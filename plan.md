# TTS 服务

---

# 🧠 一、整体架构设计（核心）

```text
                ┌───────────────┐
   HTTP Clone → │   API Server   │
                │  (FastAPI)     │
                └──────┬────────┘
                       │
        ┌──────────────┼──────────────┐
        ↓                              ↓
 Voice Store                    WebSocket Server
 (Embedding DB)               (Streaming TTS)
        ↓                              ↓
        └──────────────┬──────────────┘
                       ↓
                TTS Worker Pool
                (Qwen TTS Runtime)
```

---

# 🎯 功能拆分

## 1️⃣ HTTP：音色克隆

```http
POST /add_voice_timbre
Content-Type: multipart/form-data

voice_wav: xxx.wav / mp3
voice_id: 音色ID
```

返回：

```json
true / false
```

---

## 2️⃣ WebSocket：实时语音生成

```json
// client -> server
{
  "content": "你好，欢迎使用TTS",
  "voice_id": "voice_xxx"
}
```

```text
// server -> client
(binary audio chunk...)
(binary audio chunk...)
...
```

一段音频结束返回空chunk，即 len([]byte) = 0 的时候表示这一段音频完毕了。

---

# 🧱 二、项目结构（推荐你直接用）

```bash
tts-service/
├── app/
│   ├── main.py                # FastAPI入口
│   ├── api/
│   │   ├── http_voice.py      # clone接口
│   │   └── ws_tts.py          # websocket接口
│   ├── core/
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── model_manager.py   # 模型加载
│   ├── service/
│   │   ├── voice_clone.py     # 音色提取
│   │   ├── tts_stream.py      # 流式推理
│   │   └── audio_utils.py
│   ├── storage/
│   │   ├── voice_store.py     # 音色存储
│   │   └── file_store.py
│   └── worker/
│       └── tts_worker.py      # 推理worker
│
├── models/
│   └── qwen_tts/
│
├── data/
│   └── voices/
│
└── requirements.txt
```

---

# ⚙️ 三、核心模块设计

---

## 1️⃣ 模型管理（单例）

```python
# app/core/model_manager.py

class ModelManager:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = load_qwen_tts_model()
        return cls._model
```

👉 要点：

* 全局单例（避免重复加载GPU）
* 支持 lazy load

---


## 3️⃣ 音色存储（关键设计点）

```python
# app/storage/voice_store.py

import pickle
import os

VOICE_DIR = "data/voices"

def save_voice_embedding(voice_id, embedding):
    path = os.path.join(VOICE_DIR, f"{voice_id}.pkl")
    with open(path, "wb") as f:
        pickle.dump(embedding, f)

def load_voice_embedding(voice_id):
    path = os.path.join(VOICE_DIR, f"{voice_id}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)
```

---

## 4️⃣ 流式TTS（最核心）

```python
# app/service/tts_stream.py

def stream_tts(text: str, voice_embedding):
    model = ModelManager.get_model()

    generator = model.stream_generate(
        text=text,
        speaker_embedding=voice_embedding
    )

    for chunk in generator:
        yield chunk  # PCM / wav bytes
```

👉 关键点：

* 必须是 generator
* chunk粒度控制延迟

---

## 5️⃣ WebSocket接口

```python
# app/api/ws_tts.py

from fastapi import WebSocket

async def websocket_tts(ws: WebSocket):
    await ws.accept()

    data = await ws.receive_json()

    text = data["content"]
    voice_id = data["voice_id"]

    embedding = load_voice_embedding(voice_id)

    for chunk in stream_tts(text, embedding):
        await ws.send_bytes(chunk)
```

---


---

## 1️⃣ GPU控制（重点）

### 限制并发

```python
import asyncio

semaphore = asyncio.Semaphore(2)  # 同时最多2个推理

async with semaphore:
    run_tts()
```

---

## 2️⃣ chunk大小（影响延迟）

建议：

```text
20ms ~ 50ms 音频块
```

👉 太大 → 延迟高
👉 太小 → CPU overhead高

---

## 3️⃣ embedding缓存

```python
# 内存缓存
voice_cache = {}
```

---

## 4️⃣ WebSocket优化

* 使用 binary（不要 base64）
* 不要一次性返回 wav header

---


## ✅ 2️⃣ 分离 Worker

```text
API Server（无GPU）
↓
TTS Worker（GPU）
```

---

## ✅ 3️⃣ 多GPU扩展

```text
worker-1 → GPU0
worker-2 → GPU1
```

---

## ✅ 4️⃣ 音色标准化（非常关键）

在 clone 时：

* 重采样 → 16k / 22k
* 单声道
* 降噪（可选）

---

# 🎯 七、落地版本建议

第一版建议这样：

```text
模型：Qwen TTS 0.6B
框架：FastAPI
协议：HTTP + WebSocket
部署：单机 + 单GPU
并发：2~4
```

---

