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
                (Qwen TTS Runtime + torch.compile)
                       ↓
                Sentence Chunking
                (逐句生成 → 立即发送)
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

def stream_tts_chunked(text: str, voice_prompt, *, language="Auto"):
    """分句流式生成：按句子分割文本，逐句合成并立即 yield 音频 chunks。"""
    chunks = _split_sentences(text)
    model = ModelManager.get_model()
    for sentence in chunks:
        pcm_bytes = _synthesize_chunk(model, sentence, voice_prompt, language)
        for chunk in _iter_fixed_f32le_chunks(pcm_bytes, ...):
            yield chunk
    yield b""  # 空帧终止符
```

👉 关键点：

* 分句流式：按句子边界分割，逐句生成并立即发送，大幅降低首包延迟
* 必须是 generator
* chunk粒度控制延迟（默认 32ms）
* 动态 max_new_tokens：根据文本长度估算合理 token 上限

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

    prompt = load_voice_embedding(voice_id)

    # 异步逐 chunk 流式发送（不阻塞事件循环）
    async def stream_and_send():
        loop = asyncio.get_running_loop()
        gen = stream_tts_chunked(text, prompt, language=language)
        while True:
            chunk = await loop.run_in_executor(None, next, gen, _SENTINEL)
            if chunk is _SENTINEL:
                break
            await ws.send_bytes(chunk)

    await run_with_tts_limit(stream_and_send)
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
* 异步逐 chunk 发送：`run_in_executor` 消费同步生成器，每得到一个 chunk 立即通过 WebSocket 发送，不阻塞事件循环

---

## 5️⃣ GPU加速优化

* **Flash Attention 2**：默认启用（`ATTN_IMPLEMENTATION=flash_attention_2`），需在服务器安装 `uv pip install flash-attn --no-build-isolation`，可回退到 SDPA
* **torch.compile**：模型加载后自动对 talker 应用 `torch.compile(mode="reduce-overhead")`，首次请求有编译开销
* **CUDA 预热**：模型加载后自动执行短推理，触发 kernel 编译缓存
* **soxr 重采样**：使用 `soxr.resample()` 替代 `scipy.signal.resample()`，重采样速度提升 3-10x
* **动态 token 上限**：根据文本长度估算 `max_new_tokens`，避免短文本无效计算

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

