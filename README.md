# QwenTTS

基于 **FastAPI** 的 TTS 服务：HTTP 注册克隆音色，WebSocket 按音色流式下发 **单声道 float32 小端（f32le，与 WAV IEEE float 样本布局一致）** 分片；段结束以 **长度为 0 的二进制帧** 标记。

- 模型：[Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)（`qwen-tts`）
- 包管理：**uv**；默认 PyPI 索引已在 [`pyproject.toml`](pyproject.toml) 中通过 `[[tool.uv.index]]` 配置为阿里云镜像。
- 模型体积较大、在线拉取慢时，请先阅读 **[手动下载模型说明](docs/MODEL_DOWNLOAD.md)**，将权重放到本机后设置环境变量 `MODEL_ID` 为本地目录。

## 环境要求

- Python 3.10+（开发时使用 3.13 亦可）
- **NVIDIA GPU + CUDA** 推荐（CPU 可设 `DEVICE=cpu`，速度与显存占用需自行评估）
- 可选：系统安装 [SoX](http://sox.sourceforge.net/)（部分环境会提示 `SoX could not be found`，若仅用 numpy 参考音频元组通常仍可工作）

## 安装

```bash
cd qwentts
uv venv
uv sync
# 可选：包含 WebSocket 测试脚本依赖（websockets）
uv sync --extra dev
```

依赖索引由仓库内 `[[tool.uv.index]]` 指向 `https://mirrors.aliyun.com/pypi/simple/`；若需临时覆盖，可设置环境变量 `UV_INDEX_URL`（一般不必）。

**PyTorch**：`uv sync` 会安装与当前平台匹配的 `torch`。若阿里云镜像缺少所需 CUDA 版本轮子，可参考 [PyTorch 官网](https://pytorch.org/) 用官方 wheel 源单独安装 `torch` / `torchaudio` 后再执行 `uv sync --inexact` 或锁定版本。

## 配置（环境变量）

| 变量 | 说明 | 默认 |
|------|------|------|
| `MODEL_ID` | Hugging Face 模型 id **或本机已下载目录的绝对/相对路径** | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` |
| `TTS_BACKEND` | 推理后端：`faster` 使用 `faster-qwen3-tts` CUDA Graph；`qwen` 使用原版 `qwen-tts` | `faster` |
| `DEVICE` | 设备，如 `cuda:0` 或 `cpu` | `cuda:0` |
| `DTYPE` | `bfloat16` / `float16` / `float32` | `bfloat16` |
| `ATTN_IMPLEMENTATION` | 如 `sdpa`、`flash_attention_2` | `sdpa` |
| `FASTER_MAX_SEQ_LEN` | `faster-qwen3-tts` 静态 KV cache 最大序列长度 | `2048` |
| `VOICE_DIR` | 音色 pkl 存储目录 | `data/voices` |
| `TTS_MAX_CONCURRENT` | 并发推理上限 | `2` |
| `CHUNK_MS` | WebSocket 每帧音频时长（毫秒），按 f32le 分块 | `32` |
| `TTS_MIN_CHUNK_LEN` | 分句流式最小文本片段长度 | `10` |
| `TTS_MAX_CHUNK_LEN` | 分句流式最大文本片段长度 | `80` |
| `TTS_CACHE_DIR` | TTS f32le 文件缓存目录 | `data/tts_cache` |
| `TTS_CACHE_MAX_BYTES` | TTS 缓存最大字节数，超过后按最近访问时间清理 | `107374182400` (100GB) |
| `TTS_CACHE_VERSION` | 缓存版本；修改切分、音频格式或生成策略时可递增以整体失效 | `v1` |
| `TARGET_SAMPLE_RATE` | 参考音频重采样目标；**合成流**也输出为该采样率的 f32le | `24000` |
| `MAX_UPLOAD_BYTES` | 上传参考音频最大字节 | `20971520` (20MB) |
| `SAVE_RAW_UPLOADS` | 是否保存原始上传文件到 `data/voices/raw/` | `false` |
| `HOST` / `PORT` | 仅文档说明；启动时用 uvicorn 参数 | `0.0.0.0` / `8000` |
| `LOG_LEVEL` | 日志级别 | `INFO` |

## 启动

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

健康检查：`GET http://127.0.0.1:8000/health`

## API

### 1. 注册音色

`POST /add_voice_timbre`，`multipart/form-data`：

- `voice_wav`：文件，`.wav` 或 `.mp3`
- `voice_id`：字符串 ID（字母数字与 `_` `-`，最长 128）

成功返回 JSON：`true`。失败为 4xx/5xx（见响应体 `detail`）。

示例：

```bash
curl -s -X POST "http://127.0.0.1:8000/add_voice_timbre" ^
  -F "voice_wav=@ref.wav" ^
  -F "voice_id=my_voice_1"
```

说明：当前实现使用 **x-vector-only** 克隆（无需参考文本），与仅提供音频文件的接口一致；若需更高还原度，可后续扩展可选字段 `ref_text` 走 ICL 模式。

### 2. WebSocket 合成

连接：`WS /ws/stream`

客户端发送一条 JSON：

```json
{
  "content": "你好，欢迎使用TTS",
  "voice_id": "my_voice_1",
  "language": "Auto"
}
```

`language` 可省略，默认 `"Auto"`。

服务端连续发送 **二进制** 分片：每帧为 **单声道 float32 小端（f32le）原始样本字节**（与 WAV 中 IEEE float 样本布局一致，**不含** RIFF 头），每样本 4 字节；合成结果会先重采样到环境变量 **`TARGET_SAMPLE_RATE`（默认 24000）** 再输出，便于客户端固定采样率播放。**最后一帧为空 payload**（`len == 0`）表示本段语音结束。

最小客户端脚本（需 **`uv sync --extra dev`** 安装 `websockets`）：

```bash
uv run python scripts/ws_tts_client.py ^
  --voice-id my_voice_1 ^
  --content "你好，欢迎使用TTS" ^
  --output-wav out.wav
```

`--sample-rate` 默认 `24000`，须与服务的 **`TARGET_SAMPLE_RATE`** 一致。仅打印分片信息不写文件时，省略 `--output-wav` 即可。

### 3. 预热 TTS 缓存

将高频话术写入 UTF-8 文本文件（每行一句，空行和 `#` 开头的行会跳过），部署后可提前生成句子级缓存：

```bash
uv run python scripts/prewarm_tts_cache.py ^
  --voice-id my_voice_1 ^
  --text-file common_phrases.txt ^
  --language Auto
```

同一批话术需要多个音色时，可重复传 `--voice-id`。脚本也支持 JSON 字符串数组作为 `--text-file`。

## 项目结构

与需求文档一致：`app/main.py` 入口，`app/api/` HTTP 与 WebSocket，`app/core/` 配置与模型单例，`app/service/` 音频与合成，`app/storage/` 音色持久化，`app/worker/` 并发封装。

## License

模型与 `qwen-tts` 以各自声明为准；本仓库应用代码按项目需要选择许可证。
