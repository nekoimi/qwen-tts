# QwenTTS

基于 **FastAPI** 的 TTS 服务：HTTP 注册克隆音色，WebSocket 按音色流式下发 **PCM（int16 小端）** 分片；段结束以 **长度为 0 的二进制帧** 标记。

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
| `DEVICE` | 设备，如 `cuda:0` 或 `cpu` | `cuda:0` |
| `DTYPE` | `bfloat16` / `float16` / `float32` | `bfloat16` |
| `ATTN_IMPLEMENTATION` | 如 `sdpa`、`flash_attention_2` | `sdpa` |
| `VOICE_DIR` | 音色 pkl 存储目录 | `data/voices` |
| `TTS_MAX_CONCURRENT` | 并发推理上限 | `2` |
| `CHUNK_MS` | WebSocket 每帧 PCM 时长（毫秒） | `32` |
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

连接：`WS /ws/tts`

客户端发送一条 JSON：

```json
{
  "content": "你好，欢迎使用TTS",
  "voice_id": "my_voice_1",
  "language": "Auto"
}
```

`language` 可省略，默认 `"Auto"`。

服务端连续发送 **二进制** PCM 分片；**最后一帧为空 payload**（`len == 0`）表示本段语音结束。

最小客户端脚本（需 **`uv sync --extra dev`** 安装 `websockets`）：

```bash
uv run python scripts/ws_tts_client.py ^
  --voice-id my_voice_1 ^
  --content "你好，欢迎使用TTS" ^
  --output-wav out.wav
```

`--sample-rate` 默认 `24000`，应与模型实际输出采样率一致（若播放速度不对，可按服务端合成结果调整）。仅打印分片信息不写文件时，省略 `--output-wav` 即可。

## 项目结构

与需求文档一致：`app/main.py` 入口，`app/api/` HTTP 与 WebSocket，`app/core/` 配置与模型单例，`app/service/` 音频与合成，`app/storage/` 音色持久化，`app/worker/` 并发封装。

## License

模型与 `qwen-tts` 以各自声明为准；本仓库应用代码按项目需要选择许可证。
