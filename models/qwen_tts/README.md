本地或 Hugging Face 缓存的模型权重目录占位。默认通过 `Qwen3TTSModel.from_pretrained` 在线拉取到本机缓存。

若下载过慢，请按仓库内 **[docs/MODEL_DOWNLOAD.md](../../docs/MODEL_DOWNLOAD.md)** 将模型完整下载到本机目录，并将环境变量 `MODEL_ID` 指向该目录（例如 `models/Qwen3-TTS-12Hz-0.6B-Base`）。
