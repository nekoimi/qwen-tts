# 手动下载 Qwen3-TTS 模型

默认模型为 **[Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)**。首次启动时，`Qwen3TTSModel.from_pretrained` 会从 Hugging Face Hub 拉取权重，体积较大、在网络一般时可能很慢。可以**先在稳定网络环境下把模型完整下载到本机**，再通过环境变量让服务**只读本地目录**，避免运行时长时间阻塞。

## 你需要准备什么

- 足够磁盘空间（完整仓库通常为 **数 GB 级别**，以 Hub 页面显示为准）。
- 若使用 **Git LFS**：已安装 Git 与 [Git LFS](https://git-lfs.com/)。
- 若使用 **命令行下载**：已安装 `huggingface-cli`（来自 `huggingface_hub` 包）。

本仓库已依赖 `transformers` / `huggingface_hub`（由 `qwen-tts` 引入），可在项目 venv 中使用：

```bash
uv sync
uv run huggingface-cli download --help
```

## 方式一：huggingface-cli（推荐）

在项目目录外或 `models/` 下指定本地目录，例如下载到 `models/Qwen3-TTS-12Hz-0.6B-Base`：

```bash
# 进入项目根目录（或你希望存放权重的目录）
cd qwentts

# 下载整个模型仓库到本地目录（示例：相对路径；cmd 用 ^ 换行，PowerShell 可写成一行）
uv run huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir models/Qwen3-TTS-12Hz-0.6B-Base
```

Linux / macOS 可直接使用同一行命令。

**同等效果（Python，跨平台）**：

```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-TTS-12Hz-0.6B-Base', local_dir='models/Qwen3-TTS-12Hz-0.6B-Base')"
```

下载完成后，目录内应包含 `config.json`、权重（如 `*.safetensors` 或分片）、`tokenizer` 相关文件等，与 Hub 仓库结构一致。

**启动服务前**设置环境变量，让 `MODEL_ID` 指向该**本地目录**（绝对路径最稳妥）：

```text
# Windows PowerShell 示例
$env:MODEL_ID = "C:\Users\你的用户名\Working\PythonProjects\qwentts\models\Qwen3-TTS-12Hz-0.6B-Base"
```

```bash
# Linux / macOS 示例
export MODEL_ID="/path/to/qwentts/models/Qwen3-TTS-12Hz-0.6B-Base"
```

`Qwen3TTSModel.from_pretrained` 与 Transformers 行为一致：**若 `MODEL_ID` 为存在的本地路径，则直接从磁盘加载，不再走在线下载**（除非实现内部仍请求元数据；绝大多数情况下可完全离线加载已完整下载的目录）。

## 方式二：Git Clone（含 Git LFS）

适合已熟悉 Hugging Face 仓库克隆方式的场景：

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base models/Qwen3-TTS-12Hz-0.6B-Base
```

克隆完成后，同样将 `MODEL_ID` 设为该目录的路径。

若在国内直连 `huggingface.co` 较慢，可自行使用代理、离线拷贝已下载目录，或使用下文镜像加速（以合规与可用性为准）。

## 使用 Hugging Face 镜像加速（可选）

在仅使用 **官方仓库名** 下载时，可为 `huggingface_hub` 设置镜像端点（常见示例，请以实际服务条款为准）：

```text
# 示例：使用 hf-mirror（请在浏览器确认其当前用法与域名）
set HF_ENDPOINT=https://hf-mirror.com
```

PowerShell：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

然后再执行 `huggingface-cli download` 或首次在线拉取。下载完成后，**建议仍通过本地路径 `MODEL_ID` 指向已下载目录**，减少对镜像的依赖。

## 与全局缓存目录的关系

即使不设 `MODEL_ID` 为本地路径，Hub 也会把文件缓存在本机（常见为 `HF_HOME` / `~/.cache/huggingface`）。手动下载到**固定目录**并设置 `MODEL_ID` 的好处是：

- 路径清晰，便于备份、迁移、离线机部署；
- 避免误以为“没联网就自动从缓存读”时的路径不一致问题。

若你希望继续使用 Hub 默认缓存而不单独指定目录，可只设置缓存根目录，例如：

```text
HF_HOME=D:\hf-cache
```

具体行为以当前 `huggingface_hub` / `transformers` 版本为准。

## 校验与排错

1. **目录是否完整**：本地目录中应有 `config.json`，以及模型权重相关文件；若体积极小，可能是 LFS 未拉取完整，需执行 `git lfs pull` 或换用 `huggingface-cli download`。
2. **`MODEL_ID` 格式**：使用**文件夹路径**时，不要多加引号写入 `.env` 时注意转义；Windows 推荐 `C:/path/to/model` 或转义的反斜杠。
3. **权限与 Token**：若未来模型变为需登录访问，请在 Hub 申请 Token，并设置 `HF_TOKEN` 或 `huggingface-cli login`（当前公开模型一般不需要）。

配置完成后，正常启动：

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

日志中应出现从本地路径加载模型；若仍长时间无响应，请检查路径是否指向**包含 `config.json` 的那一层目录**。
