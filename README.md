# YouTube → 中文字幕 → Bilibili

一个可审计的命令行管线：下载单个 YouTube 视频及可用字幕，用 OpenAI 兼容的
LLM API 翻译成中文字幕；如果 YouTube 没有字幕，则用开源
[`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) 从音轨生成时间轴；最后把
中文字幕压入视频并通过 [`biliup`](https://github.com/biliup/biliup) 投稿到 B 站。

> 仅处理你有权下载、翻译和转载的内容，并遵守 YouTube、Bilibili 及原作者的条款。
> 默认关闭上传，避免配置尚未检查时意外投稿。

## 工作流

```text
YouTube URL
  → yt-dlp 下载视频、元数据、人工/自动字幕
  → 无字幕时 faster-whisper 本地转写
  → 按句末标点、停顿、时长和字符上限进行语义合并
  → OpenAI 兼容 /chat/completions API 分批翻译
  → 输出 SRT，并由 ffmpeg 烧录硬字幕
  → biliup 上传（可选，默认关闭）
```

每个 URL 使用固定哈希作为工作目录名，最终产物包括源字幕、译文 SRT、压制后的
MP4 和 `manifest.json`。LLM 密钥只从环境变量读取。

## 环境要求

- Python 3.11+
- `ffmpeg`，构建时需包含 `libass` 字幕滤镜
- B 站上传时需要 `biliup`
- 首次 Whisper 回退会下载所选模型；`small` 的 CPU 模式也可运行，GPU 更快

安装项目及 Whisper 回退：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[whisper]'
cp config.example.toml config.toml
```

`yt-dlp` 会随项目安装。Ubuntu/Debian 可用 `apt install ffmpeg fonts-noto-cjk`；macOS
可用 `brew install ffmpeg`。如果只处理始终带字幕的视频，可以先不安装 Whisper extra，
但遇到无字幕视频时管线会明确报错。

## 配置 LLM

默认使用 DeepSeek 的 OpenAI 兼容 Chat Completions 接口，并从 password-store 读取
API Key：

```bash
pass insert api/deepseek
pass show api/deepseek
```

配置如下；`pass` 输出的第一行会作为密钥，内容不会写入日志或配置文件：

```toml
[llm]
base_url = "https://api.deepseek.com"
api_key_pass_entry = "api/deepseek"
model = "deepseek-chat"
```

如需改用环境变量，将 `api_key_pass_entry = ""`，再通过 `api_key_env` 指定变量名。

若兼容服务不接受 `response_format = json_object`，设置 `json_mode = false`；管线仍会
解析并严格校验返回的 JSON。

翻译按 cue 批处理，返回的每个 ID 都会校验，时间轴不会交给模型改写。失败批次会
指数退避重试。

## 先在本地运行

检查依赖：

```bash
subtitle-pipeline --config config.toml check
```

下载、翻译并压制，但不上传：

```bash
subtitle-pipeline --config config.toml run --no-upload 'https://www.youtube.com/watch?v=...'
```

结果位于 `work/<URL哈希>/translated.mp4`，译文位于
`work/<URL哈希>/translated.zh-CN.srt`。先抽查专名、数字、断句和 Whisper 可能出现的
幻觉，再启用上传。

年龄限制、地区限制或需要登录的视频，可以在 `[download]` 配置
`cookies_from_browser = "chrome"`，或配置 Netscape 格式的 `cookies_file`。

## 上传 Bilibili

安装并登录（登录信息默认保存到 `cookies.json`）：

```bash
uv tool install biliup
biliup login
```

确认 `config.toml` 的分区 `tid`、转载标记、来源和标签。两种启用方法任选其一：

```bash
# 单次强制上传
subtitle-pipeline --config config.toml run --upload 'https://www.youtube.com/watch?v=...'

# 或设置 [upload] enabled = true 后正常运行
subtitle-pipeline --config config.toml run 'https://www.youtube.com/watch?v=...'
```

默认 `copyright = 2` 表示转载，来源自动使用 YouTube URL；若 `source` 非空则使用配置值。
上传使用 `biliup --user-cookie ... upload`，不会把 Cookie 内容放到命令行。

## 常用调整

- `download.subtitle_languages`：按 YouTube 语言标签筛选字幕，例如 `en.*`、`ja.*`。
- `whisper.model`：CPU 可先用 `small`；有足够显存时可改为 `large-v3` 或
  `turbo`（取决于 faster-whisper 支持的模型名）。
- `whisper.language`：已知源语言时显式设置可减少误判。
- `segmentation.max_gap_seconds`：超过该停顿不跨 cue 合并。
- `segmentation.max_duration_seconds` / `max_source_chars`：语义句缺少标点时的硬边界。
- `llm.batch_size`：字幕很长或模型上下文较小时调低。
- `render.font_name`：必须是机器上已安装且包含中文字形的字体。
- `upload.enabled`：生产环境才建议开启；命令行 `--no-upload` 始终优先关闭上传。

## 开发与测试

测试完全离线，不会下载、调用 LLM 或上传：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src tests
```

外部命令均通过参数数组调用，不经 shell 展开；工作目录、API 密钥和 Cookie 已加入
`.gitignore`。真实端到端测试需要自行提供 URL、API 凭据及转载授权。
