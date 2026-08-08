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
  → 用同一 LLM 翻译投稿标题和简介，并生成 B 站标签
  → 输出 SRT，并由 ffmpeg 烧录硬字幕
  → biliup 上传（可选，默认关闭）
```

每个 URL 使用固定哈希作为工作目录名，最终产物包括源字幕、译文 SRT、译后的
`translated.metadata.json`、压制后的 MP4 和 `manifest.json`。LLM 密钥默认从 `pass`
读取。

## 环境要求

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)；负责 Python、虚拟环境、依赖和锁文件
- `ffmpeg`，构建时需包含 `libass` 字幕滤镜
- B 站上传时需要 `biliup`
- 首次 Whisper 回退会下载所选模型；`small` 的 CPU 模式也可运行，GPU 更快

安装项目及 Whisper 回退：

```bash
uv sync --extra whisper
cp config.example.toml config.toml
```

`uv` 会根据 `.python-version` 准备 Python 3.11，并严格按照 `uv.lock` 创建 `.venv`；
`yt-dlp` 会随项目安装。Ubuntu/Debian 可用 `apt install ffmpeg fonts-noto-cjk`；macOS
可用 `brew install ffmpeg`。如果只处理始终带字幕的视频，可执行 `uv sync` 而不安装
Whisper extra，并设置 `[whisper] enabled = false`。遇到无字幕或原语言字幕不可用的
视频时，管线会立即报错，不会尝试加载转写模型。

完整的 YouTube 格式解析还需要 JavaScript runtime。管线会依次自动寻找 Deno、Node
和 QuickJS；推荐安装 Deno 2.3+，Node 则需 22+。PyPI 依赖已启用 yt-dlp 的 EJS 和
`curl-cffi` extras。

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

LLM HTTPS 请求会在系统 CA 基础上补充 `certifi` CA bundle，兼容 uv 独立 Python、
NixOS、macOS 和 Windows，同时保留 `SSL_CERT_FILE` 等自定义 CA 配置。

若兼容服务不接受 `response_format = json_object`，设置 `json_mode = false`；管线仍会
解析并严格校验返回的 JSON。

翻译按 cue 批处理，返回的每个 ID 都会校验，时间轴不会交给模型改写。失败批次会
指数退避重试。

## 先在本地运行

检查依赖：

```bash
uv run --extra whisper subtitle-pipeline --config config.toml check
```

下载、翻译并压制，但不上传：

```bash
uv run --extra whisper subtitle-pipeline --config config.toml run --no-upload 'https://www.youtube.com/watch?v=...'
```

URL 放在单引号中时不要再写 `\?` 或 `\=`。为兼容常见的复制方式，管线会自动移除
这两个位置的多余反斜杠。视频与字幕采用两个独立下载阶段：字幕遇到 429 或其他网络
错误时，已下载的视频仍会保留并自动进入 Whisper 转写。

字幕请求会根据 yt-dlp 元数据自动加入一个最佳原语言轨道：优先原语言人工字幕，其次
`<语言>-orig`，再其次原语言普通轨道。`download.subtitle_languages` 仅表示额外的
备用或翻译轨道，默认为空，因此不会主动下载英文、韩文等翻译字幕。

结果位于 `work/<URL哈希>/translated.mp4`，译文位于
`work/<URL哈希>/translated.zh-CN.srt`，投稿标题、简介和最终标签位于
`work/<URL哈希>/translated.metadata.json`。先抽查专名、数字、断句和 Whisper 可能
出现的幻觉，再启用上传。

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
uv run --extra whisper subtitle-pipeline --config config.toml run --upload 'https://www.youtube.com/watch?v=...'

# 或设置 [upload] enabled = true 后正常运行
uv run --extra whisper subtitle-pipeline --config config.toml run 'https://www.youtube.com/watch?v=...'
```

默认 `copyright = 2` 表示转载，来源自动使用 YouTube URL；若 `source` 非空则使用配置值。
上传使用 `biliup --user-cookie ... upload`，不会把 Cookie 内容放到命令行。

## 常用调整

- `download.subtitle_languages`：按 YouTube 语言标签筛选字幕，例如 `en.*`、`ja.*`。
- `whisper.model`：CPU 可先用 `small`；有足够显存时可改为 `large-v3` 或
  `turbo`（取决于 faster-whisper 支持的模型名）。
- `whisper.enabled`：设为 `false` 时禁止语音转写回退；此时可以不安装 whisper extra。
- `whisper.language`：已知源语言时显式设置可减少误判。
- `segmentation.max_gap_seconds`：超过该停顿不跨 cue 合并。
- `segmentation.max_duration_seconds` / `max_source_chars`：语义句缺少标点时的硬边界。
- `llm.batch_size`：字幕很长或模型上下文较小时调低。
- `llm.translate_metadata`：是否翻译 YouTube 标题和简介。
- `llm.metadata_description_max_chars`：发送给 LLM 的源简介字符上限。
- `llm.metadata_tag_count`：同一次元数据翻译请求生成的 B 站标签数量。
- `llm.metadata_subtitle_max_chars`：用于识别内容/IP 的字幕首、中、尾证据字符上限。
- `llm.ip_aliases_file`：已知 IP 的规范名及中英日别名 JSON 文件。
- `llm.glossary_files`：附加翻译术语表；后加载的自定义译名覆盖内置译名。
- `render.font_name`：必须是机器上已安装且包含中文字形的字体。
- `upload.enabled`：生产环境才建议开启；命令行 `--no-upload` 始终优先关闭上传。
- `upload.tags`：始终保留的固定标签；会与自动标签去重合并。
- `upload.max_tags`：投稿使用的固定标签与自动标签总数上限。
- `upload.tag_catalog_file`：经授权获取或人工维护的 B 站规范标签与热度目录。

IP 别名文件格式参考 `ip_aliases.example.json`：

```json
{
  "BanG Dream!": ["BanG Dream", "バンドリ", "邦邦"]
}
```

B 站标签目录格式参考 `bilibili-tags.example.json`：

```json
{
  "BanG Dream": {
    "heat": 100000,
    "aliases": ["BanG Dream!", "バンドリ", "邦邦"]
  }
}
```

标签分析会综合频道名、上传者、YouTube 分类/标签、系列/季度/剧集字段、音乐元数据、
字幕首中尾摘要、IP 别名和上述 B 站目录。目录中的别名会规范化为正式标签；同一别名
匹配多个标签时选择热度更高者。公开 B 站搜索会对自动请求返回验证码，项目不会调用
未公开搜索接口；实时热度应通过已获授权的开放平台应用导出后更新本地目录。

项目内置 Bang Dream 翻译术语表，并根据频道、标题、简介及 YouTube 元数据中的
`BanG Dream`、`バンドリ`、`ガルパ`、乐队名等标识自动启用。命中后，字幕每个
翻译批次以及标题/简介翻译都会收到作品背景、乐队名、角色名、舞台名和常见声优姓名。
未命中的普通视频不会收到该术语表。内置资料参考 BanG Dream 官方角色/乐队页面、
萌娘百科简中条目及 BanG Dream Fandom 角色目录，来源 URL 保存在术语 JSON 中。

可通过 `llm.glossary_files` 添加同格式 JSON；自定义文件在内置术语之后加载，所以
可覆盖有争议或偏好的译名。设置 `"always": true` 可让某个自定义术语表对所有视频
启用，否则应提供 `match` 字符串数组用于自动识别。

## 开发与测试

测试完全离线，不会下载、调用 LLM 或上传：

```bash
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src tests
```

修改依赖时使用 `uv add <package>`，可选依赖使用
`uv add --optional whisper <package>`，并提交同步更新的 `pyproject.toml` 和 `uv.lock`。

外部命令均通过参数数组调用，不经 shell 展开；工作目录、API 密钥和 Cookie 已加入
`.gitignore`。真实端到端测试需要自行提供 URL、API 凭据及转载授权。
