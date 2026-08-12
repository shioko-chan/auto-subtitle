# YouTube → 中文字幕 → Bilibili

一个可审计的命令行管线：下载单个 YouTube 视频，用
[`Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) 从音轨转写，并由
[`Qwen3-ForcedAligner-0.6B`](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
生成词级时间轴，再用 OpenAI 兼容的 LLM API 翻译成中文字幕；最后把
中文字幕压入视频并通过 [`biliup`](https://github.com/biliup/biliup) 投稿到 B 站。

> 仅处理你有权下载、翻译和转载的内容，并遵守 YouTube、Bilibili 及原作者的条款。
> 默认关闭上传，避免配置尚未检查时意外投稿。

## 工作流

```text
YouTube URL
  → yt-dlp 下载视频和元数据
  → Qwen3-ASR-1.7B 分块转写音轨
  → Qwen3-ForcedAligner-0.6B 生成词级时间戳
  → LLM 联合决定 cue 边界并翻译为单行中文字幕
  → 本地按 ID 恢复时间轴并校验完整覆盖和整帧宽度
  → 用同一 LLM 翻译投稿标题和简介，并生成 B 站标签
  → 输出 SRT，并由 ffmpeg 烧录硬字幕
  → biliup 上传（可选，默认关闭）
```

每个 URL 使用固定哈希作为工作目录名，最终产物包括源字幕、译文 SRT、译后的
`source.semantic.srt`、`translated.metadata.json`、压制后的 MP4 和 `manifest.json`。
LLM 密钥默认从 `pass` 读取。

## 环境要求

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)；负责 Python、虚拟环境、依赖和锁文件
- `ffmpeg`，构建时需包含 `libass` 字幕滤镜
- B 站上传时需要 `biliup`
- NVIDIA GPU；默认配置针对 RTX 2080 Ti 使用 FP16
- 首次转写会从 Hugging Face 下载 ASR 和 aligner 两个模型

进入包含 Python 3.11、uv、ffmpeg 和 CUDA 运行库的 Nix 开发环境，再安装项目及
Qwen ASR 运行时：

```bash
nix develop
uv sync --extra asr
cp config.example.toml config.toml
```

`uv` 会根据 `.python-version` 准备 Python 3.11，并严格按照 `uv.lock` 创建 `.venv`；
`yt-dlp` 会随项目安装。当前新管线始终从音轨重新生成源字幕，不使用 YouTube 人工或
自动字幕，因此运行完整管线必须安装 `asr` extra。

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
model = "deepseek-v4-flash"
thinking = "disabled"
max_tokens = 16384
max_retries = 5
```

如需改用环境变量，将 `api_key_pass_entry = ""`，再通过 `api_key_env` 指定变量名。

LLM HTTPS 请求会在系统 CA 基础上补充 `certifi` CA bundle，兼容 uv 独立 Python、
NixOS、macOS 和 Windows，同时保留 `SSL_CERT_FILE` 等自定义 CA 配置。

字幕使用联合 JSON 请求：LLM 直接把 forced aligner 单元组成 cue 并翻译，首选返回
`{"cues":[{"start_id":...省略...}]}`。解析器同时兼容旧 NDJSON、裸数组和连续 JSON
对象，再统一执行相同校验。程序根据 ID 恢复日文原文和时间轴，并严格校验范围连续、
有序、无遗漏、无重复。助词、专名和术语完整性由模型结合上下文判断，不由本地规则禁止
特定边界。时长、字符数、停顿和窗口边缘都不会成为硬边界；停顿只作为语义判断证据。

每个非最终窗口都会舍弃模型返回的最后一条 cue，并从它的 `start_id` 带入下一窗口重新
规划。若窗口只有一条 cue，程序会扩大范围；请求连续失败后则缩小范围，无法在 API
上下文限制内得到有效结果时停止，不使用硬裁断回退。已确认的连续前缀会逐窗原子写入
job 目录的 `cue-translation-cache.json`，重跑时只恢复其中最长的合法连续前缀。旧的
`source-segments-cache.json`、`translation-cache.json` 和 `display-segments-cache.json`
不会再读取。

联合输入使用 `[id,duration_ms,gap_after_ms,text]`，绝对时间只在本地保存。固定规则、
术语表和视频信息位于请求前缀，窗口数据随后，重试错误放在末尾，以提高 DeepSeek
上下文缓存命中。日志会记录 `prompt_cache_hit_tokens`、`prompt_cache_miss_tokens`、
命中率及输出 token；同起始时间单元只发送紧凑的 ID 范围，不枚举所有合法终点。

提示词中的建议字数按当前画幅扣除左右安全边距后计算。译文超过一行但不超过整帧宽度
两倍时，本地保留同一 cue 并自动平衡为两行；超过两倍才拒绝响应。ASS 左右边距统一为
`1px`，不按 cue 单独调整。渲染阶段所有 cue 使用统一标准字号，不再逐 cue 缩字，也不
调用 LLM 二次分段。最终
写入 ASS 时会删除行末的逗号、句号、分号、冒号和顿号，但保留问号、感叹号与省略号；
此显示清理不修改审计 SRT。翻译前会删除 `[音楽]`、`[歌声]`、`[拍手]`、`[笑]`、
`[鼻息]` 等非语音标记，并丢弃清理后为空的 aligner 单元。任何字幕完整性错误都会
阻止渲染和上传。

## 先在本地运行

检查依赖：

```bash
uv run --extra asr subtitle-pipeline --config config.toml check
```

下载、翻译并压制，但不上传：

```bash
uv run --extra asr subtitle-pipeline --config config.toml run --no-upload 'https://www.youtube.com/watch?v=...'
```

URL 放在单引号中时不要再写 `\?` 或 `\=`。为兼容常见的复制方式，管线会自动移除
这两个位置的多余反斜杠。音轨按 170 秒分块，每个切点两侧额外提供 2 秒上下文，减少
边界处截词。若检测到 ASR 连续生成同一长文本，当前块会自动递归二分并重新识别，最短约
20 秒；仍然循环时整条任务失败，异常文本不会进入翻译。每个成功块会立即原子写入 job
目录的 `asr-cache.json`；中断重跑时只转写
缺失块。上下文区的时间戳只归属相应核心区间，因此不会产生重复 cue。分块总长度受
forced aligner 的 180 秒输入限制约束。

启用 `[audio_analysis]` 后，管线会先用 pyannote `community-1` 做 VAD 和说话人
分离，用 AST 检测歌声。普通说话仍由 Qwen forced aligner 生成细粒度时间；歌声先由
Demucs 分离主唱，再按主唱静音切成歌词短句，由 Qwen ASR 直接生成句级时间，避免把
歌声送入语音 aligner。检测到重叠说话时，pyannote ToTaToNet 会输出独立声源并分别
转写。所有音频分析结果和带说话人字段的 cue sidecar 都写入 job 目录并可断点复用。

梦限大 MewType 五位成员的应援色与声纹配置位于独立的
`character_styles.json`，不会发给 LLM。独播频道会从无重叠语音自动建立声纹原型；
团播仅在余弦距离达到阈值时映射身份，否则保留匿名说话人和默认白色字幕。重叠人物
字幕会保留各自时间，并在 ASS 中分配不同垂直行。

结果位于 `work/<URL哈希>/translated.mp4`，译文位于
`work/<URL哈希>/translated.zh-CN.srt`，投稿标题、简介和最终标签位于
`work/<URL哈希>/translated.metadata.json`。先抽查专名、数字、断句和 ASR 可能
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
uv run --extra asr subtitle-pipeline --config config.toml run --upload 'https://www.youtube.com/watch?v=...'

# 或设置 [upload] enabled = true 后正常运行
uv run --extra asr subtitle-pipeline --config config.toml run 'https://www.youtube.com/watch?v=...'
```

默认 `copyright = 2` 表示转载，来源自动使用 YouTube URL；若 `source` 非空则使用配置值。
上传使用 `biliup --user-cookie ... upload`，不会把 Cookie 内容放到命令行。
B 站简介会按 2000 个 UTF-16 code units 安全截断，避免补充平面字符导致服务端误判超长。

批量处理 2026-07-27 至 2026-08-10 的梦限大MewType公开直播录播：

```bash
./scripts/upload-recent-yumemita.sh
```

精简的完成状态写入 `work/yumemita-2026-08-10-status.log`，其中只包含每条
录播的 `RUN`、`OK`、`FAIL`、`SKIP` 和批处理停止状态。需要在当前条目完成后
暂停时，创建 `work/yumemita-2026-08-10.stop`；恢复前删除该文件并重新运行脚本。

脚本按日期串行上传，会员限定录播不在队列中。成功投稿的任务会根据工作目录中的
`manifest.json` 自动跳过，因此中断或部分失败后可以运行同一命令继续。完成状态保存到
`work/yumemita-2026-08-10-status.log`。可将其他配置文件作为第一个参数传入。

## 常用调整

- `asr.model` / `aligner_model`：分别指定 Qwen3-ASR 和 forced aligner 模型。
- `asr.device` / `dtype`：这台 RTX 2080 Ti 使用 `cuda:0` 和 `float16`；不要改为
  该显卡不支持的 `bfloat16`。
- `asr.language`：梦限大MewType直播固定为 `Japanese`，减少语言误判。
- `asr.context`：提供节目、团体和专名背景，辅助识别罕见词。
- `asr.chunk_seconds` / `chunk_context_seconds`：控制可恢复分块和切点上下文；总输入
  长度不能超过 180 秒。
- `asr.max_new_tokens`：单块 ASR 最多生成的 token 数，默认 `2048`。
- `segmentation.model_window_cues`：联合请求初始新增的 forced aligner 单元数，默认
  `600`；非最终窗口的末条 cue 会自动带入下一窗口。
- `llm.batch_size`：旧字幕翻译接口及元数据辅助任务的批大小；联合字幕窗口由
  `segmentation.model_window_cues` 控制。
- `llm.max_tokens`：单次 LLM 响应的输出 token 上限，DeepSeek V4 建议设为 `16384`。
- `llm.max_retries`：拆分批次前的请求重试次数，建议设为 `5`。
- `llm.context_cues`：联合请求附带的已确认相邻 cue 数量，默认 `3` 条。
- `llm.thinking`：DeepSeek V4 的严格 JSON 翻译应设为 `"disabled"`；其他服务不支持该参数时省略。
- `llm.translate_metadata`：是否翻译 YouTube 标题和简介。
- `llm.metadata_description_max_chars`：发送给 LLM 的源简介字符上限。
- `llm.metadata_tag_count`：同一次元数据翻译请求生成的 B 站标签数量。
- `llm.metadata_subtitle_max_chars`：用于识别内容/IP 的字幕首、中、尾证据字符上限。
- `llm.ip_aliases_file`：已知 IP 的规范名及中英日别名 JSON 文件。
- `llm.glossary_files`：附加翻译术语表；后加载的自定义译名覆盖内置译名。
- `render.font_name`：必须是机器上已安装且包含中文字形的字体。
- `render.font_size_ratio` / `portrait_font_size_ratio`：横屏与竖屏字号相对于视频短边的比例，并受最小/最大字号限制。
- `render.margin_horizontal_ratio` / `portrait_margin_horizontal_ratio`：横屏与竖屏左右安全边距各自占视频宽度的比例。
- `render.margin_vertical_ratio`：字幕底边距占视频高度的比例。
- `render.outline_ratio`：黑色描边相对于视频短边的比例，默认 `0.003`。
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
可覆盖有争议或偏好的译名。普通作品名、歌曲名等放在 `terms` 字符串映射中；人物可放在
`characters` 数组中，以 `id`、`canonical`、`source_name`、`aliases` 和
`short_names` 描述同一实体。`short_names` 的 `source` 只翻译成对应 `target`，不会扩写
为全名；设置 `context_only: true` 后，仅在视频、频道或当前语境支持该人物身份时采用。
相同 `id` 的人物由后加载的自定义术语表整体覆盖。旧的纯 `terms` 格式继续兼容。
设置 `"always": true` 可让某个自定义术语表对所有视频启用，否则应提供 `match` 字符串
数组用于自动识别。

## 开发与测试

测试完全离线，不会下载、调用 LLM 或上传：

```bash
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src tests
```

修改依赖时使用 `uv add <package>`，可选依赖使用
`uv add --optional asr <package>`，并提交同步更新的 `pyproject.toml` 和 `uv.lock`。

外部命令均通过参数数组调用，不经 shell 展开；工作目录、API 密钥和 Cookie 已加入
`.gitignore`。真实端到端测试需要自行提供 URL、API 凭据及转载授权。
