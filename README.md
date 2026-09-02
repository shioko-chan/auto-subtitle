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
  → Qwen3-ASR-1.7B 分块转写音轨并自动识别语言
  → Qwen3-ForcedAligner-0.6B 生成词级时间戳
  → Sudachi 形态分析和本地评分生成候选单元
  → LLM 单阶段决定 cue 范围并翻译，同时处理可能的 ASR 误听
  → 本地按 ID 恢复时间轴并校验完整覆盖和整帧宽度
  → 用同一 LLM 翻译投稿标题和简介，并生成 B 站标签
  → 输出 SRT/ASS，由稀疏 libass/CUDA/NVENC 或 ffmpeg CPU 后端烧录硬字幕
  → biliup 上传（可选，默认关闭）
```

每个 URL 使用固定哈希作为工作目录名，最终产物包括源字幕、译文 SRT、译后的
`source.semantic.srt`、`translated.metadata.json`、压制后的 MP4 和 `manifest.json`。
LLM 密钥默认从 `pass` 读取。

当前实际流程、各模型的职责、缓存边界及已知风险见
[当前字幕管线与技术选型](docs/current-pipeline.md)。

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

默认 `render.backend = "auto"`：优先调用 Nix 构建的 `ass-cuda-render`。该后端使用
NVDEC 将视频帧保留在显存，CPU libass 只生成文字、描边和阴影的小型 alpha mask；
mask 仅在字幕画面变化时上传，并由 CUDA 直接混合到 NV12 帧后交给 NVENC。GPU 不支持
输入编码、旋转或像素格式时自动回退到原有 ffmpeg CPU/libass 路径；设为 `cuda` 可要求
失败即停止，设为 `cpu` 可禁用该后端。Turing 显卡不支持 AV1 NVDEC，因此下载默认优先
VP9/H.264，已缓存的 AV1 视频仍可经 CPU 回退处理。

启用 `[song_identification]` 后，管线优先用歌唱 ASR 在本地正式歌词库中做连续字符锚点
匹配；未命中才由独立 `ddgs` worker 搜索受支持歌词站并解析结构化完整歌词。搜歌与匹配
不调用 LLM，ASR、call 和未发行歌曲不会写入歌词库。匹配成功后在持久化的 Demucs 人声轨
上切成不超过 20 秒的窗口，由 pySHIRO 生成音素时间。英文歌词保留官方拼写作为显示文本，
同时用 `alkana` 生成隐藏的日式片假名发音，供 pySHIRO 日语模型近似对齐；无法可靠转读的专名不强制对齐。
译词优先复用库内官方或外部翻译，
缺失时才使用歌词专用 LLM 并写回库中。PaddleOCR、搜索和 pySHIRO 均运行在各自锁定的
worker 环境中。
原始 `source.qwen3-asr.srt` 始终保留；核验结果写入
`song-identification-cache.json`，修正版写入 `source.lyrics-corrected.srt`。单首识别
失败会保留原 ASR 并继续，不会阻塞非歌曲内容。

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
api_style = "chat_completions"
api_key_pass_entry = "api/deepseek"
model = "deepseek-v4-flash"
thinking = "disabled"
max_tokens = 16384
max_retries = 5
max_concurrency = 16
```

如需改用环境变量，将 `api_key_pass_entry = ""`，再通过 `api_key_env` 指定变量名。

也可以使用 OpenAI 原生 Responses API。需要 OpenAI Platform API Key；ChatGPT 网页版
订阅本身不等同于 API 额度：

```toml
[llm]
base_url = "https://api.openai.com/v1"
api_style = "responses"
api_key_pass_entry = "api/openai"
api_key_env = "OPENAI_API_KEY"
model = "gpt-5.6"
reasoning_effort = "low"
max_tokens = 16384
max_retries = 5
max_concurrency = 16
```

`responses` 模式会自动转换消息、JSON mode、函数工具、工具返回值、token 上限和 usage；
请求设置 `store = false`。不要同时设置 DeepSeek 专用的 `thinking`。

LLM HTTPS 请求会在系统 CA 基础上补充 `certifi` CA bundle，兼容 uv 独立 Python、
NixOS、macOS 和 Windows，同时保留 `SSL_CERT_FILE` 等自定义 CA 配置。

字幕使用单阶段联合 JSON 请求。Forced Aligner 单元首先按 ordinary diarization 的真实
时间交集归属 speaker；未归属片段依次经过同 speaker 桥接、Sudachi + GiNZA 句法回填和
0.5 秒最近 speaker 兜底，距离仍过大时删除并审计。随后按 ASR 记录的语言分流：日语使用
SudachiPy SplitMode.A 与 GiNZA；英语保留词间空格，使用英语标点、停顿和 spaCy English
tokenizer；混合语言保留原始空格，仅对日语片段运行 Sudachi。语言对应的语法证据与静音、
当前块时长一起打分，贪心合并为可供模型选择的本地单元。每个 speaker 建立独立时间轨，因此不同人物
的字幕可以重叠显示；同轨间隔达到 2 秒时建立硬 episode 边界。逐词人物归属、评分与形态
信息写入 `local-segmentation.json`。

每个 LLM 请求只处理一条 speaker 轨，模型同时选择左闭右闭的本地单元范围并翻译，例如
`{"start_id":0,"end_id":2,"text":"中文字幕"}` 表示覆盖 0、1、2。每个窗口都从 0 重新编号，范围必须连续、
无遗漏、无重复；单单元 cue 合法。响应通过校验后映射回 speaker 轨的全局单元 ID，缓存与
最终字幕仍使用全局 ID。源语言文本由本地按范围恢复，模型只返回中文。请求附带目标前后 5 秒的只读
对话上下文、视频信息和术语表，并明确 ASR 可能误听。歌声与 DiCoW `conditioned_speech`
保持不可拆分的原子单元，但使用相同响应契约。

进入 DiCoW 前，pyannote 匿名标签会先通过 ERes2NetV2 映射为人物；映射到同一人物的
多个匿名标签合并为一条人物活动掩码，再判断真正的多人重叠。无法确认人物的匿名标签
仍分别保留，原标签继续写入音频分析缓存供审计。

TARGET 以 160 个本地单元或 8000 源字符为上限，靠近上限时选择末段得分最高的本地边界。
各窗口按 `llm.max_concurrency` 并行，不再运行 Map 边界 Reduce。`cue-joint-cache.json`
会在每个窗口成功后立即原子更新；签名包含本地单元、speaker 轨、Sudachi/词典版本、评分
配置、提示词、模型、REFERENCE 和宽度限制。

JSON 等结构错误立即重试一次，再失败便递归缩窗。可无损转换为整数的字符串 ID 会先归一化。
范围遗漏、重复或乱序时，程序保留最长可信前后缀，并从已经确认的 cue 边界取错误区及前后
各一条作局部联合补丁，不重发整个窗口。空译文记录轨道、范围和源文后使用本地
`facebook/m2m100_418M` 本地机翻，并按 cue 语言选择 `ja` 或 `en` 源语言；残留日文先保护
REFERENCE 中的姓名、昵称和术语，再对未保护部分执行机翻。模型按需在 CPU 加载。
超宽译文只记录实际宽度和限制并继续，
不再触发 LLM 重试。
网络错误和超时使用带随机抖动的指数退避；HTTP 5xx 也采用相同策略，但耗尽重试后直接
终止而不缩小窗口。HTTP 429 优先遵守服务端的 `Retry-After` 响应头，并在规定等待时间
之后增加少量随机抖动；缺失该响应头时才使用带抖动的指数退避，同样在耗尽后直接终止而
不缩窗。其他 HTTP 状态视为非暂时性错误，首次遇到便直接终止。本地输出校验失败会立即
重试。

联合输入使用紧凑的 `<speaker>` 与 `<id>text`，绝对时间只在本地保存。固定规则、
术语表和视频信息位于请求前缀，只读对话上下文与窗口数据随后，重试错误放在末尾，以提高 DeepSeek
上下文缓存命中。日志会记录 `prompt_cache_hit_tokens`、`prompt_cache_miss_tokens`、
命中率及输出 token。

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

启用 `[audio_analysis]` 后，默认由 pyannote Community-1 同时生成 ordinary 和 exclusive
两条说话人时间轴。ordinary 时间轴保留真实重叠及参与者，用于身份聚合、审计和重叠 ASR
掩码；exclusive 时间轴保证每一时刻只有一个主说话人，供普通 Qwen ASR 使用。真实重叠
小于 0.4 秒时视作边界误差；0.4–1.0 秒仍走带完整上下文的 Qwen；严格超过 1.0 秒时，
使用固定 revision 的 DiCoW 对包含重叠前后完整讲话轮次的局部窗口逐 speaker 转写，并替换
整个局部基线结果。重叠时长不含 padding，DiCoW 不可用或遗漏活跃 speaker 时任务失败，
不会静默丢字幕。MOSS 仍可作为较慢的回归后端显式启用。

原音 AST 只提出歌声候选，Demucs 人声轨上的 AST 再确认实际歌唱；候选中存在
连续讲话但人声轨没有歌唱证据时按“讲话+BGM”处理。证据模糊时同时运行普通 forced
aligner 与句级歌声 ASR，时间轴坍缩或循环输出时采用歌声结果。release hysteresis 只在
确认进入歌唱状态后桥接短暂漏检，不会让一次 AST 误报扩成整段歌曲。所有音频分析结果和
带说话人字段的 cue sidecar 都写入 job 目录并可断点复用。

作业只用 ffmpeg 解码一次完整的 16 kHz 单声道音频，并由 `AudioBufferPool` 在共享 CPU
内存中提供给 pyannote、Qwen、DiCoW 和 ERes2NetV2。Qwen 与 DiCoW 直接接收 NumPy
切片或共享内存描述符，正常运行不再
创建 `asr-chunks` 或 `asr-analysis-chunks`。Demucs 只按需解码候选歌曲区间的高质量
立体声音频。各阶段日志包含耗时和可用时的峰值显存；pyannote 与原音 AST 可通过
`initial_analysis_concurrency = 2` 并行。Qwen 完成后会先释放显存再启动 DiCoW。

人物身份使用独立 uv worker 中的 ERes2NetV2 embedding 和余弦距离匹配。成员个人频道
的独播会自动将 2–15 秒、非歌唱且非重叠的主说话人片段注册到
`work/speaker-profiles-eres2netv2/`；每人最多保留 400 条。profile 带模型签名，不能与
旧 WeSpeaker embedding 混用。识别时保留原始样本，并用确定性的 cosine k-means 为
每人建立最多 5 个中心；每个中心默认至少需要 20 条样本，小簇会通过减少中心数重新
聚类。同一全局匿名 speaker 的非歌唱、非重叠干净片段会共同参与一次身份判断：
先按该标签 embedding 的 medoid 删除最远 15%，再按片段时长加权，单段权重最多计 10 秒。
重叠片段不参与投票，但会继承该匿名标签的身份。不同匿名标签独立匹配，允许都映射为同一
成员；匹配仍要求第一候选相对第二候选保持足够距离，证据不足时保留匿名标签。
MOSS 和 ERes2NetV2 使用独立 uv 环境，避免模型依赖影响 Qwen ASR。
共享内存、按需高质量立体声和受控并行的实现说明见
[音频管线内存与并行优化](docs/audio-pipeline-optimization.md)。

梦限大 MewType 五位成员的应援色与声纹配置位于独立的
`character_styles.json`，不会发给 LLM。独播频道会从无重叠语音自动建立声纹原型；
团播仅在余弦距离达到阈值时映射身份，否则保留匿名说话人和默认白色字幕。重叠人物
字幕会保留各自时间，并在 ASS 中分配不同垂直行。

结果位于 `work/<URL哈希>/translated.mp4`，译文位于
`work/<URL哈希>/translated.zh-CN.srt`，投稿标题、简介和最终标签位于
`work/<URL哈希>/translated.metadata.json`。先抽查专名、数字、断句和 ASR 可能
出现的幻觉，再启用上传。

默认 LLM 后端是由流水线按需管理的 llama.cpp，模型为
`unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M`。首次运行会下载约 16.5 GB；服务在 ASR
结束后启动，并在视频渲染前停止。服务日志写入
`work/<URL哈希>/local-llm-server.log`。也可把 GGUF 放入 `models/`，并在 `[llm]`
中用 `local_server_model_path` 指向它。

年龄限制、地区限制或需要登录的视频，可以在 `[download]` 配置
`cookies_from_browser = "chrome"`，或配置 Netscape 格式的 `cookies_file`。
`concurrent_fragments = 8` 会让 yt-dlp 并行下载 DASH/HLS 分片；普通单文件流不受影响。

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

默认 `copyright = 2` 表示转载，但不会另填转载来源，以免 B 站把 URL 自动加到简介首行。
上传使用 `biliup --user-cookie ... upload`，不会把 Cookie 内容放到命令行。
B 站简介默认限制为 1800 个字符且同时检查 UTF-16 长度，为服务端计数差异留出余量。
超长正文优先在段落或整行边界缩短，并为 `description_prefix` 预留空间；可通过
`upload.description_max_chars` 调整上限。`description_prefix` 中的 `{youtube_url}` 会在
上传时替换为当前任务的 YouTube URL。

原始 YouTube 标题明确含有 `歌枠` 且歌曲识别得到有效歌名时，默认在投稿成功后生成
`时间 歌名` 格式的顶层歌单评论。biliup 返回的 `aid`、`bvid`，评论接口响应及 `rpid`
保存在任务目录的 `bilibili-setlist-comment.json`。流水线等待投稿成功后 5 分钟，再通过
Playwright 和已登录的 Chromium 页面自动发布。每次发送前会分页检查当前登录账号是否
已经发布完全相同的顶层评论，避免重复留言；评论失败会记录接口响应和页面截图，但不会
把已经成功的投稿标记为失败。

## 自动切片

`clips` 是独立命令，只读取已经完成的任务目录，不会调用或修改主流水线。它根据直播
弹幕密度、独立观众数、重复反应及付费/会员事件发现候选，再让 LLM 在连续字幕范围内
选择完整上下文。歌回中有可靠识别和时间证据的歌曲会各自生成一个分 P，且不会再生成
与歌曲重叠的语音高光。

```bash
# 仅分析并生成切片
uv run subtitle-pipeline --config config.toml clips --no-upload \
  'https://www.youtube.com/watch?v=...'

# 生成后作为一个多 P 投稿上传
uv run subtitle-pipeline --config config.toml clips --upload \
  'https://www.youtube.com/watch?v=...'
```

命令要求 `work/<video-id>/` 已有 `translated.mp4`、最终字幕、翻译元数据和源视频信息；
缺少时会直接报错，不会补跑主流水线。结果写到 `work/<video-id>/clips/`：分析与审计数据
保存在 `analysis.json`，视频保存在 `parts/`。上传成功后会写入 `upload.json`；后续重复
运行可以补齐本地切片，但只要该记录存在就绝不再次上传。上传失败不会写成功记录。

配置项位于独立的 `[clips]`：

```toml
[clips]
upload = false
max_speech_seconds = 480
chat_peak_zscore = 3.0
chat_min_unique_authors = 5
```

更新任务列表为六个官方 YouTube 频道最近 14 天的公开直播录播（需要 Chromium
已登录 YouTube，会员限定和未开播视频会被排除）：

```bash
uv run python scripts/update-recent-yumemita-tasks.py
```

可先加 `--dry-run` 预览；通过 `--days`、`--browser` 调整时间范围和浏览器。
脚本保留 `work/yumemita-2026-08-10-uploaded.txt` 中的成功投稿历史，并原子更新下面
批处理脚本的 `RECORDS` 队列。队列按发布时间从新到旧排列，优先处理最新视频。

批量处理队列中的梦限大MewType公开直播录播：

```bash
./scripts/upload-recent-yumemita.sh
```

脚本会自动进入项目的 `nix develop .#default` 环境，因此从普通终端直接运行即可；
PyTorch、CUDA 和 `libstdc++` 等运行库会由开发环境提供。

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
- `asr.language`：默认不设置，由 Qwen 自动识别语言并保存到 cue/对齐单元；特定任务仍可显式强制语言。
- `asr.context`：提供节目、团体和专名背景，辅助识别罕见词。
- `asr.chunk_seconds` / `chunk_context_seconds`：控制可恢复分块和切点上下文；总输入
  长度不能超过 180 秒。
- `asr.max_new_tokens`：单块 ASR 最多生成的 token 数，默认 `2048`。
- `audio_analysis.diarization_backend`：默认 `pyannote`，使用 Community-1 的 ordinary
  diarization；`moss` 仅保留作回归比较。
- `audio_analysis.overlap_conditioned_asr_seconds`：默认 `0.5` 秒；更短重叠视为
  换人边界误差并保留 Qwen 基线，达到阈值后使用 DiCoW。
- `audio_analysis.conditioned_asr_model` / `conditioned_asr_revision`：长重叠局部修复所用
  DiCoW 模型及固定代码 revision。
- `audio_analysis.moss_window_seconds` / `moss_max_window_seconds`：MOSS 长窗目标和硬上限，
  默认 480/540 秒，避免长直播的注意力张量耗尽显存。
- `audio_analysis.initial_analysis_concurrency`：`1` 为顺序执行 diarization 与原音 AST；
  在显存和实测耗时允许时可设为 `2`。
- `audio_analysis.debug_audio_artifacts`：仅调试时持久化分离音轨，默认 `false`。
- `asr_correction.batch_windows` / `batch_chars`：单次 ASR 纠错请求的窗口数和原文字符上限，
  默认 `6` / `3000`。
- `asr_correction.max_tokens`：ASR 纠错响应的输出 token 上限，默认 `8192`。
- `asr_correction.context_before_seconds` / `context_after_seconds` /
  `context_max_chars`：ASR 纠错的只读前后文范围和字符上限。
- `segmentation.boundary_score_threshold`：本地候选边界的贪心切分阈值，默认 `3`。
- `segmentation.local_unit_max_seconds`：本地单元最长目标，默认 `6` 秒；超过时从 2 秒后的
  候选中选择最高分边界。
- `segmentation.model_window_units` / `model_window_chars`：划句请求的本地单元和源字符上限，
  默认 `240` / `3000`。
- `segmentation.max_tokens`：划句响应的输出 token 上限，默认 `8192`。
- `segmentation.dialogue_context_before_seconds` / `dialogue_context_after_seconds` /
  `dialogue_context_max_chars`：划句阶段的只读前后文范围和字符上限。
- `translation.batch_cues` / `batch_chars`：单次固定 cue 翻译请求的 cue 数和源字符上限，
  默认 `32` / `3000`。
- `translation.max_tokens`：翻译、歌词和元数据响应的输出 token 上限，默认 `8192`。
- `translation.context_before_seconds` / `context_after_seconds` / `context_max_chars`：
  翻译阶段的只读前后文范围和字符上限。
- `llm.max_retries`：同一窗口、边界或定点修复请求的重试次数，建议设为 `5`。
- `llm.max_concurrency`：窗口和边界 LLM 请求的最大并发数，默认 `16`。
- `translation.local_model` / `local_device`：空译文和残留源文使用的本地
  后备机翻模型及设备，默认 `facebook/m2m100_418M` / `cpu`，并按 cue 语言选择 `ja`/`en`。
- `llm.thinking`：DeepSeek V4 的严格 JSON 翻译应设为 `"disabled"`；其他服务不支持该参数时省略。
- `translation.translate_metadata`：是否翻译 YouTube 标题和简介。
- `translation.metadata_description_max_chars`：发送给 LLM 的源简介字符上限。
- `translation.metadata_tag_count`：同一次元数据翻译请求生成的 B 站标签数量。
- `translation.metadata_subtitle_max_chars`：用于识别内容/IP 的字幕首、中、尾证据字符上限。
- `translation.ip_aliases_file`：已知 IP 的规范名及中英日别名 JSON 文件。
- `translation.glossary_files`：附加翻译术语表；后加载的自定义译名覆盖内置译名。
- `render.font_name`：必须是机器上已安装且包含中文字形的字体。
- `render.font_size_ratio` / `portrait_font_size_ratio`：横屏与竖屏字号相对于视频短边的比例，并受最小/最大字号限制。
- `render.margin_horizontal_ratio` / `portrait_margin_horizontal_ratio`：横屏与竖屏左右安全边距各自占视频宽度的比例。
- `render.margin_vertical_ratio`：字幕底边距占视频高度的比例。
- `render.outline_ratio`：字幕描边相对于视频短边的比例，默认 `0.0045`。
- `upload.enabled`：生产环境才建议开启；命令行 `--no-upload` 始终优先关闭上传。
- `upload.tags`：始终保留的固定标签；会与自动标签去重合并。
- `upload.max_tags`：投稿使用的固定标签与自动标签总数上限。
- `upload.tag_catalog_file`：经授权获取或人工维护的 B 站规范标签与热度目录。
- `upload.description_max_chars`：简介的保守字符上限，同时作为 UTF-16 单位上限。
- `upload.song_setlist_comment`：仅对原始标题含 `歌枠` 的投稿生成并幂等发布歌单评论。
- `upload.cooldown_min_seconds` / `cooldown_max_seconds`：成功投稿后写入下一次投稿的
  随机冷却期限，默认 `60–120` 秒，跨进程生效。
- `upload.rate_limit_retry_delays_seconds`：biliup 输出 `406/429` 且未提供
  `Retry-After` 时的重试间隔；每次会重新执行完整上传。
- `upload.pause_marker_file`：遇到 B 站 `412` 风控时写入的暂停标记；删除标记前拒绝
  后续投稿。

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

可通过 `translation.glossary_files` 添加同格式 JSON；自定义文件在内置术语之后加载，所以
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
