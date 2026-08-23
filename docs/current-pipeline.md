# 当前字幕管线与技术选型

本文描述截至 2026-08-23 的实际实现。配置以 `config.toml` 与
`src/subtitle_pipeline/config.py` 为准；实验计划见
[音频管线内存与并行优化](audio-pipeline-optimization.md)。

每次运行都会在任务目录写入 `performance.json`。其中 `stages` 保留各阶段及各批次的
墙钟耗时、状态和可获取的 PyTorch 峰值显存，`summary` 汇总同名阶段的调用次数、总耗时
和最大耗时；`pipeline.total` 是完整管线墙钟时间。报告使用临时文件原子替换，管线失败
或被中断时也会保留已完成阶段和错误原因。

```mermaid
flowchart TD
    A[YouTube 视频与元数据] --> B[yt-dlp 下载 VP9 或 H.264 视频]
    B --> C[ffmpeg 一次解码<br/>16 kHz 单声道 float32]
    C --> D[(共享 CPU 内存<br/>AudioBufferPool)]

    D --> E[pyannote Community-1<br/>ordinary diarization]
    D --> F[AudioSet AST<br/>5 秒窗 / 2.5 秒步长]
    E --> G[ERes2NetV2 声纹匹配<br/>匿名 speaker 映射到成员]
    F --> H{原音歌唱候选?}
    H -->|是| I[Demucs 按候选区间提取人声]
    I --> J[人声轨再次运行 AST]
    H -->|否| K[讲话区域]
    J --> L{歌曲路由仲裁}
    E -. 讲话覆盖证据 .-> L

    L -->|讲话| M[讲话窗口规划<br/>目标 60 秒 / 硬上限 90 秒]
    L -->|确认歌唱| N[HeartTranscriptor 歌词 ASR<br/>目标 10 秒 / 6–15 秒]
    L -->|证据模糊| O[讲话与歌声双路识别<br/>按时间轴质量选择]
    K --> M

    M --> P[Qwen3-ASR-1.7B<br/>整窗转写 + 自动语言识别]
    P --> Q[Qwen3-ForcedAligner-0.6B<br/>细粒度词素时间轴]
    Q --> R[按 diarization 时间交集<br/>赋予 speaker]
    G --> R

    E --> S{真实重叠 >= 0.5 秒?}
    S -->|是| T[DiCoW v3.3<br/>按 speaker 分路的段级转写]
    G --> T
    S -->|否| U[保留 Qwen 基线]
    T --> V[保留 DiCoW 段级边界<br/>异常时回退 Qwen]
    R --> W
    U --> W

    N --> W[统一带时间源单元]
    O --> W
    V --> W
    W --> X[清理非语音标记]

    X --> Y{启用歌曲识别?}
    Y -->|是| Z[本地正式歌词库优先<br/>未命中才确定性联网检索]
    Z --> AA[Qwen 粗定位 + 连续字符锚点<br/>pySHIRO 人声轨精对齐]
    Y -->|否| AB[保持原 ASR]
    AA --> AC[待翻译源单元]
    AB --> AC

    AC --> AD{源单元类型}
    AD -->|普通讲话| AE[按语言分流形态与边界分析<br/>按 speaker 建立独立时间轨]
    AD -->|DiCoW 重叠讲话| AY[保留不可拆分的 DiCoW 原子单元]
    AD -->|歌唱| AX[保留不可拆分的歌唱原子单元]
    AE --> AF[单阶段 LLM 划句 + 翻译<br/>左闭右闭范围]
    AY --> AF
    AX --> AZ[优先库内官方/外部译词<br/>缺失时歌词专用 LLM]
    AZ --> AF
    AF --> AG[同轨局部联合修复<br/>完整性与宽度校验]

    AG --> AH[source.semantic.srt<br/>translated.zh-CN.srt]
    AH --> AI[翻译标题、简介并生成标签]
    AH --> AJ[ASS 自适应布局与人物样式]
    AJ --> AK{渲染后端}
    AK -->|优先| AL[libass 稀疏位图 + CUDA 混合<br/>NVDEC + NVENC]
    AK -->|回退| AM[ffmpeg libass + libx264]
    AL --> AN[translated.mp4]
    AM --> AN
    AI --> AO[translated.metadata.json]
    AN --> AP{允许上传?}
    AO --> AP
    AP -->|是| AQ[biliup 投稿 Bilibili]
    AP -->|否| AR[保留本地产物]
```

## 1. 下载与音频承载

`yt-dlp` 下载视频及 `source.info.json`。当前格式顺序优先 VP9，其次 H.264，原因是
RTX 2080 Ti 可以硬解这两种编码，但不能硬解 AV1。已经存在的源视频会直接复用。

进入语音阶段后，ffmpeg 将完整音轨一次性解码成 16 kHz、单声道、`float32` PCM。
`AudioBufferPool` 把它放入共享 CPU 内存；pyannote、Qwen、DiCoW 和声纹 worker
通过描述符或 NumPy 切片读取同一份数据，正常运行不为每个区间反复写临时 WAV。

Demucs `htdemucs` 是例外：歌曲分离需要较高质量的音频，因此只对 AST 提出的歌曲候选区间按需
读取源音频并提取人声，而不是将整部高采样率立体声音频常驻内存。

## 2. 音频分析

### 2.1 说话人时间轴

当前模型为 `pyannote/speaker-diarization-community-1`，管线只采用 ordinary 时间轴：

- **ordinary diarization**：保留真实重叠以及重叠中的匿名 speaker，用于普通 Qwen
  讲话窗口、词素人物归属、重叠检测和 DiCoW 掩码。
- 声纹身份阶段从每位 speaker 的 ordinary turn 中减去所有多人重叠交集，只将剩余的
  干净单人片段作为身份匹配骨架。exclusive 时间轴不再读取或参与任何决策。

当前 `initial_analysis_concurrency = 2`，所以 pyannote 与原音 AST 并行运行。

### 2.2 人物身份

匿名 speaker 使用 `iic/speech_eres2netv2_sv_zh-cn_16k-common` 提取 embedding，和
`work/speaker-profiles-eres2netv2/` 中的五位梦限大成员声纹比较余弦距离。

声纹库保留独播中的多条干净样本，并为每人建立最多 5 个聚类中心，而不是把所有状态
压成一个平均 embedding。身份判断汇总同一匿名 speaker 的非歌唱、非重叠片段，删除
距离 medoid 最远的 15%，再按片段时长加权。不同匿名标签可以映射到同一人物，不执行
强制一对一分配；证据不足时保持匿名并使用默认字幕样式。

### 2.3 歌曲分流

歌曲分流的核心模型是 `MIT/ast-finetuned-audioset-10-10-0.4593`。它是 AudioSet
音频事件分类器，并非专用歌曲边界模型。

当前算法为：

1. 原音按 5 秒窗口、2.5 秒步长运行 AST。
2. 分别保留歌唱、讲话和音乐三类证据。speech 不再从 singing 中相减，因此 call 和带
   讲话特征的歌唱不会因为 speech 分数较高而直接归零。
3. 对连续 3 个窗口分别取中位数。稳定 singing 证据会建立歌唱锚点；当原始 singing
   已超过阈值且同时存在音乐证据时，也允许保留被中位数滤掉的短歌唱锚点。
4. 只有歌唱锚点能进入歌曲状态。单纯“讲话 + BGM”不能自行建立歌曲，避免整场直播因
   背景音乐被送入歌声路径。
5. 进入歌曲状态后，音乐证据负责覆盖 call、间奏和短暂讲话；只有新的歌唱锚点会刷新
   35 秒 release。若最后一个锚点后出现明确的普通讲话接管，终点回填到讲话开始处。
6. Demucs `htdemucs` 提取候选区间的人声轨，再次运行 AST，以 `0.15` 阈值确认实际
   歌唱。有人声轨歌唱锚点的短歌也可以确认，不再仅因不足 30 秒而丢弃。
7. 仲裁后输出 `speech`、`singing` 或 `ambiguous`。歌曲区间内有声音的人声片段走句级
   Qwen 歌曲路径，纯间奏不生成字幕，不会切回普通 Forced Aligner。

这里的 `singing` 表示“歌曲表演区间”，不是“当前 5 秒全部在唱”。状态机每个窗口的
singing、speech、music 分数、状态和转换原因都会写入 `audio-analysis.json` 的
`song_detection` 字段，便于复核边界。2026-08-17 的两个失败样例用于真实回归：歌枠
约 `06:40–10:42.5` 被识别为一个连续区间，内部 call 不再切断；开头自我介绍歌也能从
`00:00` 进入歌曲路径。

## 3. 三条 ASR 路径

### 3.1 普通讲话：Qwen + Forced Aligner

相邻讲话区间在间隔不超过 2 秒时合并。窗口目标约 60 秒，允许根据自然边界延长，硬上限
90 秒；窗口中最多允许 15 秒静音且语音覆盖率至少为 60%。前后各附带 2 秒上下文，只有
核心区间的词素归属于该窗口。

当前固定批量大小为 4。`Qwen/Qwen3-ASR-1.7B` 不强制语言，自动识别并在每个 cue 和
对齐单元中记录 `Japanese`、`English` 或 `mixed`。
`Qwen/Qwen3-ForcedAligner-0.6B` 只对 **Qwen 文本**产生细粒度时间轴，不处理 DiCoW
文本。词素 speaker 首先按 ordinary diarization 的真实时间交集判断：单 speaker 的
交集至少覆盖词素时长 30% 时直接归属；同时与多个 speaker 相交时取总交集最大的一个。
这里不再扩张 diarization 边界，因此不会使用人工的 0.1 秒容差制造交集。

仍未归属的普通讲话词素在进入 LLM 前分层处理。短 unknown 段左右是同一 speaker 时
优先桥接；左右人物不同时，Sudachi 的词性/活用连接必须与 GiNZA 的依存关系和文节分析
共同指向同一侧，才执行 `grammar_left` 或 `grammar_right` 回填。最后仍无归属的词素交给
时间最近的 ordinary diarization speaker，距离不超过 0.5 秒时记为 `nearest_fallback`；
超过 0.5 秒则记为 `discarded`，不进入本地划句、LLM 或渲染。具名与匿名 speaker 都能
成为归属目标。

本地检查非单调时间、相同起点坍缩、异常长词素、空结果和重复循环。异常窗口优先在已有
时间边界附近递归缩短，子窗口不能短于约 15 秒；仍不能得到可信时间轴时整条任务停止，
不会带着缺失字幕继续渲染。

### 3.2 长重叠：DiCoW

ordinary diarization 中真实同时讲话达到 `0.5` 秒时，管线把重叠前后各 2 秒的完整讲话
上下文交给固定 revision 的 `BUT-FIT/DiCoW_v3_3`。当前批量大小为 4，单窗口不能超过
30 秒。

DiCoW 输出按 speaker 分路的**段级**日文文本与时间，不经过 Qwen Forced Aligner。
这些段落标记为 `conditioned_speech`，并作为联合翻译中的不可拆分原子单元：

- 在建立 DiCoW 窗口前，先将 ERes2NetV2 映射到同一人物的多个 pyannote 匿名标签合并为
  一条人物活动掩码；重叠判断和条件转写使用人物 ID，原匿名标签仅保留作审计。
- DiCoW 正常时，用其分路结果替换相应 speaker 的局部 Qwen 基线。
- DiCoW 遗漏某个活跃 speaker 时，保留该 speaker 的 Qwen 基线。
- DiCoW 出现强重复循环时，丢弃异常结果并保留 Qwen 基线。
- 正常 DiCoW 段直接进入联合翻译，翻译模型结合术语和相邻 cue 处理可能的误听，
  但不能合并或拆分 DiCoW 时间段。

当前 DiCoW worker 尚未启用 token timestamp，因此不能把其文本描述为词级对齐结果。

### 3.3 歌唱与模糊区域

确认歌唱区间先在 Demucs 人声轨上使用 HeartTranscriptor-oss 作粗转写。离线窗口目标为
10 秒，允许 6–15 秒并保留 1 秒重叠；在目标点前后 4 秒内依次优先选择 Demucs 人声轨的非歌唱
gap、平滑 RMS 能量谷值和 phrase 边界，没有证据时才硬切，同时重新平衡不足 6 秒的
尾窗。各窗口通过文本重叠去重，HeartTranscriptor 使用自动语言识别，不接收普通讲话的
context 或强制语言，以避免在模糊歌声或间奏中复述直播介绍。HeartTranscriptor 不负责
最终时间轴，Qwen Forced Aligner 也不处理歌唱；正式歌词命中后改由 pySHIRO 在持久化的
Demucs 人声候选轨上对不超过 20 秒的片段执行音素对齐。

`ambiguous` 区域同时尝试普通讲话对齐和歌声句级转写，再根据时间轴健康度选择结果。
任何路径出现长重复循环都会被拒绝。

## 4. 歌名识别与歌词校正

歌曲分流解决“用哪条 ASR 路径”，歌曲识别解决“唱的是哪首歌”，两者不是同一阶段。

启用 `[song_identification]` 后，管线先把歌唱 ASR 规范化为假名，在本地 SQLite 歌词库
中做字符级模糊匹配。只有多条连续锚点达到阈值才确认歌曲和演唱片段。未命中时根据稳定
OCR 和较长 ASR 片段生成搜索词，由独立 `ddgs` worker 搜索受支持歌词站；当前 UtaTen
适配器从结构化页面提取歌名、歌手和完整分行歌词，再用同一套本地匹配验证。

搜索和歌词匹配不调用 LLM。只有带外部来源、结构化完整正文且通过连续锚点验证的已发行
歌曲可以写入 `work/lyrics/library.sqlite3`；ASR 文本、call、ad-lib、喊声和搜不到的
未发行歌曲都不能写库。库中每首歌曲只保存一份完整标准歌词，视频只唱一部分时也不会
裁短库内正文。

匹配后使用 HeartTranscriptor ASR anchor 粗时间定位，在 Demucs 人声轨上切成不超过 20 秒的窗口，再由固定提交
的 pySHIRO 日文 v2 模型生成音素和歌词行时间。日文歌词直接使用官方读音或 Sudachi 读音；
英文歌词的官方拼写仍用于显示和翻译，仅在对齐请求中用 `alkana` 转成日式片假名发音。因为仍是
日语声学模型，这是对日本歌曲中英语唱法的近似对齐；词典无法转读的英文专名会放弃动态对齐，不伪造时间。
对齐审计同时保存歌词语言和隐藏读音。未匹配歌词的 call、ad-lib 和喊声被忽略；
句法上明确是普通讲话的片段会重新读取原始混音，用普通 Qwen + Forced Aligner 路径转写。

歌词定位采用半全局动态规划：完整使用按时间排列的歌唱 ASR，但允许免费跳过官方歌词的
开头和结尾，因此只唱副歌或歌曲中段时仍能定位。同一歌曲片段连续播放多遍时，匹配器会
反复提取时间上互不重叠的单调路径；每条路径是独立 take，可重新从完整歌词任意位置开始，
但一次 take 内不允许歌词倒序。ASR anchor 只作为高置信起点。路径内部跳过的歌词行，
以及相邻 take 之间上一轮的后继歌词和下一轮前缀，可以直接在 Demucs 人声轨上通过 pySHIRO
声学对齐补全，不再要求 ASR 再次听出歌词。内部缺口会比较仅含相邻锚点与加入缺失行两种
对齐的每帧似然；轮次边缘还会校验歌词密度、人声活动、时间覆盖和逐单元时长。只有绝对似然
合格、音素时长合理且时间轴单调时才补回缺失歌词。每次接受或拒绝及其证据均写入歌曲对齐审计。

译词优先级为同版本官方中文、已验证逐行外部翻译、歌词专用 LLM、本地机翻。缺失译词
通过专用提示词按完整歌曲逐行翻译并写回歌词库，记录模型、来源与日文歌词哈希。命中库内
译文的歌唱 cue 直接使用固定译文，不再进入通用对话翻译。

## 5. 本地分段与联合翻译

普通讲话完成上述归属后按 speaker 建立独立时间轨，不再为 unresolved unknown 建轨。
同一 speaker 的相邻发言间隔达到 2 秒会建立硬边界。这样 A 可以跨过 B 的短插话继续
组成一句，同时不同人物最终可以拥有相互重叠的字幕时间。

每个 episode 按语言路由分析。日语使用 SudachiPy `SplitMode.A` 补充词性、活用型和活用形，
并使用 GiNZA 依存证据回填 speaker；英语保留词间空格，使用英语标点、话语起始词和 spaCy
English tokenizer；混合语言保留原始空格，仅对日语字符片段运行 Sudachi，不把跨语言边界当作
日语语法连接。相邻 Forced Aligner 单元的候选边界累计评分：静音 120/250/400/600 ms
分别贡献 1/2/3/4 分；当前块达到 2 秒加 1，达到 4 秒再加 2；再叠加当前语言的终止、
起始和强连接证据。
总分达到 3 时贪心切分。若 6 秒仍没有切点，则从当前块 2 秒之后选择最高分边界，分数
相同取最靠后的一个。Qwen 标点不参与评分。

`local-segmentation.json` 保存每个词素的语言、speaker 归属原因、最近 speaker 距离、回填与
删除记录，以及每个边界候选的总分、命中因素、两侧形态证据、Nagisa POS 和最终
本地单元范围。Sudachi、GiNZA 或所需词典不可用时直接终止，不静默退化。

LLM 使用单阶段划句与翻译。每个请求只覆盖一条 speaker 轨，TARGET 采用紧凑格式：

```text
DIALOGUE_CONTEXT:
<speaker_b>只读的邻近发言

TARGET:
<speaker_a>
<0>本地源语言单元
<1>下一个本地源语言单元
```

Context 按真实时间排列，覆盖目标前后各 5 秒且最多 4000 字符；超限时优先保留最近内容。
TARGET 最多 160 个本地单元或 8000 个源字符，触及上限时在末段选择评分最高的边界作为
硬窗口边缘，不再运行 Boundary Reduce。

模型同时选择自然 cue 范围并翻译，返回左闭右闭的 JSON：

```json
{"cues":[{"start_id":0,"end_id":2,"text":"中文字幕"}]}
```

每个请求窗口都使用从 0 开始的相对 ID；范围必须连续、无遗漏、无重复地覆盖 TARGET，且
单单元 cue 使用 `end_id = start_id`。下一条 cue 的 `start_id` 必须等于上一条的
`end_id + 1`。校验成功后闭区间相对 ID 转换为内部半开范围，再映射回 speaker 轨全局 ID；缓存和最终字幕
仍保存全局 ID。每个窗口显式附带 `SOURCE_LANGUAGE`，TARGET 被说明为
`source-language ASR evidence`。源文由本地按范围恢复，模型不能回传或修改源文；ASR 纠错只反映在中文译文。`singing` 与
`conditioned_speech` 各自作为不可拆分、不可跨越的单单元 TARGET，但仍使用同一契约。

JSON 等结构错误立即重试一次，再失败便递归缩窗。整数形式的字符串 ID 会先归一化。范围
遗漏、重复、乱序或结尾未覆盖时，程序计算最长可信前缀和后缀，从已确认 cue 边界取错误区
及前后各一条作为局部补丁；补丁成功后覆盖该区，窗口其余结果保持不变。

空译文不会重试 LLM：程序记录轨道、全局单元范围和源文，再调用按需加载的本地
`facebook/m2m100_418M` 机翻。源语言为英语时使用 `en`，其他日语/混合情况使用 `ja`。译文残留日文时，
先把 REFERENCE 中的姓名、昵称和术语
切出保护，再仅机翻未保护部分并拼回。默认在 CPU 推理，避免与 ASR 争用显存。超宽译文
记录实际宽度、限制和文本后照常进入字幕，不再触发重试。网络超时、HTTP 429 和 5xx 沿用
Retry-After 或带 jitter 的指数退避，耗尽后直接终止且不缩窗；HTTP 400、401、403 等
非暂时错误立即终止。

运行时字幕提示词为 `src/subtitle_pipeline/prompts/joint-segment-translate.md` 和批量小窗口使用的
`src/subtitle_pipeline/prompts/joint-segment-translate-batch.md`。模板哈希、
本地单元、评分配置、Sudachi/词典版本、本地机翻模型、LLM、REFERENCE 和显示宽度都进入
缓存签名。

## 6. 字幕、元数据与渲染

处理结果分别生成：

- `source.semantic.srt`：按最终 cue 边界恢复的日文审计字幕；
- `translated.zh-CN.srt`：简体中文字幕；
- `translated.metadata.json`：标题、简介、内容摘要、歌曲报告和 B 站标签。

相邻 cue 时间重叠时，前一条在后一条开始时立即结束。最终 cue 的人物取其中已知源单元
数量最多的人物，unknown 不参与多数计算；并列或全部 unknown 时使用默认白字黑边样式。

字号按视频短边计算：横屏 6.6%，竖屏 7.7%，限制在 28–144。提示词安全边距为横屏左右
各 7.5%、竖屏各 2.5%，底边距为画面高度 5%；实际 ASS 左右边距统一为 1 px。描边为
短边的 0.45%。人物字幕使用白字和对应应援色描边；样式来自 `character_styles.json`，不会发给翻译 LLM。
歌唱 cue 在保留人物颜色的基础上增加下划线，以便与讲话字幕区分。

渲染前删除每个显示行末普通逗号、句号、分号、冒号和顿号，但保留问号、感叹号和省略号；
该操作不修改审计 SRT。

`render.backend = "auto"` 优先使用原生 `ass-cuda-render`：libass 在 CPU 只生成字幕文字、
描边和阴影的小型位图，CUDA 将其混合到 NVDEC 帧，再交给 NVENC。输入编码或显卡不支持
时回退到 ffmpeg `libass + libx264`。最后由 ffmpeg 将原始音轨和元数据无损封装回成片。

## 7. 缓存与恢复

每个 URL 使用 SHA-256 前 12 位作为 `work/<job-id>/`。主要缓存如下：

| 文件 | 内容 | 恢复粒度 |
|---|---|---|
| `audio-analysis.json` | speech、singing、ambiguous、diarization | 整个分析签名 |
| `asr-analysis-cache.json` | 各讲话/歌唱区域及递归子范围结果 | 单窗口或子窗口 |
| `conditioned-asr-cache.json` | DiCoW 重叠分路结果 | 全部重叠窗口签名 |
| `song-ocr-cache.json` | 每首歌的 OCR 候选 | 歌曲集合签名 |
| `song-identification-cache.json` | 歌名、来源与歌词对齐报告 | 歌曲集合及歌词库签名 |
| `work/lyrics/library.sqlite3` | 完整标准歌词、逐行译词及来源 | 持久歌词库，不存 ASR |
| `song-alignment/*/alignment.json` | pySHIRO 音素与歌词行时间 | 视频音频和歌词匹配 |
| `local-segmentation.json` | 本地边界评分、Sudachi 形态与 speaker 轨 | 整个源时间轴 |
| `cue-joint-cache.json` | 联合划句翻译窗口与最终结果 | 单 speaker 窗口 |
| `manifest.json` | 最终产物与上传完成状态 | 整个作业 |

缓存签名包含相关模型、配置、源时间轴和提示词版本。签名变化时不会误用旧结果；成功结果
使用临时文件原子替换，中断后只补未完成部分。共享内存只用于当前进程生命周期，不属于
持久缓存，退出时由所有者 `close()` 并 `unlink()`。

## 8. 当前主要技术风险

1. **歌曲路由召回率**：通用 AudioSet AST 对 call、自我介绍歌和讲话特征明显的歌曲
   不够可靠，且当前差分打分与中位数平滑会进一步降低召回率。
2. **歌声时间轴粒度**：歌唱路径只有句级窗口时间，不提供可靠词级对齐。
3. **重叠语音文本**：DiCoW 能分 speaker，但目前只有段级时间；严重串音仍可能导致遗漏
   或幻觉，因此保留 Qwen 混合基线作为证据和回退。
4. **说话人身份域偏移**：ERes2NetV2 预训练域与日语 VTuber 直播并不完全一致，角色声、
   情绪变化、BGM 和压缩失真都会增大声纹距离。
5. **LLM 尾延迟**：少数 Planner 窗口或翻译批次可能出现长输出、格式错误或内容校验失败；
   分阶段并发、独立缓存、快速缩窗和定点修复降低了影响，但不能消除服务端波动。

## 9. 代码入口

- 总编排：`src/subtitle_pipeline/pipeline.py`
- 共享音频：`src/subtitle_pipeline/audio_buffer.py`
- 音频分析与歌曲分流：`src/subtitle_pipeline/audio_analysis.py`
- 声纹身份：`src/subtitle_pipeline/speakers.py`
- Qwen ASR 与 Forced Aligner：`src/subtitle_pipeline/asr.py`
- DiCoW 重叠修复：`src/subtitle_pipeline/conditioned_asr.py`
- 歌名识别：`src/subtitle_pipeline/song_identification.py`
- 本地候选分段：`src/subtitle_pipeline/local_segmentation.py`
- 联合划句与翻译：`src/subtitle_pipeline/joint_translation.py`
- ASS 与视频渲染：`src/subtitle_pipeline/media.py`
- Bilibili 投稿：`src/subtitle_pipeline/upload.py`
