# 粉丝知识库

知识库使用 `databases/fan-knowledge.sqlite3` 长期保存资料，歌词库使用 `databases/lyrics.sqlite3`。`databases` 是指向持久数据盘的软链接。JSON 词表只提供稳定名称和人工确认规则；历史直播、聊天、SC、SNS 和官网资料都作为文档增量写入 SQLite。

## 导入现有任务

```bash
subtitle-pipeline --config config.toml knowledge ingest-work
```

该命令读取每个任务的 YouTube metadata 和 SC，不导入本项目生成的 ASR 或字幕正文。普通观众 chat 默认不写入长期知识库；重复运行时根据内容哈希跳过未变化文档。

单视频字幕管线会另行抓取有限数量、按 YouTube“热门评论”排序的顶层评论。
达到点赞阈值的评论以 `youtube_comment` 低可信背景入库，可参与当前视频的主题
检索，但不会被当作主播原话或官方事实。抓取数量、点赞阈值和最终保留数量由
`[download]` 中的 `top_comment_*` 配置控制。

## 获取历史直播字幕

视频、播放列表和频道 URL 都可交给同一命令：

```bash
subtitle-pipeline --config config.toml knowledge ingest-youtube \
  'https://www.youtube.com/@channel/streams' \
  --browser chromium \
  --playlist-end 100
```

这里只下载 metadata、人工/自动字幕和直播聊天，不下载视频。人工字幕优先于自动字幕；完整 chat 仅作临时输入，SC 入库后立即删除。

## 导入 SNS 和官网资料

官网页面及 RSS/Atom Feed 可以直接增量采集：

```bash
subtitle-pipeline --config config.toml knowledge ingest-official \
  'https://example.jp/news/feed.xml' \
  --source-type official_news
```

没有 Feed 的归档页可受控遍历同域公告链接：

```bash
subtitle-pipeline --config config.toml knowledge ingest-official \
  'https://example.jp/news/' \
  --source-type official_news \
  --follow-links --link-pattern '/news/' --maximum-documents 1000
```

X 和 Instagram 使用 `gallery-dl` 的 metadata-only 模式，不下载图片或视频：

```bash
subtitle-pipeline --config config.toml knowledge ingest-sns \
  'https://x.com/account' \
  'https://www.instagram.com/account/' \
  --browser chromium
```

其他采集器也可以统一输出 JSONL，再导入：

```bash
subtitle-pipeline --config config.toml knowledge ingest-jsonl \
  work/knowledge/import/x.jsonl \
  work/knowledge/import/official.jsonl
```

每行格式：

```json
{"source_type":"official_event","external_id":"event-2026-08","source_url":"https://example.jp/news/1","title":"活动标题","text":"正文","author":"官方名称","published_at":"2026-08-01T00:00:00+09:00","language":"ja","reliability":0.95,"metadata":{"people":["成员 ID"]}}
```

推荐 `source_type`：`x_post`、`instagram_post`、`official_news`、`official_event`、`concert`、`promotion`。网页抓取器只负责取得原始资料和出处；切块、去重、FTS 索引及检索审计由知识库统一处理。

## 萃取术语

新增或变化的文档会进入术语萃取队列，但视频字幕流水线不会自动审核或新增
term。需要维护术语库时显式运行：

```bash
subtitle-pipeline --config config.toml knowledge extract-terms
```

程序先使用 Sudachi、GiNZA、引号、标签及片假名/英文/混合文字模式，从全部待处理文档中提取原样候选。候选在调用 LLM 前按规范化词形跨文档累计，汇总出现次数、文档数、来源分布和最多 8 条代表性上下文。SC 用户名会在本地提取前移除。

LLM 只能筛选程序给出的候选，不能自行新增或改写词形。它为保留项返回稳定中文译法，并选择 `accept` 或 `search`：证据充分的 `accept` 直接入库；只有 `search` 才使用模型给出的查询词联网，最多保留 4 条标题、摘要和 URL，再针对单个候选做一次确认。未返回的候选视为拒绝。首轮不推断 alias 或 ASR 误听关系；别名只来自人工词表或后续有可靠书面证据的独立处理。

自动 term 只是 ASR 纠错和翻译 LLM 的检索参考，不是强制替换规则。人工 JSON 词表对应 `verified`，冲突时始终优先；LLM 复核通过的自动 term 为 `active`。最终是否进入某个提示词仍由窗口级 RAG 相关度决定。

每个候选给 LLM 的内容为：原样候选、出现次数、文档数、来源计数，以及候选附近的代表性文本。上下文优先采用官网、视频 metadata、X/Instagram 和人工字幕，再补充自动字幕与 SC。提示词不包含 document ID、chunk ID 或精确时间；数据库内部仍保留命中 chunk，用于来源追溯。

DeepSeek 默认使用 262144 token 总上下文、220000 token 目标输入和 16384 token 最大输出；本地模型自动受 llama-server context 上限约束。候选记录达到输入预算时按完整候选切批，不拆开单个候选。首轮不限制 term 数量，但要求严格排除普通词、临时称呼、偶然提及和证据不足的候选。

本地候选出现记录会跨增量运行保留。某个候选今天只出现一次时不会调用 LLM；以后在另一个逻辑文档中再次出现后，程序会合并历史上下文再筛选。搜索查询、结果、错误和最终确认结果写入 `knowledge-term-extraction-audit.jsonl`。不存在对所有 active term 再统一搜索的第二遍流程。

## 查看状态

```bash
subtitle-pipeline --config config.toml knowledge stats
```

完整字幕流水线只按配置执行增量资料更新，不运行 term 萃取。术语新增仅由
`knowledge extract-terms` 独立命令或人工维护触发。已经处理且内容未变化的资料不会重复抓取。每次召回及各项分数记录到任务目录的 `fan-knowledge-audit.jsonl`。
