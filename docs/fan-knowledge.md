# 粉丝知识库

知识库使用 `databases/fan-knowledge.sqlite3` 长期保存资料，歌词库使用 `databases/lyrics.sqlite3`。`databases` 是指向持久数据盘的软链接。JSON 词表只提供稳定名称和人工确认规则；历史直播、聊天、SC、SNS 和官网资料都作为文档增量写入 SQLite。

## 导入现有任务

```bash
subtitle-pipeline --config config.toml knowledge ingest-work
```

该命令读取每个任务的 YouTube metadata、讲话 ASR 和 SC。普通观众 chat 默认不写入长期知识库；重复运行时根据内容哈希跳过未变化文档。

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

## 查看状态

```bash
subtitle-pipeline --config config.toml knowledge stats
```

字幕流水线不会自动联网扩充知识库。它只读取已经入库的资料，并将每次召回及各项分数记录到任务目录的 `fan-knowledge-audit.jsonl`。
