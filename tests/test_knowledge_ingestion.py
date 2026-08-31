from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from subtitle_pipeline.chat_context import (
    read_youtube_live_chat,
    remove_youtube_chat_files,
)
from subtitle_pipeline.config import DownloadConfig
from subtitle_pipeline.fan_knowledge import FanKnowledgeRetriever, KnowledgeQuery
from subtitle_pipeline.knowledge_ingestion import (
    chunk_timed_cues,
    chunk_youtube_chat,
    download_youtube_subtitles,
    ingest_jsonl,
    ingest_work_directory,
    ingest_youtube_cache,
    ingest_youtube_top_comments,
)
from subtitle_pipeline.subtitles import Cue


class KnowledgeIngestionTests(unittest.TestCase):
    def test_ingests_only_high_like_top_level_youtube_comments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "comments.info.json"
            path.write_text(
                json.dumps(
                    {
                        "comments": [
                            {
                                "id": "high",
                                "text": "等身大フィギュアの話だね",
                                "author": "fan-a",
                                "like_count": 42,
                            },
                            {
                                "id": "reply",
                                "parent": "high",
                                "text": "返信",
                                "author": "fan-b",
                                "like_count": 100,
                            },
                            {
                                "id": "low",
                                "text": "低評価ではなく低いいね",
                                "author": "fan-c",
                                "like_count": 2,
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            retriever = FanKnowledgeRetriever(root / "knowledge.sqlite3")
            result = ingest_youtube_top_comments(
                retriever,
                path,
                video_id="abcdefghijk",
                title="配信",
                source_url="https://www.youtube.com/watch?v=abcdefghijk",
                published_at="2026-08-31",
                minimum_likes=10,
                maximum_comments=30,
            )
            hits = retriever.retrieve(
                KnowledgeQuery("等身大フィギュア", exclude_video_id="abcdefghijk")
            )
            retriever.close()

        self.assertIsNotNone(result)
        self.assertEqual(result.chunk_count, 1)
        self.assertEqual(len(hits), 1)
        self.assertIn("YouTube高赞评论", hits[0].title)
        self.assertIn("fan-a", hits[0].body)
        self.assertNotIn("返信", hits[0].body)
    def test_timed_chunks_preserve_speaker_and_split_tracks(self) -> None:
        chunks = chunk_timed_cues(
            [
                Cue(0, 10, "最初の話", "A", language="Japanese"),
                Cue(10, 20, "続き", "A", language="Japanese"),
                Cue(20, 25, "別の人", "B", language="Japanese"),
            ],
            target_seconds=60,
        )
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].speaker, "A")
        self.assertEqual(chunks[0].start_seconds, 0)
        self.assertEqual(chunks[1].speaker, "B")

    def test_jsonl_import_is_incremental(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "sns.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "source_type": "x_post",
                        "external_id": "post-1",
                        "source_url": "https://x.example/post-1",
                        "title": "演唱会告知",
                        "text": "新宿着陆计划将在八月举行。",
                        "author": "official",
                        "published_at": "2026-07-01T00:00:00+00:00",
                        "reliability": 0.95,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            retriever = FanKnowledgeRetriever(root / "knowledge.sqlite3")
            first = ingest_jsonl(retriever, path)
            second = ingest_jsonl(retriever, path)
            hits = retriever.retrieve(KnowledgeQuery("新宿着陆计划 演唱会"))
            retriever.close()

        self.assertEqual(first.inserted_or_updated, 1)
        self.assertEqual(second.unchanged, 1)
        self.assertEqual(hits[0].kind, "document_chunk")

    def test_work_import_ignores_pipeline_asr(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = root / "work" / "job"
            job.mkdir(parents=True)
            (job / "source.info.json").write_text(
                json.dumps(
                    {
                        "id": "video-1",
                        "title": "配信タイトル",
                        "description": "公式イベントのお知らせ",
                        "channel": "仲町あられ",
                        "upload_date": "20260811",
                        "webpage_url": "https://youtube.example/video-1",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (job / "source.qwen3-asr.cues.json").write_text(
                json.dumps(
                    {
                        "version": 7,
                        "cues": [
                            {
                                "start": 10.0,
                                "end": 15.0,
                                "text": "アクスタが安く感じる",
                                "speaker": "nakamachi_arale",
                                "language": "Japanese",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            retriever = FanKnowledgeRetriever(root / "knowledge.sqlite3")
            summary = ingest_work_directory(retriever, root / "work")
            metadata_hits = retriever.retrieve(KnowledgeQuery("公式イベント"))
            asr_hits = retriever.retrieve(KnowledgeQuery("アクスタ"))
            retriever.close()

        self.assertEqual(summary.inserted_or_updated, 1)
        self.assertEqual(metadata_hits[0].kind, "document_chunk")
        self.assertEqual(asr_hits, [])

    def test_youtube_cache_prefers_manual_subtitle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "cache" / "video-1"
            directory.mkdir(parents=True)
            (directory / "source.info.json").write_text(
                json.dumps(
                    {
                        "id": "video-1",
                        "title": "歌枠",
                        "description": "概要",
                        "subtitles": {"ja": [{}]},
                        "automatic_captions": {"en": [{}]},
                        "upload_date": "20260811",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (directory / "source.ja.srt").write_text(
                "1\n00:00:01,000 --> 00:00:03,000\n公式字幕です\n",
                encoding="utf-8",
            )
            (directory / "source.en.srt").write_text(
                "1\n00:00:01,000 --> 00:00:03,000\nauto caption\n",
                encoding="utf-8",
            )
            retriever = FanKnowledgeRetriever(root / "knowledge.sqlite3")
            summary = ingest_youtube_cache(retriever, root / "cache")
            hits = retriever.retrieve(KnowledgeQuery("公式字幕"))
            retriever.close()

        self.assertEqual(summary.inserted_or_updated, 2)
        self.assertTrue(hits)

    def test_youtube_cache_keeps_metadata_without_description(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "cache" / "video-1"
            directory.mkdir(parents=True)
            (directory / "source.info.json").write_text(
                json.dumps(
                    {
                        "id": "video-1",
                        "title": "短い公式動画",
                        "description": "",
                        "channel": "夢限大みゅーたいぷ",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            retriever = FanKnowledgeRetriever(root / "knowledge.sqlite3")
            summary = ingest_youtube_cache(retriever, root / "cache")
            hits = retriever.retrieve(KnowledgeQuery("短い公式動画"))
            retriever.close()

        self.assertEqual(summary.inserted_or_updated, 1)
        self.assertEqual(hits[0].kind, "document_chunk")

    def test_youtube_cache_ignores_playlist_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "cache" / "channel-id"
            directory.mkdir(parents=True)
            (directory / "source.info.json").write_text(
                json.dumps(
                    {
                        "_type": "playlist",
                        "id": "channel-id",
                        "title": "频道直播列表",
                        "description": "不是视频",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            retriever = FanKnowledgeRetriever(root / "knowledge.sqlite3")
            summary = ingest_youtube_cache(retriever, root / "cache")
            retriever.close()

        self.assertEqual(summary.skipped, 1)
        self.assertEqual(summary.inserted_or_updated, 0)

    def test_live_chat_import_separates_super_chat_and_filters_emoji_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chat_path = root / "source.live_chat.json"
            values = [
                _chat_action("liveChatTextMessageRenderer", "罠かな", 1000),
                _chat_action("liveChatTextMessageRenderer", ":_TAIKI::_TAIKI:", 2000),
                _chat_action(
                    "liveChatPaidMessageRenderer",
                    "本日分の貢ぎ物です",
                    3000,
                    amount="¥2,000",
                ),
            ]
            chat_path.write_text(
                "\n".join(json.dumps(value, ensure_ascii=False) for value in values),
                encoding="utf-8",
            )

            messages = read_youtube_live_chat(chat_path)
            chat_chunks, paid_chunks = chunk_youtube_chat(messages)

        self.assertEqual(len(messages), 2)
        self.assertEqual(chat_chunks[0].text, "viewer: 罠かな")
        self.assertIn("SC ¥2,000", paid_chunks[0].text)
        self.assertIn("本日分の貢ぎ物", paid_chunks[0].text)

    def test_youtube_chat_cleanup_removes_partial_fragments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in (
                "source.live_chat.json",
                "source.live_chat.json.part",
                "source.live_chat.json.part-Frag1",
            ):
                (directory / name).write_text("chat", encoding="utf-8")
            (directory / "source.info.json").write_text("{}", encoding="utf-8")

            remove_youtube_chat_files(directory)

            self.assertFalse(any(directory.glob("source.live_chat.json*")))

    def test_youtube_cache_indexes_super_chat_but_not_regular_chat_by_default(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "cache" / "video-1"
            directory.mkdir(parents=True)
            (directory / "source.info.json").write_text(
                json.dumps({"id": "video-1", "title": "直播"}, ensure_ascii=False),
                encoding="utf-8",
            )
            (directory / "source.live_chat.json").write_text(
                "\n".join(
                    json.dumps(value, ensure_ascii=False)
                    for value in (
                        _chat_action("liveChatTextMessageRenderer", "普通聊天", 1000),
                        _chat_action(
                            "liveChatPaidMessageRenderer",
                            "本日分の貢ぎ物",
                            2000,
                            amount="¥2,000",
                        ),
                    )
                ),
                encoding="utf-8",
            )
            retriever = FanKnowledgeRetriever(root / "knowledge.sqlite3")
            summary = ingest_youtube_cache(retriever, root / "cache")
            paid_hits = retriever.retrieve(KnowledgeQuery("本日分の貢ぎ物"))
            chat_hits = retriever.retrieve(KnowledgeQuery("普通聊天"))
            retriever.close()

        self.assertEqual(summary.inserted_or_updated, 2)
        self.assertEqual(paid_hits[0].kind, "document_chunk")
        self.assertEqual(chat_hits, [])

    @patch("subtitle_pipeline.knowledge_ingestion.subprocess.run")
    @patch(
        "subtitle_pipeline.knowledge_ingestion.require_command",
        return_value="/usr/bin/yt-dlp",
    )
    def test_youtube_download_uses_subtitle_only_mode(
        self, _require: object, run: object
    ) -> None:
        run.side_effect = [
            subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps(
                    {
                        "id": "abcdefghijk",
                        "live_status": "was_live",
                        "availability": "public",
                    }
                ),
                stderr="",
            ),
            subprocess.CompletedProcess([], 0),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            download_youtube_subtitles(
                ["https://youtube.example/channel"],
                Path(temporary),
                DownloadConfig(cookies_from_browser="chromium"),
                playlist_end=20,
            )
        listing = run.call_args_list[0].args[0]
        command = run.call_args_list[1].args[0]
        self.assertIn("--flat-playlist", listing)
        self.assertIn("20", listing)
        self.assertIn("--skip-download", command)
        self.assertIn("--no-progress", command)
        self.assertIn("--ignore-no-formats-error", command)
        self.assertIn("--download-archive", command)
        self.assertIn("--force-write-archive", command)
        self.assertIn("--write-auto-subs", command)
        self.assertIn("ja.*,ja,en.*,en,live_chat", command)
        self.assertIn("--cookies-from-browser", command)
        self.assertIn("--js-runtimes", command)
        self.assertIn("ejs:github", command)
        self.assertIn("--extractor-args", command)
        self.assertIn("youtube:player_client=web_creator", command)
        self.assertIn("--concurrent-fragments", command)
        self.assertIn("8", command)
        self.assertIn("https://www.youtube.com/watch?v=abcdefghijk", command)


def _chat_action(
    renderer_name: str,
    text: str,
    offset_milliseconds: int,
    *,
    amount: str | None = None,
) -> dict[str, object]:
    renderer: dict[str, object] = {
        "message": {"runs": [{"text": text}]},
        "authorName": {"simpleText": "viewer"},
    }
    if amount:
        renderer["purchaseAmountText"] = {"simpleText": amount}
    return {
        "replayChatItemAction": {
            "videoOffsetTimeMsec": str(offset_milliseconds),
            "actions": [
                {
                    "addChatItemAction": {
                        "item": {renderer_name: renderer},
                    }
                }
            ],
        }
    }


if __name__ == "__main__":
    unittest.main()
