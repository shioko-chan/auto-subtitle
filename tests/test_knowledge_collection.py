from __future__ import annotations

import json
import subprocess
import unittest
from unittest.mock import patch

from subtitle_pipeline.knowledge_collection import (
    collect_official_documents,
    collect_sns_documents,
    normalize_sns_metadata,
)


class KnowledgeCollectionTests(unittest.TestCase):
    @patch("subtitle_pipeline.knowledge_collection._download")
    def test_official_html_extracts_article_metadata(self, download) -> None:
        download.return_value = (
            """
            <html><head><title>Fallback</title>
            <meta property="og:title" content="演唱会公告">
            <meta property="article:published_time" content="2026-08-01">
            </head><body><nav>菜单噪声</nav><article>
            <h1>新宿着陆计划</h1><p>八月举行特别演唱会。</p>
            </article><footer>版权噪声</footer></body></html>
            """,
            "text/html",
        )

        documents = collect_official_documents(["https://example.jp/news/1"])

        self.assertEqual(documents[0]["title"], "演唱会公告")
        self.assertIn("特别演唱会", documents[0]["text"])
        self.assertNotIn("菜单噪声", documents[0]["text"])
        self.assertEqual(documents[0]["published_at"], "2026-08-01")

    @patch("subtitle_pipeline.knowledge_collection._download")
    def test_official_feed_creates_one_document_per_entry(self, download) -> None:
        download.return_value = (
            """<?xml version="1.0"?><rss><channel><item>
            <guid>event-1</guid><title>活动通知</title>
            <link>https://example.jp/event-1</link>
            <description><![CDATA[<p>活动正文</p>]]></description>
            <pubDate>2026-08-02</pubDate>
            </item></channel></rss>""",
            "application/rss+xml",
        )

        documents = collect_official_documents(["https://example.jp/feed.xml"])

        self.assertEqual(documents[0]["external_id"], "event-1")
        self.assertEqual(documents[0]["text"], "活动正文")

    @patch("subtitle_pipeline.knowledge_collection._download")
    def test_official_archive_follows_matching_same_host_links(self, download) -> None:
        pages = {
            "https://example.jp/news/": (
                '<html><body><a href="/news/1">one</a>'
                '<a href="https://other.jp/news/2">other</a></body></html>',
                "text/html",
            ),
            "https://example.jp/news/1": (
                "<html><body><h1>公告</h1><p>正文</p></body></html>",
                "text/html",
            ),
        }
        download.side_effect = lambda url, _timeout: pages[url]

        documents = collect_official_documents(
            ["https://example.jp/news/"],
            follow_links=True,
            link_pattern=r"/news/\d+$",
        )

        self.assertEqual(len(documents), 1)
        self.assertIn("正文", documents[0]["text"])
        self.assertEqual(download.call_count, 2)

    def test_normalizes_x_and_instagram_metadata(self) -> None:
        x_post = normalize_sns_metadata(
            {
                "category": "twitter",
                "tweet_id": "123",
                "content": "ライブのお知らせ",
                "author": {"name": "公式", "username": "official"},
                "date": "2026-08-01T00:00:00+09:00",
            }
        )
        instagram = normalize_sns_metadata(
            {
                "category": "instagram",
                "shortcode": "ABC",
                "description": "イベント写真",
                "username": "official",
            }
        )

        self.assertEqual(x_post["source_url"], "https://x.com/official/status/123")
        self.assertEqual(instagram["source_url"], "https://www.instagram.com/p/ABC/")

    @patch("subtitle_pipeline.knowledge_collection.subprocess.run")
    @patch(
        "subtitle_pipeline.knowledge_collection.require_command",
        return_value="/usr/bin/gallery-dl",
    )
    def test_sns_collection_uses_metadata_only_jsonl(self, _require, run) -> None:
        run.return_value = subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps(
                {
                    "category": "twitter",
                    "tweet_id": "123",
                    "content": "告知",
                    "author": {"username": "official"},
                },
                ensure_ascii=False,
            ),
            stderr="",
        )

        documents = collect_sns_documents(
            ["https://x.com/official"], cookies_from_browser="chromium"
        )

        command = run.call_args.args[0]
        self.assertIn("--no-download", command)
        self.assertIn("mode=jsonl", command)
        self.assertIn("--cookies-from-browser", command)
        self.assertEqual(documents[0]["external_id"], "123")


if __name__ == "__main__":
    unittest.main()
