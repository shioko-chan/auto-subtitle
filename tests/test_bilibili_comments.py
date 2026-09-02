import json
import tempfile
import unittest
import urllib.parse
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from subtitle_pipeline.bilibili_comments import (
    _request_json,
    _submit_comment_on_page,
    build_song_setlist_comment,
    create_comment_task,
    publish_comment_task,
)
from subtitle_pipeline.config import UploadConfig


class BilibiliCommentTests(unittest.TestCase):
    def test_comment_requests_use_browser_headers(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"code": 0}'
        with patch(
            "subtitle_pipeline.bilibili_comments.urllib.request.urlopen",
            return_value=response,
        ) as urlopen:
            _request_json("https://api.bilibili.com/test", cookies={"a": "b"})

        request = urlopen.call_args.args[0]
        self.assertIn("Chrome/151.0.0.0", request.get_header("User-agent"))
        self.assertEqual(
            request.get_header("Sec-ch-ua"),
            '"Chromium";v="151", "Not=A?Brand";v="99"',
        )
        self.assertEqual(request.get_header("Sec-fetch-dest"), "empty")
        self.assertEqual(request.get_header("Sec-fetch-mode"), "cors")
        self.assertEqual(request.get_header("Sec-fetch-site"), "same-site")

    def test_builds_setlist_only_for_literal_song_stream_title(self):
        reports = [
            {
                "song_id": "song-1",
                "song": "壱雫空",
                "alignments": [{"start": 135.2}],
                "search_group": {"start": 130.0, "end": 420.0},
            },
            {
                "song_id": "song-2",
                "song": "八月の夜",
                "alignments": [{"start": 678.1}],
                "search_group": {"start": 670.0, "end": 900.0},
            },
        ]
        self.assertEqual(
            build_song_setlist_comment({"title": "【歌枠】歌います"}, reports),
            "歌单：\n2:15 壱雫空\n11:18 八月の夜",
        )
        self.assertIsNone(
            build_song_setlist_comment({"title": "【歌回】歌います"}, reports)
        )

    def test_merges_adjacent_reports_for_the_same_performance(self):
        reports = [
            {
                "song_id": "same",
                "song": "INSIDE IDENTITY",
                "alignments": [{"start": 4234.0}],
                "search_group": {"start": 4230.0, "end": 4807.5},
            },
            {
                "song_id": "same",
                "song": "INSIDE IDENTITY",
                "alignments": [{"start": 4873.0}],
                "search_group": {"start": 4847.5, "end": 4957.5},
            },
        ]
        self.assertEqual(
            build_song_setlist_comment({"title": "歌枠"}, reports),
            "歌单：\n1:10:34 INSIDE IDENTITY",
        )

    def test_reposts_highest_liked_complete_setlist_verbatim(self):
        comments = [
            {
                "parent": "root",
                "like_count": 10,
                "text": "SETLIST\n0:10 Song A\n1:20 Song B\n2:30 Song C",
            },
            {
                "parent": "root",
                "like_count": 15,
                "text": "📢== SETLIST ==🌟\n0:19:16 ふでペン\n0:34:54 ドラえもん\n0:48:41 高音厨音域テスト",
            },
        ]
        self.assertEqual(
            build_song_setlist_comment(
                {"title": "【歌枠】歌います"}, [], comments
            ),
            comments[1]["text"],
        )

    def test_adds_the_solo_members_bilibili_emoji(self):
        reports = [{"song": "壱雫空", "alignments": [{"start": 10.0}]}]
        cases = {
            "仲町あられ": "[梦限大_阿拉蕾耶]",
            "峰月律": "[梦限大_律敬礼]",
            "宮永ののか": "[梦限大_野乃花来啦]",
            "藤都子": "[梦限大_都子期待]",
            "千石ユノ": "[梦限大_由乃坏笑]",
        }
        for channel, emoji in cases.items():
            with self.subTest(channel=channel):
                comment = build_song_setlist_comment(
                    {"title": "【歌枠】歌います", "channel": channel}, reports
                )
                self.assertEqual(comment, f"{emoji}\n歌单：\n0:10 壱雫空")

    def test_adds_character_emoji_to_forwarded_setlist(self):
        comments = [
            {
                "parent": "root",
                "like_count": 20,
                "text": "SETLIST\n0:10 Song A\n1:20 Song B\n2:30 Song C",
            }
        ]

        self.assertEqual(
            build_song_setlist_comment(
                {"title": "【藤都子】歌枠"}, [], comments
            ),
            "[梦限大_都子期待]\nSETLIST\n0:10 Song A\n1:20 Song B\n2:30 Song C",
        )

    def test_ignores_non_setlist_zero_like_reply_and_oversized_comments(self):
        reports = [
            {
                "song": "Fallback Song",
                "alignments": [{"start": 10.0}],
            }
        ]
        comments = [
            {"parent": "root", "like_count": 20, "text": "セトリ最高でした"},
            {
                "parent": "root",
                "like_count": 9,
                "text": "SETLIST\n0:10 A\n1:20 B\n2:30 C",
            },
            {
                "parent": "another-comment",
                "like_count": 30,
                "text": "SETLIST\n0:10 A\n1:20 B\n2:30 C",
            },
            {
                "parent": "root",
                "like_count": 40,
                "text": "SETLIST\n0:10 A\n1:20 B\n2:30 C\n" + "x" * 1000,
            },
        ]
        self.assertEqual(
            build_song_setlist_comment({"title": "歌枠"}, reports, comments),
            "歌单：\n0:10 Fallback Song",
        )

    def test_creates_task_scheduled_five_minutes_after_upload(self):
        with tempfile.TemporaryDirectory() as temp, patch(
            "subtitle_pipeline.bilibili_comments.time.time", return_value=1000.0
        ):
            root = Path(temp)
            job = root / "job"
            task = create_comment_task(
                job,
                aid=123,
                bvid="BV123",
                message="2:15 壱雫空",
                source_url="https://youtube.test/watch?v=1",
            )
            payload = json.loads(task.read_text())
            self.assertEqual(payload["status"], "scheduled")
            self.assertEqual(payload["created_at"], 1000.0)
            self.assertEqual(payload["publish_at"], 1300.0)

    def test_records_existing_comment_rpid_without_posting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root)
            task = create_comment_task(
                root,
                aid=123,
                bvid="BV123",
                message="2:15 壱雫空",
                source_url="https://youtube.test/watch?v=1",
            )
            response = {
                "code": 0,
                "data": {
                    "replies": [
                        {
                            "rpid": 456,
                            "member": {"mid": "42"},
                            "content": {"message": "2:15 壱雫空"},
                        }
                    ],
                    "page": {"count": 1},
                },
            }
            with (
                patch(
                    "subtitle_pipeline.bilibili_comments._request_json",
                    return_value=response,
                ) as request,
                patch("subtitle_pipeline.bilibili_comments.time.sleep") as sleep,
            ):
                status = publish_comment_task(task, config)
            self.assertEqual(status, "already_exists")
            self.assertAlmostEqual(sleep.call_args.args[0], 300.0, delta=0.1)
            self.assertEqual(json.loads(task.read_text())["rpid"], 456)
            self.assertEqual(request.call_count, 1)
            query = urllib.parse.parse_qs(
                urllib.parse.urlparse(request.call_args.args[0]).query
            )
            self.assertEqual(query["ps"], ["20"])

    def test_posts_and_records_interface_response(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root)
            task = create_comment_task(
                root,
                aid=123,
                bvid="BV123",
                message="2:15 壱雫空",
                source_url="https://youtube.test/watch?v=1",
            )
            responses = [
                {"code": 0, "data": {"replies": [], "page": {"count": 0}}},
                {
                    "code": 0,
                    "data": {
                        "root": {
                            "rpid": 789,
                            "state": 0,
                            "member": {"mid": "42"},
                            "content": {"message": "2:15 壱雫空"},
                        }
                    },
                },
            ]
            with (
                patch(
                    "subtitle_pipeline.bilibili_comments._request_json",
                    side_effect=responses,
                ),
                patch(
                    "subtitle_pipeline.bilibili_comments._post_comment_with_browser",
                    return_value={
                        "code": 0,
                        "message": "0",
                        "data": {"rpid": 789},
                    },
                ) as post,
                patch("subtitle_pipeline.bilibili_comments.time.sleep"),
            ):
                status = publish_comment_task(task, config)
            payload = json.loads(task.read_text())
            self.assertEqual(status, "posted")
            self.assertEqual(payload["rpid"], 789)
            self.assertEqual(payload["last_response"]["data"]["rpid"], 789)
            self.assertEqual(post.call_args.args, ("BV123", "2:15 壱雫空"))

    def test_browser_page_fills_editor_and_uses_the_page_reply_response(self):
        page = MagicMock()
        comment_area = MagicMock()
        editor = MagicMock()
        send = MagicMock()
        response_context = MagicMock()
        response_context.__enter__.return_value = response_context
        response_context.value.json.return_value = {
            "code": 0,
            "data": {"rpid": 789},
        }
        page.expect_response.return_value = response_context

        with patch(
            "subtitle_pipeline.bilibili_comments._first_visible_locator",
            side_effect=[comment_area, editor, send],
        ):
            result = _submit_comment_on_page(page, "BV123", "歌单内容")

        page.goto.assert_called_once_with(
            "https://www.bilibili.com/video/BV123",
            wait_until="domcontentloaded",
            timeout=60_000,
        )
        comment_area.scroll_into_view_if_needed.assert_called_once()
        editor.fill.assert_called_once_with("")
        editor.press_sequentially.assert_called_once_with("歌单内容")
        send.click.assert_called_once_with(timeout=10_000)
        self.assertEqual(result["data"]["rpid"], 789)

    def test_browser_page_types_multiline_comments_through_the_editor(self):
        page = MagicMock()
        editor = MagicMock()
        send = MagicMock()
        response_context = MagicMock()
        response_context.__enter__.return_value = response_context
        response_context.value.json.return_value = {
            "code": 0,
            "data": {"rpid": 789},
        }
        page.expect_response.return_value = response_context

        with patch(
            "subtitle_pipeline.bilibili_comments._first_visible_locator",
            side_effect=[None, editor, send],
        ):
            _submit_comment_on_page(page, "BV123", "表情\n歌单：\n0:10 Song")

        self.assertEqual(
            editor.mock_calls,
            [
                call.fill(""),
                call.press_sequentially("表情"),
                call.press("Shift+Enter"),
                call.press_sequentially("歌单："),
                call.press("Shift+Enter"),
                call.press_sequentially("0:10 Song"),
            ],
        )

    def test_records_successful_but_invisible_post_as_hidden(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root)
            task = create_comment_task(
                root,
                aid=123,
                bvid="BV123",
                message="2:15 壱雫空",
                source_url="https://youtube.test/watch?v=1",
            )
            responses = [
                {"code": 0, "data": {"replies": [], "page": {"count": 0}}},
                {
                    "code": 0,
                    "data": {
                        "root": {
                            "rpid": 789,
                            "state": 17,
                            "member": {"mid": "42"},
                            "content": {"message": "2:15 壱雫空"},
                        }
                    },
                },
            ]
            with (
                patch(
                    "subtitle_pipeline.bilibili_comments._request_json",
                    side_effect=responses,
                ),
                patch(
                    "subtitle_pipeline.bilibili_comments._post_comment_with_browser",
                    return_value={
                        "code": 0,
                        "message": "OK",
                        "data": {"rpid": 789},
                    },
                ),
                patch("subtitle_pipeline.bilibili_comments.time.sleep"),
            ):
                status = publish_comment_task(task, config)

            payload = json.loads(task.read_text())
            self.assertEqual(status, "hidden")
            self.assertEqual(payload["status"], "hidden")
            self.assertEqual(payload["rpid"], 789)
            self.assertEqual(
                payload["last_response"]["post_response"]["data"]["rpid"], 789
            )

    def test_published_or_hidden_task_cannot_be_sent_again(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root)
            task = create_comment_task(
                root,
                aid=123,
                bvid="BV123",
                message="2:15 壱雫空",
                source_url="https://youtube.test/watch?v=1",
            )
            payload = json.loads(task.read_text())
            payload["status"] = "hidden"
            payload["rpid"] = 789
            task.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not scheduled"):
                publish_comment_task(task, config)

    def test_records_query_failure_without_posting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root)
            task = create_comment_task(
                root,
                aid=123,
                bvid="BV123",
                message="2:15 壱雫空",
                source_url="https://youtube.test/watch?v=1",
            )
            with (
                patch(
                    "subtitle_pipeline.bilibili_comments._request_json",
                    return_value={"code": -404, "message": "啥都木有"},
                ) as request,
                patch("subtitle_pipeline.bilibili_comments.time.sleep"),
            ):
                status = publish_comment_task(task, config)
            self.assertEqual(status, "failed")
            self.assertEqual(request.call_count, 1)

    @staticmethod
    def _config(root: Path) -> UploadConfig:
        cookie = root / "cookies.json"
        cookie.write_text(
            json.dumps(
                {
                    "cookie_info": {
                        "cookies": [
                            {"name": "SESSDATA", "value": "session"},
                            {"name": "bili_jct", "value": "csrf"},
                            {"name": "DedeUserID", "value": "42"},
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        return UploadConfig(cookie_file=str(cookie))


if __name__ == "__main__":
    unittest.main()
