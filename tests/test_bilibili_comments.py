import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from subtitle_pipeline.bilibili_comments import (
    build_song_setlist_comment,
    create_comment_task,
    process_bilibili_comment_task,
)
from subtitle_pipeline.config import UploadConfig


class BilibiliCommentTests(unittest.TestCase):
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
            build_song_setlist_comment("【歌枠】歌います", reports),
            "2:15 壱雫空\n11:18 八月の夜",
        )
        self.assertIsNone(build_song_setlist_comment("【歌回】歌います", reports))

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
            build_song_setlist_comment("歌枠", reports),
            "1:10:34 INSIDE IDENTITY",
        )

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
                upload_response="success",
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
            with patch(
                "subtitle_pipeline.bilibili_comments._request_json",
                return_value=response,
            ) as request:
                status = process_bilibili_comment_task(task, config)
            self.assertEqual(status, "already_exists")
            self.assertEqual(json.loads(task.read_text())["rpid"], 456)
            self.assertEqual(request.call_count, 1)

    def test_new_task_waits_one_hour_before_first_attempt(self):
        with tempfile.TemporaryDirectory() as temp, patch(
            "subtitle_pipeline.bilibili_comments.time.time", return_value=1000.0
        ):
            task = create_comment_task(
                Path(temp),
                aid=123,
                bvid="BV123",
                message="2:15 壱雫空",
                source_url="https://youtube.test/watch?v=1",
                upload_response="success",
            )
            payload = json.loads(task.read_text())
        self.assertEqual(payload["created_at"], 1000.0)
        self.assertEqual(payload["next_attempt_at"], 4600.0)

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
                upload_response="success",
            )
            responses = [
                {"code": 0, "data": {"replies": [], "page": {"count": 0}}},
                {"code": 0, "message": "0", "data": {"rpid": 789}},
            ]
            with patch(
                "subtitle_pipeline.bilibili_comments._request_json",
                side_effect=responses,
            ):
                status = process_bilibili_comment_task(task, config)
            payload = json.loads(task.read_text())
            self.assertEqual(status, "posted")
            self.assertEqual(payload["rpid"], 789)
            self.assertEqual(payload["last_response"]["data"]["rpid"], 789)

    def test_keeps_unavailable_draft_pending(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root)
            task = create_comment_task(
                root,
                aid=123,
                bvid="BV123",
                message="2:15 壱雫空",
                source_url="https://youtube.test/watch?v=1",
                upload_response="success",
            )
            with patch(
                "subtitle_pipeline.bilibili_comments._request_json",
                return_value={"code": -404, "message": "啥都木有"},
            ):
                status = process_bilibili_comment_task(task, config)
            self.assertEqual(status, "pending_review")

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
