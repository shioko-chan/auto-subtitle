import importlib.util
import sys
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "update-recent-yumemita-tasks.py"
SPEC = importlib.util.spec_from_file_location("update_recent_yumemita_tasks", SCRIPT)
assert SPEC and SPEC.loader
tasks = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tasks
SPEC.loader.exec_module(tasks)


class RecentTaskTests(unittest.TestCase):
    def test_feed_and_playlist_select_only_recent_public_archives(self):
        feed = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:yt="http://www.youtube.com/xml/schemas/2015">
  <entry><yt:videoId>public</yt:videoId><published>2026-08-20T12:00:00+00:00</published></entry>
  <entry><yt:videoId>members</yt:videoId><published>2026-08-20T13:00:00+00:00</published></entry>
  <entry><yt:videoId>upcoming</yt:videoId><published>2026-08-21T13:00:00+00:00</published></entry>
  <entry><yt:videoId>old</yt:videoId><published>2026-07-01T13:00:00+00:00</published></entry>
</feed>"""
        playlist = {
            "entries": [
                {"id": "public", "title": "配信", "live_status": "was_live"},
                {
                    "id": "members",
                    "title": "メン限配信",
                    "live_status": "was_live",
                    "availability": "subscriber_only",
                },
                {"id": "upcoming", "title": "予定", "live_status": "is_upcoming"},
                {"id": "old", "title": "古い配信", "live_status": "was_live"},
            ]
        }
        selected = tasks.select_recent_streams(
            playlist,
            tasks.parse_feed(feed),
            channel="miyako",
            cutoff=datetime(2026, 8, 10, tzinfo=UTC),
        )
        self.assertEqual([record.video_id for record in selected], ["public"])

    def test_rewrite_keeps_uploaded_history_and_adds_recent(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            batch = directory / "batch.sh"
            uploaded = directory / "uploaded.txt"
            batch.write_text(
                '# comment\n# Public archived streams old.\n# Members-only streams are deliberately excluded.\n'
                'RECORDS=(\n    "2026-07-01|ritsu|done"\n'
                '    "2026-07-02|ritsu|failed"\n)\necho run\n',
                encoding="utf-8",
            )
            uploaded.write_text("done\n", encoding="utf-8")
            history = tasks.read_uploaded_records(batch, uploaded)
            recent = tasks.Record(
                datetime(2026, 8, 20, tzinfo=UTC), "miyako", "new", "配信"
            )
            tasks.replace_records(batch, tasks.deduplicate([*history, recent]))
            result = batch.read_text(encoding="utf-8")
            self.assertIn('"2026-07-01|ritsu|done"', result)
            self.assertIn('"2026-08-20|miyako|new"', result)
            self.assertNotIn("failed", result)
            self.assertIn("echo run", result)


if __name__ == "__main__":
    unittest.main()
