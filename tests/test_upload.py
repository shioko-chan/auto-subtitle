import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from subtitle_pipeline.config import UploadConfig
from subtitle_pipeline.upload import (
    _prepare_description,
    _truncate_utf16,
    _utf16_units,
    upload_to_bilibili,
)


class UploadTests(unittest.TestCase):
    def test_truncates_description_by_utf16_units_without_splitting_surrogate_pair(self):
        value = "a" * 1999 + "🎶" + "tail"
        result = _truncate_utf16(value, 2000)
        self.assertEqual(result, "a" * 1999)
        self.assertEqual(len(result.encode("utf-16-le")) // 2, 1999)

    def test_prepares_description_at_paragraph_boundary_and_preserves_suffix(self):
        result = _prepare_description(
            "first paragraph\n\n" + "x" * 80 + "\n\nlast paragraph",
            suffix="generated subtitle",
            max_chars=70,
        )
        self.assertEqual(result, "first paragraph\n\ngenerated subtitle")
        self.assertLessEqual(len(result), 70)
        self.assertLessEqual(_utf16_units(result), 70)

    def test_prepares_description_with_unicode_under_both_limits(self):
        result = _prepare_description(
            "字幕🎶" * 100,
            suffix="固定说明",
            max_chars=80,
        )
        self.assertTrue(result.endswith("固定说明"))
        self.assertLessEqual(len(result), 80)
        self.assertLessEqual(_utf16_units(result), 80)

    def test_builds_repost_command_without_shell_interpolation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cookie = root / "cookies.json"
            cookie.write_text("{}", encoding="utf-8")
            video = root / "video.mp4"
            config = UploadConfig(cookie_file=str(cookie), tags=["中字", "科技"])
            with patch("subtitle_pipeline.upload.require_command", return_value="/bin/biliup"), patch(
                "subtitle_pipeline.upload.run"
            ) as run:
                upload_to_bilibili(
                    video,
                    title="A title",
                    description="Description",
                    source_url="https://youtube.test/watch?v=1",
                    tags=["中字", "自动生成"],
                    config=config,
                )
            command = run.call_args.args[0]
            self.assertEqual(command[:4], ["/bin/biliup", "--user-cookie", str(cookie), "upload"])
            self.assertIn("https://youtube.test/watch?v=1", command)
            self.assertIn("中字,自动生成", command)
            description = command[command.index("--desc") + 1]
            self.assertEqual(description, "Description")
            self.assertEqual(command[-1], str(video))


if __name__ == "__main__":
    unittest.main()
