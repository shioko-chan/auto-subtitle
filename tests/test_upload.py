import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from subtitle_pipeline.config import UploadConfig
from subtitle_pipeline.upload import upload_to_bilibili


class UploadTests(unittest.TestCase):
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
            self.assertEqual(command[-1], str(video))


if __name__ == "__main__":
    unittest.main()
