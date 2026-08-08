import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from subtitle_pipeline.commands import CommandError
from subtitle_pipeline.config import DownloadConfig
from subtitle_pipeline.media import download_youtube


class MediaDownloadTests(unittest.TestCase):
    def test_subtitle_failure_falls_back_without_discarding_video(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)

            def fake_run(command):
                if "--skip-download" in command:
                    raise CommandError("HTTP 429")
                (directory / "source.info.json").write_text(
                    json.dumps({"title": "Video", "language": "ja"}),
                    encoding="utf-8",
                )
                (directory / "source.mp4").write_bytes(b"video")

            def find_runtime(name):
                return "/usr/bin/node" if name == "node" else None

            with patch(
                "subtitle_pipeline.media.require_command", return_value="/venv/bin/yt-dlp"
            ), patch("subtitle_pipeline.media.run", side_effect=fake_run) as run, patch(
                "subtitle_pipeline.media.shutil.which", side_effect=find_runtime
            ):
                result = download_youtube(
                    "https://www.youtube.com/watch?v=test",
                    directory,
                    DownloadConfig(),
                )

            self.assertEqual(result.video, directory / "source.mp4")
            self.assertIsNone(result.subtitle)
            self.assertEqual(run.call_count, 2)
            video_command, subtitle_command = [call.args[0] for call in run.call_args_list]
            self.assertNotIn("--write-subs", video_command)
            self.assertIn("--skip-download", subtitle_command)
            self.assertIn("node:/usr/bin/node", video_command)
            languages = subtitle_command[subtitle_command.index("--sub-langs") + 1]
            self.assertTrue(languages.startswith("ja-orig,ja,"))

    def test_existing_nonempty_subtitle_skips_subtitle_request(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)

            def fake_run(_command):
                (directory / "source.info.json").write_text(
                    '{"language":"en"}', encoding="utf-8"
                )
                (directory / "source.mp4").write_bytes(b"video")
                (directory / "source.en.srt").write_text("subtitle", encoding="utf-8")

            with patch(
                "subtitle_pipeline.media.require_command", return_value="yt-dlp"
            ), patch("subtitle_pipeline.media.run", side_effect=fake_run) as run:
                result = download_youtube(
                    "https://youtu.be/test",
                    directory,
                    DownloadConfig(js_runtime=None),
                )

            self.assertEqual(result.subtitle, directory / "source.en.srt")
            run.assert_called_once()

    def test_prefers_original_language_over_english_translation(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for language in ("en", "ja", "ja-orig"):
                (directory / f"source.{language}.srt").write_text(
                    language, encoding="utf-8"
                )
            metadata = {
                "language": "ja",
                "subtitles": {},
                "automatic_captions": {"en": [], "ja": [], "ja-orig": []},
            }
            from subtitle_pipeline.media import _find_subtitle

            self.assertEqual(
                _find_subtitle(directory, metadata), directory / "source.ja-orig.srt"
            )


if __name__ == "__main__":
    unittest.main()
