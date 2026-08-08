import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from subtitle_pipeline.config import AppConfig, UploadConfig, WhisperConfig
from subtitle_pipeline.media import DownloadResult
from subtitle_pipeline.pipeline import normalize_youtube_url, run_pipeline
from subtitle_pipeline.subtitles import Cue, write_srt


class PipelineTests(unittest.TestCase):
    def test_normalizes_shell_escaped_youtube_query(self):
        self.assertEqual(
            normalize_youtube_url(r"https://www.youtube.com/watch\?v\=abc"),
            "https://www.youtube.com/watch?v=abc",
        )

    def test_rejects_non_youtube_url(self):
        with self.assertRaisesRegex(ValueError, "supported YouTube"):
            normalize_youtube_url("https://example.com/watch?v=abc")

    def test_disabled_whisper_fails_before_loading_transcriber(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "source.mp4"
            video.write_bytes(b"video")
            downloaded = DownloadResult(video=video, subtitle=None, metadata={})
            config = AppConfig(
                work_dir=root / "work", whisper=WhisperConfig(enabled=False)
            )
            with patch(
                "subtitle_pipeline.pipeline.download_youtube", return_value=downloaded
            ), patch("subtitle_pipeline.pipeline.transcribe_with_whisper") as transcribe:
                with self.assertRaisesRegex(RuntimeError, "fallback is disabled"):
                    run_pipeline("https://youtu.be/test", config, upload_override=False)
            transcribe.assert_not_called()

    def test_uses_existing_subtitle_and_respects_no_upload_override(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "source.mp4"
            subtitle = root / "source.en.srt"
            video.write_bytes(b"video")
            write_srt([Cue(0, 1, "hello")], subtitle)
            downloaded = DownloadResult(
                video=video,
                subtitle=subtitle,
                metadata={"title": "Title", "description": "Description"},
            )

            class FakeTranslator:
                def __init__(self, config, api_key):
                    pass

                def translate(self, cues):
                    return [Cue(cue.start, cue.end, "你好") for cue in cues]

                def translate_metadata(self, title, description):
                    return "中文标题", "中文简介"

            config = AppConfig(work_dir=root / "work", upload=UploadConfig(enabled=True))
            with patch("subtitle_pipeline.pipeline.download_youtube", return_value=downloaded), patch(
                "subtitle_pipeline.pipeline.llm_api_key", return_value="secret"
            ), patch("subtitle_pipeline.pipeline.OpenAICompatibleTranslator", FakeTranslator), patch(
                "subtitle_pipeline.pipeline.render_subtitles",
                side_effect=lambda _video, _subtitle, output, _config: output.write_bytes(b"out")
                or output,
            ), patch("subtitle_pipeline.pipeline.transcribe_with_whisper") as whisper, patch(
                "subtitle_pipeline.pipeline.upload_to_bilibili"
            ) as upload:
                result = run_pipeline(
                    "https://www.youtube.com/watch?v=1",
                    config,
                    upload_override=False,
                )

            self.assertFalse(result.uploaded)
            self.assertEqual(result.translated_subtitle.read_text(encoding="utf-8").count("你好"), 1)
            self.assertTrue((result.job_dir / "manifest.json").is_file())
            metadata = result.translated_metadata.read_text(encoding="utf-8")
            self.assertIn("中文标题", metadata)
            self.assertIn("中文简介", metadata)
            whisper.assert_not_called()
            upload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
