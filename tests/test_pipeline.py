import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from subtitle_pipeline.config import AppConfig, UploadConfig, WhisperConfig
from subtitle_pipeline.media import DownloadResult
from subtitle_pipeline.pipeline import (
    _canonicalize_catalog_tags,
    _merge_tags,
    _subtitle_evidence,
    _youtube_metadata_context,
    normalize_youtube_url,
    run_pipeline,
)
from subtitle_pipeline.subtitles import Cue, write_srt


class PipelineTests(unittest.TestCase):
    def test_builds_rich_youtube_context_and_subtitle_evidence(self):
        context = _youtube_metadata_context(
            {
                "channel": "BanG Dream Channel☆",
                "uploader": "Bushiroad",
                "categories": ["Gaming"],
                "series": "Girls Band Party",
                "unrelated": "ignored",
            }
        )
        self.assertEqual(context["channel"], "BanG Dream Channel☆")
        self.assertEqual(context["series"], "Girls Band Party")
        self.assertNotIn("unrelated", context)
        evidence = _subtitle_evidence(
            [Cue(0, 1, "beginning"), Cue(1, 2, "middle"), Cue(2, 3, "ending")],
            20,
        )
        self.assertLessEqual(len(evidence), 20)
        self.assertIn("begin", evidence)

    def test_catalog_alias_resolves_to_hottest_canonical_tag(self):
        tags, matches = _canonicalize_catalog_tags(
            ["バンドリ"],
            {
                "BanG Dream": {"heat": 100, "aliases": ["バンドリ"]},
                "邦邦": {"heat": 20, "aliases": ["バンドリ"]},
            },
        )
        self.assertEqual(tags, ["BanG Dream"])
        self.assertEqual(matches[0]["heat"], 100)

    def test_merges_fixed_and_generated_tags_without_duplicates(self):
        self.assertEqual(
            _merge_tags(["中文字幕", "#动画"], ["动画", "音乐企划"], 10),
            ["中文字幕", "动画", "音乐企划"],
        )

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

                def translate(self, cues, **context):
                    return [Cue(cue.start, cue.end, "你好") for cue in cues]

                def translate_metadata(self, title, description, **context):
                    self.context = context
                    return "中文标题", "中文简介", "内容摘要", ["动画", "音乐企划"]

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
            self.assertIn("内容摘要", metadata)
            self.assertIn("音乐企划", metadata)
            whisper.assert_not_called()
            upload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
