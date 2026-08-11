import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from subtitle_pipeline.config import AppConfig, UploadConfig
from subtitle_pipeline.media import DownloadResult
from subtitle_pipeline.pipeline import (
    _canonicalize_catalog_tags,
    _merge_tags,
    _subtitle_evidence,
    _translation_context,
    _youtube_metadata_context,
    normalize_youtube_url,
    run_pipeline,
)
from subtitle_pipeline.subtitles import Cue, write_srt
from subtitle_pipeline.translate import CueTranslationResult


class PipelineTests(unittest.TestCase):
    def test_bang_dream_glossary_models_miyako_as_character(self):
        context = _translation_context(
            {
                "title": "夢限大みゅーたいぷ 藤都子",
                "channel": "藤都子 -Fuji Miyako-",
            },
            [],
        )

        miyako = next(
            character
            for character in context["characters"]
            if character["id"] == "fuji_miyako"
        )
        self.assertEqual(miyako["canonical"], "藤都子")
        self.assertIn("Fuji Miyako", miyako["aliases"])
        short_names = {
            item["source"]: item for item in miyako["short_names"]
        }
        self.assertEqual(short_names["Miyako"]["target"], "都子")
        self.assertTrue(short_names["Miyako"]["context_only"])
        self.assertNotIn("Miyako", context["terms"])

    def test_translation_glossary_keeps_legacy_terms_format_compatible(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "legacy.json"
            path.write_text(
                json.dumps(
                    {
                        "name": "Legacy",
                        "background": "Legacy glossary",
                        "match": ["legacy-video"],
                        "terms": {"旧名": "旧译名"},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            context = _translation_context(
                {"title": "legacy-video"}, [str(path)]
            )

        self.assertEqual(context["terms"]["旧名"], "旧译名")

    def test_translation_glossary_accepts_characters_without_terms(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "characters.json"
            path.write_text(
                json.dumps(
                    {
                        "name": "Characters",
                        "background": "Character glossary",
                        "match": ["entity-video"],
                        "characters": [
                            {
                                "id": "example",
                                "canonical": "例子",
                                "source_name": "れいこ",
                                "aliases": ["レイコ"],
                                "short_names": [
                                    {
                                        "source": "れい",
                                        "target": "小例",
                                        "context_only": True,
                                    }
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            context = _translation_context(
                {"title": "entity-video"}, [str(path)]
            )

        self.assertEqual(context["characters"][-1]["id"], "example")
        self.assertEqual(context["terms"], {})

    def test_custom_character_overrides_builtin_character_by_id(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "override.json"
            path.write_text(
                json.dumps(
                    {
                        "name": "Preferred names",
                        "background": "User-preferred translation",
                        "match": ["夢限大みゅーたいぷ"],
                        "characters": [
                            {
                                "id": "fuji_miyako",
                                "canonical": "自定义都子",
                                "source_name": "藤都子",
                                "aliases": [],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            context = _translation_context(
                {"title": "夢限大みゅーたいぷ"}, [str(path)]
            )

        matches = [
            character
            for character in context["characters"]
            if character["id"] == "fuji_miyako"
        ]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["canonical"], "自定义都子")

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

    def test_uses_qwen_subtitle_and_respects_no_upload_override(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "source.mp4"
            subtitle = root / "source.en.srt"
            video.write_bytes(b"video")
            write_srt([Cue(0, 1, "hello")], subtitle)
            downloaded = DownloadResult(
                video=video,
                metadata={"title": "Title", "description": "Description"},
            )

            class FakeTranslator:
                def __init__(self, config, api_key):
                    pass

                def plan_and_translate(self, cues, config, **context):
                    self.joint_context = context
                    return CueTranslationResult(
                        cues,
                        [Cue(cue.start, cue.end, "你好") for cue in cues],
                    )

                def translate_metadata(self, title, description, **context):
                    self.context = context
                    return "中文标题", "中文简介", "内容摘要", ["动画", "音乐企划"]

            config = AppConfig(
                work_dir=root / "work", upload=UploadConfig(enabled=True)
            )
            with patch(
                "subtitle_pipeline.pipeline.download_youtube", return_value=downloaded
            ), patch(
                "subtitle_pipeline.pipeline.transcribe_with_qwen",
                return_value=subtitle,
            ) as asr, patch(
                "subtitle_pipeline.pipeline.llm_api_key", return_value="secret"
            ), patch(
                "subtitle_pipeline.pipeline.OpenAICompatibleTranslator", FakeTranslator
            ), patch(
                "subtitle_pipeline.pipeline.subtitle_layout"
            ) as layout, patch(
                "subtitle_pipeline.pipeline.render_subtitles",
                side_effect=lambda _video, _subtitle, output, _config, **kwargs: (
                    output.write_bytes(b"out") or output
                ),
            ), patch(
                "subtitle_pipeline.pipeline.upload_to_bilibili"
            ) as upload:
                layout.return_value.max_line_units = 12
                layout.return_value.font_size = 48
                result = run_pipeline(
                    "https://www.youtube.com/watch?v=1",
                    config,
                    upload_override=False,
                )

            self.assertFalse(result.uploaded)
            self.assertEqual(
                result.translated_subtitle.read_text(encoding="utf-8").count("你好"),
                1,
            )
            self.assertTrue((result.job_dir / "manifest.json").is_file())
            self.assertTrue((result.job_dir / "source.semantic.srt").is_file())
            self.assertFalse((result.job_dir / "source-segments-cache.json").exists())
            self.assertFalse((result.job_dir / "translation-cache.json").exists())
            self.assertFalse((result.job_dir / "display-segments-cache.json").exists())
            metadata = result.translated_metadata.read_text(encoding="utf-8")
            self.assertIn("中文标题", metadata)
            self.assertIn("中文简介", metadata)
            self.assertIn("内容摘要", metadata)
            self.assertIn("音乐企划", metadata)
            asr.assert_called_once()
            upload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
