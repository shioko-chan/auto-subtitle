import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from subtitle_pipeline.config import LLMConfig, SegmentationConfig
from subtitle_pipeline.subtitles import Cue
from subtitle_pipeline.translate import (
    OpenAICompatibleTranslator,
    TranslationError,
    _coerce_display_segments,
    _parse_json_object,
    _remove_terminal_period,
    _validate_display_segments,
)


class TranslationTests(unittest.TestCase):
    def test_source_segmentation_uses_model_boundaries_for_long_unpunctuated_text(self):
        translator = OpenAICompatibleTranslator(LLMConfig(), "secret")
        cues = [
            Cue(0, 2, "夢"),
            Cue(2, 4, "は"),
            Cue(4, 6, "パワー"),
            Cue(6, 8, "次"),
        ]
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": '{"id":0,"segments":["夢はパワー","次"]}'
                    },
                }
            ]
        }
        with patch.object(translator, "_request", return_value=response) as request:
            result = translator.segment_source_cues(
                cues, SegmentationConfig(review_duration_seconds=6)
            )
        self.assertEqual(result, [Cue(0, 6, "夢はパワー"), Cue(6, 8, "次")])
        prompt = request.call_args.args[0]["messages"][1]["content"]
        self.assertIn("Decision token range: 0-2", prompt)
        self.assertIn("TARGET:\n夢はパワー次", prompt)
        self.assertNotIn('"gap_after"', prompt)

    def test_source_segmentation_never_uses_model_window_edges_as_boundaries(self):
        translator = OpenAICompatibleTranslator(LLMConfig(), "secret")
        cues = [Cue(index * 2, index * 2 + 2, text) for index, text in enumerate("甲乙丙丁戊")]
        target_texts = ["甲乙丙丁", "乙丙丁戊"]
        responses = [
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"id": window_id, "segments": [target_text]},
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }
            for window_id, target_text in enumerate(target_texts)
        ]
        config = SegmentationConfig(
            review_duration_seconds=6,
            model_window_cues=2,
            model_context_cues=1,
        )
        with patch.object(translator, "_request", side_effect=responses) as request:
            result = translator.segment_source_cues(cues, config)
        self.assertEqual(result, [Cue(0, 10, "甲乙丙丁戊")])
        self.assertEqual(request.call_count, 2)

    def test_short_punctuated_source_uses_trusted_boundary_without_model(self):
        translator = OpenAICompatibleTranslator(LLMConfig(), "secret")
        cues = [Cue(0, 1, "そう"), Cue(1, 2, "です。"), Cue(2, 3, "はい")]
        with patch.object(translator, "_request") as request:
            result = translator.segment_source_cues(cues, SegmentationConfig())
        self.assertEqual(result, [Cue(0, 2, "そうです。"), Cue(2, 3, "はい")])
        request.assert_not_called()

    def test_source_semantic_boundary_inside_aligner_item_snaps_to_item_edge(self):
        translator = OpenAICompatibleTranslator(LLMConfig(), "secret")
        cues = [Cue(0, 2, "はいそう"), Cue(2, 4, "次")]
        response = {
            "choices": [
                {
                    "message": {
                        "content": '{"id":0,"segments":["はい","そう次"]}'
                    }
                }
            ]
        }
        config = SegmentationConfig(review_source_chars=1)
        with patch.object(translator, "_request", return_value=response):
            result = translator.segment_source_cues(cues, config)
        self.assertEqual(result, cues)

    def test_semantically_splits_only_cues_too_wide_at_85_percent(self):
        translator = OpenAICompatibleTranslator(LLMConfig(), "secret")
        cues = [Cue(0, 4, "短句"), Cue(4, 8, "一二三四五六七八九十甲乙")]
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": '{"id":1,"segments":["一二三四","五六七八","九十甲乙"]}'
                    },
                }
            ]
        }
        with patch.object(translator, "_request", return_value=response) as request:
            result = translator.segment_for_single_line(
                cues,
                max_line_units=10,
                font_size=100,
                min_font_scale=0.85,
            )
        self.assertEqual(result, {1: ["一二三四五六七八", "九十甲乙"]})
        prompt = request.call_args.args[0]["messages"][1]["content"]
        self.assertIn("Required target IDs: [1]", prompt)
        self.assertIn('"id": 0', prompt)

    def test_display_segments_must_preserve_text_and_fit(self):
        self.assertIsNone(_validate_display_segments("甲乙丙丁", ["甲乙", "丙丁"], 2))
        self.assertIn(
            "preserve",
            _validate_display_segments("甲乙丙丁", ["甲乙", "丁"], 2),
        )
        self.assertIn(
            "one-line",
            _validate_display_segments("甲乙丙丁", ["甲乙丙", "丁"], 2),
        )

    def test_recovers_semantic_boundaries_without_dropping_original_punctuation(self):
        segments, error = _coerce_display_segments(
            "前半句，后半句", ["前半句", "后半句"], 4
        )
        self.assertIsNone(error)
        self.assertEqual(segments, ["前半句，", "后半句"])

    def test_ssl_context_combines_platform_and_certifi_ca(self):
        with patch("subtitle_pipeline.translate.ssl.create_default_context") as create, patch(
            "subtitle_pipeline.translate.certifi.where", return_value="/ca/certifi.pem"
        ):
            translator = OpenAICompatibleTranslator(LLMConfig(), "secret")
        self.assertIs(translator.ssl_context, create.return_value)
        create.return_value.load_verify_locations.assert_called_once_with(
            cafile="/ca/certifi.pem"
        )

    def test_translates_in_batches_and_preserves_timing(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(batch_size=2, thinking="disabled"), "secret"
        )
        responses = [
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"id":0,"text":"甲。"}\n{"id":1,"text":"乙？"}'
                        }
                    }
                ]
            },
            {
                "choices": [
                    {"message": {"content": '{"id":2,"text":"丙."}'}}
                ]
            },
        ]
        cues = [Cue(0, 1, "a"), Cue(1, 2, "b"), Cue(2, 3, "c")]
        with patch.object(translator, "_request", side_effect=responses) as request:
            result = translator.translate(cues)
        self.assertEqual([cue.text for cue in result], ["甲", "乙？", "丙"])
        self.assertEqual([(cue.start, cue.end) for cue in result], [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(request.call_count, 2)
        for call in request.call_args_list:
            self.assertEqual(call.args[0]["thinking"], {"type": "disabled"})
            self.assertEqual(call.args[0]["max_tokens"], 16384)
            self.assertNotIn("response_format", call.args[0])

    def test_retries_length_response_and_names_expected_ids(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(batch_size=2, max_retries=2), "secret"
        )
        responses = [
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "content": '{"id":0,"text":"甲"}\n{"id":1,"text":'
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": '{"id":1,"text":"乙"}'
                        },
                    }
                ]
            },
        ]
        with patch.object(translator, "_request", side_effect=responses) as request, patch(
            "subtitle_pipeline.translate.time.sleep"
        ):
            result = translator.translate([Cue(0, 1, "a"), Cue(1, 2, "b")])
        self.assertEqual([cue.text for cue in result], ["甲", "乙"])
        retry_prompt = request.call_args_list[1].args[0]["messages"][1]["content"]
        self.assertIn("Required target IDs: [1]", retry_prompt)
        self.assertIn("finish_reason=length", retry_prompt)
        self.assertIn('"translation": "甲"', retry_prompt)

    def test_splits_failed_batch_until_valid_sub_batches(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(batch_size=4, max_retries=1), "secret"
        )
        responses = [
            {"choices": [{"message": {"content": "not json"}}]},
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"id":0,"text":"甲"}\n{"id":1,"text":"乙"}'
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": '{"id":2,"text":""}'}}]},
            {
                "choices": [
                    {"message": {"content": '{"id":2,"text":"丙"}'}}
                ]
            },
            {
                "choices": [
                    {"message": {"content": '{"id":3,"text":"丁"}'}}
                ]
            },
        ]
        cues = [Cue(i, i + 1, str(i)) for i in range(4)]
        with patch.object(translator, "_request", side_effect=responses) as request:
            result = translator.translate(cues)
        self.assertEqual([cue.text for cue in result], ["甲", "乙", "丙", "丁"])
        self.assertEqual(request.call_count, 5)

    def test_persists_successful_batches_and_resumes_only_missing_cues(self):
        config = LLMConfig(batch_size=2, max_retries=1)
        cues = [Cue(i, i + 1, str(i)) for i in range(4)]
        with tempfile.TemporaryDirectory() as temp:
            cache_path = Path(temp) / "translation-cache.json"
            first = OpenAICompatibleTranslator(config, "secret")
            first_responses = [
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"id":0,"text":"甲"}\n{"id":1,"text":"乙"}'
                            }
                        }
                    ]
                },
                {"choices": [{"message": {"content": "not json"}}]},
                {"choices": [{"message": {"content": "still not json"}}]},
            ]
            with patch.object(first, "_request", side_effect=first_responses):
                with self.assertRaisesRegex(TranslationError, "translation failed"):
                    first.translate(cues, cache_path=cache_path)

            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertEqual(cached["translations"], {"0": "甲", "1": "乙"})

            second = OpenAICompatibleTranslator(config, "secret")
            response = {
                "choices": [
                    {
                        "message": {
                            "content": '{"id":2,"text":"丙"}\n{"id":3,"text":"丁"}'
                        }
                    }
                ]
            }
            with patch.object(second, "_request", return_value=response) as request:
                result = second.translate(cues, cache_path=cache_path)
            self.assertEqual([cue.text for cue in result], ["甲", "乙", "丙", "丁"])
            request.assert_called_once()

    def test_persists_valid_ndjson_line_before_truncated_line(self):
        config = LLMConfig(batch_size=2, max_retries=1)
        cues = [Cue(0, 1, "a"), Cue(1, 2, "b")]
        response = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": '{"id":0,"text":"甲"}\n{"id":1,"text":'
                    },
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temp:
            cache_path = Path(temp) / "translation-cache.json"
            translator = OpenAICompatibleTranslator(config, "secret")
            with patch.object(translator, "_request", return_value=response):
                with self.assertRaisesRegex(TranslationError, "translation failed"):
                    translator.translate(cues, cache_path=cache_path)
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertEqual(cached["translations"], {"0": "甲"})

    def test_parses_fenced_json_from_less_strict_provider(self):
        parsed = _parse_json_object('```json\n{"translations": []}\n```')
        self.assertEqual(parsed, {"translations": []})

    def test_removes_only_terminal_subtitle_periods(self):
        self.assertEqual(_remove_terminal_period("你好。"), "你好")
        self.assertEqual(_remove_terminal_period('他说：“好的。”'), '他说：“好的”')
        self.assertEqual(_remove_terminal_period("All right."), "All right")
        self.assertEqual(_remove_terminal_period("真的吗？"), "真的吗？")
        self.assertEqual(_remove_terminal_period("太好了！"), "太好了！")
        self.assertEqual(_remove_terminal_period("等等..."), "等等...")

    def test_translates_title_and_description_together(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(metadata_description_max_chars=13, thinking="disabled"), "secret"
        )
        response = {
            "choices": [
                {
                    "message": {
                        "content": '{"title":"中文标题","description":"中文简介\\nhttps://example.com","content_summary":"内容摘要","tags":["#动画","音乐"]}'
                    }
                }
            ]
        }
        with patch.object(translator, "_request", return_value=response) as request:
            result = translator.translate_metadata(
                "Original",
                "A very long description",
                youtube_context={"channel": "BanG Dream Channel☆"},
                subtitle_evidence="ガルパの新しいゲーム",
                ip_aliases={"BanG Dream!": ["バンドリ"]},
                bilibili_tag_catalog={"BanG Dream": {"heat": 100}},
            )
        self.assertEqual(
            result,
            ("中文标题", "中文简介\nhttps://example.com", "内容摘要", ["动画", "音乐"]),
        )
        body = request.call_args.args[0]
        self.assertEqual(body["thinking"], {"type": "disabled"})
        prompt = body["messages"][1]["content"]
        self.assertIn('"description": "A very long d"', prompt)
        self.assertIn("BanG Dream Channel☆", prompt)
        self.assertIn("ガルパの新しいゲーム", prompt)
        self.assertIn('"heat": 100', prompt)

    def test_cleans_and_deduplicates_generated_tags(self):
        from subtitle_pipeline.translate import _clean_tags

        self.assertEqual(
            _clean_tags(["#动画", "动画", "音乐, BanG Dream", ""], 3),
            ["动画", "音乐", "BanG Dream"],
        )


if __name__ == "__main__":
    unittest.main()
