import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from subtitle_pipeline.config import LLMConfig, SegmentationConfig
from subtitle_pipeline.subtitles import Cue
from subtitle_pipeline.translate import (
    CueTranslationRecord,
    OpenAICompatibleTranslator,
    TranslationError,
    _collapsed_start_runs,
    _coerce_display_segments,
    _joint_translation_signature,
    _load_joint_translation_cache,
    _log_response_usage,
    _parse_joint_records,
    _parse_json_object,
    _protected_source_boundaries,
    _remove_terminal_period,
    _without_source_punctuation,
    _validate_joint_target_language,
    _validate_joint_timing,
    _validate_joint_records,
    _validate_display_segments,
)


class TranslationTests(unittest.TestCase):
    def test_joint_translation_restores_timing_and_compact_gap_input(self):
        translator = OpenAICompatibleTranslator(LLMConfig(thinking="disabled"), "secret")
        cues = [
            Cue(0, 0.4, "夢"),
            Cue(0.7, 1.0, "パワー。"),
            Cue(1.0, 1.8, "次"),
        ]
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"cues":['
                            '{"start_id":0,"end_id":1,"text":"梦想就是力量"},'
                            '{"start_id":2,"end_id":2,"text":"Power"}'
                            "]}"
                        )
                    },
                }
            ]
        }
        with patch.object(translator, "_request", return_value=response) as request:
            result = translator.plan_and_translate(
                cues,
                SegmentationConfig(),
                max_line_units=20,
            )
        self.assertEqual(
            result.source_cues,
            [Cue(0, 1.0, "夢パワー。"), Cue(1.0, 1.8, "次")],
        )
        self.assertEqual(
            result.translated_cues,
            [Cue(0, 1.0, "梦想就是力量"), Cue(1.0, 1.8, "Power")],
        )
        prompt = request.call_args.args[0]["messages"][1]["content"]
        self.assertIn(
            "Unit columns: [id,duration_ms,gap_after_ms,speaker,kind,text]",
            prompt,
        )
        self.assertIn('[[0,400,300,null,"speech","夢"]', prompt)
        self.assertIn('[1,300,0,null,"speech","パワー"]', prompt)
        self.assertNotIn("パワー。", prompt)
        self.assertIn("ASR punctuation has been removed", prompt)
        self.assertNotIn("Required individual IDs", prompt)
        self.assertNotIn("valid end_id values", prompt)
        self.assertLess(prompt.index("REFERENCE:"), prompt.index("Required ID range:"))
        self.assertIn("20 full-width characters", prompt)
        self.assertIn("REFERENCE.characters contains entity instances", prompt)
        self.assertIn("context_only=true", prompt)
        self.assertIn("Return exactly one JSON object with a cues array", prompt)
        self.assertIn("Never omit, genericize, or paraphrase a source name", prompt)
        self.assertIn("that record's text must explicitly contain the mapped target name", prompt)
        self.assertEqual(request.call_args.args[0]["max_tokens"], 16384)
        self.assertEqual(request.call_args.args[0]["thinking"], {"type": "disabled"})
        self.assertEqual(
            request.call_args.args[0]["response_format"], {"type": "json_object"}
        )

    def test_joint_prompt_source_view_removes_unicode_punctuation(self):
        self.assertEqual(
            _without_source_punctuation('「使ってる。ね？」 BanG Dream!'),
            "使ってるね BanG Dream",
        )

    def test_joint_parser_accepts_legacy_and_equivalent_record_containers(self):
        records = [
            {"start_id": 0, "end_id": 1, "text": "甲"},
            {"start_id": 2, "end_id": 3, "text": "乙"},
        ]
        ndjson = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
        adjacent = ",".join(json.dumps(record, ensure_ascii=False) for record in records)

        self.assertEqual(_parse_joint_records(json.dumps({"cues": records})), records)
        self.assertEqual(_parse_joint_records(json.dumps(records)), records)
        self.assertEqual(_parse_joint_records(ndjson), records)
        self.assertEqual(_parse_joint_records(adjacent), records)

    def test_joint_parser_rejects_explanatory_text(self):
        with self.assertRaisesRegex(ValueError, "invalid joint cue JSON"):
            _parse_joint_records(
                'Here you go: {"start_id":0,"end_id":0,"text":"甲"}'
            )

    def test_logs_deepseek_cache_usage(self):
        with self.assertLogs(level="INFO") as captured:
            _log_response_usage(
                {
                    "usage": {
                        "prompt_tokens": 1000,
                        "prompt_cache_hit_tokens": 750,
                        "prompt_cache_miss_tokens": 250,
                        "completion_tokens": 80,
                        "total_tokens": 1080,
                    }
                }
            )

        message = "\n".join(captured.output)
        self.assertIn("cache_hit=750", message)
        self.assertIn("cache_miss=250", message)
        self.assertIn("cache_hit_rate=75.0%", message)

    def test_nonfinal_window_always_carries_and_replans_last_cue(self):
        translator = OpenAICompatibleTranslator(LLMConfig(), "secret")
        cues = [Cue(index, index + 1, text) for index, text in enumerate("甲乙丙丁戊")]
        responses = [
            {
                "choices": [{"message": {"content": (
                    '{"start_id":0,"end_id":0,"text":"一"}\n'
                    '{"start_id":1,"end_id":2,"text":"旧尾句"}'
                )}}]
            },
            {
                "choices": [{"message": {"content": (
                    '{"start_id":1,"end_id":3,"text":"重新分组"}\n'
                    '{"start_id":4,"end_id":4,"text":"五"}'
                )}}]
            },
        ]
        config = SegmentationConfig(model_window_cues=3)
        with patch.object(translator, "_request", side_effect=responses) as request:
            result = translator.plan_and_translate(
                cues,
                config,
                max_line_units=20,
            )
        self.assertEqual(
            [cue.text for cue in result.translated_cues], ["一", "重新分组", "五"]
        )
        self.assertEqual(request.call_count, 2)
        second_prompt = request.call_args_list[1].args[0]["messages"][1]["content"]
        self.assertIn("Required ID range: 1-4", second_prompt)
        self.assertNotIn("旧尾句", second_prompt)

    def test_joint_translation_tolerates_text_beyond_prompt_limit_within_frame(self):
        translator = OpenAICompatibleTranslator(LLMConfig(), "secret")
        cues = [Cue(0, 1, "長い文")]
        response = {
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": '{"start_id":0,"end_id":0,"text":"一二三四五六七八九十甲"}'
                },
            }]
        }
        with patch.object(translator, "_request", return_value=response) as request:
            result = translator.plan_and_translate(
                cues,
                SegmentationConfig(),
                max_line_units=10,
                hard_max_line_units=12,
            )

        self.assertEqual(result.translated_cues[0].text, "一二三四五六七八九十甲")
        prompt = request.call_args.args[0]["messages"][1]["content"]
        self.assertIn("10 full-width characters", prompt)
        self.assertNotIn("12 full-width characters", prompt)

    def test_single_cue_nonfinal_window_expands_without_hard_cut(self):
        translator = OpenAICompatibleTranslator(LLMConfig(max_retries=1), "secret")
        cues = [Cue(index, index + 1, text) for index, text in enumerate("甲乙丙丁戊")]
        responses = [
            {"choices": [{"message": {"content": (
                '{"start_id":0,"end_id":1,"text":"未完成"}'
            )}}]},
            {"choices": [{"message": {"content": (
                '{"start_id":0,"end_id":1,"text":"前句"}\n'
                '{"start_id":2,"end_id":3,"text":"未完成尾句"}'
            )}}]},
            {"choices": [{"message": {"content": (
                '{"start_id":2,"end_id":4,"text":"最终句"}'
            )}}]},
        ]
        with patch.object(translator, "_request", side_effect=responses) as request:
            result = translator.plan_and_translate(
                cues,
                SegmentationConfig(model_window_cues=2),
                max_line_units=20,
            )
        self.assertEqual([cue.text for cue in result.translated_cues], ["前句", "最终句"])
        ranges = [
            call.args[0]["messages"][1]["content"]
            for call in request.call_args_list
        ]
        self.assertIn("Required ID range: 0-1", ranges[0])
        self.assertIn("Required ID range: 0-3", ranges[1])
        self.assertIn("Required ID range: 2-4", ranges[2])

    def test_discarded_nonfinal_tail_may_have_collapsed_timing(self):
        translator = OpenAICompatibleTranslator(LLMConfig(max_retries=1), "secret")
        cues = [
            Cue(0, 1, "甲"),
            Cue(1, 2, "乙"),
            Cue(1, 3, "丙"),
        ]
        responses = [
            {"choices": [{"message": {"content": (
                '{"start_id":0,"end_id":0,"text":"一"}\n'
                '{"start_id":1,"end_id":1,"text":"未完成"}'
            )}}]},
            {"choices": [{"message": {"content": (
                '{"start_id":1,"end_id":2,"text":"完成"}'
            )}}]},
        ]
        with patch.object(translator, "_request", side_effect=responses):
            result = translator.plan_and_translate(
                cues,
                SegmentationConfig(model_window_cues=2),
                max_line_units=20,
            )
        self.assertEqual([cue.text for cue in result.translated_cues], ["一", "完成"])

    def test_collapsed_timing_runs_are_compact(self):
        cues = [
            Cue(0, 1, "甲"),
            Cue(1, 1.2, "乙"),
            Cue(1, 1.2, "丙"),
            Cue(1, 2, "丁"),
        ]
        runs = _collapsed_start_runs(cues, 0, 4)
        self.assertEqual(runs, [[1, 3]])

    def test_failed_joint_range_shrinks_after_retries(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(max_retries=1), "secret"
        )
        cues = [Cue(index, index + 1, text) for index, text in enumerate("甲乙丙丁")]
        responses = [
            {"choices": [{"message": {"content": "not json"}}]},
            {"choices": [{"message": {"content": (
                '{"start_id":0,"end_id":0,"text":"一"}\n'
                '{"start_id":1,"end_id":1,"text":"尾"}'
            )}}]},
            {"choices": [{"message": {"content": (
                '{"start_id":1,"end_id":3,"text":"后句"}'
            )}}]},
        ]
        with patch.object(translator, "_request", side_effect=responses) as request:
            result = translator.plan_and_translate(
                cues,
                SegmentationConfig(model_window_cues=4),
                max_line_units=20,
            )
        self.assertEqual([cue.text for cue in result.translated_cues], ["一", "后句"])
        prompts = [call.args[0]["messages"][1]["content"] for call in request.call_args_list]
        self.assertIn("Required ID range: 0-3", prompts[0])
        self.assertIn("Required ID range: 0-1", prompts[1])

    def test_joint_finish_reason_is_retried_with_expected_range(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(max_retries=2), "secret"
        )
        cues = [Cue(0, 1, "甲")]
        record = '{"start_id":0,"end_id":0,"text":"一"}'
        responses = [
            {"choices": [{"finish_reason": "length", "message": {"content": record}}]},
            {"choices": [{"finish_reason": "stop", "message": {"content": record}}]},
        ]
        with patch.object(translator, "_request", side_effect=responses) as request, patch(
            "subtitle_pipeline.translate.time.sleep"
        ):
            result = translator.plan_and_translate(
                cues,
                SegmentationConfig(),
                max_line_units=20,
            )
        self.assertEqual([cue.text for cue in result.translated_cues], ["一"])
        retry_prompt = request.call_args_list[1].args[0]["messages"][1]["content"]
        self.assertIn("finish_reason=length", retry_prompt)
        self.assertIn("Required ID range: 0-0", retry_prompt)
        self.assertGreater(retry_prompt.index("RETRY:"), retry_prompt.index("TARGET:"))

    def test_joint_validation_rejects_gaps_empty_and_width(self):
        records = _validate_joint_records(
            [
                {"start_id": 0, "end_id": 0, "text": "虽然如此"},
                {"start_id": 1, "end_id": 1, "text": "头很大"},
            ],
            0,
            2,
            10,
        )
        self.assertEqual([record.end_id for record in records], [0, 1])
        with self.assertRaisesRegex(TranslationError, "expected start_id=1"):
            _validate_joint_records(
                [
                    {"start_id": 0, "end_id": 0, "text": "一"},
                    {"start_id": 2, "end_id": 2, "text": "三"},
                ],
                0,
                3,
                10,
            )
        with self.assertRaisesRegex(TranslationError, "empty"):
            _validate_joint_records(
                [{"start_id": 0, "end_id": 0, "text": " "}], 0, 1, 10
            )
        with self.assertRaisesRegex(TranslationError, "not one-line"):
            _validate_joint_records(
                [{"start_id": 0, "end_id": 0, "text": "一二三"}],
                0,
                1,
                2,
            )
        with self.assertRaisesRegex(TranslationError, "Japanese kana"):
            _validate_joint_target_language(
                [CueTranslationRecord(0, 0, "実存でもない")], "简体中文"
            )
        with self.assertRaisesRegex(TranslationError, "non-positive"):
            _validate_joint_timing(
                [CueTranslationRecord(0, 0, "字幕")],
                [Cue(1, 2, "甲"), Cue(1, 3, "乙")],
            )

    def test_protects_particles_and_adnominals_from_cue_boundaries(self):
        cues = [
            Cue(0, 1, "この"),
            Cue(1, 2, "夢"),
            Cue(2, 3, "だ"),
            Cue(3, 4, "って"),
            Cue(4, 5, "分"),
            Cue(5, 6, "だけ"),
            Cue(6, 7, "目印"),
        ]
        protected = _protected_source_boundaries(cues, {})
        self.assertTrue({0, 2, 4, 5}.issubset(protected))

    def test_joint_cache_keeps_only_longest_valid_prefix(self):
        cues = [Cue(index, index + 1, str(index)) for index in range(4)]
        config = SegmentationConfig(model_window_cues=2)
        llm = LLMConfig()
        signature = _joint_translation_signature(cues, config, llm, {}, 10)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cue-translation-cache.json"
            path.write_text(
                json.dumps(
                    {
                        "signature": signature,
                        "records": [
                            {"start_id": 0, "end_id": 0, "text": "零"},
                            {"start_id": 2, "end_id": 2, "text": "坏"},
                        ],
                        "next_window_end": 4,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            records, next_end = _load_joint_translation_cache(
                path, signature, cues, 10, "简体中文"
            )
        self.assertEqual(records, [CueTranslationRecord(0, 0, "零")])
        self.assertEqual(next_end, 4)

    def test_joint_cache_resumes_from_confirmed_prefix_only(self):
        config = SegmentationConfig(model_window_cues=2)
        llm = LLMConfig(max_retries=1)
        cues = [Cue(index, index + 1, text) for index, text in enumerate("甲乙丙丁")]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cue-translation-cache.json"
            first = OpenAICompatibleTranslator(llm, "secret")
            first_responses = [
                {"choices": [{"message": {"content": (
                    '{"start_id":0,"end_id":0,"text":"一"}\n'
                    '{"start_id":1,"end_id":1,"text":"待重做"}'
                )}}]},
                {"choices": [{"message": {"content": "invalid"}}]},
                {"choices": [{"message": {"content": "invalid"}}]},
            ]
            with patch.object(first, "_request", side_effect=first_responses):
                with self.assertRaises(TranslationError):
                    first.plan_and_translate(
                        cues,
                        config,
                        max_line_units=20,
                        cache_path=path,
                    )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["records"], [
                {"start_id": 0, "end_id": 0, "text": "一"}
            ])

            second = OpenAICompatibleTranslator(llm, "secret")
            response = {"choices": [{"message": {"content": (
                '{"start_id":1,"end_id":3,"text":"完成后句"}'
            )}}]}
            with patch.object(second, "_request", return_value=response) as request:
                result = second.plan_and_translate(
                    cues,
                    config,
                    max_line_units=20,
                    cache_path=path,
                )
        self.assertEqual([cue.text for cue in result.translated_cues], ["一", "完成后句"])
        prompt = request.call_args.args[0]["messages"][1]["content"]
        self.assertIn("Required ID range: 1-3", prompt)
        self.assertIn('"translation":"一"', prompt)
        self.assertNotIn("待重做", prompt)

    def test_semantically_splits_only_cues_too_wide_at_standard_size(self):
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
            prompt = call.args[0]["messages"][1]["content"]
            self.assertIn("ASR may substitute homophonic kanji", prompt)
            self.assertIn("never expand a short name into a full name", prompt)
            self.assertIn("If the evidence is ambiguous or insufficient, do not guess", prompt)

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
