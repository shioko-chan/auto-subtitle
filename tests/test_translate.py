import json
import tempfile
import unittest
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import call, patch

from subtitle_pipeline.config import LLMConfig, SegmentationConfig
from subtitle_pipeline.subtitles import Cue
from subtitle_pipeline.translate import (
    CueTranslationRecord,
    LLMHTTPError,
    OpenAICompatibleTranslator,
    TranslationError,
    _compact_fixed_translation_text,
    _compact_prompt_units_text,
    _compact_reference_text,
    _cue_plan_prompt,
    _cue_plan_range_groups,
    _cue_plan_signature,
    _is_nontransient_http_error,
    _log_response_usage,
    _majority_speaker,
    _merge_planned_and_fixed_records,
    _normalize_api_response,
    _parse_joint_records,
    _parse_json_object,
    _parse_retry_after,
    _prepare_api_request,
    _prompt_translation_context,
    _transient_retry_delay,
    _translation_window_ranges,
    _validate_joint_records,
    _validate_joint_target_language,
    _validated_boundary_repairs,
    _validated_translation_repairs,
    _without_source_punctuation,
)


class TranslationTests(unittest.TestCase):
    def test_split_planning_and_translation_have_independent_contracts_and_caches(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(thinking="disabled"), "secret"
        )
        cues = [
            Cue(0, 0.4, "夢"),
            Cue(0.7, 1.0, "パワー。"),
            Cue(1.0, 1.8, "次"),
        ]

        def response(body):
            prompt = body["messages"][1]["content"]
            if "Group every TARGET" in prompt:
                content = (
                    '{"cues":['
                    '{"start_id":0,"end_id":1},'
                    '{"start_id":2,"end_id":2}'
                    "]}"
                )
            else:
                self.assertIn("Cue boundaries are already final", prompt)
                self.assertIn("Japanese comes from ASR", prompt)
                self.assertIn("<0>夢パワー", prompt)
                self.assertNotIn('"start_id":0,"end_id":2', prompt)
                content = (
                    '{"translations":['
                    '{"cue_id":0,"text":"梦想就是力量"},'
                    '{"cue_id":1,"text":"接下来"}'
                    "]}"
                )
            return {
                "choices": [{"finish_reason": "stop", "message": {"content": content}}]
            }

        with tempfile.TemporaryDirectory() as temp:
            plan_cache = Path(temp) / "cue-plan-cache.json"
            translation_cache = Path(temp) / "cue-translation-cache.json"
            with patch.object(translator, "_request", side_effect=response) as request:
                result = translator.plan_and_translate(
                    cues,
                    SegmentationConfig(),
                    max_line_units=20,
                    plan_cache_path=plan_cache,
                    cache_path=translation_cache,
                )
            self.assertEqual(request.call_count, 2)
            plan_payload = json.loads(plan_cache.read_text(encoding="utf-8"))
            translation_payload = json.loads(
                translation_cache.read_text(encoding="utf-8")
            )

            cached = OpenAICompatibleTranslator(
                LLMConfig(thinking="disabled"), "secret"
            )
            with patch.object(cached, "_request") as cached_request:
                cached_result = cached.plan_and_translate(
                    cues,
                    SegmentationConfig(),
                    max_line_units=20,
                    plan_cache_path=plan_cache,
                    cache_path=translation_cache,
                )

        self.assertEqual(result, cached_result)
        self.assertEqual([cue.text for cue in result.source_cues], ["夢パワー。", "次"])
        self.assertEqual(
            [cue.text for cue in result.translated_cues],
            ["梦想就是力量", "接下来"],
        )
        self.assertEqual(
            set(plan_payload["records"][0]),
            {"start_id", "end_id", "source_text"},
        )
        self.assertEqual(translation_payload["records"][0]["cue_id"], 0)
        self.assertEqual(translation_payload["records"][0]["status"], "confirmed")
        cached_request.assert_not_called()

    def test_singing_cues_bypass_planner_and_keep_existing_boundaries(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(thinking="disabled"), "secret"
        )
        cues = [
            Cue(0.0, 2.0, "夢はパワー", kind="singing"),
            Cue(2.0, 4.0, "歌い続ける", kind="singing"),
        ]
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"translations":['
                            '{"cue_id":0,"text":"梦想就是力量"},'
                            '{"cue_id":1,"text":"继续歌唱"}'
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

        self.assertEqual(request.call_count, 1)
        prompt = request.call_args.args[0]["messages"][1]["content"]
        self.assertNotIn("Group every TARGET", prompt)
        self.assertEqual(
            [(cue.start, cue.end, cue.text) for cue in result.source_cues],
            [(0.0, 2.0, "夢はパワー"), (2.0, 4.0, "歌い続ける")],
        )

    def test_conditioned_speech_bypasses_planner_and_keeps_dicow_boundaries(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(thinking="disabled"), "secret"
        )
        cues = [
            Cue(1.0, 2.5, "お願いします", "A", "conditioned_speech"),
            Cue(1.8, 3.0, "よろしくお願いします", "B", "conditioned_speech"),
        ]
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"translations":['
                            '{"cue_id":0,"text":"拜托了"},'
                            '{"cue_id":1,"text":"请多关照"}'
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

        self.assertEqual(request.call_count, 1)
        prompt = request.call_args.args[0]["messages"][1]["content"]
        self.assertNotIn("Group every TARGET", prompt)
        self.assertEqual(
            [
                (cue.start, cue.end, cue.text, cue.speaker, cue.kind)
                for cue in result.source_cues
            ],
            [
                (1.0, 2.5, "お願いします", "A", "conditioned_speech"),
                (1.8, 3.0, "よろしくお願いします", "B", "conditioned_speech"),
            ],
        )

    def test_fixed_translation_retries_only_invalid_cue_text(self):
        translator = OpenAICompatibleTranslator(LLMConfig(max_retries=2), "secret")
        cues = [Cue(0, 1, "みやこ"), Cue(1, 2, "です")]
        responses = [
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": (
                                '{"cues":['
                                '{"start_id":0,"end_id":0},'
                                '{"start_id":1,"end_id":1}'
                                "]}"
                            )
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": (
                                '{"translations":['
                                '{"cue_id":0,"text":"みやこ"},'
                                '{"cue_id":1,"text":"是"}'
                                "]}"
                            )
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": ('{"translations":[{"cue_id":0,"text":"都子"}]}')
                        },
                    }
                ]
            },
        ]
        with patch.object(translator, "_request", side_effect=responses) as request:
            result = translator.plan_and_translate(
                cues,
                SegmentationConfig(),
                max_line_units=20,
            )
        self.assertEqual([cue.text for cue in result.translated_cues], ["都子", "是"])
        retry_prompt = request.call_args_list[2].args[0]["messages"][1]["content"]
        repair_target = retry_prompt.split("TARGET:\n", 1)[1]
        repair = json.loads(repair_target.split("\nRETRY:", 1)[0])
        self.assertEqual(repair["cue_id"], 0)
        self.assertEqual(repair["source"], "みやこ")
        self.assertEqual(repair["invalid_text"], "みやこ")
        self.assertIn("Japanese kana", repair["errors"][0])
        self.assertNotIn('"cue_id":1', repair_target)

    def test_fixed_translation_repair_records_width_and_missing_reasons(self):
        accepted, rejected = _validated_translation_repairs(
            [{"cue_id": 11, "text": "一二三"}],
            [11, 12],
            2,
            "简体中文",
        )

        self.assertEqual(accepted, {})
        self.assertEqual(rejected[11]["invalid_text"], "一二三")
        self.assertIn("exceeds maximum", rejected[11]["errors"][0])
        self.assertEqual(rejected[12]["invalid_text"], "")
        self.assertIn("missing", rejected[12]["errors"][0])

    def test_split_planner_repairs_only_artificial_window_boundary(self):
        translator = OpenAICompatibleTranslator(LLMConfig(max_concurrency=1), "secret")
        cues = [Cue(index, index + 1, text) for index, text in enumerate("甲乙丙丁")]

        def response(body):
            prompt = body["messages"][1]["content"]
            if "Required ID range: 0-1" in prompt:
                content = (
                    '{"cues":['
                    '{"start_id":0,"end_id":0},'
                    '{"start_id":1,"end_id":1}'
                    "]}"
                )
            elif "Required ID range: 2-3" in prompt:
                content = (
                    '{"cues":['
                    '{"start_id":2,"end_id":2},'
                    '{"start_id":3,"end_id":3}'
                    "]}"
                )
            elif "Independently replan every BOUNDARY block" in prompt:
                self.assertIn("<boundary:1|2 range=1-2>", prompt)
                self.assertNotIn("artificial Map boundary", prompt)
                self.assertNotIn("READ_ONLY_CUES", prompt)
                self.assertNotIn("WRITABLE", prompt)
                self.assertNotIn('"translation"', prompt)
                content = (
                    '{"repairs":[{"boundary_id":"1|2","cues":['
                    '{"start_id":1,"end_id":2}]}]}'
                )
            else:
                self.assertIn("<1>乙丙", prompt)
                content = (
                    '{"translations":['
                    '{"cue_id":0,"text":"一"},'
                    '{"cue_id":1,"text":"二三"},'
                    '{"cue_id":2,"text":"四"}'
                    "]}"
                )
            return {
                "choices": [{"finish_reason": "stop", "message": {"content": content}}]
            }

        with patch.object(translator, "_request", side_effect=response) as request:
            result = translator.plan_and_translate(
                cues,
                SegmentationConfig(model_window_cues=2),
                max_line_units=20,
            )
        self.assertEqual(request.call_count, 4)
        self.assertEqual([cue.text for cue in result.source_cues], ["甲", "乙丙", "丁"])
        self.assertEqual(
            [cue.text for cue in result.translated_cues], ["一", "二三", "四"]
        )

    def test_boundary_reduce_batches_multiple_boundaries_in_one_request(self):
        translator = OpenAICompatibleTranslator(LLMConfig(max_concurrency=1), "secret")
        cues = [Cue(index, index + 1, text) for index, text in enumerate("甲乙丙丁戊己")]
        boundary_prompts: list[str] = []

        def response(body):
            prompt = body["messages"][1]["content"]
            if "Required ID range" in prompt:
                match = next(
                    value
                    for value in ("0-1", "2-3", "4-5")
                    if f"Required ID range: {value}" in prompt
                )
                start, end = (int(value) for value in match.split("-"))
                content = json.dumps(
                    {
                        "cues": [
                            {"start_id": cue_id, "end_id": cue_id}
                            for cue_id in range(start, end + 1)
                        ]
                    }
                )
            elif "Independently replan every BOUNDARY block" in prompt:
                boundary_prompts.append(prompt)
                content = json.dumps(
                    {
                        "repairs": [
                            {
                                "boundary_id": "1|2",
                                "cues": [
                                    {"start_id": 1, "end_id": 1},
                                    {"start_id": 2, "end_id": 2},
                                ],
                            },
                            {
                                "boundary_id": "3|4",
                                "cues": [
                                    {"start_id": 3, "end_id": 3},
                                    {"start_id": 4, "end_id": 4},
                                ],
                            },
                        ]
                    }
                )
            else:
                content = json.dumps(
                    {
                        "translations": [
                            {"cue_id": cue_id, "text": text}
                            for cue_id, text in enumerate("一二三四五六")
                        ]
                    },
                    ensure_ascii=False,
                )
            return {
                "choices": [{"finish_reason": "stop", "message": {"content": content}}]
            }

        with patch.object(translator, "_request", side_effect=response) as request:
            result = translator.plan_and_translate(
                cues,
                SegmentationConfig(model_window_cues=2),
                max_line_units=20,
            )

        self.assertEqual(request.call_count, 5)
        self.assertEqual(len(boundary_prompts), 1)
        self.assertIn("<boundary:1|2 range=1-2>", boundary_prompts[0])
        self.assertIn("<boundary:3|4 range=3-4>", boundary_prompts[0])
        self.assertEqual([cue.text for cue in result.translated_cues], list("一二三四五六"))

    def test_boundary_validation_accepts_independent_partial_results(self):
        cues = [Cue(index, index + 1, text) for index, text in enumerate("甲乙丙丁")]
        specs = [
            (
                "0|1",
                CueTranslationRecord(0, 0, "", "甲"),
                CueTranslationRecord(1, 1, "", "乙"),
            ),
            (
                "2|3",
                CueTranslationRecord(2, 2, "", "丙"),
                CueTranslationRecord(3, 3, "", "丁"),
            ),
        ]

        accepted, rejected = _validated_boundary_repairs(
            [
                {
                    "boundary_id": "0|1",
                    "cues": [{"start_id": 0, "end_id": 1}],
                }
            ],
            specs,
            cues,
        )

        self.assertEqual(list(accepted), ["0|1"])
        self.assertEqual(accepted["0|1"][0].source_text, "甲乙")
        self.assertIn("missing", rejected["2|3"])

    def test_boundary_reduce_retries_only_missing_boundary(self):
        translator = OpenAICompatibleTranslator(LLMConfig(max_retries=3), "secret")
        cues = [Cue(index, index + 1, text) for index, text in enumerate("甲乙丙丁")]
        specs = [
            (
                "0|1",
                CueTranslationRecord(0, 0, "", "甲"),
                CueTranslationRecord(1, 1, "", "乙"),
            ),
            (
                "2|3",
                CueTranslationRecord(2, 2, "", "丙"),
                CueTranslationRecord(3, 3, "", "丁"),
            ),
        ]
        prompts: list[str] = []
        accepted_callbacks: list[list[str]] = []

        def response(body):
            prompt = body["messages"][1]["content"]
            prompts.append(prompt)
            key = "0|1" if len(prompts) == 1 else "2|3"
            start = 0 if key == "0|1" else 2
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "repairs": [
                                        {
                                            "boundary_id": key,
                                            "cues": [
                                                {
                                                    "start_id": start,
                                                    "end_id": start + 1,
                                                }
                                            ],
                                        }
                                    ]
                                }
                            )
                        },
                    }
                ]
            }

        with patch.object(translator, "_request", side_effect=response) as request:
            result = translator._repair_plan_boundaries(
                cues,
                specs,
                {},
                20,
                on_accept=lambda values: accepted_callbacks.append(list(values)),
            )

        self.assertEqual(request.call_count, 2)
        self.assertIn("<boundary:0|1 range=0-1>", prompts[0])
        self.assertIn("<boundary:2|3 range=2-3>", prompts[0])
        self.assertNotIn("<boundary:0|1 range=0-1>", prompts[1])
        self.assertIn("<boundary:2|3 range=2-3>", prompts[1])
        self.assertEqual(set(result), {"0|1", "2|3"})
        self.assertEqual(accepted_callbacks, [["0|1"], ["2|3"]])

    def test_split_planner_retries_content_error_before_translation(self):
        translator = OpenAICompatibleTranslator(LLMConfig(max_retries=3), "secret")
        cues = [Cue(0, 1, "甲")]
        responses = [
            {"choices": [{"finish_reason": "stop", "message": {"content": "{"}}]},
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": (
                                '{"cues":[{"start_id":0,"end_id":0}]}'
                            )
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": ('{"translations":[{"cue_id":0,"text":"甲"}]}')
                        },
                    }
                ]
            },
        ]
        with patch.object(translator, "_request", side_effect=responses) as request:
            result = translator.plan_and_translate(
                cues,
                SegmentationConfig(),
                max_line_units=20,
            )
        self.assertEqual(result.translated_cues[0].text, "甲")
        self.assertEqual(request.call_count, 3)
        retry_prompt = request.call_args_list[1].args[0]["messages"][1]["content"]
        self.assertNotIn("RETRY:", retry_prompt)

    def test_fixed_translation_prompt_uses_safe_width_but_validation_uses_hard_width(
        self,
    ):
        translator = OpenAICompatibleTranslator(LLMConfig(), "secret")
        cues = [Cue(0, 1, "長い文")]

        def response(body):
            prompt = body["messages"][1]["content"]
            if "Group every TARGET" in prompt:
                return {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": (
                                    '{"cues":[{"start_id":0,"end_id":0}]}'
                                )
                            },
                        }
                    ]
                }
            self.assertIn("no wider than 5.000", prompt)
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": (
                                '{"translations":[{"cue_id":0,"text":"一二三四五六"}]}'
                            )
                        },
                    }
                ]
            }

        with patch.object(translator, "_request", side_effect=response):
            result = translator.plan_and_translate(
                cues,
                SegmentationConfig(),
                max_line_units=5,
                hard_max_line_units=10,
            )
        self.assertEqual(result.translated_cues[0].text, "一二三四五六")

    def test_majority_speaker_ignores_unknown_units(self):
        cues = [
            Cue(0, 1, "甲", "ritsu"),
            Cue(1, 2, "乙", None),
            Cue(2, 3, "丙", "yuno"),
            Cue(3, 4, "丁", "ritsu"),
            Cue(4, 5, "戊", None),
        ]

        self.assertEqual(_majority_speaker(cues, 0, 4), "ritsu")
        self.assertEqual(_majority_speaker(cues, 0, 1), "ritsu")
        self.assertIsNone(_majority_speaker(cues, 0, 2))
        self.assertIsNone(_majority_speaker(cues, 1, 1))

    def test_joint_prompt_source_view_removes_unicode_punctuation(self):
        self.assertEqual(
            _without_source_punctuation("「使ってる。ね？」 BanG Dream!"),
            "使ってるね BanG Dream",
        )

    def test_prompt_reference_orders_stable_glossary_before_video_fields(self):
        context = {
            "video": {"title": "changing"},
            "franchises": [{"name": "stable"}],
            "characters": [{"id": "stable-character"}],
            "terms": {"stable-term": "固定术语"},
            "identified_songs": [{"song": "changing-song"}],
        }
        ordered = list(_prompt_translation_context(context))
        self.assertEqual(
            ordered,
            ["franchises", "characters", "terms", "video", "identified_songs"],
        )

    def test_prompt_reference_uses_compact_semantic_sections(self):
        compact = _compact_reference_text(
            {
                "franchises": [{"name": "BanG Dream!", "background": "背景"}],
                "characters": [
                    {
                        "id": "miyako",
                        "source_name": "藤都子",
                        "canonical": "藤都子",
                        "aliases": ["ふじみやこ", "Miyako"],
                        "short_names": [
                            {
                                "source": "みやこ",
                                "target": "都子",
                                "context_only": True,
                            }
                        ],
                    }
                ],
                "terms": {"ゆめみた": "梦限大MewType"},
                "video": {"title": "标题", "categories": ["Entertainment"]},
            }
        )

        self.assertIn("<franchises>\nBanG Dream!｜背景", compact)
        self.assertIn("<characters>\nmiyako｜藤都子=>藤都子", compact)
        self.assertIn("aliases:ふじみやこ｜Miyako", compact)
        self.assertIn("short:みやこ=>都子[context]", compact)
        self.assertIn("<terms>\nゆめみた=>梦限大MewType", compact)
        self.assertIn("<video>\ntitle:标题\ncategories:Entertainment", compact)
        self.assertNotIn('"characters"', compact)

    def test_compact_reference_preserves_trusted_text(self):
        compact = _compact_reference_text(
            {
                "characters": [
                    {
                        "id": "a｜b",
                        "source_name": "A=>B",
                        "canonical": "<C>",
                    }
                ],
                "terms": {"x｜y": "z=>w"},
            }
        )

        self.assertIn("a｜b｜A=>B=><C>", compact)
        self.assertIn("x｜y=>z=>w", compact)

    def test_joint_cache_signature_changes_with_api_provider(self):
        cues = [Cue(0, 1, "字幕")]
        segmentation = SegmentationConfig()
        deepseek = _cue_plan_signature(
            cues,
            segmentation,
            LLMConfig(),
            {},
            20,
        )
        openai = _cue_plan_signature(
            cues,
            segmentation,
            LLMConfig(
                base_url="https://api.openai.com/v1",
                api_style="responses",
                model="gpt-5.6",
                thinking=None,
                reasoning_effort="low",
            ),
            {},
            20,
        )

        self.assertNotEqual(deepseek, openai)

    def test_joint_parser_accepts_equivalent_record_containers(self):
        records = [
            {"start_id": 0, "end_id": 1, "text": "甲"},
            {"start_id": 2, "end_id": 3, "text": "乙"},
        ]
        ndjson = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
        adjacent = ",".join(
            json.dumps(record, ensure_ascii=False) for record in records
        )

        self.assertEqual(_parse_joint_records(json.dumps({"cues": records})), records)
        self.assertEqual(_parse_joint_records(json.dumps(records)), records)
        self.assertEqual(_parse_joint_records(ndjson), records)
        self.assertEqual(_parse_joint_records(adjacent), records)

    def test_joint_parser_rejects_explanatory_text(self):
        with self.assertRaisesRegex(ValueError, "invalid joint cue JSON"):
            _parse_joint_records('Here you go: {"start_id":0,"end_id":0,"text":"甲"}')

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

    def test_openai_responses_request_converts_messages_json_and_tools(self):
        config = LLMConfig(
            base_url="https://api.openai.com/v1",
            api_style="responses",
            model="gpt-5.6",
            reasoning_effort="low",
            thinking=None,
        )
        url, body = _prepare_api_request(
            config,
            {
                "model": "gpt-5.6",
                "max_tokens": 1234,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": "固定规则"},
                    {"role": "user", "content": "判断歌曲"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search_web",
                                    "arguments": '{"query":"歌名"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": "搜索结果",
                    },
                ],
                "response_format": {"type": "json_object"},
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "description": "search",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "tool_choice": "auto",
            },
        )

        self.assertEqual(url, "https://api.openai.com/v1/responses")
        self.assertEqual(body["instructions"], "固定规则")
        self.assertEqual(body["max_output_tokens"], 1234)
        self.assertEqual(body["text"], {"format": {"type": "json_object"}})
        self.assertEqual(body["reasoning"], {"effort": "low"})
        self.assertFalse(body["store"])
        self.assertNotIn("temperature", body)
        self.assertEqual(body["tools"][0]["name"], "search_web")
        self.assertEqual(body["input"][1]["type"], "function_call")
        self.assertEqual(body["input"][2]["type"], "function_call_output")

    def test_openai_responses_output_normalizes_text_tools_and_usage(self):
        config = LLMConfig(api_style="responses", thinking=None)
        normalized = _normalize_api_response(
            config,
            {
                "id": "resp_123",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": '{"title":"标题"}'}
                        ],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "search_web",
                        "arguments": '{"query":"歌名"}',
                    },
                ],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                    "input_tokens_details": {"cached_tokens": 80},
                },
            },
        )

        choice = normalized["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(choice["message"]["content"], '{"title":"标题"}')
        self.assertEqual(
            choice["message"]["tool_calls"][0]["function"]["name"],
            "search_web",
        )
        self.assertEqual(normalized["usage"]["prompt_tokens"], 100)
        self.assertEqual(
            normalized["usage"]["prompt_tokens_details"]["cached_tokens"], 80
        )

    def test_openai_incomplete_response_maps_token_limit_to_length(self):
        normalized = _normalize_api_response(
            LLMConfig(api_style="responses", thinking=None),
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [],
            },
        )
        self.assertEqual(normalized["choices"][0]["finish_reason"], "length")

    def test_window_ranges_never_leave_one_unit_chunk(self):
        self.assertEqual(_translation_window_ranges(5, 2), [(0, 2), (2, 5)])
        self.assertEqual(_translation_window_ranges(601, 600), [(0, 599), (599, 601)])

    def test_fixed_cues_split_planner_ranges_and_bypass_planning(self):
        cues = [
            Cue(0.0, 0.4, "話", kind="speech"),
            Cue(0.4, 0.8, "す", kind="speech"),
            Cue(0.8, 1.8, "歌詞", kind="singing"),
            Cue(1.8, 2.2, "重複話者", kind="conditioned_speech"),
            Cue(2.2, 2.6, "続", kind="speech"),
            Cue(2.6, 3.0, "き", kind="speech"),
        ]

        self.assertEqual(_cue_plan_range_groups(cues, 1), [[(0, 2)], [(4, 6)]])
        records = _merge_planned_and_fixed_records(
            cues,
            [
                CueTranslationRecord(0, 1, "", "話す"),
                CueTranslationRecord(4, 5, "", "続き"),
            ],
        )
        self.assertEqual(
            [
                (record.start_id, record.end_id, record.source_text)
                for record in records
            ],
            [
                (0, 1, "話す"),
                (2, 2, "歌詞"),
                (3, 3, "重複話者"),
                (4, 5, "続き"),
            ],
        )

    def test_planner_units_use_compact_speaker_and_gap_markers(self):
        units = _compact_prompt_units_text(
            [
                Cue(0.0, 0.5, "こんにちは。", "speaker_00", "speech"),
                Cue(0.7, 1.0, "まだ<確認>", "speaker_00", "speech"),
                Cue(1.6, 2.0, "はい", "speaker_01", "speech"),
                Cue(2.1, 2.4, "次", None, "speech"),
            ],
            0,
            4,
        )

        self.assertEqual(
            units,
            "\n".join(
                [
                    "<speaker_00>",
                    "<0>こんにちは <1>まだ＜確認＞",
                    "<gap:600ms>",
                    "<speaker_01>",
                    "<2>はい",
                    "<unknown>",
                    "<3>次",
                ]
            ),
        )

    def test_fixed_translation_uses_compact_speaker_and_cue_markers(self):
        cues = [
            Cue(index, index + 1, "unused", "A", "speech")
            for index in range(13)
        ]
        records = [
            CueTranslationRecord(
                index,
                index,
                "",
                (
                    "いやそれは違うと思うけど"
                    if index == 11
                    else "昨日の話じゃなくてその前のやつ"
                    if index == 12
                    else "unused"
                ),
            )
            for index in range(13)
        ]

        self.assertEqual(
            _compact_fixed_translation_text(cues, records, [11, 12]),
            "\n".join(
                [
                    "<A>",
                    "<11>いやそれは違うと思うけど",
                    "<12>昨日の話じゃなくてその前のやつ",
                ]
            ),
        )

    def test_planner_ignores_overlap_evidence(self):
        cues = [Cue(216.0, 222.0, "対象", "minetsuki_ritsu", "speech")]
        context = {
            "asr_evidence": [
                {
                    "kind": "overlap_reconciliation",
                    "start": 216.427,
                    "end": 221.068,
                    "qwen_mixed": [
                        {
                            "start": 216.0,
                            "end": 222.0,
                            "text": "一で始めますね、行きますよ。はい三",
                            "units": [{"start": 216.0, "end": 216.2, "text": "一"}],
                        }
                    ],
                    "dicow": [
                        {
                            "start": 216.5,
                            "end": 217.5,
                            "text": "で始めますね",
                            "speaker": "minetsuki_ritsu",
                            "kind": "speech",
                        },
                        {
                            "start": 218.0,
                            "end": 218.5,
                            "text": "right",
                            "speaker": "sengoku_yuno",
                            "kind": "speech",
                        },
                        {
                            "start": 219.0,
                            "end": 221.0,
                            "text": "いきますよ。3",
                            "speaker": "minetsuki_ritsu",
                            "kind": "speech",
                        },
                    ],
                    "turns": [{"start": 216.0, "end": 221.0, "speaker": "SPEAKER_00"}],
                }
            ]
        }

        prompt = _cue_plan_prompt(
            cues,
            0,
            1,
            context,
            20,
            "Simplified Chinese",
            previous_error=None,
        )
        self.assertIn("TARGET:\n<minetsuki_ritsu>\n<0>対象", prompt)
        self.assertNotIn("<overlap>", prompt)
        self.assertNotIn("一で始めますね", prompt)
    def test_only_transient_transport_failures_receive_backoff(self):
        with patch(
            "subtitle_pipeline.translate.random.uniform",
            side_effect=[4.5, 2.25, 1.75, 3.5],
        ) as uniform:
            self.assertEqual(
                _transient_retry_delay(urllib.error.URLError("network"), 3), 4.5
            )
            self.assertEqual(_transient_retry_delay(TimeoutError("timeout"), 2), 2.25)
            self.assertEqual(
                _transient_retry_delay(LLMHTTPError(429, "limited"), 2), 1.75
            )
            self.assertEqual(_transient_retry_delay(LLMHTTPError(503, "busy"), 3), 3.5)
        self.assertEqual(
            uniform.call_args_list,
            [call(3.0, 5.0), call(1.5, 2.5), call(1.5, 2.5), call(3.0, 5.0)],
        )
        self.assertIsNone(_transient_retry_delay(LLMHTTPError(400, "bad"), 3))
        self.assertIsNone(_transient_retry_delay(TranslationError("invalid"), 3))
        self.assertIsNone(_transient_retry_delay(ValueError("invalid JSON"), 3))
        self.assertTrue(_is_nontransient_http_error(LLMHTTPError(400, "bad")))
        self.assertTrue(_is_nontransient_http_error(LLMHTTPError(401, "auth")))
        self.assertTrue(_is_nontransient_http_error(LLMHTTPError(403, "denied")))
        self.assertFalse(_is_nontransient_http_error(LLMHTTPError(429, "limited")))
        self.assertFalse(_is_nontransient_http_error(LLMHTTPError(503, "busy")))

    def test_retry_after_overrides_429_backoff(self):
        error = LLMHTTPError(429, "limited", retry_after_seconds=12.5)
        with patch(
            "subtitle_pipeline.translate.random.uniform", return_value=1.25
        ) as uniform:
            self.assertEqual(_transient_retry_delay(error, 4), 13.75)
        uniform.assert_called_once_with(0.0, 2.0)
        self.assertEqual(_parse_retry_after("7"), 7)
        self.assertEqual(_parse_retry_after("0"), 0)
        self.assertEqual(
            _parse_retry_after(
                "Fri, 14 Aug 2026 00:00:10 GMT",
                now=datetime(2026, 8, 14, tzinfo=timezone.utc),
            ),
            10,
        )
        self.assertIsNone(_parse_retry_after("not-a-delay"))

    def test_429_exhaustion_stops_without_window_shrink(self):
        translator = OpenAICompatibleTranslator(LLMConfig(max_retries=3), "secret")
        cues = [Cue(index, index + 1, str(index)) for index in range(4)]
        error = LLMHTTPError(429, "limited", retry_after_seconds=6)
        with (
            patch.object(
                translator,
                "_request",
                side_effect=error,
            ) as request,
            patch("subtitle_pipeline.translate.random.uniform", return_value=0),
            patch("subtitle_pipeline.translate.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(LLMHTTPError, "HTTP 429"):
                translator.plan_and_translate(
                    cues,
                    SegmentationConfig(model_window_cues=4),
                    max_line_units=20,
                )
        self.assertEqual(request.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [6, 6])

    def test_5xx_exhaustion_stops_without_window_shrink(self):
        translator = OpenAICompatibleTranslator(LLMConfig(max_retries=3), "secret")
        cues = [Cue(index, index + 1, str(index)) for index in range(4)]
        error = LLMHTTPError(503, "server overloaded")
        with (
            patch.object(
                translator,
                "_request",
                side_effect=error,
            ) as request,
            patch("subtitle_pipeline.translate.random.uniform", side_effect=[1, 2]),
            patch("subtitle_pipeline.translate.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(LLMHTTPError, "HTTP 503"):
                translator.plan_and_translate(
                    cues,
                    SegmentationConfig(model_window_cues=4),
                    max_line_units=20,
                )
        self.assertEqual(request.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])

    def test_nontransient_http_error_stops_without_retry_or_window_shrink(self):
        translator = OpenAICompatibleTranslator(LLMConfig(max_retries=5), "secret")
        cues = [Cue(index, index + 1, str(index)) for index in range(4)]
        with (
            patch.object(
                translator,
                "_request",
                side_effect=LLMHTTPError(401, "invalid API key"),
            ) as request,
            patch("subtitle_pipeline.translate.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(LLMHTTPError, "HTTP 401"):
                translator.plan_and_translate(
                    cues,
                    SegmentationConfig(model_window_cues=4),
                    max_line_units=20,
                )
        self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()

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
        pending_empty = _validate_joint_records(
            [{"start_id": 0, "end_id": 0, "text": " "}], 0, 1, 10
        )
        with self.assertRaisesRegex(TranslationError, "empty"):
            _validate_joint_target_language(pending_empty, "简体中文")
        with self.assertRaisesRegex(TranslationError, "not one-line"):
            _validate_joint_records(
                [{"start_id": 0, "end_id": 0, "text": "一二三"}],
                0,
                1,
                2,
            )
        nonfinal = _validate_joint_records(
            [
                {"start_id": 0, "end_id": 0, "text": "一"},
                {"start_id": 1, "end_id": 1, "text": "一二三"},
            ],
            0,
            2,
            2,
            skip_last_width=True,
        )
        self.assertEqual([record.end_id for record in nonfinal], [0, 1])
        with self.assertRaisesRegex(TranslationError, "record 0 is not one-line"):
            _validate_joint_records(
                [
                    {"start_id": 0, "end_id": 0, "text": "一二三"},
                    {"start_id": 1, "end_id": 1, "text": "一"},
                ],
                0,
                2,
                2,
                skip_last_width=True,
            )
        with self.assertRaisesRegex(TranslationError, "Japanese kana"):
            _validate_joint_target_language(
                [CueTranslationRecord(0, 0, "実存でもない")], "简体中文"
            )
    def test_ssl_context_combines_platform_and_certifi_ca(self):
        with (
            patch("subtitle_pipeline.translate.ssl.create_default_context") as create,
            patch(
                "subtitle_pipeline.translate.certifi.where",
                return_value="/ca/certifi.pem",
            ),
        ):
            translator = OpenAICompatibleTranslator(LLMConfig(), "secret")
        self.assertIs(translator.ssl_context, create.return_value)
        create.return_value.load_verify_locations.assert_called_once_with(
            cafile="/ca/certifi.pem"
        )

    def test_parses_fenced_json_from_less_strict_provider(self):
        parsed = _parse_json_object('```json\n{"translations": []}\n```')
        self.assertEqual(parsed, {"translations": []})

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
