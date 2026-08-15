import json
import tempfile
import threading
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
    _collapsed_start_runs,
    _is_nontransient_http_error,
    _joint_translation_signature,
    _load_joint_translation_cache,
    _log_response_usage,
    _parse_joint_records,
    _parse_json_object,
    _parse_retry_after,
    _prompt_translation_context,
    _transient_retry_delay,
    _translation_window_ranges,
    _validate_joint_records,
    _validate_joint_target_language,
    _validate_joint_timing,
    _without_source_punctuation,
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
        self.assertLess(prompt.index("REFERENCE:"), prompt.index("DISPLAY_CONSTRAINT:"))
        self.assertLess(
            prompt.index("DISPLAY_CONSTRAINT:"), prompt.index("Required ID range:")
        )
        self.assertIn("20 full-width characters", prompt)
        self.assertIn("REFERENCE.characters contains entity instances", prompt)
        self.assertIn("context_only=true", prompt)
        self.assertIn("Return exactly one JSON object with a cues array", prompt)
        self.assertIn("Never omit, genericize, or paraphrase a source name", prompt)
        self.assertIn("that record's text must explicitly contain the mapped target name", prompt)
        self.assertIn("Usually omit さん", prompt)
        self.assertIn("only in a formal context", prompt)
        self.assertIn("Translate ちゃん as 酱", prompt)
        self.assertIn("Usually omit くん", prompt)
        self.assertIn("Translate さま or 様 as 大人", prompt)
        self.assertIn("Translate 先生 as 老师, 医生, or 先生", prompt)
        self.assertIn("complete name-plus-honorific form overrides", prompt)
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

    def test_joint_parser_accepts_equivalent_record_containers(self):
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

    def test_parallel_windows_then_repair_only_two_edge_cues(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(max_concurrency=2), "secret"
        )
        cues = [Cue(index, index + 1, text) for index, text in enumerate("甲乙丙丁戊")]

        def response(body):
            prompt = body["messages"][1]["content"]
            system = body["messages"][0]["content"]
            if "repair one provisional" in system:
                self.assertIn("IDs 1-3", prompt)
                self.assertIn('"start_id":1', prompt)
                self.assertIn('"start_id":3', prompt)
                return {"choices": [{"message": {"content": (
                    '{"cues":['
                    '{"start_id":1,"end_id":3,"text":"重新分组"}'
                    "]}"
                )}}]}
            if "Required ID range: 0-2" in prompt:
                content = (
                    '{"cues":['
                    '{"start_id":0,"end_id":0,"text":"一"},'
                    '{"start_id":1,"end_id":2,"text":"暂定左边"}'
                    "]}"
                )
            else:
                self.assertIn("Required ID range: 3-4", prompt)
                content = (
                    '{"cues":['
                    '{"start_id":3,"end_id":3,"text":"暂定右边"},'
                    '{"start_id":4,"end_id":4,"text":"五"}'
                    "]}"
                )
            return {"choices": [{"message": {"content": content}}]}

        config = SegmentationConfig(model_window_cues=3)
        with patch.object(translator, "_request", side_effect=response) as request:
            result = translator.plan_and_translate(
                cues,
                config,
                max_line_units=20,
            )
        self.assertEqual(
            [cue.text for cue in result.translated_cues], ["一", "重新分组", "五"]
        )
        self.assertEqual(request.call_count, 3)
        boundary_calls = [
            call
            for call in request.call_args_list
            if "repair one provisional"
            in call.args[0]["messages"][0]["content"]
        ]
        self.assertEqual(len(boundary_calls), 1)

    def test_boundary_repair_accepts_translation_field_alias(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(max_retries=1, max_concurrency=1), "secret"
        )
        cues = [Cue(index, index + 1, text) for index, text in enumerate("甲乙丙丁")]

        def response(body):
            prompt = body["messages"][1]["content"]
            if "WRITABLE:" in prompt:
                self.assertIn('"text":"中文字幕"', prompt)
                content = (
                    '{"cues":['
                    '{"start_id":1,"end_id":1,"translation":"一"},'
                    '{"start_id":2,"end_id":2,"translation":"二"}'
                    "]}"
                )
            elif "Required ID range: 0-1" in prompt:
                content = (
                    '{"cues":['
                    '{"start_id":0,"end_id":0,"text":"零"},'
                    '{"start_id":1,"end_id":1,"text":"一"}'
                    "]}"
                )
            else:
                content = (
                    '{"cues":['
                    '{"start_id":2,"end_id":2,"text":"二"},'
                    '{"start_id":3,"end_id":3,"text":"三"}'
                    "]}"
                )
            return {"choices": [{"message": {"content": content}}]}

        with patch.object(translator, "_request", side_effect=response):
            result = translator.plan_and_translate(
                cues,
                SegmentationConfig(model_window_cues=2),
                max_line_units=20,
            )
        self.assertEqual(
            [cue.text for cue in result.translated_cues], ["零", "一", "二", "三"]
        )

    def test_failed_map_window_shrinks_and_repairs_new_boundary(self):
        translator = OpenAICompatibleTranslator(LLMConfig(max_retries=1), "secret")
        cues = [Cue(index, index + 1, text) for index, text in enumerate("甲乙丙丁")]

        def plan(_cues, start, end, *_args):
            if (start, end) == (0, 4):
                raise TranslationError("finish_reason=length")
            return [
                CueTranslationRecord(index, index, str(index))
                for index in range(start, end)
            ]

        with patch.object(
            translator, "_plan_and_translate_window", side_effect=plan
        ) as planned, patch.object(
            translator,
            "_repair_translation_boundary",
            return_value=[CueTranslationRecord(1, 2, "一二")],
        ) as repaired:
            result = translator._plan_and_translate_window_resilient(
                cues, 0, 4, {}, [], 20, 40
            )

        self.assertEqual(
            result,
            [
                CueTranslationRecord(0, 0, "0"),
                CueTranslationRecord(1, 2, "一二"),
                CueTranslationRecord(3, 3, "3"),
            ],
        )
        self.assertEqual(planned.call_count, 3)
        repaired.assert_called_once()

    def test_length_output_retries_once_before_window_shrink(self):
        translator = OpenAICompatibleTranslator(LLMConfig(max_retries=5), "secret")
        response = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": '{"text":"' + "我，" * 30},
                }
            ]
        }

        with patch.object(translator, "_request", return_value=response) as request:
            with self.assertRaisesRegex(TranslationError, "failed after 2 attempts"):
                translator._plan_and_translate_window(
                    [Cue(0, 1, "私"), Cue(1, 2, "です")],
                    0,
                    2,
                    {},
                    [],
                    20,
                    40,
                )

        self.assertEqual(request.call_count, 2)

    def test_independent_windows_execute_concurrently(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(max_retries=1, max_concurrency=2), "secret"
        )
        cues = [Cue(index, index + 1, str(index)) for index in range(4)]
        barrier = threading.Barrier(2)
        lock = threading.Lock()
        active = 0
        maximum_active = 0

        def response(body):
            nonlocal active, maximum_active
            prompt = body["messages"][1]["content"]
            if "WRITABLE:" in prompt:
                content = (
                    '{"cues":['
                    '{"start_id":1,"end_id":1,"text":"一"},'
                    '{"start_id":2,"end_id":2,"text":"二"}'
                    "]}"
                )
                return {"choices": [{"message": {"content": content}}]}
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            barrier.wait(timeout=2)
            if "Required ID range: 0-1" in prompt:
                content = (
                    '{"cues":['
                    '{"start_id":0,"end_id":0,"text":"零"},'
                    '{"start_id":1,"end_id":1,"text":"一"}'
                    "]}"
                )
            else:
                content = (
                    '{"cues":['
                    '{"start_id":2,"end_id":2,"text":"二"},'
                    '{"start_id":3,"end_id":3,"text":"三"}'
                    "]}"
                )
            with lock:
                active -= 1
            return {"choices": [{"message": {"content": content}}]}

        with patch.object(translator, "_request", side_effect=response):
            result = translator.plan_and_translate(
                cues,
                SegmentationConfig(model_window_cues=2),
                max_line_units=20,
            )
        self.assertEqual(maximum_active, 2)
        self.assertEqual(
            [cue.text for cue in result.translated_cues], ["零", "一", "二", "三"]
        )

    def test_window_ranges_never_leave_one_unit_chunk(self):
        self.assertEqual(_translation_window_ranges(5, 2), [(0, 2), (2, 5)])
        self.assertEqual(
            _translation_window_ranges(601, 600), [(0, 599), (599, 601)]
        )

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

    def test_single_cue_window_retries_same_range_immediately(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(max_retries=2, max_concurrency=1), "secret"
        )
        cues = [Cue(index, index + 1, text) for index, text in enumerate("甲乙丙丁")]
        responses = [
            {"choices": [{"message": {"content": (
                '{"start_id":0,"end_id":1,"text":"未完成"}'
            )}}]},
            {"choices": [{"message": {"content": (
                '{"start_id":0,"end_id":0,"text":"一"}\n'
                '{"start_id":1,"end_id":1,"text":"二"}'
            )}}]},
            {"choices": [{"message": {"content": (
                '{"start_id":2,"end_id":2,"text":"三"}\n'
                '{"start_id":3,"end_id":3,"text":"四"}'
            )}}]},
            {"choices": [{"message": {"content": (
                '{"start_id":1,"end_id":1,"text":"二"}\n'
                '{"start_id":2,"end_id":2,"text":"三"}'
            )}}]},
        ]
        with patch.object(translator, "_request", side_effect=responses) as request:
            result = translator.plan_and_translate(
                cues,
                SegmentationConfig(model_window_cues=2),
                max_line_units=20,
            )
        self.assertEqual(
            [cue.text for cue in result.translated_cues], ["一", "二", "三", "四"]
        )
        prompts = [
            call.args[0]["messages"][1]["content"]
            for call in request.call_args_list
        ]
        self.assertIn("Required ID range: 0-1", prompts[0])
        self.assertIn("Required ID range: 0-1", prompts[1])
        self.assertIn("RETRY:", prompts[1])
        self.assertIn("only one cue", prompts[1])
        self.assertNotIn("Required ID range: 0-3", "\n".join(prompts))

    def test_collapsed_timing_runs_are_compact(self):
        cues = [
            Cue(0, 1, "甲"),
            Cue(1, 1.2, "乙"),
            Cue(1, 1.2, "丙"),
            Cue(1, 2, "丁"),
        ]
        runs = _collapsed_start_runs(cues, 0, 4)
        self.assertEqual(runs, [[1, 3]])

    def test_failed_joint_range_retries_without_shrinking(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(max_retries=2), "secret"
        )
        cues = [Cue(index, index + 1, text) for index, text in enumerate("甲乙丙丁")]
        responses = [
            {"choices": [{"message": {"content": "not json"}}]},
            {"choices": [{"message": {"content": (
                '{"start_id":0,"end_id":1,"text":"前句"}\n'
                '{"start_id":2,"end_id":3,"text":"后句"}'
            )}}]},
        ]
        with patch.object(translator, "_request", side_effect=responses) as request:
            result = translator.plan_and_translate(
                cues,
                SegmentationConfig(model_window_cues=4),
                max_line_units=20,
            )
        self.assertEqual([cue.text for cue in result.translated_cues], ["前句", "后句"])
        prompts = [call.args[0]["messages"][1]["content"] for call in request.call_args_list]
        self.assertIn("Required ID range: 0-3", prompts[0])
        self.assertIn("Required ID range: 0-3", prompts[1])
        self.assertNotIn("Required ID range: 0-1", prompts[1])

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
        ) as sleep:
            result = translator.plan_and_translate(
                cues,
                SegmentationConfig(),
                max_line_units=20,
            )
        self.assertEqual([cue.text for cue in result.translated_cues], ["一"])
        first_prompt = request.call_args_list[0].args[0]["messages"][1]["content"]
        retry_prompt = request.call_args_list[1].args[0]["messages"][1]["content"]
        self.assertEqual(first_prompt, retry_prompt)
        self.assertNotIn("RETRY:", retry_prompt)
        self.assertIn("Required ID range: 0-0", retry_prompt)
        sleep.assert_not_called()

    def test_joint_json_error_is_retried_without_prompt_feedback(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(max_retries=2), "secret"
        )
        cues = [Cue(0, 1, "甲")]
        responses = [
            {"choices": [{"finish_reason": "stop", "message": {"content": "{"}}]},
            {"choices": [{"finish_reason": "stop", "message": {"content": (
                '{"start_id":0,"end_id":0,"text":"一"}'
            )}}]},
        ]
        with patch.object(translator, "_request", side_effect=responses) as request:
            result = translator.plan_and_translate(
                cues,
                SegmentationConfig(),
                max_line_units=20,
            )
        self.assertEqual([cue.text for cue in result.translated_cues], ["一"])
        first_prompt = request.call_args_list[0].args[0]["messages"][1]["content"]
        second_prompt = request.call_args_list[1].args[0]["messages"][1]["content"]
        self.assertEqual(first_prompt, second_prompt)
        self.assertNotIn("RETRY:", second_prompt)

    def test_joint_id_error_is_retried_with_prompt_feedback(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(max_retries=2), "secret"
        )
        cues = [Cue(0, 1, "甲"), Cue(1, 2, "乙")]
        responses = [
            {"choices": [{"finish_reason": "stop", "message": {"content": (
                '{"start_id":0,"end_id":0,"text":"一"}'
            )}}]},
            {"choices": [{"finish_reason": "stop", "message": {"content": (
                '{"start_id":0,"end_id":0,"text":"一"}\n'
                '{"start_id":1,"end_id":1,"text":"二"}'
            )}}]},
        ]
        with patch.object(translator, "_request", side_effect=responses) as request:
            result = translator.plan_and_translate(
                cues,
                SegmentationConfig(),
                max_line_units=20,
            )
        self.assertEqual([cue.text for cue in result.translated_cues], ["一", "二"])
        retry_prompt = request.call_args_list[1].args[0]["messages"][1]["content"]
        self.assertIn("RETRY:", retry_prompt)
        self.assertIn("missing IDs 1-1", retry_prompt)

    def test_joint_network_failure_uses_exponential_backoff(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(max_retries=2), "secret"
        )
        cues = [Cue(0, 1, "甲")]
        response = {
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": '{"start_id":0,"end_id":0,"text":"一"}'
                },
            }]
        }
        with patch.object(
            translator,
            "_request",
            side_effect=[urllib.error.URLError("temporary network failure"), response],
        ) as request, patch(
            "subtitle_pipeline.translate.random.uniform", return_value=1
        ), patch("subtitle_pipeline.translate.time.sleep") as sleep:
            translator.plan_and_translate(
                cues,
                SegmentationConfig(),
                max_line_units=20,
            )
        sleep.assert_called_once_with(1)
        first_prompt = request.call_args_list[0].args[0]["messages"][1]["content"]
        second_prompt = request.call_args_list[1].args[0]["messages"][1]["content"]
        self.assertEqual(first_prompt, second_prompt)
        self.assertNotIn("RETRY:", second_prompt)

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
        translator = OpenAICompatibleTranslator(
            LLMConfig(max_retries=3), "secret"
        )
        cues = [Cue(index, index + 1, str(index)) for index in range(4)]
        error = LLMHTTPError(429, "limited", retry_after_seconds=6)
        with patch.object(
            translator,
            "_request",
            side_effect=error,
        ) as request, patch(
            "subtitle_pipeline.translate.random.uniform", return_value=0
        ), patch("subtitle_pipeline.translate.time.sleep") as sleep:
            with self.assertRaisesRegex(LLMHTTPError, "HTTP 429"):
                translator.plan_and_translate(
                    cues,
                    SegmentationConfig(model_window_cues=4),
                    max_line_units=20,
                )
        self.assertEqual(request.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [6, 6])

    def test_5xx_exhaustion_stops_without_window_shrink(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(max_retries=3), "secret"
        )
        cues = [Cue(index, index + 1, str(index)) for index in range(4)]
        error = LLMHTTPError(503, "server overloaded")
        with patch.object(
            translator,
            "_request",
            side_effect=error,
        ) as request, patch(
            "subtitle_pipeline.translate.random.uniform", side_effect=[1, 2]
        ), patch("subtitle_pipeline.translate.time.sleep") as sleep:
            with self.assertRaisesRegex(LLMHTTPError, "HTTP 503"):
                translator.plan_and_translate(
                    cues,
                    SegmentationConfig(model_window_cues=4),
                    max_line_units=20,
                )
        self.assertEqual(request.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])

    def test_nontransient_http_error_stops_without_retry_or_window_shrink(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(max_retries=5), "secret"
        )
        cues = [Cue(index, index + 1, str(index)) for index in range(4)]
        with patch.object(
            translator,
            "_request",
            side_effect=LLMHTTPError(401, "invalid API key"),
        ) as request, patch("subtitle_pipeline.translate.time.sleep") as sleep:
            with self.assertRaisesRegex(LLMHTTPError, "HTTP 401"):
                translator.plan_and_translate(
                    cues,
                    SegmentationConfig(model_window_cues=4),
                    max_line_units=20,
                )
        self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()

    def test_text_invalid_cues_are_repaired_together_after_planning(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(max_retries=2), "secret"
        )
        cues = [Cue(index, index + 1, text) for index, text in enumerate("甲乙丙")]
        responses = [
            {"choices": [{"finish_reason": "stop", "message": {"content": (
                '{"cues":['
                '{"start_id":0,"end_id":0,"text":"まだ"},'
                '{"start_id":1,"end_id":1,"text":"正确"},'
                '{"start_id":2,"end_id":2,"text":"テスト"}'
                "]}"
            )}}]},
            {"choices": [{"finish_reason": "stop", "message": {"content": (
                '{"repairs":['
                '{"repair_id":0,"text":"还没有"},'
                '{"repair_id":2,"text":"测试"}'
                "]}"
            )}}]},
        ]
        with patch.object(translator, "_request", side_effect=responses) as request:
            result = translator.plan_and_translate(
                cues,
                SegmentationConfig(),
                max_line_units=20,
            )
        self.assertEqual(
            [cue.text for cue in result.translated_cues],
            ["还没有", "正确", "测试"],
        )
        self.assertEqual(request.call_count, 2)
        repair_prompt = request.call_args_list[1].args[0]["messages"][1]["content"]
        repair_target = repair_prompt.split("TARGET: ", 1)[1]
        self.assertIn('"repair_id":0', repair_prompt)
        self.assertIn('"repair_id":2', repair_prompt)
        self.assertNotIn('"repair_id":1', repair_target)
        self.assertIn("never change, merge, split", repair_prompt)
        self.assertLess(
            repair_prompt.index("REFERENCE:"),
            repair_prompt.index("DISPLAY_CONSTRAINT:"),
        )

    def test_partial_repairs_are_cached_and_only_missing_ids_are_retried(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(max_retries=2), "secret"
        )
        cues = [Cue(index, index + 1, text) for index, text in enumerate("甲乙")]
        responses = [
            {"choices": [{"finish_reason": "stop", "message": {"content": (
                '{"cues":['
                '{"start_id":0,"end_id":0,"text":"まだ"},'
                '{"start_id":1,"end_id":1,"text":"テスト"}'
                "]}"
            )}}]},
            {"choices": [{"finish_reason": "stop", "message": {"content": (
                '{"repairs":[{"repair_id":0,"text":"还没有"}]}'
            )}}]},
            {"choices": [{"finish_reason": "stop", "message": {"content": (
                '{"repairs":[{"repair_id":1,"text":"测试"}]}'
            )}}]},
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cue-translation-cache.json"
            with patch.object(translator, "_request", side_effect=responses) as request:
                result = translator.plan_and_translate(
                    cues,
                    SegmentationConfig(),
                    max_line_units=20,
                    cache_path=path,
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual([cue.text for cue in result.translated_cues], ["还没有", "测试"])
        second_repair = request.call_args_list[2].args[0]["messages"][1]["content"]
        self.assertNotIn('"repair_id":0', second_repair)
        self.assertIn('"repair_id":1', second_repair)
        self.assertEqual(
            [record["status"] for record in payload["records"]],
            ["confirmed", "confirmed"],
        )

    def test_unrepaired_text_error_blocks_completion(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(max_retries=1), "secret"
        )
        cues = [Cue(0, 1, "甲")]
        responses = [
            {"choices": [{"finish_reason": "stop", "message": {
                "content": '{"cues":[{"start_id":0,"end_id":0,"text":"まだ"}]}'
            }}]},
            {"choices": [{"finish_reason": "stop", "message": {
                "content": '{"repairs":[{"repair_id":0,"text":"まだ"}]}'
            }}]},
        ]
        with patch.object(translator, "_request", side_effect=responses):
            with self.assertRaisesRegex(TranslationError, "repair"):
                translator.plan_and_translate(
                    cues,
                    SegmentationConfig(),
                    max_line_units=20,
                )

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
        with self.assertRaisesRegex(TranslationError, "non-positive"):
            _validate_joint_timing(
                [CueTranslationRecord(0, 0, "字幕")],
                [Cue(1, 2, "甲"), Cue(1, 3, "乙")],
            )

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

    def test_parallel_cache_resumes_only_missing_windows_then_boundaries(self):
        config = SegmentationConfig(model_window_cues=2)
        llm = LLMConfig(max_retries=1, max_concurrency=1)
        cues = [Cue(index, index + 1, text) for index, text in enumerate("甲乙丙丁")]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cue-translation-cache.json"
            first = OpenAICompatibleTranslator(llm, "secret")
            first_responses = [
                {"choices": [{"message": {"content": (
                    '{"start_id":0,"end_id":0,"text":"一"}\n'
                    '{"start_id":1,"end_id":1,"text":"二"}'
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
            self.assertNotIn("records", payload)
            self.assertEqual(set(payload["windows"]), {"0:2"})

            second = OpenAICompatibleTranslator(llm, "secret")
            responses = [
                {"choices": [{"message": {"content": (
                    '{"start_id":2,"end_id":2,"text":"三"}\n'
                    '{"start_id":3,"end_id":3,"text":"四"}'
                )}}]},
                {"choices": [{"message": {"content": (
                    '{"start_id":1,"end_id":1,"text":"二"}\n'
                    '{"start_id":2,"end_id":2,"text":"三"}'
                )}}]},
            ]
            with patch.object(second, "_request", side_effect=responses) as request:
                result = second.plan_and_translate(
                    cues,
                    config,
                    max_line_units=20,
                    cache_path=path,
                )
            final_payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            [cue.text for cue in result.translated_cues], ["一", "二", "三", "四"]
        )
        self.assertEqual(request.call_count, 2)
        prompts = [call.args[0]["messages"][1]["content"] for call in request.call_args_list]
        self.assertIn("Required ID range: 2-3", prompts[0])
        self.assertNotIn("Required ID range: 0-1", "\n".join(prompts))
        self.assertEqual(set(final_payload["windows"]), {"0:2", "2:4"})
        self.assertEqual(set(final_payload["boundaries"]), {"1|2"})

    def test_complete_cached_plan_resumes_at_pending_repairs_only(self):
        config = SegmentationConfig()
        llm = LLMConfig(max_retries=1)
        cues = [Cue(0, 1, "甲"), Cue(1, 2, "乙")]
        signature = _joint_translation_signature(cues, config, llm, {}, 20)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cue-translation-cache.json"
            path.write_text(
                json.dumps(
                    {
                        "signature": signature,
                        "records": [
                            {
                                "start_id": 0,
                                "end_id": 0,
                                "text": "正确",
                                "status": "confirmed",
                                "errors": [],
                            },
                            {
                                "start_id": 1,
                                "end_id": 1,
                                "text": "まだ",
                                "status": "pending",
                                "errors": ["contains Japanese kana"],
                            },
                        ],
                        "next_window_end": 2,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            translator = OpenAICompatibleTranslator(llm, "secret")
            response = {"choices": [{"finish_reason": "stop", "message": {
                "content": '{"repairs":[{"repair_id":1,"text":"还没有"}]}'
            }}]}
            with patch.object(translator, "_request", return_value=response) as request:
                result = translator.plan_and_translate(
                    cues,
                    config,
                    max_line_units=20,
                    cache_path=path,
                )
        self.assertEqual([cue.text for cue in result.translated_cues], ["正确", "还没有"])
        body = request.call_args.args[0]
        self.assertIn("repair specific translated subtitle", body["messages"][0]["content"])
        self.assertNotIn("Group every TARGET", body["messages"][1]["content"])

    def test_ssl_context_combines_platform_and_certifi_ca(self):
        with patch("subtitle_pipeline.translate.ssl.create_default_context") as create, patch(
            "subtitle_pipeline.translate.certifi.where", return_value="/ca/certifi.pem"
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
