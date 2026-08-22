import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from subtitle_pipeline.config import LLMConfig, SegmentationConfig
from subtitle_pipeline.joint_translation import (
    CoverageValidationError,
    _dialogue_context,
    _prompt,
    _reference_replacements,
    _request_resilient,
    _validate_records,
    _window_ranges,
    normalize_residual_japanese,
)
from subtitle_pipeline.local_segmentation import LocalUnit, SpeakerTrack
from subtitle_pipeline.subtitles import Cue
from subtitle_pipeline.translate import (
    LLMHTTPError,
    OpenAICompatibleTranslator,
    _finish_reason,
    _is_nontransient_http_error,
    _normalize_api_response,
    _parse_joint_records,
    _parse_json_object,
    _transient_retry_delay,
)


def _response(content):
    return {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}


class JointTranslationTests(unittest.TestCase):
    def test_inclusive_ranges_allow_single_unit_and_require_full_coverage(self):
        track = SpeakerTrack(
            "A", "A",
            (
                LocalUnit("A", 0, (0,), 0, 1, "一", "A", "speech"),
                LocalUnit("A", 1, (1,), 1, 2, "二", "A", "speech"),
            ),
        )
        records = _validate_records(
            [
                {"start_id": 0, "end_id": 0, "text": "一"},
                {"start_id": 1, "end_id": 1, "text": "二"},
            ], track, 0, 2, 20, "简体中文", validate_language=True,
        )
        self.assertEqual([(item.start_id, item.end_id) for item in records], [(0, 1), (1, 2)])
        with self.assertRaisesRegex(CoverageValidationError, "coverage failed"):
            _validate_records(
                [{"start_id": 0, "end_id": 0, "text": "一"}],
                track, 0, 2, 20, "简体中文", validate_language=True,
            )

    def test_window_relative_ids_map_back_to_track_ids(self):
        track = SpeakerTrack(
            "A",
            "A",
            tuple(
                LocalUnit(
                    "A", index, (index,), index, index + 1, str(index), "A", "speech"
                )
                for index in range(5)
            ),
        )
        records = _validate_records(
            [
                {"start_id": 0, "end_id": 0, "text": "甲"},
                {"start_id": 1, "end_id": 2, "text": "乙"},
            ],
            track,
            2,
            5,
            20,
            "简体中文",
            validate_language=True,
        )
        self.assertEqual(
            [(item.start_id, item.end_id) for item in records],
            [(2, 3), (3, 5)],
        )

    def test_integer_string_ids_are_accepted(self):
        track = SpeakerTrack(
            "A", "A", (LocalUnit("A", 0, (0,), 0, 1, "一", "A", "speech"),)
        )
        records = _validate_records(
            [{"start_id": "0", "end_id": "0", "text": "一"}],
            track,
            0,
            1,
            20,
            "简体中文",
            validate_language=True,
        )
        self.assertEqual((records[0].start_id, records[0].end_id), (0, 1))

    def test_coverage_error_selects_confirmed_neighbor_boundaries(self):
        track = SpeakerTrack(
            "A",
            "A",
            tuple(
                LocalUnit("A", index, (index,), index, index + 1, str(index), "A", "speech")
                for index in range(6)
            ),
        )
        with self.assertRaises(CoverageValidationError) as raised:
            _validate_records(
                [
                    {"start_id": 0, "end_id": 0, "text": "零"},
                    {"start_id": 1, "end_id": 1, "text": "一"},
                    {"start_id": 3, "end_id": 3, "text": "三"},
                    {"start_id": 4, "end_id": 4, "text": "四"},
                    {"start_id": 5, "end_id": 5, "text": "五"},
                ],
                track,
                0,
                6,
                20,
                "简体中文",
                validate_language=True,
            )
        self.assertEqual((raised.exception.patch_start, raised.exception.patch_end), (1, 4))
        self.assertEqual(
            [(record.start_id, record.end_id) for record in raised.exception.preserved],
            [(0, 1), (4, 5), (5, 6)],
        )

    def test_coverage_error_is_repaired_with_a_local_patch(self):
        track = SpeakerTrack(
            "A",
            "A",
            tuple(
                LocalUnit("A", index, (index,), index, index + 1, str(index), "A", "speech")
                for index in range(6)
            ),
        )
        targets: list[str] = []

        def request(body):
            target = body["messages"][1]["content"].split("TARGET:\n", 1)[1]
            targets.append(target)
            if len(targets) == 1:
                cues = [
                    {"start_id": 0, "end_id": 0, "text": "零"},
                    {"start_id": 1, "end_id": 1, "text": "一"},
                    {"start_id": 3, "end_id": 3, "text": "三"},
                    {"start_id": 4, "end_id": 4, "text": "四"},
                    {"start_id": 5, "end_id": 5, "text": "五"},
                ]
            else:
                cues = [{"start_id": 0, "end_id": 2, "text": "补丁"}]
            return _response(json.dumps({"cues": cues,}, ensure_ascii=False))

        records = _request_resilient(
            track,
            0,
            6,
            list(track.units),
            SegmentationConfig(),
            LLMConfig(),
            request,
            {},
            20,
            20,
            "",
            _parse_joint_records,
            _finish_reason,
            lambda _error, _attempt: None,
            lambda _error: False,
            lambda _kind, _error, _content, _request, _response: None,
            lambda text: text,
        )
        self.assertEqual(len(targets), 2)
        self.assertIn("<0>1\n<1>2\n<2>3", targets[1])
        self.assertEqual(
            [(record.start_id, record.end_id) for record in records],
            [(0, 1), (1, 4), (4, 5), (5, 6)],
        )

    def test_noninitial_window_prompt_uses_zero_based_request_ids(self):
        track = SpeakerTrack(
            "A",
            "A",
            tuple(
                LocalUnit(
                    "A", index, (index,), index, index + 1, f"单元{index}", "A", "speech"
                )
                for index in range(5)
            ),
        )
        prompt = _prompt(
            track,
            2,
            5,
            list(track.units),
            SegmentationConfig(),
            {},
            20,
            "简体中文",
            "",
            None,
        )
        target = prompt.split("TARGET:\n", 1)[1]
        self.assertIn("<A>\n<0>单元2\n<1>单元3\n<2>单元4", target)
        self.assertNotIn("<3>单元3", target)

    def test_each_speaker_uses_an_independent_track_and_cache(self):
        translator = OpenAICompatibleTranslator(LLMConfig(max_concurrency=2), "secret")
        cues = [
            Cue(0.0, 1.0, "おはよう", "A"),
            Cue(0.5, 1.4, "はい", "B"),
            Cue(1.1, 2.0, "ございます", "A"),
        ]

        def request(body):
            target = body["messages"][1]["content"].split("TARGET:\n", 1)[1]
            ids = [
                int(part.split(">", 1)[0])
                for part in target.split("<")
                if part and part.split(">", 1)[0].isdigit()
            ]
            speaker = target.splitlines()[0][1:-1]
            content = {"cues": [{"start_id": min(ids), "end_id": max(ids), "text": f"{speaker}字幕"}]}
            return _response(json.dumps(content, ensure_ascii=False))

        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp) / "cue-joint-cache.json"
            audit = Path(temp) / "local-segmentation.json"
            with patch.object(translator, "_request", side_effect=request) as mocked:
                result = translator.plan_and_translate(
                    cues, SegmentationConfig(), max_line_units=20,
                    cache_path=cache, audit_path=audit,
                )
            self.assertEqual(mocked.call_count, 2)
            self.assertTrue(audit.is_file())
            self.assertEqual(json.loads(cache.read_text(encoding="utf-8"))["version"], 5)

            cached = OpenAICompatibleTranslator(LLMConfig(max_concurrency=2), "secret")
            with patch.object(cached, "_request") as cached_request:
                repeated = cached.plan_and_translate(
                    cues, SegmentationConfig(), max_line_units=20, cache_path=cache,
                )
            cached_request.assert_not_called()

        self.assertEqual(result, repeated)
        self.assertEqual([cue.speaker for cue in result.source_cues], ["A", "B"])
        self.assertGreater(result.source_cues[0].end, result.source_cues[1].start)

    def test_coverage_failures_write_complete_llm_audit_records(self):
        with tempfile.TemporaryDirectory() as temp:
            audit = Path(temp) / "llm-audit.jsonl"
            translator = OpenAICompatibleTranslator(
                LLMConfig(), "secret", audit_path=audit
            )
            invalid = _response(
                '{"cues":[{"start_id":1,"end_id":2,"text":"错误范围"}]}'
            )
            with (
                patch.object(translator, "_request", return_value=invalid),
                patch.object(
                    translator.local_translator, "translate", return_value="本地译文"
                ) as local_translate,
                self.assertLogs(
                    "subtitle_pipeline.joint_translation", "WARNING"
                ) as captured,
            ):
                result = translator.plan_and_translate(
                    [Cue(0, 1, "原文", "A")],
                    SegmentationConfig(),
                    max_line_units=20,
                )

            local_translate.assert_called_once_with("原文")
            self.assertEqual(result.translated_cues[0].text, "本地译文")
            self.assertIn(
                "reason=single_unit_CoverageValidationError",
                "\n".join(captured.output),
            )

            entries = [
                json.loads(line)
                for line in audit.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0]["error_type"], "CoverageValidationError")
            self.assertEqual(entries[0]["expected_start"], 0)
            self.assertEqual(entries[0]["expected_end"], 0)
            self.assertEqual(entries[0]["received_ranges"], [[1, 2]])
            self.assertIn("错误范围", entries[0]["response_content"])
            self.assertEqual(entries[0]["response"], invalid)
            self.assertIn("TARGET:", entries[0]["request"]["messages"][1]["content"])

    def test_singing_and_conditioned_speech_are_atomic_windows(self):
        units = (
            LocalUnit("A", 0, (0,), 0, 2, "歌", "A", "singing"),
            LocalUnit("A", 1, (1,), 2, 3, "話", "A", "speech"),
            LocalUnit("A", 2, (2,), 3, 4, "重", "A", "conditioned_speech"),
        )
        self.assertEqual(_window_ranges(units, SegmentationConfig()), [(0, 1), (1, 2), (2, 3)])

    def test_window_limit_chooses_strongest_recent_boundary(self):
        units = tuple(
            LocalUnit(
                "A", index, (index,), index, index + 1, str(index), "A", "speech",
                score,
            )
            for index, score in enumerate((1, 5, 0, None))
        )
        config = SegmentationConfig(model_window_units=3)
        self.assertEqual(_window_ranges(units, config), [(0, 2), (2, 4)])

    def test_window_edge_avoids_leading_dependent_particle(self):
        units = (
            LocalUnit("A", 0, (0,), 0, 1, "前", "A", "speech", 10),
            LocalUnit("A", 1, (1,), 1, 2, "を続ける", "A", "speech", 4),
            LocalUnit("A", 2, (2,), 2, 3, "次", "A", "speech", 1),
            LocalUnit("A", 3, (3,), 3, 4, "終", "A", "speech"),
        )
        config = SegmentationConfig(model_window_units=3)
        self.assertEqual(_window_ranges(units, config)[0], (0, 2))

    def test_context_is_read_only_chronological_nearby_dialogue(self):
        selected = (LocalUnit("A", 1, (1,), 10, 11, "目标", "A", "speech"),)
        values = [
            LocalUnit("B", 0, (0,), 8, 9, "之前", "B", "speech"),
            selected[0],
            LocalUnit("B", 1, (2,), 11.5, 12, "之后", "B", "speech"),
            LocalUnit("C", 0, (3,), 30, 31, "太远", "C", "speech"),
        ]
        context = _dialogue_context(values, selected, SegmentationConfig())
        self.assertEqual(context, "<B>之前\n<B>之后")
        self.assertNotIn("目标", context)

    def test_joint_parser_accepts_object_array_and_ndjson(self):
        records = [{"start_id": 0, "end_id": 0, "text": "甲"}]
        self.assertEqual(_parse_joint_records(json.dumps({"cues": records})), records)
        self.assertEqual(_parse_joint_records(json.dumps(records)), records)
        self.assertEqual(_parse_joint_records(json.dumps(records[0])), records)

    def test_content_failure_retries_once_then_shrinks(self):
        translator = OpenAICompatibleTranslator(LLMConfig(max_retries=5), "secret")
        cues = [Cue(0, 4.1, "行きます", "A"), Cue(4.2, 4.8, "でも", "A")]
        calls = 0

        def request(body):
            nonlocal calls
            calls += 1
            target = body["messages"][1]["content"].split("TARGET:\n", 1)[1]
            ids = [
                int(part.split(">", 1)[0])
                for part in target.split("<")
                if part and part.split(">", 1)[0].isdigit()
            ]
            if len(ids) > 1:
                return _response('{"cues":[]}')
            return _response(json.dumps({"cues": [{"start_id": ids[0], "end_id": ids[0], "text": "好"}]}, ensure_ascii=False))

        with patch.object(translator, "_request", side_effect=request):
            result = translator.plan_and_translate(cues, SegmentationConfig(), max_line_units=20)
        self.assertEqual(calls, 4)
        self.assertEqual([cue.text for cue in result.translated_cues], ["好", "好"])
        self.assertEqual(
            [cue.source_text for cue in result.translated_cues],
            ["行きます", "でも"],
        )

    def test_kana_result_is_machine_translated_locally_without_retry(self):
        translator = OpenAICompatibleTranslator(LLMConfig(), "secret")
        context = {
            "characters": [
                {
                    "canonical": "藤都子",
                    "source_name": "藤都子",
                    "aliases": ["フジミヤコ"],
                    "short_names": [
                        {"source": "ミヤコ", "target": "都子", "context_only": True}
                    ],
                }
            ]
        }

        response = _response(
            json.dumps(
                {"cues": [{"start_id": 0, "end_id": 0, "text": "姓名ミヤコです"}]},
                ensure_ascii=False,
            )
        )

        with (
            patch.object(translator, "_request", return_value=response) as request,
            patch.object(
                translator.local_translator, "translate", return_value="是"
            ) as local_translate,
            self.assertLogs(
                "subtitle_pipeline.joint_translation", "WARNING"
            ) as captured,
        ):
            result = translator.plan_and_translate(
                [Cue(0, 1, "かな", "A")],
                SegmentationConfig(),
                translation_context=context,
                max_line_units=20,
            )
        request.assert_called_once()
        local_translate.assert_called_once_with("です")
        self.assertIn("reason=residual_japanese", "\n".join(captured.output))
        self.assertIn("protected_terms=1", "\n".join(captured.output))
        self.assertEqual(result.translated_cues[0].text, "姓名都子是")

    def test_empty_translation_uses_local_machine_translation(self):
        track = SpeakerTrack(
            "A", "A", (LocalUnit("A", 0, (0,), 0, 1, "ミヤコです", "A", "speech"),)
        )
        translated: list[str] = []

        def local_translate(text):
            translated.append(text)
            return "是"

        with self.assertLogs(
            "subtitle_pipeline.joint_translation", "WARNING"
        ) as captured:
            records = _validate_records(
                [{"start_id": 0, "end_id": 0, "text": "  "}],
                track,
                0,
                1,
                20,
                "简体中文",
                validate_language=True,
                reference_replacements=(("ミヤコ", "都子"),),
                local_translate=local_translate,
            )
        self.assertEqual(translated, ["です"])
        self.assertEqual(records[0].text, "都子是")
        self.assertIn("reason=empty_translation", "\n".join(captured.output))
        self.assertIn("protected_terms=1", "\n".join(captured.output))

    def test_overwide_translation_is_logged_and_accepted(self):
        track = SpeakerTrack(
            "A", "A", (LocalUnit("A", 0, (0,), 0, 1, "原文", "A", "speech"),)
        )
        with self.assertLogs("subtitle_pipeline.joint_translation", "WARNING"):
            records = _validate_records(
                [{"start_id": 0, "end_id": 0, "text": "很长的中文字幕"}],
                track,
                0,
                1,
                2,
                "简体中文",
                validate_language=True,
            )
        self.assertEqual(records[0].text, "很长的中文字幕")

    def test_residual_japanese_is_machine_translated(self):
        self.assertEqual(
            normalize_residual_japanese(
                "角色ありがとう OK", (), lambda text: "谢谢"
            ),
            "谢谢",
        )

    def test_reference_names_are_resolved_before_residual_kana(self):
        replacements = _reference_replacements(
            {
                "characters": [
                    {
                        "canonical": "藤都子",
                        "source_name": "藤都子",
                        "aliases": ["フジミヤコ"],
                        "short_names": [
                            {"source": "ミヤコ", "target": "都子"}
                        ],
                    }
                ],
                "terms": {"バンドリ": "BanG Dream!"},
            }
        )
        self.assertEqual(
            normalize_residual_japanese(
                "バンドリのミヤコです",
                replacements,
                lambda text: {"の": "的", "です": "是"}[text],
            ),
            "BanG Dream!的都子是",
        )

    def test_429_exhaustion_does_not_shrink_window(self):
        translator = OpenAICompatibleTranslator(LLMConfig(max_retries=2), "secret")
        cues = [Cue(0, 0.2, "一", "A"), Cue(0.8, 1.0, "二", "A")]
        with patch.object(
            translator,
            "_request",
            side_effect=LLMHTTPError(429, "rate", retry_after_seconds=0),
        ) as request, patch(
            "subtitle_pipeline.joint_translation.time.sleep"
        ), self.assertRaises(LLMHTTPError):
            translator.plan_and_translate(cues, SegmentationConfig(), max_line_units=20)
        self.assertEqual(request.call_count, 2)

    def test_config_change_invalidates_joint_cache(self):
        translator = OpenAICompatibleTranslator(LLMConfig(), "secret")
        response = _response('{"cues":[{"start_id":0,"end_id":0,"text":"中文"}]}')
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp) / "cache.json"
            with patch.object(translator, "_request", return_value=response):
                translator.plan_and_translate(
                    [Cue(0, 1, "原文", "A")], SegmentationConfig(),
                    max_line_units=20, cache_path=cache,
                )
            with patch.object(translator, "_request", return_value=response) as request:
                translator.plan_and_translate(
                    [Cue(0, 1, "原文", "A")],
                    SegmentationConfig(boundary_score_threshold=4),
                    max_line_units=20, cache_path=cache,
                )
            request.assert_called_once()


class ApiCompatibilityTests(unittest.TestCase):
    def test_only_transient_http_failures_receive_backoff(self):
        self.assertIsNotNone(_transient_retry_delay(LLMHTTPError(429, "rate"), 1))
        self.assertIsNotNone(_transient_retry_delay(LLMHTTPError(503, "busy"), 1))
        self.assertIsNone(_transient_retry_delay(LLMHTTPError(401, "auth"), 1))
        self.assertTrue(_is_nontransient_http_error(LLMHTTPError(401, "auth")))

    def test_parses_fenced_json(self):
        self.assertEqual(_parse_json_object('```json\n{"value":1}\n```'), {"value": 1})

    def test_normalizes_openai_responses_output(self):
        normalized = _normalize_api_response(
            LLMConfig(api_style="responses"),
            {
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
        )
        self.assertEqual(normalized["choices"][0]["message"]["content"], "ok")


if __name__ == "__main__":
    unittest.main()
