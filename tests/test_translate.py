import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from subtitle_pipeline.config import LLMConfig, SegmentationConfig, TranslationConfig
from subtitle_pipeline.joint_translation import (
    CoverageValidationError,
    _dialogue_context,
    _prompt,
    _reference_replacements,
    _reference_text,
    _request_resilient,
    _validate_records,
    _window_ranges,
    normalize_residual_japanese,
)
from subtitle_pipeline.local_segmentation import LocalUnit, SpeakerTrack
from subtitle_pipeline.subtitles import Cue, TimedTextUnit
from subtitle_pipeline.translate import (
    LLMHTTPError,
    OpenAICompatibleTranslator,
    _finish_reason,
    _is_nontransient_http_error,
    _normalize_api_response,
    _parse_joint_records,
    _parse_json_object,
    _prompt_section_sizes,
    _transient_retry_delay,
)


class PromptBudgetTests(unittest.TestCase):
    def test_prompt_section_sizes_separate_dynamic_evidence(self):
        sections = _prompt_section_sizes(
            "rules\nREFERENCE:\nref\nCURRENT_VIDEO_CHAT:\nchat\n"
            "DIALOGUE_CONTEXT:\ncontext\nSOURCE:\nsource"
        )

        self.assertEqual(sections["reference"]["characters"], 3)
        self.assertEqual(sections["current_video_chat"]["characters"], 4)
        self.assertEqual(sections["dialogue_context"]["characters"], 7)
        self.assertEqual(sections["source"]["characters"], 6)
        self.assertEqual(sections["fixed_prompt"]["characters"], 5)

    def test_request_context_audit_records_actual_usage_and_section_sizes(self):
        with tempfile.TemporaryDirectory() as temporary:
            audit = Path(temporary) / "llm-audit.jsonl"
            translator = OpenAICompatibleTranslator(
                LLMConfig(local_server_enabled=True, local_server_context_size=16384),
                TranslationConfig(),
                "secret",
                audit_path=audit,
            )
            translator._log_request_context(
                {
                    "model": "test",
                    "max_tokens": 8192,
                    "messages": [
                        {
                            "role": "user",
                            "content": "SOURCE:\n原文\nCURRENT_VIDEO_CHAT:\n聊天",
                        }
                    ],
                },
                {
                    "usage": {
                        "prompt_tokens": 123,
                        "completion_tokens": 45,
                    }
                },
            )
            event = json.loads(audit.read_text())

        self.assertEqual(event["prompt_tokens"], 123)
        self.assertEqual(event["completion_tokens"], 45)
        self.assertEqual(event["context_capacity"], 16384)
        self.assertIn("source", event["sections"])
        self.assertIn("current_video_chat", event["sections"])


def _response(content):
    return {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}


class JointTranslationTests(unittest.TestCase):
    def test_reference_text_excludes_large_audit_payloads(self):
        reference = json.loads(
            _reference_text(
                {
                    "video": {"title": "配信"},
                    "terms": {"ミヤコ": "都子"},
                    "identified_songs": [{"alignment": "large"}],
                    "asr_evidence": [{"text": "large"}],
                }
            )
        )
        self.assertEqual(
            reference,
            {"video": {"title": "配信"}, "terms": {"ミヤコ": "都子"}},
        )

    def test_inclusive_ranges_allow_single_unit_and_require_full_coverage(self):
        track = SpeakerTrack(
            "A",
            "A",
            (
                LocalUnit("A", 0, (0,), 0, 1, "一", "A", "speech"),
                LocalUnit("A", 1, (1,), 1, 2, "二", "A", "speech"),
            ),
        )
        records = _validate_records(
            [
                {"start_id": 0, "end_id": 0, "text": "一"},
                {"start_id": 1, "end_id": 1, "text": "二"},
            ],
            track,
            0,
            2,
            20,
            "简体中文",
            validate_language=True,
        )
        self.assertEqual(
            [(item.start_id, item.end_id) for item in records], [(0, 1), (1, 2)]
        )
        with self.assertRaisesRegex(CoverageValidationError, "coverage failed"):
            _validate_records(
                [{"start_id": 0, "end_id": 0, "text": "一"}],
                track,
                0,
                2,
                20,
                "简体中文",
                validate_language=True,
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
                LocalUnit(
                    "A", index, (index,), index, index + 1, str(index), "A", "speech"
                )
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
        self.assertEqual(
            (raised.exception.patch_start, raised.exception.patch_end), (1, 4)
        )
        self.assertEqual(
            [(record.start_id, record.end_id) for record in raised.exception.preserved],
            [(0, 1), (4, 5), (5, 6)],
        )

    def test_coverage_error_is_repaired_with_a_local_patch(self):
        track = SpeakerTrack(
            "A",
            "A",
            tuple(
                LocalUnit(
                    "A", index, (index,), index, index + 1, str(index), "A", "speech"
                )
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
            return _response(
                json.dumps(
                    {
                        "cues": cues,
                    },
                    ensure_ascii=False,
                )
            )

        records = _request_resilient(
            track,
            0,
            6,
            list(track.units),
            SegmentationConfig(),
            TranslationConfig(),
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
                    "A",
                    index,
                    (index,),
                    index,
                    index + 1,
                    f"单元{index}",
                    "A",
                    "speech",
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
        translator = OpenAICompatibleTranslator(
            LLMConfig(max_concurrency=2), TranslationConfig(), "secret"
        )
        cues = [
            Cue(0.0, 1.0, "おはよう", "A"),
            Cue(0.5, 1.4, "はい", "B"),
            Cue(1.1, 2.0, "ございます", "A"),
        ]

        def request(body):
            prompt = body["messages"][1]["content"]
            if "WINDOWS:\n" in prompt:
                return _response(
                    json.dumps(
                        {
                            "windows": [
                                {
                                    "window_id": 0,
                                    "cues": [
                                        {"start_id": 0, "end_id": 0, "text": "A字幕"}
                                    ],
                                },
                                {
                                    "window_id": 1,
                                    "cues": [
                                        {"start_id": 0, "end_id": 0, "text": "B字幕"}
                                    ],
                                },
                            ]
                        },
                        ensure_ascii=False,
                    )
                )
            target = prompt.split("TARGET:\n", 1)[1]
            ids = [
                int(part.split(">", 1)[0])
                for part in target.split("<")
                if part and part.split(">", 1)[0].isdigit()
            ]
            speaker = target.splitlines()[0][1:-1]
            content = {
                "cues": [
                    {"start_id": min(ids), "end_id": max(ids), "text": f"{speaker}字幕"}
                ]
            }
            return _response(json.dumps(content, ensure_ascii=False))

        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp) / "cue-joint-cache.json"
            audit = Path(temp) / "local-segmentation.json"
            with patch.object(translator, "_request", side_effect=request) as mocked:
                result = translator.plan_and_translate(
                    cues,
                    SegmentationConfig(),
                    max_line_units=20,
                    cache_path=cache,
                    audit_path=audit,
                )
            self.assertEqual(mocked.call_count, 1)
            self.assertTrue(audit.is_file())
            self.assertEqual(
                json.loads(cache.read_text(encoding="utf-8"))["version"], 8
            )

            cached = OpenAICompatibleTranslator(
                LLMConfig(max_concurrency=2), TranslationConfig(), "secret"
            )
            with patch.object(cached, "_request") as cached_request:
                repeated = cached.plan_and_translate(
                    cues,
                    SegmentationConfig(),
                    max_line_units=20,
                    cache_path=cache,
                )
            cached_request.assert_not_called()

        self.assertEqual(result, repeated)
        self.assertEqual([cue.speaker for cue in result.source_cues], ["A", "B"])
        self.assertGreater(result.source_cues[0].end, result.source_cues[1].start)

    def test_distant_same_speaker_episodes_share_request_but_not_cue(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(max_concurrency=2), TranslationConfig(), "secret"
        )
        cues = [
            Cue(0.0, 0.5, "なんか", "A"),
            Cue(1000.0, 1000.5, "嬉しい", "A"),
        ]

        def request(body):
            prompt = body["messages"][1]["content"]
            self.assertEqual(prompt.count('<WINDOW id="'), 2)
            return _response(
                json.dumps(
                    {
                        "windows": [
                            {
                                "window_id": 0,
                                "cues": [
                                    {"start_id": 0, "end_id": 0, "text": "总觉得"}
                                ],
                            },
                            {
                                "window_id": 1,
                                "cues": [
                                    {"start_id": 0, "end_id": 0, "text": "很开心"}
                                ],
                            },
                        ]
                    },
                    ensure_ascii=False,
                )
            )

        with patch.object(translator, "_request", side_effect=request) as mocked:
            result = translator.plan_and_translate(
                cues,
                SegmentationConfig(),
                max_line_units=20,
            )

        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(
            [cue.text for cue in result.translated_cues], ["总觉得", "很开心"]
        )
        self.assertEqual(
            [(cue.start, cue.end) for cue in result.translated_cues],
            [(0.0, 0.5), (1000.0, 1000.5)],
        )

    def test_missing_batch_window_retries_only_that_window(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(max_concurrency=2), TranslationConfig(), "secret"
        )
        cues = [
            Cue(0.0, 0.5, "なんか", "A"),
            Cue(1000.0, 1000.5, "嬉しい", "A"),
        ]

        def request(body):
            prompt = body["messages"][1]["content"]
            if "WINDOWS:\n" in prompt:
                return _response(
                    json.dumps(
                        {
                            "windows": [
                                {
                                    "window_id": 0,
                                    "cues": [
                                        {
                                            "start_id": 0,
                                            "end_id": 0,
                                            "text": "总觉得",
                                        }
                                    ],
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )
                )
            self.assertIn("<0>嬉しい", prompt)
            return _response(
                json.dumps(
                    {"cues": [{"start_id": 0, "end_id": 0, "text": "很开心"}]},
                    ensure_ascii=False,
                )
            )

        with patch.object(translator, "_request", side_effect=request) as mocked:
            result = translator.plan_and_translate(
                cues,
                SegmentationConfig(),
                max_line_units=20,
            )

        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(
            [cue.text for cue in result.translated_cues], ["总觉得", "很开心"]
        )

    def test_verified_lyric_translation_bypasses_general_llm(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(), TranslationConfig(), "secret"
        )
        cue = Cue(
            1.0,
            3.0,
            "伝えたくて",
            "singer",
            "singing",
            preferred_translation="想要告诉你",
        )
        with patch.object(
            translator,
            "_request",
            side_effect=AssertionError(
                "general LLM must not translate verified lyrics"
            ),
        ):
            result = translator.plan_and_translate(
                [cue], SegmentationConfig(), max_line_units=30
            )
        self.assertEqual(result.translated_cues[0].text, "想要告诉你")

    def test_coverage_failures_write_complete_llm_audit_records(self):
        with tempfile.TemporaryDirectory() as temp:
            audit = Path(temp) / "llm-audit.jsonl"
            translator = OpenAICompatibleTranslator(
                LLMConfig(), TranslationConfig(), "secret", audit_path=audit
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

            local_translate.assert_called_once_with("原文", source_language="Japanese")
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
        self.assertEqual(
            _window_ranges(units, SegmentationConfig()), [(0, 1), (1, 2), (2, 3)]
        )

    def test_window_never_crosses_same_speaker_episode_gap(self):
        units = (
            LocalUnit("A", 0, (0,), 10.0, 10.5, "なんか", "A", "speech"),
            LocalUnit("A", 1, (1,), 10.8, 11.2, "少し", "A", "speech"),
            LocalUnit("A", 2, (2,), 1011.0, 1011.5, "嬉しい", "A", "speech"),
        )
        config = SegmentationConfig(speaker_episode_gap_seconds=2.0)

        self.assertEqual(_window_ranges(units, config), [(0, 2), (2, 3)])

    def test_window_limit_chooses_strongest_recent_boundary(self):
        units = tuple(
            LocalUnit(
                "A",
                index,
                (index,),
                index,
                index + 1,
                str(index),
                "A",
                "speech",
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
        self.assertEqual(
            context,
            "<B language=Japanese>之前\n<B language=Japanese>之后",
        )
        self.assertNotIn("目标", context)

    def test_joint_parser_accepts_object_array_and_ndjson(self):
        records = [{"start_id": 0, "end_id": 0, "text": "甲"}]
        self.assertEqual(_parse_joint_records(json.dumps({"cues": records})), records)
        self.assertEqual(_parse_joint_records(json.dumps(records)), records)
        self.assertEqual(_parse_joint_records(json.dumps(records[0])), records)

    def test_content_failure_retries_once_then_shrinks(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(max_retries=5), TranslationConfig(), "secret"
        )
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
            return _response(
                json.dumps(
                    {"cues": [{"start_id": ids[0], "end_id": ids[0], "text": "好"}]},
                    ensure_ascii=False,
                )
            )

        with patch.object(translator, "_request", side_effect=request):
            result = translator.plan_and_translate(
                cues, SegmentationConfig(), max_line_units=20
            )
        self.assertEqual(calls, 4)
        self.assertEqual([cue.text for cue in result.translated_cues], ["好", "好"])
        self.assertEqual(
            [cue.source_text for cue in result.translated_cues],
            ["行きます", "でも"],
        )
        self.assertEqual(
            [cue.source_units for cue in result.translated_cues],
            [
                (TimedTextUnit("行きます", 0, 4.1),),
                (TimedTextUnit("でも", 4.2, 4.8),),
            ],
        )

    def test_kana_result_is_machine_translated_locally_without_retry(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(), TranslationConfig(), "secret"
        )
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
        local_translate.assert_called_once_with("です", source_language="Japanese")
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
            normalize_residual_japanese("角色ありがとう OK", (), lambda text: "谢谢"),
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
                        "short_names": [{"source": "ミヤコ", "target": "都子"}],
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
        translator = OpenAICompatibleTranslator(
            LLMConfig(max_retries=2), TranslationConfig(), "secret"
        )
        cues = [Cue(0, 0.2, "一", "A"), Cue(0.8, 1.0, "二", "A")]
        with (
            patch.object(
                translator,
                "_request",
                side_effect=LLMHTTPError(429, "rate", retry_after_seconds=0),
            ) as request,
            patch("subtitle_pipeline.joint_translation.time.sleep"),
            self.assertRaises(LLMHTTPError),
        ):
            translator.plan_and_translate(cues, SegmentationConfig(), max_line_units=20)
        self.assertEqual(request.call_count, 2)

    def test_config_change_invalidates_joint_cache(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(), TranslationConfig(), "secret"
        )
        response = _response('{"cues":[{"start_id":0,"end_id":0,"text":"中文"}]}')
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp) / "cache.json"
            with patch.object(translator, "_request", return_value=response):
                translator.plan_and_translate(
                    [Cue(0, 1, "原文", "A")],
                    SegmentationConfig(),
                    max_line_units=20,
                    cache_path=cache,
                )
            with patch.object(translator, "_request", return_value=response) as request:
                translator.plan_and_translate(
                    [Cue(0, 1, "原文", "A")],
                    SegmentationConfig(boundary_score_threshold=4),
                    max_line_units=20,
                    cache_path=cache,
                )
            request.assert_called_once()


class ApiCompatibilityTests(unittest.TestCase):
    def test_lyrics_review_only_applies_explicit_corrections(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(), TranslationConfig(), "secret"
        )
        response = _response(
            json.dumps(
                {
                    "corrections": [
                        {
                            "line_id": "1",
                            "text": "和这个世界一起蜕变吧",
                            "reason": "外来语误译和无依据增译",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )
        with patch.object(translator, "_request", return_value=response) as request:
            reviewed = translator.review_lyrics(
                "Song",
                "Artist",
                ["前の行", "この世とメタモルフォーゼしようぜ", "次の行"],
                {0: "前一行", 1: "让这世界与变形虫共舞吧", 2: "下一行"},
                translation_context={"terms": {"メタモルフォーゼ": "蜕变"}},
            )

        self.assertEqual(
            reviewed,
            {0: "前一行", 1: "和这个世界一起蜕变吧", 2: "下一行"},
        )
        prompt = request.call_args.args[0]["messages"][1]["content"]
        self.assertIn("この世とメタモルフォーゼしようぜ", prompt)
        self.assertIn("让这世界与变形虫共舞吧", prompt)
        self.assertIn("メタモルフォーゼ", prompt)

    def test_lyrics_review_accepts_no_corrections(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(), TranslationConfig(), "secret"
        )
        response = _response('{"corrections":[]}')
        with patch.object(translator, "_request", return_value=response):
            reviewed = translator.review_lyrics(
                "Song", "Artist", ["正しい歌詞"], {0: "正确的歌词"}
            )

        self.assertEqual(reviewed, {0: "正确的歌词"})

    def test_lyrics_prompt_excludes_runtime_audit_payloads(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(), TranslationConfig(), "secret"
        )
        response = _response(
            json.dumps(
                {"lines": [{"line_id": 0, "text": "准备好去寻找答案"}]},
                ensure_ascii=False,
            )
        )
        with patch.object(translator, "_request", return_value=response) as request:
            translator.translate_lyrics(
                "Song",
                "Artist",
                ["Ready set and find out"],
                translation_context={
                    "video": {"description": "large video context"},
                    "franchises": [
                        {"name": "夢限大みゅーたいぷ", "background": "large background"}
                    ],
                    "terms": {"ミヤコ": "都子"},
                    "asr_evidence": [{"text": "large ASR audit"}],
                    "identified_songs": [{"pyshiro": "large alignment audit"}],
                },
            )

        prompt = request.call_args.args[0]["messages"][1]["content"]
        self.assertIn("夢限大みゅーたいぷ", prompt)
        self.assertIn("ミヤコ", prompt)
        self.assertNotIn("large background", prompt)
        self.assertNotIn("large video context", prompt)
        self.assertNotIn("large ASR audit", prompt)
        self.assertNotIn("large alignment audit", prompt)

    def test_metadata_prompt_excludes_large_audit_payloads(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(), TranslationConfig(), "secret"
        )
        response = _response(
            json.dumps(
                {
                    "title": "标题",
                    "description": "简介",
                    "content_summary": "摘要",
                    "tags": ["标签"],
                },
                ensure_ascii=False,
            )
        )
        with patch.object(translator, "_request", return_value=response) as request:
            translator.translate_metadata(
                "title",
                "description",
                translation_context={
                    "terms": {"ミヤコ": "都子"},
                    "identified_songs": [{"alignment": "large"}],
                    "asr_evidence": [{"text": "large"}],
                },
            )
        prompt = request.call_args.args[0]["messages"][1]["content"]
        self.assertIn("ミヤコ", prompt)
        self.assertNotIn("identified_songs", prompt)
        self.assertNotIn("asr_evidence", prompt)

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
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
        )
        self.assertEqual(normalized["choices"][0]["message"]["content"], "ok")


if __name__ == "__main__":
    unittest.main()
