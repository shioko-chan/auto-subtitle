import json
import tempfile
import unittest
from pathlib import Path

from subtitle_pipeline.config import LLMConfig, SegmentationConfig
from subtitle_pipeline.fan_knowledge import KnowledgeHit, KnowledgeScore
from subtitle_pipeline.local_segmentation import LocalUnit, SpeakerTrack
from subtitle_pipeline.song_identification import (
    SongIdentificationResult,
    split_aligned_song_cues,
)
from subtitle_pipeline.staged_translation import (
    run_fixed_translation,
    run_segmentation,
)
from subtitle_pipeline.subtitles import Cue, TimedTextUnit, text_display_width


def parse_cues(content):
    return json.loads(content)["cues"]


def finish_reason(_response):
    return "stop"


def no_delay(_error, _attempt):
    return None


def not_nontransient(_error):
    return False


def ignore_audit(*_args):
    return None


class StagedTranslationTests(unittest.TestCase):
    def test_short_cues_use_the_expanded_request_group_limit(self):
        source = [Cue(i, i + 1, f"短句{i}", "A") for i in range(20)]
        prompts: list[str] = []

        def request(body):
            prompts.append(body["messages"][1]["content"])
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "cues": [
                                        {"cue_id": i, "text": f"译文{i}"}
                                        for i in range(20)
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        result = run_fixed_translation(
            source_cues=source,
            llm=LLMConfig(),
            segmentation=SegmentationConfig(),
            request=request,
            parse_content=parse_cues,
            finish_reason=finish_reason,
            retry_delay=no_delay,
            is_nontransient=not_nontransient,
            log_invalid_response=ignore_audit,
            local_translate=lambda text: text,
            translation_context={},
            maximum_units=20.0,
            honorific_rules="",
            cache_path=None,
        )

        self.assertEqual(len(prompts), 1)
        self.assertEqual(len(result), 20)

    def test_segmentation_retries_an_overwide_source_group(self):
        cues = [Cue(0, 1, "あいう", "A"), Cue(1, 2, "えおか", "A")]
        track = SpeakerTrack(
            "A",
            "A",
            (
                LocalUnit("A", 0, (0,), 0, 1, "あいう", "A", "speech"),
                LocalUnit("A", 1, (1,), 1, 2, "えおか", "A", "speech"),
            ),
        )
        responses = iter(
            [
                {"cues": [{"start_id": 0, "end_id": 1}]},
                {
                    "cues": [
                        {"start_id": 0, "end_id": 0},
                        {"start_id": 1, "end_id": 1},
                    ]
                },
            ]
        )

        def request(_body):
            return {"choices": [{"message": {"content": json.dumps(next(responses))}}]}

        result = run_segmentation(
            tracks=[track],
            source_cues=cues,
            segmentation=SegmentationConfig(),
            llm=LLMConfig(max_retries=2),
            request=request,
            parse_content=parse_cues,
            finish_reason=finish_reason,
            retry_delay=no_delay,
            is_nontransient=not_nontransient,
            log_invalid_response=ignore_audit,
            source_maximum_units=4.0,
            cache_path=None,
            sudachi_versions={},
        )

        self.assertEqual([cue.text for cue in result], ["あいう", "えおか"])

    def test_fixed_translation_cannot_change_source_boundaries(self):
        source = [Cue(0, 1, "原文一", "A"), Cue(1, 2, "原文二", "A")]

        def request(_body):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "cues": [
                                        {"cue_id": 0, "text": "译文一"},
                                        {"cue_id": 1, "text": "译文二"},
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        translated = run_fixed_translation(
            source_cues=source,
            llm=LLMConfig(),
            segmentation=SegmentationConfig(),
            request=request,
            parse_content=parse_cues,
            finish_reason=finish_reason,
            retry_delay=no_delay,
            is_nontransient=not_nontransient,
            log_invalid_response=ignore_audit,
            local_translate=lambda text: f"MT:{text}",
            translation_context={},
            maximum_units=20.0,
            honorific_rules="",
            cache_path=None,
        )

        self.assertEqual([(cue.start, cue.end) for cue in translated], [(0, 1), (1, 2)])
        self.assertEqual([cue.text for cue in translated], ["译文一", "译文二"])

    def test_fixed_translation_retrieves_knowledge_for_each_request_group(self):
        source = [Cue(0, 1, "本日分の貢ぎ物", "A")]
        retrieved: list[list[Cue]] = []
        prompts: list[str] = []

        def retrieve(cues, _chat_text):
            retrieved.append(cues)
            return [
                KnowledgeHit(
                    "knowledge:ty",
                    "catchphrase",
                    "TY",
                    "TY在这里表示Thank You。",
                    None,
                    KnowledgeScore(1, 0, 0, 1, 1, 0, 1, 5),
                    ("TY",),
                )
            ]

        def request(body):
            prompts.append(body["messages"][1]["content"])
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"cues": [{"cue_id": 0, "text": "今日的礼物"}]},
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        run_fixed_translation(
            source_cues=source,
            llm=LLMConfig(),
            segmentation=SegmentationConfig(),
            request=request,
            parse_content=parse_cues,
            finish_reason=finish_reason,
            retry_delay=no_delay,
            is_nontransient=not_nontransient,
            log_invalid_response=ignore_audit,
            local_translate=lambda text: f"MT:{text}",
            translation_context={},
            maximum_units=20.0,
            honorific_rules="",
            cache_path=None,
            retrieve_knowledge=retrieve,
        )

        self.assertEqual(
            [[cue.text for cue in group] for group in retrieved], [["本日分の貢ぎ物"]]
        )
        self.assertIn("TY在这里表示Thank You", prompts[0])

    def test_fixed_translation_retrieves_chat_once_for_request_group(self):
        source = [Cue(10, 11, "これ", "A"), Cue(11, 12, "それ", "A")]
        prompts: list[str] = []

        def request(body):
            prompts.append(body["messages"][1]["content"])
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "cues": [
                                        {"cue_id": 0, "text": "这个"},
                                        {"cue_id": 1, "text": "那个"},
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        run_fixed_translation(
            source_cues=source,
            llm=LLMConfig(),
            segmentation=SegmentationConfig(),
            request=request,
            parse_content=parse_cues,
            finish_reason=finish_reason,
            retry_delay=no_delay,
            is_nontransient=not_nontransient,
            log_invalid_response=ignore_audit,
            local_translate=lambda text: text,
            translation_context={},
            maximum_units=20.0,
            honorific_rules="",
            cache_path=None,
            retrieve_chat=lambda cues: (
                f"chat@{min(cue.start for cue in cues):.0f}-"
                f"{max(cue.end for cue in cues):.0f}"
            ),
        )

        self.assertIn("chat@10-12", prompts[0])
        self.assertEqual(prompts[0].count("chat@10-12"), 1)

    def test_fixed_translation_deduplicates_chat_shared_by_adjacent_cues(self):
        source = [Cue(10, 11, "一", "A"), Cue(11, 12, "二", "A")]
        prompts: list[str] = []

        def request(body):
            prompts.append(body["messages"][1]["content"])
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "cues": [
                                        {"cue_id": 0, "text": "一"},
                                        {"cue_id": 1, "text": "二"},
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        run_fixed_translation(
            source_cues=source,
            llm=LLMConfig(),
            segmentation=SegmentationConfig(),
            request=request,
            parse_content=parse_cues,
            finish_reason=finish_reason,
            retry_delay=no_delay,
            is_nontransient=not_nontransient,
            log_invalid_response=ignore_audit,
            local_translate=lambda text: text,
            translation_context={},
            maximum_units=20.0,
            honorific_rules="",
            cache_path=None,
            retrieve_chat=lambda _cue: "[+1.0s chat] 共同证据",
        )

        self.assertEqual(prompts[0].count("共同证据"), 1)
        self.assertIn("[+1.0s chat] 共同证据", prompts[0])

    def test_fixed_translation_cache_does_not_depend_on_knowledge(self):
        source = [Cue(0, 1, "原文", "A")]

        def request(_body):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"cues": [{"cue_id": 0, "text": "译文"}]},
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "translation.json"
            arguments = {
                "source_cues": source,
                "llm": LLMConfig(),
                "segmentation": SegmentationConfig(),
                "parse_content": parse_cues,
                "finish_reason": finish_reason,
                "retry_delay": no_delay,
                "is_nontransient": not_nontransient,
                "log_invalid_response": ignore_audit,
                "local_translate": lambda text: f"MT:{text}",
                "translation_context": {},
                "maximum_units": 20.0,
                "honorific_rules": "",
                "cache_path": cache,
            }
            first = run_fixed_translation(
                **arguments,
                request=request,
                retrieve_knowledge=lambda _cues, _chat: [],
            )
            second = run_fixed_translation(
                **arguments,
                request=lambda _body: self.fail("LLM should not run on cache hit"),
                retrieve_knowledge=lambda _cues, _chat: self.fail(
                    "knowledge retrieval should not run on cache hit"
                ),
                retrieve_chat=lambda _cue: self.fail(
                    "chat retrieval should not run on cache hit"
                ),
            )

        self.assertEqual(first, second)

    def test_song_line_splits_on_pyshiro_units_at_japanese_limit(self):
        units = tuple(
            TimedTextUnit(character, index * 0.5, (index + 1) * 0.5)
            for index, character in enumerate("一二三四五六")
        )
        result = split_aligned_song_cues(
            SongIdentificationResult(
                [
                    Cue(
                        0,
                        3,
                        "一二三四五六",
                        "A",
                        "singing",
                        preferred_translation="甲乙丙丁戊己",
                        source_units=units,
                    )
                ],
                [],
            ),
            3.0,
        )

        self.assertEqual(
            [cue.text for cue in result.corrected_cues], ["一二三", "四五六"]
        )
        self.assertEqual(
            [cue.preferred_translation for cue in result.corrected_cues],
            ["甲乙丙", "丁戊己"],
        )
        self.assertEqual(
            [(cue.start, cue.end) for cue in result.corrected_cues],
            [(0.0, 1.5), (1.5, 3.0)],
        )
        self.assertTrue(
            all(text_display_width(cue.text) <= 3 for cue in result.corrected_cues)
        )


if __name__ == "__main__":
    unittest.main()
