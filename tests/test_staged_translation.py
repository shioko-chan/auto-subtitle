import json
import unittest

from subtitle_pipeline.config import LLMConfig, SegmentationConfig
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
