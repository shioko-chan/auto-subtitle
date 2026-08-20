import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from subtitle_pipeline.audio_analysis import AudioRegion
from subtitle_pipeline.conditioned_asr import (
    _conditioned_windows,
    _replace_windows,
    repair_long_overlaps,
)
from subtitle_pipeline.config import AudioAnalysisConfig
from subtitle_pipeline.subtitles import Cue


class ConditionedASRTests(unittest.TestCase):
    def test_half_second_overlap_creates_conditioned_window_by_default(self):
        diarization = [
            AudioRegion(0, 4, "speech", "A", anonymous_speaker="S0"),
            AudioRegion(2.5, 5, "speech", "B", anonymous_speaker="S1"),
            AudioRegion(8, 10, "speech", "A", anonymous_speaker="S0"),
            AudioRegion(9.5, 11, "speech", "B", anonymous_speaker="S1"),
        ]
        config = AudioAnalysisConfig(overlap_context_seconds=1.0)

        result = _conditioned_windows(diarization, 20, config)

        self.assertEqual(len(result), 2)
        self.assertEqual((result[0].start, result[0].end), (1.5, 5.0))
        self.assertEqual(result[0].speakers, ("S0", "S1"))
        self.assertEqual((result[1].start, result[1].end), (8.5, 11.0))

    def test_explicit_long_overlap_threshold_filters_shorter_overlap(self):
        diarization = [
            AudioRegion(0, 4, "speech", "A", anonymous_speaker="S0"),
            AudioRegion(2.5, 5, "speech", "B", anonymous_speaker="S1"),
            AudioRegion(8, 10, "speech", "A", anonymous_speaker="S0"),
            AudioRegion(9.5, 11, "speech", "B", anonymous_speaker="S1"),
        ]
        config = AudioAnalysisConfig(
            overlap_conditioned_asr_seconds=1.5,
            overlap_context_seconds=1.0,
        )

        result = _conditioned_windows(diarization, 20, config)

        self.assertEqual(len(result), 1)
        self.assertEqual((result[0].start, result[0].end), (1.5, 5.0))

    def test_context_clips_long_surrounding_turn_to_model_window(self):
        diarization = [
            AudioRegion(0, 40, "speech", "A", anonymous_speaker="S0"),
            AudioRegion(10, 13, "speech", "B", anonymous_speaker="S1"),
        ]
        config = AudioAnalysisConfig(overlap_context_seconds=2.0)

        [result] = _conditioned_windows(diarization, 60, config)

        self.assertEqual((result.start, result.end), (8.0, 15.0))
        self.assertEqual(
            [(turn.start, turn.end) for turn in result.turns],
            [(8.0, 15.0), (10, 13)],
        )

    def test_conditioned_repair_replaces_complete_local_window(self):
        baseline = [
            Cue(0, 1, "before", "A"),
            Cue(2, 3, "mixed", "A"),
            Cue(5, 6, "after", "B"),
            Cue(8, 9, "outside", "A"),
            Cue(4, 5, "song", "A", "singing"),
        ]
        repaired = [Cue(1.5, 3.2, "speaker A", "A")]
        window = SimpleNamespace(start=1.0, end=6.5)

        result = _replace_windows(baseline, repaired, [window])

        self.assertEqual(
            [cue.text for cue in result],
            ["before", "speaker A", "song", "after", "outside"],
        )

    def test_disabled_backend_fails_instead_of_dropping_long_overlap(self):
        diarization = [
            AudioRegion(0, 4, "speech", "A", anonymous_speaker="S0"),
            AudioRegion(2, 5, "speech", "B", anonymous_speaker="S1"),
        ]
        audio = SimpleNamespace(duration=10)
        config = AudioAnalysisConfig(conditioned_asr_backend="disabled")
        with (
            tempfile.TemporaryDirectory() as temp,
            self.assertRaisesRegex(RuntimeError, "backend is disabled"),
        ):
            repair_long_overlaps([], diarization, audio, Path(temp), config)

    def test_successful_conditioned_result_is_cached(self):
        diarization = [
            AudioRegion(0, 4, "speech", "A", anonymous_speaker="S0"),
            AudioRegion(2, 5, "speech", "B", anonymous_speaker="S1"),
        ]
        audio = SimpleNamespace(duration=10)
        config = AudioAnalysisConfig()
        with (
            tempfile.TemporaryDirectory() as temp,
            patch(
                "subtitle_pipeline.conditioned_asr._run_dicow",
                return_value=[Cue(1, 3, "fixed", "A")],
            ) as run,
        ):
            first = repair_long_overlaps([], diarization, audio, Path(temp), config)
            second = repair_long_overlaps([], diarization, audio, Path(temp), config)

        self.assertEqual(first, second)
        self.assertEqual(first.cues[0].kind, "conditioned_speech")
        run.assert_called_once()

    def test_qwen_overlap_units_are_preserved_as_read_only_evidence(self):
        diarization = [
            AudioRegion(0, 4, "speech", "A", anonymous_speaker="S0"),
            AudioRegion(2, 5, "speech", "B", anonymous_speaker="S1"),
        ]
        qwen_windows = [
            {
                "core_start": 0,
                "core_end": 60,
                "text": "整窗文本不应直接发送",
                "cues": [
                    {"start": 0.5, "end": 1.0, "text": "范围外"},
                    {"start": 1.5, "end": 2.0, "text": "お願いします"},
                    {"start": 2.0, "end": 2.4, "text": "よろしく"},
                    {"start": 8.0, "end": 9.0, "text": "范围外"},
                ],
            }
        ]
        audio = SimpleNamespace(duration=10)
        config = AudioAnalysisConfig(overlap_context_seconds=0.5)
        with (
            tempfile.TemporaryDirectory() as temp,
            patch(
                "subtitle_pipeline.conditioned_asr._run_dicow",
                return_value=[Cue(1.5, 3.0, "googlegoogle", "A")],
            ),
        ):
            result = repair_long_overlaps(
                [],
                diarization,
                audio,
                Path(temp),
                config,
                qwen_windows=qwen_windows,
            )

        self.assertEqual([cue.text for cue in result.cues], ["googlegoogle"])
        [evidence] = result.evidence
        [qwen] = evidence["qwen_mixed"]
        self.assertEqual(qwen["text"], "お願いしますよろしく")
        self.assertEqual(len(qwen["units"]), 2)
        self.assertNotIn("整窗文本不应直接发送", str(evidence))

    def test_dicow_repetition_hallucination_preserves_qwen_baseline(self):
        diarization = [
            AudioRegion(0, 4, "speech", "A", anonymous_speaker="S0"),
            AudioRegion(2, 5, "speech", "B", anonymous_speaker="S1"),
        ]
        baseline = [
            Cue(1.5, 2.5, "Qwen A", "A"),
            Cue(2.5, 3.5, "Qwen B", "B"),
        ]
        audio = SimpleNamespace(duration=10)
        config = AudioAnalysisConfig(overlap_context_seconds=0.5)
        with (
            tempfile.TemporaryDirectory() as temp,
            patch(
                "subtitle_pipeline.conditioned_asr._run_dicow",
                return_value=[
                    Cue(1.5, 3.0, "DiCoW A", "A"),
                    Cue(2.0, 4.0, "私は" * 100, "B"),
                ],
            ),
        ):
            result = repair_long_overlaps(
                baseline, diarization, audio, Path(temp), config
            )

        self.assertEqual(
            [(cue.text, cue.speaker, cue.kind) for cue in result.cues],
            [
                ("DiCoW A", "A", "conditioned_speech"),
                ("Qwen B", "B", "speech"),
            ],
        )
        self.assertNotIn("私は私は", str(result.evidence))


if __name__ == "__main__":
    unittest.main()
