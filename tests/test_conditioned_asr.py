import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from subtitle_pipeline.asr import (
    _conditioned_aligned_cues,
    _conditioned_asr_records,
)
from subtitle_pipeline.audio_analysis import AudioRegion
from subtitle_pipeline.cache import CacheStore
from subtitle_pipeline.conditioned_asr import (
    ConditionedWindow,
    _conditioned_windows,
    _replace_windows,
    reconcile_long_overlaps,
    repair_long_overlaps,
    transcribe_long_overlaps,
)
from subtitle_pipeline.config import AudioAnalysisConfig
from subtitle_pipeline.subtitles import Cue


def stream_cues(cues):
    def run(_audio, windows, _config, on_window):
        for index, window in enumerate(windows):
            on_window(index, [cue for cue in cues if window.start <= (cue.start + cue.end) / 2 < window.end])
    return run


class ConditionedASRTests(unittest.TestCase):
    def test_conditioned_records_preserve_source_and_use_corrected_alignment(self):
        raw = [Cue(1.0, 3.0, "raw", "A", "conditioned_speech")]

        records, regions = _conditioned_asr_records(raw, 7)
        corrected_records = [
            {
                **records[0],
                "text": "corrected",
                "cues": [
                    {
                        "start": 1.1,
                        "end": 2.9,
                        "text": "corrected",
                        "kind": "speech",
                    }
                ],
            }
        ]
        result = _conditioned_aligned_cues(corrected_records, 7)

        self.assertEqual(records[0]["window_id"], 7)
        self.assertEqual(records[0]["asr_source"], "dicow")
        self.assertEqual(regions[0].asr_route, "dicow")
        self.assertEqual(
            [(cue.text, cue.speaker, cue.kind) for cue in result],
            [("corrected", "A", "conditioned_speech")],
        )

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
        self.assertEqual(result[0].speakers, ("A", "B"))
        self.assertEqual((result[1].start, result[1].end), (8.5, 11.0))

    def test_anonymous_labels_resolved_to_same_person_do_not_overlap(self):
        diarization = [
            AudioRegion(0, 4, "speech", "A", anonymous_speaker="S0"),
            AudioRegion(2, 5, "speech", "A", anonymous_speaker="S1"),
        ]

        result = _conditioned_windows(
            diarization,
            10,
            AudioAnalysisConfig(overlap_context_seconds=1.0),
        )

        self.assertEqual(result, [])

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
                side_effect=stream_cues([Cue(1, 3, "fixed", "A")]),
            ) as run,
        ):
            first = repair_long_overlaps([], diarization, audio, Path(temp), config)
            second = repair_long_overlaps([], diarization, audio, Path(temp), config)

        self.assertEqual(first, second)
        self.assertEqual(first.cues[0].kind, "conditioned_speech")
        run.assert_called_once()

    def test_transcription_and_reconciliation_accept_corrected_aligned_cues(self):
        diarization = [
            AudioRegion(0, 4, "speech", "A", anonymous_speaker="S0"),
            AudioRegion(2, 5, "speech", "B", anonymous_speaker="S1"),
        ]
        baseline = [Cue(1.7, 2.8, "Qwen baseline", "A")]
        audio = SimpleNamespace(duration=10)
        config = AudioAnalysisConfig(overlap_context_seconds=0.5)
        with (
            tempfile.TemporaryDirectory() as temp,
            patch(
                "subtitle_pipeline.conditioned_asr._run_dicow",
                side_effect=stream_cues([Cue(1.5, 3.0, "raw DiCoW", "A")]),
            ),
        ):
            transcription = transcribe_long_overlaps(
                diarization, audio, Path(temp), config
            )

        corrected = [Cue(1.6, 2.9, "corrected DiCoW", "A", "conditioned_speech")]
        result = reconcile_long_overlaps(baseline, corrected, transcription.windows)

        self.assertEqual([cue.text for cue in transcription.cues], ["raw DiCoW"])
        self.assertEqual([cue.text for cue in result.cues], ["corrected DiCoW"])
        self.assertEqual(result.evidence[0]["dicow"][0]["text"], "corrected DiCoW")

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
                side_effect=stream_cues([Cue(1.5, 3.0, "googlegoogle", "A")]),
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
                side_effect=stream_cues([
                    Cue(1.5, 3.0, "DiCoW A", "A"),
                    Cue(2.0, 4.0, "私は" * 100, "B"),
                ]),
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

    def test_long_overlap_windows_cover_the_timeline_with_clipped_turns(self):
        for diarization in (
            [AudioRegion(0, 95, "speech", "A"), AudioRegion(0, 95, "speech", "B")],
            [AudioRegion(0, 50, "speech", "A")]
            + [AudioRegion(start, start + 1, "speech", "B") for start in range(1, 50, 5)],
        ):
            with self.subTest(diarization=diarization):
                windows = _conditioned_windows(diarization, 100, AudioAnalysisConfig())
                self.assertGreater(len(windows), 1)
                self.assertTrue(all(0 < window.end - window.start <= 30 for window in windows))
                self.assertEqual(windows[0].start, 0)
                self.assertEqual(windows[-1].end, 97 if len(diarization) == 2 else 49)
                self.assertTrue(all(left.end == right.start for left, right in zip(windows, windows[1:])))
                for window in windows:
                    self.assertTrue(all(window.start <= turn.start < turn.end <= window.end for turn in window.turns))
                    self.assertEqual(set(window.speakers), {turn.speaker for turn in window.turns})

    def test_partial_repetition_keeps_uncovered_baseline_from_same_speaker(self):
        diarization = [AudioRegion(0, 20, "speech", "A"), AudioRegion(5, 15, "speech", "B")]
        baseline = [Cue(3.5, 4.5, "replace A", "A"), Cue(10, 12, "keep A", "A")]
        with tempfile.TemporaryDirectory() as temp, patch(
            "subtitle_pipeline.conditioned_asr._run_dicow",
            side_effect=stream_cues([Cue(3, 5, "clean A", "A"), Cue(10, 12, "私は" * 100, "A")]),
        ):
            result = repair_long_overlaps(
                baseline, diarization, SimpleNamespace(duration=20), Path(temp), AudioAnalysisConfig()
            )
        self.assertEqual([cue.text for cue in result.cues], ["clean A", "keep A"])

    def test_partial_time_coverage_and_empty_repair_keep_baseline(self):
        turns = (AudioRegion(0, 10, "speech", "A"), AudioRegion(0, 10, "speech", "B"))
        window = ConditionedWindow(0, 10, ("A", "B"), turns)
        baseline = [Cue(1, 4, "whole phrase", "A"), Cue(5, 6, "unknown")]
        for repaired in ([], [Cue(1, 2, "partial A", "A")]):
            with self.subTest(repaired=repaired):
                result = _replace_windows(baseline, repaired, [window])
                self.assertIn(baseline[0], result)
                self.assertIn(baseline[1], result)

    def test_adjacent_repair_units_cover_baseline_without_duplicates(self):
        window = ConditionedWindow(0, 10, ("A",), (AudioRegion(0, 10, "speech", "A"),))
        baseline = [Cue(1, 3, "baseline", "A")]
        repaired = [Cue(1, 2, "first", "A"), Cue(2, 3, "second", "A")]
        self.assertEqual(_replace_windows(baseline, repaired, [window]), repaired)

    def test_batches_missing_windows_and_resumes_after_later_batch_failure(self):
        diarization = [
            region
            for start in range(0, 60, 10)
            for region in (AudioRegion(start, start + 3, "speech", "A"), AudioRegion(start + 1, start + 4, "speech", "B"))
        ]
        audio = SimpleNamespace(duration=60, descriptor=SimpleNamespace(as_dict=lambda: {}))
        popen = subprocess.Popen
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            worker = directory / "worker.py"
            failure = directory / "fail"
            failure.touch()
            worker.write_text('''import json, pathlib, subprocess, sys, time
request = json.loads(sys.stdin.read())
assert request["batch_size"] == 2
log = pathlib.Path(__file__).with_name("requests.json")
previous = json.loads(log.read_text()) if log.exists() else []
log.write_text(json.dumps(previous + [[window["start"] for window in request["windows"]]]))
for index, window in enumerate(request["windows"]):
    if index == 2 and pathlib.Path(__file__).with_name("fail").exists():
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        pathlib.Path(__file__).with_name("child.pid").write_text(str(child.pid))
        print(json.dumps({"error": "worker interrupted"}), flush=True)
        time.sleep(60)
    cues = [{"start": window["start"], "end": window["end"], "text": "speech", "speaker": speaker} for speaker in window["speakers"]]
    print(json.dumps({"window_index": index, "cues": cues}), flush=True)
print(json.dumps({"complete": True}), flush=True)
''', encoding="utf-8")
            config = AudioAnalysisConfig(conditioned_asr_batch_size=2, conditioned_asr_worker_project=str(directory))

            def start_process(_command, **kwargs):
                return popen([sys.executable, str(worker)], **kwargs)

            with patch("subtitle_pipeline.conditioned_asr.shutil.which", return_value="uv"), patch(
                "subtitle_pipeline.conditioned_asr.subprocess.Popen", side_effect=start_process
            ) as run:
                with self.assertRaisesRegex(RuntimeError, "worker interrupted"):
                    transcribe_long_overlaps(diarization, audio, directory, config)
            run.assert_called_once()
            self.assertTrue(run.call_args.kwargs["start_new_session"])
            child = int((directory / "child.pid").read_text())
            child_status = Path(f"/proc/{child}/stat")
            for _ in range(100):
                if not child_status.exists() or child_status.read_text().split()[2] == "Z":
                    break
                time.sleep(0.01)
            else:
                self.fail("DiCoW worker left its child running after failure")
            stage = CacheStore(directory / "cache.sqlite3").existing("conditioned_asr")
            self.assertIsNotNone(stage.get("0"))
            self.assertIsNotNone(stage.get("1"))
            self.assertIsNone(stage.get("2"))
            failure.unlink()
            with patch("subtitle_pipeline.conditioned_asr.shutil.which", return_value="uv"), patch(
                "subtitle_pipeline.conditioned_asr.subprocess.Popen", side_effect=start_process
            ) as resumed:
                result = transcribe_long_overlaps(diarization, audio, directory, config)
                repeated = transcribe_long_overlaps(diarization, audio, directory, config)
            resumed.assert_called_once()
            requests = json.loads((directory / "requests.json").read_text())
            self.assertEqual([len(values) for values in requests], [6, 4])
            self.assertEqual(requests[1], requests[0][2:])
            self.assertEqual(len(result.cues), 12)
            self.assertEqual(repeated, result)


if __name__ == "__main__":
    unittest.main()
