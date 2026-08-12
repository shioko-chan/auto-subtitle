import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from subtitle_pipeline.asr import (
    _analysis_regions,
    _record_timeline_is_healthy,
    _remove_text_overlap,
    _repetition_hallucination,
    _restore_punctuation,
    _result_to_cues,
    _song_windows,
    _transcribe_ambiguous_range,
    _valid_cached_record,
    transcribe_with_qwen,
)
from subtitle_pipeline.audio_analysis import AudioAnalysis, AudioRegion
from subtitle_pipeline.config import ASRConfig


class QwenASRTests(unittest.TestCase):
    def test_groups_fragments_from_one_separated_speaker_track(self):
        result = _analysis_regions(
            AudioAnalysis(
                speech=[
                    AudioRegion(
                        10.0, 11.0, "speech", "A", overlap=True,
                        source_path="a.wav", source_offset=9.0,
                    ),
                    AudioRegion(
                        12.0, 14.0, "speech", "A", overlap=True,
                        source_path="a.wav", source_offset=9.0,
                    ),
                    AudioRegion(
                        10.5, 13.0, "speech", "B", overlap=True,
                        source_path="b.wav", source_offset=9.0,
                    ),
                ],
                singing=[],
            )
        )

        self.assertEqual(len(result), 2)
        self.assertEqual((result[0].start, result[0].end), (10.0, 14.0))
        self.assertEqual(result[0].source_offset, 9.0)

    def test_singing_regions_are_subtracted_from_speech_timing(self):
        result = _analysis_regions(
            AudioAnalysis(
                speech=[AudioRegion(10, 30, "speech", "A")],
                singing=[AudioRegion(15, 25, "singing", "A")],
            )
        )

        self.assertEqual(
            [(item.start, item.end, item.kind) for item in result],
            [(10, 15, "speech"), (15, 25, "singing"), (25, 30, "speech")],
        )

    def test_ambiguous_regions_are_subtracted_and_routed_once(self):
        result = _analysis_regions(
            AudioAnalysis(
                speech=[AudioRegion(0, 20, "speech", "A")],
                singing=[],
                ambiguous=[AudioRegion(5, 15, "ambiguous", "A")],
            )
        )
        self.assertEqual(
            [(item.start, item.end, item.kind) for item in result],
            [(0, 5, "speech"), (5, 15, "ambiguous"), (15, 20, "speech")],
        )

    def test_rejects_collapsed_ambiguous_forced_alignment(self):
        record = {
            "text": "这是足够长的识别结果但是整个时间轴已经完全坍缩了",
            "cues": [
                {"start": 10.0, "end": 10.1, "text": str(index)}
                for index in range(25)
            ],
        }
        self.assertFalse(
            _record_timeline_is_healthy(
                record, AudioRegion(10, 20, "ambiguous")
            )
        )

    def test_ambiguous_region_runs_both_routes_and_keeps_healthy_alignment(self):
        speech = {
            "text": "これは十分に長く正常に整列された発話です",
            "cues": [{"start": 10.0, "end": 18.0, "text": "正常な発話"}],
        }
        singing = {
            "text": "歌声候補",
            "cues": [{"start": 10.0, "end": 20.0, "text": "歌声候補"}],
        }
        with (
            patch("subtitle_pipeline.asr._media_duration", return_value=30.0),
            patch(
                "subtitle_pipeline.asr._transcribe_range", return_value=speech
            ) as speech_route,
            patch(
                "subtitle_pipeline.asr._transcribe_song_range",
                return_value=singing,
            ) as singing_route,
        ):
            result = _transcribe_ambiguous_range(
                object(),
                Path("source.mp4"),
                Path("chunks"),
                ASRConfig(),
                AudioRegion(10, 20, "ambiguous", "A"),
                index=0,
                window_seconds=12,
                overlap_seconds=2,
            )
        self.assertEqual(result["ambiguous_route"], "speech")
        speech_route.assert_called_once()
        singing_route.assert_called_once()

    def test_ambiguous_region_falls_back_when_alignment_collapses(self):
        speech = {
            "text": "これは十分に長いのに時間軸が完全に壊れた発話です",
            "cues": [
                {"start": 10.0, "end": 10.1, "text": str(index)}
                for index in range(25)
            ],
        }
        singing = {
            "text": "歌声候補",
            "cues": [{"start": 10.0, "end": 20.0, "text": "歌声候補"}],
        }
        with (
            patch("subtitle_pipeline.asr._media_duration", return_value=30.0),
            patch("subtitle_pipeline.asr._transcribe_range", return_value=speech),
            patch(
                "subtitle_pipeline.asr._transcribe_song_range",
                return_value=singing,
            ),
        ):
            result = _transcribe_ambiguous_range(
                object(),
                Path("source.mp4"),
                Path("chunks"),
                ASRConfig(),
                AudioRegion(10, 20, "ambiguous", "A"),
                index=0,
                window_seconds=12,
                overlap_seconds=2,
            )
        self.assertEqual(result["ambiguous_route"], "singing")

    def test_aligner_cues_are_clamped_to_owned_analysis_region(self):
        result = SimpleNamespace(
            text="前後",
            time_stamps=SimpleNamespace(
                items=[
                    SimpleNamespace(text="前", start_time=8.0, end_time=10.2),
                    SimpleNamespace(text="後", start_time=10.2, end_time=11.0),
                ]
            ),
        )

        cues = _result_to_cues(
            result,
            offset=0,
            keep_start=5,
            keep_end=10,
            final_chunk=True,
        )

        self.assertEqual(
            [(cue.start, cue.end, cue.text) for cue in cues],
            [(8.0, 10, "前")],
        )

    def test_singing_windows_overlap_without_extending_past_region(self):
        self.assertEqual(
            _song_windows(5, 29, 12, 2),
            [(5, 17), (15, 27), (25, 29)],
        )

    def test_removes_repeated_text_from_overlapping_song_window(self):
        self.assertEqual(
            _remove_text_overlap("君とここで歌う", "ここで歌う明日へ"),
            "明日へ",
        )

    def test_detects_repeated_asr_generation_loop(self):
        phrase = "私が食べてるのでちょっとこっちに移動します"
        repetition = _repetition_hallucination(phrase * 10)
        self.assertIsNotNone(repetition)
        assert repetition is not None
        self.assertGreaterEqual(repetition[1], 4)

    def test_does_not_flag_short_natural_repetition(self):
        self.assertIsNone(_repetition_hallucination("はいはいはいはい、大丈夫です"))

    def test_restores_punctuation_between_japanese_aligner_tokens(self):
        self.assertEqual(
            _restore_punctuation(
                "こんにちは、今日はライブです。",
                ["こんにちは", "今日", "は", "ライブ", "です"],
            ),
            ["こんにちは、", "今日", "は", "ライブ", "です。"],
        )

    def test_caches_each_completed_audio_chunk_and_reuses_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "source.mp4"
            destination = root / "source.qwen3-asr.srt"
            video.write_bytes(b"video")
            config = ASRConfig(chunk_seconds=10, chunk_context_seconds=1)
            item = SimpleNamespace(text="一", start_time=1, end_time=2)
            result = SimpleNamespace(
                language="Japanese",
                text="一。",
                time_stamps=SimpleNamespace(items=[item]),
            )
            model = SimpleNamespace(transcribe=lambda **_kwargs: [result])

            with patch(
                "subtitle_pipeline.asr._media_duration", return_value=15
            ), patch(
                "subtitle_pipeline.asr._load_qwen_model", return_value=model
            ) as load_model, patch(
                "subtitle_pipeline.asr._extract_audio_chunk"
            ):
                transcribe_with_qwen(video, destination, config)
                transcribe_with_qwen(video, destination, config)

            load_model.assert_called_once()
            self.assertTrue((root / "asr-cache.json").is_file())
            self.assertEqual(destination.read_text(encoding="utf-8").count("一。"), 2)

    def test_rejects_zero_duration_cached_cue(self):
        self.assertFalse(
            _valid_cached_record(
                {"cues": [{"start": 1.0, "end": 1.0, "text": "字幕"}]}
            )
        )

    def test_rejects_cached_chunk_with_repeated_generation_loop(self):
        phrase = "私が食べてるのでちょっとこっちに移動します"
        self.assertFalse(
            _valid_cached_record(
                {
                    "text": phrase * 10,
                    "cues": [{"start": 1.0, "end": 2.0, "text": "字幕"}],
                }
            )
        )

    def test_retries_repetition_with_two_shorter_chunks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "source.mp4"
            destination = root / "source.qwen3-asr.srt"
            video.write_bytes(b"video")
            config = ASRConfig(chunk_seconds=40, chunk_context_seconds=1)
            phrase = "私が食べてるのでちょっとこっちに移動します"
            looping = SimpleNamespace(
                language="Japanese",
                text=phrase * 10,
                time_stamps=SimpleNamespace(items=[]),
            )
            left = SimpleNamespace(
                language="Japanese",
                text="左。",
                time_stamps=SimpleNamespace(
                    items=[SimpleNamespace(text="左", start_time=5, end_time=6)]
                ),
            )
            right = SimpleNamespace(
                language="Japanese",
                text="右。",
                time_stamps=SimpleNamespace(
                    items=[SimpleNamespace(text="右", start_time=5, end_time=6)]
                ),
            )

            class Model:
                def __init__(self):
                    self.results = iter([looping, left, right])
                    self.calls = 0

                def transcribe(self, **_kwargs):
                    self.calls += 1
                    return [next(self.results)]

            model = Model()
            with patch(
                "subtitle_pipeline.asr._media_duration", return_value=40
            ), patch(
                "subtitle_pipeline.asr._load_qwen_model", return_value=model
            ), patch("subtitle_pipeline.asr._extract_audio_chunk"):
                transcribe_with_qwen(video, destination, config)

            self.assertEqual(model.calls, 3)
            self.assertIn("左。", destination.read_text(encoding="utf-8"))
            self.assertIn("右。", destination.read_text(encoding="utf-8"))
            cache = __import__("json").loads(
                (root / "asr-cache.json").read_text(encoding="utf-8")
            )
            self.assertTrue(cache["chunks"]["0"]["recovered_from_repetition"])


if __name__ == "__main__":
    unittest.main()
