import unittest

import numpy as np

from subtitle_pipeline.moss_diarization import (
    _decode_segments,
    _decode_transcripts,
    _window_ranges,
)


class MossDiarizationTests(unittest.TestCase):
    def test_short_audio_uses_one_window(self):
        samples = np.zeros(60 * 16000, dtype=np.float32)

        self.assertEqual(
            _window_ranges(
                samples,
                16000,
                target_seconds=4800,
                maximum_seconds=5400,
                search_seconds=300,
            ),
            [(0.0, 60.0)],
        )

    def test_long_audio_splits_near_quiet_region_below_ninety_minutes(self):
        sample_rate = 10
        samples = np.ones(6000 * sample_rate, dtype=np.float32)
        samples[4700 * sample_rate : 4701 * sample_rate] = 0

        windows = _window_ranges(
            samples,
            sample_rate,
            target_seconds=4800,
            maximum_seconds=5400,
            search_seconds=300,
        )

        self.assertEqual(len(windows), 2)
        self.assertAlmostEqual(windows[0][1], 4700.25)
        self.assertTrue(all(end - start <= 5400 for start, end in windows))
        self.assertEqual(windows[-1][1], 6000.0)

    def test_decodes_window_scoped_speaker_labels_and_offsets(self):
        segments = _decode_segments(
            [
                {
                    "start": 4801.0,
                    "end": 4802.5,
                    "speaker": "MOSS_W001_S01",
                    "text": "こんにちは",
                },
                {
                    "start": 1.0,
                    "end": 2.0,
                    "speaker": "MOSS_W000_S01",
                    "text": "はい",
                },
            ],
            [(0.0, 4800.0), (4800.0, 6000.0)],
        )

        self.assertEqual(
            [item.speaker for item in segments],
            [
                "MOSS_W000_S01",
                "MOSS_W001_S01",
            ],
        )
        self.assertEqual([item.text for item in segments], ["はい", "こんにちは"])

    def test_rejects_segment_outside_its_window(self):
        with self.assertRaisesRegex(RuntimeError, "out-of-range"):
            _decode_segments(
                [
                    {
                        "start": 10.0,
                        "end": 11.0,
                        "speaker": "MOSS_W001_S01",
                        "text": "wrong window",
                    }
                ],
                [(0.0, 20.0), (20.0, 40.0)],
            )

    def test_validates_one_audit_transcript_per_window(self):
        windows = [(0.0, 20.0), (20.0, 40.0)]
        transcripts = _decode_transcripts(
            [
                {"window": 0, "start": 0.0, "end": 20.0, "raw": "first"},
                {"window": 1, "start": 20.0, "end": 40.0, "raw": "second"},
            ],
            windows,
        )

        self.assertEqual([item["raw"] for item in transcripts], ["first", "second"])

        with self.assertRaisesRegex(RuntimeError, "wrong number"):
            _decode_transcripts(transcripts[:1], windows)


if __name__ == "__main__":
    unittest.main()
