import importlib.util
import unittest
from pathlib import Path

import torch

_WORKER_PATH = Path(__file__).parents[1] / "tools" / "dicow" / "worker.py"
_SPEC = importlib.util.spec_from_file_location("dicow_worker", _WORKER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
worker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(worker)


class DiCoWWorkerTests(unittest.TestCase):
    def test_stno_mask_marks_target_non_target_and_overlap(self):
        masks = worker._diarization_masks(
            ["S0", "S1"],
            [
                {"start": 0.0, "end": 2.0, "speaker": "S0"},
                {"start": 1.0, "end": 3.0, "speaker": "S1"},
            ],
            0.0,
            150,
        )

        self.assertEqual(tuple(masks.shape), (2, 4, 150))
        self.assertTrue(torch.all(masks[0, 1, :50] == 1))
        self.assertTrue(torch.all(masks[0, 3, 50:100] == 1))
        self.assertTrue(torch.all(masks[0, 2, 100:] == 1))

    def test_timestamp_pairs_are_restored_to_video_time(self):
        class Tokenizer:
            def batch_decode(self, *_args, **_kwargs):
                return ["<|0.20|>こんにちは<|1.10|>"]

        result = worker._decode_segments(
            Tokenizer(), [[1]], ["S0"], offset=10.0, duration=2.0
        )

        self.assertEqual(
            result,
            [
                {
                    "start": 10.2,
                    "end": 11.1,
                    "speaker": "S0",
                    "text": "こんにちは",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
