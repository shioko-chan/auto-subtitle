import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

_WORKER_PATH = Path(__file__).parents[1] / "tools" / "dicow" / "worker.py"
_SPEC = importlib.util.spec_from_file_location("dicow_worker", _WORKER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
worker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(worker)


class DiCoWWorkerTests(unittest.TestCase):
    def test_all_batches_share_one_model_and_emit_before_later_failure(self):
        for fail_later in (False, True):
            with self.subTest(fail_later=fail_later):
                windows = [{"start": start, "end": start + 2, "speakers": ["A"], "turns": []} for start in range(0, 24, 4)]
                request = {
                    "model": "local-test", "revision": "test", "device": "cuda", "batch_size": 2,
                    "audio": {"shared_memory": "test", "sample_rate": 16000, "shape": [24 * 16000], "dtype": "float32"},
                    "windows": windows,
                }
                model = Mock()
                model.to.return_value = model
                loader = Mock(return_value=model)
                memory = SimpleNamespace(buf=bytearray(24 * 16000 * 4), _name="test", close=Mock())
                modules = {
                    "transformers": SimpleNamespace(
                        AutoModelForSpeechSeq2Seq=SimpleNamespace(from_pretrained=loader),
                        AutoFeatureExtractor=SimpleNamespace(from_pretrained=Mock()),
                        AutoTokenizer=SimpleNamespace(from_pretrained=Mock()),
                    ),
                    "transformers.dynamic_module_utils": SimpleNamespace(get_class_from_dynamic_module=Mock()),
                    "transformers.utils": SimpleNamespace(logging=SimpleNamespace(set_verbosity_error=Mock())),
                }

                def transcribe(_model, _features, _tokenizer, _audio, batch, _language):
                    if fail_later and batch[0]["start"] >= 8:
                        raise RuntimeError("later batch failed")
                    return [{"start": window["start"], "end": window["end"], "text": "speech", "speaker": "A"} for window in batch]

                emitted = []
                with (
                    patch.dict("sys.modules", modules),
                    patch("torch.cuda.is_available", return_value=True),
                    patch.object(worker.shared_memory, "SharedMemory", return_value=memory),
                    patch.object(worker.resource_tracker, "unregister"),
                    patch.object(worker, "_prepare_remote_code"),
                    patch.object(worker, "_case_mapping"),
                    patch.object(worker, "_transcribe_windows", side_effect=transcribe) as run,
                ):
                    if fail_later:
                        with self.assertRaisesRegex(RuntimeError, "later batch failed"):
                            worker._handle(request, emitted.append)
                    else:
                        worker._handle(request, emitted.append)
                loader.assert_called_once()
                self.assertEqual([len(call.args[4]) for call in run.call_args_list], [2, 2] if fail_later else [2, 2, 2])
                self.assertEqual([result["window_index"] for result in emitted], list(range(2 if fail_later else 6)))
                memory.close.assert_called_once()

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
