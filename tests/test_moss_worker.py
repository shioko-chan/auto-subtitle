import importlib.util
import unittest
from pathlib import Path

_WORKER_PATH = Path(__file__).parents[1] / "tools" / "moss_transcribe" / "worker.py"
_SPEC = importlib.util.spec_from_file_location("moss_transcribe_worker", _WORKER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_WORKER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_WORKER)


class MossWorkerTests(unittest.TestCase):
    def test_accepts_structured_segment_with_no_speech_text(self):
        self.assertEqual(_WORKER._parse_transcript("[41.12][S01][44.48]", 0, 0), [])

    def test_rejects_nonempty_unstructured_response(self):
        with self.assertRaisesRegex(RuntimeError, "unparseable"):
            _WORKER._parse_transcript("无法识别语音", 0, 0)


if __name__ == "__main__":
    unittest.main()
