import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from subtitle_pipeline.telemetry import pipeline_metrics, stage_metrics


class TelemetryTests(unittest.TestCase):
    def test_pipeline_report_aggregates_repeated_and_threaded_stages(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "performance.json"
            with pipeline_metrics(path):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [
                        executor.submit(self._record_stage, "parallel")
                        for _ in range(2)
                    ]
                    for future in futures:
                        future.result()
                with stage_metrics("serial"):
                    pass
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "completed")
        self.assertEqual(len(payload["stages"]), 3)
        self.assertEqual(payload["summary"]["parallel"]["calls"], 2)
        self.assertEqual(payload["summary"]["parallel"]["completed"], 2)
        self.assertEqual(payload["summary"]["parallel"]["failed"], 0)
        self.assertIn("total_elapsed_seconds", payload["summary"]["parallel"])
        self.assertEqual(payload["summary"]["serial"]["calls"], 1)

    def test_failed_pipeline_still_writes_completed_work_and_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "performance.json"
            with (
                self.assertRaisesRegex(ValueError, "broken"),
                pipeline_metrics(path),
            ):
                with stage_metrics("completed"):
                    pass
                with stage_metrics("failed"):
                    raise ValueError("broken")
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "failed")
        self.assertIn("ValueError: broken", payload["error"])
        self.assertEqual(payload["summary"]["completed"]["completed"], 1)
        self.assertEqual(payload["summary"]["failed"]["failed"], 1)

    @staticmethod
    def _record_stage(name: str) -> None:
        with stage_metrics(name):
            pass


if __name__ == "__main__":
    unittest.main()
