import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from subtitle_pipeline.config import UploadConfig
from subtitle_pipeline.upload import (
    BiliupCommandError,
    _bilibili_failure_code,
    _prepare_description,
    _record_upload_cooldown,
    _retry_after_from_output,
    _truncate_utf16,
    _utf16_units,
    _wait_for_upload_cooldown,
    upload_to_bilibili,
)


class UploadTests(unittest.TestCase):
    def test_truncates_description_by_utf16_units_without_splitting_surrogate_pair(self):
        value = "a" * 1999 + "🎶" + "tail"
        result = _truncate_utf16(value, 2000)
        self.assertEqual(result, "a" * 1999)
        self.assertEqual(len(result.encode("utf-16-le")) // 2, 1999)

    def test_prepares_description_at_paragraph_boundary_and_preserves_suffix(self):
        result = _prepare_description(
            "first paragraph\n\n" + "x" * 80 + "\n\nlast paragraph",
            suffix="generated subtitle",
            max_chars=70,
        )
        self.assertEqual(result, "first paragraph\n\ngenerated subtitle")
        self.assertLessEqual(len(result), 70)
        self.assertLessEqual(_utf16_units(result), 70)

    def test_prepares_description_with_unicode_under_both_limits(self):
        result = _prepare_description(
            "字幕🎶" * 100,
            suffix="固定说明",
            max_chars=80,
        )
        self.assertTrue(result.endswith("固定说明"))
        self.assertLessEqual(len(result), 80)
        self.assertLessEqual(_utf16_units(result), 80)

    def test_builds_repost_command_without_shell_interpolation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cookie = root / "cookies.json"
            cookie.write_text("{}", encoding="utf-8")
            video = root / "video.mp4"
            config = UploadConfig(cookie_file=str(cookie), tags=["中字", "科技"])
            with patch(
                "subtitle_pipeline.upload.require_command", return_value="/bin/biliup"
            ), patch("subtitle_pipeline.upload._wait_for_upload_cooldown"), patch(
                "subtitle_pipeline.upload._record_upload_cooldown"
            ), patch("subtitle_pipeline.upload._run_biliup") as run:
                upload_to_bilibili(
                    video,
                    title="A title",
                    description="Description",
                    source_url="https://youtube.test/watch?v=1",
                    tags=["中字", "自动生成"],
                    config=config,
                )
            command = run.call_args.args[0]
            self.assertEqual(command[:4], ["/bin/biliup", "--user-cookie", str(cookie), "upload"])
            self.assertIn("https://youtube.test/watch?v=1", command)
            self.assertIn("中字,自动生成", command)
            description = command[command.index("--desc") + 1]
            self.assertEqual(description, "Description")
            self.assertEqual(command[-1], str(video))

    def test_retries_rate_limited_complete_upload_with_configured_delays(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cookie = root / "cookies.json"
            cookie.write_text("{}", encoding="utf-8")
            config = UploadConfig(
                cookie_file=str(cookie),
                rate_limit_retry_delays_seconds=[2, 5],
                throttle_state_file=str(root / "throttle.json"),
                pause_marker_file=str(root / "paused.json"),
            )
            failures = [
                BiliupCommandError(1, '{"code":406,"message":"too fast"}'),
                BiliupCommandError(1, "HTTP 429 Retry-After: 7"),
                "success",
            ]
            with patch(
                "subtitle_pipeline.upload.require_command", return_value="/bin/biliup"
            ), patch("subtitle_pipeline.upload._run_biliup", side_effect=failures) as run, patch(
                "subtitle_pipeline.upload.time.sleep"
            ) as sleep, patch("subtitle_pipeline.upload._record_upload_cooldown"):
                upload_to_bilibili(
                    root / "video.mp4",
                    title="title",
                    description="description",
                    source_url="https://youtube.test/video",
                    tags=["中字"],
                    config=config,
                )
            self.assertEqual(run.call_count, 3)
            self.assertEqual([item.args[0] for item in sleep.call_args_list], [2, 7])

    def test_412_writes_pause_marker_and_aborts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cookie = root / "cookies.json"
            marker = root / "paused.json"
            cookie.write_text("{}", encoding="utf-8")
            config = UploadConfig(
                cookie_file=str(cookie),
                throttle_state_file=str(root / "throttle.json"),
                pause_marker_file=str(marker),
            )
            error = BiliupCommandError(1, '{"code":412,"message":"risk"}')
            with patch(
                "subtitle_pipeline.upload.require_command", return_value="/bin/biliup"
            ), patch("subtitle_pipeline.upload._run_biliup", side_effect=error):
                with self.assertRaisesRegex(RuntimeError, "queue paused"):
                    upload_to_bilibili(
                        root / "video.mp4",
                        title="title",
                        description="description",
                        source_url="https://youtube.test/video",
                        tags=["中字"],
                        config=config,
                    )
            payload = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(payload["code"], 412)

    def test_cooldown_state_delays_the_next_process(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "throttle.json"
            config = UploadConfig(
                cooldown_min_seconds=60,
                cooldown_max_seconds=120,
                throttle_state_file=str(path),
            )
            with patch("subtitle_pipeline.upload.random.uniform", return_value=75), patch(
                "subtitle_pipeline.upload.time.time", return_value=1000
            ):
                _record_upload_cooldown(config)
            with patch("subtitle_pipeline.upload.time.time", return_value=1020), patch(
                "subtitle_pipeline.upload.time.sleep"
            ) as sleep:
                _wait_for_upload_cooldown(path)
            sleep.assert_called_once_with(55)

    def test_classifies_rate_limit_and_retry_after_output(self):
        self.assertEqual(_bilibili_failure_code("{'code': 406}"), 406)
        self.assertEqual(_bilibili_failure_code("HTTP status 429"), 429)
        self.assertEqual(_bilibili_failure_code('{"code":412}'), 412)
        self.assertIsNone(_bilibili_failure_code('{"code":500}'))
        self.assertEqual(_retry_after_from_output("Retry-After: 12.5"), 12.5)
        self.assertEqual(_retry_after_from_output('"retry_after": "7"'), 7)


if __name__ == "__main__":
    unittest.main()
