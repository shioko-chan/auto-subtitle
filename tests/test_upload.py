import json
import select
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from subtitle_pipeline.config import UploadConfig
from subtitle_pipeline.upload import (
    BiliupCommandError,
    UploadNotStartedError,
    _bilibili_failure_code,
    _prepare_description,
    _record_upload_cooldown,
    _retry_after_from_output,
    _submission_ids,
    _truncate_utf16,
    _utf16_units,
    _upload_lock,
    _wait_for_upload_cooldown,
    upload_to_bilibili,
    upload_videos_to_bilibili,
)


class UploadTests(unittest.TestCase):
    BILIUP_SUCCESS = (
        'ResponseData { code: 0, data: Some(Object {"aid": Number(123), '
        '"bvid": String("BV123")}), message: "OK" }'
    )

    def test_extracts_submission_ids_from_biliup_rust_debug_output(self):
        output = (
            "\x1b[32mINFO\x1b[0m ResponseData { code: 0, data: Some(Object "
            '{"aid": Number(117147313377632), "bvid": String("BV1Wy8h6BEo7")}), '
            'message: "OK" }'
        )
        self.assertEqual(_submission_ids(output), (117147313377632, "BV1Wy8h6BEo7"))

    def test_truncates_description_by_utf16_units_without_splitting_surrogate_pair(
        self,
    ):
        value = "a" * 1999 + "🎶" + "tail"
        result = _truncate_utf16(value, 2000)
        self.assertEqual(result, "a" * 1999)
        self.assertEqual(len(result.encode("utf-16-le")) // 2, 1999)

    def test_prepares_header_with_unicode_under_both_limits(self):
        result = _prepare_description("固定说明\n" + "字幕🎶" * 100, max_chars=80)
        self.assertTrue(result.startswith("固定说明"))
        self.assertLessEqual(len(result), 80)
        self.assertLessEqual(_utf16_units(result), 80)

    def test_builds_repost_command_without_shell_interpolation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cookie = root / "cookies.json"
            cookie.write_text("{}", encoding="utf-8")
            video = root / "video.mp4"
            config = UploadConfig(cookie_file=str(cookie), tags=["中字", "科技"],
                                  throttle_state_file=str(root / "throttle.json"),
                                  pause_marker_file=str(root / "paused.json"))
            with (
                patch(
                    "subtitle_pipeline.upload.require_command",
                    return_value="/bin/biliup",
                ),
                patch("subtitle_pipeline.upload._wait_for_upload_cooldown"),
                patch("subtitle_pipeline.upload._record_upload_cooldown"),
                patch(
                    "subtitle_pipeline.upload._run_biliup",
                    return_value=self.BILIUP_SUCCESS,
                ) as run,
            ):
                upload_to_bilibili(
                    video,
                    title="A title",

                    source_url="https://youtube.test/watch?v=1",
                    tags=["中字", "自动生成"],
                    config=config,
                )
            command = run.call_args.args[0]
            self.assertEqual(
                command[:4], ["/bin/biliup", "--user-cookie", str(cookie), "upload"]
            )
            self.assertEqual(command[command.index("--source") + 1], "https://youtube.test/watch?v=1")
            self.assertNotIn("--tid", command)
            self.assertEqual(json.loads(command[command.index("--extra-fields") + 1]), {"tid_v2": 2047})
            self.assertIn("中字,自动生成", command)
            description = command[command.index("--desc") + 1]
            self.assertEqual(description, "")
            self.assertEqual(command[-1], str(video))

    def test_expands_youtube_url_placeholder_in_description_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cookie = root / "cookies.json"
            cookie.write_text("{}", encoding="utf-8")
            source_url = "https://www.youtube.com/watch?v=example"
            config = UploadConfig(
                cookie_file=str(cookie),
                throttle_state_file=str(root / "throttle.json"),
                pause_marker_file=str(root / "paused.json"),
                description_prefix="原视频：{youtube_url}\n字幕说明",
            )
            with (
                patch(
                    "subtitle_pipeline.upload.require_command",
                    return_value="/bin/biliup",
                ),
                patch("subtitle_pipeline.upload._wait_for_upload_cooldown"),
                patch("subtitle_pipeline.upload._record_upload_cooldown"),
                patch(
                    "subtitle_pipeline.upload._run_biliup",
                    return_value=self.BILIUP_SUCCESS,
                ) as run,
            ):
                upload_to_bilibili(
                    root / "video.mp4",
                    title="title",

                    source_url=source_url,
                    tags=["中字"],
                    config=config,
                )
            command = run.call_args.args[0]
            description = command[command.index("--desc") + 1]
            self.assertEqual(
                description,
                f"原视频：{source_url}\n字幕说明",
            )

    def test_builds_one_multi_part_command_in_supplied_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cookie = root / "cookies.json"
            cookie.write_text("{}", encoding="utf-8")
            parts = [root / "001_first.mp4", root / "002_second.mp4"]
            config = UploadConfig(cookie_file=str(cookie),
                                  throttle_state_file=str(root / "throttle.json"),
                                  pause_marker_file=str(root / "paused.json"))
            with (
                patch(
                    "subtitle_pipeline.upload.require_command",
                    return_value="/bin/biliup",
                ),
                patch("subtitle_pipeline.upload._wait_for_upload_cooldown"),
                patch("subtitle_pipeline.upload._record_upload_cooldown"),
                patch(
                    "subtitle_pipeline.upload._run_biliup",
                    return_value=self.BILIUP_SUCCESS,
                ) as run,
            ):
                upload_videos_to_bilibili(
                    parts,
                    title="合集",

                    source_url="https://youtube.test/video",
                    tags=["切片"],
                    config=config,
                )

            command = run.call_args.args[0]
            self.assertEqual(command[-2:], [str(parts[0]), str(parts[1])])

    def test_rejects_empty_multi_part_upload(self):
        with self.assertRaisesRegex(UploadNotStartedError, "at least one video"):
            upload_videos_to_bilibili(
                [],
                title="empty",

                source_url="https://youtube.test/video",
                tags=["切片"],
                config=UploadConfig(),
            )

    def test_rate_limit_retry_and_pause_policy(self):
        cases = [
            ('recover', [BiliupCommandError(1, '{"code":406}'),
                         BiliupCommandError(1, 'HTTP 429 Retry-After: 7'),
                         self.BILIUP_SUCCESS], [2, 7], False, False),
            ('exhausted', [BiliupCommandError(1, 'ResponseData { code: 21566, data: None }')] * 3,
             [2, 5], True, True),
            ('timeout', [BiliupCommandError(1, 'connection timed out')], [], True, False),
            ('success_then_error', [BiliupCommandError(1,
                'ResponseData { code: 0, data: Some(x) } HTTP 429 Retry-After: 7')], [], True, False),
        ]
        for name, responses, waits, fails, paused in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                cookie = root / 'cookies.json'
                cookie.write_text('{}')
                marker = root / 'paused.json'
                config = UploadConfig(cookie_file=str(cookie),
                    rate_limit_retry_delays_seconds=[2, 5],
                    throttle_state_file=str(root / 'throttle.json'),
                    pause_marker_file=str(marker))
                with patch('subtitle_pipeline.upload.require_command', return_value='/bin/biliup'), \
                     patch('subtitle_pipeline.upload._run_biliup', side_effect=responses) as run, \
                     patch('subtitle_pipeline.upload.time.sleep') as sleep, \
                     patch('subtitle_pipeline.upload._record_upload_cooldown'):
                    def upload():
                        return upload_to_bilibili(root / 'video.mp4', title='title',
                            source_url='https://youtube.test/video', tags=['中字'], config=config)
                    if fails:
                        with self.assertRaises(BiliupCommandError):
                            upload()
                    else:
                        self.assertEqual(upload().bvid, 'BV123')
                self.assertEqual(run.call_count, len(responses))
                self.assertEqual([call.args[0] for call in sleep.call_args_list], waits)
                self.assertEqual(marker.exists(), paused)
                if paused:
                    self.assertEqual(json.loads(marker.read_text())['code'], 21566)

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
            with (
                patch(
                    "subtitle_pipeline.upload.require_command",
                    return_value="/bin/biliup",
                ),
                patch("subtitle_pipeline.upload._run_biliup", side_effect=error),
            ):
                with self.assertRaises(BiliupCommandError):
                    upload_to_bilibili(
                        root / "video.mp4",
                        title="title",

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
            with (
                patch("subtitle_pipeline.upload.random.uniform", return_value=75),
                patch("subtitle_pipeline.upload.time.time", return_value=1000),
            ):
                _record_upload_cooldown(config)
            with (
                patch("subtitle_pipeline.upload.time.time", return_value=1020),
                patch("subtitle_pipeline.upload.time.sleep") as sleep,
            ):
                _wait_for_upload_cooldown(path)
            sleep.assert_called_once_with(55)

    def test_classifies_rate_limit_and_retry_after_output(self):
        self.assertEqual(_bilibili_failure_code("{'code': 406}"), 406)
        self.assertEqual(_bilibili_failure_code("HTTP status 429"), 429)
        self.assertEqual(_bilibili_failure_code('{"code":412}'), 412)
        self.assertIsNone(_bilibili_failure_code('{"code":500}'))
        self.assertEqual(_retry_after_from_output("Retry-After: 12.5"), 12.5)
        self.assertEqual(_retry_after_from_output('"retry_after": "7"'), 7)

    def test_pause_created_during_cooldown_or_retry_prevents_submission(self):
        for waiting in ("cooldown", "retry"):
            with self.subTest(waiting=waiting), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                cookie = root / "cookies.json"
                marker = root / "paused.json"
                cookie.write_text("{}")
                config = UploadConfig(cookie_file=str(cookie),
                    throttle_state_file=str(root / "throttle.json"),
                    pause_marker_file=str(marker), rate_limit_retry_delays_seconds=[1])
                wait_target = ("_wait_for_upload_cooldown" if waiting == "cooldown" else "time.sleep")
                with patch("subtitle_pipeline.upload.require_command", return_value="biliup"), \
                     patch("subtitle_pipeline.upload." + wait_target,
                           side_effect=lambda _: marker.write_text("paused")), \
                     patch("subtitle_pipeline.upload._run_biliup",
                           side_effect=BiliupCommandError(1, '{"code":429}')) as submit, \
                     self.assertRaisesRegex(UploadNotStartedError, "uploads are paused"):
                    upload_to_bilibili(root / "video.mp4", title="test", source_url="url",
                                       tags=["test"], config=config)
                self.assertEqual(submit.call_count, 0 if waiting == "cooldown" else 1)

    def test_process_lock_covers_cooldown_submission_and_state_update(self):
        code = textwrap.dedent('''
            import fcntl
            import sys
            from pathlib import Path
            from unittest.mock import patch
            from subtitle_pipeline.config import UploadConfig
            from subtitle_pipeline.upload import upload_to_bilibili, _record_upload_cooldown

            root = Path(sys.argv[1])
            config = UploadConfig(cookie_file=str(root / 'cookies.json'),
                throttle_state_file=str(root / 'throttle.json'),
                pause_marker_file=str(root / 'paused.json'),
                cooldown_min_seconds=60, cooldown_max_seconds=60)
            def locked_event(name):
                with (root / 'throttle.json.lock').open('a+') as handle:
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        print(name, flush=True)
                        return
                raise AssertionError(name + ' ran outside the shared process lock')
            def submit(command):
                locked_event('submit')
                return '{"aid":123,"bvid":"BV123"}'
            def record(config):
                locked_event('record')
                _record_upload_cooldown(config)
            print('ready', flush=True)
            with patch('subtitle_pipeline.upload.require_command', return_value='biliup'), \\
                 patch('subtitle_pipeline.upload._run_biliup', side_effect=submit), \\
                 patch('subtitle_pipeline.upload.time.time', return_value=1020), \\
                 patch('subtitle_pipeline.upload.time.sleep', side_effect=lambda seconds: locked_event('sleep:' + str(seconds))), \\
                 patch('subtitle_pipeline.upload._record_upload_cooldown', side_effect=record):
                upload_to_bilibili(root / 'video.mp4', title='test', source_url='url', tags=['test'], config=config)
        ''')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "cookies.json").write_text("{}")
            state = root / "throttle.json"
            child = None
            try:
                with _upload_lock(state):
                    state.write_text('{"next_allowed_at":1060}')
                    child = subprocess.Popen([sys.executable, "-c", code, str(root)],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    self.assertEqual(child.stdout.readline().strip(), "ready")
                    self.assertFalse(select.select([child.stdout], [], [], 0.2)[0],
                                     "another process entered upload while the lock was held")
                stdout, stderr = child.communicate(timeout=10)
                self.assertEqual(child.returncode, 0, stderr)
                self.assertEqual(stdout.splitlines(), ["sleep:40.0", "submit", "record"])
                self.assertEqual(json.loads(state.read_text())["next_allowed_at"], 1080)
            finally:
                if child is not None and child.poll() is None:
                    child.kill()
                    child.communicate()

    def test_failed_submission_releases_lock_for_another_process(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cookie = root / "cookies.json"
            cookie.write_text("{}")
            state = root / "throttle.json"
            config = UploadConfig(cookie_file=str(cookie), throttle_state_file=str(state),
                                  pause_marker_file=str(root / "paused.json"))
            with patch("subtitle_pipeline.upload.require_command", return_value="biliup"), \
                 patch("subtitle_pipeline.upload._run_biliup",
                       side_effect=BiliupCommandError(1, "transport failed")), \
                 self.assertRaises(BiliupCommandError):
                upload_to_bilibili(root / "video.mp4", title="test", source_url="url",
                                   tags=["test"], config=config)
            code = ("from pathlib import Path; from subtitle_pipeline.upload import _upload_lock; "
                    "import sys\nwith _upload_lock(Path(sys.argv[1])): print('acquired')")
            result = subprocess.run([sys.executable, "-c", code, str(state)],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "acquired")


if __name__ == "__main__":
    unittest.main()
