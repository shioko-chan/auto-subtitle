import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from subtitle_pipeline.config import DownloadConfig, RenderConfig
from subtitle_pipeline.media import (
    RenderCue,
    _adaptive_font_size,
    _adaptive_horizontal_margin,
    _layout_subtitle_cues,
    _video_dimensions,
    _write_ass,
    download_youtube,
)
from subtitle_pipeline.subtitles import Cue


class MediaDownloadTests(unittest.TestCase):
    def test_downloads_video_and_metadata_without_youtube_subtitles(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)

            def fake_run(_command):
                (directory / "source.info.json").write_text(
                    json.dumps({"title": "Video", "language": "ja"}),
                    encoding="utf-8",
                )
                (directory / "source.mp4").write_bytes(b"video")

            def find_runtime(name):
                return "/usr/bin/node" if name == "node" else None

            with patch(
                "subtitle_pipeline.media.require_command", return_value="/venv/bin/yt-dlp"
            ), patch("subtitle_pipeline.media.run", side_effect=fake_run) as run, patch(
                "subtitle_pipeline.media.shutil.which", side_effect=find_runtime
            ):
                result = download_youtube(
                    "https://www.youtube.com/watch?v=test",
                    directory,
                    DownloadConfig(),
                )

            self.assertEqual(result.video, directory / "source.mp4")
            self.assertEqual(result.metadata["title"], "Video")
            run.assert_called_once()
            video_command = run.call_args.args[0]
            self.assertNotIn("--write-subs", video_command)
            self.assertNotIn("--write-auto-subs", video_command)
            self.assertIn("node:/usr/bin/node", video_command)

    def test_existing_subtitle_is_ignored_by_qwen_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)

            def fake_run(_command):
                (directory / "source.info.json").write_text(
                    '{"language":"en"}', encoding="utf-8"
                )
                (directory / "source.mp4").write_bytes(b"video")
                (directory / "source.en.srt").write_text("subtitle", encoding="utf-8")

            with patch(
                "subtitle_pipeline.media.require_command", return_value="yt-dlp"
            ), patch("subtitle_pipeline.media.run", side_effect=fake_run) as run:
                result = download_youtube(
                    "https://youtu.be/test",
                    directory,
                    DownloadConfig(js_runtime=None),
                )

            self.assertEqual(result.video, directory / "source.mp4")
            run.assert_called_once()


class SubtitleRenderTests(unittest.TestCase):
    def test_font_size_uses_orientation_specific_short_edge_ratio(self):
        config = RenderConfig(max_font_size=100)
        self.assertEqual(_adaptive_font_size(1920, 1080, config), 71)
        self.assertEqual(_adaptive_font_size(1080, 1920, config), 83)

    def test_horizontal_margin_uses_portrait_ratio(self):
        config = RenderConfig()
        self.assertEqual(_adaptive_horizontal_margin(1920, 1080, config), 144)
        self.assertEqual(_adaptive_horizontal_margin(1080, 1920, config), 27)

    def test_allows_slightly_oversized_cue_into_safe_margin(self):
        text = "一二三四五六七八九十一"
        result = _layout_subtitle_cues(
            [Cue(10, 18, text)],
            max_line_units=10,
            hard_max_line_units=12,
        )

        self.assertEqual(result, [RenderCue(10, 18, text)])

    def test_splits_oversized_cue_into_single_line_semantic_events(self):
        text = "一二三四五六七八九十甲乙"
        result = _layout_subtitle_cues(
            [Cue(10, 18, text)],
            max_line_units=10,
            semantic_segments={0: ["一二三四五六", "七八九十甲乙"]},
        )

        self.assertEqual([cue.text for cue in result], ["一二三四五六", "七八九十甲乙"])
        self.assertEqual([(cue.start, cue.end) for cue in result], [(10, 14), (14, 18)])
        self.assertTrue(all("\n" not in cue.text for cue in result))

    def test_rendered_lines_drop_plain_terminal_punctuation(self):
        text = '前半句，后半句。真的吗？太好了！等等...他说：“好的，”'
        segments = [
            "前半句，",
            "后半句。",
            "真的吗？",
            "太好了！",
            "等等...",
            '他说：“好的，”',
        ]
        result = _layout_subtitle_cues(
            [Cue(0, 12, text)],
            max_line_units=20,
            semantic_segments={0: segments},
        )

        self.assertEqual(
            [cue.text for cue in result],
            ["前半句", "后半句", "真的吗？", "太好了！", "等等...", '他说：“好的”'],
        )

    def test_wraps_oversized_cue_into_two_balanced_lines(self):
        result = _layout_subtitle_cues(
            [Cue(0, 2, "一二三四五六，七八九十甲乙")],
            max_line_units=10,
        )

        self.assertEqual(result, [RenderCue(0, 2, "一二三四五六\n七八九十甲乙")])

    def test_rejects_cue_exceeding_two_lines(self):
        with self.assertRaisesRegex(ValueError, "exceeds two lines"):
            _layout_subtitle_cues(
                [Cue(0, 1, "一二三四五六七八九十甲乙丙丁戊己庚辛壬癸子")],
                max_line_units=10,
            )

    def test_ass_uses_video_resolution_and_global_one_pixel_margins(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "subtitle.ass"
            _write_ass(
                [Cue(1, 2, "单行字幕")],
                path,
                width=1080,
                height=1920,
                font_name="Noto Sans CJK SC",
                font_size=48,
                margin_vertical=96,
                outline=2,
            )
            content = path.read_text(encoding="utf-8")

        self.assertIn("PlayResX: 1080", content)
        self.assertIn("PlayResY: 1920", content)
        self.assertIn(
            "Style: Default,Noto Sans CJK SC,48,", content
        )
        self.assertIn(",2,1,1,96,1", content)
        self.assertNotIn(r"\fs", content)
        self.assertIn("单行字幕", content)
        self.assertIn("Default,,0,0,0,,单行字幕", content)

    def test_video_dimensions_respect_rotation_metadata(self):
        response = subprocess.CompletedProcess(
            ["ffprobe"],
            0,
            json.dumps(
                {
                    "streams": [
                        {
                            "width": 1920,
                            "height": 1080,
                            "side_data_list": [{"rotation": -90}],
                        }
                    ]
                }
            ),
            "",
        )
        with patch(
            "subtitle_pipeline.media.require_command", return_value="ffprobe"
        ), patch("subtitle_pipeline.media.subprocess.run", return_value=response):
            self.assertEqual(_video_dimensions(Path("video.mp4")), (1080, 1920))


if __name__ == "__main__":
    unittest.main()
