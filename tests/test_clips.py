import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from subtitle_pipeline.chat_context import YouTubeChatMessage
from subtitle_pipeline.cache import CacheStore
from subtitle_pipeline.cli import main
from subtitle_pipeline.clips import (
    ChatWindow,
    ClipPart,
    ClipsResult,
    _download_chat,
    _merge_speech_parts,
    _render_clip,
    _review_seeds,
    _semantic_seeds,
    _song_seeds,
    _upload_metadata,
    _validated_response_range,
    analyze_chat_windows,
    run_clips,
    select_chat_peaks,
)
from subtitle_pipeline.config import AppConfig, ClipsConfig
from subtitle_pipeline.subtitles import Cue, write_srt
from subtitle_pipeline.upload import BilibiliSubmission


class ClipsTests(unittest.TestCase):
    def _complete_pipeline_stages(self, job):
        store = CacheStore(job / "cache.sqlite3")
        for name in ("translation", "render", "metadata"):
            store.stage(name, lambda: {}).finish({"completed": True})
        return store

    def test_cli_clips_is_independent_from_main_pipeline(self):
        result = ClipsResult(
            Path("work/video123"),
            Path("work/video123/clips/analysis.json"),
            (),
            False,
        )
        with (
            patch("subtitle_pipeline.cli.load_config", return_value=AppConfig()),
            patch("subtitle_pipeline.cli.run_clips", return_value=result) as clips,
            patch("subtitle_pipeline.cli.run_pipeline") as pipeline,
        ):
            exit_code = main(
                [
                    "--config",
                    "config.toml",
                    "clips",
                    "--no-upload",
                    "https://youtu.be/video123",
                ]
            )

        self.assertEqual(exit_code, 0)
        clips.assert_called_once()
        self.assertFalse(clips.call_args.kwargs["upload_override"])
        pipeline.assert_not_called()

    def test_chat_density_counts_reactions_authors_and_paid_events(self):
        messages = [
            YouTubeChatMessage(35, "a", "草"),
            YouTubeChatMessage(36, "b", "888"),
            YouTubeChatMessage(37, "c", "すごい"),
            YouTubeChatMessage(38, "d", "最高"),
            YouTubeChatMessage(39, "e", "wow", "$5"),
        ]

        windows = analyze_chat_windows(messages, duration=90)
        peak = next(window for window in windows if window.start == 30)

        self.assertEqual(peak.messages, 5)
        self.assertEqual(peak.unique_authors, 5)
        self.assertEqual(peak.reactions, 2)
        self.assertEqual(peak.paid, 1)
        self.assertGreater(peak.zscore, 0)
        self.assertTrue(any(window.start == 5 for window in windows))

    def test_upload_metadata_reuses_complete_video_title(self):
        metadata = {
            "translated_title": "完整翻译视频",
        }

        result = _upload_metadata(metadata)

        self.assertEqual(result["title"], metadata["translated_title"])
        self.assertEqual(result, {"title": metadata["translated_title"]})

    def test_select_chat_peaks_merges_adjacent_qualifying_windows(self):
        windows = [
            ChatWindow(0, 30, 1, 1, 0, 0, 0, 0, 0, 0),
            ChatWindow(30, 60, 10, 8, 5, 0, 0, 2, 4, 4.5, ("草",)),
            ChatWindow(60, 90, 8, 6, 3, 1, 0, 1, 3.5, 5, ("最高",)),
        ]

        peaks = select_chat_peaks(windows, minimum_zscore=3, minimum_unique_authors=5)

        self.assertEqual(len(peaks), 1)
        self.assertEqual((peaks[0].start, peaks[0].end), (30, 90))
        self.assertEqual(peaks[0].paid, 1)
        self.assertEqual(peaks[0].snippets, ("草", "最高"))

    def test_no_chat_fallback_scans_contiguous_transcript_windows(self):
        cues = [Cue(10, 20, "开场"), Cue(310, 320, "后半段")]

        seeds = _semantic_seeds(cues, duration=600)

        self.assertEqual(
            [(seed["context_start"], seed["context_end"]) for seed in seeds],
            [(0.0, 300.0), (300.0, 600)],
        )

    def test_song_seed_keeps_verified_range_and_context_between_songs(self):
        metadata = {
            "identified_songs": [
                {
                    "song": "Song A",
                    "artist": "Artist",
                    "confidence": "high",
                    "score": 0.8,
                    "alignments": [{"start": 100, "end": 180}],
                    "search_group": {"start": 90, "end": 190},
                },
                {
                    "song": "Song B",
                    "confidence": "medium",
                    "alignments": [{"start": 250, "end": 320}],
                    "search_group": {"start": 240, "end": 330},
                },
            ]
        }

        seeds = _song_seeds(metadata, [], 500)

        self.assertEqual(len(seeds), 2)
        self.assertEqual(seeds[0]["required_start"], 100)
        self.assertEqual(seeds[0]["performance_start"], 90)
        self.assertEqual(seeds[0]["context_end"], 240)
        self.assertEqual(seeds[1]["context_start"], 190)

    def test_llm_range_is_limited_to_candidate_and_preserves_song(self):
        cues = [(1, Cue(80, 95, "报幕")), (2, Cue(190, 210, "感想"))]
        seed = {
            "context_start": 70,
            "context_end": 220,
            "performance_start": 90,
            "performance_end": 200,
        }

        start, end = _validated_response_range(
            {"start_id": 1, "end_id": 2}, cues, seed, True
        )

        self.assertEqual((start, end), (80, 210))
        with self.assertRaisesRegex(RuntimeError, "outside the candidate"):
            _validated_response_range({"start_id": 0, "end_id": 2}, cues, seed, True)

    def test_low_confidence_song_review_falls_back_to_verified_range(self):
        seed = {
            "kind": "song",
            "context_start": 80,
            "context_end": 220,
            "performance_start": 100,
            "performance_end": 200,
            "song": "Song A",
            "artist": "Artist",
            "score": 0.8,
        }
        cues = [Cue(80, 100, "报幕"), Cue(200, 220, "感想")]

        with patch(
            "subtitle_pipeline.clips._request_json",
            return_value={
                "worthy": True,
                "confidence": "medium",
                "start_id": 0,
                "end_id": 1,
                "title": "边界不可靠",
                "reason": "置信度不足",
            },
        ):
            result = _review_seeds(
                [seed], cues, translator=None, config=AppConfig(), required=True
            )

        self.assertEqual((result[0].start, result[0].end), (100, 200))
        self.assertIn("完整歌曲范围", result[0].reason)

    def test_speech_parts_overlapping_songs_are_removed_and_duplicates_merge(self):
        speech = [
            ClipPart("speech", 10, 50, "one", "r", "high", 3),
            ClipPart("speech", 40, 80, "two", "r", "high", 5),
            ClipPart("speech", 150, 190, "song overlap", "r", "high", 9),
        ]
        songs = [ClipPart("song", 180, 260, "song", "r", "high", 1)]

        result = _merge_speech_parts(speech, songs, 480)

        self.assertEqual(len(result), 1)
        self.assertEqual((result[0].start, result[0].end), (10, 80))
        self.assertEqual(result[0].title, "two")

    def test_run_clips_uses_cache_renders_in_order_and_never_reuploads(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job = root / "video123"
            job.mkdir()
            (job / "manifest.json").write_text(json.dumps({
                "uploaded": True, "aid": 123, "bvid": "BV123"}))
            (job / "translated.mp4").write_bytes(b"video")
            write_srt([Cue(0, 60, "字幕")], job / "translated.zh-CN.srt")
            (job / "translated.metadata.json").write_text(
                json.dumps({"upload_tags": ["切片"]}), encoding="utf-8"
            )
            (job / "source.info.json").write_text(
                json.dumps({"duration": 60}), encoding="utf-8"
            )
            self._complete_pipeline_stages(job)
            config = AppConfig(
                work_dir=root,
                clips=ClipsConfig(upload=True),
            )
            analysis = {
                "version": 1,
                "signature": "signature",
                "parts": [
                    {
                        "kind": "speech",
                        "start": 20,
                        "end": 30,
                        "title": "second",
                        "file": "parts/002_second.mp4",
                    },
                    {
                        "kind": "speech",
                        "start": 0,
                        "end": 10,
                        "title": "first",
                        "file": "parts/001_first.mp4",
                    },
                ],
                "upload_metadata": {"title": "合集", "description": "简介"},
            }

            def render(_source, destination, _start, _end, _config):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"part")

            submission = BilibiliSubmission(1, "BV1test", "ok")
            with (
                patch(
                    "subtitle_pipeline.clips._analyze",
                    return_value=analysis,
                ),
                patch("subtitle_pipeline.clips._render_clip", side_effect=render),
                patch(
                    "subtitle_pipeline.clips.upload_videos_to_bilibili",
                    return_value=submission,
                ) as upload,
            ):
                first = run_clips("https://www.youtube.com/watch?v=video123", config)
                second = run_clips("https://www.youtube.com/watch?v=video123", config)

            self.assertTrue(first.uploaded)
            self.assertTrue(second.uploaded)
            self.assertEqual(upload.call_count, 1)
            self.assertEqual(upload.call_args.kwargs["append_aid"], 123)
            self.assertEqual(
                [path.name for path in upload.call_args.args[0]],
                ["001_first.mp4", "002_second.mp4"],
            )
            self.assertTrue((job / "clips" / "upload.json").is_file())

    def test_run_clips_requires_completed_job(self):
        with tempfile.TemporaryDirectory() as temp:
            config = AppConfig(work_dir=Path(temp))
            with self.assertRaisesRegex(RuntimeError, "completed pipeline job"):
                run_clips("https://www.youtube.com/watch?v=missing", config)

    def test_reset_parent_rejects_stale_files_until_pipeline_completes(self):
        for parent in ("translation", "metadata"):
            with self.subTest(parent=parent), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                job = root / "video123"
                job.mkdir()
                for name in ("translated.mp4", "translated.zh-CN.srt",
                             "translated.metadata.json", "source.info.json"):
                    (job / name).write_text("old")
                store = self._complete_pipeline_stages(job)
                config = AppConfig(work_dir=root)
                def analyze(*args):
                    return {"parts": [], "subtitle": (job / "translated.zh-CN.srt").read_text()}
                with patch("subtitle_pipeline.clips._analyze", side_effect=analyze) as analysis:
                    run_clips("https://youtu.be/video123", config, upload_override=False)
                    store.reset(parent)
                    with self.assertRaisesRegex(RuntimeError, "completed pipeline job"):
                        run_clips("https://youtu.be/video123", config, upload_override=False)
                    self.assertIsNone(store.existing("clip_analysis"))
                    analysis.assert_called_once()
                    (job / "translated.zh-CN.srt").write_text("new")
                    self._complete_pipeline_stages(job)
                    result = run_clips("https://youtu.be/video123", config, upload_override=False)
                    self.assertEqual(analysis.call_count, 2)
                self.assertEqual(json.loads(result.analysis.read_text())["subtitle"], "new")

    def test_cached_clips_reject_a_different_llm_provider(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job = root / "video123"
            job.mkdir()
            for name in ("translated.mp4", "translated.zh-CN.srt",
                         "translated.metadata.json", "source.info.json"):
                (job / name).write_text("artifact")
            self._complete_pipeline_stages(job)
            config = AppConfig(work_dir=root)
            with patch("subtitle_pipeline.clips._analyze", return_value={"parts": []}) as analysis:
                run_clips("https://youtu.be/video123", config, upload_override=False)
                changed = replace(config, llm=replace(config.llm, base_url="https://different.invalid/v1"))
                with self.assertRaisesRegex(RuntimeError, "cached LLM provider differs"):
                    run_clips("https://youtu.be/video123", changed, upload_override=False)
                analysis.assert_called_once()

    def test_chat_download_failure_uses_fallback_and_removes_raw_files(self):
        with tempfile.TemporaryDirectory() as temp:
            clips_dir = Path(temp)

            def fail_download(_url, directory, _config):
                (directory / "source.live_chat.json.part").write_text(
                    "partial", encoding="utf-8"
                )
                raise RuntimeError("download failed")

            with patch(
                "subtitle_pipeline.clips._download_youtube_chat_replay",
                side_effect=fail_download,
            ):
                messages = _download_chat(
                    "https://youtu.be/video123", clips_dir, AppConfig()
                )

            self.assertEqual(messages, [])
            self.assertFalse(
                (clips_dir / "chat-download" / "source.live_chat.json.part").exists()
            )

    def test_render_clip_reencodes_exact_range(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "translated.mp4"
            destination = root / "parts" / "001_clip.mp4"
            source.write_bytes(b"source")

            def fake_run(command):
                Path(command[-1]).parent.mkdir(parents=True, exist_ok=True)
                Path(command[-1]).write_bytes(b"clip")

            with (
                patch("subtitle_pipeline.clips.require_command", return_value="ffmpeg"),
                patch("subtitle_pipeline.clips.run", side_effect=fake_run) as run,
            ):
                _render_clip(source, destination, 12.3456, 45.6789, AppConfig())

            command = run.call_args.args[0]
            self.assertEqual(command[command.index("-ss") + 1], "12.346")
            self.assertEqual(command[command.index("-t") + 1], "33.333")
            self.assertEqual(command[command.index("-c:v") + 1], "libx264")
            self.assertEqual(destination.read_bytes(), b"clip")

    def test_render_failure_prevents_incomplete_upload(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job = root / "video123"
            job.mkdir()
            (job / "translated.mp4").write_bytes(b"video")
            write_srt([Cue(0, 60, "字幕")], job / "translated.zh-CN.srt")
            (job / "translated.metadata.json").write_text("{}", encoding="utf-8")
            (job / "source.info.json").write_text(
                json.dumps({"duration": 60}), encoding="utf-8"
            )
            self._complete_pipeline_stages(job)
            analysis = {
                "version": 1,
                "signature": "signature",
                "parts": [
                    {
                        "kind": "speech",
                        "start": 0,
                        "end": 30,
                        "title": "first",
                        "file": "parts/001_first.mp4",
                    },
                    {
                        "kind": "speech",
                        "start": 30,
                        "end": 60,
                        "title": "second",
                        "file": "parts/002_second.mp4",
                    },
                ],
                "upload_metadata": {"title": "合集", "description": "简介"},
            }

            def fail_second(_source, destination, _start, _end, _config):
                if destination.name.startswith("002_"):
                    raise RuntimeError("ffmpeg failed")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"part")

            config = AppConfig(work_dir=root, clips=ClipsConfig(upload=True))
            with (
                patch(
                    "subtitle_pipeline.clips._analyze",
                    return_value=analysis,
                ),
                patch("subtitle_pipeline.clips._render_clip", side_effect=fail_second),
                patch("subtitle_pipeline.clips.upload_videos_to_bilibili") as upload,
                self.assertRaisesRegex(RuntimeError, "ffmpeg failed"),
            ):
                run_clips("https://www.youtube.com/watch?v=video123", config)

            upload.assert_not_called()
            self.assertFalse((job / "clips" / "upload.json").exists())


if __name__ == "__main__":
    unittest.main()
