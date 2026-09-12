from subtitle_pipeline.repetition import find_repetition_loop
"""Offline recovery checks across cache, model, media and publication boundaries."""
import json
import re
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from subtitle_pipeline.asr import _transcribe_raw_range, _transcribe_range, transcribe_with_qwen
from subtitle_pipeline.asr_correction import correct_asr_windows as _correct_asr_windows
from subtitle_pipeline.audio_analysis import AudioAnalysis, AudioRegion, analyze_audio
from subtitle_pipeline.bilibili_comments import create_comment_task, publish_comment_task
from subtitle_pipeline.cache import CacheStore, STAGES, StageDefinition
from subtitle_pipeline.config import ASRConfig, AudioAnalysisConfig, DownloadConfig, LLMConfig, RenderConfig, TranslationConfig, SegmentationConfig, UploadConfig
from subtitle_pipeline.media import DownloadResult, download_youtube, render_subtitles
from subtitle_pipeline.publication import write_record
from subtitle_pipeline.staged_translation import run_joint_translation
from subtitle_pipeline.subtitles import Cue


from functools import partial
from subtitle_pipeline.prompt_budget import estimate_prompt_tokens, validate_request_budget

def validate_test_request(body):
    validate_request_budget(body, context_size=16384,
        count_tokens=lambda value: estimate_prompt_tokens(json.dumps(value['messages'], ensure_ascii=False)))

correct_asr_windows = partial(_correct_asr_windows, validate_request=validate_test_request)


def response(cues):
    return {"choices": [{"message": {"content": json.dumps({"cues": cues})}}]}


class CacheRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "cache.sqlite3"
        self.store = CacheStore(self.path)
        self.audio = SimpleNamespace(sample_rate=16000, slice=lambda *a, **k: np.zeros(1600, dtype=np.float32))

    def test_source_hit_precedes_audio_probe_and_models(self):
        cue = Cue(0, 1, "こんにちは", language="Japanese")
        self.store.stage("source_cues", lambda: {}).finish({"cues": [asdict(cue)], "evidence": []})
        with patch("subtitle_pipeline.asr._media_duration", side_effect=AssertionError("audio probe")), patch("subtitle_pipeline.asr.AudioBufferPool", side_effect=AssertionError("audio pool")):
            output = transcribe_with_qwen(self.root / "source.mp4", self.root / "source.srt", ASRConfig())
        self.assertIn(cue.text, output.read_text())
        self.assertTrue(output.with_suffix(".cues.json").is_file())

    def test_raw_recursive_resume_reuses_parent_plan_and_completed_child(self):
        stage = self.store.stage("raw_speech", lambda: {})
        repeated = SimpleNamespace(repetition=find_repetition_loop("私が食べてるのでちょっとこっちに移動します" * 10), text="私が食べてるのでちょっとこっちに移動します" * 10, language="Japanese")
        left = SimpleNamespace(repetition=find_repetition_loop("左です"), text="左です", language="Japanese")
        right = SimpleNamespace(repetition=find_repetition_loop("右です"), text="右です", language="Japanese")
        model = SimpleNamespace(max_new_tokens=128, transcribe=Mock(side_effect=[[repeated], [left], KeyboardInterrupt()]))
        with self.assertRaises(KeyboardInterrupt):
            _transcribe_raw_range(model, ASRConfig(), 0, 40, 40, self.audio, cache=stage)
        resumed = SimpleNamespace(max_new_tokens=128, transcribe=Mock(return_value=[right]))
        result = _transcribe_raw_range(resumed, ASRConfig(), 0, 40, 40, self.audio, cache=CacheStore(self.path).existing("raw_speech"))
        self.assertEqual(resumed.transcribe.call_count, 1)
        self.assertIn("左です", result["text"])
        self.assertIn("右です", result["text"])

    def test_aligned_recursive_resume_does_not_request_parent_or_left(self):
        stage = self.store.stage("raw_speech", lambda: {})
        def result(text):
            return SimpleNamespace(repetition=find_repetition_loop(text), text=text, language="Japanese", time_stamps=SimpleNamespace(items=[SimpleNamespace(text=text, start_time=5, end_time=6)]))
        model = SimpleNamespace(transcribe=Mock(side_effect=[[result("私が食べてるのでちょっとこっちに移動します" * 10)], [result("左")], KeyboardInterrupt()]))
        kwargs = dict(core_start=0, core_end=40, media_duration=40, final_chunk=True, label="0", audio_buffer=self.audio, cache=stage)
        with self.assertRaises(KeyboardInterrupt):
            _transcribe_range(model, self.root / "source.mp4", None, ASRConfig(), **kwargs)
        model.transcribe = Mock(return_value=[result("右")])
        value = _transcribe_range(model, self.root / "source.mp4", None, ASRConfig(), **kwargs)
        self.assertEqual(model.transcribe.call_count, 1)
        self.assertEqual(len(value["cues"]), 2)

    def test_correction_version_keeps_raw_and_song_results(self):
        self.store.stage("raw_speech", lambda: {}).put("0", {"text": "原文"})
        self.store.stage("song_identification", lambda: {}).finish({"reports": []})
        kwargs = dict(records=[{"text": "原文", "window_id": 0}], entities=[], model="old", cache_path=self.path, audit_path=self.root / "audit.jsonl")
        request = Mock(return_value=response([{"text": "訂正"}]))
        # Use the correction response schema, independently of translation.
        request.return_value = {"choices": [{"message": {"content": '{"windows":[{"window_id":0,"text":"訂正"}]}'}}]}
        first = correct_asr_windows(**kwargs, request=request)
        self.store.stage("translation", lambda: {}).finish(["old"])
        with patch.dict(STAGES, {"asr_correction": StageDefinition(STAGES["asr_correction"].version + 1, STAGES["asr_correction"].parents)}):
            store = CacheStore(self.path)
            self.assertEqual(store.existing("raw_speech").get("0")["text"], "原文")
            self.assertIsNotNone(store.existing("song_identification").get("__result__"))
            self.assertIsNone(store.existing("translation"))
            correct_asr_windows(**kwargs, request=request)
        self.assertEqual(request.call_count, 2)
        self.assertTrue(first)

    def translation(self, request, **changes):
        args = dict(source_cues=[Cue(0, 1, "原文一"), Cue(1, 2, "原文二")], llm=LLMConfig(max_concurrency=1, max_retries=1), translation=TranslationConfig(batch_cues=1), request=request, parse_content=lambda value: json.loads(value)["cues"], finish_reason=lambda _: "stop", retry_delay=lambda *_: None, is_nontransient=lambda _: True, log_invalid_response=lambda *_: None, local_translate=lambda text: "回退" + text, translation_context={}, honorific_rules="", cache_path=self.path)
        args.update(changes)
        from subtitle_pipeline.local_segmentation import LocalUnit, SpeakerTrack
        args["tracks"] = [SpeakerTrack("A", "A", tuple(
            LocalUnit("A", i, (i,), cue.start, cue.end, cue.text, "A", cue.kind, 0.0)
            for i, cue in enumerate(args["source_cues"])))]
        return run_joint_translation(**args, segmentation=SegmentationConfig(), maximum_units=20)[1]

    def test_partial_translation_freezes_groups_evidence_and_settings(self):
        count = 0
        def interrupted(body):
            nonlocal count
            count += 1
            if count == 2:
                raise KeyboardInterrupt()
            return response([{"start_id": 0, "end_id": 0, "text": "第一行"}])
        retrieval = Mock(return_value=[])
        with self.assertRaises(KeyboardInterrupt):
            self.translation(interrupted, retrieve_knowledge=retrieval)
        pending = Mock(return_value=response([{"start_id": 0, "end_id": 0, "text": "第二行"}]))
        result = self.translation(pending, translation=TranslationConfig(batch_cues=20), retrieve_knowledge=Mock(side_effect=AssertionError("evidence changed")))
        self.assertEqual(pending.call_count, 1)
        self.assertEqual([cue.text for cue in result], ["第一行", "第二行"])
        self.assertEqual(self.store.existing("translation").plan["translation"]["batch_cues"], 1)

    def test_degraded_retry_keeps_good_group_and_failed_retry_keeps_fallback(self):
        request = Mock(side_effect=[response([{"start_id": 0, "end_id": 0, "text": "第一行"}]), response([{"start_id": 0, "end_id": 0, "text": ""}])])
        first = self.translation(request, is_nontransient=lambda exc: str(exc) == "offline")
        self.translation(Mock(side_effect=AssertionError("complete")))
        self.assertEqual(self.store.retry_degraded("translation"), 1)
        with self.assertRaises(RuntimeError):
            self.translation(Mock(side_effect=RuntimeError("offline")))
        restored = self.translation(Mock(side_effect=AssertionError("fallback preserved")))
        self.assertEqual(restored, first)
        self.store.retry_degraded("translation")
        retry = Mock(return_value=response([{"start_id": 0, "end_id": 0, "text": "第二行"}]))
        result = self.translation(retry)
        self.assertEqual(retry.call_count, 1)
        self.assertEqual([cue.text for cue in result], ["第一行", "第二行"])

    def test_missing_download_metadata_is_rebuilt_without_network(self):
        video = self.root / "source.mp4"
        video.write_bytes(b"video")
        comments = self.root / "comments.info.json"
        comments.write_text('{"comments": []}')
        config = DownloadConfig(download_chat_replay=False, download_top_comments=True)
        with patch("subtitle_pipeline.media._download_youtube", return_value=DownloadResult(video, {"title": "old"})) as downloader, patch("subtitle_pipeline.media._download_youtube_top_comments", return_value=comments) as fetch, patch("subtitle_pipeline.media._video_dimensions", return_value=(1920, 1080)):
            download_youtube("url", self.root, config)
            comments.unlink()
            (self.root / "source.info.json").unlink()
            restored = download_youtube("changed", self.root, replace(config, download_top_comments=False))
        self.assertEqual(downloader.call_count, 1)
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(restored.metadata["title"], "old")
        self.assertTrue(restored.comments.is_file())

    def test_missing_render_repairs_only_artifact_using_frozen_cues(self):
        target = self.root / "translated.mp4"
        calls = []
        def render(video, subtitle, destination, config, **kwargs):
            calls.append(kwargs["cues"])
            destination.write_bytes(b"rendered")
        with patch("subtitle_pipeline.media._render_subtitles", side_effect=render):
            render_subtitles(self.root / "source.mp4", self.root / "source.srt", target, RenderConfig(), cues=[Cue(0, 1, "旧")])
            target.unlink()
            render_subtitles(self.root / "source.mp4", self.root / "source.srt", target, RenderConfig(), cues=[Cue(0, 1, "新")])
        self.assertEqual([values[0].text for values in calls], ["旧", "旧"])

    def test_missing_vocal_artifact_does_not_rerun_analysis_models(self):
        stem = self.root / "stem.wav"
        analysis = AudioAnalysis([], [AudioRegion(0, 10, "singing", source_path=str(stem))])
        stage = self.store.stage("audio_analysis", lambda: {"config": asdict(AudioAnalysisConfig()), "metadata": {}})
        stage.put("vocal_artifacts", [{"start": 0, "end": 10, "path": str(stem)}], kind="plan")
        stage.finish(asdict(analysis))
        with patch("subtitle_pipeline.audio_analysis.separate_vocal_ranges", side_effect=lambda *_: stem.write_bytes(b"restored")) as separate, patch("subtitle_pipeline.audio_analysis._run_initial_audio_analysis", side_effect=AssertionError("analysis")):
            result = analyze_audio(self.root / "source.mp4", self.root, AudioAnalysisConfig())
        self.assertEqual(separate.call_count, 1)
        self.assertEqual(result.singing[0].source_path, str(stem))

    def test_comment_resume_preserves_schedule_and_terminal_status(self):
        kwargs = dict(aid=1, bvid="BVexample", message="歌单", source_url="url")
        with patch("subtitle_pipeline.bilibili_comments.time.time", return_value=100):
            path = create_comment_task(self.root, **kwargs)
        original = json.loads(path.read_text())
        with patch("subtitle_pipeline.bilibili_comments.time.time", return_value=1000):
            create_comment_task(self.root, **kwargs)
        self.assertEqual(json.loads(path.read_text())["publish_at"], original["publish_at"])
        write_record(path, {**original, "status": "posted"})
        with patch("subtitle_pipeline.bilibili_comments._load_cookies", side_effect=AssertionError("already posted")):
            self.assertEqual(publish_comment_task(path, UploadConfig()), "posted")
