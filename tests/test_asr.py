import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from subtitle_pipeline.asr import (
    _add_punctuation_boundary_hints,
    _analysis_region_signature,
    _analysis_regions,
    _align_speech_records,
    _asr_generation_token_limit,
    _cache_signature,
    _HeartTranscriptorAdapter,
    _load_cache,
    _record_timeline_is_healthy,
    _remove_text_overlap,
    _repetition_hallucination,
    _result_to_cues,
    _song_cut_candidates,
    _song_windows,
    _SongCutCandidate,
    _speaker_assignment_for_aligned_cue,
    _speaker_assignment_timeline,
    _speaker_for_aligned_cue,
    _speech_asr_windows,
    _timeline_retry_split,
    _transcribe_analyzed,
    _transcribe_range,
    _transcribe_song_range,
    _transcribe_speech_batch,
    _valid_cached_record,
    read_cue_sidecar,
    transcribe_with_qwen,
)
from subtitle_pipeline.audio_analysis import AcousticPhrase, AudioAnalysis, AudioRegion
from subtitle_pipeline.config import ASRConfig, AudioAnalysisConfig
from subtitle_pipeline.subtitles import Cue


class QwenASRTests(unittest.TestCase):
    def test_corrected_and_original_unalignable_speech_is_discarded(self):
        aligner = Mock()
        aligner.align.side_effect = [[object()], [object()]]
        record = {
            "window_id": 0,
            "core_start": 10.0,
            "core_end": 20.0,
            "language": "English",
            "text": "corrected text",
            "original_text": "original text",
        }
        audio_buffer = SimpleNamespace(
            sample_rate=16000,
            slice=Mock(return_value=np.zeros(16000, dtype=np.float32)),
        )

        with (
            patch(
                "subtitle_pipeline.asr._result_to_cues",
                return_value=[Cue(10.0, 10.5, "text")],
            ),
            patch(
                "subtitle_pipeline.asr._record_timeline_is_healthy",
                return_value=False,
            ),
        ):
            result = _align_speech_records(
                aligner,
                [record],
                [AudioRegion(10.0, 20.0, "speech")],
                audio_buffer,
                ASRConfig(max_inference_batch_size=1),
                30.0,
            )[0]

        self.assertEqual(result["text"], "")
        self.assertEqual(result["cues"], [])
        self.assertTrue(result["skipped_empty"])
        self.assertEqual(
            result["correction_method"], "alignment_unusable_discarded"
        )
        self.assertEqual(
            result["alignment_error"],
            "corrected_and_original_timeline_invalid",
        )

    def test_heart_transcriptor_adapter_accepts_buffer_audio(self):
        transcriber = Mock(return_value={"text": "聞こえた歌詞"})
        model = _HeartTranscriptorAdapter(
            transcriber,
            max_new_tokens=123,
            num_beams=3,
        )

        result = model.transcribe(
            audio=(np.zeros(1600, dtype=np.float32), 16000),
            context="ignored",
            language=None,
            return_time_stamps=False,
        )

        self.assertEqual(result[0].text, "聞こえた歌詞")
        self.assertEqual(result[0].language, "Japanese")
        source = transcriber.call_args.args[0]
        self.assertEqual(source["sampling_rate"], 16000)
        self.assertEqual(source["raw"].shape, (1600,))
        self.assertEqual(
            transcriber.call_args.kwargs["generate_kwargs"]["max_new_tokens"],
            123,
        )
        self.assertEqual(
            transcriber.call_args.kwargs["generate_kwargs"]["num_beams"], 3
        )

    def test_asr_punctuation_becomes_boundary_metadata_only_when_text_matches(self):
        cues = [Cue(0, 1, "まもなく"), Cue(1, 2, "開演"), Cue(2, 3, "です")]

        hinted = _add_punctuation_boundary_hints("まもなく開演、です。", cues)
        mismatched = _add_punctuation_boundary_hints("完全に別の文章、です。", cues)

        self.assertEqual([cue.boundary_hint for cue in hinted], [None, "weak", None])
        self.assertEqual([cue.boundary_hint for cue in mismatched], [None, None, None])

    def test_forced_aligner_pos_is_preserved_on_aligned_units(self):
        result = SimpleNamespace(
            text="配信です",
            language="Japanese",
            time_stamps=SimpleNamespace(
                items=[
                    SimpleNamespace(
                        text="配信", start_time=0.0, end_time=0.5, pos="名詞"
                    ),
                    SimpleNamespace(
                        text="です", start_time=0.5, end_time=0.8, pos="助動詞"
                    ),
                ]
            ),
        )

        cues = _result_to_cues(
            result,
            offset=0.0,
            keep_start=0.0,
            keep_end=1.0,
            final_chunk=True,
        )

        self.assertEqual([cue.pos for cue in cues], ["名詞", "助動詞"])
        self.assertEqual([cue.language for cue in cues], ["Japanese", "Japanese"])

    def test_forced_aligner_language_is_recorded_per_source_unit(self):
        result = SimpleNamespace(
            text="Ready set",
            language="English",
            time_stamps=SimpleNamespace(
                items=[
                    SimpleNamespace(text="Ready", start_time=0.0, end_time=0.4),
                    SimpleNamespace(text="set", start_time=0.4, end_time=0.8),
                ]
            ),
        )

        cues = _result_to_cues(
            result,
            offset=0.0,
            keep_start=0.0,
            keep_end=1.0,
            final_chunk=True,
        )

        self.assertEqual([cue.language for cue in cues], ["English", "English"])

    def test_single_word_list_participates_in_asr_cache_signature(self):
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "source.mp4"
            video.write_bytes(b"video")
            config = ASRConfig()

            first = _cache_signature(video, 1.0, config, ["藤都子"])
            reordered = _cache_signature(
                video, 1.0, config, ["藤都子", "宮永ののか", "藤都子"]
            )
            changed = _cache_signature(video, 1.0, config, ["宮永ののか"])

        self.assertEqual(first["japanese_single_word_list"], ["藤都子"])
        self.assertEqual(
            reordered["japanese_single_word_list"], ["宮永ののか", "藤都子"]
        )
        self.assertNotEqual(first, changed)

    def test_speech_batch_transcribes_four_windows_in_one_model_call(self):
        calls = []

        class Model:
            max_new_tokens = 2048

            def transcribe(self, **kwargs):
                calls.append(kwargs)
                return [
                    SimpleNamespace(
                        language="Japanese",
                        text=f"字幕{index}",
                        time_stamps=SimpleNamespace(
                            items=[
                                SimpleNamespace(
                                    text=f"字幕{index}",
                                    start_time=0.1,
                                    end_time=0.5,
                                )
                            ]
                        ),
                    )
                    for index in range(4)
                ]

        audio = SimpleNamespace(
            sample_rate=16000,
            slice=lambda *_args, **_kwargs: np.zeros(16000, dtype=np.float32),
        )
        regions = [
            (index, AudioRegion(index * 2.0, index * 2.0 + 1.0, "speech"))
            for index in range(4)
        ]

        records = _transcribe_speech_batch(
            Model(),
            Path("source.mp4"),
            ASRConfig(chunk_context_seconds=0, max_inference_batch_size=4),
            regions,
            media_duration=8.0,
            audio_buffer=audio,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]["audio"]), 4)
        self.assertEqual(list(records), [0, 1, 2, 3])

    def test_short_audio_uses_bounded_generation_token_limit(self):
        config = ASRConfig(max_new_tokens=2048)

        self.assertEqual(_asr_generation_token_limit(config, 0.5), 128)
        self.assertEqual(_asr_generation_token_limit(config, 4.5), 176)
        self.assertEqual(_asr_generation_token_limit(config, 170), 2048)

    def test_explicitly_skipped_empty_speech_is_a_healthy_record(self):
        self.assertTrue(
            _record_timeline_is_healthy(
                {"text": "", "cues": [], "skipped_empty": True},
                AudioRegion(10.0, 10.8, "speech"),
            )
        )

    def test_empty_speech_is_audited_and_skipped_without_split(self):
        empty = SimpleNamespace(
            language="Japanese",
            text="",
            time_stamps=SimpleNamespace(items=[]),
        )
        model = SimpleNamespace(max_new_tokens=2048)
        model.transcribe = Mock(return_value=[empty])
        audio = SimpleNamespace(
            sample_rate=16000,
            slice=lambda *_args, **_kwargs: np.zeros(640000, dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as temp:
            audit_path = Path(temp) / "empty.jsonl"
            record = _transcribe_range(
                model,
                Path("source.mp4"),
                None,
                ASRConfig(chunk_context_seconds=0),
                core_start=0,
                core_end=40,
                media_duration=40,
                final_chunk=True,
                label="empty",
                audio_buffer=audio,
                validate_timeline=True,
                empty_speech_audit_path=audit_path,
            )
            events = [json.loads(line) for line in audit_path.read_text().splitlines()]

        self.assertEqual(record["cues"], [])
        self.assertTrue(record["skipped_empty"])
        model.transcribe.assert_called_once()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["reason"], "empty_aligned_cues")
        self.assertEqual(events[0]["action"], "skip")
        self.assertEqual(events[0]["core_start"], 0)
        self.assertEqual(events[0]["core_end"], 40)

    def test_short_empty_speech_is_audited_and_skipped(self):
        empty = SimpleNamespace(
            language="Japanese",
            text="",
            time_stamps=SimpleNamespace(items=[]),
        )
        model = SimpleNamespace(max_new_tokens=2048)
        model.transcribe = Mock(return_value=[empty])
        audio = SimpleNamespace(
            sample_rate=16000,
            slice=lambda *_args, **_kwargs: np.zeros(16000, dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as temp:
            audit_path = Path(temp) / "empty.jsonl"
            record = _transcribe_range(
                model,
                Path("source.mp4"),
                None,
                ASRConfig(chunk_context_seconds=0),
                core_start=10,
                core_end=11,
                media_duration=20,
                final_chunk=True,
                label="short-empty",
                audio_buffer=audio,
                validate_timeline=True,
                empty_speech_audit_path=audit_path,
            )
            events = [json.loads(line) for line in audit_path.read_text().splitlines()]

        self.assertTrue(record["skipped_empty"])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["action"], "skip")

    def test_transcribe_range_temporarily_applies_dynamic_token_limit(self):
        observed_limits = []
        result = SimpleNamespace(
            language="Japanese",
            text="短い音声",
            time_stamps=SimpleNamespace(
                items=[SimpleNamespace(text="短い音声", start_time=0.1, end_time=0.4)]
            ),
        )

        class Model:
            max_new_tokens = 2048

            def transcribe(self, **_kwargs):
                observed_limits.append(self.max_new_tokens)
                return [result]

        model = Model()
        audio = SimpleNamespace(
            sample_rate=16000,
            slice=lambda *_args, **_kwargs: np.zeros(8000, dtype=np.float32),
        )

        from subtitle_pipeline.asr import _transcribe_range

        record = _transcribe_range(
            model,
            Path("source.mp4"),
            None,
            ASRConfig(chunk_context_seconds=0, max_new_tokens=2048),
            core_start=0,
            core_end=0.5,
            media_duration=0.5,
            final_chunk=True,
            label="short",
            audio_buffer=audio,
        )

        self.assertEqual(observed_limits, [128])
        self.assertEqual(model.max_new_tokens, 2048)
        self.assertEqual(record["generation_token_limit"], 128)

    def test_cache_signature_normalizes_json_container_types(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cache.json"
            path.write_text(
                '{"version":10,"signature":{"speakers":["A","B"]},'
                '"chunks":{"0":{"text":"x","cues":[]}}}',
                encoding="utf-8",
            )

            result = _load_cache(path, {"speakers": ("A", "B")})

        self.assertIn("0", result["chunks"])

    def test_old_cue_sidecar_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.cues.json"
            path.write_text(
                '{"version":2,"cues":[{"start":0,"end":1,"text":"字幕"}]}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "invalid cue sidecar"):
                read_cue_sidecar(path)

    def test_cue_sidecar_preserves_detected_language(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.cues.json"
            path.write_text(
                '{"version":7,"cues":[{"start":0,"end":1,'
                '"text":"Ready","language":"English"}]}',
                encoding="utf-8",
            )

            cues = read_cue_sidecar(path)

        self.assertEqual(cues[0].language, "English")

    def test_shared_audio_names_do_not_invalidate_asr_cache_signature(self):
        left = _analysis_region_signature(
            AudioRegion(1, 2, "speech", "A", source_path="shm://first")
        )
        right = _analysis_region_signature(
            AudioRegion(1, 2, "speech", "A", source_path="shm://second")
        )
        self.assertEqual(left, right)
        self.assertEqual(left["source_path"], "shared-memory")

    def test_groups_fragments_from_one_separated_speaker_track(self):
        result = _analysis_regions(
            AudioAnalysis(
                speech=[
                    AudioRegion(
                        10.0,
                        11.0,
                        "speech",
                        "A",
                        overlap=True,
                        source_path="a.wav",
                        source_offset=9.0,
                    ),
                    AudioRegion(
                        12.0,
                        14.0,
                        "speech",
                        "A",
                        overlap=True,
                        source_path="a.wav",
                        source_offset=9.0,
                    ),
                    AudioRegion(
                        10.5,
                        13.0,
                        "speech",
                        "B",
                        overlap=True,
                        source_path="b.wav",
                        source_offset=9.0,
                    ),
                ],
                singing=[],
            )
        )

        self.assertEqual(len(result), 2)
        self.assertEqual((result[0].start, result[0].end), (10.0, 14.0))
        self.assertEqual(result[0].source_offset, 9.0)

    def test_analyzed_speech_uses_original_mix_buffer_and_global_timeline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "source.mp4"
            destination = root / "source.qwen3-asr.srt"
            video.write_bytes(b"video")
            region = AudioRegion(
                10.0,
                11.0,
                "speech",
                "A",
                source_path="shm://track",
                source_offset=9.0,
            )
            buffer = SimpleNamespace(duration=2.0)
            pool = SimpleNamespace(
                contains=lambda _uri: True,
                resolve=lambda _uri: buffer,
                main=lambda: buffer,
            )
            record = {
                "text": "字幕",
                "cues": [{"start": 1.0, "end": 1.5, "text": "字幕"}],
            }
            with (
                patch(
                    "subtitle_pipeline.asr._media_duration", return_value=30.0
                ) as media_duration,
                patch("subtitle_pipeline.asr._load_qwen_model", return_value=object()),
                patch(
                    "subtitle_pipeline.asr._transcribe_raw_speech_batch",
                    return_value={
                        0: {
                            "core_start": 10.0,
                            "core_end": 11.0,
                            "text": "字幕",
                            "language": "Japanese",
                        }
                    },
                ) as transcribe,
                patch(
                    "subtitle_pipeline.asr._load_qwen_aligner", return_value=object()
                ),
                patch(
                    "subtitle_pipeline.asr._align_speech_records",
                    return_value={0: record},
                ) as align,
            ):
                _transcribe_analyzed(
                    video,
                    destination,
                    ASRConfig(),
                    AudioAnalysisConfig(),
                    AudioAnalysis([region], []),
                    pool,
                )

            media_duration.assert_called_once_with(video)
            self.assertEqual(transcribe.call_args.kwargs["media_duration"], 30.0)
            self.assertIs(transcribe.call_args.kwargs["audio_buffer"], buffer)
            indexed = transcribe.call_args.args[2]
            self.assertEqual((indexed[0][1].start, indexed[0][1].end), (10.0, 11.0))
            self.assertIs(align.call_args.args[3], buffer)

    def test_speech_windows_merge_nearby_turns_across_speakers(self):
        analysis = AudioAnalysis(
            speech=[],
            singing=[],
            diarization=[
                AudioRegion(0, 25, "speech", "A"),
                AudioRegion(26, 45, "speech", "B"),
                AudioRegion(44, 70, "speech", "A"),
            ],
        )

        windows = _speech_asr_windows(
            analysis,
            ASRConfig(
                chunk_context_seconds=2,
                speech_window_target_seconds=60,
                speech_window_max_seconds=90,
            ),
        )

        self.assertEqual(
            [(window.start, window.end, window.kind) for window in windows],
            [(0, 70, "speech")],
        )

    def test_speech_windows_do_not_cross_more_than_two_seconds_of_silence(self):
        analysis = AudioAnalysis(
            speech=[],
            singing=[],
            diarization=[
                AudioRegion(0, 20, "speech", "A"),
                AudioRegion(22.001, 40, "speech", "A"),
            ],
        )

        windows = _speech_asr_windows(analysis, ASRConfig())

        self.assertEqual(
            [(window.start, window.end) for window in windows],
            [(0, 20), (22.001, 40)],
        )

    def test_short_speech_episode_merges_into_nearest_window(self):
        analysis = AudioAnalysis(
            speech=[],
            singing=[],
            diarization=[
                AudioRegion(60.0, 90.0, "speech", "A"),
                AudioRegion(94.0, 94.7, "speech", "A"),
                AudioRegion(104.0, 130.0, "speech", "A"),
            ],
        )

        windows = _speech_asr_windows(analysis, ASRConfig())

        self.assertEqual(
            [(window.start, window.end) for window in windows],
            [(60.0, 94.7), (104.0, 130.0)],
        )

    def test_short_speech_without_nearby_episode_keeps_original_core(self):
        analysis = AudioAnalysis(
            speech=[],
            singing=[],
            diarization=[AudioRegion(100.0, 100.7, "speech", "A")],
        )

        windows = _speech_asr_windows(analysis, ASRConfig())

        self.assertEqual(
            [(window.start, window.end) for window in windows],
            [(100.0, 100.7)],
        )

    def test_speech_windows_rebalance_instead_of_stranding_a_short_tail(self):
        spans = [
            (0.0, 1.435),
            (2.346, 4.911),
            (5.518, 6.497),
            (7.527, 8.893),
            (10.294, 13.517),
            (13.568, 16.065),
            (16.335, 17.533),
            (18.613, 19.170),
            (20.875, 22.157),
            (23.743, 31.152),
            (31.793, 32.502),
            (33.902, 36.180),
            (37.935, 38.880),
            (39.302, 41.749),
            (42.745, 44.348),
            (46.086, 46.930),
        ]
        analysis = AudioAnalysis(
            speech=[],
            singing=[],
            diarization=[
                AudioRegion(start, end, "speech", "A") for start, end in spans
            ],
        )

        windows = _speech_asr_windows(analysis, ASRConfig())

        self.assertEqual(
            [(window.start, window.end) for window in windows],
            [(0.0, 22.157), (23.743, 46.93)],
        )

    def test_aligned_cue_speaker_uses_diarization_intersection(self):
        diarization = [
            AudioRegion(0, 3, "speech", "A"),
            AudioRegion(2, 4, "speech", "B"),
            AudioRegion(4, 6, "speech", "B"),
        ]

        self.assertEqual(_speaker_for_aligned_cue(0.5, 1.5, diarization), "A")
        self.assertEqual(_speaker_for_aligned_cue(2.2, 2.8, diarization), "A")
        self.assertEqual(_speaker_for_aligned_cue(4.2, 5.0, diarization), "B")
        self.assertEqual(
            _speaker_for_aligned_cue(
                1.8,
                2.2,
                [
                    AudioRegion(0, 2, "speech", "A"),
                    AudioRegion(2, 4, "speech", "B"),
                ],
            ),
            "B",
        )
        self.assertEqual(
            _speaker_for_aligned_cue(
                10.0,
                10.4,
                [
                    AudioRegion(9.0, 10.38, "speech", "A"),
                    AudioRegion(10.38, 12.0, "speech", "B"),
                ],
            ),
            "A",
        )
        self.assertEqual(
            _speaker_for_aligned_cue(
                12.0,
                13.0,
                [
                    AudioRegion(12.0, 12.1, "speech", "A"),
                    AudioRegion(12.8, 13.0, "speech", "B"),
                ],
            ),
            "B",
        )
        self.assertEqual(
            _speaker_for_aligned_cue(
                20.0,
                21.0,
                [AudioRegion(20.69, 21.0, "speech", "A")],
            ),
            "A",
        )
        self.assertIsNone(
            _speaker_for_aligned_cue(
                20.0,
                21.0,
                [AudioRegion(20.81, 21.0, "speech", "A")],
            )
        )
        self.assertIsNone(
            _speaker_for_aligned_cue(
                30.0,
                30.4,
                [AudioRegion(29.0, 30.1, "speech", "A")],
            ),
        )
        self.assertIsNone(
            _speaker_for_aligned_cue(
                40.0,
                40.1,
                [AudioRegion(39.0, 39.95, "speech", "A")],
            ),
        )
        self.assertIsNone(
            _speaker_for_aligned_cue(
                50.0,
                50.1,
                [AudioRegion(49.0, 49.89, "speech", "A")],
            )
        )

        overlapping = _speaker_assignment_for_aligned_cue(
            20.0, 21.0, [AudioRegion(20.81, 21.0, "speech", "A")]
        )
        nearby = _speaker_assignment_for_aligned_cue(
            40.0, 40.1, [AudioRegion(39.0, 39.95, "speech", "A")]
        )
        self.assertEqual(
            (overlapping.fallback_speaker, overlapping.fallback_distance),
            ("A", 0.0),
        )
        self.assertEqual(nearby.fallback_speaker, "A")
        self.assertAlmostEqual(nearby.fallback_distance or 0.0, 0.05)

    def test_speaker_assignment_prefers_ordinary_diarization(self):
        exclusive = [AudioRegion(0, 2, "speech", "A")]
        ordinary = [
            AudioRegion(0, 2, "speech", "A"),
            AudioRegion(1, 2, "speech", "B"),
        ]

        self.assertIs(
            _speaker_assignment_timeline(
                AudioAnalysis(exclusive, [], diarization=ordinary)
            ),
            ordinary,
        )
        self.assertIs(
            _speaker_assignment_timeline(AudioAnalysis(exclusive, [])),
            exclusive,
        )

    def test_timeline_retry_prefers_a_long_silence_over_the_midpoint(self):
        record = {
            "cues": [
                {"start": 0, "end": 10, "text": "前"},
                {"start": 32, "end": 40, "text": "中"},
                {"start": 42, "end": 60, "text": "后"},
            ]
        }

        self.assertEqual(_timeline_retry_split(record, 0, 60), 21)

    def test_successful_retry_child_is_cached_before_its_sibling_finishes(self):
        from subtitle_pipeline.asr import _transcribe_range

        repeated = "私が食べてるのでちょっとこっちに移動します" * 10

        def result(text, start, end):
            return SimpleNamespace(
                language="Japanese",
                text=text,
                time_stamps=SimpleNamespace(
                    items=[SimpleNamespace(text=text, start_time=start, end_time=end)]
                ),
            )

        first_model = SimpleNamespace(
            max_new_tokens=2048,
            transcribe=SimpleNamespace(),
        )
        first_model.transcribe = Mock(
            side_effect=[
                [result(repeated, 0, 40)],
                [result("前半", 0, 10)],
                RuntimeError("right child failed"),
            ]
        )
        audio = SimpleNamespace(
            sample_rate=16000,
            slice=lambda *_args, **_kwargs: np.zeros(640000, dtype=np.float32),
        )
        completed = {}
        writes = []
        arguments = dict(
            video=Path("source.mp4"),
            chunk_dir=None,
            config=ASRConfig(chunk_context_seconds=0),
            core_start=0,
            core_end=40,
            media_duration=40,
            final_chunk=True,
            label="retry",
            audio_buffer=audio,
            completed_ranges=completed,
            completed_range_callback=lambda: writes.append(dict(completed)),
        )
        with self.assertRaisesRegex(RuntimeError, "right child failed"):
            _transcribe_range(first_model, **arguments)

        self.assertIn("0.000:20.000:0", completed)
        self.assertEqual(len(writes), 1)

        second_model = SimpleNamespace(max_new_tokens=2048)
        second_model.transcribe = Mock(
            side_effect=[
                [result(repeated, 0, 40)],
                [result("后半", 0, 10)],
            ]
        )
        recovered = _transcribe_range(second_model, **arguments)

        self.assertEqual(second_model.transcribe.call_count, 2)
        self.assertEqual([cue["text"] for cue in recovered["cues"]], ["前半", "后半"])

    def test_singing_regions_do_not_take_ownership_from_speech_timing(self):
        result = _analysis_regions(
            AudioAnalysis(
                speech=[AudioRegion(10, 30, "speech", "A")],
                singing=[AudioRegion(15, 25, "singing", "A")],
            )
        )

        self.assertEqual(
            [(item.start, item.end, item.kind) for item in result],
            [(10, 30, "speech"), (15, 25, "singing")],
        )

    def test_strong_acoustic_speech_without_diarization_adds_supplemental_route(self):
        phrase = AcousticPhrase(
            12,
            20,
            "high",
            "strong",
            0.8,
            0.7,
            0.6,
            0.9,
            0.0,
            True,
            True,
        )

        result = _analysis_regions(
            AudioAnalysis(
                [],
                [AudioRegion(12, 20, "singing")],
                acoustic_phrases=[phrase],
            )
        )

        self.assertEqual(
            [(item.kind, item.asr_route) for item in result],
            [("speech", "song_speech_fallback"), ("singing", "qwen")],
        )

    def test_aligner_cues_are_clamped_to_owned_analysis_region(self):
        result = SimpleNamespace(
            text="前後",
            time_stamps=SimpleNamespace(
                items=[
                    SimpleNamespace(text="前", start_time=8.0, end_time=10.2),
                    SimpleNamespace(text="後", start_time=10.2, end_time=11.0),
                ]
            ),
        )

        cues = _result_to_cues(
            result,
            offset=0,
            keep_start=5,
            keep_end=10,
            final_chunk=True,
        )

        self.assertEqual(
            [(cue.start, cue.end, cue.text) for cue in cues],
            [(8.0, 10, "前")],
        )

    def test_singing_windows_target_thirty_seconds_and_balance_tail(self):
        self.assertEqual(
            _song_windows(0, 45, 30, 2),
            [(0, 27), (25, 45)],
        )
        self.assertEqual(
            _song_windows(5, 68, 30, 2),
            [(5, 35), (33, 68)],
        )

    def test_singing_windows_treat_thirty_five_to_thirty_eight_as_normal(self):
        audit = []

        windows = _song_windows(0, 36, 30, 2, audit=audit)

        self.assertEqual(windows, [(0, 36)])
        self.assertEqual(audit[0]["cut_reason"], "region_end")

    def test_singing_windows_allow_a_thirty_seven_second_tail(self):
        windows = _song_windows(
            0,
            70,
            30,
            2,
            candidates=[
                _SongCutCandidate(35, "non_singing_gap", -1.0),
                _SongCutCandidate(30, "vocal_energy_valley", 0.1),
            ],
        )

        self.assertEqual(windows, [(0, 35), (33, 70)])

    def test_singing_windows_prefer_cut_evidence_lexicographically(self):
        audit = []
        windows = _song_windows(
            0,
            62,
            30,
            2,
            candidates=[
                _SongCutCandidate(30, "vocal_energy_valley", 0.01),
                _SongCutCandidate(28, "phrase_boundary"),
                _SongCutCandidate(29, "non_singing_gap", -1.2),
            ],
            audit=audit,
        )

        self.assertEqual(windows, [(0, 29), (27, 62)])
        self.assertEqual(audit[0]["cut_reason"], "non_singing_gap")

    def test_singing_windows_use_energy_valley_before_phrase_boundary(self):
        windows = _song_windows(
            0,
            60,
            30,
            2,
            candidates=[
                _SongCutCandidate(27, "vocal_energy_valley", 0.05),
                _SongCutCandidate(30, "vocal_energy_valley", 0.20),
                _SongCutCandidate(29, "phrase_boundary"),
            ],
        )

        self.assertEqual(windows, [(0, 27), (25, 60)])

    def test_singing_cut_candidates_detect_long_vocal_gap(self):
        samples = np.ones(60000, dtype=np.float32)
        samples[29000:31000] = 0
        audio = SimpleNamespace(
            sample_rate=1000,
            slice=lambda start, end: samples[round(start * 1000) : round(end * 1000)],
        )

        candidates = _song_cut_candidates(
            AudioRegion(0, 60, "singing"),
            audio,
            AudioAnalysisConfig(singing_phrase_silence_seconds=0.45),
        )

        gaps = [item for item in candidates if item.kind == "non_singing_gap"]
        self.assertTrue(gaps)
        self.assertAlmostEqual(gaps[0].time, 30, delta=0.2)

    def test_singing_asr_does_not_receive_general_stream_context(self):
        model = SimpleNamespace()
        model.transcribe = Mock(
            return_value=[SimpleNamespace(text="聞こえた歌詞", language="Japanese")]
        )
        audio = SimpleNamespace(
            sample_rate=16000,
            slice=lambda *_args, **_kwargs: np.zeros(16000, dtype=np.float32),
        )

        _transcribe_song_range(
            model,
            Path("source.mp4"),
            None,
            ASRConfig(
                context="配信名や人物名を含む一般コンテキスト",
                language="Japanese",
            ),
            AudioRegion(0, 1, "singing"),
            label="song",
            window_config=AudioAnalysisConfig(),
            audio_buffer=audio,
        )

        self.assertEqual(model.transcribe.call_args.kwargs["context"], "")
        self.assertIsNone(model.transcribe.call_args.kwargs["language"])

    def test_singing_asr_retries_repetition_with_shorter_windows(self):
        repeated = "同じ長い歌詞を繰り返してしまう" * 12
        model = SimpleNamespace()
        model.transcribe = Mock(
            side_effect=[
                [SimpleNamespace(text=repeated, language="Japanese")],
                [SimpleNamespace(text="前半の歌詞", language="Japanese")],
                [SimpleNamespace(text="ready set and find out", language="English")],
            ]
        )
        audio = SimpleNamespace(
            sample_rate=16000,
            slice=lambda *_args, **_kwargs: np.zeros(320000, dtype=np.float32),
        )

        record = _transcribe_song_range(
            model,
            Path("source.mp4"),
            None,
            ASRConfig(),
            AudioRegion(0, 20, "singing"),
            label="song",
            window_config=AudioAnalysisConfig(
                singing_asr_target_seconds=30,
                singing_asr_min_seconds=20,
                singing_asr_max_seconds=38,
                singing_asr_overlap_seconds=2,
            ),
            audio_buffer=audio,
        )

        self.assertEqual(model.transcribe.call_count, 3)
        self.assertEqual(record["language"], "mixed")
        self.assertEqual(
            [(item["start"], item["end"]) for item in record["cues"]],
            [(0, 10), (10, 20)],
        )
        self.assertTrue(
            any(
                item.get("cut_reason") == "repetition_retry"
                for item in record["singing_windows"]
            )
        )

    def test_removes_repeated_text_from_overlapping_song_window(self):
        self.assertEqual(
            _remove_text_overlap("君とここで歌う", "ここで歌う明日へ"),
            "明日へ",
        )

    def test_detects_repeated_asr_generation_loop(self):
        phrase = "私が食べてるのでちょっとこっちに移動します"
        repetition = _repetition_hallucination(phrase * 10)
        self.assertIsNotNone(repetition)
        assert repetition is not None
        self.assertGreaterEqual(repetition[1], 4)

    def test_does_not_flag_short_natural_repetition(self):
        self.assertIsNone(_repetition_hallucination("はいはいはいはい、大丈夫です"))

    def test_aligner_cues_do_not_restore_asr_punctuation(self):
        result = SimpleNamespace(
            text="こんにちは、今日はライブです。",
            time_stamps=SimpleNamespace(
                items=[
                    SimpleNamespace(text="こんにちは", start_time=0, end_time=1),
                    SimpleNamespace(text="今日", start_time=1, end_time=2),
                    SimpleNamespace(text="は", start_time=2, end_time=3),
                    SimpleNamespace(text="ライブ", start_time=3, end_time=4),
                    SimpleNamespace(text="です", start_time=4, end_time=5),
                ]
            ),
        )

        self.assertEqual(
            [
                cue.text
                for cue in _result_to_cues(
                    result,
                    offset=0,
                    keep_start=0,
                    keep_end=5,
                    final_chunk=True,
                )
            ],
            ["こんにちは", "今日", "は", "ライブ", "です"],
        )

    def test_caches_each_completed_audio_chunk_and_reuses_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "source.mp4"
            destination = root / "source.qwen3-asr.srt"
            video.write_bytes(b"video")
            config = ASRConfig(chunk_seconds=10, chunk_context_seconds=1)
            item = SimpleNamespace(text="一", start_time=1, end_time=2)
            result = SimpleNamespace(
                language="Japanese",
                text="一。",
                time_stamps=SimpleNamespace(items=[item]),
            )
            calls = []
            model = SimpleNamespace(
                transcribe=lambda **kwargs: calls.append(kwargs) or [result]
            )
            audio = SimpleNamespace(
                sample_rate=16000,
                slice=lambda *_args, **_kwargs: np.zeros(16000, dtype=np.float32),
            )

            with (
                patch("subtitle_pipeline.asr._media_duration", return_value=15),
                patch(
                    "subtitle_pipeline.asr._load_qwen_model", return_value=model
                ) as load_model,
                patch("subtitle_pipeline.asr.AudioBufferPool.main", return_value=audio),
            ):
                transcribe_with_qwen(video, destination, config)
                transcribe_with_qwen(video, destination, config)

            load_model.assert_called_once()
            self.assertTrue(calls)
            self.assertTrue(
                all(
                    isinstance(call["audio"], tuple)
                    and call["audio"][1] == 16000
                    and isinstance(call["audio"][0], np.ndarray)
                    for call in calls
                )
            )
            self.assertTrue((root / "asr-cache.json").is_file())
            self.assertFalse((root / "asr-chunks").exists())
            self.assertEqual(destination.read_text(encoding="utf-8").count("一"), 2)
            self.assertNotIn("一。", destination.read_text(encoding="utf-8"))

    def test_rejects_zero_duration_cached_cue(self):
        self.assertFalse(
            _valid_cached_record({"cues": [{"start": 1.0, "end": 1.0, "text": "字幕"}]})
        )

    def test_rejects_cached_chunk_with_repeated_generation_loop(self):
        phrase = "私が食べてるのでちょっとこっちに移動します"
        self.assertFalse(
            _valid_cached_record(
                {
                    "text": phrase * 10,
                    "cues": [{"start": 1.0, "end": 2.0, "text": "字幕"}],
                }
            )
        )

    def test_retries_repetition_with_two_shorter_chunks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "source.mp4"
            destination = root / "source.qwen3-asr.srt"
            video.write_bytes(b"video")
            config = ASRConfig(chunk_seconds=40, chunk_context_seconds=1)
            phrase = "私が食べてるのでちょっとこっちに移動します"
            looping = SimpleNamespace(
                language="Japanese",
                text=phrase * 10,
                time_stamps=SimpleNamespace(items=[]),
            )
            left = SimpleNamespace(
                language="Japanese",
                text="左。",
                time_stamps=SimpleNamespace(
                    items=[SimpleNamespace(text="左", start_time=5, end_time=6)]
                ),
            )
            right = SimpleNamespace(
                language="Japanese",
                text="右。",
                time_stamps=SimpleNamespace(
                    items=[SimpleNamespace(text="右", start_time=5, end_time=6)]
                ),
            )

            class Model:
                def __init__(self):
                    self.results = iter([looping, left, right])
                    self.calls = 0

                def transcribe(self, **_kwargs):
                    self.calls += 1
                    return [next(self.results)]

            model = Model()
            audio = SimpleNamespace(
                sample_rate=16000,
                slice=lambda *_args, **_kwargs: np.zeros(16000, dtype=np.float32),
            )
            with (
                patch("subtitle_pipeline.asr._media_duration", return_value=40),
                patch("subtitle_pipeline.asr._load_qwen_model", return_value=model),
                patch("subtitle_pipeline.asr.AudioBufferPool.main", return_value=audio),
            ):
                transcribe_with_qwen(video, destination, config)

            self.assertEqual(model.calls, 3)
            self.assertIn("左", destination.read_text(encoding="utf-8"))
            self.assertIn("右", destination.read_text(encoding="utf-8"))
            self.assertNotIn("。", destination.read_text(encoding="utf-8"))
            cache = __import__("json").loads(
                (root / "asr-cache.json").read_text(encoding="utf-8")
            )
            self.assertTrue(cache["chunks"]["0"]["recovered_from_repetition"])


if __name__ == "__main__":
    unittest.main()
