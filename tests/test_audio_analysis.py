import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from subtitle_pipeline.audio_analysis import (
    AudioRegion,
    DiarizationTimelines,
    _annotate_exclusive_overlaps,
    _arbitrate_singing_regions,
    _coalesce_singing_phrases,
    _exclude_timeline_regions,
    _extract_audio,
    _mark_overlaps,
    _maximum_concurrent_speakers,
    _merge_regions,
    _overlap_route,
    _overlap_spans,
    _run_initial_audio_analysis,
    _run_mossformer2_worker,
    _run_overlap_separation,
    _separated_source_is_usable,
    _singing_evidence_score,
    _singing_regions_from_scores,
    _subtract_regions,
)
from subtitle_pipeline.audio_buffer import AudioBufferPool
from subtitle_pipeline.config import AudioAnalysisConfig
from subtitle_pipeline.speakers import (
    _aggregate_profile_distances,
    _can_embed_region,
    _evenly_spaced,
    _extract_eres2netv2_embeddings,
    _identity_candidates,
    _load_profiles,
    _moss_identity_candidates,
    _profile_centers,
    _profile_distance,
    _profile_match_is_confident,
    _update_profile,
    identify_speakers,
    load_character_styles,
    metadata_character,
)


class AudioAnalysisTests(unittest.TestCase):
    def test_initial_diarization_and_ast_can_run_concurrently(self):
        barrier = threading.Barrier(2)

        def diarize(*_args):
            barrier.wait(timeout=1)
            regions = [AudioRegion(0, 1, "speech")]
            return DiarizationTimelines(regions, regions)

        def singing(*_args):
            barrier.wait(timeout=1)
            return [AudioRegion(0, 1, "singing")]

        with (
            patch(
                "subtitle_pipeline.audio_analysis._run_diarization",
                side_effect=diarize,
            ),
            patch(
                "subtitle_pipeline.audio_analysis._score_singing_windows",
                side_effect=singing,
            ),
        ):
            timelines, scores = _run_initial_audio_analysis(
                Path("source.mp4"),
                Path("job"),
                np.zeros((1, 16000), dtype=np.float32),
                16000,
                AudioAnalysisConfig(
                    device="cpu",
                    initial_analysis_concurrency=2,
                    diarization_backend="pyannote",
                ),
                {},
            )

        self.assertEqual(timelines.exclusive[0].kind, "speech")
        self.assertEqual(scores[0].kind, "singing")

    def test_overlap_routes_use_real_intersection_thresholds(self):
        self.assertEqual(
            _overlap_route(1.499, conditioned_seconds=1.5),
            "exclusive",
        )
        self.assertEqual(
            _overlap_route(1.5, conditioned_seconds=1.5),
            "conditioned",
        )

    def test_exclusive_timeline_is_annotated_from_ordinary_overlap(self):
        ordinary = [
            AudioRegion(0, 4, "speech", "S0"),
            AudioRegion(2.75, 4.5, "speech", "S1"),
        ]
        exclusive = [
            AudioRegion(0, 3.5, "speech", "S0", anonymous_speaker="S0"),
            AudioRegion(3.5, 4.5, "speech", "S1", anonymous_speaker="S1"),
        ]

        result = _annotate_exclusive_overlaps(
            exclusive,
            ordinary,
            conditioned_seconds=1.0,
        )

        overlap = [item for item in result if item.overlap]
        clean = [item for item in result if not item.overlap]
        self.assertEqual(
            [(item.start, item.end) for item in overlap],
            [(2.75, 3.5), (3.5, 4)],
        )
        self.assertEqual(
            [(item.start, item.end) for item in clean], [(0, 2.75), (4, 4.5)]
        )
        self.assertTrue(all(item.overlap_seconds == 1.25 for item in overlap))
        self.assertTrue(all(item.overlap_speakers == ("S0", "S1") for item in overlap))
        self.assertTrue(all(item.asr_route == "conditioned" for item in overlap))

    def test_song_regions_are_removed_from_overlap_diarization_timeline(self):
        result = _exclude_timeline_regions(
            [AudioRegion(0, 10, "speech", "S0", anonymous_speaker="S0")],
            [AudioRegion(3, 7, "singing")],
        )

        self.assertEqual(
            [(item.start, item.end, item.anonymous_speaker) for item in result],
            [(0, 3, "S0"), (7, 10, "S0")],
        )

    def test_moss_identity_candidates_exclude_overlap_and_trim_switch_edges(self):
        config = AudioAnalysisConfig(
            speaker_identity_edge_trim_seconds=0.2,
            speaker_identity_min_segment_seconds=1.5,
            speaker_identity_max_weight_seconds=10.0,
        )
        regions = [
            AudioRegion(0, 22, "speech", "MOSS_W000_S01"),
            AudioRegion(30, 35, "speech", "MOSS_W000_S02", overlap=True),
            AudioRegion(40, 41, "speech", "MOSS_W000_S01"),
        ]

        candidates = _moss_identity_candidates(regions, config)

        self.assertEqual(len(candidates), 3)
        self.assertTrue(all(item.speaker == "MOSS_W000_S01" for item in candidates))
        self.assertAlmostEqual(candidates[0].start, 0.2)
        self.assertAlmostEqual(candidates[-1].end, 21.8)
        self.assertTrue(all(item.end - item.start <= 10.0 for item in candidates))

    def test_moss_identity_distance_trims_outlier_before_duration_weighting(self):
        evidence = [(np.asarray([1.0, 0.0], dtype=np.float32), 1.0) for _ in range(9)]
        evidence.append((np.asarray([0.0, 1.0], dtype=np.float32), 10.0))
        profiles = {
            "member_a": np.asarray([[1.0, 0.0]], dtype=np.float32),
            "member_b": np.asarray([[0.0, 1.0]], dtype=np.float32),
        }

        distances = _aggregate_profile_distances(
            evidence,
            profiles,
            trim_ratio=0.15,
            maximum_weight=10.0,
        )

        self.assertEqual(distances[0][1], "member_a")
        self.assertAlmostEqual(distances[0][0], 0.0)

    def test_moss_labels_map_independently_and_overlap_inherits_identity(self):
        regions = [
            AudioRegion(0, 4, "speech", "MOSS_W000_S01"),
            AudioRegion(5, 9, "speech", "MOSS_W000_S02"),
            AudioRegion(10, 12, "speech", "MOSS_W000_S01", overlap=True),
        ]
        seen_candidates = []

        def snippets(_waveform, _rate, candidates, **_kwargs):
            seen_candidates.extend(candidates)
            return [(None, 16000) for _candidate in candidates]

        profiles = {
            "fuji_miyako": np.asarray([[1.0, 0.0]], dtype=np.float32),
            "sengoku_yuno": np.asarray([[0.0, 1.0]], dtype=np.float32),
        }
        with (
            patch(
                "subtitle_pipeline.speakers._candidate_snippets", side_effect=snippets
            ),
            patch(
                "subtitle_pipeline.speakers._extract_embeddings",
                return_value=[np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0])],
            ),
            patch("subtitle_pipeline.speakers._load_profiles", return_value=profiles),
        ):
            resolved = identify_speakers(
                np.zeros((1, 16000), dtype=np.float32),
                16000,
                regions,
                AudioAnalysisConfig(device="cpu"),
            )

        self.assertEqual(len(seen_candidates), 2)
        self.assertTrue(all(not item.overlap for item in seen_candidates))
        self.assertEqual(
            [item.speaker for item in resolved],
            [
                "fuji_miyako",
                "fuji_miyako",
                "fuji_miyako",
            ],
        )
        self.assertEqual(
            [item.anonymous_speaker for item in resolved],
            ["MOSS_W000_S01", "MOSS_W000_S02", "MOSS_W000_S01"],
        )

    def test_mossformer_worker_uses_shared_memory_descriptors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "source.mp4"
            video.write_bytes(b"video")
            pool = AudioBufferPool(video, root, 1.0)
            source = pool.add(np.ones(16000, dtype=np.float32), 16000)
            response = SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"items": [{"id": 0}]}),
                stderr="",
            )
            with (
                patch.object(pool, "main", return_value=source),
                patch(
                    "subtitle_pipeline.audio_analysis.subprocess.run",
                    return_value=response,
                ) as invoke,
            ):
                outputs = _run_mossformer2_worker(
                    np.zeros((1, 16000), dtype=np.float32),
                    16000,
                    [(0, 0.0, 1.0)],
                    AudioAnalysisConfig(),
                    root / "debug",
                    audio_pool=pool,
                )
            payload = json.loads(invoke.call_args.kwargs["input"])
            self.assertIn("audio", payload)
            self.assertNotIn("input_path", payload["items"][0])
            self.assertEqual(len(payload["items"][0]["outputs"]), 2)
            self.assertEqual(len(outputs[0]), 2)
            self.assertFalse((root / "debug").exists())
            pool.close()

    def test_eres2net_worker_uses_one_shared_audio_batch(self):
        import torch

        response = SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"embeddings": [[0.1, 0.2]]}),
            stderr="",
        )
        with patch(
            "subtitle_pipeline.speakers.subprocess.run", return_value=response
        ) as invoke:
            embeddings = _extract_eres2netv2_embeddings(
                [(torch.ones((1, 32000)), 16000)], AudioAnalysisConfig()
            )
        payload = json.loads(invoke.call_args.kwargs["input"])
        self.assertIn("audio", payload)
        self.assertNotIn("paths", payload)
        self.assertEqual(
            payload["items"], [{"id": 0, "start_sample": 0, "end_sample": 32000}]
        )
        self.assertEqual(embeddings, [[0.1, 0.2]])

    def test_counts_distinct_simultaneous_speakers(self):
        regions = [
            AudioRegion(0, 5, "speech", "A"),
            AudioRegion(1, 4, "speech", "B"),
            AudioRegion(2, 3, "speech", "C"),
            AudioRegion(2.5, 6, "speech", "A"),
        ]

        self.assertEqual(_maximum_concurrent_speakers(regions, 0, 6), 3)
        self.assertEqual(_maximum_concurrent_speakers(regions, 3, 6), 2)

    def test_rejects_silent_and_tiny_separated_speaker_sources(self):
        segment = type("Segment", (), {"start": 0.1, "end": 0.4})()
        self.assertFalse(_separated_source_is_usable(np.zeros(8000), [segment], 16000))
        tiny = type("Segment", (), {"start": 0.1, "end": 0.108})()
        self.assertFalse(
            _separated_source_is_usable(np.full(8000, 0.01), [tiny], 16000)
        )
        self.assertTrue(
            _separated_source_is_usable(np.full(8000, 0.01), [segment], 16000)
        )

    def test_audio_is_reextracted_when_analysis_cache_is_invalid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "source.mp4"
            destination = root / "source.analysis.wav"
            video.write_bytes(b"new video")
            destination.write_bytes(b"stale audio")

            with (
                patch(
                    "subtitle_pipeline.audio_analysis.require_command",
                    return_value="ffmpeg",
                ),
                patch("subtitle_pipeline.audio_analysis.run") as run,
            ):
                _extract_audio(video, destination)

            run.assert_called_once()

    def test_speech_does_not_erase_independent_singing_evidence(self):
        self.assertAlmostEqual(_singing_evidence_score(0.2, 0.6), 0.2)
        self.assertAlmostEqual(_singing_evidence_score(0.8, 0.1), 0.8)

    def test_marks_simultaneous_different_speakers(self):
        result = _mark_overlaps(
            [
                AudioRegion(0, 3, "speech", "SPEAKER_00"),
                AudioRegion(2, 4, "speech", "SPEAKER_01"),
                AudioRegion(5, 6, "speech", "SPEAKER_00"),
            ]
        )
        self.assertEqual([item.overlap for item in result], [True, True, False])

    def test_merges_nearby_singing_windows(self):
        result = _merge_regions(
            [
                AudioRegion(0, 5, "singing", confidence=0.6),
                AudioRegion(4, 9, "singing", confidence=0.8),
                AudioRegion(12, 15, "singing", confidence=0.7),
            ],
            1.0,
        )
        self.assertEqual(
            result,
            [
                AudioRegion(0, 9, "singing", confidence=0.8),
                AudioRegion(12, 15, "singing", confidence=0.7),
            ],
        )

    def test_singing_hysteresis_bridges_short_ast_miss(self):
        windows = [
            AudioRegion(index * 5, index * 5 + 5, "singing", confidence=score)
            for index, score in enumerate([0.8, 0.7, 0.0, 0.0, 0.0, 0.8, 0.7])
        ]
        result = _singing_regions_from_scores(
            windows,
            threshold=0.05,
            smoothing_windows=3,
            release_seconds=35,
        )
        self.assertEqual(result, [AudioRegion(0, 35, "singing", confidence=0.8)])

    def test_singing_hysteresis_ends_after_sustained_ast_miss(self):
        windows = [
            AudioRegion(index * 10, index * 10 + 5, "singing", confidence=score)
            for index, score in enumerate([0.8, 0.7, 0.0, 0.0, 0.0, 0.0, 0.8, 0.7])
        ]
        result = _singing_regions_from_scores(
            windows,
            threshold=0.05,
            smoothing_windows=3,
            release_seconds=20,
        )
        self.assertEqual(
            result,
            [
                AudioRegion(0, 15, "singing", confidence=0.8),
                AudioRegion(60, 75, "singing", confidence=0.8),
            ],
        )

    def test_song_state_covers_calls_instrumentals_and_short_speech(self):
        windows = [
            AudioRegion(
                0,
                5,
                "singing",
                confidence=0.8,
                speech_confidence=0.2,
                music_confidence=0.7,
            ),
            AudioRegion(
                5,
                10,
                "singing",
                confidence=0.0,
                speech_confidence=0.8,
                music_confidence=0.7,
            ),
            AudioRegion(10, 15, "singing", music_confidence=0.8),
            AudioRegion(15, 20, "singing", speech_confidence=0.9),
            AudioRegion(
                20,
                25,
                "singing",
                confidence=0.8,
                music_confidence=0.7,
            ),
        ]

        result = _singing_regions_from_scores(
            windows,
            threshold=0.05,
            music_threshold=0.05,
            smoothing_windows=1,
            release_seconds=35,
        )

        self.assertEqual(result, [AudioRegion(0, 25, "singing", confidence=0.8)])

    def test_music_and_speech_cannot_start_song_without_singing_anchor(self):
        windows = [
            AudioRegion(
                index * 5,
                index * 5 + 5,
                "singing",
                speech_confidence=0.8,
                music_confidence=0.7,
            )
            for index in range(4)
        ]

        result = _singing_regions_from_scores(
            windows,
            threshold=0.05,
            music_threshold=0.05,
            smoothing_windows=1,
            release_seconds=35,
        )

        self.assertEqual(result, [])

    def test_raw_singing_hit_with_music_rescues_short_song(self):
        result = _singing_regions_from_scores(
            [
                AudioRegion(0, 5, "singing", music_confidence=0.7),
                AudioRegion(
                    5,
                    10,
                    "singing",
                    confidence=0.06,
                    music_confidence=0.7,
                ),
                AudioRegion(10, 15, "singing", music_confidence=0.7),
            ],
            threshold=0.05,
            music_threshold=0.05,
            smoothing_windows=3,
            release_seconds=35,
        )

        self.assertEqual(result, [AudioRegion(0, 15, "singing", confidence=0.06)])

    def test_sustained_speech_takeover_closes_after_final_anchor(self):
        result = _singing_regions_from_scores(
            [
                AudioRegion(
                    0,
                    5,
                    "singing",
                    confidence=0.8,
                    music_confidence=0.7,
                ),
                AudioRegion(
                    5,
                    10,
                    "singing",
                    speech_confidence=0.8,
                    music_confidence=0.2,
                ),
            ],
            threshold=0.05,
            music_threshold=0.05,
            smoothing_windows=1,
            release_seconds=35,
        )

        self.assertEqual(result, [AudioRegion(0, 5, "singing", confidence=0.8)])

    def test_song_state_audit_records_evidence_and_transitions(self):
        audit = {}
        _singing_regions_from_scores(
            [
                AudioRegion(
                    0,
                    5,
                    "singing",
                    confidence=0.8,
                    music_confidence=0.6,
                ),
                AudioRegion(5, 10, "singing", music_confidence=0.7),
            ],
            threshold=0.05,
            music_threshold=0.05,
            smoothing_windows=1,
            release_seconds=35,
            audit=audit,
        )

        self.assertEqual(audit["windows"][1]["state"], "in_song")
        self.assertEqual(
            [(item["from"], item["to"]) for item in audit["transitions"]],
            [("outside", "in_song"), ("in_song", "outside")],
        )

    def test_singing_score_smoothing_removes_isolated_ast_hit(self):
        windows = [
            AudioRegion(index * 2.5, index * 2.5 + 5, "singing", confidence=score)
            for index, score in enumerate([0.0, 0.0, 0.9, 0.0, 0.0])
        ]
        self.assertEqual(
            _singing_regions_from_scores(
                windows,
                threshold=0.05,
                smoothing_windows=3,
                release_seconds=35,
            ),
            [],
        )

    def test_singing_score_smoothing_removes_isolated_edge_hit(self):
        windows = [
            AudioRegion(index * 2.5, index * 2.5 + 5, "singing", confidence=score)
            for index, score in enumerate([0.9, 0.0, 0.0])
        ]
        self.assertEqual(
            _singing_regions_from_scores(
                windows,
                threshold=0.05,
                smoothing_windows=3,
                release_seconds=35,
            ),
            [],
        )

    def test_speech_with_bgm_does_not_become_singing(self):
        singing, ambiguous = _arbitrate_singing_regions(
            [AudioRegion(0, 10, "singing", confidence=0.8)],
            [],
            [AudioRegion(0, 8, "speech", "A")],
            speech_bgm_coverage=0.35,
            release_seconds=35,
        )
        self.assertEqual(singing, [])
        self.assertEqual(ambiguous, [])

    def test_vocal_stem_confirms_singing_before_release_hysteresis(self):
        singing, ambiguous = _arbitrate_singing_regions(
            [
                AudioRegion(0, 5, "singing", confidence=0.8),
                AudioRegion(30, 35, "singing", confidence=0.7),
            ],
            [
                AudioRegion(0, 5, "singing", confidence=0.7),
                AudioRegion(30, 35, "singing", confidence=0.6),
            ],
            [],
            speech_bgm_coverage=0.35,
            release_seconds=35,
        )
        self.assertEqual(singing, [AudioRegion(0, 35, "singing", confidence=0.7)])
        self.assertEqual(ambiguous, [])

    def test_uncertain_non_speech_candidate_uses_dual_asr(self):
        singing, ambiguous = _arbitrate_singing_regions(
            [AudioRegion(10, 20, "singing", confidence=0.4)],
            [],
            [AudioRegion(10, 11, "speech", "A")],
            speech_bgm_coverage=0.35,
            release_seconds=35,
        )
        self.assertEqual(singing, [])
        self.assertEqual(ambiguous, [AudioRegion(10, 20, "singing", confidence=0.4)])

    def test_confirmed_song_fragments_are_coalesced_to_long_asr_blocks(self):
        result = _coalesce_singing_phrases(
            [
                AudioRegion(45.0, 61.742, "singing", confidence=0.7),
                AudioRegion(63.837, 65.068, "singing", confidence=0.7),
                AudioRegion(70.06, 117.5, "singing", confidence=0.7),
            ],
            minimum_seconds=30,
        )
        self.assertEqual(result, [AudioRegion(45.0, 117.5, "singing", confidence=0.7)])

    def test_short_vocal_episode_is_confirmed_by_vocal_stem(self):
        singing, ambiguous = _arbitrate_singing_regions(
            [AudioRegion(0, 20, "singing", confidence=0.8)],
            [AudioRegion(0, 20, "singing", confidence=0.8)],
            [],
            speech_bgm_coverage=0.35,
            release_seconds=35,
            minimum_singing_seconds=30,
        )
        self.assertEqual(singing, [AudioRegion(0, 20, "singing", confidence=0.8)])
        self.assertEqual(ambiguous, [])

    def test_limits_separation_to_padded_overlap_region(self):
        regions = [
            AudioRegion(0, 10, "speech", "A"),
            AudioRegion(4, 6, "speech", "B"),
            AudioRegion(12, 14, "speech", "A"),
        ]
        self.assertEqual(_overlap_spans(regions, 20), [(3.25, 6.75)])
        self.assertEqual(
            _subtract_regions(regions[0], [(3.25, 6.75)]),
            [
                AudioRegion(0, 3.25, "speech", "A"),
                AudioRegion(6.75, 10, "speech", "A"),
            ],
        )

    def test_three_speaker_overlap_skips_two_source_separator(self):
        regions = _mark_overlaps(
            [
                AudioRegion(1, 4, "speech", "A"),
                AudioRegion(1.5, 3.5, "speech", "B"),
                AudioRegion(2, 3, "speech", "C"),
            ]
        )
        with patch(
            "subtitle_pipeline.audio_analysis._run_mossformer2_worker"
        ) as worker:
            result = _run_overlap_separation(
                np.zeros((1, 80000), dtype=np.float32),
                16000,
                regions,
                AudioAnalysisConfig(),
                Path("unused"),
            )

        worker.assert_not_called()
        self.assertEqual(result, regions)

    def test_mossformer_replaces_overlap_only_when_both_tracks_are_usable(self):
        import soundfile as sf

        regions = _mark_overlaps(
            [
                AudioRegion(0, 5, "speech", "A"),
                AudioRegion(2, 4, "speech", "B"),
            ]
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [root / "source-0.wav", root / "source-1.wav"]
            for path in paths:
                sf.write(path, np.full(56000, 0.01, dtype=np.float32), 16000)
            with patch(
                "subtitle_pipeline.audio_analysis._run_mossformer2_worker",
                return_value={0: paths},
            ):
                result = _run_overlap_separation(
                    np.zeros((1, 80000), dtype=np.float32),
                    16000,
                    regions,
                    AudioAnalysisConfig(),
                    root,
                )

        separated = [region for region in result if region.source_path]
        self.assertEqual(len(separated), 2)
        self.assertTrue(all(region.overlap for region in separated))
        self.assertEqual(
            {region.speaker for region in separated},
            {"OVERLAP_00000_SOURCE_0", "OVERLAP_00000_SOURCE_1"},
        )

    def test_mossformer_empty_track_falls_back_to_original_regions(self):
        import soundfile as sf

        regions = _mark_overlaps(
            [
                AudioRegion(0, 5, "speech", "A"),
                AudioRegion(2, 4, "speech", "B"),
            ]
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            voiced = root / "voiced.wav"
            silent = root / "silent.wav"
            sf.write(voiced, np.full(56000, 0.01, dtype=np.float32), 16000)
            sf.write(silent, np.zeros(56000, dtype=np.float32), 16000)
            with patch(
                "subtitle_pipeline.audio_analysis._run_mossformer2_worker",
                return_value={0: [voiced, silent]},
            ):
                result = _run_overlap_separation(
                    np.zeros((1, 80000), dtype=np.float32),
                    16000,
                    regions,
                    AudioAnalysisConfig(),
                    root,
                )

        self.assertEqual(result, regions)

    def test_separated_overlap_track_can_be_embedded_but_raw_overlap_cannot(self):
        self.assertTrue(
            _can_embed_region(
                AudioRegion(0, 20, "speech", "A", overlap=True, source_path="a.wav")
            )
        )
        self.assertFalse(
            _can_embed_region(AudioRegion(0, 10, "speech", "A", overlap=True))
        )

    def test_short_overlap_fragments_are_aggregated_for_identity(self):
        regions = [
            AudioRegion(
                10.0,
                10.6,
                "speech",
                "OVERLAP_A",
                overlap=True,
                source_path="a.wav",
                source_offset=10.0,
            ),
            AudioRegion(
                10.8,
                11.4,
                "speech",
                "OVERLAP_A",
                overlap=True,
                source_path="a.wav",
                source_offset=10.0,
            ),
        ]

        candidates = _identity_candidates(regions)

        self.assertEqual(len(candidates), 1)
        self.assertEqual((candidates[0].start, candidates[0].end), (10.0, 11.4))

    def test_channel_metadata_identifies_solo_member(self):
        self.assertEqual(
            metadata_character({"channel": "藤都子 -Fuji Miyako-"}),
            "fuji_miyako",
        )

    def test_character_styles_are_separate_from_translation_glossary(self):
        styles = load_character_styles()
        self.assertEqual(styles["minetsuki_ritsu"].primary_color, "#65A9FF")

    def test_speaker_profiles_are_scoped_to_embedding_model(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            _update_profile(
                directory,
                "minetsuki_ritsu",
                [np.asarray([1.0, 0.0], dtype=np.float32)],
                "eres2netv2:test-model",
            )
            self.assertIn(
                "minetsuki_ritsu",
                _load_profiles(directory, "eres2netv2:test-model"),
            )
            self.assertEqual(_load_profiles(directory, "wespeaker:test-model"), {})
            payload = json.loads(
                (directory / "minetsuki_ritsu.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["version"], 2)
            self.assertEqual(payload["model"], "eres2netv2:test-model")

    def test_speaker_profile_preserves_distinct_embedding_centers(self):
        embeddings = np.asarray(
            [[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 4,
            dtype=np.float32,
        )

        centers = _profile_centers(
            embeddings,
            max_centers=3,
            min_samples_per_center=3,
        )

        self.assertEqual(centers.shape, (2, 2))
        self.assertTrue(any(np.allclose(center, [1.0, 0.0]) for center in centers))
        self.assertTrue(any(np.allclose(center, [0.0, 1.0]) for center in centers))

    def test_speaker_profile_reduces_centers_for_small_clusters(self):
        embeddings = np.asarray(
            [[1.0, 0.0]] * 9 + [[0.0, 1.0]],
            dtype=np.float32,
        )

        centers = _profile_centers(
            embeddings,
            max_centers=2,
            min_samples_per_center=3,
        )

        self.assertEqual(centers.shape, (1, 2))

    def test_speaker_profile_matches_nearest_of_multiple_centers(self):
        distance = _profile_distance(
            np.asarray([0.0, 1.0], dtype=np.float32),
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )

        self.assertAlmostEqual(distance, 0.0)

    def test_speaker_profile_rejects_ambiguous_nearest_identity(self):
        self.assertFalse(
            _profile_match_is_confident(
                [(0.258, "fuji_miyako"), (0.261, "minetsuki_ritsu")],
                threshold=0.32,
                margin=0.03,
            )
        )

    def test_speaker_profile_accepts_clear_nearest_identity(self):
        self.assertTrue(
            _profile_match_is_confident(
                [(0.143, "minetsuki_ritsu"), (0.304, "fuji_miyako")],
                threshold=0.32,
                margin=0.03,
            )
        )

    def test_speaker_enrollment_samples_across_full_video(self):
        self.assertEqual(_evenly_spaced(list(range(10)), 4), [0, 3, 6, 9])


if __name__ == "__main__":
    unittest.main()
