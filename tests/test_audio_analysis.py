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
    _acoustic_phrase_route,
    _clean_speaker_timeline,
    _exclude_timeline_regions,
    _extract_audio,
    _mark_overlaps,
    _merge_regions,
    _run_initial_audio_analysis,
    _singing_evidence_score,
    _singing_regions_from_scores,
)
from subtitle_pipeline.config import AudioAnalysisConfig
from subtitle_pipeline.speakers import (
    _aggregate_profile_distances,
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
    def test_acoustic_phrase_route_covers_full_singing_speech_matrix(self):
        config = AudioAnalysisConfig(
            singing_threshold=0.2,
            singing_vocal_threshold=0.6,
            singing_speech_takeover_threshold=0.5,
        )
        cases = {
            ("high", "strong"): (0.3, 0.6, 0.7, 0.0, True, True),
            ("high", "weak"): (0.3, 0.2, 0.7, 0.0, True, False),
            ("high", "none"): (0.3, 0.0, 0.7, 0.0, True, False),
            ("medium", "strong"): (0.3, 0.6, 0.2, 0.0, True, True),
            ("medium", "weak"): (0.3, 0.2, 0.2, 0.0, True, False),
            ("medium", "none"): (0.3, 0.0, 0.2, 0.0, True, False),
            ("low", "strong"): (0.1, 0.2, 0.2, 0.1, False, True),
            ("low", "weak"): (0.1, 0.2, 0.2, 0.0, False, False),
            ("low", "none"): (0.1, 0.0, 0.2, 0.0, False, False),
        }
        for levels, values in cases.items():
            singing, speech, vocal, overlap, route_alt, route_speech = values
            with self.subTest(levels=levels):
                self.assertEqual(
                    _acoustic_phrase_route(singing, speech, vocal, overlap, config),
                    (*levels, route_alt, route_speech),
                )

    def test_initial_diarization_and_ast_can_run_concurrently(self):
        barrier = threading.Barrier(2)

        def diarize(*_args):
            barrier.wait(timeout=1)
            return [AudioRegion(0, 1, "speech")]

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
            diarization, scores = _run_initial_audio_analysis(
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

        self.assertEqual(diarization[0].kind, "speech")
        self.assertEqual(scores[0].kind, "singing")

    def test_clean_speaker_timeline_subtracts_ordinary_overlap(self):
        ordinary = [
            AudioRegion(0, 4, "speech", "S0"),
            AudioRegion(2.75, 4.5, "speech", "S1"),
        ]

        result = _clean_speaker_timeline(ordinary)

        self.assertEqual(
            [
                (item.start, item.end, item.speaker, item.anonymous_speaker)
                for item in result
            ],
            [(0, 2.75, "S0", "S0"), (4, 4.5, "S1", "S1")],
        )
        self.assertTrue(all(not item.overlap for item in result))

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

    def test_identity_candidates_use_only_nonoverlap_speech(self):
        regions = [
            AudioRegion(0, 3, "speech", "A"),
            AudioRegion(3, 6, "speech", "A", overlap=True),
            AudioRegion(6, 7, "speech", "A"),
        ]

        self.assertEqual(_identity_candidates(regions), [regions[0]])

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

    def test_singing_candidates_do_not_bridge_classifier_misses(self):
        windows = [
            AudioRegion(index * 10, index * 10 + 5, "singing", confidence=score)
            for index, score in enumerate([0.8, 0.7, 0.0, 0.0, 0.0, 0.0, 0.8, 0.7])
        ]
        result = _singing_regions_from_scores(
            windows,
            threshold=0.05,
            smoothing_windows=3,
        )
        self.assertEqual(
            [(item.start, item.end) for item in result],
            [(0, 5), (10, 15), (60, 65), (70, 75)],
        )

    def test_speech_evidence_does_not_erase_singing_candidate(self):
        windows = [
            AudioRegion(
                0,
                5,
                "singing",
                confidence=0.8,
                speech_confidence=0.2,
                music_confidence=0.7,
            ),
        ]

        result = _singing_regions_from_scores(
            windows,
            threshold=0.05,
            music_threshold=0.05,
            smoothing_windows=1,
        )

        self.assertEqual(result, [AudioRegion(0, 5, "singing", confidence=0.8)])

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
        )

        self.assertEqual(result, [AudioRegion(5, 10, "singing", confidence=0.06)])

    def test_singing_candidate_audit_records_independent_decisions(self):
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
            audit=audit,
        )

        self.assertTrue(audit["windows"][0]["selected"])
        self.assertFalse(audit["windows"][1]["selected"])
        self.assertEqual(len(audit["candidates"]), 1)

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
            ),
            [],
        )

    def test_channel_metadata_identifies_solo_member(self):
        self.assertEqual(
            metadata_character({"channel": "藤都子 -Fuji Miyako-"}),
            "fuji_miyako",
        )

    def test_character_styles_are_separate_from_translation_glossary(self):
        styles = load_character_styles()
        self.assertEqual(styles["minetsuki_ritsu"].primary_color, "#FFFFFF")
        self.assertEqual(styles["minetsuki_ritsu"].outline_color, "#65A9FF")

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
