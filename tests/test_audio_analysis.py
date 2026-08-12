import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from subtitle_pipeline.audio_analysis import (
    AudioRegion,
    _arbitrate_singing_regions,
    _coalesce_singing_phrases,
    _extract_audio,
    _mark_overlaps,
    _merge_regions,
    _overlap_spans,
    _separated_source_is_usable,
    _singing_evidence_score,
    _singing_regions_from_scores,
    _subtract_regions,
)
from subtitle_pipeline.speakers import (
    _can_embed_region,
    _evenly_spaced,
    _identity_candidates,
    _load_profiles,
    _profile_centers,
    _profile_distance,
    _profile_match_is_confident,
    _update_profile,
    load_character_styles,
    metadata_character,
)


class AudioAnalysisTests(unittest.TestCase):
    def test_rejects_silent_and_tiny_separated_speaker_sources(self):
        segment = type("Segment", (), {"start": 0.1, "end": 0.4})()
        self.assertFalse(
            _separated_source_is_usable(np.zeros(8000), [segment], 16000)
        )
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

    def test_singing_evidence_must_outscore_speech(self):
        self.assertEqual(_singing_evidence_score(0.2, 0.6), 0.0)
        self.assertAlmostEqual(_singing_evidence_score(0.8, 0.1), 0.7)

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
        self.assertEqual(
            singing, [AudioRegion(0, 35, "singing", confidence=0.7)]
        )
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
        self.assertEqual(
            ambiguous, [AudioRegion(10, 20, "singing", confidence=0.4)]
        )

    def test_confirmed_song_fragments_are_coalesced_to_long_asr_blocks(self):
        result = _coalesce_singing_phrases(
            [
                AudioRegion(45.0, 61.742, "singing", confidence=0.7),
                AudioRegion(63.837, 65.068, "singing", confidence=0.7),
                AudioRegion(70.06, 117.5, "singing", confidence=0.7),
            ],
            minimum_seconds=30,
        )
        self.assertEqual(
            result, [AudioRegion(45.0, 117.5, "singing", confidence=0.7)]
        )

    def test_short_vocal_episode_remains_ambiguous(self):
        singing, ambiguous = _arbitrate_singing_regions(
            [AudioRegion(0, 20, "singing", confidence=0.8)],
            [AudioRegion(0, 20, "singing", confidence=0.8)],
            [],
            speech_bgm_coverage=0.35,
            release_seconds=35,
            minimum_singing_seconds=30,
        )
        self.assertEqual(singing, [])
        self.assertEqual(
            ambiguous, [AudioRegion(0, 20, "singing", confidence=0.8)]
        )

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

    def test_separated_overlap_track_can_be_embedded_but_raw_overlap_cannot(self):
        self.assertTrue(
            _can_embed_region(
                AudioRegion(
                    0, 20, "speech", "A", overlap=True, source_path="a.wav"
                )
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
        self.assertTrue(
            any(np.allclose(center, [1.0, 0.0]) for center in centers)
        )
        self.assertTrue(
            any(np.allclose(center, [0.0, 1.0]) for center in centers)
        )

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
