import unittest

from subtitle_pipeline.audio_analysis import (
    AudioRegion,
    _mark_overlaps,
    _merge_regions,
    _overlap_spans,
    _subtract_regions,
)
from subtitle_pipeline.speakers import (
    _can_embed_region,
    load_character_styles,
    metadata_character,
)


class AudioAnalysisTests(unittest.TestCase):
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
    def test_channel_metadata_identifies_solo_member(self):
        self.assertEqual(
            metadata_character({"channel": "藤都子 -Fuji Miyako-"}),
            "fuji_miyako",
        )

    def test_character_styles_are_separate_from_translation_glossary(self):
        styles = load_character_styles()
        self.assertEqual(styles["minetsuki_ritsu"].primary_color, "#65A9FF")


if __name__ == "__main__":
    unittest.main()
