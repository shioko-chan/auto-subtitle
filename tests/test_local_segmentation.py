import json
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

from subtitle_pipeline.config import SegmentationConfig
from subtitle_pipeline.local_segmentation import (
    Morphology,
    SudachiAnalyzer,
    _choose_cuts_and_scores,
    _score_boundary,
    build_speaker_tracks,
)
from subtitle_pipeline.subtitles import Cue


class FakeAnalyzer:
    versions: ClassVar[dict[str, str]] = {
        "SudachiPy": "test",
        "SudachiDict-core": "test",
    }

    def analyze(self, text):
        return []


class ParticleAnalyzer(FakeAnalyzer):
    def analyze(self, text):
        if text == "私は猫です":
            return [
                Morphology("私", 0, 1, ("代名詞",), "*", "*"),
                Morphology("は", 1, 2, ("助詞", "係助詞"), "*", "*"),
                Morphology("猫", 2, 3, ("名詞",), "*", "*"),
                Morphology("です", 3, 5, ("助動詞",), "助動詞", "終止形-一般"),
            ]
        if text == "私は":
            return [
                Morphology("私", 0, 1, ("代名詞",), "*", "*"),
                Morphology("は", 1, 2, ("助詞", "係助詞"), "*", "*"),
            ]
        if text == "は猫です":
            return [
                Morphology("は", 0, 1, ("助詞", "係助詞"), "*", "*"),
                Morphology("猫", 1, 2, ("名詞",), "*", "*"),
                Morphology("です", 2, 4, ("助動詞",), "助動詞", "終止形-一般"),
            ]
        return []


class LocalSegmentationTests(unittest.TestCase):
    def test_four_gap_bands_score_one_through_four(self):
        for gap, expected in [(0.12, 1), (0.25, 2), (0.4, 3), (0.6, 4)]:
            cues = [Cue(0, 0.1, "左", "A"), Cue(0.1 + gap, 0.4 + gap, "右", "A")]
            self.assertEqual(_score_boundary(cues, [0, 1], 0, 0, 1, []).score, expected)

    def test_duration_rewards_accumulate_at_four_seconds(self):
        cues = [Cue(0, 4.1, "左", "A"), Cue(4.1, 4.5, "右", "A")]
        score = _score_boundary(cues, [0, 1], 0, 0, 1, [])
        self.assertEqual(score.score, 3)
        self.assertIn("duration>=2s:+1", score.factors)
        self.assertIn("duration>=4s:+2", score.factors)

    def test_morphology_rewards_terminal_and_penalizes_connections(self):
        cues = [Cue(0, 1, "行く", "A"), Cue(1, 2, "ので", "A")]
        terminal = Morphology("行く", 0, 2, ("動詞", "一般", "*", "*", "五段", "終止形-一般"), "五段", "終止形-一般")
        particle = Morphology("ので", 2, 4, ("助詞", "接続助詞", "*", "*", "*", "*"), "*", "*")
        score = _score_boundary(cues, [0, 1], 0, 0, 2, [terminal, particle])
        self.assertEqual(score.score, -1)
        self.assertIn("terminal_predicate:+2", score.factors)
        self.assertIn("strong_connection:-3", score.factors)

    def test_inside_sudachi_morpheme_is_strong_connection(self):
        cues = [Cue(0, 1, "夢限", "A"), Cue(1, 2, "大", "A")]
        word = Morphology("夢限大", 0, 3, ("名詞", "固有名詞"), "*", "*")
        score = _score_boundary(cues, [0, 1], 0, 0, 2, [word])
        self.assertIn("strong_connection:-3", score.factors)

    def test_six_second_fallback_selects_a_boundary_after_two_seconds(self):
        cues = [Cue(i * 1.1, i * 1.1 + 1.0, str(i), "A") for i in range(7)]
        cuts, _ = _choose_cuts_and_scores(
            cues, list(range(7)), list(range(1, 8)), [],
            SegmentationConfig(boundary_score_threshold=99),
        )
        self.assertTrue(cuts)
        self.assertGreaterEqual(cuts[0], 2)

    def test_tracks_are_independent_unknown_is_cut_by_known_activity(self):
        cues = [
            Cue(0, 1, "A1", "A"), Cue(0.5, 1.2, "U1"),
            Cue(1, 2, "B", "B"), Cue(1.3, 2.0, "U2"), Cue(1.2, 2.2, "A2", "A"),
        ]
        tracks, _ = build_speaker_tracks(cues, SegmentationConfig(), analyzer=FakeAnalyzer())
        values = {track.key: track for track in tracks}
        self.assertEqual(set(values), {"A", "B", "unknown"})
        self.assertEqual(len(values["unknown"].units), 2)
        self.assertEqual(len(values["A"].units), 1)

    def test_unknown_between_same_speaker_is_bridged_before_segmentation(self):
        cues = [
            Cue(0, 0.4, "私", "A"),
            Cue(0.4, 0.7, "は"),
            Cue(0.7, 1.2, "猫です", "A"),
        ]
        tracks, _ = build_speaker_tracks(
            cues, SegmentationConfig(), analyzer=ParticleAnalyzer()
        )
        self.assertEqual([track.key for track in tracks], ["A"])
        self.assertEqual(tracks[0].units[0].source_indices, (0, 1, 2))

    def test_unknown_is_not_bridged_across_competing_speaker_activity(self):
        cues = [
            Cue(0, 0.4, "私", "A"),
            Cue(0.4, 0.7, "は"),
            Cue(0.45, 0.65, "插话", "B"),
            Cue(0.7, 1.2, "猫です", "A"),
        ]
        tracks, _ = build_speaker_tracks(
            cues, SegmentationConfig(), analyzer=ParticleAnalyzer()
        )
        self.assertIn("unknown", {track.key for track in tracks})

    def test_unknown_between_different_speakers_follows_stronger_grammar(self):
        cues = [
            Cue(0, 0.4, "私", "A"),
            Cue(0.4, 0.7, "は"),
            Cue(0.7, 1.2, "猫です", "B"),
        ]
        tracks, _ = build_speaker_tracks(
            cues, SegmentationConfig(), analyzer=ParticleAnalyzer()
        )
        values = {track.key: track for track in tracks}
        self.assertNotIn("unknown", values)
        self.assertIn(1, values["A"].units[0].source_indices)

    def test_atomic_kinds_remain_single_units_and_audit_is_written(self):
        cues = [Cue(0, 1, "歌", "A", "singing"), Cue(1, 2, "重叠", "A", "conditioned_speech")]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "local-segmentation.json"
            tracks, _ = build_speaker_tracks(cues, SegmentationConfig(), analyzer=FakeAnalyzer(), audit_path=path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual([unit.kind for unit in tracks[0].units], ["singing", "conditioned_speech"])
        self.assertIn("episodes", payload)

    def test_real_sudachi_exposes_conjugation(self):
        values = SudachiAnalyzer().analyze("行きました")
        self.assertTrue(any(item.conjugation_form != "*" for item in values))


if __name__ == "__main__":
    unittest.main()
