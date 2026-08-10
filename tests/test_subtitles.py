import json
import tempfile
import unittest
from pathlib import Path

from subtitle_pipeline.subtitles import (
    Cue,
    apply_translations,
    clean_non_speech_markers,
    merge_semantic_cues,
    read_subtitles,
    translation_payload,
    write_srt,
)
from subtitle_pipeline.config import SegmentationConfig


class SubtitleTests(unittest.TestCase):
    def test_srt_round_trip_preserves_timing_and_unicode(self):
        cues = [
            Cue(1.234, 3.456, "你好，世界"),
            Cue(65.0, 66.001, "第二行\n仍是同一条"),
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.srt"
            write_srt(cues, path)
            self.assertEqual(read_subtitles(path), cues)

    def test_reader_accepts_webvtt_header_and_strips_tags(self):
        content = """WEBVTT

00:00:01.000 --> 00:00:02.500 align:start
<c.color>hello</c>
"""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.vtt"
            path.write_text(content, encoding="utf-8")
            self.assertEqual(read_subtitles(path), [Cue(1.0, 2.5, "hello")])

    def test_translation_payload_uses_local_ids(self):
        payload = json.loads(translation_payload([Cue(0, 1, "a"), Cue(1, 2, "b")]))
        self.assertEqual(payload, [{"id": 0, "text": "a"}, {"id": 1, "text": "b"}])

    def test_apply_translations_rejects_missing_ids(self):
        cues = [Cue(0, 1, "a"), Cue(1, 2, "b")]
        with self.assertRaisesRegex(ValueError, "ids"):
            apply_translations(cues, [{"id": 0, "text": "甲"}])

    def test_translation_validation_reports_id_differences(self):
        cues = [Cue(0, 1, "a"), Cue(1, 2, "b")]
        with self.assertRaisesRegex(
            ValueError,
            r"missing=\[1\], unexpected=\[2\], duplicates=\[0\]",
        ):
            apply_translations(
                cues,
                [
                    {"id": 0, "text": "甲"},
                    {"id": 0, "text": "重复"},
                    {"id": 2, "text": "越界"},
                ],
            )

    def test_cleans_non_speech_markers_and_drops_empty_cues(self):
        cues = [
            Cue(0, 1, "[音楽]"),
            Cue(1, 2, "好き[歌声]なのよ [拍手]"),
            Cue(2, 3, "[鼻息] ごめん"),
            Cue(3, 4, "[Aメロ] は残す"),
        ]
        self.assertEqual(
            clean_non_speech_markers(cues),
            [
                Cue(1, 2, "好きなのよ"),
                Cue(2, 3, "ごめん"),
                Cue(3, 4, "[Aメロ] は残す"),
            ],
        )

    def test_merges_fragments_until_sentence_punctuation(self):
        cues = [
            Cue(0, 1, "The model was trained on"),
            Cue(1.1, 2, "more than a million samples."),
            Cue(2.1, 3, "It generalizes well."),
        ]
        result = merge_semantic_cues(cues, SegmentationConfig())
        self.assertEqual(
            result,
            [
                Cue(0, 2, "The model was trained on more than a million samples."),
                Cue(2.1, 3, "It generalizes well."),
            ],
        )

    def test_does_not_merge_across_long_pause(self):
        cues = [Cue(0, 1, "an unfinished"), Cue(2, 3, "thought")]
        config = SegmentationConfig(max_gap_seconds=0.5)
        self.assertEqual(merge_semantic_cues(cues, config), cues)

    def test_hard_duration_limit_prevents_oversized_cue(self):
        cues = [Cue(0, 4, "one fragment"), Cue(4.1, 8, "another fragment")]
        config = SegmentationConfig(max_duration_seconds=5)
        self.assertEqual(merge_semantic_cues(cues, config), cues)

    def test_joins_cjk_without_inserting_a_space(self):
        cues = [Cue(0, 1, "这是一个，"), Cue(1, 2, "完整句子。")]
        self.assertEqual(
            merge_semantic_cues(cues, SegmentationConfig()),
            [Cue(0, 2, "这是一个，完整句子。")],
        )


if __name__ == "__main__":
    unittest.main()
