from __future__ import annotations

import unittest

from subtitle_pipeline.reference_context import (
    compact_reference_context,
    compact_translation_reference_context,
)


class ReferenceContextTests(unittest.TestCase):
    def test_window_reference_omits_full_youtube_description(self) -> None:
        compact = compact_reference_context(
            {
                "video": {
                    "title": "直播标题",
                    "description": "很长的 YouTube 视频简介",
                    "channel": "频道",
                },
                "terms": {"原词": "译词"},
            }
        )

        self.assertEqual(compact["video"], {"title": "直播标题", "channel": "频道"})
        self.assertEqual(compact["terms"], {"原词": "译词"})

    def test_translation_reference_keeps_only_batch_relevant_entries(self) -> None:
        compact = compact_translation_reference_context(
            {
                "video": {
                    "title": "直播标题",
                    "description": "不应注入",
                    "tags": ["不应注入"],
                    "channel": "频道",
                },
                "franchises": [{"name": "企划", "background": "背景"}],
                "characters": [
                    {
                        "id": "speaker_a",
                        "source_name": "人物甲",
                        "canonical": "人物甲译名",
                    },
                    {
                        "id": "speaker_b",
                        "source_name": "人物乙",
                        "canonical": "人物乙译名",
                    },
                ],
                "terms": {"アクスタ": "亚克力立牌", "無関係": "无关"},
                "asr_entities": [{"surface": "不应注入"}],
                "fan_knowledge": [{"body": "不应重复注入"}],
            },
            evidence_text="アクスタの話",
            speakers={"speaker_a"},
        )

        self.assertEqual(
            compact["video"], {"title": "直播标题", "channel": "频道"}
        )
        self.assertEqual(
            [value["id"] for value in compact["characters"]], ["speaker_a"]
        )
        self.assertEqual(compact["terms"], {"アクスタ": "亚克力立牌"})
        self.assertNotIn("asr_entities", compact)
        self.assertNotIn("fan_knowledge", compact)


if __name__ == "__main__":
    unittest.main()
