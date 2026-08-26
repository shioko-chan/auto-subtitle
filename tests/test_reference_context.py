from __future__ import annotations

import unittest

from subtitle_pipeline.reference_context import compact_reference_context


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


if __name__ == "__main__":
    unittest.main()
