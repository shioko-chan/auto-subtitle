import unittest

from subtitle_pipeline.config import SegmentationConfig
from subtitle_pipeline.local_segmentation import LocalUnit
from subtitle_pipeline.translation_support import (
    dialogue_context,
    window_ranges,
)


def _unit(
    local_id: int,
    start: float,
    end: float,
    text: str,
    *,
    score: int | None = None,
) -> LocalUnit:
    return LocalUnit(
        "speaker",
        local_id,
        (local_id,),
        start,
        end,
        text,
        "speaker",
        "speech",
        boundary_score_after=score,
    )


class TranslationSupportTests(unittest.TestCase):
    def test_windows_never_cross_speaker_episode_gap(self) -> None:
        units = (
            _unit(0, 0, 1, "前半"),
            _unit(1, 4, 5, "後半"),
        )

        self.assertEqual(window_ranges(units, SegmentationConfig()), [(0, 1), (1, 2)])

    def test_window_limit_prefers_a_strong_recent_boundary(self) -> None:
        units = tuple(
            _unit(index, index, index + 1, str(index), score=10 if index == 2 else 0)
            for index in range(5)
        )
        config = SegmentationConfig(model_window_units=4)

        self.assertEqual(window_ranges(units, config)[0], (0, 3))

    def test_dialogue_context_excludes_target_and_remains_chronological(self) -> None:
        before = _unit(0, 0, 1, "前")
        target = _unit(1, 1, 2, "対象")
        after = _unit(2, 2, 3, "後")

        context = dialogue_context(
            [after, target, before], (target,), SegmentationConfig()
        )

        self.assertLess(context.index("前"), context.index("後"))
        self.assertNotIn("対象", context)

if __name__ == "__main__":
    unittest.main()
