import unittest
from unittest.mock import patch

from subtitle_pipeline.config import SongIdentificationConfig
from subtitle_pipeline.song_identification import (
    _run_song_agent,
    aggregate_ocr_observations,
    apply_lyric_corrections,
    group_singing_episodes,
)
from subtitle_pipeline.subtitles import Cue


class SongIdentificationTests(unittest.TestCase):
    def test_groups_singing_phrases_without_absorbing_speech(self):
        cues = [
            Cue(0, 2, "intro", "a", "speech"),
            Cue(10, 14, "line one", "a", "singing"),
            Cue(15, 19, "line two", "a", "singing"),
            Cue(20, 25, "talk", "a", "speech"),
            Cue(45, 50, "next song", "a", "singing"),
        ]

        episodes = group_singing_episodes(cues, 20)

        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0].cue_ids, (1, 2))
        self.assertEqual(episodes[1].cue_ids, (4,))

    def test_ocr_candidate_requires_distinct_persistent_frames(self):
        observations = [
            ("おじゃま虫 / DECO*27", 0.9, 0, 10.0),
            ("おじゃま虫/DECO*27", 0.8, 1, 11.0),
            ("chat message", 0.99, 2, 12.0),
        ]

        candidates = aggregate_ocr_observations(observations, 2)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].frames, 2)
        self.assertIn("おじゃま虫", candidates[0].text)

    def test_episode_includes_cues_between_singing_anchors(self):
        cues = [
            Cue(0, 2, "song start", "a", "singing"),
            Cue(2, 8, "misclassified lyric", "a", "speech"),
            Cue(8, 10, "song end", "a", "singing"),
        ]

        episodes = group_singing_episodes(cues, 10)

        self.assertEqual(episodes[0].cue_ids, (0, 1, 2))

    def test_applies_only_verified_ordered_lyric_alignment(self):
        cues = [
            Cue(0, 1, "talk", "a", "speech"),
            Cue(1, 4, "wrong one", "a", "singing"),
            Cue(4, 7, "wrong two", "a", "singing"),
            Cue(7, 8, "talk", "a", "speech"),
        ]
        reports = [
            {
                "confidence": "medium",
                "episode": {"cue_ids": [1, 2]},
                "alignments": [
                    {
                        "asr_cue_ids": [1, 2],
                        "match": "lyrics",
                        "corrected_text": "corrected lyric",
                    }
                ],
            }
        ]

        corrected = apply_lyric_corrections(cues, reports)

        self.assertEqual(len(corrected), 3)
        self.assertEqual(corrected[1], Cue(1, 7, "corrected lyric", "a", "singing"))

    def test_does_not_apply_low_confidence_or_out_of_episode_alignment(self):
        cues = [Cue(0, 2, "raw", "a", "singing")]
        reports = [
            {
                "confidence": "low",
                "episode": {"cue_ids": [0]},
                "alignments": [
                    {"asr_cue_ids": [0], "match": "lyrics", "corrected_text": "bad"}
                ],
            },
            {
                "confidence": "high",
                "episode": {"cue_ids": []},
                "alignments": [
                    {"asr_cue_ids": [0], "match": "lyrics", "corrected_text": "bad"}
                ],
            },
        ]

        self.assertEqual(apply_lyric_corrections(cues, reports), cues)

    def test_tool_budget_forces_final_answer_without_more_tools(self):
        responses = [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call-{index}",
                                    "type": "function",
                                    "function": {
                                        "name": "search_web",
                                        "arguments": '{"query":"song"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
            for index in range(1)
        ]
        responses.append(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                '{"song":null,"artist":null,"confidence":"low",'
                                '"evidence":[],"sources":[],"alignments":[]}'
                            ),
                        }
                    }
                ]
            }
        )

        with patch(
            "subtitle_pipeline.song_identification._WebTools.execute",
            return_value=("call", "[]"),
        ):
            report = _run_song_agent(
                {},
                SongIdentificationConfig(max_tool_calls=1),
                lambda _: responses.pop(0),
            )

        self.assertIsNone(report["song"])
        self.assertEqual(report["confidence"], "low")
        self.assertEqual(responses, [])

    def test_unfetched_source_cannot_authorize_lyric_correction(self):
        from subtitle_pipeline.song_identification import _finalize_report

        report = _finalize_report(
            {
                "song": "Example",
                "artist": "Artist",
                "confidence": "high",
                "sources": ["https://example.com/lyrics"],
                "alignments": [
                    {
                        "asr_cue_ids": [0],
                        "lyric_line_ids": [1],
                        "match": "lyrics",
                        "corrected_text": "line",
                    }
                ],
            },
            set(),
        )

        self.assertEqual(report["sources"], [])
        self.assertEqual(report["alignments"][0]["match"], "uncertain")


if __name__ == "__main__":
    unittest.main()
