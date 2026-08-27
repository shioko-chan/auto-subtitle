from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from subtitle_pipeline.chat_context import (
    CurrentVideoChatIndex,
    YouTubeChatMessage,
)


class CurrentVideoChatIndexTests(unittest.TestCase):
    def test_includes_all_useful_time_local_messages_and_audits_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            audit = Path(temporary) / "audit.jsonl"
            index = CurrentVideoChatIndex(
                [
                    YouTubeChatMessage(95, "a", "アクスタの話？"),
                    YouTubeChatMessage(101, "b", "アクスタは罠"),
                    YouTubeChatMessage(104, "c", "アクスタ高い"),
                    YouTubeChatMessage(300, "d", "別の話題"),
                ],
                audit_path=audit,
            )

            evidence = index.evidence(
                100,
                105,
                "アクスタフィギュアが安く感じる",
                stage="translation",
                target_id=7,
            )

            event = json.loads(audit.read_text(encoding="utf-8"))

            self.assertIn("[chat] アクスタの話？", evidence)
            self.assertIn("[chat] アクスタは罠", evidence)
            self.assertIn("[chat] アクスタ高い", evidence)
            self.assertNotIn("別の話題", evidence)
            self.assertEqual(event["stage"], "translation")
            self.assertEqual(event["target_id"], 7)
            self.assertEqual(event["candidate_count"], 3)
            self.assertEqual(event["selected_count"], 3)

    def test_query_does_not_filter_time_local_messages(self) -> None:
        index = CurrentVideoChatIndex(
            [YouTubeChatMessage(10, "a", "今日の晩ご飯は何？")]
        )

        evidence = index.evidence(
            10,
            12,
            "夢限大みゅーたいぷです",
            stage="asr_correction",
            target_id=0,
        )

        self.assertEqual(evidence, "[chat] 今日の晩ご飯は何？")

    def test_message_ids_remove_download_duplicates_but_preserve_viewer_repeats(self):
        index = CurrentVideoChatIndex(
            [
                YouTubeChatMessage(10, "a", "アクスタ", message_id="same"),
                YouTubeChatMessage(10, "a", "アクスタ", message_id="same"),
                YouTubeChatMessage(11, "b", "アクスタ", message_id="other"),
                YouTubeChatMessage(12, "c", "アクスタ", message_id="third"),
            ]
        )

        evidence = index.evidence(
            10,
            13,
            "アクスタについて",
            stage="translation",
            target_id="window",
        )

        self.assertIn("[chat ×3] アクスタ", evidence)
        self.assertNotIn("×4", evidence)

    def test_repeated_text_is_only_grouped_within_ten_seconds(self) -> None:
        index = CurrentVideoChatIndex(
            [
                YouTubeChatMessage(10, "a", "等身大フィギュア", message_id="1"),
                YouTubeChatMessage(14, "b", "等身大フィギュア", message_id="2"),
                YouTubeChatMessage(30, "c", "等身大フィギュア", message_id="3"),
            ]
        )

        evidence = index.evidence(
            10,
            30,
            "透芯材フィギュア",
            stage="asr_correction",
            target_id=4,
        )

        self.assertEqual(
            evidence,
            "[chat ×2] 等身大フィギュア\n[chat] 等身大フィギュア",
        )

    def test_filters_reaction_only_spam_and_omits_timestamps(self) -> None:
        index = CurrentVideoChatIndex(
            [
                YouTubeChatMessage(10, "a", "ｗｗｗｗ"),
                YouTubeChatMessage(11, "b", "👏👏👏"),
                YouTubeChatMessage(12, "c", "TY!"),
            ]
        )

        evidence = index.evidence(
            10,
            12,
            "ありがとう",
            stage="asr_correction",
            target_id=0,
        )

        self.assertEqual(evidence, "[chat] TY!")
        self.assertNotIn("s chat", evidence)


if __name__ == "__main__":
    unittest.main()
