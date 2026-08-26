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
    def test_retrieves_repeated_time_local_terms_and_audits_selection(self) -> None:
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

        self.assertIn("アクスタ (3 viewers)", evidence)
        self.assertNotIn("別の話題", evidence)
        self.assertEqual(event["stage"], "translation")
        self.assertEqual(event["target_id"], 7)

    def test_irrelevant_single_messages_are_not_injected(self) -> None:
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

        self.assertEqual(evidence, "")

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

        self.assertIn("×3 viewers", evidence)
        self.assertNotIn("×4 viewers", evidence)


if __name__ == "__main__":
    unittest.main()
