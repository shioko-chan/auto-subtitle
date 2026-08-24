from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from subtitle_pipeline.asr_correction import (
    ASREntity,
    _pending_batches,
    correct_asr_windows,
    entities_from_context,
)


class ASRCorrectionTests(unittest.TestCase):
    def test_batches_respect_window_and_character_limits(self) -> None:
        pending = [
            (index, {}, text, [])
            for index, text in enumerate(("a" * 7, "b" * 7, "c" * 3))
        ]
        batches = _pending_batches(pending, maximum_windows=2, maximum_chars=10)
        self.assertEqual(
            [[item[0] for item in batch] for batch in batches], [[0], [1, 2]]
        )

    def test_entities_from_context_ignores_translation_terms(self) -> None:
        entities = entities_from_context(
            {
                "terms": {"誤変換": "translation only"},
                "asr_entities": [
                    {
                        "surface": "夢限大みゅーたいぷ",
                        "reading": "むげんだいみゅーたいぷ",
                        "aliases": ["無限大ミュータイプ"],
                    }
                ],
            }
        )
        self.assertEqual(entities[0].surface, "夢限大みゅーたいぷ")
        self.assertEqual(entities[0].aliases, ("無限大ミュータイプ",))

    def test_exact_alias_is_normalized_before_llm(self) -> None:
        requests: list[dict[str, object]] = []

        def request(body: dict[str, object]) -> dict[str, object]:
            requests.append(body)
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "windows": [
                                        {
                                            "window_id": 0,
                                            "corrected_text": "夢限大みゅーたいぷボーカル仲町あられです",
                                        }
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = correct_asr_windows(
                [
                    {
                        "window_id": 4,
                        "text": "無限大ミュータイプボーカル仲間ちゃあられです",
                        "language": "Japanese",
                    }
                ],
                entities=[
                    ASREntity(
                        "夢限大みゅーたいぷ",
                        "むげんだいみゅーたいぷ",
                        ("無限大ミュータイプ",),
                    ),
                    ASREntity("仲町あられ", "なかまちあられ"),
                ],
                request=request,
                model="test-model",
                cache_path=root / "cache.json",
                audit_path=root / "audit.jsonl",
            )
        self.assertEqual(
            result[0]["text"],
            "夢限大みゅーたいぷボーカル仲町あられです",
        )
        prompt = requests[0]["messages"][1]["content"]
        self.assertIn(
            "<0 candidates=夢限大みゅーたいぷ｜仲町あられ>",
            prompt,
        )
        self.assertIn("<仲町あられ>", prompt)

    def test_invalid_llm_response_fails_open_to_rule_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = correct_asr_windows(
                [{"text": "無限大ミュータイプ", "language": "Japanese"}],
                entities=[
                    ASREntity(
                        "夢限大みゅーたいぷ",
                        "むげんだいみゅーたいぷ",
                        ("無限大ミュータイプ",),
                    )
                ],
                request=lambda _: {"choices": []},
                model="test-model",
                cache_path=root / "cache.json",
                audit_path=root / "audit.jsonl",
            )
            audit_lines = (
                (root / "audit.jsonl").read_text(encoding="utf-8").splitlines()
            )
            audit = json.loads(audit_lines[-1])
        self.assertEqual(result[0]["text"], "夢限大みゅーたいぷ")
        self.assertEqual(result[0]["correction_method"], "rule_fallback")
        self.assertEqual(audit["error"], "request_or_validation_failed")


if __name__ == "__main__":
    unittest.main()
