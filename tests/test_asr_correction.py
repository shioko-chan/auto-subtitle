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
from subtitle_pipeline.fan_knowledge import KnowledgeHit, KnowledgeScore


class ASRCorrectionTests(unittest.TestCase):
    def test_neighboring_asr_windows_are_read_only_context(self) -> None:
        prompts: list[str] = []

        def request(body):
            prompts.append(body["messages"][1]["content"])
            target = body["messages"][1]["content"].split("TARGET:\n", 1)[1]
            text = target.split(">", 1)[1]
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"windows": [{"window_id": 0, "corrected_text": text}]},
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        records = [
            {
                "window_id": index,
                "text": text,
                "language": "Japanese",
                "core_start": index * 10.0,
                "core_end": index * 10.0 + 5.0,
            }
            for index, text in enumerate(("前の発言", "対象の発言", "後の発言"))
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            correct_asr_windows(
                records,
                entities=[],
                request=request,
                model="test",
                cache_path=root / "cache.json",
                audit_path=root / "audit.jsonl",
                batch_windows=1,
                context_before_seconds=20,
                context_after_seconds=10,
            )

        self.assertIn("READ_ONLY_CONTEXT:\n", prompts[1])
        self.assertIn("前の発言", prompts[1])
        self.assertIn("後の発言", prompts[1])
        target = prompts[1].split("TARGET:\n", 1)[1]
        self.assertNotIn("前の発言", target)
        self.assertNotIn("後の発言", target)

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

    def test_retrieved_fan_knowledge_is_injected_and_audited(self) -> None:
        requests: list[dict[str, object]] = []
        hit = KnowledgeHit(
            "knowledge:ty",
            "catchphrase",
            "TY",
            "TY表示Thank You，不是人物姓名。",
            None,
            KnowledgeScore(1, 0, 0, 0, 0, 0, 1, 3.75),
            ("TY",),
        )

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
                                            "corrected_text": "TYありがとう",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            correct_asr_windows(
                [{"window_id": 1, "text": "TYありがとう", "language": "Japanese"}],
                entities=[],
                request=request,
                model="test-model",
                cache_path=root / "cache.json",
                audit_path=root / "audit.jsonl",
                retrieve_knowledge=lambda _record, _text: [hit],
            )
            audit = json.loads(
                (root / "audit.jsonl").read_text(encoding="utf-8").splitlines()[-1]
            )

        prompt = requests[0]["messages"][1]["content"]
        self.assertIn("TY表示Thank You", prompt)
        self.assertEqual(audit["knowledge_ids"], ["knowledge:ty"])

    def test_chat_evidence_is_labeled_per_asr_window(self) -> None:
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
                                        {"window_id": 0, "corrected_text": "一"},
                                        {"window_id": 1, "corrected_text": "二"},
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
            correct_asr_windows(
                [
                    {"text": "一", "chat_text": "第一窗口聊天"},
                    {"text": "二", "chat_text": "第二窗口聊天"},
                ],
                entities=[],
                request=request,
                model="test-model",
                cache_path=root / "cache.json",
                audit_path=root / "audit.jsonl",
            )

        prompt = requests[0]["messages"][1]["content"]
        self.assertIn(
            "<0>\nFAN_KNOWLEDGE:\n(none)\nCURRENT_VIDEO_CHAT:\n第一窗口聊天", prompt
        )
        self.assertIn(
            "<1>\nFAN_KNOWLEDGE:\n(none)\nCURRENT_VIDEO_CHAT:\n第二窗口聊天", prompt
        )

    def test_asr_cache_does_not_depend_on_retrieved_knowledge(self) -> None:
        hit = KnowledgeHit(
            "knowledge:first",
            "note",
            "first",
            "first context",
            None,
            KnowledgeScore(1, 0, 0, 0, 0, 0, 1, 1),
            (),
        )

        def request(_body: dict[str, object]) -> dict[str, object]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "windows": [
                                        {"window_id": 0, "corrected_text": "修正"}
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
            arguments = {
                "records": [{"text": "原文", "language": "Japanese"}],
                "entities": [],
                "model": "test-model",
                "cache_path": root / "cache.json",
                "audit_path": root / "audit.jsonl",
            }
            first = correct_asr_windows(
                **arguments,
                request=request,
                retrieve_knowledge=lambda _record, _text: [hit],
            )
            second = correct_asr_windows(
                **arguments,
                request=lambda _body: self.fail("LLM should not run on cache hit"),
                retrieve_knowledge=lambda _record, _text: [],
            )

        self.assertEqual(first[0]["text"], "修正")
        self.assertEqual(second[0]["text"], "修正")


if __name__ == "__main__":
    unittest.main()
