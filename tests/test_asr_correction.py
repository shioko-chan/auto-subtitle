from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from subtitle_pipeline.asr_correction import (
    ASREntity,
    _correction_windows,
    correct_asr_windows,
    entities_from_context,
)
from subtitle_pipeline.fan_knowledge import KnowledgeHit, KnowledgeScore


class ASRCorrectionTests(unittest.TestCase):
    def test_all_retrieval_precedes_llm_and_survives_interruption(self):
        events = []
        records = [{"text": value, "language": "Japanese"} for value in ["あ", "い"]]

        def retrieve(items):
            events.append(("retrieve", [text for _, text in items]))
            return [[] for _ in items]

        def interrupted(body):
            events.append(("llm", None))
            raise KeyboardInterrupt()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kwargs = dict(records=records, entities=[], model="test", window_chars=1,
                          cache_path=root / "cache.sqlite3", audit_path=root / "audit.jsonl")
            with self.assertRaises(KeyboardInterrupt):
                correct_asr_windows(**kwargs, request=interrupted, retrieve_knowledge=retrieve)
            self.assertEqual(events, [("retrieve", ["あ", "い"]), ("llm", None)])
            responses = iter(["あ", "い"])

            def request(body):
                return {"choices": [{"message": {"content": json.dumps({"segments": [
                    {"segment_id": 0, "corrected_text": next(responses)}
                ]})}}]}

            result = correct_asr_windows(
                **kwargs, request=request,
                retrieve_knowledge=lambda _: self.fail("cached requests must not retrieve again"),
            )
            self.assertEqual([record["text"] for record in result], ["あ", "い"])

    def test_prompt_requests_active_phonetic_and_lexical_correction(self) -> None:
        prompts: list[str] = []

        def request(body: dict[str, object]) -> dict[str, object]:
            prompts.append(body["messages"][0]["content"])
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"segments":[{"segment_id":0,"corrected_text":"等身大フィギュア"}]}'
                        }
                    }
                ]
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            correct_asr_windows(
                [{"text": "透芯材フィギュア", "language": "Japanese"}],
                entities=[],
                request=request,
                model="test",
                cache_path=root / "cache.json",
                audit_path=root / "audit.jsonl",
            )

        prompt = " ".join(prompts[0].split())
        self.assertIn("phonetic near miss", prompt)
        self.assertIn("does not require an entity candidate", prompt)
        self.assertIn("ordinary collocation", prompt)
        self.assertIn("Do not limit corrections to character-level edits", prompt)
        self.assertIn("The correct wording may use completely different kanji", prompt)
        self.assertIn("Consider nearby pronunciations", prompt)
        self.assertIn("Faithfulness means faithfulness to the likely spoken audio", prompt)
        self.assertIn("not to the literal ASR characters", prompt)

    def test_correction_windows_use_only_the_character_limit(self) -> None:
        pending = [
            (index, {}, text, [])
            for index, text in enumerate(("a" * 7, "b" * 7, "c" * 3))
        ]
        batches = _correction_windows(pending, maximum_chars=10)
        self.assertEqual(
            [[item[0] for item in batch] for batch in batches], [[0], [1, 2]]
        )

        short = [(index, {}, "x", []) for index in range(9)]
        self.assertEqual(len(_correction_windows(short, maximum_chars=10)), 1)

    def test_long_segment_retrieves_knowledge_by_local_fragment(self) -> None:
        queries: list[str] = []

        def request(_body: dict[str, object]) -> dict[str, object]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "segments": [
                                        {"segment_id": 0, "corrected_text": "修正"}
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
                [{"text": "あ" * 250, "language": "Japanese"}],
                entities=[],
                request=request,
                model="test-model",
                cache_path=root / "cache.json",
                audit_path=root / "audit.jsonl",
                retrieve_knowledge=lambda items: [queries.append(text) or [] for _, text in items],
            )

        self.assertEqual([len(value) for value in queries], [120, 120, 10])

    def test_request_uses_stage_output_limit(self) -> None:
        bodies: list[dict[str, object]] = []

        def request(body: dict[str, object]) -> dict[str, object]:
            bodies.append(body)
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"segments":[{"segment_id":0,"corrected_text":"修正"}]}'
                        }
                    }
                ]
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            correct_asr_windows(
                [{"text": "原文", "language": "Japanese"}],
                entities=[],
                request=request,
                model="test-model",
                cache_path=root / "cache.json",
                audit_path=root / "audit.jsonl",
                max_tokens=1234,
            )

        self.assertEqual(bodies[0]["max_tokens"], 1234)

    def test_fenced_json_response_is_accepted(self) -> None:
        response_content = (
            "```json\n"
            '{"segments":[{"segment_id":0,'
            '"corrected_text":"等身大フィギュア"}]}\n'
            "```"
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = correct_asr_windows(
                [{"text": "透芯材フィギュア", "language": "Japanese"}],
                entities=[],
                request=lambda _body: {
                    "choices": [{"message": {"content": response_content}}]
                },
                model="test-model",
                cache_path=root / "cache.json",
                audit_path=root / "audit.jsonl",
            )
            audit = json.loads(
                (root / "audit.jsonl").read_text(encoding="utf-8").splitlines()[-1]
            )

        self.assertEqual(result[0]["text"], "等身大フィギュア")
        self.assertEqual(result[0]["correction_method"], "llm")
        self.assertIsNone(audit["error"])

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
                                    "segments": [
                                        {
                                            "segment_id": 0,
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
            '<SEGMENT id="0" candidates="夢限大みゅーたいぷ｜仲町あられ">',
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
                                    "segments": [
                                        {
                                            "segment_id": 0,
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
                retrieve_knowledge=lambda items: [[hit] for _ in items],
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
                                    "segments": [
                                        {"segment_id": 0, "corrected_text": "一"},
                                        {"segment_id": 1, "corrected_text": "二"},
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
            '<SEGMENT id="0" candidates="(none)">\n'
            "CURRENT_VIDEO_CHAT:\n第一窗口聊天\nLOCAL_RETRIEVAL:\n"
            '<RETRIEVAL_FRAGMENT id="0">\nTEXT:\n一\n'
            "FAN_KNOWLEDGE:\n(none)\n</RETRIEVAL_FRAGMENT>\n"
            "ASR_TEXT:\n一\n</SEGMENT>",
            prompt,
        )
        self.assertIn(
            '<SEGMENT id="1" candidates="(none)">\n'
            "CURRENT_VIDEO_CHAT:\n第二窗口聊天\nLOCAL_RETRIEVAL:\n"
            '<RETRIEVAL_FRAGMENT id="0">\nTEXT:\n二\n'
            "FAN_KNOWLEDGE:\n(none)\n</RETRIEVAL_FRAGMENT>\n"
            "ASR_TEXT:\n二\n</SEGMENT>",
            prompt,
        )
        first_end = prompt.index("</SEGMENT>")
        second_start = prompt.index('<SEGMENT id="1"')
        self.assertLess(first_end, second_start)

    def test_asr_cache_reuses_original_result_after_knowledge_changes(self) -> None:
        calls = 0
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
            nonlocal calls
            calls += 1
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "segments": [
                                        {"segment_id": 0, "corrected_text": "修正"}
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
                retrieve_knowledge=lambda items: [[hit] for _ in items],
            )
            second = correct_asr_windows(
                **arguments,
                request=request,
                retrieve_knowledge=lambda items: [[] for _ in items],
            )
            third = correct_asr_windows(
                **arguments,
                request=lambda _body: self.fail("unchanged evidence should use cache"),
                retrieve_knowledge=lambda items: [[] for _ in items],
            )

        self.assertEqual(first[0]["text"], "修正")
        self.assertEqual(second[0]["text"], "修正")
        self.assertEqual(third[0]["correction_method"], "llm")
        self.assertEqual(calls, 1)

    def test_asr_cache_keeps_stage_configuration(self) -> None:
        calls = 0

        def request(_body: dict[str, object]) -> dict[str, object]:
            nonlocal calls
            calls += 1
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"segments":[{"segment_id":0,"corrected_text":"修正"}]}'
                        }
                    }
                ]
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = {
                "records": [{"text": "原文", "language": "Japanese"}],
                "entities": [],
                "request": request,
                "model": "test-model",
                "cache_path": root / "cache.json",
                "audit_path": root / "audit.jsonl",
            }
            correct_asr_windows(**arguments, max_tokens=1024)
            correct_asr_windows(**arguments, max_tokens=2048)

        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
