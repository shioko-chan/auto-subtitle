from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from subtitle_pipeline.fan_knowledge import (
    FanKnowledgeRetriever,
    KnowledgeChunk,
    KnowledgeDocument,
    PendingTermDocument,
    TermCandidateOccurrence,
    TermExtractionChunk,
)
from subtitle_pipeline.term_extraction import (
    LocalTermCandidate,
    LocalTermOccurrence,
    _background_zipf_frequency,
    _candidate_batches,
    _candidate_has_valid_structure,
    _candidate_is_worth_screening,
    _candidate_prompt_value,
    _candidate_source_text,
    _collect_local_term_candidates,
    _local_term_surfaces,
    _strip_candidate_source_artifacts,
    extract_pending_terms,
    prepare_pending_terms,
    validate_pending_terms,
)


def _response(value: dict[str, object], *, finish_reason: str = "stop") -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {"content": json.dumps(value, ensure_ascii=False)},
                "finish_reason": finish_reason,
            }
        ]
    }


def _document(
    document_id: str,
    source_type: str,
    text: str,
    *,
    title: str = "配信",
    author: str | None = None,
) -> PendingTermDocument:
    return PendingTermDocument(
        document_id,
        source_type,
        title,
        None,
        0.8,
        f"hash-{document_id}",
        (TermExtractionChunk(f"chunk-{document_id}", text),),
        author=author,
    )


class TermExtractionTests(unittest.TestCase):
    def test_structural_filter_rejects_non_term_artifacts_before_idf(self) -> None:
        for value in (
            "2026年8月31日",
            "12345",
            "0.5周年",
            "1000円",
            "001東京都中野区",
            "!!️shorts投稿!!️",
            "shorts\n投稿",
            "イベント……",
            "<div>イベント</div>",
        ):
            with self.subTest(value=value):
                self.assertFalse(_candidate_has_valid_structure(value))

        for value in (
            "すやバラ",
            "#AveMujica",
            "BanG Dream",
            "第3回ガルパ杯",
            "22/7",
            "7ORDER",
            "3Dライブ",
        ):
            with self.subTest(value=value):
                self.assertTrue(_candidate_has_valid_structure(value))

    def test_source_artifacts_are_removed_before_candidate_extraction(self) -> None:
        cleaned = _strip_candidate_source_artifacts(
            "配信はこちら https://www.youtube.com/live/Ax9lrff2kYc?si=token "
            "公式 example.com/news/123 連絡 staff@example.com ?si=token&v=123 "
            "#すやバラ"
        )

        self.assertEqual(cleaned, "配信はこちら 公式 連絡 #すやバラ")
        surfaces = _local_term_surfaces(cleaned)
        self.assertNotIn("Ax9lrff2kYc", surfaces)
        self.assertNotIn("example.com", surfaces)
        self.assertIn("#すやバラ", surfaces)

    def test_source_artifacts_do_not_create_candidate_occurrences(self) -> None:
        candidates = _collect_local_term_candidates(
            [
                _document(
                    "url-source",
                    "x_post",
                    "配信はこちら https://www.youtube.com/live/Ax9lrff2kYc?si=token "
                    "#すやバラ",
                    title="公式 example.com/news/123",
                    author="メンバー",
                )
            ],
            screenable_only=False,
        )
        surfaces = {candidate.surface for candidate in candidates}

        self.assertNotIn("Ax9lrff2kYc", surfaces)
        self.assertNotIn("example.com", surfaces)
        self.assertIn("#すやバラ", surfaces)

    def test_background_frequency_rejects_common_terms_but_keeps_domain_terms(
        self,
    ) -> None:
        occurrence = LocalTermOccurrence(
            "document:one",
            "document:one",
            "chunk:one",
            "context",
            "official_news",
            1.0,
            strong_name_evidence=True,
        )

        self.assertGreater(_background_zipf_frequency("ライブ"), 4.8)
        self.assertLess(_background_zipf_frequency("すやバラ"), 4.8)
        self.assertFalse(
            _candidate_is_worth_screening("ライブ", [occurrence])
        )
        self.assertFalse(_candidate_is_worth_screening("YouTube", [occurrence]))
        self.assertFalse(_candidate_is_worth_screening("shorts", [occurrence]))
        self.assertTrue(
            _candidate_is_worth_screening("すやバラ", [occurrence])
        )
        self.assertTrue(_candidate_is_worth_screening("AveMujica", [occurrence]))

    def test_prepare_pending_terms_consumes_document_queue_and_queues_reviews(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            retriever = FanKnowledgeRetriever(Path(temporary) / "knowledge.sqlite3")
            retriever.upsert_document(
                KnowledgeDocument(
                    "document:local-only",
                    "official_news",
                    "post:local-only",
                    "告知",
                    "今夜も「すやバラ」です。",
                ),
                [KnowledgeChunk(0, "今夜も「すやバラ」です。")],
            )

            summary = prepare_pending_terms(retriever)
            pending = retriever.pending_term_documents()
            occurrence_count = retriever._database.execute(
                "SELECT COUNT(*) AS value "
                "FROM knowledge_term_candidate_occurrences"
            ).fetchone()["value"]
            extracted_count = retriever._database.execute(
                "SELECT COUNT(*) AS value FROM knowledge_extracted_terms"
            ).fetchone()["value"]
            review_count = retriever._database.execute(
                "SELECT COUNT(*) AS value FROM knowledge_term_review_queue"
            ).fetchone()["value"]
            retriever.close()

        self.assertEqual(summary.documents, 1)
        self.assertGreater(summary.local_candidates, 0)
        self.assertEqual(pending, [])
        self.assertGreater(occurrence_count, 0)
        self.assertEqual(extracted_count, 0)
        self.assertGreater(review_count, 0)

    def test_validation_prunes_stale_candidates_before_calling_llm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            retriever = FanKnowledgeRetriever(Path(temporary) / "knowledge.sqlite3")
            retriever.upsert_document(
                KnowledgeDocument(
                    "document:stale-term",
                    "youtube_metadata",
                    "stale-term",
                    "YouTube",
                    "YouTube",
                ),
                [KnowledgeChunk(0, "YouTube")],
            )
            document = retriever.pending_term_documents()[0]
            retriever.replace_term_candidate_occurrences(
                [document.document_id],
                [
                    TermCandidateOccurrence(
                        "youtube",
                        "YouTube",
                        document.document_id,
                        document.document_id,
                        document.chunks[0].chunk_id,
                        "[VIDEO]YouTube",
                        "youtube_metadata",
                        1.0,
                        True,
                        1,
                    )
                ],
            )
            retriever.finish_term_preparation(
                [document],
                touched_forms=["youtube"],
                pending_review_forms=["youtube"],
                backfill=False,
            )

            summary = validate_pending_terms(
                retriever,
                request=lambda _body: self.fail("LLM must not be called"),
                model="test-model",
                max_tokens=1024,
                thinking=None,
                max_retries=1,
            )
            pending = retriever.pending_term_review_forms()
            retriever.close()

        self.assertEqual(summary.candidates, 0)
        self.assertEqual(pending, [])

    def test_backfill_reviews_are_hidden_from_incremental_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            retriever = FanKnowledgeRetriever(Path(temporary) / "knowledge.sqlite3")
            retriever.upsert_document(
                KnowledgeDocument(
                    "document:backfill-only",
                    "official_news",
                    "backfill-only",
                    "告知",
                    "今夜も「すやバラ」です。",
                ),
                [KnowledgeChunk(0, "今夜も「すやバラ」です。")],
            )
            retriever.queue_all_documents_for_term_extraction()

            prepare_pending_terms(retriever, include_backfill=True)
            incremental = retriever.pending_term_review_forms()
            backfill = retriever.pending_term_review_forms(include_backfill=True)
            retriever.close()

        self.assertEqual(incremental, [])
        self.assertGreater(len(backfill), 0)

    def test_review_queue_prioritizes_reliable_written_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            retriever = FanKnowledgeRetriever(Path(temporary) / "knowledge.sqlite3")
            retriever.upsert_document(
                KnowledgeDocument(
                    "document:ranking",
                    "official_news",
                    "ranking",
                    "告知",
                    "すやバラ公式とすやバラ配信",
                ),
                [KnowledgeChunk(0, "すやバラ公式とすやバラ配信")],
            )
            document = retriever.pending_term_documents()[0]
            chunk_id = document.chunks[0].chunk_id
            retriever.replace_term_candidate_occurrences(
                [document.document_id],
                [
                    TermCandidateOccurrence(
                        "すやバラ公式",
                        "すやバラ公式",
                        document.document_id,
                        document.document_id,
                        chunk_id,
                        "[OFFICIAL]すやバラ公式",
                        "official_news",
                        1.0,
                        True,
                        1,
                    ),
                    TermCandidateOccurrence(
                        "すやバラ配信",
                        "すやバラ配信",
                        document.document_id,
                        document.document_id,
                        chunk_id,
                        "[ASR]すやバラ配信",
                        "youtube_auto_subtitle",
                        0.6,
                        True,
                        100,
                    ),
                ],
            )
            retriever.finish_term_preparation(
                [document],
                touched_forms=["すやバラ公式", "すやバラ配信"],
                pending_review_forms=["すやバラ公式", "すやバラ配信"],
                backfill=False,
            )

            pending = retriever.pending_term_review_forms()
            retriever.close()

        self.assertEqual(pending, ["すやバラ公式", "すやバラ配信"])

    def test_validation_caps_new_terms_and_defers_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            retriever = FanKnowledgeRetriever(Path(temporary) / "knowledge.sqlite3")
            retriever.upsert_document(
                KnowledgeDocument(
                    "document:term-cap",
                    "official_news",
                    "term-cap",
                    "告知",
                    "すやバラ甲、すやバラ乙、すやバラ丙",
                ),
                [KnowledgeChunk(0, "すやバラ甲、すやバラ乙、すやバラ丙")],
            )
            document = retriever.pending_term_documents()[0]
            chunk_id = document.chunks[0].chunk_id
            surfaces = ("すやバラ甲", "すやバラ乙", "すやバラ丙")
            retriever.replace_term_candidate_occurrences(
                [document.document_id],
                [
                    TermCandidateOccurrence(
                        surface,
                        surface,
                        document.document_id,
                        document.document_id,
                        chunk_id,
                        f"[OFFICIAL]{surface}",
                        "official_news",
                        1.0,
                        True,
                        1,
                    )
                    for surface in surfaces
                ],
            )
            retriever.finish_term_preparation(
                [document],
                touched_forms=surfaces,
                pending_review_forms=surfaces,
                backfill=False,
            )

            def request(_body: dict[str, object]) -> dict[str, object]:
                return _response(
                    {
                        "terms": [
                            {
                                "candidate": surface,
                                "canonical_zh": surface,
                                "confidence": 0.9,
                                "action": "accept",
                                "search_query": "",
                            }
                            for surface in surfaces
                        ]
                    }
                )

            validate_pending_terms(
                retriever,
                request=request,
                model="test-model",
                max_tokens=1024,
                thinking=None,
                max_retries=1,
                maximum_new_terms=2,
            )
            active = retriever._database.execute(
                "SELECT surface FROM knowledge_extracted_terms "
                "WHERE status = 'active' ORDER BY surface"
            ).fetchall()
            pending = retriever.pending_term_review_forms()
            retriever.close()

        self.assertEqual(len(active), 2)
        self.assertEqual(len(pending), 1)
        self.assertNotIn(pending[0], {row["surface"] for row in active})

    def test_local_patterns_cover_domain_name_forms(self) -> None:
        values = _local_term_surfaces(
            "夢限大みゅーたいぷのすやバラ配信。"
            "『新宿着陸計画』とプロジェクトセカイ、BanG Dream!、"
            "#ゆめみたライブ、の話です。"
        )

        self.assertIn("夢限大みゅーたいぷ", values)
        self.assertIn("すやバラ", values)
        self.assertIn("新宿着陸計画", values)
        self.assertIn("プロジェクトセカイ", values)
        self.assertIn("#ゆめみたライブ", values)

    def test_candidates_accumulate_before_llm_screening(self) -> None:
        candidates = _collect_local_term_candidates(
            [
                _document("one", "youtube_auto_subtitle", "すやバラで歌います"),
                _document("two", "x_post", "今夜もすやバラです", author="メンバー"),
            ]
        )
        candidate = next(value for value in candidates if value.surface == "すやバラ")

        self.assertEqual(candidate.document_count, 2)
        self.assertEqual(len(candidate.occurrences), 2)
        self.assertEqual(
            candidate.source_counts,
            {"youtube_auto_subtitle": 1, "x_post": 1},
        )
        self.assertTrue(any("[ASR]" in value.context for value in candidate.occurrences))
        self.assertTrue(any("[X][メンバー]" in value.context for value in candidate.occurrences))
        self.assertEqual(
            _candidate_prompt_value(candidate)["source_counts"],
            {"ASR": 1, "X": 1},
        )

    def test_candidates_accumulate_across_incremental_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            retriever = FanKnowledgeRetriever(Path(temporary) / "knowledge.sqlite3")
            calls: list[dict[str, object]] = []

            def request(body: dict[str, object]) -> dict[str, object]:
                calls.append(body)
                return _response(
                    {
                        "terms": [
                            {
                                "candidate": "すやバラ",
                                "canonical_zh": "Suyapara",
                                "confidence": 0.8,
                                "action": "accept",
                                "search_query": "",
                            }
                        ]
                    }
                )

            retriever.upsert_document(
                KnowledgeDocument(
                    "document:first",
                    "youtube_auto_subtitle",
                    "video:first",
                    "第一场",
                    "すやバラで歌います。",
                ),
                [KnowledgeChunk(0, "すやバラで歌います。")],
            )
            first = extract_pending_terms(
                retriever,
                request=request,
                model="test-model",
                max_tokens=1024,
                thinking=None,
                max_retries=1,
            )
            retriever.upsert_document(
                KnowledgeDocument(
                    "document:second",
                    "x_post",
                    "post:second",
                    "第二次提及",
                    "今夜もすやバラです。",
                ),
                [KnowledgeChunk(0, "今夜もすやバラです。")],
            )
            second = extract_pending_terms(
                retriever,
                request=request,
                model="test-model",
                max_tokens=1024,
                thinking=None,
                max_retries=1,
            )
            retriever.upsert_document(
                KnowledgeDocument(
                    "document:third",
                    "youtube_auto_subtitle",
                    "video:third",
                    "第三次提及",
                    "すやバラを始めます。",
                ),
                [KnowledgeChunk(0, "すやバラを始めます。")],
            )
            third = extract_pending_terms(
                retriever,
                request=request,
                model="test-model",
                max_tokens=1024,
                thinking=None,
                max_retries=1,
            )
            occurrence_count = retriever._database.execute(
                "SELECT COUNT(*) AS value "
                "FROM knowledge_term_candidate_occurrences "
                "WHERE normalized_form = 'すやバラ'"
            ).fetchone()["value"]
            evidence_count = retriever._database.execute(
                "SELECT COUNT(*) AS value FROM knowledge_term_evidence"
            ).fetchone()["value"]
            retriever.close()

        self.assertEqual(first.candidates, 0)
        self.assertEqual(second.candidates, 1)
        self.assertEqual(third.candidates, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(occurrence_count, 3)
        self.assertEqual(evidence_count, 3)

    def test_super_chat_username_is_not_a_local_candidate(self) -> None:
        document = _document(
            "sc",
            "youtube_super_chat",
            "SC ¥500 ビクターさん: プロジェクトセカイ最高です",
        )
        text = _candidate_source_text(document, document.chunks[0])
        surfaces = set(_local_term_surfaces(text))

        self.assertNotIn("ビクター", surfaces)
        self.assertNotIn("ビクターさん", surfaces)
        self.assertIn("プロジェクトセカイ", surfaces)

        candidates = _collect_local_term_candidates(
            [
                document,
                _document(
                    "x",
                    "x_post",
                    "プロジェクトセカイを遊びます",
                    author="公式アカウント",
                ),
            ]
        )
        candidate = next(
            value for value in candidates if value.surface == "プロジェクトセカイ"
        )
        prompt_value = _candidate_prompt_value(candidate)
        self.assertIn(
            "[SC][ビクターさん]《配信》プロジェクトセカイ最高です",
            prompt_value["contexts"],
        )
        self.assertEqual(prompt_value["source_counts"], {"SC": 1, "X": 1})

    def test_llm_screens_only_local_candidates_and_stores_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            retriever = FanKnowledgeRetriever(Path(temporary) / "knowledge.sqlite3")
            retriever.upsert_document(
                KnowledgeDocument(
                    "document:project",
                    "official_news",
                    "project",
                    "企划公告",
                    "『プロジェクトセカイ』を開催します。",
                ),
                [KnowledgeChunk(0, "『プロジェクトセカイ』を開催します。")],
            )

            def request(_body: dict[str, object]) -> dict[str, object]:
                return _response(
                    {
                        "terms": [
                            {
                                "candidate": "プロジェクトセカイ",
                                "canonical_zh": "世界计划",
                                "confidence": 0.94,
                                "action": "accept",
                                "search_query": "",
                            },
                            {
                                "candidate": "输入中不存在的词",
                                "canonical_zh": "幻觉",
                                "confidence": 1.0,
                                "action": "accept",
                                "search_query": "",
                            },
                        ]
                    }
                )

            summary = extract_pending_terms(
                retriever,
                request=request,
                model="test-model",
                max_tokens=1024,
                thinking=None,
                max_retries=1,
            )
            rows = retriever._database.execute(
                "SELECT surface, canonical_zh FROM knowledge_extracted_terms"
            ).fetchall()
            evidence = retriever._database.execute(
                "SELECT evidence_text FROM knowledge_term_evidence"
            ).fetchall()
            retriever.close()

        self.assertEqual((summary.stored, summary.published), (1, 1))
        self.assertEqual([(row["surface"], row["canonical_zh"]) for row in rows], [("プロジェクトセカイ", "世界计划")])
        self.assertIn("プロジェクトセカイ", evidence[0]["evidence_text"])

    def test_search_candidate_gets_one_focused_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            retriever = FanKnowledgeRetriever(Path(temporary) / "knowledge.sqlite3")
            retriever.upsert_document(
                KnowledgeDocument(
                    "document:event",
                    "x_post",
                    "event",
                    "ライブ告知",
                    "『めたもるふぉーぜ』を開催します。",
                ),
                [KnowledgeChunk(0, "『めたもるふぉーぜ』を開催します。")],
            )
            responses = iter(
                [
                    {
                        "terms": [
                            {
                                "candidate": "めたもるふぉーぜ",
                                "canonical_zh": "蜕变",
                                "confidence": 0.75,
                                "action": "search",
                                "search_query": "めたもるふぉーぜ ライブ",
                            }
                        ]
                    },
                    {"decision": "accept", "canonical_zh": "Metamorphose"},
                ]
            )
            prompts: list[str] = []

            def request(body: dict[str, object]) -> dict[str, object]:
                prompts.append(json.dumps(body, ensure_ascii=False))
                return _response(next(responses))

            queries: list[str] = []

            def search(query: str) -> dict[str, object]:
                queries.append(query)
                return {
                    "results": [
                        {
                            "title": "公式ライブ",
                            "url": "https://official.example/event",
                            "snippet": "めたもるふぉーぜ",
                        }
                    ]
                }

            summary = extract_pending_terms(
                retriever,
                request=request,
                search_web=search,
                model="test-model",
                max_tokens=1024,
                thinking=None,
                max_retries=1,
            )
            row = retriever._database.execute(
                "SELECT canonical_zh FROM knowledge_extracted_terms"
            ).fetchone()
            retriever.close()

        self.assertEqual(summary.stored, 1)
        self.assertEqual(queries, ["めたもるふぉーぜ ライブ"])
        self.assertEqual(row["canonical_zh"], "Metamorphose")
        self.assertEqual(len(prompts), 2)

    def test_candidate_batching_keeps_candidate_records_whole(self) -> None:
        document = _document("one", "official_news", "プロジェクトセカイ")
        candidates = [
            LocalTermCandidate(
                f"プロジェクト{index}",
                (
                    LocalTermOccurrence(
                        document_id=document.document_id,
                        logical_document_id=document.document_id,
                        chunk_id=document.chunks[0].chunk_id,
                        context=f"[OFFICIAL]プロジェクト{index}" * 40,
                        source_type=document.source_type,
                        reliability=document.reliability,
                    ),
                ),
            )
            for index in range(12)
        ]

        batches = _candidate_batches(
            candidates,
            context_size=4096,
            max_tokens=512,
            target_input_tokens=2500,
        )

        self.assertGreater(len(batches), 1)
        self.assertEqual(
            [value.surface for batch in batches for value in batch],
            [value.surface for value in candidates],
        )

    def test_empty_local_candidate_set_completes_document_without_llm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            retriever = FanKnowledgeRetriever(Path(temporary) / "knowledge.sqlite3")
            retriever.upsert_document(
                KnowledgeDocument(
                    "document:empty", "official_news", "empty", "配信", "。"
                ),
                [KnowledgeChunk(0, "。")],
            )
            calls = 0

            def request(_body: dict[str, object]) -> dict[str, object]:
                nonlocal calls
                calls += 1
                return _response({"terms": []})

            summary = extract_pending_terms(
                retriever,
                request=request,
                model="test-model",
                max_tokens=1024,
                thinking=None,
                max_retries=1,
            )
            pending = retriever.pending_term_documents()
            retriever.close()

        self.assertEqual(calls, 0)
        self.assertEqual(summary.documents, 1)
        self.assertEqual(pending, [])


if __name__ == "__main__":
    unittest.main()
