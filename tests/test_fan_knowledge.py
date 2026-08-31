from __future__ import annotations

import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from subtitle_pipeline.fan_knowledge import (
    ExtractedTerm,
    FanKnowledgeRetriever,
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeQuery,
    KnowledgeRecord,
    _sudachi_tokenizer,
    records_from_translation_context,
)


class FanKnowledgeRetrieverTests(unittest.TestCase):
    def test_changed_documents_are_queued_for_term_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            retriever = FanKnowledgeRetriever(Path(temporary) / "knowledge.sqlite3")
            document = KnowledgeDocument(
                "document:term-source",
                "youtube_metadata",
                "term-source",
                "すやバラ配信",
                "すやバラは、すやすやバラード歌枠の略です。",
            )
            chunks = [KnowledgeChunk(0, "すやバラは、すやすやバラード歌枠の略です。")]

            first = retriever.upsert_document(document, chunks)
            second = retriever.upsert_document(document, chunks)
            pending = retriever.pending_term_documents()

            self.assertTrue(first.changed)
            self.assertFalse(second.changed)
            self.assertEqual(
                [value.document_id for value in pending], [document.document_id]
            )
            self.assertEqual(len(pending[0].chunks), 1)
            retriever.close()

    def test_existing_documents_can_be_queued_for_backfill(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            retriever = FanKnowledgeRetriever(Path(temporary) / "knowledge.sqlite3")
            retriever.upsert_document(
                KnowledgeDocument(
                    "document:backfill",
                    "official_news",
                    "backfill",
                    "既存資料",
                    "既存資料の本文です。",
                ),
                [KnowledgeChunk(0, "既存資料の本文です。")],
            )
            pending = retriever.pending_term_documents()[0]
            retriever.store_extracted_terms(pending, [])
            retriever.finish_term_preparation(
                [pending],
                touched_forms=(),
                pending_review_forms=(),
                backfill=False,
            )
            self.assertEqual(retriever.pending_term_documents(), [])

            queued = retriever.queue_all_documents_for_term_extraction()

            self.assertEqual(queued, 1)
            self.assertEqual(
                [
                    value.document_id
                    for value in retriever.pending_term_documents(include_backfill=True)
                ],
                ["document:backfill"],
            )
            retriever.close()

    def test_reviewed_terms_are_published_as_references(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            retriever = FanKnowledgeRetriever(Path(temporary) / "knowledge.sqlite3")
            retriever.upsert_document(
                KnowledgeDocument(
                    "document:term-source",
                    "youtube_metadata",
                    "term-source",
                    "すやバラ配信",
                    "すやバラは、すやすやバラード歌枠の略です。",
                    reliability=0.95,
                ),
                [KnowledgeChunk(0, "すやバラは、すやすやバラード歌枠の略です。")],
            )
            document = retriever.pending_term_documents()[0]
            stored, published = retriever.store_extracted_terms(
                document,
                [
                    ExtractedTerm(
                        "すやバラ",
                        "助眠抒情歌回",
                        ("すやすやバラード歌枠",),
                        "すやばら",
                        "definition",
                        (document.chunks[0].chunk_id,),
                        0.94,
                    )
                ],
            )
            retriever.finish_term_preparation(
                [document],
                touched_forms=(),
                pending_review_forms=(),
                backfill=False,
            )

            hits = retriever.retrieve_term_references(
                KnowledgeQuery("すやバラ", top_k=4)
            )
            self.assertEqual((stored, published), (1, 1))
            self.assertEqual(hits[0].title, "すやバラ")
            self.assertIn("助眠抒情歌回", hits[0].body)
            self.assertEqual(retriever.pending_term_documents(), [])
            retriever.close()

    def test_term_forms_are_deduplicated_after_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            retriever = FanKnowledgeRetriever(Path(temporary) / "knowledge.sqlite3")
            retriever.upsert_document(
                KnowledgeDocument(
                    "document:term-forms",
                    "youtube_metadata",
                    "term-forms",
                    "USJ",
                    "USJについて話します。",
                ),
                [KnowledgeChunk(0, "USJについて話します。")],
            )
            document = retriever.pending_term_documents()[0]
            retriever.store_extracted_terms(
                document,
                [
                    ExtractedTerm(
                        "USJ",
                        "日本环球影城",
                        ("usj", "ＵＳＪ"),
                        "",
                        "name",
                        (document.chunks[0].chunk_id,),
                        0.9,
                    )
                ],
            )
            rows = retriever._database.execute(
                "SELECT normalized_form, form FROM knowledge_term_forms"
            ).fetchall()
            retriever.close()

        self.assertEqual(
            [dict(row) for row in rows],
            [{"normalized_form": "usj", "form": "USJ"}],
        )

    def test_extracted_term_schema_has_no_entity_kind(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            retriever = FanKnowledgeRetriever(Path(temporary) / "knowledge.sqlite3")
            columns = {
                str(row["name"])
                for row in retriever._database.execute(
                    "PRAGMA table_info(knowledge_extracted_terms)"
                ).fetchall()
            }
            retriever.close()

        self.assertNotIn("kind", columns)

    def test_reviewed_term_from_weak_source_is_still_a_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            retriever = FanKnowledgeRetriever(Path(temporary) / "knowledge.sqlite3")
            retriever.upsert_document(
                KnowledgeDocument(
                    "document:sc",
                    "youtube_superchat",
                    "sc",
                    "SC",
                    "すやバラは助眠抒情歌回です。",
                    reliability=0.5,
                ),
                [KnowledgeChunk(0, "すやバラは助眠抒情歌回です。")],
            )
            document = retriever.pending_term_documents()[0]
            stored, published = retriever.store_extracted_terms(
                document,
                [
                    ExtractedTerm(
                        "すやバラ",
                        "助眠抒情歌回",
                        (),
                        "",
                        "definition",
                        (document.chunks[0].chunk_id,),
                        0.99,
                    )
                ],
            )

            self.assertEqual((stored, published), (1, 1))
            hits = retriever.retrieve_term_references(KnowledgeQuery("すやバラ"))
            self.assertEqual(len(hits), 1)
            self.assertIn("助眠抒情歌回", hits[0].body)
            retriever.close()

    def test_existing_provisional_terms_are_activated_on_open(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "knowledge.sqlite3"
            retriever = FanKnowledgeRetriever(database_path)
            retriever.upsert_document(
                KnowledgeDocument(
                    "document:term-source",
                    "youtube_metadata",
                    "term-source",
                    "企画名",
                    "ミライは企画名です。",
                ),
                [KnowledgeChunk(0, "ミライは企画名です。")],
            )
            document = retriever.pending_term_documents()[0]
            retriever.store_extracted_terms(
                document,
                [
                    ExtractedTerm(
                        "ミライ",
                        "未来企划",
                        (),
                        "みらい",
                        "name",
                        (document.chunks[0].chunk_id,),
                        0.8,
                    )
                ],
            )
            retriever._database.execute(
                "UPDATE knowledge_extracted_terms SET status = 'provisional'"
            )
            retriever._database.execute(
                "DELETE FROM knowledge_records WHERE record_id LIKE 'extracted:%'"
            )
            retriever._database.commit()
            retriever.close()

            reopened = FanKnowledgeRetriever(database_path)
            row = reopened._database.execute(
                "SELECT status FROM knowledge_extracted_terms"
            ).fetchone()
            hits = reopened.retrieve_term_references(KnowledgeQuery("ミライ"))
            reopened.close()

        self.assertEqual(row["status"], "active")
        self.assertEqual(len(hits), 1)

    def test_conflicting_sources_do_not_replace_an_active_translation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            retriever = FanKnowledgeRetriever(Path(temporary) / "knowledge.sqlite3")
            for document_id, source_type, translation in (
                ("official", "official_news", "助眠抒情歌回"),
                ("subtitle", "youtube_auto_subtitle", "睡觉芭乐"),
            ):
                retriever.upsert_document(
                    KnowledgeDocument(
                        f"document:{document_id}",
                        source_type,
                        document_id,
                        "すやバラ",
                        "すやバラは配信企画名です。",
                        reliability=0.95,
                    ),
                    [KnowledgeChunk(0, "すやバラは配信企画名です。")],
                )
                document = next(
                    value
                    for value in retriever.pending_term_documents()
                    if value.document_id == f"document:{document_id}"
                )
                retriever.store_extracted_terms(
                    document,
                    [
                        ExtractedTerm(
                            "すやバラ",
                            translation,
                            (),
                            "",
                            "definition",
                            (document.chunks[0].chunk_id,),
                            0.95,
                        )
                    ],
                )

            hits = retriever.retrieve_term_references(
                KnowledgeQuery("すやバラ", top_k=4)
            )
            retriever.close()

        self.assertEqual(len(hits), 1)
        self.assertIn("助眠抒情歌回", hits[0].body)
        self.assertNotIn("睡觉芭乐", hits[0].body)

    def test_curated_mapping_overrides_extracted_translation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            retriever = FanKnowledgeRetriever(Path(temporary) / "knowledge.sqlite3")
            retriever.upsert(
                [
                    KnowledgeRecord(
                        "term:yumemita",
                        "term",
                        "ゆめみた",
                        "ゆめみた的固定中文译法为梦限大MewType。",
                        source_type="curated_glossary",
                        reliability=0.99,
                    )
                ]
            )
            retriever.upsert_document(
                KnowledgeDocument(
                    "document:yumemita",
                    "youtube_auto_subtitle",
                    "yumemita",
                    "ゆめみた配信",
                    "ゆめみたの配信です。",
                ),
                [KnowledgeChunk(0, "ゆめみたの配信です。")],
            )
            document = retriever.pending_term_documents()[0]

            stored, published = retriever.store_extracted_terms(
                document,
                [
                    ExtractedTerm(
                        "ゆめみた",
                        "梦见",
                        (),
                        "",
                        "name",
                        (document.chunks[0].chunk_id,),
                        0.95,
                    )
                ],
            )
            row = retriever._database.execute(
                "SELECT canonical_zh, status FROM knowledge_extracted_terms"
            ).fetchone()
            retriever.close()

        self.assertEqual((stored, published), (1, 0))
        self.assertEqual(
            dict(row),
            {
                "canonical_zh": "梦限大MewType",
                "status": "verified",
            },
        )

    def test_term_references_are_retrieved_by_alias_and_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            retriever = FanKnowledgeRetriever(Path(temporary) / "knowledge.sqlite3")
            retriever.upsert(
                [
                    KnowledgeRecord(
                        "term:suyabara",
                        "term",
                        "すやバラ",
                        "すやバラ的固定中文译法为助眠抒情歌回。",
                        aliases=("スヤバラ",),
                        reliability=0.98,
                    ),
                    KnowledgeRecord(
                        "character:arale",
                        "character",
                        "仲町あられ",
                        "仲町あられ的固定中文名为仲町阿拉蕾。",
                        speaker="nakamachi_arale",
                        reliability=0.98,
                    ),
                    KnowledgeRecord(
                        "note:background",
                        "note",
                        "背景",
                        "普通背景资料不属于术语引用。",
                    ),
                ]
            )
            retriever.upsert_document(
                KnowledgeDocument(
                    "document:suyabara",
                    "youtube_metadata",
                    "suyabara",
                    "スヤバラ配信",
                    "スヤバラは安眠向けの歌枠です。",
                ),
                [KnowledgeChunk(0, "スヤバラは安眠向けの歌枠です。")],
            )

            query = KnowledgeQuery(
                "スヤバラについて話す",
                speaker="nakamachi_arale",
                top_k=8,
            )
            hits = retriever.retrieve_term_references(query)
            background = retriever.retrieve_background(query)
            retriever.close()

        self.assertEqual(
            {hit.record_id for hit in hits},
            {"term:suyabara", "character:arale"},
        )
        self.assertTrue(all("term_reference" in hit.retrieval_ranks for hit in hits))
        self.assertTrue(background)
        self.assertTrue(
            all(hit.kind not in {"term", "character", "entity"} for hit in background)
        )

    def test_cross_encoder_reranks_body_and_rejects_title_only_match(self) -> None:
        class FakeCrossEncoder:
            def predict(self, pairs, **_kwargs):
                return np.asarray(
                    [
                        0.9 if "すやすやバラード" in passage else 0.001
                        for _query, passage in pairs
                    ],
                    dtype=np.float32,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            retriever = FanKnowledgeRetriever(
                root / "knowledge.sqlite3",
                reranker_model="fake-reranker",
                reranker_minimum_score=0.05,
            )
            retriever.upsert_document(
                KnowledgeDocument(
                    "document:useful",
                    "youtube_auto_subtitle",
                    "useful",
                    "すやバラ",
                    "すやバラは、すやすやバラードを略した配信名です。",
                ),
                [KnowledgeChunk(0, "すやバラは、すやすやバラードを略した配信名です。")],
            )
            retriever.upsert_document(
                KnowledgeDocument(
                    "document:title-only",
                    "youtube_auto_subtitle",
                    "title-only",
                    "すやバラ",
                    "今日は全く別の商品について話しています。",
                ),
                [KnowledgeChunk(0, "今日は全く別の商品について話しています。")],
            )
            assert retriever._reranker is not None
            retriever._reranker._model = FakeCrossEncoder()

            hits = retriever.retrieve(KnowledgeQuery("すやバラ", top_k=4))
            retriever.close()

        self.assertEqual(len(hits), 1)
        self.assertIn("すやすやバラード", hits[0].body)
        self.assertGreater(hits[0].score.reranker, 0.8)

    def test_sudachi_tokenizer_is_thread_local(self) -> None:
        workers = 4
        barrier = threading.Barrier(workers)

        def tokenizer_pair() -> tuple[object, object]:
            first = _sudachi_tokenizer()
            barrier.wait()
            return first, _sudachi_tokenizer()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            pairs = list(executor.map(lambda _index: tokenizer_pair(), range(workers)))

        self.assertTrue(all(first is second for first, second in pairs))
        self.assertEqual(len({id(first) for first, _second in pairs}), workers)

    def test_vector_search_recalls_semantic_hit_without_lexical_overlap(self) -> None:
        class FakeEmbeddingModel:
            def get_embedding_dimension(self) -> int:
                return 2

            def encode(self, values, **_kwargs):
                return np.asarray(
                    [
                        [1.0, 0.0]
                        if "雕像" in value or "expensive figure" in value
                        else [0.0, 1.0]
                        for value in values
                    ],
                    dtype=np.float32,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            retriever = FanKnowledgeRetriever(
                root / "knowledge.sqlite3",
                embedding_model="fake-model",
                vector_index_path=root / "knowledge.faiss",
            )
            retriever.upsert_document(
                KnowledgeDocument(
                    "document:figure",
                    "official_news",
                    "figure",
                    "商品新闻",
                    "四十万日元的等身大雕像手办正式发售",
                ),
                [KnowledgeChunk(0, "四十万日元的等身大雕像手办正式发售")],
            )
            assert retriever._vector_index is not None
            retriever._vector_index._model = FakeEmbeddingModel()
            retriever.sync_vector_index()

            hits = retriever.retrieve(KnowledgeQuery("expensive figure"))
            retriever.close()

        self.assertEqual(hits[0].kind, "document_chunk")
        self.assertGreaterEqual(hits[0].score.vector, 0.99)
        self.assertIn("vector", hits[0].retrieval_ranks)

    def test_context_records_preserve_entities_aliases_and_fixed_translation(
        self,
    ) -> None:
        records = records_from_translation_context(
            {
                "characters": [
                    {
                        "id": "fuji_miyako",
                        "canonical": "藤都子",
                        "source_name": "藤都子",
                        "aliases": ["ふじみやこ"],
                        "short_names": [{"source": "ミヤコ", "target": "都子"}],
                    }
                ],
                "terms": {"夢限大みゅーたいぷ": "梦限大MewType"},
                "asr_entities": [
                    {
                        "surface": "夢限大みゅーたいぷ",
                        "reading": "むげんだいみゅーたいぷ",
                        "aliases": ["無限大ミュータイプ"],
                    }
                ],
                "knowledge_records": [
                    {
                        "id": "miyako-ty",
                        "kind": "catchphrase",
                        "title": "TY",
                        "body": "TY表示Thank You。",
                        "aliases": ["ティーワイ"],
                        "keywords": ["感谢"],
                        "speaker": "fuji_miyako",
                        "reliability": 1.0,
                    }
                ],
            }
        )
        character = next(record for record in records if record.kind == "character")
        term = next(record for record in records if record.kind == "term")
        self.assertEqual(character.speaker, "fuji_miyako")
        self.assertIn("ミヤコ", character.aliases)
        self.assertEqual(term.reading, "むげんだいみゅーたいぷ")
        self.assertIn("無限大ミュータイプ", term.aliases)
        self.assertIn("梦限大MewType", term.body)
        knowledge = next(record for record in records if record.kind == "catchphrase")
        self.assertEqual(knowledge.record_id, "knowledge:miyako-ty")
        self.assertEqual(knowledge.speaker, "fuji_miyako")

    def test_retrieval_combines_alias_speaker_date_and_chat_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            retriever = FanKnowledgeRetriever(
                root / "knowledge.sqlite3",
                audit_path=root / "audit.jsonl",
            )
            retriever.upsert(
                [
                    KnowledgeRecord(
                        "event:current",
                        "event",
                        "高价雕像手办",
                        "主播谈到高价雕像后，觉得亚克力立牌相对便宜。",
                        aliases=("アクスタ", "フィギュア"),
                        speaker="nakamachi_arale",
                        valid_from="2026-08-01",
                        reliability=0.95,
                    ),
                    KnowledgeRecord(
                        "event:other",
                        "event",
                        "其他成员的手办",
                        "另一位成员谈过普通手办。",
                        aliases=("フィギュア",),
                        speaker="fuji_miyako",
                        valid_to="2025-12-31",
                        reliability=0.7,
                    ),
                ]
            )
            hits = retriever.retrieve(
                KnowledgeQuery(
                    "アクスタフィギュアが安く感じる",
                    speaker="nakamachi_arale",
                    video_date="2026-08-11",
                    chat_text="アクスタは罠 フィギュア高い",
                )
            )
            retriever.close()

            audit = json.loads(
                (root / "audit.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )

        self.assertEqual(hits[0].record_id, "event:current")
        self.assertGreater(hits[0].score.speaker, 0)
        self.assertGreater(hits[0].score.date, 0)
        self.assertGreater(hits[0].score.cross_evidence, 0)
        self.assertIn("score_components", audit["hits"][0])

    def test_fts_finds_background_text_without_alias_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            retriever = FanKnowledgeRetriever(Path(temporary) / "knowledge.sqlite3")
            retriever.upsert(
                [
                    KnowledgeRecord(
                        "note:ty",
                        "note",
                        "TY",
                        "TY在这里表示Thank You，不是人物姓名。",
                        keywords=("ありがとう", "貢ぎ物", "お納めください"),
                        reliability=1.0,
                    )
                ]
            )
            hits = retriever.retrieve(
                KnowledgeQuery("本日分の貢ぎ物をお納めください、次回")
            )
            retriever.close()

        self.assertEqual(hits[0].record_id, "note:ty")
        self.assertGreater(hits[0].score.keyword, 0)

    def test_retrieval_rejects_fts_floor_without_substantive_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            retriever = FanKnowledgeRetriever(
                root / "knowledge.sqlite3", audit_path=root / "audit.jsonl"
            )
            retriever.upsert_document(
                KnowledgeDocument(
                    "document:weak",
                    "youtube_auto_subtitle",
                    "weak-video:ja",
                    "无关直播",
                    "配信とは無関係な長い記録です。",
                ),
                [KnowledgeChunk(0, "配信とは無関係な長い記録です。")],
            )

            hits = retriever.retrieve(KnowledgeQuery("今日は配信を始めます"))
            retriever.close()
            audit = json.loads(root.joinpath("audit.jsonl").read_text())

        self.assertEqual(hits, [])
        self.assertGreater(audit["weak_candidate_count"], 0)
        self.assertEqual(
            audit["rejected"][0]["gate_reason"],
            "no substantive lexical, semantic, entity, or cross evidence",
        )

    def test_results_are_diversified_across_source_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            retriever = FanKnowledgeRetriever(Path(temporary) / "knowledge.sqlite3")
            retriever.upsert_document(
                KnowledgeDocument(
                    "document:one",
                    "youtube_auto_subtitle",
                    "video-one:ja",
                    "第一场直播",
                    "特製限定アクリルスタンド販売情報",
                    source_url="https://youtube.example/watch?v=one",
                ),
                [
                    KnowledgeChunk(
                        index,
                        f"特製限定アクリルスタンド販売情報 {suffix}",
                    )
                    for index, suffix in enumerate(
                        (
                            "価格について詳しく話した",
                            "予約期間を案内した",
                            "商品写真を紹介した",
                        )
                    )
                ],
            )
            retriever.upsert_document(
                KnowledgeDocument(
                    "document:two",
                    "official_news",
                    "news-two",
                    "公式商品公告",
                    "特製限定アクリルスタンド販売情報",
                    source_url="https://official.example/news/two",
                ),
                [KnowledgeChunk(0, "特製限定アクリルスタンド販売情報 公式発表")],
            )

            hits = retriever.retrieve(
                KnowledgeQuery("特製限定アクリルスタンド販売情報", top_k=4)
            )
            retriever.close()

        first_source = [
            hit
            for hit in hits
            if hit.source_url == "https://youtube.example/watch?v=one"
        ]
        self.assertLessEqual(len(first_source), 2)
        self.assertTrue(
            any(hit.source_url == "https://official.example/news/two" for hit in hits)
        )

    def test_document_chunks_are_incremental_and_searchable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            retriever = FanKnowledgeRetriever(Path(temporary) / "knowledge.sqlite3")
            document = KnowledgeDocument(
                "document:stream",
                "youtube_auto_subtitle",
                "video-1:ja",
                "直播标题",
                "主播讨论二十七万日元的雕像手办。",
                source_url="https://youtube.example/watch?v=video-1",
                author="nakamachi_arale",
                published_at="2026-08-11T00:00:00+00:00",
                language="ja",
                reliability=0.7,
            )
            chunks = [
                KnowledgeChunk(
                    0,
                    "几十万日元的雕像让亚克力立牌显得便宜",
                    320.0,
                    345.0,
                    "nakamachi_arale",
                    "ja",
                )
            ]
            first = retriever.upsert_document(document, chunks)
            second = retriever.upsert_document(document, chunks)
            hits = retriever.retrieve(
                KnowledgeQuery(
                    "亚克力立牌为什么显得便宜",
                    speaker="nakamachi_arale",
                    video_date="2026-08-11",
                )
            )

            self.assertTrue(first.changed)
            self.assertFalse(second.changed)
            self.assertEqual(retriever.document_count(), 1)
            self.assertEqual(retriever.chunk_count(), 1)
            retriever.close()

        self.assertEqual(hits[0].kind, "document_chunk")
        self.assertIn("几十万日元", hits[0].body)

    def test_retrieval_excludes_documents_from_current_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            retriever = FanKnowledgeRetriever(Path(temporary) / "knowledge.sqlite3")
            for video_id, text in (
                ("abc_def1234", "アクスタは現在の動画"),
                ("abcXdef1234", "アクスタは過去の配信でも話した"),
            ):
                retriever.upsert_document(
                    KnowledgeDocument(
                        f"document:{video_id}",
                        "youtube_auto_subtitle",
                        f"{video_id}:ja",
                        video_id,
                        text,
                    ),
                    [KnowledgeChunk(0, text)],
                )

            hits = retriever.retrieve(
                KnowledgeQuery("アクスタ", exclude_video_id="abc_def1234")
            )
            retriever.close()

        self.assertTrue(hits)
        self.assertTrue(all("現在の動画" not in hit.body for hit in hits))


if __name__ == "__main__":
    unittest.main()
