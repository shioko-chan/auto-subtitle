from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from subtitle_pipeline.fan_knowledge import (
    FanKnowledgeRetriever,
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeQuery,
    KnowledgeRecord,
    records_from_translation_context,
)


class FanKnowledgeRetrieverTests(unittest.TestCase):
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
                        ("価格について詳しく話した", "予約期間を案内した", "商品写真を紹介した")
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
