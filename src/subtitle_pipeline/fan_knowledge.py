from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import threading
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, date, datetime
from difflib import SequenceMatcher
from functools import cache
from pathlib import Path
from typing import Self

from .cross_encoder import LocalCrossEncoderReranker
from .vector_index import LocalVectorIndex

_LOGGER = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.-]*|[\u3040-\u30ff\u3400-\u9fff]+")
_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_SCHEMA_VERSION = 7
_LEXICAL_CANDIDATES = 40
_VECTOR_CANDIDATES = 40
_FUSION_CANDIDATES = 30
_RRF_K = 60
_RRF_WEIGHTS = {"fts": 1.0, "vector": 1.0, "entity": 2.0}
_MAX_RESULTS_PER_SOURCE = 2
_TERM_REFERENCE_KINDS = frozenset({"term", "character", "entity"})
_SUDACHI_LOCAL = threading.local()
_CURATED_TRANSLATION_RE = re.compile(r"固定中文(?:名|译法)为(.+?)。$")


@dataclass(frozen=True)
class KnowledgeRecord:
    record_id: str
    kind: str
    title: str
    body: str
    aliases: tuple[str, ...] = ()
    reading: str = ""
    keywords: tuple[str, ...] = ()
    speaker: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    source_url: str | None = None
    source_type: str = "curated_glossary"
    reliability: float = 0.8


@dataclass(frozen=True)
class KnowledgeDocument:
    document_id: str
    source_type: str
    external_id: str
    title: str
    text: str
    source_url: str | None = None
    author: str | None = None
    published_at: str | None = None
    fetched_at: str | None = None
    language: str | None = None
    metadata: dict[str, object] | None = None
    reliability: float = 0.8


@dataclass(frozen=True)
class KnowledgeChunk:
    ordinal: int
    text: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    speaker: str | None = None
    language: str | None = None


@dataclass(frozen=True)
class DocumentUpsertResult:
    document_id: str
    changed: bool
    chunk_count: int


@dataclass(frozen=True)
class TermExtractionChunk:
    chunk_id: str
    text: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    speaker: str | None = None
    language: str | None = None
    document_id: str = ""
    source_type: str = ""
    sender: str | None = None
    published_at: str | None = None
    document_title: str = ""
    source_url: str | None = None
    relation: str | None = None
    related_post_id: str | None = None


@dataclass(frozen=True)
class PendingTermDocument:
    document_id: str
    source_type: str
    title: str
    source_url: str | None
    reliability: float
    content_hash: str
    chunks: tuple[TermExtractionChunk, ...]
    backfill: bool = False
    external_id: str = ""
    author: str | None = None
    published_at: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TermCandidateOccurrence:
    normalized_form: str
    surface: str
    document_id: str
    logical_document_id: str
    chunk_id: str
    context: str
    source_type: str
    reliability: float
    strong_name_evidence: bool
    occurrence_count: int = 1


@dataclass(frozen=True)
class ExtractedTerm:
    surface: str
    canonical_zh: str
    aliases: tuple[str, ...]
    reading: str
    relation: str
    evidence_chunk_ids: tuple[str, ...]
    confidence: float
    asr_aliases: tuple[str, ...] = ()
    reviewed_conflict: bool = False
    description_zh: str = ""
    source_urls: tuple[str, ...] = ()
    enrichment_reviewed: bool = False


@dataclass(frozen=True)
class CuratedTermMapping:
    canonical_zh: str
    record_id: str


@dataclass(frozen=True)
class KnownTermMapping:
    surface: str
    canonical_zh: str
    aliases: tuple[str, ...]
    asr_aliases: tuple[str, ...]
    reading: str
    confidence: float
    curated: bool = False


@dataclass(frozen=True)
class KnowledgeQuery:
    text: str
    speaker: str | None = None
    video_date: str | None = None
    ocr_text: str = ""
    chat_text: str = ""
    exclude_video_id: str | None = None
    top_k: int = 8


@dataclass(frozen=True)
class KnowledgeScore:
    keyword: float
    kana: float
    vector: float
    speaker: float
    date: float
    cross_evidence: float
    reliability: float
    total: float
    rrf: float = 0.0
    bm25: float = 0.0
    entity: float = 0.0
    idf: float = 0.0
    lexical_terms: int = 0
    specificity: int = 0
    reranker: float = 0.0


@dataclass(frozen=True)
class KnowledgeHit:
    record_id: str
    kind: str
    title: str
    body: str
    source_url: str | None
    score: KnowledgeScore
    matched_terms: tuple[str, ...]
    retrieval_ranks: dict[str, int] = field(default_factory=dict)

    def prompt_value(self, maximum_body_chars: int = 360) -> dict[str, object]:
        return {
            "id": self.record_id,
            "kind": self.kind,
            "title": self.title,
            "context": self.body[:maximum_body_chars],
            "score": round(self.score.total, 4),
            "matched": list(self.matched_terms),
        }


@dataclass(frozen=True)
class RetrievalWeights:
    keyword: float = 3.0
    kana: float = 2.0
    vector: float = 2.0
    speaker: float = 1.25
    date: float = 0.75
    cross_evidence: float = 1.5
    reliability: float = 0.75
    rrf: float = 2.0


@dataclass
class _Candidate:
    row: sqlite3.Row
    ranks: dict[str, int]
    raw_bm25: float | None = None


@dataclass(frozen=True)
class _NormalizedQuery:
    text: str
    reading: str
    fts_terms: tuple[str, ...]


class FanKnowledgeRetriever:
    """Small, local and explainable retriever for fan-domain context."""

    def __init__(
        self,
        database_path: Path,
        *,
        audit_path: Path | None = None,
        weights: RetrievalWeights | None = None,
        embedding_model: str | None = None,
        vector_index_path: Path | None = None,
        vector_minimum_score: float = 0.62,
        reranker_model: str | None = None,
        reranker_minimum_score: float = 0.05,
    ) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._database_path = database_path
        self._audit_path = audit_path
        self._weights = weights or RetrievalWeights()
        self._vector_minimum_score = vector_minimum_score
        self._reranker_minimum_score = reranker_minimum_score
        self._vector_index = (
            LocalVectorIndex(vector_index_path, embedding_model)
            if embedding_model and vector_index_path is not None
            else None
        )
        self._reranker = (
            LocalCrossEncoderReranker(reranker_model) if reranker_model else None
        )
        self._lock = threading.Lock()
        self._database = sqlite3.connect(database_path, check_same_thread=False)
        self._database.row_factory = sqlite3.Row
        self._initialize()

    def close(self) -> None:
        self.release_models()
        self._database.close()

    def release_models(self) -> int:
        released = sum(
            (
                int(self._vector_index.release_model())
                if self._vector_index is not None
                else 0,
                int(self._reranker.release_model())
                if self._reranker is not None
                else 0,
            )
        )
        if not released:
            return 0

        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        _LOGGER.info("released %d fan-knowledge retrieval model(s)", released)
        return released

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def upsert(self, records: Iterable[KnowledgeRecord]) -> int:
        count = 0
        with self._lock, self._database:
            for record in records:
                _validate_record(record)
                aliases = json.dumps(record.aliases, ensure_ascii=False)
                keywords = json.dumps(record.keywords, ensure_ascii=False)
                search_text = _fts_search_text(
                    (
                        record.title,
                        record.body,
                        record.reading,
                        *record.aliases,
                        *record.keywords,
                    )
                )
                self._database.execute(
                    """
                    INSERT INTO knowledge_records (
                        record_id, kind, title, body, aliases, reading, keywords,
                        speaker, valid_from, valid_to, source_url, source_type,
                        reliability, search_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(record_id) DO UPDATE SET
                        kind=excluded.kind, title=excluded.title, body=excluded.body,
                        aliases=excluded.aliases, reading=excluded.reading,
                        keywords=excluded.keywords, speaker=excluded.speaker,
                        valid_from=excluded.valid_from, valid_to=excluded.valid_to,
                        source_url=excluded.source_url,
                        source_type=excluded.source_type,
                        reliability=excluded.reliability,
                        search_text=excluded.search_text
                    """,
                    (
                        record.record_id,
                        record.kind,
                        record.title,
                        record.body,
                        aliases,
                        record.reading,
                        keywords,
                        record.speaker,
                        record.valid_from,
                        record.valid_to,
                        record.source_url,
                        record.source_type,
                        record.reliability,
                        search_text,
                    ),
                )
                count += 1
            self._database.execute(
                "INSERT INTO knowledge_fts(knowledge_fts) VALUES('rebuild')"
            )
        return count

    def ingest_translation_context(self, context: dict[str, object]) -> int:
        return self.upsert(records_from_translation_context(context))

    def upsert_document(
        self,
        document: KnowledgeDocument,
        chunks: Iterable[KnowledgeChunk],
    ) -> DocumentUpsertResult:
        _validate_document(document)
        chunk_values = [chunk for chunk in chunks if chunk.text.strip()]
        content_hash = _document_hash(document, chunk_values)
        metadata_json = json.dumps(
            document.metadata or {}, ensure_ascii=False, sort_keys=True
        )
        with self._lock, self._database:
            existing = self._database.execute(
                "SELECT content_hash FROM knowledge_documents WHERE document_id = ?",
                (document.document_id,),
            ).fetchone()
            if existing is not None and existing["content_hash"] == content_hash:
                return DocumentUpsertResult(
                    document.document_id, False, len(chunk_values)
                )
            self._database.execute(
                """
                INSERT INTO knowledge_documents (
                    document_id, source_type, external_id, source_url, title,
                    author, published_at, fetched_at, language, raw_text,
                    metadata_json, reliability, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    source_type=excluded.source_type,
                    external_id=excluded.external_id,
                    source_url=excluded.source_url,
                    title=excluded.title,
                    author=excluded.author,
                    published_at=excluded.published_at,
                    fetched_at=excluded.fetched_at,
                    language=excluded.language,
                    raw_text=excluded.raw_text,
                    metadata_json=excluded.metadata_json,
                    reliability=excluded.reliability,
                    content_hash=excluded.content_hash
                """,
                (
                    document.document_id,
                    document.source_type,
                    document.external_id,
                    document.source_url,
                    document.title,
                    document.author,
                    document.published_at,
                    document.fetched_at,
                    document.language,
                    document.text,
                    metadata_json,
                    document.reliability,
                    content_hash,
                ),
            )
            self._database.execute(
                "DELETE FROM knowledge_document_chunks WHERE document_id = ?",
                (document.document_id,),
            )
            for chunk in chunk_values:
                chunk_id = _chunk_id(document.document_id, chunk)
                self._database.execute(
                    """
                    INSERT INTO knowledge_document_chunks (
                        chunk_id, document_id, ordinal, start_seconds,
                        end_seconds, speaker, language, text, search_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        document.document_id,
                        chunk.ordinal,
                        chunk.start_seconds,
                        chunk.end_seconds,
                        chunk.speaker,
                        chunk.language or document.language,
                        chunk.text.strip(),
                        _fts_search_text((document.title, chunk.text.strip())),
                    ),
                )
            self._database.execute(
                """
                INSERT INTO knowledge_term_extraction_queue (
                    document_id, content_hash, queued_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    content_hash=excluded.content_hash,
                    queued_at=excluded.queued_at
                """,
                (document.document_id, content_hash, _utc_now()),
            )
        return DocumentUpsertResult(document.document_id, True, len(chunk_values))

    def queue_all_documents_for_term_extraction(self) -> int:
        with self._lock, self._database:
            before = self._database.total_changes
            self._database.execute(
                """
                INSERT INTO knowledge_term_backfill_queue (
                    document_id, content_hash, queued_at
                )
                SELECT document_id, content_hash, ? FROM knowledge_documents
                WHERE TRUE
                ON CONFLICT(document_id) DO UPDATE SET
                    content_hash=excluded.content_hash,
                    queued_at=excluded.queued_at
                """,
                (_utc_now(),),
            )
            return self._database.total_changes - before

    def pending_term_documents(
        self,
        maximum_documents: int | None = None,
        *,
        include_backfill: bool = False,
    ) -> list[PendingTermDocument]:
        queue_table = (
            "knowledge_term_backfill_queue"
            if include_backfill
            else "knowledge_term_extraction_queue"
        )
        limit = "" if maximum_documents is None else "LIMIT ?"
        parameters: tuple[object, ...] = (
            () if maximum_documents is None else (maximum_documents,)
        )
        rows = self._database.execute(
            f"""
            SELECT documents.document_id, documents.source_type,
                   documents.external_id, documents.title, documents.source_url,
                   documents.author, documents.published_at,
                   documents.metadata_json, documents.reliability,
                   documents.content_hash
            FROM {queue_table} AS queue
            JOIN knowledge_documents AS documents
              ON documents.document_id = queue.document_id
            WHERE queue.content_hash = documents.content_hash
            ORDER BY queue.queued_at, documents.document_id
            {limit}
            """,
            parameters,
        ).fetchall()
        return self._pending_term_documents_from_rows(rows, backfill=include_backfill)

    def _pending_term_documents_from_rows(
        self, rows: Iterable[sqlite3.Row], *, backfill: bool
    ) -> list[PendingTermDocument]:
        pending: list[PendingTermDocument] = []
        for row in rows:
            chunk_rows = self._database.execute(
                """
                SELECT chunk_id, text, start_seconds, end_seconds,
                       speaker, language
                FROM knowledge_document_chunks
                WHERE document_id = ?
                ORDER BY ordinal
                """,
                (row["document_id"],),
            ).fetchall()
            pending.append(
                PendingTermDocument(
                    document_id=str(row["document_id"]),
                    source_type=str(row["source_type"]),
                    title=str(row["title"]),
                    source_url=(
                        str(row["source_url"])
                        if row["source_url"] is not None
                        else None
                    ),
                    reliability=float(row["reliability"]),
                    content_hash=str(row["content_hash"]),
                    chunks=tuple(
                        TermExtractionChunk(
                            str(chunk["chunk_id"]),
                            str(chunk["text"]),
                            (
                                float(chunk["start_seconds"])
                                if chunk["start_seconds"] is not None
                                else None
                            ),
                            (
                                float(chunk["end_seconds"])
                                if chunk["end_seconds"] is not None
                                else None
                            ),
                            (
                                str(chunk["speaker"])
                                if chunk["speaker"] is not None
                                else None
                            ),
                            (
                                str(chunk["language"])
                                if chunk["language"] is not None
                                else None
                            ),
                            str(row["document_id"]),
                            str(row["source_type"]),
                            (str(row["author"]) if row["author"] is not None else None),
                            (
                                str(row["published_at"])
                                if row["published_at"] is not None
                                else None
                            ),
                            str(row["title"]),
                            (
                                str(row["source_url"])
                                if row["source_url"] is not None
                                else None
                            ),
                        )
                        for chunk in chunk_rows
                    ),
                    backfill=backfill,
                    external_id=str(row["external_id"]),
                    author=(str(row["author"]) if row["author"] is not None else None),
                    published_at=(
                        str(row["published_at"])
                        if row["published_at"] is not None
                        else None
                    ),
                    metadata=json.loads(str(row["metadata_json"])),
                )
            )
        return pending

    def replace_term_candidate_occurrences(
        self,
        document_ids: Iterable[str],
        occurrences: Iterable[TermCandidateOccurrence],
    ) -> None:
        ids = tuple(dict.fromkeys(value for value in document_ids if value))
        values = tuple(occurrences)
        allowed_ids = set(ids)
        if any(value.document_id not in allowed_ids for value in values):
            raise ValueError("term candidate occurrence belongs to an unknown document")
        with self._lock, self._database:
            for start in range(0, len(ids), 500):
                batch = ids[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                self._database.execute(
                    "DELETE FROM knowledge_term_candidate_occurrences "
                    f"WHERE document_id IN ({placeholders})",
                    batch,
                )

            self._database.executemany(
                """
                INSERT INTO knowledge_term_candidate_occurrences (
                    normalized_form, surface, document_id, logical_document_id,
                    chunk_id, context, source_type, reliability,
                    strong_name_evidence, occurrence_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(normalized_form, chunk_id, context) DO UPDATE SET
                    occurrence_count=(
                        knowledge_term_candidate_occurrences.occurrence_count
                        + excluded.occurrence_count
                    ),
                    strong_name_evidence=MAX(
                        knowledge_term_candidate_occurrences.strong_name_evidence,
                        excluded.strong_name_evidence
                    )
                """,
                [
                    (
                        value.normalized_form,
                        value.surface,
                        value.document_id,
                        value.logical_document_id,
                        value.chunk_id,
                        value.context,
                        value.source_type,
                        value.reliability,
                        int(value.strong_name_evidence),
                        value.occurrence_count,
                    )
                    for value in values
                ],
            )

    def term_candidate_forms_for_documents(
        self, document_ids: Iterable[str]
    ) -> set[str]:
        ids = tuple(dict.fromkeys(value for value in document_ids if value))
        forms: set[str] = set()
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = self._database.execute(
                f"SELECT DISTINCT normalized_form "
                f"FROM knowledge_term_candidate_occurrences "
                f"WHERE document_id IN ({placeholders})",
                batch,
            ).fetchall()
            forms.update(str(row["normalized_form"]) for row in rows)
        return forms

    def term_candidate_occurrences(
        self, normalized_forms: Iterable[str]
    ) -> list[TermCandidateOccurrence]:
        forms = tuple(dict.fromkeys(value for value in normalized_forms if value))
        if not forms:
            return []
        rows: list[sqlite3.Row] = []
        for start in range(0, len(forms), 500):
            batch = forms[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows.extend(
                self._database.execute(
                    f"""
                    SELECT normalized_form, surface, document_id,
                           logical_document_id, chunk_id, context, source_type,
                           reliability, strong_name_evidence, occurrence_count
                    FROM knowledge_term_candidate_occurrences
                    WHERE normalized_form IN ({placeholders})
                    ORDER BY normalized_form, document_id, chunk_id
                    """,
                    batch,
                ).fetchall()
            )
        return [
            TermCandidateOccurrence(
                normalized_form=str(row["normalized_form"]),
                surface=str(row["surface"]),
                document_id=str(row["document_id"]),
                logical_document_id=str(row["logical_document_id"]),
                chunk_id=str(row["chunk_id"]),
                context=str(row["context"]),
                source_type=str(row["source_type"]),
                reliability=float(row["reliability"]),
                strong_name_evidence=bool(row["strong_name_evidence"]),
                occurrence_count=int(row["occurrence_count"]),
            )
            for row in rows
        ]

    def finish_term_preparation(
        self,
        documents: Iterable[PendingTermDocument],
        *,
        touched_forms: Iterable[str],
        pending_review_forms: Iterable[str],
        backfill: bool,
    ) -> None:
        document_values = tuple(documents)
        touched = tuple(dict.fromkeys(value for value in touched_forms if value))
        pending = tuple(dict.fromkeys(value for value in pending_review_forms if value))
        with self._lock, self._database:
            for document in document_values:
                for queue_table in (
                    "knowledge_term_extraction_queue",
                    "knowledge_term_backfill_queue",
                ):
                    self._database.execute(
                        f"DELETE FROM {queue_table} "
                        "WHERE document_id = ? AND content_hash = ?",
                        (document.document_id, document.content_hash),
                    )
            for start in range(0, len(touched), 500):
                batch = touched[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                self._database.execute(
                    f"DELETE FROM knowledge_term_review_queue "
                    f"WHERE normalized_form IN ({placeholders})",
                    batch,
                )
            self._database.executemany(
                """
                INSERT INTO knowledge_term_review_queue (
                    normalized_form, queued_at, backfill
                ) VALUES (?, ?, ?)
                ON CONFLICT(normalized_form) DO UPDATE SET
                    queued_at=excluded.queued_at,
                    backfill=MIN(knowledge_term_review_queue.backfill,
                                 excluded.backfill)
                """,
                [(value, _utc_now(), int(backfill)) for value in pending],
            )

    def pending_term_review_forms(
        self,
        maximum_candidates: int | None = None,
        *,
        include_backfill: bool = False,
    ) -> list[str]:
        limit = "" if maximum_candidates is None else "LIMIT ?"
        parameters: tuple[object, ...] = (
            () if maximum_candidates is None else (maximum_candidates,)
        )
        rows = self._database.execute(
            f"""
            WITH evidence AS (
                SELECT
                    queue.normalized_form,
                    queue.queued_at,
                    MAX(candidate.strong_name_evidence) AS strong_name,
                    MAX(CASE candidate.source_type
                        WHEN 'official_news' THEN 4
                        WHEN 'official_event' THEN 4
                        WHEN 'official_music_notice' THEN 4
                        WHEN 'x_post' THEN 3
                        WHEN 'instagram_post' THEN 3
                        WHEN 'youtube_manual_subtitle' THEN 2
                        WHEN 'youtube_metadata' THEN 2
                        ELSE 0
                    END) AS written_source_tier,
                    COUNT(DISTINCT candidate.source_type) AS source_count,
                    COUNT(DISTINCT candidate.logical_document_id) AS document_count,
                    SUM(candidate.occurrence_count) AS occurrence_count,
                    MAX(candidate.reliability) AS reliability
                FROM knowledge_term_review_queue AS queue
                JOIN knowledge_term_candidate_occurrences AS candidate
                  ON candidate.normalized_form = queue.normalized_form
                WHERE queue.backfill = 0 OR ?
                GROUP BY queue.normalized_form, queue.queued_at
            )
            SELECT normalized_form
            FROM evidence
            ORDER BY
                (strong_name AND written_source_tier > 0) DESC,
                written_source_tier DESC,
                strong_name DESC,
                source_count DESC,
                document_count DESC,
                occurrence_count DESC,
                reliability DESC,
                queued_at,
                normalized_form
            {limit}
            """,
            (int(include_backfill), *parameters),
        ).fetchall()
        return [str(row["normalized_form"]) for row in rows]

    def complete_term_reviews(self, normalized_forms: Iterable[str]) -> None:
        forms = tuple(dict.fromkeys(value for value in normalized_forms if value))
        if not forms:
            return
        with self._lock, self._database:
            for start in range(0, len(forms), 500):
                batch = forms[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                self._database.execute(
                    f"DELETE FROM knowledge_term_review_queue "
                    f"WHERE normalized_form IN ({placeholders})",
                    batch,
                )

    def term_documents(self, document_ids: Iterable[str]) -> list[PendingTermDocument]:
        ids = tuple(dict.fromkeys(value for value in document_ids if value))
        if not ids:
            return []
        rows: list[sqlite3.Row] = []
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows.extend(
                self._database.execute(
                    f"""
                    SELECT document_id, source_type, external_id, title, source_url,
                           author, published_at, metadata_json, reliability,
                           content_hash
                    FROM knowledge_documents
                    WHERE document_id IN ({placeholders})
                    ORDER BY document_id
                    """,
                    batch,
                ).fetchall()
            )
        return self._pending_term_documents_from_rows(rows, backfill=False)

    def store_extracted_terms(
        self,
        document: PendingTermDocument,
        terms: Iterable[ExtractedTerm],
    ) -> tuple[int, int]:
        valid_chunk_ids = {chunk.chunk_id for chunk in document.chunks}
        stored = 0
        published = 0
        with self._lock, self._database:
            for term in terms:
                evidence_ids = tuple(
                    value
                    for value in term.evidence_chunk_ids
                    if value in valid_chunk_ids
                )
                if not evidence_ids:
                    continue
                term_id = self._resolve_extracted_term_id(
                    term.surface, (*term.aliases, *term.asr_aliases)
                )
                term_id = term_id or _extracted_term_id(term.surface)
                curated = self._curated_term_mapping(term.surface, term.aliases)
                existing = self._database.execute(
                    "SELECT * FROM knowledge_extracted_terms WHERE term_id = ?",
                    (term_id,),
                ).fetchone()
                stored_surface = (
                    str(existing["surface"])
                    if existing is not None
                    else term.surface.strip()
                )
                new_aliases = tuple(
                    dict.fromkeys(
                        value.strip()
                        for value in (term.surface, *term.aliases)
                        if value.strip() and value.strip() != stored_surface
                    )
                )
                old_aliases = (
                    tuple(json.loads(str(existing["aliases_json"])))
                    if existing is not None
                    else ()
                )
                aliases = tuple(dict.fromkeys((*old_aliases, *new_aliases)))
                old_asr_aliases = (
                    tuple(json.loads(str(existing["asr_aliases_json"])))
                    if existing is not None
                    else ()
                )
                asr_aliases = tuple(
                    dict.fromkeys(
                        value.strip()
                        for value in (*old_asr_aliases, *term.asr_aliases)
                        if value.strip()
                        and value.strip() != stored_surface
                        and value.strip() not in aliases
                    )
                )
                status = "verified" if curated is not None else "active"
                preserve_existing = existing is not None and (
                    str(existing["status"]) == "verified"
                    or (
                        str(existing["status"]) == "active"
                        and not term.reviewed_conflict
                        and str(existing["canonical_zh"]) != term.canonical_zh.strip()
                    )
                )
                canonical_zh = (
                    curated.canonical_zh
                    if curated is not None
                    else (
                        str(existing["canonical_zh"])
                        if preserve_existing
                        else term.canonical_zh.strip()
                    )
                )
                stored_status = str(existing["status"]) if preserve_existing else status
                confidence = (
                    float(existing["confidence"])
                    if preserve_existing
                    else term.confidence
                )
                self._database.execute(
                    """
                    INSERT INTO knowledge_extracted_terms (
                        term_id, surface, canonical_zh, aliases_json,
                        asr_aliases_json,
                        reading, status, confidence, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(term_id) DO UPDATE SET
                        canonical_zh=excluded.canonical_zh,
                        aliases_json=excluded.aliases_json,
                        asr_aliases_json=excluded.asr_aliases_json,
                        reading=excluded.reading,
                        status=excluded.status,
                        confidence=MAX(knowledge_extracted_terms.confidence,
                                       excluded.confidence),
                        updated_at=excluded.updated_at
                    """,
                    (
                        term_id,
                        stored_surface,
                        canonical_zh,
                        json.dumps(aliases, ensure_ascii=False),
                        json.dumps(asr_aliases, ensure_ascii=False),
                        term.reading.strip(),
                        stored_status,
                        confidence,
                        _utc_now(),
                    ),
                )
                self._replace_term_forms(
                    term_id, stored_surface, (*aliases, *asr_aliases)
                )
                if term.enrichment_reviewed:
                    self._database.execute(
                        """
                        INSERT INTO knowledge_extracted_term_details (
                            term_id, description_zh, source_urls_json, updated_at
                        ) VALUES (?, ?, ?, ?)
                        ON CONFLICT(term_id) DO UPDATE SET
                            description_zh=excluded.description_zh,
                            source_urls_json=excluded.source_urls_json,
                            updated_at=excluded.updated_at
                        """,
                        (
                            term_id,
                            term.description_zh.strip(),
                            json.dumps(term.source_urls, ensure_ascii=False),
                            _utc_now(),
                        ),
                    )
                for chunk_id in evidence_ids:
                    chunk = next(
                        value for value in document.chunks if value.chunk_id == chunk_id
                    )
                    self._database.execute(
                        """
                        INSERT OR REPLACE INTO knowledge_term_evidence (
                            term_id, chunk_id, document_id, relation,
                            canonical_zh, evidence_text, source_type, confidence
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            term_id,
                            chunk_id,
                            document.document_id,
                            term.relation,
                            term.canonical_zh.strip(),
                            chunk.text,
                            document.source_type,
                            term.confidence,
                        ),
                    )
                stored += 1
                if stored_status in {"active", "verified"} and curated is None:
                    self._publish_extracted_term(term_id)
                    published += 1
            if published:
                self._database.execute(
                    "INSERT INTO knowledge_fts(knowledge_fts) VALUES('rebuild')"
                )
        return stored, published

    def known_term_mapping(
        self, surface: str, aliases: Iterable[str]
    ) -> KnownTermMapping | None:
        alias_values = tuple(value.strip() for value in aliases if value.strip())
        with self._lock:
            curated = self._curated_term_mapping(surface, alias_values)
            if curated is not None:
                return KnownTermMapping(
                    surface.strip(),
                    curated.canonical_zh,
                    alias_values,
                    (),
                    "",
                    1.0,
                    True,
                )
            term_id = self._resolve_extracted_term_id(surface, alias_values)
            if term_id is None:
                return None
            row = self._database.execute(
                "SELECT * FROM knowledge_extracted_terms WHERE term_id = ?", (term_id,)
            ).fetchone()
            if row is None or str(row["status"]) not in {"active", "verified"}:
                return None
            return KnownTermMapping(
                str(row["surface"]),
                str(row["canonical_zh"]),
                tuple(json.loads(str(row["aliases_json"]))),
                tuple(json.loads(str(row["asr_aliases_json"]))),
                str(row["reading"]),
                float(row["confidence"]),
            )

    def _resolve_extracted_term_id(
        self, surface: str, aliases: Iterable[str]
    ) -> str | None:
        normalized_forms = tuple(
            dict.fromkeys(
                _normalize_term_form(value)
                for value in (surface, *aliases)
                if value.strip()
            )
        )
        if not normalized_forms:
            return None
        placeholders = ",".join("?" for _ in normalized_forms)
        rows = self._database.execute(
            f"SELECT DISTINCT forms.term_id "
            f"FROM knowledge_term_forms AS forms "
            f"JOIN knowledge_extracted_terms AS terms "
            f"ON terms.term_id = forms.term_id "
            f"WHERE forms.normalized_form IN ({placeholders})",
            normalized_forms,
        ).fetchall()
        term_ids = {str(row["term_id"]) for row in rows}
        if len(term_ids) == 1:
            return next(iter(term_ids))
        surface_id = _extracted_term_id(surface)
        return surface_id if surface_id in term_ids else None

    def _replace_term_forms(
        self, term_id: str, surface: str, aliases: Iterable[str]
    ) -> None:
        forms: dict[str, str] = {}
        for value in (surface, *aliases):
            form = value.strip()
            normalized = _normalize_term_form(form)
            if normalized:
                forms.setdefault(normalized, form)
        self._database.execute(
            "DELETE FROM knowledge_term_forms WHERE term_id = ?", (term_id,)
        )
        self._database.executemany(
            "INSERT INTO knowledge_term_forms(term_id, normalized_form, form) "
            "VALUES(?, ?, ?)",
            [(term_id, normalized, form) for normalized, form in forms.items()],
        )

    def reconcile_extracted_terms_with_curated(self) -> int:
        rows = self._database.execute(
            "SELECT term_id, surface, aliases_json, asr_aliases_json "
            "FROM knowledge_extracted_terms"
        ).fetchall()
        updates: list[tuple[str, str]] = []
        for row in rows:
            aliases = (
                *json.loads(str(row["aliases_json"])),
                *json.loads(str(row["asr_aliases_json"])),
            )
            curated = self._curated_term_mapping(str(row["surface"]), aliases)
            if curated is not None:
                updates.append((curated.canonical_zh, str(row["term_id"])))
        with self._lock, self._database:
            self._database.executemany(
                "UPDATE knowledge_extracted_terms SET canonical_zh = ?, "
                "status = 'verified', updated_at = ? WHERE term_id = ?",
                [
                    (canonical_zh, _utc_now(), term_id)
                    for canonical_zh, term_id in updates
                ],
            )
            self._database.executemany(
                "DELETE FROM knowledge_records WHERE record_id = ?",
                [(f"extracted:{term_id}",) for _target, term_id in updates],
            )
            if updates:
                self._database.execute(
                    "INSERT INTO knowledge_fts(knowledge_fts) VALUES('rebuild')"
                )
            self._sync_term_forms()
        return len(updates)

    def _sync_term_forms(self) -> None:
        rows = self._database.execute(
            "SELECT term_id, surface, aliases_json, asr_aliases_json "
            "FROM knowledge_extracted_terms"
        ).fetchall()
        for row in rows:
            self._replace_term_forms(
                str(row["term_id"]),
                str(row["surface"]),
                (
                    *json.loads(str(row["aliases_json"])),
                    *json.loads(str(row["asr_aliases_json"])),
                ),
            )

    def _curated_term_mapping(
        self, surface: str, aliases: Iterable[str]
    ) -> CuratedTermMapping | None:
        forms = {
            surface.strip(),
            *(value.strip() for value in aliases if value.strip()),
        }
        rows = self._database.execute(
            """
            SELECT record_id, title, body, aliases
            FROM knowledge_records
            WHERE source_type = 'curated_glossary'
              AND kind IN ('term', 'character')
            ORDER BY reliability DESC, record_id
            """
        ).fetchall()
        for row in rows:
            record_forms = {str(row["title"]), *json.loads(str(row["aliases"]))}
            if not forms & record_forms:
                continue
            match = _CURATED_TRANSLATION_RE.search(str(row["body"]))
            if match is None:
                continue
            return CuratedTermMapping(
                canonical_zh=match.group(1).strip(),
                record_id=str(row["record_id"]),
            )
        return None

    def _publish_extracted_term(self, term_id: str) -> None:
        row = self._database.execute(
            """
            SELECT terms.*, details.description_zh, details.source_urls_json
            FROM knowledge_extracted_terms AS terms
            LEFT JOIN knowledge_extracted_term_details AS details
              ON details.term_id = terms.term_id
            WHERE terms.term_id = ?
            """,
            (term_id,),
        ).fetchone()
        if row is None:
            return
        aliases = tuple(json.loads(str(row["aliases_json"])))
        asr_aliases = tuple(json.loads(str(row["asr_aliases_json"])))
        body = f"{row['surface']}的参考中文译名为{row['canonical_zh']}。"
        if asr_aliases:
            body += (
                f"{'、'.join(asr_aliases)}是自动语音识别中出现过的误听形式，"
                "仅用于检索和纠错，不是正式别名。"
            )
        description = str(row["description_zh"] or "").strip()
        if description:
            body += f"{description}"
        source_urls = tuple(json.loads(str(row["source_urls_json"] or "[]")))
        source_url = source_urls[0] if source_urls else None
        search_text = _fts_search_text(
            (
                str(row["surface"]),
                body,
                str(row["reading"]),
                *aliases,
                *asr_aliases,
            )
        )
        self._database.execute(
            """
            INSERT INTO knowledge_records (
                record_id, kind, title, body, aliases, reading, keywords,
                speaker, valid_from, valid_to, source_url, source_type,
                reliability, search_text
            ) VALUES (?, 'term', ?, ?, ?, ?, '[]', NULL, NULL, NULL,
                      ?, 'auto_extracted', ?, ?)
            ON CONFLICT(record_id) DO UPDATE SET
                kind=excluded.kind, title=excluded.title, body=excluded.body,
                aliases=excluded.aliases, reading=excluded.reading,
                source_type=excluded.source_type,
                source_url=excluded.source_url,
                reliability=excluded.reliability, search_text=excluded.search_text
            """,
            (
                f"extracted:{term_id}",
                row["surface"],
                body,
                json.dumps((*aliases, *asr_aliases), ensure_ascii=False),
                row["reading"],
                source_url,
                row["confidence"],
                search_text,
            ),
        )

    def document_count(self) -> int:
        row = self._database.execute(
            "SELECT COUNT(*) AS value FROM knowledge_documents"
        ).fetchone()
        return int(row["value"])

    def chunk_count(self) -> int:
        row = self._database.execute(
            "SELECT COUNT(*) AS value FROM knowledge_document_chunks"
        ).fetchone()
        return int(row["value"])

    def metadata(self, key: str) -> str | None:
        row = self._database.execute(
            "SELECT value FROM knowledge_meta WHERE key = ?", (key,)
        ).fetchone()
        return str(row["value"]) if row is not None else None

    def set_metadata(self, key: str, value: str) -> None:
        with self._lock, self._database:
            self._database.execute(
                "INSERT OR REPLACE INTO knowledge_meta(key, value) VALUES(?, ?)",
                (key, value),
            )

    def external_ids(self, source_types: tuple[str, ...]) -> set[str]:
        if not source_types:
            return set()
        placeholders = ",".join("?" for _ in source_types)
        rows = self._database.execute(
            f"SELECT external_id FROM knowledge_documents "
            f"WHERE source_type IN ({placeholders})",
            source_types,
        ).fetchall()
        return {str(row["external_id"]) for row in rows}

    def source_urls(self, source_types: tuple[str, ...]) -> set[str]:
        if not source_types:
            return set()
        placeholders = ",".join("?" for _ in source_types)
        rows = self._database.execute(
            f"SELECT source_url FROM knowledge_documents "
            f"WHERE source_type IN ({placeholders}) AND source_url IS NOT NULL",
            source_types,
        ).fetchall()
        return {str(row["source_url"]) for row in rows}

    def retrieve(self, query: KnowledgeQuery) -> list[KnowledgeHit]:
        return self._retrieve(query, include_term_records=True)

    def retrieve_background(self, query: KnowledgeQuery) -> list[KnowledgeHit]:
        return self._retrieve(query, include_term_records=False)

    def _retrieve(
        self, query: KnowledgeQuery, *, include_term_records: bool
    ) -> list[KnowledgeHit]:
        return self._retrieve_many([query], include_term_records=include_term_records)[0]

    def retrieve_background_many(self, queries: list[KnowledgeQuery]) -> list[list[KnowledgeHit]]:
        return self._retrieve_many(queries, include_term_records=False)

    def _retrieve_many(
        self, queries: list[KnowledgeQuery], *, include_term_records: bool
    ) -> list[list[KnowledgeHit]]:
        results: list[list[KnowledgeHit]] = [[] for _ in queries]
        active = [(i, query) for i, query in enumerate(queries) if query.top_k > 0 and query.text.strip()]
        if not active:
            return results
        _LOGGER.info("Knowledge batch retrieval: encoding %d queries", len(active))
        normalized = [_normalize_query(query.text) for _, query in active]
        vector_rows = (
            self._vector_index.search_many([value.text for value in normalized], _VECTOR_CANDIDATES)
            if self._vector_index is not None else [{} for _ in active]
        )
        _LOGGER.info("Knowledge batch retrieval: selecting candidates")
        prepared = [
            self._retrieval_candidates(query, norm, vectors, include_term_records=include_term_records)
            for (_, query), norm, vectors in zip(active, normalized, vector_rows, strict=True)
        ]
        _LOGGER.info("Knowledge batch retrieval: reranking %d pairs", sum(len(hits) for hits, _ in prepared))
        reranked = (
            self._reranker.score_many([
                (norm.text, [hit.body for hit in hits])
                for norm, (hits, _) in zip(normalized, prepared, strict=True)
            ])
            if self._reranker is not None else [None for _ in active]
        )
        for (position, query), (hits, count), scores in zip(active, prepared, reranked, strict=True):
            results[position] = self._finish_retrieval(
                query, hits, count, scores, include_term_records=include_term_records
            )
        return results

    def _retrieval_candidates(self, query, normalized, vector_scores, *, include_term_records):
        with self._lock:
            candidates = self._candidates(
                query,
                normalized,
                vector_scores,
                include_term_records=include_term_records,
            )
            idf = self._idf_scores(normalized.fts_terms)
        fused = sorted(
            candidates.values(),
            key=lambda candidate: (
                -_rrf_score(candidate.ranks),
                _candidate_id(candidate),
            ),
        )[:_FUSION_CANDIDATES]
        hits = [
            self._score(candidate, query, normalized, vector_scores, idf)
            for candidate in fused
        ]
        return hits, len(candidates)

    def _finish_retrieval(self, query, hits, candidate_count, reranker_scores, *, include_term_records):
        if reranker_scores is not None:
            hits = [
                replace(
                    hit,
                    score=replace(hit.score, reranker=reranker_score),
                )
                for hit, reranker_score in zip(hits, reranker_scores, strict=True)
            ]
        accepted = [
            hit
            for hit in hits
            if _passes_relevance_gate(hit)
            and (
                self._reranker is None
                or hit.score.reranker >= self._reranker_minimum_score
            )
        ]
        rejected = [hit for hit in hits if hit not in accepted]
        hits = accepted
        hits.sort(
            key=lambda hit: (
                -hit.score.reranker if self._reranker is not None else 0.0,
                -hit.score.total,
                hit.record_id,
            )
        )
        selected = _diversify_hits(hits, query.top_k)
        self._audit(
            query,
            selected,
            rejected,
            candidate_count,
            event=(
                "fan_knowledge_retrieval"
                if include_term_records
                else "fan_background_knowledge_retrieval"
            ),
        )
        return selected

    def retrieve_term_references(self, query: KnowledgeQuery) -> list[KnowledgeHit]:
        """Return deterministic terminology matches outside background ranking."""
        normalized = _normalize_query(query.text)
        with self._lock:
            rows = self._database.execute(
                "SELECT *, NULL AS fts_rank FROM knowledge_records"
            ).fetchall()
            matched_rows: list[tuple[int, int, sqlite3.Row, tuple[str, ...]]] = []
            for row in rows:
                if str(row["kind"]) not in _TERM_REFERENCE_KINDS:
                    continue
                matched = _exact_entity_matches(normalized, _row_forms(row))
                speaker_match = bool(
                    query.speaker
                    and row["speaker"]
                    and query.speaker == str(row["speaker"])
                )
                if not matched and not speaker_match:
                    continue
                matched_rows.append(
                    (
                        1 if speaker_match else 0,
                        max((len(_compact(value)) for value in matched), default=0),
                        row,
                        matched,
                    )
                )

            matched_rows.sort(
                key=lambda value: (
                    -value[0],
                    -value[1],
                    str(value[2]["record_id"]),
                )
            )
            hits: list[KnowledgeHit] = []
            for rank, (_speaker, _length, row, matched) in enumerate(
                matched_rows[: query.top_k], 1
            ):
                reliability = float(row["reliability"])
                hits.append(
                    KnowledgeHit(
                        record_id=str(row["record_id"]),
                        kind=str(row["kind"]),
                        title=str(row["title"]),
                        body=str(row["body"]),
                        source_url=row["source_url"],
                        score=KnowledgeScore(
                            keyword=1.0 if matched else 0.0,
                            kana=0.0,
                            vector=0.0,
                            speaker=1.0 if _speaker else 0.0,
                            date=0.0,
                            cross_evidence=0.0,
                            reliability=reliability,
                            total=(2.0 if matched else 0.0)
                            + (1.0 if _speaker else 0.0)
                            + reliability,
                            entity=1.0 if matched else 0.0,
                            specificity=_length,
                        ),
                        matched_terms=matched,
                        retrieval_ranks={"term_reference": rank},
                    )
                )
        self._audit(
            query,
            hits,
            [],
            len(matched_rows),
            event="fan_term_reference_retrieval",
        )
        return hits

    def retrieve_asr_term_references(
        self, query: KnowledgeQuery
    ) -> list[KnowledgeHit]:
        """Return exact and phonetically close known terms for ASR correction."""
        if query.top_k < 1 or not query.text.strip():
            return []
        normalized = _normalize_query(query.text)
        with self._lock:
            rows = self._database.execute(
                "SELECT *, NULL AS fts_rank FROM knowledge_records"
            ).fetchall()

        candidates: list[
            tuple[bool, float, int, sqlite3.Row, tuple[str, ...]]
        ] = []
        for row in rows:
            if str(row["kind"]) not in _TERM_REFERENCE_KINDS:
                continue
            forms = _row_forms(row)
            exact = _exact_entity_matches(normalized, forms)
            phonetic_forms = _term_phonetic_forms(
                str(row["title"]),
                str(row["reading"] or ""),
                tuple(json.loads(str(row["aliases"]))),
            )
            phonetic = max(
                (
                    _best_window_ratio(normalized.reading, reading)
                    for reading in phonetic_forms
                ),
                default=0.0,
            )
            if not exact and phonetic < 0.78:
                continue
            candidates.append(
                (
                    bool(exact),
                    1.0 if exact else phonetic,
                    max((len(value) for value in phonetic_forms), default=0),
                    row,
                    exact,
                )
            )

        candidates.sort(
            key=lambda value: (
                -int(value[0]),
                -value[1],
                -value[2],
                str(value[3]["record_id"]),
            )
        )
        hits: list[KnowledgeHit] = []
        for rank, (exact, phonetic, _length, row, matched) in enumerate(
            candidates[: query.top_k], 1
        ):
            reliability = float(row["reliability"])
            hits.append(
                KnowledgeHit(
                    record_id=str(row["record_id"]),
                    kind=str(row["kind"]),
                    title=str(row["title"]),
                    body=str(row["body"]),
                    source_url=row["source_url"],
                    score=KnowledgeScore(
                        keyword=1.0 if exact else 0.0,
                        kana=phonetic,
                        vector=0.0,
                        speaker=0.0,
                        date=0.0,
                        cross_evidence=0.0,
                        reliability=reliability,
                        total=(3.0 if exact else 2.0 * phonetic)
                        + 0.75 * reliability,
                        entity=1.0 if exact else 0.0,
                    ),
                    matched_terms=(
                        matched
                        if matched
                        else (f"近音:{row['title']!s}",)
                    ),
                    retrieval_ranks={"entity" if exact else "kana": rank},
                )
            )
        return hits

    def sync_vector_index(self) -> tuple[int, int]:
        if self._vector_index is None:
            return 0, 0
        rows = self._database.execute(
            """
            SELECT chunks.chunk_id, documents.title, chunks.text
            FROM knowledge_document_chunks AS chunks
            JOIN knowledge_documents AS documents
              ON documents.document_id = chunks.document_id
            ORDER BY chunks.chunk_id
            """
        ).fetchall()
        return self._vector_index.sync(
            [(str(row["chunk_id"]), f"{row['title']}\n{row['text']}") for row in rows]
        )

    def _initialize(self) -> None:
        with self._database:
            self._database.executescript(
                """
                CREATE TABLE IF NOT EXISTS knowledge_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS knowledge_records (
                    record_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    aliases TEXT NOT NULL,
                    reading TEXT NOT NULL,
                    keywords TEXT NOT NULL,
                    speaker TEXT,
                    valid_from TEXT,
                    valid_to TEXT,
                    source_url TEXT,
                    source_type TEXT NOT NULL,
                    reliability REAL NOT NULL,
                    search_text TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                    search_text,
                    content='knowledge_records',
                    content_rowid='rowid',
                    tokenize='unicode61 remove_diacritics 2'
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts_vocab
                    USING fts5vocab(knowledge_fts, 'row');
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    document_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    external_id TEXT NOT NULL,
                    source_url TEXT,
                    title TEXT NOT NULL,
                    author TEXT,
                    published_at TEXT,
                    fetched_at TEXT,
                    language TEXT,
                    raw_text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    reliability REAL NOT NULL,
                    content_hash TEXT NOT NULL,
                    UNIQUE(source_type, external_id)
                );
                CREATE TABLE IF NOT EXISTS knowledge_document_chunks (
                    chunk_id TEXT UNIQUE NOT NULL,
                    document_id TEXT NOT NULL REFERENCES knowledge_documents(document_id)
                        ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    start_seconds REAL,
                    end_seconds REAL,
                    speaker TEXT,
                    language TEXT,
                    text TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    UNIQUE(document_id, ordinal)
                );
                CREATE TABLE IF NOT EXISTS knowledge_term_extraction_queue (
                    document_id TEXT PRIMARY KEY
                        REFERENCES knowledge_documents(document_id) ON DELETE CASCADE,
                    content_hash TEXT NOT NULL,
                    queued_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS knowledge_term_backfill_queue (
                    document_id TEXT PRIMARY KEY
                        REFERENCES knowledge_documents(document_id) ON DELETE CASCADE,
                    content_hash TEXT NOT NULL,
                    queued_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS knowledge_term_candidate_occurrences (
                    normalized_form TEXT NOT NULL,
                    surface TEXT NOT NULL,
                    document_id TEXT NOT NULL
                        REFERENCES knowledge_documents(document_id) ON DELETE CASCADE,
                    logical_document_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL
                        REFERENCES knowledge_document_chunks(chunk_id) ON DELETE CASCADE,
                    context TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    reliability REAL NOT NULL,
                    strong_name_evidence INTEGER NOT NULL,
                    occurrence_count INTEGER NOT NULL,
                    PRIMARY KEY(normalized_form, chunk_id, context)
                );
                CREATE INDEX IF NOT EXISTS knowledge_term_candidates_form
                    ON knowledge_term_candidate_occurrences(normalized_form);
                CREATE TABLE IF NOT EXISTS knowledge_term_review_queue (
                    normalized_form TEXT PRIMARY KEY,
                    queued_at TEXT NOT NULL,
                    backfill INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS knowledge_extracted_terms (
                    term_id TEXT PRIMARY KEY,
                    surface TEXT NOT NULL,
                    canonical_zh TEXT NOT NULL,
                    aliases_json TEXT NOT NULL,
                    asr_aliases_json TEXT NOT NULL,
                    reading TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS knowledge_term_forms (
                    term_id TEXT NOT NULL
                        REFERENCES knowledge_extracted_terms(term_id) ON DELETE CASCADE,
                    normalized_form TEXT NOT NULL,
                    form TEXT NOT NULL,
                    PRIMARY KEY(term_id, normalized_form)
                );
                CREATE INDEX IF NOT EXISTS knowledge_term_forms_normalized
                    ON knowledge_term_forms(normalized_form);
                CREATE TABLE IF NOT EXISTS knowledge_extracted_term_details (
                    term_id TEXT PRIMARY KEY
                        REFERENCES knowledge_extracted_terms(term_id) ON DELETE CASCADE,
                    description_zh TEXT NOT NULL,
                    source_urls_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS knowledge_term_evidence (
                    term_id TEXT NOT NULL
                        REFERENCES knowledge_extracted_terms(term_id) ON DELETE CASCADE,
                    chunk_id TEXT NOT NULL
                        REFERENCES knowledge_document_chunks(chunk_id) ON DELETE CASCADE,
                    document_id TEXT NOT NULL
                        REFERENCES knowledge_documents(document_id) ON DELETE CASCADE,
                    relation TEXT NOT NULL,
                    canonical_zh TEXT NOT NULL,
                    evidence_text TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    PRIMARY KEY(term_id, chunk_id)
                );
                CREATE INDEX IF NOT EXISTS knowledge_term_evidence_document
                    ON knowledge_term_evidence(document_id);
                CREATE INDEX IF NOT EXISTS knowledge_chunks_document
                    ON knowledge_document_chunks(document_id, ordinal);
                CREATE INDEX IF NOT EXISTS knowledge_documents_published
                    ON knowledge_documents(published_at);
                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_document_chunks_fts
                    USING fts5(
                        search_text,
                        content='knowledge_document_chunks',
                        content_rowid='rowid',
                        tokenize='unicode61 remove_diacritics 2'
                    );
                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_chunks_fts_vocab
                    USING fts5vocab(knowledge_document_chunks_fts, 'row');
                CREATE TRIGGER IF NOT EXISTS knowledge_chunks_ai AFTER INSERT
                    ON knowledge_document_chunks BEGIN
                    INSERT INTO knowledge_document_chunks_fts(rowid, search_text)
                    VALUES (new.rowid, new.search_text);
                END;
                CREATE TRIGGER IF NOT EXISTS knowledge_chunks_ad AFTER DELETE
                    ON knowledge_document_chunks BEGIN
                    INSERT INTO knowledge_document_chunks_fts(
                        knowledge_document_chunks_fts, rowid, search_text
                    ) VALUES ('delete', old.rowid, old.search_text);
                END;
                CREATE TRIGGER IF NOT EXISTS knowledge_chunks_au AFTER UPDATE
                    ON knowledge_document_chunks BEGIN
                    INSERT INTO knowledge_document_chunks_fts(
                        knowledge_document_chunks_fts, rowid, search_text
                    ) VALUES ('delete', old.rowid, old.search_text);
                    INSERT INTO knowledge_document_chunks_fts(rowid, search_text)
                    VALUES (new.rowid, new.search_text);
                END;
                """
            )
            self._database.execute("PRAGMA foreign_keys = ON")
            self._database.execute(
                "INSERT OR REPLACE INTO knowledge_meta(key, value) "
                "VALUES('schema_version', ?)",
                (str(_SCHEMA_VERSION),),
            )
            self._sync_term_forms()
            provisional = self._database.execute(
                "SELECT term_id FROM knowledge_extracted_terms "
                "WHERE status = 'provisional'"
            ).fetchall()
            if provisional:
                self._database.execute(
                    "UPDATE knowledge_extracted_terms SET status = 'active', "
                    "updated_at = ? WHERE status = 'provisional'",
                    (_utc_now(),),
                )
                for row in provisional:
                    self._publish_extracted_term(str(row["term_id"]))
                self._database.execute(
                    "INSERT INTO knowledge_fts(knowledge_fts) VALUES('rebuild')"
                )

    def _candidates(
        self,
        query: KnowledgeQuery,
        normalized: _NormalizedQuery,
        vector_scores: dict[str, float],
        *,
        include_term_records: bool,
    ) -> dict[str, _Candidate]:
        candidates: dict[str, _Candidate] = {}

        def add(
            row: sqlite3.Row,
            channel: str,
            rank: int,
            raw_bm25: float | None = None,
        ) -> None:
            record_id = str(row["record_id"])
            candidate = candidates.get(record_id)
            if candidate is None:
                candidates[record_id] = _Candidate(row, {channel: rank}, raw_bm25)
                return
            candidate.ranks[channel] = min(rank, candidate.ranks.get(channel, rank))
            if raw_bm25 is not None and (
                candidate.raw_bm25 is None or raw_bm25 < candidate.raw_bm25
            ):
                candidate.raw_bm25 = raw_bm25

        tokens = normalized.fts_terms
        if tokens:
            expression = " OR ".join(
                f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens
            )
            record_filter = (
                ""
                if include_term_records
                else "AND records.kind NOT IN ('term', 'character', 'entity')"
            )
            try:
                matches = self._database.execute(
                    f"""
                    SELECT records.*, bm25(knowledge_fts) AS fts_rank
                    FROM knowledge_fts
                    JOIN knowledge_records AS records
                      ON records.rowid = knowledge_fts.rowid
                    WHERE knowledge_fts MATCH ?
                    {record_filter}
                    ORDER BY fts_rank
                    LIMIT 20
                    """,
                    (expression,),
                ).fetchall()
            except sqlite3.OperationalError:
                matches = []
            try:
                exclusion = ""
                parameters: list[object] = [expression]
                if query.exclude_video_id:
                    exclusion = (
                        "AND (documents.source_type = 'youtube_comment' OR ("
                        "documents.external_id != ? "
                        "AND substr(documents.external_id, 1, length(?) + 1) != ? || ':'"
                        "))"
                    )
                    parameters.extend(
                        [
                            query.exclude_video_id,
                            query.exclude_video_id,
                            query.exclude_video_id,
                        ]
                    )
                chunk_matches = self._database.execute(
                    f"""
                    SELECT
                        'chunk:' || chunks.chunk_id AS record_id,
                        'document_chunk' AS kind,
                        documents.title AS title,
                        chunks.text AS body,
                        '[]' AS aliases,
                        '' AS reading,
                        '[]' AS keywords,
                        chunks.speaker AS speaker,
                        substr(documents.published_at, 1, 10) AS valid_from,
                        NULL AS valid_to,
                        documents.source_url AS source_url,
                        documents.source_type AS source_type,
                        documents.reliability AS reliability,
                        chunks.search_text AS search_text,
                        bm25(knowledge_document_chunks_fts) AS fts_rank
                    FROM knowledge_document_chunks_fts
                    JOIN knowledge_document_chunks AS chunks
                      ON chunks.rowid = knowledge_document_chunks_fts.rowid
                    JOIN knowledge_documents AS documents
                      ON documents.document_id = chunks.document_id
                    WHERE knowledge_document_chunks_fts MATCH ?
                    {exclusion}
                    ORDER BY fts_rank
                    LIMIT 20
                    """,
                    parameters,
                ).fetchall()
            except sqlite3.OperationalError:
                chunk_matches = []
            lexical_rows = _interleave(matches, chunk_matches)[:_LEXICAL_CANDIDATES]
            for rank, row in enumerate(lexical_rows, 1):
                add(row, "fts", rank, float(row["fts_rank"]))

        if vector_scores:
            vector_rank = {
                chunk_id: rank
                for rank, (chunk_id, _score) in enumerate(
                    sorted(vector_scores.items(), key=lambda value: -value[1]), 1
                )
            }
            values = list(vector_rank)
            for offset in range(0, len(values), 500):
                batch = values[offset : offset + 500]
                placeholders = ",".join("?" for _ in batch)
                parameters: list[object] = [*batch]
                exclusion = ""
                if query.exclude_video_id:
                    exclusion = (
                        "AND (documents.source_type = 'youtube_comment' OR ("
                        "documents.external_id != ? "
                        "AND substr(documents.external_id, 1, length(?) + 1) != ? || ':'"
                        "))"
                    )
                    parameters.extend(
                        [
                            query.exclude_video_id,
                            query.exclude_video_id,
                            query.exclude_video_id,
                        ]
                    )
                matches = self._database.execute(
                    f"""
                    SELECT
                        'chunk:' || chunks.chunk_id AS record_id,
                        'document_chunk' AS kind,
                        documents.title AS title,
                        chunks.text AS body,
                        '[]' AS aliases, '' AS reading, '[]' AS keywords,
                        chunks.speaker AS speaker,
                        substr(documents.published_at, 1, 10) AS valid_from,
                        NULL AS valid_to, documents.source_url AS source_url,
                        documents.source_type AS source_type,
                        documents.reliability AS reliability,
                        chunks.search_text AS search_text,
                        NULL AS fts_rank
                    FROM knowledge_document_chunks AS chunks
                    JOIN knowledge_documents AS documents
                      ON documents.document_id = chunks.document_id
                    WHERE chunks.chunk_id IN ({placeholders}) {exclusion}
                    """,
                    parameters,
                ).fetchall()
                for row in matches:
                    chunk_id = str(row["record_id"]).removeprefix("chunk:")
                    add(row, "vector", vector_rank[chunk_id])

        entity_rows: list[tuple[int, sqlite3.Row]] = []
        for row in self._database.execute(
            "SELECT *, NULL AS fts_rank FROM knowledge_records"
        ).fetchall():
            if not include_term_records and str(row["kind"]) in _TERM_REFERENCE_KINDS:
                continue
            forms = _row_forms(row)
            matched = _exact_entity_matches(normalized, forms)
            if matched:
                entity_rows.append(
                    (max(len(_compact(value)) for value in matched), row)
                )
        entity_rows.sort(key=lambda value: (-value[0], str(value[1]["record_id"])))
        for rank, (_length, row) in enumerate(entity_rows, 1):
            add(row, "entity", rank)
        return candidates

    def _idf_scores(self, terms: tuple[str, ...]) -> dict[str, float]:
        if not terms:
            return {}
        total = self._database.execute(
            "SELECT "
            "(SELECT COUNT(*) FROM knowledge_records) + "
            "(SELECT COUNT(*) FROM knowledge_document_chunks) AS value"
        ).fetchone()["value"]
        if not total:
            return {}
        values: dict[str, int] = {}
        for table in ("knowledge_fts_vocab", "knowledge_chunks_fts_vocab"):
            for offset in range(0, len(terms), 400):
                batch = terms[offset : offset + 400]
                placeholders = ",".join("?" for _ in batch)
                rows = self._database.execute(
                    f"SELECT term, doc FROM {table} WHERE term IN ({placeholders})",
                    batch,
                ).fetchall()
                for row in rows:
                    term = str(row["term"])
                    values[term] = values.get(term, 0) + int(row["doc"])
        denominator = math.log(float(total) + 1.0) + 1.0
        return {
            term: (math.log((float(total) + 1.0) / (count + 1.0)) + 1.0) / denominator
            for term, count in values.items()
        }

    def _score(
        self,
        candidate: _Candidate,
        query: KnowledgeQuery,
        normalized: _NormalizedQuery,
        vector_scores: dict[str, float],
        idf_scores: dict[str, float],
    ) -> KnowledgeHit:
        row = candidate.row
        forms = _row_forms(row)
        matched = _exact_entity_matches(normalized, forms)
        query_lexemes = set(normalized.fts_terms)
        score_forms = [*forms, str(row["body"])]
        form_tokens = set(_query_tokens(" ".join(score_forms)))
        form_lexemes = form_tokens | set(_cjk_ngrams(" ".join(score_forms)))
        overlap = query_lexemes & form_lexemes
        overlap_weight = sum(idf_scores.get(value, 0.0) for value in overlap)
        query_weight = sum(idf_scores.get(value, 0.0) for value in query_lexemes)
        token_overlap = overlap_weight / max(1.0, query_weight)
        bm25 = (
            1.0 / math.log2(candidate.ranks["fts"] + 1.0)
            if "fts" in candidate.ranks
            else 0.0
        )
        exact_score = min(1.0, 0.5 * len(matched))
        keyword = max(token_overlap, exact_score)
        entity = 1.0 if matched else 0.0
        idf = max((idf_scores.get(value, 0.0) for value in overlap), default=0.0)
        specificity = max((len(_compact(value)) for value in overlap), default=0)

        query_kana = normalized.reading
        kana = max(
            (_best_window_ratio(query_kana, _kana(form)) for form in score_forms),
            default=0.0,
        )
        if kana < 0.58:
            kana = 0.0

        row_speaker = str(row["speaker"] or "")
        if query.speaker and row_speaker:
            speaker = 1.0 if query.speaker == row_speaker else -0.5
        elif row_speaker:
            speaker = 0.0
        else:
            speaker = 0.2

        date_score = _date_score(
            query.video_date,
            row["valid_from"],
            row["valid_to"],
        )
        cross = max(
            _evidence_score(query.ocr_text, forms),
            _evidence_score(query.chat_text, forms),
        )
        reliability = float(row["reliability"])
        record_id = str(row["record_id"])
        vector = vector_scores.get(record_id.removeprefix("chunk:"), 0.0)
        if vector < self._vector_minimum_score:
            vector = 0.0
        weights = self._weights
        total = (
            weights.keyword * keyword
            + weights.kana * kana
            + weights.vector * vector
            + weights.speaker * speaker
            + weights.date * date_score
            + weights.cross_evidence * cross
            + weights.reliability * reliability
            + weights.rrf * _normalized_rrf_score(candidate.ranks)
        )
        score = KnowledgeScore(
            keyword=keyword,
            kana=kana,
            vector=vector,
            speaker=speaker,
            date=date_score,
            cross_evidence=cross,
            reliability=reliability,
            total=total,
            rrf=_normalized_rrf_score(candidate.ranks),
            bm25=bm25,
            entity=entity,
            idf=idf,
            lexical_terms=len(overlap),
            specificity=specificity,
        )
        return KnowledgeHit(
            record_id=record_id,
            kind=str(row["kind"]),
            title=str(row["title"]),
            body=str(row["body"]),
            source_url=row["source_url"],
            score=score,
            matched_terms=matched,
            retrieval_ranks=dict(candidate.ranks),
        )

    def _audit(
        self,
        query: KnowledgeQuery,
        hits: list[KnowledgeHit],
        rejected: list[KnowledgeHit],
        candidate_count: int,
        *,
        event: str = "fan_knowledge_retrieval",
    ) -> None:
        if self._audit_path is None:
            return
        payload = {
            "event": event,
            "query": asdict(query),
            "candidate_count": candidate_count,
            "weak_candidate_count": len(rejected),
            "hits": [
                {
                    **hit.prompt_value(),
                    "source_url": hit.source_url,
                    "retrieval_ranks": hit.retrieval_ranks,
                    "score_components": asdict(hit.score),
                    "gate_evidence": list(_relevance_evidence(hit)),
                }
                for hit in hits
            ],
            "rejected": [
                {
                    **hit.prompt_value(),
                    "retrieval_ranks": hit.retrieval_ranks,
                    "score_components": asdict(hit.score),
                    "gate_reason": (
                        "cross_encoder_below_absolute_threshold"
                        if self._reranker is not None
                        and hit.score.reranker < self._reranker_minimum_score
                        else "no substantive lexical, semantic, entity, or cross evidence"
                    ),
                }
                for hit in rejected[:10]
            ],
        }
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _passes_relevance_gate(hit: KnowledgeHit) -> bool:
    return bool(_relevance_evidence(hit))


def _relevance_evidence(hit: KnowledgeHit) -> tuple[str, ...]:
    score = hit.score
    values: list[str] = []
    if score.entity > 0:
        values.append("exact_entity_or_alias")
    if score.vector > 0:
        values.append("vector_above_absolute_threshold")
    if (
        score.bm25 >= 0.2
        and score.idf >= 0.35
        and (score.lexical_terms >= 2 or score.specificity >= 4)
    ):
        values.append("strong_bm25_with_informative_terms")
    if score.cross_evidence > 0 and (score.keyword > 0 or score.kana >= 0.72):
        values.append("ocr_or_chat_corroboration")
    return tuple(values)


def _candidate_id(candidate: _Candidate) -> str:
    return str(candidate.row["record_id"])


def _rrf_score(ranks: dict[str, int]) -> float:
    return sum(
        _RRF_WEIGHTS[channel] / (_RRF_K + rank) for channel, rank in ranks.items()
    )


def _normalized_rrf_score(ranks: dict[str, int]) -> float:
    maximum = sum(weight / (_RRF_K + 1) for weight in _RRF_WEIGHTS.values())
    return _rrf_score(ranks) / maximum


def _interleave(*groups: list[sqlite3.Row]) -> list[sqlite3.Row]:
    values: list[sqlite3.Row] = []
    for index in range(max((len(group) for group in groups), default=0)):
        values.extend(group[index] for group in groups if index < len(group))
    return values


def _diversify_hits(hits: list[KnowledgeHit], limit: int) -> list[KnowledgeHit]:
    selected: list[KnowledgeHit] = []
    source_counts: dict[str, int] = {}
    bodies: list[str] = []
    for hit in hits:
        body = _compact(hit.body)
        if body in bodies or any(
            SequenceMatcher(None, body, existing).ratio() >= 0.92 for existing in bodies
        ):
            continue
        source = hit.source_url or f"record:{hit.record_id}"
        if source_counts.get(source, 0) >= _MAX_RESULTS_PER_SOURCE:
            continue
        selected.append(hit)
        bodies.append(body)
        source_counts[source] = source_counts.get(source, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def records_from_translation_context(
    context: dict[str, object],
) -> list[KnowledgeRecord]:
    records: dict[str, KnowledgeRecord] = {}
    for value in context.get("knowledge_records", []):
        if not isinstance(value, dict):
            continue
        raw_id = str(value.get("id") or value.get("record_id") or "").strip()
        title = str(value.get("title") or "").strip()
        body = str(value.get("body") or "").strip()
        if not raw_id or not title or not body:
            continue
        record = KnowledgeRecord(
            record_id=f"knowledge:{raw_id}",
            kind=str(value.get("kind") or "note").strip(),
            title=title,
            body=body,
            aliases=_string_tuple(value.get("aliases")),
            reading=str(value.get("reading") or "").strip(),
            keywords=_string_tuple(value.get("keywords")),
            speaker=str(value.get("speaker") or "").strip() or None,
            valid_from=str(value.get("valid_from") or "").strip() or None,
            valid_to=str(value.get("valid_to") or "").strip() or None,
            source_url=str(value.get("source_url") or "").strip() or None,
            source_type=str(value.get("source_type") or "curated_glossary").strip(),
            reliability=float(value.get("reliability", 0.8)),
        )
        records[record.record_id] = record
    for franchise in context.get("franchises", []):
        if not isinstance(franchise, dict):
            continue
        name = str(franchise.get("name") or "").strip()
        body = str(franchise.get("background") or "").strip()
        if name and body:
            record = KnowledgeRecord(
                _record_id("franchise", name),
                "franchise",
                name,
                body,
                reliability=0.9,
            )
            records[record.record_id] = record

    for character in context.get("characters", []):
        if not isinstance(character, dict):
            continue
        character_id = str(character.get("id") or "").strip()
        source_name = str(character.get("source_name") or "").strip()
        canonical = str(character.get("canonical") or "").strip()
        aliases = [
            str(alias).strip()
            for alias in character.get("aliases", [])
            if isinstance(alias, str) and alias.strip()
        ]
        for short_name in character.get("short_names", []):
            if isinstance(short_name, dict) and isinstance(
                short_name.get("source"), str
            ):
                aliases.append(str(short_name["source"]).strip())
        if character_id and source_name:
            record = KnowledgeRecord(
                _record_id("character", character_id),
                "character",
                source_name,
                f"{source_name}的固定中文名为{canonical}。",
                tuple(dict.fromkeys(aliases)),
                speaker=character_id,
                reliability=0.98,
            )
            records[record.record_id] = record

    entity_by_surface = {
        str(entity.get("surface")): entity
        for entity in context.get("asr_entities", [])
        if isinstance(entity, dict) and entity.get("surface")
    }
    for source, target in dict(context.get("terms", {})).items():
        source = str(source).strip()
        target = str(target).strip()
        if not source or not target:
            continue
        entity = entity_by_surface.get(source, {})
        aliases = tuple(
            str(alias).strip()
            for alias in entity.get("aliases", [])
            if isinstance(alias, str) and alias.strip()
        )
        reading = str(entity.get("reading") or "").strip()
        record = KnowledgeRecord(
            _record_id("term", source),
            "term",
            source,
            f"{source}的固定中文译法为{target}。",
            aliases,
            reading,
            reliability=0.97,
        )
        records[record.record_id] = record

    for surface, entity in entity_by_surface.items():
        record_id = _record_id("term", surface)
        if record_id in records:
            continue
        aliases = tuple(
            str(alias).strip()
            for alias in entity.get("aliases", [])
            if isinstance(alias, str) and alias.strip()
        )
        record = KnowledgeRecord(
            _record_id("entity", surface),
            "entity",
            surface,
            f"应将这个专有名称识别为{surface}。",
            aliases,
            str(entity.get("reading") or "").strip(),
            reliability=0.95,
        )
        records[record.record_id] = record
    return list(records.values())


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        str(item).strip() for item in value if isinstance(item, str) and item.strip()
    )


def _record_id(kind: str, value: str) -> str:
    digest = hashlib.sha256(f"{kind}\0{value}".encode()).hexdigest()[:20]
    return f"{kind}:{digest}"


def _validate_record(record: KnowledgeRecord) -> None:
    if not record.record_id or not record.kind or not record.title:
        raise ValueError("knowledge record requires id, kind and title")
    if not 0 <= record.reliability <= 1:
        raise ValueError("knowledge record reliability must be between 0 and 1")
    for value in (record.valid_from, record.valid_to):
        if value:
            date.fromisoformat(value)


def _validate_document(document: KnowledgeDocument) -> None:
    if not document.document_id or not document.source_type or not document.external_id:
        raise ValueError("knowledge document requires id, source type and external id")
    if not document.title or not document.text.strip():
        raise ValueError("knowledge document requires title and text")
    if not 0 <= document.reliability <= 1:
        raise ValueError("knowledge document reliability must be between 0 and 1")


def _document_hash(document: KnowledgeDocument, chunks: list[KnowledgeChunk]) -> str:
    payload = {
        "source_type": document.source_type,
        "external_id": document.external_id,
        "source_url": document.source_url,
        "title": document.title,
        "author": document.author,
        "published_at": document.published_at,
        "language": document.language,
        "text": document.text,
        "metadata": document.metadata or {},
        "reliability": document.reliability,
        "chunks": [asdict(chunk) for chunk in chunks],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def _chunk_id(document_id: str, chunk: KnowledgeChunk) -> str:
    payload = (
        document_id,
        chunk.ordinal,
        chunk.start_seconds,
        chunk.end_seconds,
        chunk.speaker,
        chunk.language,
        chunk.text,
    )
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()[
        :24
    ]


def _extracted_term_id(surface: str) -> str:
    normalized = _normalize_term_form(surface)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def _normalize_term_form(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().casefold()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _fts_search_text(values: Iterable[str]) -> str:
    text = " ".join(value for value in values if value)
    return " ".join(dict.fromkeys((text, *_cjk_ngrams(text))))


def _normalize_query(text: str) -> _NormalizedQuery:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    lexical: list[str] = []
    readings: list[str] = []
    if _JAPANESE_RE.search(normalized):
        try:
            from sudachipy import tokenizer

            morphemes = _sudachi_tokenizer().tokenize(
                normalized, tokenizer.Tokenizer.SplitMode.A
            )
            for morpheme in morphemes:
                part = morpheme.part_of_speech()[0]
                surface = morpheme.surface().strip().casefold()
                if part not in {"助詞", "助動詞", "補助記号", "空白"}:
                    dictionary_form = morpheme.dictionary_form().strip().casefold()
                    lexical.extend(
                        value for value in (surface, dictionary_form) if value
                    )
                reading = morpheme.reading_form()
                if reading and reading != "*":
                    readings.append(_kana(reading))
        except (ImportError, OSError):
            lexical.extend(_query_tokens(normalized))
    else:
        lexical.extend(_query_tokens(normalized))
    lexical.extend(_query_tokens(normalized))
    content_terms = list(dict.fromkeys(value for value in lexical if value))
    ngrams = [ngram for value in content_terms for ngram in _cjk_ngrams(value)]
    terms = sorted(
        dict.fromkeys((*content_terms, *ngrams)),
        key=lambda value: (-len(_compact(value)), value),
    )[:64]
    reading = "".join(readings) or _kana(normalized)
    return _NormalizedQuery(normalized, reading, tuple(terms))


def _sudachi_tokenizer() -> object:
    from sudachipy import dictionary

    instance = getattr(_SUDACHI_LOCAL, "tokenizer", None)
    if instance is None:
        instance = dictionary.Dictionary().create()
        _SUDACHI_LOCAL.tokenizer = instance
    return instance


@cache
def _term_phonetic_forms(
    title: str, reading: str, aliases: tuple[str, ...]
) -> tuple[str, ...]:
    from sudachipy import tokenizer

    values: list[str] = []
    if reading:
        values.append(_kana(reading))
    for form in (title, *aliases):
        if not form:
            continue
        parts = [
            _kana(morpheme.reading_form())
            for morpheme in _sudachi_tokenizer().tokenize(
                form, tokenizer.Tokenizer.SplitMode.A
            )
            if morpheme.reading_form() not in {"", "*"}
        ]
        for start in range(len(parts)):
            combined = ""
            for part in parts[start:]:
                combined += part
                if len(combined) > 24:
                    break
                if len(combined) >= 5:
                    values.append(combined)
    return tuple(dict.fromkeys(values))


def _exact_entity_matches(query: _NormalizedQuery, forms: list[str]) -> tuple[str, ...]:
    return tuple(
        form
        for form in forms
        if _contains_form(query.text, form)
        or _contains_form(query.reading, _kana(form))
    )


def _cjk_ngrams(text: str) -> list[str]:
    terms: list[str] = []
    for run in re.findall(r"[\u3040-\u30ff\u3400-\u9fff]+", text):
        for width in (2, 3):
            terms.extend(
                run[index : index + width] for index in range(len(run) - width + 1)
            )
    return terms


def _row_forms(row: sqlite3.Row) -> list[str]:
    return list(
        dict.fromkeys(
            value
            for value in (
                str(row["title"]),
                str(row["reading"]),
                *json.loads(row["aliases"]),
                *json.loads(row["keywords"]),
            )
            if value
        )
    )


def _query_tokens(text: str) -> list[str]:
    return list(
        dict.fromkeys(match.group(0).casefold() for match in _TOKEN_RE.finditer(text))
    )


def _compact(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def _kana(text: str) -> str:
    normalized = _compact(text)
    return "".join(
        chr(ord(character) - 0x60) if "\u30a1" <= character <= "\u30f6" else character
        for character in normalized
    )


def _contains_form(text: str, form: str) -> bool:
    compact_text = _compact(text)
    compact_form = _compact(form)
    return bool(compact_form) and compact_form in compact_text


def _lexical_matches(text: str, forms: list[str]) -> bool:
    if any(_contains_form(text, form) for form in forms):
        return True
    if not _JAPANESE_RE.search(text):
        return False
    query = _kana(text)
    return any(
        len(_compact(form)) >= 3 and _best_window_ratio(query, _kana(form)) >= 0.68
        for form in forms
    )


def _best_window_ratio(text: str, target: str) -> float:
    if not text or not target:
        return 0.0
    minimum = max(1, len(target) - 3)
    maximum = min(len(text), len(target) + 4)
    if len(text) <= maximum:
        return SequenceMatcher(None, text, target).ratio()
    return max(
        SequenceMatcher(None, text[start : start + size], target).ratio()
        for size in range(minimum, maximum + 1)
        for start in range(len(text) - size + 1)
    )


def _date_score(
    video_date: str | None,
    valid_from: str | None,
    valid_to: str | None,
) -> float:
    if not video_date or (not valid_from and not valid_to):
        return 0.0
    try:
        current = date.fromisoformat(video_date)
        start = date.fromisoformat(valid_from) if valid_from else None
        end = date.fromisoformat(valid_to) if valid_to else None
    except ValueError:
        return 0.0
    if start and current < start or end and current > end:
        return -1.0
    return 1.0


def _evidence_score(text: str, forms: list[str]) -> float:
    if not text:
        return 0.0
    matches = sum(_contains_form(text, form) for form in forms)
    return min(1.0, matches / 2)


def video_date_from_metadata(metadata: dict[str, object]) -> str | None:
    upload_date = metadata.get("upload_date")
    if isinstance(upload_date, str) and len(upload_date) == 8 and upload_date.isdigit():
        return f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"
    for key in ("release_timestamp", "timestamp"):
        value = metadata.get(key)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, UTC).date().isoformat()
    return None
