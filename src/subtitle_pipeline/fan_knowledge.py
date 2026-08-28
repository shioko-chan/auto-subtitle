from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Self

from .vector_index import LocalVectorIndex

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.-]*|[\u3040-\u30ff\u3400-\u9fff]+")
_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_SCHEMA_VERSION = 3
_LEXICAL_CANDIDATES = 40
_VECTOR_CANDIDATES = 40
_FUSION_CANDIDATES = 30
_RRF_K = 60
_RRF_WEIGHTS = {"fts": 1.0, "vector": 1.0, "entity": 2.0}
_MAX_RESULTS_PER_SOURCE = 2


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
    ) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._database_path = database_path
        self._audit_path = audit_path
        self._weights = weights or RetrievalWeights()
        self._vector_minimum_score = vector_minimum_score
        self._vector_index = (
            LocalVectorIndex(vector_index_path, embedding_model)
            if embedding_model and vector_index_path is not None
            else None
        )
        self._lock = threading.Lock()
        self._database = sqlite3.connect(database_path, check_same_thread=False)
        self._database.row_factory = sqlite3.Row
        self._initialize()

    def close(self) -> None:
        self._database.close()

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
        return DocumentUpsertResult(document.document_id, True, len(chunk_values))

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
        if query.top_k < 1 or not query.text.strip():
            return []
        normalized = _normalize_query(query.text)
        vector_scores = (
            self._vector_index.search(normalized.text, _VECTOR_CANDIDATES)
            if self._vector_index is not None
            else {}
        )
        with self._lock:
            candidates = self._candidates(query, normalized, vector_scores)
            idf = self._idf_scores(normalized.fts_terms)
        fused = sorted(
            candidates.values(),
            key=lambda candidate: (-_rrf_score(candidate.ranks), _candidate_id(candidate)),
        )[:_FUSION_CANDIDATES]
        hits = [
            self._score(candidate, query, normalized, vector_scores, idf)
            for candidate in fused
        ]
        rejected = [hit for hit in hits if not _passes_relevance_gate(hit)]
        hits = [hit for hit in hits if _passes_relevance_gate(hit)]
        hits.sort(key=lambda hit: (-hit.score.total, hit.record_id))
        selected = _diversify_hits(hits, query.top_k)
        self._audit(query, selected, rejected, len(candidates))
        return selected

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

    def _candidates(
        self,
        query: KnowledgeQuery,
        normalized: _NormalizedQuery,
        vector_scores: dict[str, float],
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
                candidates[record_id] = _Candidate(
                    row, {channel: rank}, raw_bm25
                )
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
            try:
                matches = self._database.execute(
                    """
                    SELECT records.*, bm25(knowledge_fts) AS fts_rank
                    FROM knowledge_fts
                    JOIN knowledge_records AS records
                      ON records.rowid = knowledge_fts.rowid
                    WHERE knowledge_fts MATCH ?
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
                        "AND documents.external_id != ? "
                        "AND substr(documents.external_id, 1, length(?) + 1) != ? || ':'"
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
                        "AND documents.external_id != ? "
                        "AND substr(documents.external_id, 1, length(?) + 1) != ? || ':'"
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
            forms = _row_forms(row)
            matched = _exact_entity_matches(normalized, forms)
            if matched:
                entity_rows.append((max(len(_compact(value)) for value in matched), row))
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
                    f"SELECT term, doc FROM {table} "
                    f"WHERE term IN ({placeholders})",
                    batch,
                ).fetchall()
                for row in rows:
                    term = str(row["term"])
                    values[term] = values.get(term, 0) + int(row["doc"])
        denominator = math.log(float(total) + 1.0) + 1.0
        return {
            term: (math.log((float(total) + 1.0) / (count + 1.0)) + 1.0)
            / denominator
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
    ) -> None:
        if self._audit_path is None:
            return
        payload = {
            "event": "fan_knowledge_retrieval",
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
                    "gate_reason": "no substantive lexical, semantic, entity, or cross evidence",
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
        _RRF_WEIGHTS[channel] / (_RRF_K + rank)
        for channel, rank in ranks.items()
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
            SequenceMatcher(None, body, existing).ratio() >= 0.92
            for existing in bodies
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
                    lexical.extend(value for value in (surface, dictionary_form) if value)
                reading = morpheme.reading_form()
                if reading and reading != "*":
                    readings.append(_kana(reading))
        except (ImportError, OSError):
            lexical.extend(_query_tokens(normalized))
    else:
        lexical.extend(_query_tokens(normalized))
    lexical.extend(_query_tokens(normalized))
    content_terms = list(dict.fromkeys(value for value in lexical if value))
    ngrams = [
        ngram
        for value in content_terms
        for ngram in _cjk_ngrams(value)
    ]
    terms = sorted(
        dict.fromkeys((*content_terms, *ngrams)),
        key=lambda value: (-len(_compact(value)), value),
    )[:64]
    reading = "".join(readings) or _kana(normalized)
    return _NormalizedQuery(normalized, reading, tuple(terms))


@lru_cache(maxsize=1)
def _sudachi_tokenizer() -> object:
    from sudachipy import dictionary

    return dictionary.Dictionary().create()


def _exact_entity_matches(
    query: _NormalizedQuery, forms: list[str]
) -> tuple[str, ...]:
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
