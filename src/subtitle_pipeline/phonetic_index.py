"""Persistent term readings and native, parallel substring matching."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Callable
from importlib.metadata import version

import numpy as np
from rapidfuzz import fuzz, process

# A five-kana term with one substituted kana scores 80. Shorter fuzzy forms
# are too ambiguous; spelling/alias exact matches are handled by the caller.
PHONETIC_CUTOFF = 80.0
PHONETIC_MIN_LENGTH = 5
PHONETIC_VERSION = f"term-readings-v2:sudachi-{version('SudachiPy')}:{version('SudachiDict-core')}"


class TermPhoneticIndex:
    def __init__(self, database: sqlite3.Connection, kinds: tuple[str, ...], build: Callable):
        self.database = database
        self.kinds = kinds
        self.build = build
        database.executescript("""
            CREATE TABLE IF NOT EXISTS knowledge_term_pronunciations (
                record_id TEXT PRIMARY KEY REFERENCES knowledge_records(record_id) ON DELETE CASCADE,
                fingerprint TEXT NOT NULL,
                forms_json TEXT NOT NULL
            );
            CREATE TRIGGER IF NOT EXISTS knowledge_pronunciations_delete
            AFTER DELETE ON knowledge_records BEGIN
                DELETE FROM knowledge_term_pronunciations WHERE record_id=old.record_id;
            END;
        """)

    def sync_record(self, record_id: str) -> bool:
        row = self.database.execute(
            "SELECT record_id,kind,title,reading,aliases FROM knowledge_records WHERE record_id=?",
            (record_id,),
        ).fetchone()
        if row is None or row['kind'] not in self.kinds:
            self.database.execute("DELETE FROM knowledge_term_pronunciations WHERE record_id=?", (record_id,))
            return False
        source = [PHONETIC_VERSION, row['kind'], row['title'], row['reading'], row['aliases']]
        fingerprint = hashlib.sha256(json.dumps(source, ensure_ascii=False).encode()).hexdigest()
        saved = self.database.execute(
            "SELECT fingerprint FROM knowledge_term_pronunciations WHERE record_id=?", (record_id,)
        ).fetchone()
        if saved is not None and saved['fingerprint'] == fingerprint:
            return False
        forms = self.build(row['title'], row['reading'], tuple(json.loads(row['aliases'])))
        forms = list(dict.fromkeys(value for value in forms if len(value) >= PHONETIC_MIN_LENGTH))
        self.database.execute(
            "INSERT OR REPLACE INTO knowledge_term_pronunciations VALUES(?,?,?)",
            (record_id, fingerprint, json.dumps(forms, ensure_ascii=False)),
        )
        return True

    def sync(self) -> int:
        return sum(self.sync_record(row['record_id']) for row in self.database.execute(
            "SELECT record_id FROM knowledge_records ORDER BY record_id"
        ).fetchall())


def match_readings(queries: list[str], forms: list[str]) -> np.ndarray:
    """Return 0..1 partial-Indel scores; batches bound temporary matrix memory.

    RapidFuzz's C-API scorer releases the GIL. Match positions refer to these
    normalized reading strings, not the original orthographic source text.
    """
    if not queries or not forms:
        return np.zeros((len(queries), len(forms)), dtype=np.float32)
    scores = process.cdist(
        queries, forms, scorer=fuzz.partial_ratio, processor=None,
        score_cutoff=PHONETIC_CUTOFF, workers=min(8, len(os.sched_getaffinity(0))),
        dtype=np.float32,
    )
    # partial_ratio is symmetric: do not treat a short query as a perfect match
    # for a much longer term. Compare the complete strings in that direction.
    for i, query in enumerate(queries):
        if len(query) < PHONETIC_MIN_LENGTH:
            scores[i] = 0
            continue
        for j in np.flatnonzero(scores[i]):
            if len(query) < len(forms[j]):
                scores[i, j] = fuzz.ratio(query, forms[j], score_cutoff=PHONETIC_CUTOFF)
    return scores / 100.0
