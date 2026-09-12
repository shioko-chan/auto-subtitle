from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path


class LocalVectorIndex:
    def __init__(self, path: Path, model_name: str) -> None:
        self.path = path
        self.model_name = model_name
        self._model = None
        self._device = "cpu"
        self._index = None
        self._ids: dict[str, int] = {}
        self._loaded = False
        self._lock = threading.Lock()

    def sync(self, values: list[tuple[str, str]]) -> tuple[int, int]:
        with self._lock:
            try:
                return self._sync(values)
            except BaseException:
                # A failed write must not leave uncommitted vectors in use.
                self._index = None
                self._ids = {}
                self._loaded = False
                raise

    def _sync(self, values: list[tuple[str, str]]) -> tuple[int, int]:
        import faiss
        import numpy as np

        current = {chunk_id: text for chunk_id, text in values}
        self._load_index()
        if not current and self._index is None:
            if not self._loaded:
                self._save()
                self._loaded = True
            return 0, 0
        initialized = self._index is None
        if initialized:
            dimension = int(self._load_model().get_embedding_dimension())
            self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        stale = set(self._ids) - set(current)
        if stale:
            stale_ids = np.asarray(
                [self._ids[value] for value in stale], dtype=np.int64
            )
            self._index.remove_ids(stale_ids)
            for value in stale:
                del self._ids[value]
        missing = [value for value in values if value[0] not in self._ids]
        if missing:
            vectors = self._load_model().encode(
                [f"passage: {text}" for _chunk_id, text in missing],
                batch_size=128 if self._device == "cuda" else 64,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            ).astype("float32", copy=False)
            next_id = max(self._ids.values(), default=0) + 1
            numeric = np.arange(next_id, next_id + len(missing), dtype=np.int64)
            self._index.add_with_ids(vectors, numeric)
            self._ids.update(
                {
                    chunk_id: int(vector_id)
                    for (chunk_id, _text), vector_id in zip(missing, numeric)
                }
            )
        if initialized or missing or stale:
            self._save()
        self._loaded = True
        return len(missing), len(stale)

    def is_ready(self) -> bool:
        with self._lock:
            self._load_index()
            return self._index is not None or self._loaded

    def search(self, text: str, limit: int) -> dict[str, float]:
        return self.search_many([text], limit)[0]

    def search_many(self, texts: list[str], limit: int) -> list[dict[str, float]]:
        results: list[dict[str, float]] = [{} for _ in texts]
        active = [(i, text) for i, text in enumerate(texts) if text.strip()]
        if limit < 1 or not active:
            return results
        with self._lock:
            self._load_index()
            if self._index is None or not self._ids:
                return results
            model = self._load_model()
            vectors = model.encode(
                [f"query: {text}" for _, text in active],
                batch_size=128 if self._device == "cuda" else 64,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32", copy=False)
            scores, ids = self._index.search(vectors, min(limit, len(self._ids)))
            reverse = {value: key for key, value in self._ids.items()}
            for (position, _), row_scores, row_ids in zip(active, scores, ids, strict=True):
                results[position] = {
                    reverse[int(vector_id)]: float(score)
                    for score, vector_id in zip(row_scores, row_ids, strict=True)
                    if int(vector_id) in reverse
                }
        return results

    def release_model(self) -> bool:
        with self._lock:
            if self._model is None:
                return False
            self._model = None
            self._device = "cpu"
            return True

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "embedding retrieval requires sentence-transformers and faiss-cpu"
                ) from exc
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model

    def _load_index(self) -> None:
        if self._index is not None or self._loaded:
            return
        if not self.path.is_file():
            return
        import faiss
        import numpy as np

        with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            if db.execute("SELECT name FROM sqlite_master WHERE name='vector_snapshot'").fetchone() is None:
                return
            row = db.execute("SELECT model, ids, vectors FROM vector_snapshot WHERE id=1").fetchone()
        if row is None or row[0] != self.model_name:
            return
        ids = {str(key): int(value) for key, value in json.loads(row[1]).items()}
        index = faiss.deserialize_index(np.frombuffer(row[2], dtype=np.uint8)) if row[2] is not None else None
        self._ids, self._index = ids, index
        self._loaded = True

    def _save(self) -> None:
        import faiss

        self.path.parent.mkdir(parents=True, exist_ok=True)
        vectors = faiss.serialize_index(self._index).tobytes() if self._index is not None else None
        with closing(sqlite3.connect(self.path, timeout=30)) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS vector_snapshot (
                    id INTEGER PRIMARY KEY CHECK(id=1),
                    model TEXT NOT NULL,
                    ids TEXT NOT NULL,
                    vectors BLOB
                )
            """)
            db.execute("""
                INSERT INTO vector_snapshot(id, model, ids, vectors) VALUES(1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    model=excluded.model, ids=excluded.ids, vectors=excluded.vectors
            """, (self.model_name, json.dumps(self._ids, sort_keys=True), vectors))
