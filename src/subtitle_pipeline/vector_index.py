from __future__ import annotations

import json
from pathlib import Path


class LocalVectorIndex:
    def __init__(self, path: Path, model_name: str) -> None:
        self.path = path
        self.metadata_path = path.with_suffix(path.suffix + ".json")
        self.model_name = model_name
        self._model = None
        self._device = "cpu"
        self._index = None
        self._ids: dict[str, int] = {}

    def sync(self, values: list[tuple[str, str]]) -> tuple[int, int]:
        import faiss
        import numpy as np

        model = self._load_model()
        current = {chunk_id: text for chunk_id, text in values}
        self._load_index()
        if self._index is None:
            dimension = int(model.get_embedding_dimension())
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
            vectors = model.encode(
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
        self._save()
        return len(missing), len(stale)

    def search(self, text: str, limit: int) -> dict[str, float]:
        if limit < 1 or not text.strip():
            return {}
        self._load_index()
        if self._index is None or not self._ids:
            return {}
        vector = (
            self._load_model()
            .encode(
                [f"query: {text}"],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            .astype("float32", copy=False)
        )
        scores, ids = self._index.search(vector, min(limit, len(self._ids)))
        reverse = {value: key for key, value in self._ids.items()}
        return {
            reverse[int(vector_id)]: float(score)
            for score, vector_id in zip(scores[0], ids[0])
            if int(vector_id) in reverse
        }

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
        if self._index is not None:
            return
        if not self.path.is_file() or not self.metadata_path.is_file():
            return
        import faiss

        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if metadata.get("model") != self.model_name:
            return
        self._ids = {str(key): int(value) for key, value in metadata["ids"].items()}
        self._index = faiss.read_index(str(self.path))

    def _save(self) -> None:
        import faiss

        assert self._index is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_index = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary_metadata = self.metadata_path.with_suffix(
            self.metadata_path.suffix + ".tmp"
        )
        faiss.write_index(self._index, str(temporary_index))
        temporary_metadata.write_text(
            json.dumps({"model": self.model_name, "ids": self._ids}, sort_keys=True),
            encoding="utf-8",
        )
        temporary_index.replace(self.path)
        temporary_metadata.replace(self.metadata_path)
