from __future__ import annotations

import threading


class LocalCrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self._lock = threading.Lock()

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        with self._lock:
            model = self._load_model()
            values = model.predict(
                [(query, passage) for passage in passages],
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        return [
            float(value.item() if hasattr(value, "item") else value)
            for value in values
        ]

    def release_model(self) -> bool:
        with self._lock:
            if self._model is None:
                return False
            self._model = None
            return True

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "cross-encoder reranking requires sentence-transformers"
                ) from exc
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_kwargs = {"dtype": torch.float16} if device == "cuda" else {}
            self._model = CrossEncoder(
                self.model_name,
                device=device,
                max_length=self.max_length,
                model_kwargs=model_kwargs,
            )
        return self._model
