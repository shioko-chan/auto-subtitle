from __future__ import annotations

import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from subtitle_pipeline.vector_index import LocalVectorIndex


class LocalVectorIndexTests(unittest.TestCase):
    def test_batch_encodes_queries_together_and_preserves_empty_positions(self):
        from unittest.mock import Mock
        index = LocalVectorIndex(Path("unused.faiss"), "fake")
        index._model = Mock()
        index._model.encode.return_value = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
        index._index = Mock()
        index._index.search.return_value = (np.asarray([[0.9], [0.8]]), np.asarray([[1], [2]]))
        index._ids = {"first": 1, "second": 2}
        self.assertEqual(index.search_many(["a", "", "b"], 1), [{"first": 0.9}, {}, {"second": 0.8}])
        self.assertEqual(index._model.encode.call_count, 1)
        self.assertEqual(index._model.encode.call_args.args[0], ["query: a", "query: b"])
        self.assertEqual(index._index.search.call_count, 1)

    def test_release_model_preserves_loaded_index(self) -> None:
        index = LocalVectorIndex(Path("unused.faiss"), "fake")
        loaded_index = object()
        index._model = object()
        index._device = "cuda"
        index._index = loaded_index

        self.assertTrue(index.release_model())
        self.assertIsNone(index._model)
        self.assertEqual(index._device, "cpu")
        self.assertIs(index._index, loaded_index)
        self.assertFalse(index.release_model())

    def test_search_serializes_shared_embedding_model(self) -> None:
        class ReentryDetectingModel:
            def __init__(self) -> None:
                self.active = 0
                self.maximum_active = 0
                self.lock = threading.Lock()

            def get_embedding_dimension(self) -> int:
                return 2

            def encode(self, values, **_kwargs):
                with self.lock:
                    self.active += 1
                    self.maximum_active = max(self.maximum_active, self.active)
                try:
                    time.sleep(0.02)
                    return np.asarray([[1.0, 0.0] for _value in values], dtype=np.float32)
                finally:
                    with self.lock:
                        self.active -= 1

        with tempfile.TemporaryDirectory() as temporary:
            index = LocalVectorIndex(Path(temporary) / "knowledge.faiss", "fake")
            model = ReentryDetectingModel()
            index._model = model
            index.sync([("chunk-1", "knowledge")])
            model.maximum_active = 0

            with ThreadPoolExecutor(max_workers=3) as executor:
                results = list(
                    executor.map(lambda value: index.search(value, 1), ["a", "b", "c"])
                )

        self.assertEqual(model.maximum_active, 1)
        self.assertEqual(results, [{"chunk-1": 1.0}] * 3)


if __name__ == "__main__":
    unittest.main()
