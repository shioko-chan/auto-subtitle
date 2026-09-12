from __future__ import annotations

import tempfile
import threading
import time
import unittest
import sqlite3
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

from subtitle_pipeline.vector_index import LocalVectorIndex


class LocalVectorIndexTests(unittest.TestCase):
    class Model:
        def get_embedding_dimension(self):
            return 2

        def encode(self, texts, **kwargs):
            return np.asarray([[0., 1.] if 'beta' in text else [1., 0.]
                               for text in texts], dtype=np.float32)

    def index(self, path, model_name='fake'):
        index = LocalVectorIndex(path, model_name)
        index._model = self.Model()
        return index

    def test_snapshot_restores_matching_vectors_and_ids_from_one_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'knowledge.sqlite3'
            index = self.index(path)
            self.assertEqual(index.sync([('alpha-id', 'alpha')]), (1, 0))
            self.assertEqual(index.sync([('beta-id', 'beta')]), (1, 1))
            reopened = self.index(path)
            self.assertTrue(reopened.is_ready())
            self.assertEqual(reopened.search('beta', 1), {'beta-id': 1.0})
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_interruption_before_commit_keeps_previous_complete_snapshot(self):
        class InterruptedConnection(sqlite3.Connection):
            def __exit__(self, exc_type, exc, traceback):
                if exc_type is None:
                    raise RuntimeError('interrupted before commit')
                return super().__exit__(exc_type, exc, traceback)

        connect = sqlite3.connect
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'knowledge.sqlite3'
            index = self.index(path)
            index.sync([('alpha-id', 'alpha')])
            with patch('subtitle_pipeline.vector_index.sqlite3.connect',
                       side_effect=lambda *args, **kwargs: connect(
                           *args, **kwargs, factory=InterruptedConnection)):
                with self.assertRaisesRegex(RuntimeError, 'interrupted before commit'):
                    index.sync([('beta-id', 'beta')])
            self.assertEqual(self.index(path).search('alpha', 1), {'alpha-id': 1.0})
            self.assertEqual(index.search('alpha', 1), {'alpha-id': 1.0})
            self.assertEqual(index.sync([('beta-id', 'beta')]), (1, 1))
            self.assertEqual(self.index(path).search('beta', 1), {'beta-id': 1.0})

    def test_empty_or_different_model_snapshot_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'knowledge.sqlite3'
            with closing(sqlite3.connect(path)):
                pass
            original = self.index(path)
            self.assertFalse(original.is_ready())
            original.sync([('alpha-id', 'alpha')])
            changed = self.index(path, 'changed-model')
            self.assertFalse(changed.is_ready())
            changed.sync([])
            self.assertTrue(self.index(path, 'changed-model').is_ready())
            self.assertEqual(self.index(path, 'changed-model').search('alpha', 1), {})

    def test_unchanged_snapshot_needs_no_model_load_or_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'knowledge.sqlite3'
            values = [('alpha-id', 'alpha')]
            self.index(path).sync(values)
            index = LocalVectorIndex(path, 'fake')
            with patch.object(index, '_load_model', side_effect=AssertionError('model load')), \
                 patch.object(index, '_save', side_effect=AssertionError('snapshot write')):
                self.assertEqual(index.sync(values), (0, 0))

    def test_empty_snapshot_does_not_require_an_embedding_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'knowledge.sqlite3'
            index = LocalVectorIndex(path, 'unavailable-model')
            with patch.object(index, '_load_model', side_effect=AssertionError('model load')):
                self.assertEqual(index.sync([]), (0, 0))
                self.assertTrue(index.is_ready())
                self.assertEqual(index.search('query', 1), {})
            reopened = LocalVectorIndex(path, 'unavailable-model')
            self.assertTrue(reopened.is_ready())
            with patch.object(reopened, '_load_model', side_effect=AssertionError('model load')), \
                 patch.object(reopened, '_save', side_effect=AssertionError('snapshot write')):
                self.assertEqual(reopened.sync([]), (0, 0))
            reopened._model = self.Model()
            self.assertEqual(reopened.sync([('beta-id', 'beta')]), (1, 0))
            self.assertEqual(reopened.search('beta', 1), {'beta-id': 1.0})
            with patch.object(reopened, '_load_model', side_effect=AssertionError('model load')):
                self.assertEqual(reopened.sync([]), (0, 1))
                self.assertEqual(reopened.search('beta', 1), {})

    def test_batch_encodes_queries_together_and_preserves_empty_positions(self):
        from unittest.mock import Mock
        index = LocalVectorIndex(Path("unused.sqlite3"), "fake")
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
        index = LocalVectorIndex(Path("unused.sqlite3"), "fake")
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
            index = LocalVectorIndex(Path(temporary) / "knowledge.sqlite3", "fake")
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
