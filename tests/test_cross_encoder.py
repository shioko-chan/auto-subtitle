import unittest

from subtitle_pipeline.cross_encoder import LocalCrossEncoderReranker


class LocalCrossEncoderRerankerTests(unittest.TestCase):
    def test_batch_preserves_query_boundaries_with_empty_candidates(self):
        from unittest.mock import Mock
        reranker = LocalCrossEncoderReranker("fake")
        reranker._model = Mock()
        reranker._model.predict.return_value = [0.1, 0.2, 0.3]
        scores = reranker.score_many([("a", ["x", "y"]), ("b", []), ("c", ["z"])])
        self.assertEqual(scores, [[0.1, 0.2], [], [0.3]])
        self.assertEqual(reranker._model.predict.call_count, 1)
        self.assertEqual(reranker._model.predict.call_args.args[0], [("a", "x"), ("a", "y"), ("c", "z")])

    def test_release_model_allows_lazy_reload(self) -> None:
        reranker = LocalCrossEncoderReranker("fake")
        reranker._model = object()

        self.assertTrue(reranker.release_model())
        self.assertIsNone(reranker._model)
        self.assertFalse(reranker.release_model())
