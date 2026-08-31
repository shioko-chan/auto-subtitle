import unittest

from subtitle_pipeline.cross_encoder import LocalCrossEncoderReranker


class LocalCrossEncoderRerankerTests(unittest.TestCase):
    def test_release_model_allows_lazy_reload(self) -> None:
        reranker = LocalCrossEncoderReranker("fake")
        reranker._model = object()

        self.assertTrue(reranker.release_model())
        self.assertIsNone(reranker._model)
        self.assertFalse(reranker.release_model())
