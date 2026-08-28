from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from subtitle_pipeline.config import AppConfig, FanKnowledgeConfig
from subtitle_pipeline.fan_knowledge import FanKnowledgeRetriever
from subtitle_pipeline.knowledge_update import update_knowledge_if_stale


class KnowledgeUpdateTests(unittest.TestCase):
    def test_recent_success_skips_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            retriever = FanKnowledgeRetriever(Path(temporary) / "knowledge.sqlite3")
            retriever.set_metadata(
                "automatic_update_last_success_at", datetime.now(UTC).isoformat()
            )
            config = AppConfig(
                fan_knowledge=FanKnowledgeConfig(
                    database_path=str(Path(temporary) / "knowledge.sqlite3"),
                    embedding_model=None,
                )
            )
            with patch(
                "subtitle_pipeline.knowledge_update.download_youtube_subtitles"
            ) as download:
                update_knowledge_if_stale(config, retriever)
            retriever.close()

        download.assert_not_called()

    def test_failed_collection_does_not_record_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            retriever = FanKnowledgeRetriever(Path(temporary) / "knowledge.sqlite3")
            knowledge = replace(
                FanKnowledgeConfig(
                    database_path=str(Path(temporary) / "knowledge.sqlite3"),
                    embedding_model=None,
                ),
                official_sources=["https://example.jp/news/"],
            )
            config = AppConfig(fan_knowledge=knowledge)
            with (
                patch(
                    "subtitle_pipeline.knowledge_update.collect_official_documents",
                    side_effect=RuntimeError("network failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "network failed"),
            ):
                update_knowledge_if_stale(config, retriever)
            self.assertIsNone(retriever.metadata("automatic_update_last_success_at"))
            self.assertIn(
                "network failed",
                retriever.metadata("automatic_update_last_error") or "",
            )
            retriever.close()


if __name__ == "__main__":
    unittest.main()
