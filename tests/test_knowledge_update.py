from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from subtitle_pipeline.config import AppConfig, FanKnowledgeConfig
from subtitle_pipeline.fan_knowledge import FanKnowledgeRetriever
from subtitle_pipeline.knowledge_update import update_knowledge_if_stale


class KnowledgeUpdateTests(unittest.TestCase):
    def test_failed_attempt_skips_rerun_until_interval_expires(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "knowledge.sqlite3"
            config = AppConfig(fan_knowledge=FanKnowledgeConfig(
                database_path=str(path), embedding_model=None,
                youtube_sources=[], sns_sources=[],
                official_sources=["https://example.jp/news/"],
            ))
            retriever = FanKnowledgeRetriever(path)
            try:
                with patch(
                    "subtitle_pipeline.knowledge_update.collect_official_documents",
                    side_effect=RuntimeError("network failed"),
                ) as collect:
                    update_knowledge_if_stale(config, retriever)
                    error = retriever.metadata("automatic_update_last_error")
                    update_knowledge_if_stale(config, retriever)
                    self.assertEqual(collect.call_count, 1)
                    self.assertEqual(retriever.metadata("automatic_update_last_error"), error)
                    self.assertIsNone(retriever.metadata("automatic_update_last_success_at"))
                    retriever.set_metadata("automatic_update_last_attempt_at", "2000-01-01T00:00:00+00:00")
                    update_knowledge_if_stale(config, retriever)
                    self.assertEqual(collect.call_count, 2)
            finally:
                retriever.close()

    def test_interrupted_attempt_is_recorded_before_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "knowledge.sqlite3"
            config = AppConfig(fan_knowledge=FanKnowledgeConfig(
                database_path=str(path), embedding_model=None,
                youtube_sources=[], sns_sources=[],
                official_sources=["https://example.jp/news/"],
            ))
            retriever = FanKnowledgeRetriever(path)
            try:
                with patch(
                    "subtitle_pipeline.knowledge_update.collect_official_documents",
                    side_effect=KeyboardInterrupt,
                ) as collect:
                    with self.assertRaises(KeyboardInterrupt):
                        update_knowledge_if_stale(config, retriever)
                    self.assertIsNotNone(retriever.metadata("automatic_update_last_attempt_at"))
                    update_knowledge_if_stale(config, retriever)
                    self.assertEqual(collect.call_count, 1)
            finally:
                retriever.close()

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
            with patch(
                "subtitle_pipeline.knowledge_update.collect_official_documents",
                side_effect=RuntimeError("network failed"),
            ):
                summary = update_knowledge_if_stale(config, retriever)
            self.assertEqual(summary.scanned, 0)
            self.assertIsNone(retriever.metadata("automatic_update_last_success_at"))
            self.assertIn(
                "network failed",
                retriever.metadata("automatic_update_last_error") or "",
            )
            retriever.close()

    def test_failed_collection_raises_when_existing_database_is_unreadable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(
                fan_knowledge=FanKnowledgeConfig(
                    database_path=str(Path(temporary) / "knowledge.sqlite3"),
                    embedding_model=None,
                )
            )
            retriever = MagicMock()
            retriever.metadata.side_effect = RuntimeError("database unreadable")
            with patch(
                "subtitle_pipeline.knowledge_update._update_knowledge_if_stale",
                side_effect=RuntimeError("network failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "network failed"):
                    update_knowledge_if_stale(config, retriever)

    def test_failed_collection_continues_when_error_audit_cannot_be_written(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = AppConfig(
                fan_knowledge=FanKnowledgeConfig(
                    database_path=str(Path(temporary) / "knowledge.sqlite3"),
                    embedding_model=None,
                )
            )
            retriever = MagicMock()
            retriever.metadata.return_value = "2026-09-01T00:00:00+00:00"
            retriever.set_metadata.side_effect = RuntimeError("read only")
            with patch(
                "subtitle_pipeline.knowledge_update._update_knowledge_if_stale",
                side_effect=RuntimeError("network failed"),
            ):
                summary = update_knowledge_if_stale(config, retriever)

        self.assertEqual(summary.scanned, 0)


if __name__ == "__main__":
    unittest.main()
