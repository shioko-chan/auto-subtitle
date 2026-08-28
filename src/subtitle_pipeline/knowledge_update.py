from __future__ import annotations

import hashlib
import json
import logging
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .chat_context import remove_youtube_chat_files
from .config import AppConfig
from .fan_knowledge import FanKnowledgeRetriever
from .knowledge_collection import collect_official_documents, collect_sns_documents
from .knowledge_ingestion import (
    IngestionSummary,
    download_youtube_subtitles,
    ingest_document_mapping,
    ingest_youtube_cache,
)

logger = logging.getLogger(__name__)
_LAST_SUCCESS_KEY = "automatic_update_last_success_at"


def update_knowledge_if_stale(
    config: AppConfig, retriever: FanKnowledgeRetriever
) -> IngestionSummary:
    lock_path = Path(config.fan_knowledge.database_path).with_suffix(".update.lock")
    with _exclusive_update_lock(lock_path):
        try:
            result = _update_knowledge_if_stale(config, retriever)
        except Exception as exc:
            retriever.set_metadata(
                "automatic_update_last_error",
                json.dumps(
                    {
                        "at": datetime.now(UTC).isoformat(),
                        "type": type(exc).__name__,
                        "message": str(exc)[:1000],
                    },
                    ensure_ascii=False,
                ),
            )
            logger.error("fan knowledge update failed; stopping pipeline: %s", exc)
            raise
        retriever.set_metadata("automatic_update_last_error", "")
        return result


def _update_knowledge_if_stale(
    config: AppConfig, retriever: FanKnowledgeRetriever
) -> IngestionSummary:
    settings = config.fan_knowledge
    if not settings.auto_update_enabled:
        return IngestionSummary()
    previous = _parse_time(retriever.metadata(_LAST_SUCCESS_KEY))
    now = datetime.now(UTC)
    if previous is not None and now - previous < timedelta(
        hours=settings.update_interval_hours
    ):
        logger.info("fan knowledge is current; last update %s", previous.isoformat())
        return IngestionSummary()

    logger.info("fan knowledge is stale; starting synchronous incremental update")
    summary = IngestionSummary()
    known_youtube = retriever.external_ids(
        ("youtube_metadata", "youtube_auto_subtitle", "youtube_super_chat")
    )
    cache_dir = Path(settings.collection_cache_dir)
    for source in settings.youtube_sources:
        downloaded_ids: tuple[str, ...] = ()

        def ingest_batch(video_ids: tuple[str, ...]) -> None:
            nonlocal summary
            summary = summary.merge(
                ingest_youtube_cache(
                    retriever,
                    cache_dir,
                    target_seconds=settings.chunk_target_seconds,
                    maximum_chars=settings.chunk_max_chars,
                    language_priority=tuple(
                        value.removesuffix(".*")
                        for value in settings.youtube_subtitle_languages
                    ),
                    include_regular_chat=False,
                    video_ids=set(video_ids),
                )
            )
            for video_id in video_ids:
                remove_youtube_chat_files(cache_dir / video_id)

        downloaded_ids = download_youtube_subtitles(
            [source],
            cache_dir,
            config.download,
            languages=tuple(settings.youtube_subtitle_languages),
            playlist_end=settings.youtube_playlist_limit,
            after_batch=ingest_batch,
            known_video_ids=known_youtube,
        )
        known_youtube.update(downloaded_ids)

    known_official_urls = retriever.source_urls(
        ("official_news", "official_event", "official_music_notice")
    )
    for source_index, source in enumerate(settings.official_sources):
        source_type, follow_links, link_pattern = _official_source_policy(source)
        validator_key = (
            "official_http_validator:"
            + hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        )
        saved_validators = retriever.metadata(validator_key)
        try:
            validators = json.loads(saved_validators) if saved_validators else {}
        except json.JSONDecodeError:
            validators = {}
        response_validators: dict[str, dict[str, str]] = {}
        values = collect_official_documents(
            [source],
            source_type=source_type,
            follow_links=follow_links,
            link_pattern=link_pattern,
            maximum_documents=1000,
            maximum_depth=1,
            strict_errors=True,
            known_source_urls=known_official_urls,
            conditional_headers={source: validators},
            response_validators=response_validators,
        )
        for item_index, value in enumerate(values):
            summary = summary.add(
                ingest_document_mapping(
                    retriever,
                    value,
                    fallback_external_id=(
                        f"automatic-official:{source_index}:{item_index}"
                    ),
                    maximum_chars=settings.chunk_max_chars,
                )
            )
        if source in response_validators:
            retriever.set_metadata(
                validator_key,
                json.dumps(response_validators[source], sort_keys=True),
            )

    known_sns = retriever.external_ids(("x_post", "instagram_post"))
    for source_index, source in enumerate(settings.sns_sources):
        values = collect_sns_documents(
            [source],
            cookies_from_browser=config.download.cookies_from_browser,
            known_external_ids=known_sns,
        )
        for item_index, value in enumerate(values):
            external_id = str(value.get("external_id") or "")
            if external_id in known_sns:
                continue
            summary = summary.add(
                ingest_document_mapping(
                    retriever,
                    value,
                    fallback_external_id=(f"automatic-sns:{source_index}:{item_index}"),
                    maximum_chars=settings.chunk_max_chars,
                )
            )
            known_sns.add(external_id)

    added_vectors, removed_vectors = retriever.sync_vector_index()
    logger.info(
        "fan knowledge vector index: added=%d removed=%d",
        added_vectors,
        removed_vectors,
    )
    retriever.set_metadata(_LAST_SUCCESS_KEY, datetime.now(UTC).isoformat())
    logger.info(
        "fan knowledge update complete: scanned=%d changed=%d unchanged=%d chunks=%d",
        summary.scanned,
        summary.inserted_or_updated,
        summary.unchanged,
        summary.chunks,
    )
    return summary


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return (
        parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    )


def _official_source_policy(url: str) -> tuple[str, bool, str | None]:
    if "/news" in url:
        return "official_news", True, r"/news/\d+/?$"
    if "/events" in url:
        return "official_event", True, r"/events/"
    if "bushiroad-music.com" in url:
        return "official_music_notice", False, None
    return "official_news", True, None


@contextmanager
def _exclusive_update_lock(path: Path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
