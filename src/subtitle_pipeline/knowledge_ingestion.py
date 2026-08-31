from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .chat_context import (
    YouTubeChatMessage,
    read_youtube_live_chat,
)
from .commands import require_command
from .config import DownloadConfig
from .fan_knowledge import (
    DocumentUpsertResult,
    FanKnowledgeRetriever,
    KnowledgeChunk,
    KnowledgeDocument,
)
from .subtitles import Cue, read_subtitles

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionSummary:
    scanned: int = 0
    inserted_or_updated: int = 0
    unchanged: int = 0
    chunks: int = 0
    skipped: int = 0

    def add(
        self,
        result: DocumentUpsertResult | None,
        *,
        scanned: int = 1,
    ) -> IngestionSummary:
        if result is None:
            return IngestionSummary(
                self.scanned + scanned,
                self.inserted_or_updated,
                self.unchanged,
                self.chunks,
                self.skipped + 1,
            )
        return IngestionSummary(
            self.scanned + scanned,
            self.inserted_or_updated + int(result.changed),
            self.unchanged + int(not result.changed),
            self.chunks + result.chunk_count,
            self.skipped,
        )

    def merge(self, other: IngestionSummary) -> IngestionSummary:
        return IngestionSummary(
            self.scanned + other.scanned,
            self.inserted_or_updated + other.inserted_or_updated,
            self.unchanged + other.unchanged,
            self.chunks + other.chunks,
            self.skipped + other.skipped,
        )


def ingest_work_directory(
    retriever: FanKnowledgeRetriever,
    work_dir: Path,
    *,
    maximum_chars: int = 1400,
    include_regular_chat: bool = False,
) -> IngestionSummary:
    summary = IngestionSummary()
    if not work_dir.is_dir():
        return summary
    for job_dir in sorted(path for path in work_dir.iterdir() if path.is_dir()):
        metadata_path = job_dir / "source.info.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = _read_json_object(metadata_path)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("skipping invalid work metadata %s: %s", metadata_path, exc)
            summary = summary.add(None)
            continue
        video_id = str(metadata.get("id") or job_dir.name).strip()
        title = str(metadata.get("title") or video_id).strip()
        source_url = str(metadata.get("webpage_url") or "").strip() or None
        published_at = _published_at(metadata)
        author = str(metadata.get("channel") or metadata.get("uploader") or "").strip()

        description = str(metadata.get("description") or "").strip()
        if description:
            document = KnowledgeDocument(
                document_id=_document_id("youtube_metadata", video_id),
                source_type="youtube_metadata",
                external_id=video_id,
                title=title,
                text=description,
                source_url=source_url,
                author=author or None,
                published_at=published_at,
                fetched_at=_now(),
                language=str(metadata.get("language") or "").strip() or None,
                metadata=_compact_youtube_metadata(metadata),
                reliability=0.85,
            )
            summary = summary.add(
                retriever.upsert_document(
                    document,
                    chunk_plain_text(description, maximum_chars=maximum_chars),
                )
            )

        summary = summary.merge(
            _ingest_youtube_chat(
                retriever,
                job_dir / "source.live_chat.json",
                video_id=video_id,
                title=title,
                source_url=source_url,
                author=author or None,
                published_at=published_at,
                maximum_chars=maximum_chars,
                include_regular_chat=include_regular_chat,
            )
        )
    return summary


def ingest_jsonl(
    retriever: FanKnowledgeRetriever,
    path: Path,
    *,
    maximum_chars: int = 1400,
) -> IngestionSummary:
    summary = IngestionSummary()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            result = ingest_document_mapping(
                retriever,
                value,
                fallback_external_id=f"{path.name}:{line_number}",
                maximum_chars=maximum_chars,
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("skipping %s:%d: %s", path, line_number, exc)
            summary = summary.add(None)
        else:
            summary = summary.add(result)
    return summary


def ingest_document_mapping(
    retriever: FanKnowledgeRetriever,
    value: object,
    *,
    fallback_external_id: str,
    maximum_chars: int = 1400,
) -> DocumentUpsertResult:
    document, chunks = document_from_mapping(
        value,
        fallback_external_id=fallback_external_id,
        maximum_chars=maximum_chars,
    )
    return retriever.upsert_document(document, chunks)


def document_from_mapping(
    value: object,
    *,
    fallback_external_id: str,
    maximum_chars: int = 1400,
) -> tuple[KnowledgeDocument, list[KnowledgeChunk]]:
    if not isinstance(value, dict):
        raise TypeError("knowledge JSONL record must be an object")
    source_type = str(value.get("source_type") or "external").strip()
    source_url = str(value.get("source_url") or "").strip() or None
    external_id = str(value.get("external_id") or fallback_external_id).strip()
    title = str(value.get("title") or external_id).strip()
    text = str(value.get("text") or value.get("raw_text") or "").strip()
    raw_chunks = value.get("chunks")
    if isinstance(raw_chunks, list):
        chunks = []
        for ordinal, item in enumerate(raw_chunks):
            if not isinstance(item, dict) or not str(item.get("text") or "").strip():
                continue
            chunks.append(
                KnowledgeChunk(
                    ordinal=ordinal,
                    text=str(item["text"]).strip(),
                    start_seconds=_optional_float(item.get("start_seconds")),
                    end_seconds=_optional_float(item.get("end_seconds")),
                    speaker=str(item.get("speaker") or "").strip() or None,
                    language=str(item.get("language") or "").strip() or None,
                )
            )
        if not text:
            text = "\n".join(chunk.text for chunk in chunks)
    else:
        chunks = chunk_plain_text(text, maximum_chars=maximum_chars)
    document = KnowledgeDocument(
        document_id=str(value.get("document_id") or "").strip()
        or _document_id(source_type, external_id),
        source_type=source_type,
        external_id=external_id,
        title=title,
        text=text,
        source_url=source_url,
        author=str(value.get("author") or "").strip() or None,
        published_at=str(value.get("published_at") or "").strip() or None,
        fetched_at=str(value.get("fetched_at") or "").strip() or _now(),
        language=str(value.get("language") or "").strip() or None,
        metadata=value.get("metadata")
        if isinstance(value.get("metadata"), dict)
        else {},
        reliability=float(value.get("reliability", 0.8)),
    )
    return document, chunks


def download_youtube_subtitles(
    urls: list[str],
    cache_dir: Path,
    download: DownloadConfig,
    *,
    languages: tuple[str, ...] = ("ja.*", "ja", "en.*", "en"),
    playlist_end: int = 100,
    batch_size: int = 10,
    batch_workers: int = 4,
    after_batch: Callable[[tuple[str, ...]], None] | None = None,
    known_video_ids: set[str] | None = None,
) -> tuple[str, ...]:
    if not urls:
        return ()
    yt_dlp = require_command("yt-dlp")
    cache_dir.mkdir(parents=True, exist_ok=True)
    common = [
        yt_dlp,
        "--js-runtimes",
        "node",
        "--remote-components",
        "ejs:github",
        "--extractor-args",
        "youtube:player_client=web_creator",
        "--concurrent-fragments",
        str(download.concurrent_fragments),
    ]
    if download.cookies_from_browser:
        common.extend(["--cookies-from-browser", download.cookies_from_browser])
    if download.cookies_file:
        common.extend(["--cookies", download.cookies_file])

    listing_command = [
        *common,
        "--flat-playlist",
        "--dump-json",
        "--playlist-end",
        str(playlist_end),
        *urls,
    ]
    if known_video_ids is None:
        listing_lines = subprocess.run(
            listing_command,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    else:
        listing_lines = _incremental_youtube_listing(listing_command, known_video_ids)
    video_urls: list[str] = []
    for line in listing_lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        video_id = str(item.get("id") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            continue
        if item.get("live_status") not in (None, "not_live", "was_live"):
            continue
        if item.get("availability") not in (None, "public", "unlisted"):
            continue
        if known_video_ids is not None and video_id in known_video_ids:
            logger.info("reached previously ingested YouTube video %s", video_id)
            break
        video_urls.append(f"https://www.youtube.com/watch?v={video_id}")
    video_urls = list(dict.fromkeys(video_urls))
    logger.info(
        "found %d public YouTube video(s) in %d source(s)",
        len(video_urls),
        len(urls),
    )

    command = [
        *common,
        "--ignore-errors",
        "--ignore-no-formats-error",
        "--no-progress",
        "--download-archive",
        str(cache_dir / ".download-archive"),
        "--force-write-archive",
        "--skip-download",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs",
        ",".join((*languages, "live_chat")),
        "--sub-format",
        "srt/best",
        "--convert-subs",
        "srt",
        "--output",
        str(cache_dir / "%(id)s" / "source.%(ext)s"),
    ]
    logger.info("downloading knowledge assets for %d video(s)", len(video_urls))
    batches = [
        tuple(video_urls[offset : offset + batch_size])
        for offset in range(0, len(video_urls), batch_size)
    ]
    with ThreadPoolExecutor(max_workers=batch_workers) as executor:
        futures = {
            executor.submit(subprocess.run, [*command, *batch], check=True): batch
            for batch in batches
        }
        for future in as_completed(futures):
            future.result()
            if after_batch is not None:
                after_batch(tuple(url.rsplit("=", 1)[-1] for url in futures[future]))
    return tuple(url.rsplit("=", 1)[-1] for url in video_urls)


def _incremental_youtube_listing(
    command: list[str], known_video_ids: set[str]
) -> list[str]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    lines: list[str] = []
    reached_known = False
    for line in process.stdout:
        lines.append(line)
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(value.get("id") or "") in known_video_ids:
            reached_known = True
            process.terminate()
            break
    _stdout, stderr = process.communicate()
    if not reached_known and process.returncode:
        raise RuntimeError(
            f"incremental YouTube listing failed ({process.returncode}): "
            f"{(stderr or '').strip()[-500:]}"
        )
    return lines


def ingest_youtube_cache(
    retriever: FanKnowledgeRetriever,
    cache_dir: Path,
    *,
    target_seconds: float = 45.0,
    maximum_chars: int = 1400,
    language_priority: tuple[str, ...] = ("ja", "ja-orig", "en"),
    include_regular_chat: bool = False,
    video_ids: set[str] | None = None,
) -> IngestionSummary:
    summary = IngestionSummary()
    for metadata_path in sorted(cache_dir.glob("*/source.info.json")):
        metadata = _read_json_object(metadata_path)
        directory = metadata_path.parent
        video_id = str(metadata.get("id") or directory.name).strip()
        if metadata.get("_type") == "playlist":
            summary = summary.add(None)
            continue
        if video_ids is not None and video_id not in video_ids:
            continue
        title = str(metadata.get("title") or video_id).strip()
        source_url = str(metadata.get("webpage_url") or "").strip() or None
        author = str(metadata.get("channel") or metadata.get("uploader") or "").strip()
        published_at = _published_at(metadata)
        description = str(metadata.get("description") or "").strip()
        metadata_text = description or title
        metadata_document = KnowledgeDocument(
            document_id=_document_id("youtube_metadata", video_id),
            source_type="youtube_metadata",
            external_id=video_id,
            title=title,
            text=metadata_text,
            source_url=source_url,
            author=author or None,
            published_at=published_at,
            fetched_at=_now(),
            metadata=_compact_youtube_metadata(metadata),
            reliability=0.85,
        )
        summary = summary.add(
            retriever.upsert_document(
                metadata_document,
                chunk_plain_text(metadata_text, maximum_chars=maximum_chars),
            )
        )

        subtitle_path, language, manual = _select_youtube_subtitle(
            directory, metadata, language_priority
        )
        if subtitle_path is not None:
            try:
                cues = _deduplicate_caption_cues(read_subtitles(subtitle_path))
            except (OSError, ValueError) as exc:
                logger.warning(
                    "skipping invalid YouTube subtitle %s: %s", subtitle_path, exc
                )
                summary = summary.add(None)
            else:
                text = "\n".join(cue.text for cue in cues)
                source_type = (
                    "youtube_manual_subtitle" if manual else "youtube_auto_subtitle"
                )
                document = KnowledgeDocument(
                    document_id=_document_id(source_type, f"{video_id}:{language}"),
                    source_type=source_type,
                    external_id=f"{video_id}:{language}",
                    title=title,
                    text=text,
                    source_url=source_url,
                    author=author or None,
                    published_at=published_at,
                    fetched_at=_now(),
                    language=language,
                    metadata=_compact_youtube_metadata(metadata),
                    reliability=0.9 if manual else 0.68,
                )
                summary = summary.add(
                    retriever.upsert_document(
                        document,
                        chunk_timed_cues(
                            cues,
                            target_seconds=target_seconds,
                            maximum_chars=maximum_chars,
                        ),
                    )
                )
        summary = summary.merge(
            _ingest_youtube_chat(
                retriever,
                directory / "source.live_chat.json",
                video_id=video_id,
                title=title,
                source_url=source_url,
                author=author or None,
                published_at=published_at,
                maximum_chars=maximum_chars,
                include_regular_chat=include_regular_chat,
            )
        )
    return summary


def chunk_youtube_chat(
    messages: list[YouTubeChatMessage],
    *,
    maximum_chars: int = 1400,
    target_seconds: float = 60.0,
) -> tuple[list[KnowledgeChunk], list[KnowledgeChunk]]:
    regular = [message for message in messages if not message.amount]
    paid = [message for message in messages if message.amount]
    return (
        _chunk_chat_messages(regular, maximum_chars, target_seconds),
        _chunk_chat_messages(paid, maximum_chars, target_seconds),
    )


def _ingest_youtube_chat(
    retriever: FanKnowledgeRetriever,
    path: Path,
    *,
    video_id: str,
    title: str,
    source_url: str | None,
    author: str | None,
    published_at: str | None,
    maximum_chars: int,
    include_regular_chat: bool,
) -> IngestionSummary:
    if not path.is_file():
        return IngestionSummary()
    messages = read_youtube_live_chat(path)
    chat_chunks, paid_chunks = chunk_youtube_chat(messages, maximum_chars=maximum_chars)
    summary = IngestionSummary()
    sources = [("youtube_super_chat", "super-chat", paid_chunks, 0.82)]
    if include_regular_chat:
        sources.insert(0, ("youtube_live_chat", "chat", chat_chunks, 0.52))
    for source_type, suffix, chunks, reliability in sources:
        if not chunks:
            continue
        document = KnowledgeDocument(
            document_id=_document_id(source_type, video_id),
            source_type=source_type,
            external_id=f"{video_id}:{suffix}",
            title=f"{title} ({suffix})",
            text="\n".join(chunk.text for chunk in chunks),
            source_url=source_url,
            author=author,
            published_at=published_at,
            fetched_at=_now(),
            language="mixed",
            metadata={"video_id": video_id},
            reliability=reliability,
        )
        summary = summary.add(retriever.upsert_document(document, chunks))
    return summary


def _chunk_chat_messages(
    messages: list[YouTubeChatMessage],
    maximum_chars: int,
    target_seconds: float,
) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    current: list[YouTubeChatMessage] = []
    characters = 0

    def flush() -> None:
        nonlocal current, characters
        if not current:
            return
        chunks.append(
            KnowledgeChunk(
                ordinal=len(chunks),
                text="\n".join(_chat_message_text(message) for message in current),
                start_seconds=current[0].offset_seconds,
                end_seconds=current[-1].offset_seconds,
                language="mixed",
            )
        )
        current = []
        characters = 0

    for message in messages:
        line = _chat_message_text(message)
        if current and (
            characters + len(line) > maximum_chars
            or message.offset_seconds - current[0].offset_seconds >= target_seconds
        ):
            flush()
        current.append(message)
        characters += len(line)
    flush()
    return chunks


def _chat_message_text(message: YouTubeChatMessage) -> str:
    prefix = message.author or "viewer"
    if message.amount:
        prefix = f"SC {message.amount} {prefix}"
    elif message.membership:
        prefix = f"member {prefix}"
    return f"{prefix}: {message.text}"


def chunk_timed_cues(
    cues: list[Cue],
    *,
    target_seconds: float = 45.0,
    maximum_chars: int = 1400,
    maximum_gap_seconds: float = 15.0,
) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    current: list[Cue] = []
    characters = 0

    def flush() -> None:
        nonlocal current, characters
        if not current:
            return
        speakers = {cue.speaker for cue in current if cue.speaker}
        languages = {cue.language for cue in current if cue.language}
        chunks.append(
            KnowledgeChunk(
                ordinal=len(chunks),
                text="\n".join(cue.text.strip() for cue in current),
                start_seconds=current[0].start,
                end_seconds=current[-1].end,
                speaker=next(iter(speakers)) if len(speakers) == 1 else None,
                language=next(iter(languages)) if len(languages) == 1 else "mixed",
            )
        )
        current = []
        characters = 0

    for cue in cues:
        text = cue.text.strip()
        if not text:
            continue
        speaker_changed = bool(
            current
            and current[-1].speaker
            and cue.speaker
            and current[-1].speaker != cue.speaker
        )
        gap = cue.start - current[-1].end if current else 0.0
        if current and (
            speaker_changed
            or gap > maximum_gap_seconds
            or characters + len(text) > maximum_chars
        ):
            flush()
        current.append(cue)
        characters += len(text)
        if current[-1].end - current[0].start >= target_seconds:
            flush()
    flush()
    return chunks


def chunk_plain_text(text: str, *, maximum_chars: int = 1400) -> list[KnowledgeChunk]:
    paragraphs = [
        " ".join(value.split()) for value in text.splitlines() if value.strip()
    ]
    chunks: list[KnowledgeChunk] = []
    current: list[str] = []
    characters = 0
    for paragraph in paragraphs:
        while len(paragraph) > maximum_chars:
            if current:
                chunks.append(KnowledgeChunk(len(chunks), "\n".join(current)))
                current = []
                characters = 0
            chunks.append(KnowledgeChunk(len(chunks), paragraph[:maximum_chars]))
            paragraph = paragraph[maximum_chars:]
        if current and characters + len(paragraph) > maximum_chars:
            chunks.append(KnowledgeChunk(len(chunks), "\n".join(current)))
            current = []
            characters = 0
        if paragraph:
            current.append(paragraph)
            characters += len(paragraph)
    if current:
        chunks.append(KnowledgeChunk(len(chunks), "\n".join(current)))
    return chunks


def _select_youtube_subtitle(
    directory: Path,
    metadata: dict[str, object],
    language_priority: tuple[str, ...],
) -> tuple[Path | None, str, bool]:
    candidates = list(directory.glob("source.*.srt"))
    manual_languages = set(_mapping_keys(metadata.get("subtitles"))) - {"live_chat"}

    def details(path: Path) -> tuple[int, int, str]:
        language = path.name.removeprefix("source.").removesuffix(".srt")
        base = language.split("-")[0]
        manual = language in manual_languages or base in manual_languages
        try:
            priority = language_priority.index(language)
        except ValueError:
            try:
                priority = language_priority.index(base)
            except ValueError:
                priority = len(language_priority)
        return (0 if manual else 1, priority, language)

    if not candidates:
        return None, "", False
    selected = min(candidates, key=lambda path: details(path)[:3])
    _, _, language = details(selected)
    manual = language in manual_languages or language.split("-")[0] in manual_languages
    return selected, language, manual


def _deduplicate_caption_cues(cues: list[Cue]) -> list[Cue]:
    output: list[Cue] = []
    for cue in cues:
        text = " ".join(cue.text.split())
        if not text:
            continue
        value = Cue(cue.start, cue.end, text, language=cue.language)
        if output and text == output[-1].text:
            continue
        if (
            output
            and cue.start <= output[-1].end + 0.25
            and text.startswith(output[-1].text)
        ):
            previous = output[-1]
            output[-1] = Cue(previous.start, cue.end, text, language=cue.language)
            continue
        output.append(value)
    return output


def _mapping_keys(value: object) -> list[str]:
    return [str(key) for key in value] if isinstance(value, dict) else []


def _read_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    return value


def _compact_youtube_metadata(metadata: dict[str, object]) -> dict[str, object]:
    keys = (
        "id",
        "channel_id",
        "channel",
        "uploader_id",
        "upload_date",
        "timestamp",
        "release_timestamp",
        "duration",
        "live_status",
        "tags",
        "categories",
    )
    return {
        key: metadata[key] for key in keys if metadata.get(key) not in (None, "", [])
    }


def _published_at(metadata: dict[str, object]) -> str | None:
    for key in ("release_timestamp", "timestamp"):
        value = metadata.get(key)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, UTC).isoformat()
    value = metadata.get("upload_date")
    if isinstance(value, str) and len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}T00:00:00+00:00"
    return None


def _document_id(source_type: str, external_id: str) -> str:
    digest = hashlib.sha256(f"{source_type}\0{external_id}".encode()).hexdigest()[:24]
    return f"document:{digest}"


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _now() -> str:
    return datetime.now(UTC).isoformat()
