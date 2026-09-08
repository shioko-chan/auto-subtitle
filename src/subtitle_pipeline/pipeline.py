from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

from .asr import (
    read_cue_evidence,
    read_cue_sidecar,
    transcribe_with_qwen,
)
from .asr_correction import correct_asr_windows, entities_from_context
from .bilibili_comments import (
    build_song_setlist_comment,
    create_comment_task,
    publish_comment_task,
)
from .chat_context import CurrentVideoChatIndex, remove_youtube_chat_files
from .config import AppConfig, LLMConfig, llm_api_key
from .fan_knowledge import (
    FanKnowledgeRetriever,
    KnowledgeHit,
    KnowledgeQuery,
    video_date_from_metadata,
)
from .knowledge_ingestion import ingest_youtube_top_comments
from .knowledge_update import update_knowledge_if_stale
from .local_llm_server import LocalLLMServer
from .media import download_youtube, render_subtitles, subtitle_layout
from .song_identification import (
    SongIdentificationResult,
    arbitrate_verified_lyrics,
    identify_and_align_songs,
    select_ocr_song_titles,
    split_aligned_song_cues,
    translate_aligned_song_lyrics,
)
from .speakers import load_character_styles
from .subtitles import (
    Cue,
    clean_non_speech_markers,
    filter_long_cue_pairs,
    read_subtitles,
    trim_overlapping_cues,
    write_srt,
)
from .telemetry import pipeline_metrics, stage_metrics
from .translate import OpenAICompatibleTranslator
from .upload import upload_to_bilibili

_BUILTIN_GLOSSARY_FILES = (
    "glossaries/bang-dream.json",
    "glossaries/yumemita.json",
    "glossaries/our-notes.json",
)
_JAPANESE_SCRIPT_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_BEIJING_TIME = ZoneInfo("Asia/Shanghai")
_DEEPSEEK_BLOCKED_BEIJING_HOURS = ((9, 12), (14, 18))
_MAX_SUBTITLE_DURATION_SECONDS = 30.0


@dataclass(frozen=True)
class PipelineResult:
    job_dir: Path
    source_video: Path
    source_subtitle: Path
    translated_subtitle: Path
    translated_metadata: Path
    rendered_video: Path
    uploaded: bool
    bilibili_aid: int | None
    bilibili_bvid: str | None


def run_pipeline(
    url: str,
    config: AppConfig,
    *,
    upload_override: bool | None = None,
) -> PipelineResult:
    url = normalize_youtube_url(url)
    job_id = youtube_video_id(url)
    job_dir = config.work_dir.resolve() / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    with _job_log(job_dir / "run.log"):
        logging.info("job directory: %s", job_dir)
        _wait_for_deepseek_task_window(config.llm)
        local_llm_server = LocalLLMServer(config.llm, job_dir / "local-llm-server.log")
        fan_knowledge = (
            FanKnowledgeRetriever(
                Path(config.fan_knowledge.database_path).expanduser(),
                audit_path=job_dir / "fan-knowledge-audit.jsonl",
                embedding_model=config.fan_knowledge.embedding_model,
                vector_index_path=Path(
                    config.fan_knowledge.vector_index_path
                ).expanduser(),
                vector_minimum_score=config.fan_knowledge.vector_minimum_score,
                reranker_model=config.fan_knowledge.reranker_model,
                reranker_minimum_score=(config.fan_knowledge.reranker_minimum_score),
            )
            if config.fan_knowledge.enabled
            else None
        )

        try:
            with (
                pipeline_metrics(job_dir / "performance.json"),
                stage_metrics("pipeline.total"),
            ):
                return _run_pipeline_stages(
                    url,
                    config,
                    job_dir,
                    upload_override=upload_override,
                    local_llm_server=local_llm_server,
                    fan_knowledge=fan_knowledge,
                )
        finally:
            local_llm_server.stop()
            if fan_knowledge is not None:
                fan_knowledge.close()


@contextmanager
def _job_log(path: Path) -> Iterator[None]:
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    previous_level = root.level
    if previous_level > logging.INFO:
        root.setLevel(logging.INFO)
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)
        handler.close()


def _deepseek_task_delay(config: LLMConfig, now: datetime) -> float:
    hostname = urlsplit(config.base_url).hostname
    normalized = hostname.lower().rstrip(".") if hostname else ""
    if normalized != "deepseek.com" and not normalized.endswith(".deepseek.com"):
        return 0.0
    current = now.replace(tzinfo=UTC) if now.tzinfo is None else now
    current = current.astimezone(_BEIJING_TIME)
    if current.weekday() >= 5:
        return 0.0
    for start_hour, end_hour in _DEEPSEEK_BLOCKED_BEIJING_HOURS:
        if start_hour <= current.hour < end_hour:
            resume_at = current.replace(
                hour=end_hour,
                minute=0,
                second=0,
                microsecond=0,
            )
            return max(0.0, (resume_at - current).total_seconds())
    return 0.0


def _wait_for_deepseek_task_window(config: LLMConfig) -> None:
    current = datetime.now(UTC)
    delay = _deepseek_task_delay(config, current)
    if delay <= 0:
        return
    resume_at = datetime.fromtimestamp(current.timestamp() + delay, UTC)
    resume_at_beijing = resume_at.astimezone(_BEIJING_TIME)
    logging.info(
        "DeepSeek Beijing weekday blocked window active; pausing task for %.0fs "
        "until %s (%s UTC)",
        delay,
        resume_at_beijing.isoformat(timespec="seconds"),
        resume_at.isoformat(timespec="seconds"),
    )
    time.sleep(delay)


def _run_pipeline_stages(
    url: str,
    config: AppConfig,
    job_dir: Path,
    *,
    upload_override: bool | None,
    local_llm_server: LocalLLMServer,
    fan_knowledge: FanKnowledgeRetriever | None,
) -> PipelineResult:
    if fan_knowledge is not None:
        with stage_metrics("pipeline.knowledge_update"):
            update_knowledge_if_stale(config, fan_knowledge)
        fan_knowledge.release_models()
    with stage_metrics("pipeline.download"):
        downloaded = download_youtube(url, job_dir, config.download)

    current_chat: CurrentVideoChatIndex | None = None
    if downloaded.chat_replay is not None:
        with stage_metrics("pipeline.current_video_chat"):
            try:
                current_chat = CurrentVideoChatIndex.from_path(
                    downloaded.chat_replay,
                    lookback_seconds=(
                        config.fan_knowledge.current_video_chat_lookback_seconds
                    ),
                    lookahead_seconds=(
                        config.fan_knowledge.current_video_chat_lookahead_seconds
                    ),
                    audit_path=job_dir / "current-video-chat-audit.jsonl",
                )
                logging.info(
                    "loaded %d current-video chat messages for time-local context",
                    len(current_chat),
                )
            except (OSError, UnicodeError) as exc:
                logging.warning("could not read current-video chat replay: %s", exc)
            finally:
                remove_youtube_chat_files(job_dir)

    translation_context = _translation_context(
        downloaded.metadata, config.translation.glossary_files
    )
    video_date = video_date_from_metadata(downloaded.metadata)
    current_video_id = str(downloaded.metadata.get("id") or "").strip() or None
    if fan_knowledge is not None:
        if downloaded.comments is not None and current_video_id is not None:
            comment_result = ingest_youtube_top_comments(
                fan_knowledge,
                downloaded.comments,
                video_id=current_video_id,
                title=str(downloaded.metadata.get("title") or current_video_id),
                source_url=url,
                published_at=video_date,
                minimum_likes=config.download.top_comment_min_likes,
                maximum_comments=config.download.top_comment_background_limit,
            )
            if comment_result is not None:
                logging.info(
                    "indexed %d high-like YouTube comments as current-video background",
                    comment_result.chunk_count,
                )
        imported = fan_knowledge.ingest_translation_context(translation_context)
        logging.info("indexed %d curated fan-knowledge records", imported)
    japanese_single_word_list = _japanese_single_word_list(translation_context)
    if japanese_single_word_list:
        logging.info(
            "using %d Japanese glossary names as forced-aligner single words",
            len(japanese_single_word_list),
        )
    translator = OpenAICompatibleTranslator(
        config.llm,
        config.translation,
        llm_api_key(config.llm),
        audit_path=job_dir / "llm-audit.jsonl",
    )
    asr_entities = entities_from_context(translation_context)
    layout = subtitle_layout(downloaded.video, config.render)
    japanese_guidance_units = layout.max_line_units * 1.25
    early_song_result = SongIdentificationResult([], [])

    def process_song_cues(singing_cues: list[Cue]) -> list[Cue]:
        nonlocal early_song_result

        def select_titles(candidate_sets):
            try:
                if config.llm.local_server_enabled:
                    local_llm_server.start()
                return select_ocr_song_titles(
                    candidate_sets,
                    video_title=str(downloaded.metadata.get("title") or ""),
                    request=translator.request,
                    model=config.llm.model,
                    json_mode=config.llm.json_mode,
                    thinking=config.llm.thinking,
                )
            except Exception as exc:  # noqa: BLE001 - OCR remains optional evidence.
                logging.warning(
                    "LLM song-title selection failed; continuing without OCR "
                    "titles: %s",
                    exc,
                )
                return [None for _ in candidate_sets]

        with stage_metrics("pipeline.song_identification"):
            if config.song_identification.enabled:
                early_song_result = identify_and_align_songs(
                    downloaded.video,
                    singing_cues,
                    downloaded.metadata,
                    job_dir,
                    config.song_identification,
                    source_maximum_units=japanese_guidance_units,
                    select_ocr_titles=select_titles,
                )
            else:
                early_song_result = SongIdentificationResult(singing_cues, [])
        return early_song_result.corrected_cues

    def correct_asr_text(records: list[dict[str, object]]) -> list[dict[str, object]]:
        if current_chat:
            records = [
                {
                    **record,
                    "chat_text": current_chat.evidence(
                        float(record.get("core_start") or 0.0),
                        float(record.get("core_end") or 0.0),
                        str(record.get("text") or ""),
                        stage="asr_correction",
                        target_id=record.get("window_id", index),
                    ),
                }
                for index, record in enumerate(records)
            ]
        if config.llm.local_server_enabled:
            with stage_metrics("pipeline.local_llm_asr_correction_startup"):
                local_llm_server.start()

        def retrieve_asr_knowledge(
            record: dict[str, object], text: str
        ) -> list[KnowledgeHit]:
            if fan_knowledge is None:
                return []
            query = KnowledgeQuery(
                text=text,
                speaker=str(record.get("speaker") or "") or None,
                video_date=video_date,
                ocr_text=str(record.get("ocr_text") or ""),
                chat_text=str(record.get("chat_text") or ""),
                exclude_video_id=current_video_id,
                top_k=config.fan_knowledge.top_k_asr,
            )
            values = [
                *fan_knowledge.retrieve_asr_term_references(query)[:2],
                *fan_knowledge.retrieve_background(query),
            ]
            return list({hit.record_id: hit for hit in values}.values())[
                : config.fan_knowledge.top_k_asr
            ]

        try:
            return correct_asr_windows(
                records,
                entities=asr_entities,
                request=translator.request,
                model=config.llm.model,
                cache_path=job_dir / "asr-correction-cache.json",
                audit_path=job_dir / "asr-correction-audit.jsonl",
                window_chars=config.asr_correction.window_chars,
                max_tokens=config.asr_correction.max_tokens,
                retrieve_knowledge=retrieve_asr_knowledge,
            )
        finally:
            if fan_knowledge is not None:
                fan_knowledge.release_models()

    with stage_metrics("pipeline.audio_and_asr"):
        source_subtitle = transcribe_with_qwen(
            downloaded.video,
            job_dir / "source.qwen3-asr.srt",
            config.asr,
            config.audio_analysis,
            downloaded.metadata,
            japanese_single_word_list,
            correct_asr_text,
            process_song_cues,
        )

    sidecar = source_subtitle.with_suffix(".cues.json")
    cues = (
        read_cue_sidecar(sidecar)
        if sidecar.is_file()
        else read_subtitles(source_subtitle)
    )
    asr_evidence = read_cue_evidence(sidecar) if sidecar.is_file() else []
    original_cue_count = len(cues)
    cues = clean_non_speech_markers(cues)
    cues, song_arbitration = arbitrate_verified_lyrics(
        cues, list(early_song_result.verified_lyric_spans)
    )
    logging.info(
        "non-speech marker cleanup: %d source cues -> %d spoken cues",
        original_cue_count,
        len(cues),
    )
    source_title = str(downloaded.metadata.get("title") or "YouTube video")
    source_description = str(downloaded.metadata.get("description") or "")
    youtube_context = _youtube_metadata_context(downloaded.metadata)
    if asr_evidence:
        translation_context = {
            **translation_context,
            "asr_evidence": asr_evidence,
        }
    if translation_context.get("franchises"):
        names = [
            item["name"]
            for item in translation_context["franchises"]
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        ]
        logging.info("using translation glossary: %s", ", ".join(names))
    song_result = SongIdentificationResult(
        cues,
        [
            {**report, "post_asr_arbitration": song_arbitration}
            for report in early_song_result.reports
        ],
        early_song_result.verified_lyric_spans,
    )
    if config.llm.local_server_enabled:
        with stage_metrics("pipeline.local_llm_startup"):
            local_llm_server.start()
    else:
        local_llm_server.start()
    with stage_metrics("pipeline.song_lyrics_translation"):
        if song_result.reports:
            song_result = translate_aligned_song_lyrics(
                song_result,
                config.song_identification,
                translator.translate_lyrics,
                translation_context,
                config.llm.model,
            )
    song_result = split_aligned_song_cues(song_result, japanese_guidance_units)
    cues = song_result.corrected_cues
    if song_result.reports:
        write_srt(cues, job_dir / "source.lyrics-corrected.srt")
        translation_context = {
            **translation_context,
            "identified_songs": song_result.reports,
        }
        logging.info(
            "song identification produced %d search-group reports",
            len(song_result.reports),
        )
    with stage_metrics("pipeline.llm_cue_segmentation"):
        segmented = translator.segment_cues(
            cues,
            config.segmentation,
            max_line_units=layout.max_line_units,
            cache_path=job_dir / "cue-segmentation-cache.json",
            audit_path=job_dir / "local-segmentation.json",
        )
    retrieve_translation_knowledge = None
    if fan_knowledge is not None:

        def retrieve_translation_knowledge(selected, chat_text):
            return _retrieve_translation_knowledge(
                fan_knowledge,
                selected,
                video_date=video_date,
                top_k=config.fan_knowledge.top_k_translation,
                query_chars=config.fan_knowledge.translation_query_chars,
                exclude_video_id=current_video_id,
                chat_text=chat_text,
            )

    retrieve_translation_chat = (
        (
            lambda selected: current_chat.evidence(
                min(cue.start for cue in selected),
                max(cue.end for cue in selected),
                "\n".join(cue.text for cue in selected),
                stage="translation",
                target_id=(
                    f"{min(cue.start for cue in selected):.3f}-"
                    f"{max(cue.end for cue in selected):.3f}"
                ),
            )
        )
        if current_chat
        else None
    )
    with stage_metrics("pipeline.llm_cue_translation"):
        translated = translator.translate_segmented_cues(
            segmented,
            translation_context=translation_context,
            cache_path=job_dir / "cue-translation-cache.json",
            audit_path=job_dir / "translation-audit.jsonl",
            retrieve_knowledge=retrieve_translation_knowledge,
            retrieve_chat=retrieve_translation_chat,
        )
    logging.info(
        "staged cue segmentation and ASR-aware translation: "
        "%d aligned cues -> %d subtitle cues",
        original_cue_count,
        len(segmented),
    )
    overlap_count = sum(
        current.end > following.start
        for current, following in zip(segmented, segmented[1:])
    )
    cues = trim_overlapping_cues(segmented)
    translated = trim_overlapping_cues(translated)
    logging.info("timing overlap cleanup: adjusted %d cues", overlap_count)
    cues, translated, dropped_long_cues = filter_long_cue_pairs(
        cues,
        translated,
        maximum_seconds=_MAX_SUBTITLE_DURATION_SECONDS,
    )
    for cue in dropped_long_cues:
        logging.warning(
            "dropping overlong subtitle cue start=%.3fs end=%.3fs duration=%.3fs "
            "speaker=%s text=%r",
            cue.start,
            cue.end,
            cue.end - cue.start,
            cue.speaker or "unknown",
            cue.text[:120],
        )
    if dropped_long_cues:
        logging.warning(
            "dropped %d subtitle cue(s) longer than %.1fs",
            len(dropped_long_cues),
            _MAX_SUBTITLE_DURATION_SECONDS,
        )
    write_srt(cues, job_dir / "source.semantic.srt")

    translated_path = job_dir / "translated.zh-CN.srt"
    write_srt(translated, translated_path)

    subtitle_evidence = _subtitle_evidence(
        cues, config.translation.metadata_subtitle_max_chars
    )
    ip_aliases = _load_optional_json_object(
        config.translation.ip_aliases_file, "IP aliases"
    )
    tag_catalog = _load_optional_json_object(
        config.upload.tag_catalog_file, "Bilibili tag catalog"
    )
    title, description = source_title, source_description
    content_summary = ""
    generated_tags: list[str] = []
    with stage_metrics("pipeline.metadata_translation"):
        if config.translation.translate_metadata:
            logging.info("translating video title and description and generating tags")
            title, description, content_summary, generated_tags = (
                translator.translate_metadata(
                    source_title,
                    source_description,
                    youtube_context=youtube_context,
                    subtitle_evidence=subtitle_evidence,
                    ip_aliases=ip_aliases,
                    bilibili_tag_catalog=tag_catalog,
                    translation_context=translation_context,
                )
            )
    generated_tags, tag_catalog_matches = _canonicalize_catalog_tags(
        generated_tags, tag_catalog
    )
    upload_tags = _merge_tags(
        config.upload.tags, generated_tags, config.upload.max_tags
    )
    metadata_path = job_dir / "translated.metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "source_title": source_title,
                "source_description": source_description,
                "translated_title": title,
                "translated_description": description,
                "youtube_context": youtube_context,
                "translation_glossaries": [
                    item["name"] for item in translation_context.get("franchises", [])
                ],
                "identified_songs": song_result.reports,
                "content_summary": content_summary,
                "generated_tags": generated_tags,
                "tag_catalog_matches": tag_catalog_matches,
                "upload_tags": upload_tags,
                "source_url": url,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if config.llm.local_server_enabled:
        with stage_metrics("pipeline.local_llm_shutdown"):
            local_llm_server.stop()
    else:
        local_llm_server.stop()
    rendered_path = job_dir / "translated.mp4"
    with stage_metrics("pipeline.render"):
        render_subtitles(
            downloaded.video,
            translated_path,
            rendered_path,
            config.render,
            cues=translated,
            character_styles=load_character_styles(
                config.audio_analysis.character_styles_file
            ),
        )

    should_upload = (
        config.upload.enabled if upload_override is None else upload_override
    )
    submission = None
    comment_task = None
    if should_upload:
        with stage_metrics("pipeline.upload"):
            submission = upload_to_bilibili(
                rendered_path,
                title=title,
                description=description,
                source_url=url,
                tags=upload_tags,
                config=config.upload,
            )
            setlist = (
                build_song_setlist_comment(
                    downloaded.metadata,
                    song_result.reports,
                    _youtube_comments(downloaded.comments),
                    minimum_comment_likes=config.download.top_comment_min_likes,
                )
                if config.upload.song_setlist_comment
                else None
            )
            if (
                setlist is not None
                and submission.aid is not None
                and submission.bvid is not None
            ):
                comment_task = create_comment_task(
                    job_dir,
                    aid=submission.aid,
                    bvid=submission.bvid,
                    message=setlist,
                    source_url=url,
                )
        if comment_task is not None:
            with stage_metrics("pipeline.bilibili_comment"):
                comment_status = publish_comment_task(comment_task, config.upload)
            if comment_status not in {"posted", "already_exists"}:
                logging.warning(
                    "Bilibili setlist comment publication finished with status=%s",
                    comment_status,
                )

    result = PipelineResult(
        job_dir=job_dir,
        source_video=downloaded.video,
        source_subtitle=source_subtitle,
        translated_subtitle=translated_path,
        translated_metadata=metadata_path,
        rendered_video=rendered_path,
        uploaded=should_upload,
        bilibili_aid=submission.aid if submission is not None else None,
        bilibili_bvid=submission.bvid if submission is not None else None,
    )
    _write_manifest(result, url, title)
    return result


def normalize_youtube_url(url: str) -> str:
    candidate = url.strip()
    for escaped, literal in ((r"\?", "?"), (r"\=", "="), (r"\&", "&")):
        candidate = candidate.replace(escaped, literal)
    if "\\" in candidate:
        raise ValueError("YouTube URL contains an unexpected backslash")
    parsed = urlsplit(candidate)
    hostname = (parsed.hostname or "").lower()
    supported_host = (
        hostname == "youtu.be"
        or hostname == "youtube.com"
        or hostname.endswith(".youtube.com")
        or hostname == "youtube-nocookie.com"
        or hostname.endswith(".youtube-nocookie.com")
    )
    if parsed.scheme not in {"http", "https"} or not supported_host:
        raise ValueError(f"not a supported YouTube URL: {url}")
    if candidate != url:
        logging.info("normalized escaped YouTube URL: %s", candidate)
    return candidate


def youtube_video_id(url: str) -> str:
    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").lower()
    if hostname == "youtu.be":
        video_id = parsed.path.strip("/").split("/", 1)[0]
    elif parsed.path.rstrip("/") == "/watch":
        video_id = parse_qs(parsed.query).get("v", [""])[0]
    else:
        path_parts = parsed.path.strip("/").split("/")
        video_id = (
            path_parts[1]
            if len(path_parts) >= 2
            and path_parts[0] in {"embed", "live", "shorts"}
            else ""
        )
    if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
        raise ValueError(f"YouTube URL has no valid video ID: {url}")
    return video_id


def _merge_tags(configured: list[str], generated: list[str], limit: int) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in [*configured, *generated]:
        tag = value.strip().lstrip("#").replace(",", " ").strip()[:20]
        key = tag.casefold()
        if tag and key not in seen:
            seen.add(key)
            merged.append(tag)
        if len(merged) >= limit:
            break
    if not merged:
        raise ValueError("at least one Bilibili upload tag is required")
    return merged


def _youtube_metadata_context(metadata: dict[str, object]) -> dict[str, object]:
    keys = (
        "channel",
        "channel_id",
        "uploader",
        "uploader_id",
        "creator",
        "categories",
        "tags",
        "series",
        "season",
        "season_number",
        "episode",
        "episode_number",
        "playlist",
        "playlist_title",
        "artist",
        "track",
        "album",
        "language",
    )
    return {
        key: metadata[key] for key in keys if metadata.get(key) not in (None, "", [])
    }


def _youtube_comments(path: Path | None) -> list[object]:
    if path is None or not path.is_file():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("cannot read downloaded YouTube comments: %s", exc)
        return []
    comments = value.get("comments") if isinstance(value, dict) else None
    return comments if isinstance(comments, list) else []


def _subtitle_evidence(cues: list[Cue], limit: int) -> str:
    if limit <= 0 or not cues:
        return ""
    lines = [cue.text.replace("\n", " ").strip() for cue in cues if cue.text.strip()]
    full = "\n".join(lines)
    if len(full) <= limit:
        return full
    third = max(1, limit // 3)
    middle = len(full) // 2
    return "\n...\n".join(
        (
            full[:third],
            full[max(0, middle - third // 2) : middle + third // 2],
            full[-third:],
        )
    )[:limit]


def _retrieve_translation_knowledge(
    retriever: FanKnowledgeRetriever,
    cues: list[Cue],
    *,
    video_date: str | None,
    top_k: int,
    query_chars: int,
    exclude_video_id: str | None,
    chat_text: str,
) -> list[KnowledgeHit]:
    hits_by_id: dict[str, KnowledgeHit] = {}
    term_hits_by_id: dict[str, KnowledgeHit] = {}
    remaining = query_chars
    chunk: list[str] = []
    chunk_chars = 0
    chunk_speaker: str | None = None

    def flush() -> None:
        nonlocal chunk, chunk_chars, chunk_speaker
        if not chunk:
            return
        query = KnowledgeQuery(
            text="\n".join(chunk),
            speaker=chunk_speaker,
            video_date=video_date,
            chat_text=chat_text,
            exclude_video_id=exclude_video_id,
            top_k=top_k,
        )
        for hit in retriever.retrieve_term_references(query):
            previous = term_hits_by_id.get(hit.record_id)
            if previous is None or hit.score.total > previous.score.total:
                term_hits_by_id[hit.record_id] = hit
        for hit in retriever.retrieve_background(query):
            previous = hits_by_id.get(hit.record_id)
            if previous is None or hit.score.total > previous.score.total:
                hits_by_id[hit.record_id] = hit
        chunk = []
        chunk_chars = 0
        chunk_speaker = None

    for cue in cues:
        text = cue.text.strip()
        if not text or remaining <= 0:
            continue
        text = text[:remaining]
        speaker = cue.speaker
        if chunk and (speaker != chunk_speaker or chunk_chars + len(text) > 2000):
            flush()
        if not chunk:
            chunk_speaker = speaker
        chunk.append(text)
        chunk_chars += len(text)
        remaining -= len(text)
    flush()
    terms = sorted(
        term_hits_by_id.values(),
        key=lambda hit: (-hit.score.total, hit.record_id),
    )[:top_k]
    background = sorted(
        (
            hit
            for record_id, hit in hits_by_id.items()
            if record_id not in term_hits_by_id
        ),
        key=lambda hit: (-hit.score.total, hit.record_id),
    )[:top_k]
    return [*terms, *background]


def _load_optional_json_object(path_value: str | None, label: str) -> dict[str, object]:
    if not path_value:
        return {}
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise ValueError(f"{label} file not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {label} file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} file must contain a JSON object: {path}")
    return value


def _translation_context(
    metadata: dict[str, object], configured_files: list[str]
) -> dict[str, object]:
    glossaries: list[dict[str, object]] = []
    package_root = resources.files("subtitle_pipeline")
    for relative_path in _BUILTIN_GLOSSARY_FILES:
        resource = package_root.joinpath(relative_path)
        value = json.loads(resource.read_text(encoding="utf-8"))
        glossaries.append(_validate_translation_glossary(value, relative_path))
    for path_value in configured_files:
        path = Path(path_value).expanduser()
        if not path.is_file():
            raise ValueError(f"translation glossary file not found: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid translation glossary file {path}: {exc}"
            ) from exc
        glossaries.append(_validate_translation_glossary(value, str(path)))

    identity = {
        "title": metadata.get("title"),
        "description": str(metadata.get("description") or "")[:10000],
        **_youtube_metadata_context(metadata),
    }
    evidence = json.dumps(identity, ensure_ascii=False).casefold()
    franchises: list[dict[str, str]] = []
    terms: dict[str, str] = {}
    characters_by_id: dict[str, dict[str, object]] = {}
    asr_entities_by_surface: dict[str, dict[str, object]] = {}
    knowledge_records: list[dict[str, object]] = []
    for glossary in glossaries:
        matches = glossary["match"]
        assert isinstance(matches, list)
        if not glossary.get("always") and not any(
            isinstance(candidate, str) and candidate.casefold() in evidence
            for candidate in matches
        ):
            continue
        franchises.append(
            {
                "name": str(glossary["name"]),
                "background": str(glossary["background"]),
            }
        )
        glossary_terms = glossary.get("terms", {})
        assert isinstance(glossary_terms, dict)
        terms.update(
            {
                source.strip(): target.strip()
                for source, target in glossary_terms.items()
                if isinstance(source, str)
                and isinstance(target, str)
                and source.strip()
                and target.strip()
            }
        )
        glossary_characters = glossary.get("characters", [])
        assert isinstance(glossary_characters, list)
        for character in glossary_characters:
            assert isinstance(character, dict)
            character_id = character["id"]
            assert isinstance(character_id, str)
            characters_by_id[character_id] = _translation_character(character)
        glossary_asr_entities = glossary.get("asr_entities", [])
        assert isinstance(glossary_asr_entities, list)
        for entity in glossary_asr_entities:
            assert isinstance(entity, dict)
            surface = entity["surface"]
            assert isinstance(surface, str)
            asr_entities_by_surface[surface] = entity
        glossary_knowledge = glossary.get("knowledge", [])
        assert isinstance(glossary_knowledge, list)
        knowledge_records.extend(glossary_knowledge)
    return {
        "video": identity,
        "franchises": franchises,
        "characters": list(characters_by_id.values()),
        "terms": terms,
        "asr_entities": list(asr_entities_by_surface.values()),
        "knowledge_records": knowledge_records,
    }


def _translation_character(character: dict[str, object]) -> dict[str, object]:
    """Keep rendering/training metadata out of the LLM reference payload."""
    allowed = ("id", "canonical", "source_name", "aliases", "short_names")
    return {key: character[key] for key in allowed if key in character}


def _japanese_single_word_list(context: dict[str, object]) -> list[str]:
    values: set[str] = set()
    characters = context.get("characters", [])
    if not isinstance(characters, list):
        return []

    def add(value: object) -> None:
        if not isinstance(value, str):
            return
        normalized = "".join(value.split())
        if len(normalized) > 1 and _JAPANESE_SCRIPT_RE.search(normalized):
            values.add(normalized)

    for character in characters:
        if not isinstance(character, dict):
            continue
        add(character.get("source_name"))
        aliases = character.get("aliases", [])
        if isinstance(aliases, list):
            for alias in aliases:
                add(alias)
        short_names = character.get("short_names", [])
        if isinstance(short_names, list):
            for short_name in short_names:
                if isinstance(short_name, dict):
                    add(short_name.get("source"))
    terms = context.get("terms", {})
    if isinstance(terms, dict):
        for source in terms:
            add(source)
    return sorted(values)


def _validate_translation_glossary(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"translation glossary {label} must be a JSON object")
    if not isinstance(value.get("name"), str) or not value["name"].strip():
        raise ValueError(f"translation glossary {label} requires a name")
    if not isinstance(value.get("background"), str):
        raise ValueError(f"translation glossary {label} requires background text")
    sources = value.get("sources", [])
    if not isinstance(sources, list) or not all(
        isinstance(source, str) and source.startswith(("https://", "http://"))
        for source in sources
    ):
        raise ValueError(f"translation glossary {label} sources must be HTTP URLs")
    matches = value.get("match")
    if not isinstance(matches, list) or not all(
        isinstance(item, str) and item.strip() for item in matches
    ):
        raise ValueError(f"translation glossary {label} requires string match terms")
    terms = value.get("terms", {})
    if not isinstance(terms, dict) or not all(
        isinstance(source, str) and isinstance(target, str)
        for source, target in terms.items()
    ):
        raise ValueError(f"translation glossary {label} terms must be string mappings")
    characters = value.get("characters", [])
    if not isinstance(characters, list):
        raise ValueError(f"translation glossary {label} characters must be a list")
    seen_character_ids: set[str] = set()
    for position, character in enumerate(characters):
        character_label = f"translation glossary {label} character {position}"
        if not isinstance(character, dict):
            raise ValueError(f"{character_label} must be an object")
        character_id = character.get("id")
        canonical = character.get("canonical")
        source_name = character.get("source_name")
        if not isinstance(character_id, str) or not character_id.strip():
            raise ValueError(f"{character_label} requires a non-empty id")
        if character_id in seen_character_ids:
            raise ValueError(
                f"translation glossary {label} duplicates character id {character_id}"
            )
        seen_character_ids.add(character_id)
        if not isinstance(canonical, str) or not canonical.strip():
            raise ValueError(f"{character_label} requires a non-empty canonical name")
        if not isinstance(source_name, str) or not source_name.strip():
            raise ValueError(f"{character_label} requires a non-empty source_name")
        aliases = character.get("aliases", [])
        if not isinstance(aliases, list) or not all(
            isinstance(alias, str) and alias.strip() for alias in aliases
        ):
            raise ValueError(f"{character_label} aliases must be non-empty strings")
        short_names = character.get("short_names", [])
        if not isinstance(short_names, list):
            raise ValueError(f"{character_label} short_names must be a list")
        for short_position, short_name in enumerate(short_names):
            short_label = f"{character_label} short name {short_position}"
            if not isinstance(short_name, dict):
                raise ValueError(f"{short_label} must be an object")
            source = short_name.get("source")
            target = short_name.get("target")
            context_only = short_name.get("context_only", False)
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"{short_label} requires a non-empty source")
            if not isinstance(target, str) or not target.strip():
                raise ValueError(f"{short_label} requires a non-empty target")
            if not isinstance(context_only, bool):
                raise ValueError(f"{short_label} context_only must be boolean")
    asr_entities = value.get("asr_entities", [])
    if not isinstance(asr_entities, list):
        raise ValueError(f"translation glossary {label} asr_entities must be a list")
    for position, entity in enumerate(asr_entities):
        entity_label = f"translation glossary {label} ASR entity {position}"
        if not isinstance(entity, dict):
            raise ValueError(f"{entity_label} must be an object")
        if not isinstance(entity.get("surface"), str) or not entity["surface"].strip():
            raise ValueError(f"{entity_label} requires a non-empty surface")
        if not isinstance(entity.get("reading"), str) or not entity["reading"].strip():
            raise ValueError(f"{entity_label} requires a non-empty reading")
        aliases = entity.get("aliases", [])
        if not isinstance(aliases, list) or not all(
            isinstance(alias, str) and alias.strip() for alias in aliases
        ):
            raise ValueError(f"{entity_label} aliases must be non-empty strings")
    knowledge = value.get("knowledge", [])
    if not isinstance(knowledge, list):
        raise ValueError(f"translation glossary {label} knowledge must be a list")
    seen_knowledge_ids: set[str] = set()
    for position, item in enumerate(knowledge):
        item_label = f"translation glossary {label} knowledge {position}"
        if not isinstance(item, dict):
            raise ValueError(f"{item_label} must be an object")
        record_id = item.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"{item_label} requires a non-empty id")
        if record_id in seen_knowledge_ids:
            raise ValueError(
                f"translation glossary {label} duplicates knowledge id {record_id}"
            )
        seen_knowledge_ids.add(record_id)
        for key in ("title", "body"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"{item_label} requires a non-empty {key}")
        for key in ("aliases", "keywords"):
            values = item.get(key, [])
            if not isinstance(values, list) or not all(
                isinstance(candidate, str) and candidate.strip() for candidate in values
            ):
                raise ValueError(f"{item_label} {key} must be non-empty strings")
        reliability = item.get("reliability", 0.8)
        if not isinstance(reliability, (int, float)) or not 0 <= reliability <= 1:
            raise ValueError(f"{item_label} reliability must be between 0 and 1")
    return value


def _canonicalize_catalog_tags(
    tags: list[str], catalog: dict[str, object]
) -> tuple[list[str], list[dict[str, object]]]:
    aliases: dict[str, list[tuple[int, str]]] = {}
    for canonical, raw in catalog.items():
        if not isinstance(canonical, str) or not canonical.strip():
            continue
        heat = 0
        names = [canonical]
        if isinstance(raw, dict):
            raw_heat = raw.get("heat", 0)
            if isinstance(raw_heat, (int, float)):
                heat = max(0, int(raw_heat))
            raw_aliases = raw.get("aliases", [])
            if isinstance(raw_aliases, list):
                names.extend(
                    str(value) for value in raw_aliases if isinstance(value, str)
                )
        for name in names:
            aliases.setdefault(name.strip().casefold(), []).append(
                (heat, canonical.strip())
            )

    resolved: list[str] = []
    matches: list[dict[str, object]] = []
    for tag in tags:
        candidates = aliases.get(tag.casefold(), [])
        if candidates:
            heat, canonical = max(candidates, key=lambda value: value[0])
            resolved.append(canonical)
            matches.append(
                {
                    "candidate": tag,
                    "canonical": canonical,
                    "heat": heat,
                    "existing": True,
                }
            )
        else:
            resolved.append(tag)
            matches.append(
                {"candidate": tag, "canonical": tag, "heat": 0, "existing": False}
            )
    return resolved, matches


def _write_manifest(result: PipelineResult, url: str, title: str) -> None:
    values = asdict(result)
    values.update(
        {
            "url": url,
            "title": title,
            "completed_at": datetime.now(UTC).isoformat(),
        }
    )
    serializable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in values.items()
    }
    (result.job_dir / "manifest.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
