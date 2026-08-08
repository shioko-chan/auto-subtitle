from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from .config import AppConfig, llm_api_key
from .media import download_youtube, render_subtitles, transcribe_with_whisper
from .subtitles import Cue, merge_semantic_cues, read_subtitles, write_srt
from .translate import OpenAICompatibleTranslator
from .upload import upload_to_bilibili


@dataclass(frozen=True)
class PipelineResult:
    job_dir: Path
    source_video: Path
    source_subtitle: Path
    translated_subtitle: Path
    translated_metadata: Path
    rendered_video: Path
    uploaded: bool


def run_pipeline(
    url: str,
    config: AppConfig,
    *,
    upload_override: bool | None = None,
) -> PipelineResult:
    url = normalize_youtube_url(url)
    job_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    job_dir = config.work_dir.resolve() / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    logging.info("job directory: %s", job_dir)

    downloaded = download_youtube(url, job_dir, config.download)
    source_subtitle = downloaded.subtitle
    if source_subtitle is None:
        if not config.whisper.enabled:
            raise RuntimeError(
                "no usable source-language subtitle was found and Whisper fallback is disabled"
            )
        source_subtitle = transcribe_with_whisper(
            downloaded.video, job_dir / "source.whisper.srt", config.whisper
        )

    cues = read_subtitles(source_subtitle)
    original_cue_count = len(cues)
    cues = merge_semantic_cues(cues, config.segmentation)
    logging.info(
        "semantic segmentation: %d source cues -> %d translation units",
        original_cue_count,
        len(cues),
    )
    translator = OpenAICompatibleTranslator(config.llm, llm_api_key(config.llm))
    translated = translator.translate(cues)
    translated_path = job_dir / "translated.zh-CN.srt"
    write_srt(translated, translated_path)

    source_title = str(downloaded.metadata.get("title") or "YouTube video")
    source_description = str(downloaded.metadata.get("description") or "")
    youtube_context = _youtube_metadata_context(downloaded.metadata)
    subtitle_evidence = _subtitle_evidence(
        cues, config.llm.metadata_subtitle_max_chars
    )
    ip_aliases = _load_optional_json_object(config.llm.ip_aliases_file, "IP aliases")
    tag_catalog = _load_optional_json_object(
        config.upload.tag_catalog_file, "Bilibili tag catalog"
    )
    title, description = source_title, source_description
    content_summary = ""
    generated_tags: list[str] = []
    if config.llm.translate_metadata:
        logging.info("translating video title and description and generating tags")
        title, description, content_summary, generated_tags = translator.translate_metadata(
            source_title,
            source_description,
            youtube_context=youtube_context,
            subtitle_evidence=subtitle_evidence,
            ip_aliases=ip_aliases,
            bilibili_tag_catalog=tag_catalog,
        )
    generated_tags, tag_catalog_matches = _canonicalize_catalog_tags(
        generated_tags, tag_catalog
    )
    upload_tags = _merge_tags(config.upload.tags, generated_tags, config.upload.max_tags)
    metadata_path = job_dir / "translated.metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "source_title": source_title,
                "source_description": source_description,
                "translated_title": title,
                "translated_description": description,
                "youtube_context": youtube_context,
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

    rendered_path = job_dir / "translated.mp4"
    render_subtitles(downloaded.video, translated_path, rendered_path, config.render)

    should_upload = config.upload.enabled if upload_override is None else upload_override
    if should_upload:
        upload_to_bilibili(
            rendered_path,
            title=title,
            description=description,
            source_url=url,
            tags=upload_tags,
            config=config.upload,
        )

    result = PipelineResult(
        job_dir=job_dir,
        source_video=downloaded.video,
        source_subtitle=source_subtitle,
        translated_subtitle=translated_path,
        translated_metadata=metadata_path,
        rendered_video=rendered_path,
        uploaded=should_upload,
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
    return {key: metadata[key] for key in keys if metadata.get(key) not in (None, "", [])}


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
                names.extend(str(value) for value in raw_aliases if isinstance(value, str))
        for name in names:
            aliases.setdefault(name.strip().casefold(), []).append((heat, canonical.strip()))

    resolved: list[str] = []
    matches: list[dict[str, object]] = []
    for tag in tags:
        candidates = aliases.get(tag.casefold(), [])
        if candidates:
            heat, canonical = max(candidates, key=lambda value: value[0])
            resolved.append(canonical)
            matches.append(
                {"candidate": tag, "canonical": canonical, "heat": heat, "existing": True}
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
        key: str(value) if isinstance(value, Path) else value for key, value in values.items()
    }
    (result.job_dir / "manifest.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
