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
from .subtitles import merge_semantic_cues, read_subtitles, write_srt
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
    title, description = source_title, source_description
    if config.llm.translate_metadata:
        logging.info("translating video title and description")
        title, description = translator.translate_metadata(source_title, source_description)
    metadata_path = job_dir / "translated.metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "source_title": source_title,
                "source_description": source_description,
                "translated_title": title,
                "translated_description": description,
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
