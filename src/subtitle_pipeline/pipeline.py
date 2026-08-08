from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

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
    rendered_video: Path
    uploaded: bool


def run_pipeline(
    url: str,
    config: AppConfig,
    *,
    upload_override: bool | None = None,
) -> PipelineResult:
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

    rendered_path = job_dir / "translated.mp4"
    render_subtitles(downloaded.video, translated_path, rendered_path, config.render)

    should_upload = config.upload.enabled if upload_override is None else upload_override
    title = str(downloaded.metadata.get("title") or "YouTube video")
    description = str(downloaded.metadata.get("description") or "")
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
        rendered_video=rendered_path,
        uploaded=should_upload,
    )
    _write_manifest(result, url, title)
    return result


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
