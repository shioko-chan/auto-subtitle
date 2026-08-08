from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .commands import require_command, run
from .config import DownloadConfig, RenderConfig, WhisperConfig
from .subtitles import Cue, write_srt


@dataclass(frozen=True)
class DownloadResult:
    video: Path
    subtitle: Path | None
    metadata: dict[str, object]


def download_youtube(url: str, directory: Path, config: DownloadConfig) -> DownloadResult:
    yt_dlp = require_command("yt-dlp")
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "source.%(ext)s"
    command = [
        yt_dlp,
        "--no-playlist",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs",
        ",".join(config.subtitle_languages),
        "--sub-format",
        "srt/vtt/best",
        "--convert-subs",
        "srt",
        "--format",
        "bv*+ba/b",
        "--merge-output-format",
        "mp4",
        "--output",
        str(output),
    ]
    if config.cookies_from_browser:
        command.extend(["--cookies-from-browser", config.cookies_from_browser])
    if config.cookies_file:
        command.extend(["--cookies", config.cookies_file])
    command.append(url)
    run(command)

    info_path = directory / "source.info.json"
    if not info_path.exists():
        raise RuntimeError("yt-dlp completed but did not create source.info.json")
    metadata = json.loads(info_path.read_text(encoding="utf-8"))
    candidates = [
        path
        for path in directory.glob("source.*")
        if path.suffix.lower()
        not in {".json", ".srt", ".vtt", ".part", ".ytdl", ".description"}
    ]
    if not candidates:
        raise RuntimeError("yt-dlp completed but no downloaded video was found")
    video = max(candidates, key=lambda path: path.stat().st_size)
    subtitles = sorted(directory.glob("source.*.srt"))
    if not subtitles:
        subtitles = sorted(directory.glob("source.*.vtt"))
    subtitle = subtitles[0] if subtitles else None
    logging.info("downloaded video: %s", video)
    if subtitle:
        logging.info("using YouTube subtitle: %s", subtitle.name)
    return DownloadResult(video, subtitle, metadata)


def transcribe_with_whisper(
    video: Path, destination: Path, config: WhisperConfig
) -> Path:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "no YouTube subtitle was found and faster-whisper is not installed; "
            "install the project with the [whisper] extra"
        ) from exc

    logging.info("no YouTube subtitle found; transcribing with faster-whisper")
    model = WhisperModel(
        config.model,
        device=config.device,
        compute_type=config.compute_type,
    )
    segments, info = model.transcribe(
        str(video),
        language=config.language,
        initial_prompt=config.initial_prompt,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    logging.info(
        "Whisper language: %s (probability %.3f)",
        info.language,
        info.language_probability,
    )
    cues = [
        Cue(float(segment.start), float(segment.end), segment.text.strip())
        for segment in segments
        if segment.text.strip()
    ]
    if not cues:
        raise RuntimeError("Whisper did not produce any speech segments")
    write_srt(cues, destination)
    return destination


def render_subtitles(
    video: Path, subtitle: Path, destination: Path, config: RenderConfig
) -> Path:
    ffmpeg = require_command("ffmpeg")
    # Running in the subtitle directory avoids platform-specific escaping of full paths.
    local_video = video.resolve()
    local_subtitle = subtitle.resolve()
    local_destination = destination.resolve()
    if local_subtitle.parent != local_destination.parent:
        raise ValueError("subtitle and rendered video must share a work directory")
    style = (
        f"FontName={config.font_name},FontSize={config.font_size},"
        f"MarginV={config.margin_vertical},Outline=1,Shadow=0"
    )
    subtitle_name = local_subtitle.name.replace("'", r"\'").replace(":", r"\:")
    filter_value = f"subtitles=filename='{subtitle_name}':force_style='{style}'"
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(local_video),
            "-vf",
            filter_value,
            "-c:v",
            "libx264",
            "-preset",
            config.preset,
            "-crf",
            str(config.crf),
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(local_destination),
        ],
        cwd=local_destination.parent,
    )
    return destination
