from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from .commands import CommandError, require_command, run
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
    common = [yt_dlp, "--no-playlist", *_runtime_arguments(config)]
    common.extend(_authentication_arguments(config))
    video_command = [
        *common,
        "--write-info-json",
        "--format",
        "bv*+ba/b",
        "--merge-output-format",
        "mp4",
        "--output",
        str(output),
        url,
    ]
    run(video_command)

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

    subtitle = _find_subtitle(directory, metadata)
    if subtitle is None or not _is_original_subtitle(subtitle, metadata):
        subtitle_command = [
            *common,
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            "--sub-langs",
            ",".join(_subtitle_languages(config, metadata)),
            "--sub-format",
            "srt/vtt/best",
            "--convert-subs",
            "srt",
            "--output",
            str(output),
            url,
        ]
        try:
            run(subtitle_command)
        except CommandError as exc:
            logging.warning(
                "YouTube subtitle download failed (%s); falling back to Whisper",
                exc,
            )
        subtitle = _find_subtitle(directory, metadata)

    logging.info("downloaded video: %s", video)
    if subtitle:
        logging.info("using YouTube subtitle: %s", subtitle.name)
    return DownloadResult(video, subtitle, metadata)


def _authentication_arguments(config: DownloadConfig) -> list[str]:
    arguments: list[str] = []
    if config.cookies_from_browser:
        arguments.extend(["--cookies-from-browser", config.cookies_from_browser])
    if config.cookies_file:
        arguments.extend(["--cookies", config.cookies_file])
    return arguments


def _runtime_arguments(config: DownloadConfig) -> list[str]:
    runtime = config.js_runtime
    if not runtime:
        return []
    if runtime == "auto":
        for name in ("deno", "node", "qjs"):
            path = shutil.which(name)
            if path:
                runtime_name = "quickjs" if name == "qjs" else name
                return ["--js-runtimes", f"{runtime_name}:{path}"]
        logging.warning(
            "no supported JavaScript runtime found; install Deno 2.3+ or Node 22+"
        )
        return []
    return ["--js-runtimes", runtime]


def _find_subtitle(
    directory: Path, metadata: dict[str, object]
) -> Path | None:
    subtitles = [
        path for path in directory.glob("source.*.srt") if path.stat().st_size > 0
    ]
    if not subtitles:
        subtitles = [
            path for path in directory.glob("source.*.vtt") if path.stat().st_size > 0
        ]
    if not subtitles:
        return None

    originals = _original_language_variants(metadata)
    manual = metadata.get("subtitles")
    manual_tags = (
        {str(tag).lower() for tag in manual} if isinstance(manual, dict) else set()
    )

    def priority(path: Path) -> tuple[int, str]:
        tag = _subtitle_tag(path)
        normalized = tag.lower()
        matches_original = any(
            normalized == original or normalized.startswith(f"{original}-")
            for original in originals
        )
        if normalized in manual_tags and matches_original:
            rank = 0
        elif any(normalized == f"{original}-orig" for original in originals):
            rank = 1
        elif matches_original:
            rank = 2
        elif normalized.endswith("-orig"):
            rank = 3
        elif normalized in manual_tags:
            rank = 4
        else:
            rank = 5
        return rank, normalized

    return min(subtitles, key=priority)


def _subtitle_languages(
    config: DownloadConfig, metadata: dict[str, object]
) -> list[str]:
    requested: list[str] = []
    for original in _original_language_variants(metadata):
        requested.extend([f"{original}-orig", original])
    for language in config.subtitle_languages:
        if language not in requested:
            requested.append(language)
    return requested


def _original_language(metadata: dict[str, object]) -> str | None:
    language = metadata.get("language")
    if isinstance(language, str) and language:
        normalized = language.lower()
        if all(character.isalnum() or character in "_-" for character in normalized):
            return normalized
    automatic = metadata.get("automatic_captions")
    if isinstance(automatic, dict):
        original_tags = sorted(
            tag[:-5].lower()
            for tag in automatic
            if isinstance(tag, str) and tag.lower().endswith("-orig")
        )
        if original_tags:
            return original_tags[0]
    return None


def _original_language_variants(metadata: dict[str, object]) -> list[str]:
    original = _original_language(metadata)
    if not original:
        return []
    variants = [original]
    base = original.split("-", 1)[0].split("_", 1)[0]
    if base not in variants:
        variants.append(base)
    return variants


def _subtitle_tag(path: Path) -> str:
    name = path.name
    prefix = "source."
    suffix = path.suffix
    return name[len(prefix) : -len(suffix)]


def _is_original_subtitle(path: Path, metadata: dict[str, object]) -> bool:
    tag = _subtitle_tag(path).lower()
    originals = _original_language_variants(metadata)
    if originals:
        return any(tag == original or tag == f"{original}-orig" for original in originals)
    return tag.endswith("-orig")


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
