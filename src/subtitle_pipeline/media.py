from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .commands import require_command, run
from .config import DownloadConfig, RenderConfig
from .subtitles import Cue, read_subtitles, text_display_width


_RENDER_TERMINAL_PLAIN_PUNCTUATION_RE = re.compile(
    r'''[，、；：。．,;:]+(?=["'”’」』）)\]]*$)'''
)
_RENDER_TERMINAL_ASCII_PERIOD_RE = re.compile(
    r'''(?<!\.)\.(?=["'”’」』）)\]]*$)'''
)


@dataclass(frozen=True)
class DownloadResult:
    video: Path
    metadata: dict[str, object]


@dataclass(frozen=True)
class SubtitleLayout:
    width: int
    height: int
    font_size: int
    margin_horizontal: int
    margin_vertical: int
    outline: int
    max_line_units: float
    frame_line_units: float


@dataclass(frozen=True)
class RenderCue:
    start: float
    end: float
    text: str


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

    logging.info("downloaded video: %s", video)
    return DownloadResult(video, metadata)


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


def render_subtitles(
    video: Path,
    subtitle: Path,
    destination: Path,
    config: RenderConfig,
    *,
    semantic_segments: dict[int, list[str]] | None = None,
) -> Path:
    ffmpeg = require_command("ffmpeg")
    layout = subtitle_layout(video, config)

    # Running in the subtitle directory avoids platform-specific escaping of full paths.
    local_video = video.resolve()
    local_subtitle = subtitle.resolve()
    local_destination = destination.resolve()
    if local_subtitle.parent != local_destination.parent:
        raise ValueError("subtitle and rendered video must share a work directory")

    source_cues = read_subtitles(local_subtitle)
    render_cues = _layout_subtitle_cues(
        source_cues,
        max_line_units=layout.max_line_units,
        hard_max_line_units=layout.frame_line_units,
        semantic_segments=semantic_segments,
    )
    ass_path = local_subtitle.with_suffix(".render.ass")
    _write_ass(
        render_cues,
        ass_path,
        width=layout.width,
        height=layout.height,
        font_name=config.font_name,
        font_size=layout.font_size,
        margin_vertical=layout.margin_vertical,
        outline=layout.outline,
    )
    logging.info(
        "adaptive subtitle style: %dx%d font=%d prompt_margin=%d "
        "ass_margins=1/%d outline=%d cues=%d->%d",
        layout.width,
        layout.height,
        layout.font_size,
        layout.margin_horizontal,
        layout.margin_vertical,
        layout.outline,
        len(source_cues),
        len(render_cues),
    )
    subtitle_name = ass_path.name.replace("'", r"\'").replace(":", r"\:")
    filter_value = f"subtitles=filename='{subtitle_name}'"
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


def subtitle_layout(video: Path, config: RenderConfig) -> SubtitleLayout:
    width, height = _video_dimensions(video)
    font_size = _adaptive_font_size(width, height, config)
    margin_horizontal = _adaptive_horizontal_margin(width, height, config)
    usable_width = width - 2 * margin_horizontal
    return SubtitleLayout(
        width=width,
        height=height,
        font_size=font_size,
        margin_horizontal=margin_horizontal,
        margin_vertical=round(height * config.margin_vertical_ratio),
        outline=max(1, round(min(width, height) * config.outline_ratio)),
        max_line_units=max(1.0, usable_width / font_size),
        frame_line_units=max(1.0, width / font_size),
    )


def _adaptive_font_size(width: int, height: int, config: RenderConfig) -> int:
    ratio = (
        config.portrait_font_size_ratio
        if height > width
        else config.font_size_ratio
    )
    return max(
        config.min_font_size,
        min(config.max_font_size, round(min(width, height) * ratio)),
    )


def _adaptive_horizontal_margin(
    width: int, height: int, config: RenderConfig
) -> int:
    ratio = (
        config.portrait_margin_horizontal_ratio
        if height > width
        else config.margin_horizontal_ratio
    )
    return round(width * ratio)


def _video_dimensions(video: Path) -> tuple[int, int]:
    ffprobe = require_command("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height:stream_tags=rotate:stream_side_data=rotation",
        "-of",
        "json",
        str(video.resolve()),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
    except (
        subprocess.CalledProcessError,
        json.JSONDecodeError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeError(f"could not determine video dimensions: {video}") from exc

    rotation: float = 0
    tags = stream.get("tags")
    if isinstance(tags, dict):
        try:
            rotation = float(tags.get("rotate", 0))
        except (TypeError, ValueError):
            pass
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for item in side_data:
            if isinstance(item, dict) and "rotation" in item:
                try:
                    rotation = float(item["rotation"])
                except (TypeError, ValueError):
                    pass
    if round(abs(rotation)) % 180 == 90:
        width, height = height, width
    if width < 1 or height < 1:
        raise RuntimeError(f"video has invalid dimensions: {width}x{height}")
    return width, height


def _layout_subtitle_cues(
    cues: list[Cue],
    *,
    max_line_units: float,
    hard_max_line_units: float | None = None,
    semantic_segments: dict[int, list[str]] | None = None,
) -> list[RenderCue]:
    hard_limit = max_line_units if hard_max_line_units is None else hard_max_line_units
    if hard_limit < max_line_units:
        raise ValueError("hard line limit cannot be smaller than the preferred line limit")
    segments_by_id = semantic_segments or {}
    rendered: list[RenderCue] = []
    for index, cue in enumerate(cues):
        segments = segments_by_id.get(index, [" ".join(cue.text.split())])
        if not segments or "".join(segments) != " ".join(cue.text.split()):
            raise ValueError(f"semantic segments do not preserve cue {index}")
        display_segments = [_strip_render_terminal_punctuation(item) for item in segments]
        widths = [text_display_width(segment) for segment in display_segments]
        if any(width > hard_limit + 1e-9 for width in widths):
            raise ValueError(f"semantic segment for cue {index} exceeds one line")
        total_width = sum(widths)
        elapsed = cue.start
        for segment_index, (segment, units) in enumerate(zip(display_segments, widths)):
            if segment_index == len(display_segments) - 1 or total_width <= 0:
                end = cue.end
            else:
                end = elapsed + (cue.end - cue.start) * units / total_width
            rendered.append(RenderCue(elapsed, end, segment))
            elapsed = end
    return rendered


def _strip_render_terminal_punctuation(text: str) -> str:
    without_plain = _RENDER_TERMINAL_PLAIN_PUNCTUATION_RE.sub("", text)
    without_period = _RENDER_TERMINAL_ASCII_PERIOD_RE.sub("", without_plain)
    return without_period if without_period.strip() else text


def _write_ass(
    cues: list[Cue] | list[RenderCue],
    path: Path,
    *,
    width: int,
    height: int,
    font_name: str,
    font_size: int,
    margin_vertical: int,
    outline: int,
) -> None:
    safe_font_name = font_name.replace(",", "")
    style_format = (
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding"
    )
    style = (
        f"Style: Default,{safe_font_name},{font_size},&H00FFFFFF,&H000000FF,"
        "&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,"
        f"{outline},0,2,1,1,"
        f"{margin_vertical},1"
    )
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.709\n\n"
        "[V4+ Styles]\n"
        f"{style_format}\n"
        f"{style}\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )
    events = [
        "Dialogue: 0,"
        f"{_ass_timestamp(cue.start)},{_ass_timestamp(cue.end)},"
        "Default,,0,0,0,,"
        f"{_escape_ass_text(cue.text)}"
        for cue in cues
        if cue.text.strip() and cue.end > cue.start
    ]
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


def _ass_timestamp(value: float) -> str:
    centiseconds = max(0, round(value * 100))
    hours, centiseconds = divmod(centiseconds, 360_000)
    minutes, centiseconds = divmod(centiseconds, 6_000)
    seconds, centiseconds = divmod(centiseconds, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def _escape_ass_text(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )
