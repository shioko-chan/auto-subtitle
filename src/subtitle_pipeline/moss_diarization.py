from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass
from importlib import resources
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

from .audio_buffer import AudioBuffer
from .config import AudioAnalysisConfig

logger = logging.getLogger(__name__)

_CACHE_VERSION = 2
_SILENCE_PROBE_SECONDS = 0.5


@dataclass(frozen=True)
class MossSegment:
    start: float
    end: float
    speaker: str
    text: str


@dataclass(frozen=True)
class MossDiarizationResult:
    segments: list[MossSegment]
    transcripts: list[dict[str, object]]
    windows: list[tuple[float, float]]


def transcribe_and_diarize(
    video: Path,
    job_dir: Path,
    config: AudioAnalysisConfig,
    metadata: dict[str, object],
    *,
    audio_buffer: AudioBuffer | None = None,
    waveform: Any | None = None,
    sample_rate: int = 16000,
) -> MossDiarizationResult:
    """Run MOSS once per long window and retain its speaker-aware transcript."""
    prompt = _prompt(metadata, config.character_styles_file)
    signature = _signature(video, config, prompt)
    cache_path = job_dir / "moss-transcribe-diarize.json"
    cached = _load_cache(cache_path, signature)
    if cached is not None:
        logger.info(
            "MOSS diarization cache: %d windows, %d segments",
            len(cached.windows),
            len(cached.segments),
        )
        return cached

    owns_memory = False
    memory: shared_memory.SharedMemory | None = None
    if audio_buffer is None:
        if waveform is None:
            raise RuntimeError("MOSS diarization requires an audio buffer or waveform")
        import numpy as np

        values = np.asarray(waveform.mean(dim=0).detach().cpu(), dtype=np.float32)
        memory = shared_memory.SharedMemory(create=True, size=max(1, values.nbytes))
        shared = np.ndarray(values.shape, dtype=np.float32, buffer=memory.buf)
        shared[:] = values
        descriptor = {
            "shared_memory": memory.name,
            "dtype": "float32",
            "shape": list(shared.shape),
            "sample_rate": sample_rate,
        }
        samples = shared
        owns_memory = True
    else:
        descriptor = audio_buffer.descriptor.as_dict()
        samples = audio_buffer.samples
        sample_rate = audio_buffer.sample_rate

    try:
        windows = _window_ranges(
            samples,
            sample_rate,
            target_seconds=config.moss_window_seconds,
            maximum_seconds=config.moss_max_window_seconds,
            search_seconds=config.moss_window_search_seconds,
        )
        segments, transcripts = _run_windows(
            descriptor,
            windows,
            prompt,
            config,
            cache_path=cache_path,
            signature=signature,
        )
    finally:
        if owns_memory:
            del samples, shared
            assert memory is not None
            memory.close()
            memory.unlink()

    result = MossDiarizationResult(segments, transcripts, windows)
    _write_cache(cache_path, signature, config, prompt, windows, segments, transcripts)
    logger.info(
        "MOSS diarization wrote %d windows and %d speaker segments",
        len(windows),
        len(segments),
    )
    return result


def _run_windows(
    descriptor: dict[str, object],
    windows: list[tuple[float, float]],
    prompt: str,
    config: AudioAnalysisConfig,
    *,
    cache_path: Path,
    signature: str,
) -> tuple[list[MossSegment], list[dict[str, object]]]:
    partial = _load_partial_cache(cache_path, signature, windows)
    segments = list(partial[0]) if partial is not None else []
    transcripts = list(partial[1]) if partial is not None else []
    completed = {int(item["window"]) for item in transcripts}

    for index, window in enumerate(windows):
        if index in completed:
            logger.info("MOSS window %d/%d cache hit", index + 1, len(windows))
            continue
        logger.info(
            "MOSS window %d/%d: %.3f-%.3f",
            index + 1,
            len(windows),
            window[0],
            window[1],
        )
        try:
            response = _run_worker(descriptor, [window], prompt, config)
        except RuntimeError as exc:
            raise RuntimeError(
                f"MOSS window {index + 1}/{len(windows)} failed: {exc}"
            ) from exc
        window_transcripts = _decode_transcripts(response.get("transcripts"), [window])
        window_segments = _decode_segments(response.get("segments"), [window])
        remapped_segments = [
            MossSegment(
                item.start,
                item.end,
                item.speaker.replace("MOSS_W000_", f"MOSS_W{index:03d}_", 1),
                item.text,
            )
            for item in window_segments
        ]
        transcript = dict(window_transcripts[0])
        transcript["window"] = index
        segments.extend(remapped_segments)
        transcripts.append(transcript)
        segments.sort(key=lambda item: (item.start, item.end, item.speaker))
        transcripts.sort(key=lambda item: int(item["window"]))
        _write_cache(
            cache_path,
            signature,
            config,
            prompt,
            windows,
            segments,
            transcripts,
        )
        logger.info(
            "MOSS window %d/%d cached: %d segments",
            index + 1,
            len(windows),
            len(remapped_segments),
        )
    return segments, transcripts


def _write_cache(
    path: Path,
    signature: str,
    config: AudioAnalysisConfig,
    prompt: str,
    windows: list[tuple[float, float]],
    segments: list[MossSegment],
    transcripts: list[dict[str, object]],
) -> None:
    payload = {
        "version": _CACHE_VERSION,
        "signature": signature,
        "model": config.moss_transcribe_model,
        "prompt": prompt,
        "windows": [{"start": start, "end": end} for start, end in windows],
        "segments": [asdict(segment) for segment in segments],
        "transcripts": transcripts,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _window_ranges(
    samples: Any,
    sample_rate: int,
    *,
    target_seconds: float,
    maximum_seconds: float,
    search_seconds: float,
) -> list[tuple[float, float]]:
    import numpy as np

    duration = len(samples) / sample_rate
    if duration <= target_seconds:
        return [(0.0, duration)]
    windows: list[tuple[float, float]] = []
    cursor = 0.0
    probe_samples = max(1, round(_SILENCE_PROBE_SECONDS * sample_rate))
    while duration - cursor > target_seconds:
        lower = cursor + max(1.0, target_seconds - search_seconds)
        upper = min(cursor + maximum_seconds, cursor + target_seconds + search_seconds)
        upper = min(upper, duration - 1.0)
        if upper <= lower:
            boundary = min(cursor + target_seconds, duration)
        else:
            first = round(lower * sample_rate)
            last = round(upper * sample_rate)
            offsets = range(
                first, max(first + 1, last - probe_samples + 1), probe_samples
            )
            boundary_sample = min(
                offsets,
                key=lambda offset: float(
                    np.mean(
                        np.square(
                            np.asarray(
                                samples[offset : offset + probe_samples],
                                dtype=np.float32,
                            )
                        )
                    )
                ),
            )
            boundary = (boundary_sample + probe_samples / 2) / sample_rate
        boundary = min(boundary, cursor + maximum_seconds, duration)
        if boundary <= cursor:
            raise RuntimeError("MOSS window selection made no progress")
        windows.append((round(cursor, 3), round(boundary, 3)))
        cursor = boundary
    if duration > cursor:
        windows.append((round(cursor, 3), duration))
    if any(end - start > maximum_seconds + 1e-3 for start, end in windows):
        raise RuntimeError("MOSS window exceeds configured maximum")
    return windows


def _run_worker(
    descriptor: dict[str, object],
    windows: list[tuple[float, float]],
    prompt: str,
    config: AudioAnalysisConfig,
) -> dict[str, object]:
    uv = shutil.which("uv")
    project = Path(config.moss_transcribe_worker_project).resolve()
    worker = project / "worker.py"
    if uv is None or not worker.is_file():
        raise RuntimeError(f"MOSS transcription worker is unavailable at {worker}")
    payload = {
        "model": config.moss_transcribe_model,
        "device": config.device,
        "dtype": "float16",
        "max_new_tokens": config.moss_max_new_tokens,
        "audio": descriptor,
        "windows": [{"start": start, "end": end} for start, end in windows],
        "prompt": prompt,
    }
    completed = subprocess.run(
        [uv, "run", "--project", str(project), "python", str(worker)],
        input=json.dumps(payload, ensure_ascii=False),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "MOSS transcription worker failed: "
            + (completed.stderr or completed.stdout)[-3000:]
        )
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("MOSS transcription worker returned malformed JSON") from exc
    if not isinstance(response, dict):
        raise TypeError("MOSS transcription worker returned a non-object")
    if response.get("error"):
        raw = str(response.get("raw") or "")
        tail = raw[-2000:].replace("\n", "\\n")
        raise RuntimeError(
            f"MOSS transcription failed: {response['error']}; raw_tail={tail!r}"
        )
    return response


def _decode_segments(
    value: object, windows: list[tuple[float, float]]
) -> list[MossSegment]:
    if not isinstance(value, list):
        raise TypeError("MOSS worker returned no segments")
    segments: list[MossSegment] = []
    for item in value:
        if not isinstance(item, dict):
            raise TypeError("MOSS worker returned an invalid segment")
        segment = MossSegment(
            round(float(item["start"]), 3),
            round(float(item["end"]), 3),
            str(item["speaker"]),
            str(item["text"]).strip(),
        )
        window_index = _window_index(segment.speaker)
        if window_index >= len(windows):
            raise RuntimeError("MOSS segment refers to an unknown window")
        start, end = windows[window_index]
        if (
            segment.start < start - 0.1
            or segment.end > end + 0.1
            or segment.end <= segment.start
            or not segment.text
        ):
            raise RuntimeError("MOSS worker returned an out-of-range segment")
        segments.append(segment)
    return sorted(segments, key=lambda item: (item.start, item.end, item.speaker))


def _window_index(speaker: str) -> int:
    import re

    match = re.fullmatch(r"MOSS_W(\d+)_S\d+", speaker)
    if match is None:
        raise RuntimeError(f"invalid MOSS speaker label: {speaker}")
    try:
        return int(match.group(1))
    except ValueError as exc:
        raise RuntimeError(f"invalid MOSS speaker label: {speaker}") from exc


def _decode_transcripts(
    value: object, windows: list[tuple[float, float]]
) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise TypeError("MOSS worker returned no transcript audit records")
    if len(value) != len(windows):
        raise RuntimeError("MOSS worker returned the wrong number of transcripts")
    transcripts: list[dict[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise TypeError("MOSS worker returned an invalid transcript record")
        expected_start, expected_end = windows[index]
        if (
            int(item.get("window", -1)) != index
            or abs(float(item.get("start", -1)) - expected_start) > 1e-3
            or abs(float(item.get("end", -1)) - expected_end) > 1e-3
            or not isinstance(item.get("raw"), str)
        ):
            raise RuntimeError("MOSS worker returned a mismatched transcript record")
        transcripts.append(
            {
                "window": index,
                "start": expected_start,
                "end": expected_end,
                "raw": str(item["raw"]),
            }
        )
    return transcripts


def _prompt(metadata: dict[str, object], styles_path: str | None) -> str:
    if styles_path:
        styles = json.loads(Path(styles_path).expanduser().read_text(encoding="utf-8"))
    else:
        styles = json.loads(
            resources.files("subtitle_pipeline")
            .joinpath("character_styles.json")
            .read_text(encoding="utf-8")
        )
    hotwords = [
        str(character.get("source_name") or "").strip()
        for character in styles.get("characters", [])
        if isinstance(character, dict) and character.get("source_name")
    ]
    context = "、".join(
        str(metadata.get(key) or "").strip()
        for key in ("title", "channel", "uploader")
        if metadata.get(key)
    )
    return (
        "请将音频中的日语语音准确转写为日文原文。每一段需以起始时间戳和"
        "说话人编号（[S01]、[S02]、[S03]…）开头，正文为对应语音内容，并在"
        "段末标注结束时间戳。同一说话人在整个音频中必须使用相同编号。多人同时"
        "讲话时不要合并成一个说话人；尽可能分别转写并保留重叠时间。不要翻译。"
        f"视频信息：{context or '无'}。热词提示：{'、'.join(hotwords)}。"
    )


def _signature(video: Path, config: AudioAnalysisConfig, prompt: str) -> str:
    stat = video.stat()
    payload = {
        "version": _CACHE_VERSION,
        "video_size": stat.st_size,
        "video_mtime_ns": stat.st_mtime_ns,
        "model": config.moss_transcribe_model,
        "window_seconds": config.moss_window_seconds,
        "maximum_seconds": config.moss_max_window_seconds,
        "search_seconds": config.moss_window_search_seconds,
        "max_new_tokens": config.moss_max_new_tokens,
        "prompt": prompt,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_cache(path: Path, signature: str) -> MossDiarizationResult | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("version") != _CACHE_VERSION
            or value.get("signature") != signature
        ):
            return None
        windows = [
            (float(item["start"]), float(item["end"])) for item in value["windows"]
        ]
        segments = _decode_segments(value["segments"], windows)
        transcripts = _decode_transcripts(value["transcripts"], windows)
        return MossDiarizationResult(segments, transcripts, windows)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        json.JSONDecodeError,
    ) as exc:
        logger.warning("ignoring unreadable MOSS cache %s: %s", path, exc)
        return None


def _load_partial_cache(
    path: Path,
    signature: str,
    windows: list[tuple[float, float]],
) -> tuple[list[MossSegment], list[dict[str, object]]] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("version") != _CACHE_VERSION or value.get("signature") != signature:
            return None
        cached_windows = [
            (float(item["start"]), float(item["end"])) for item in value["windows"]
        ]
        if cached_windows != windows:
            return None
        raw_transcripts = value["transcripts"]
        if not isinstance(raw_transcripts, list) or len(raw_transcripts) > len(windows):
            return None
        transcripts: list[dict[str, object]] = []
        for index, item in enumerate(raw_transcripts):
            if not isinstance(item, dict):
                return None
            start, end = windows[index]
            if (
                int(item.get("window", -1)) != index
                or abs(float(item.get("start", -1)) - start) > 1e-3
                or abs(float(item.get("end", -1)) - end) > 1e-3
                or not isinstance(item.get("raw"), str)
            ):
                return None
            transcripts.append(
                {"window": index, "start": start, "end": end, "raw": item["raw"]}
            )
        segments = _decode_segments(value["segments"], windows)
        if any(_window_index(item.speaker) >= len(transcripts) for item in segments):
            return None
        return segments, transcripts
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        json.JSONDecodeError,
    ):
        return None
