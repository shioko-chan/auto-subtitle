from __future__ import annotations

import json
import logging
import math
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .commands import require_command, run
from .config import ASRConfig
from .subtitles import Cue, write_srt


_CACHE_VERSION = 1


def transcribe_with_qwen(
    video: Path, destination: Path, config: ASRConfig
) -> Path:
    duration = _media_duration(video)
    chunk_count = max(1, math.ceil(duration / config.chunk_seconds))
    cache_path = destination.parent / "asr-cache.json"
    signature = _cache_signature(video, duration, config)
    cache = _load_cache(cache_path, signature)
    cached_chunks = cache["chunks"]
    assert isinstance(cached_chunks, dict)

    missing = [
        index
        for index in range(chunk_count)
        if not _valid_cached_record(cached_chunks.get(str(index)))
    ]
    for index in missing:
        cached_chunks.pop(str(index), None)
    model = None
    if missing:
        logging.info(
            "loading Qwen3-ASR (%d/%d audio chunks missing)",
            len(missing),
            chunk_count,
        )
        model = _load_qwen_model(config)
    else:
        logging.info("Qwen3-ASR cache complete: %d chunks", chunk_count)

    chunk_dir = destination.parent / "asr-chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for index in missing:
        core_start = index * config.chunk_seconds
        core_end = min(duration, core_start + config.chunk_seconds)
        extract_start = max(0.0, core_start - config.chunk_context_seconds)
        extract_end = min(duration, core_end + config.chunk_context_seconds)
        chunk_path = chunk_dir / f"chunk-{index:05d}.wav"
        logging.info(
            "Qwen3-ASR chunk %d/%d: %.1f-%.1fs",
            index + 1,
            chunk_count,
            core_start,
            core_end,
        )
        try:
            _extract_audio_chunk(
                video,
                chunk_path,
                start=extract_start,
                duration=extract_end - extract_start,
            )
            assert model is not None
            results = model.transcribe(
                audio=str(chunk_path),
                context=config.context,
                language=config.language,
                return_time_stamps=True,
            )
            if len(results) != 1:
                raise RuntimeError(
                    f"Qwen3-ASR returned {len(results)} results for one audio chunk"
                )
            result = results[0]
            cues = _result_to_cues(
                result,
                offset=extract_start,
                keep_start=core_start,
                keep_end=core_end,
                final_chunk=index == chunk_count - 1,
            )
            cached_chunks[str(index)] = {
                "core_start": core_start,
                "core_end": core_end,
                "language": str(getattr(result, "language", "")),
                "text": str(getattr(result, "text", "")),
                "cues": [asdict(cue) for cue in cues],
            }
            _write_cache(cache_path, cache)
            logging.info(
                "cached Qwen3-ASR chunk %d/%d: %d aligned units",
                index + 1,
                chunk_count,
                len(cues),
            )
        finally:
            chunk_path.unlink(missing_ok=True)

    all_cues: list[Cue] = []
    for index in range(chunk_count):
        record = cached_chunks.get(str(index))
        if not isinstance(record, dict) or not isinstance(record.get("cues"), list):
            raise RuntimeError(f"Qwen3-ASR cache is missing chunk {index}")
        all_cues.extend(_decode_cached_cues(record["cues"], index))
    if not all_cues:
        raise RuntimeError("Qwen3-ASR did not produce any aligned speech")
    write_srt(all_cues, destination)
    logging.info("Qwen3-ASR wrote %d aligned units: %s", len(all_cues), destination)
    return destination


def _load_qwen_model(config: ASRConfig) -> Any:
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise RuntimeError(
            "Qwen3-ASR is not installed; run `uv sync --extra asr`"
        ) from exc

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Qwen3-ASR device is {config.device}, but PyTorch cannot access CUDA"
        )
    dtype = getattr(torch, config.dtype)
    logging.info(
        "loading %s with aligner %s on %s (%s)",
        config.model,
        config.aligner_model,
        config.device,
        config.dtype,
    )
    return Qwen3ASRModel.from_pretrained(
        config.model,
        dtype=dtype,
        device_map=config.device,
        attn_implementation="sdpa",
        max_inference_batch_size=config.max_inference_batch_size,
        max_new_tokens=config.max_new_tokens,
        forced_aligner=config.aligner_model,
        forced_aligner_kwargs={
            "dtype": dtype,
            "device_map": config.device,
            "attn_implementation": "sdpa",
        },
    )


def _result_to_cues(
    result: Any,
    *,
    offset: float,
    keep_start: float,
    keep_end: float,
    final_chunk: bool,
) -> list[Cue]:
    text = str(getattr(result, "text", "")).strip()
    alignment = getattr(result, "time_stamps", None)
    items = list(getattr(alignment, "items", []) or [])
    if text and not items:
        raise RuntimeError("Qwen3 forced aligner returned no timestamps for non-empty text")
    fragments = _restore_punctuation(text, [str(item.text) for item in items])
    cues: list[Cue] = []
    previous_start = -1.0
    for item, fragment in zip(items, fragments):
        start = round(float(item.start_time) + offset, 3)
        end = round(float(item.end_time) + offset, 3)
        if start < previous_start:
            raise RuntimeError("Qwen3 forced aligner returned non-monotonic timestamps")
        previous_start = start
        midpoint = (start + end) / 2
        in_owned_range = midpoint >= keep_start and (
            midpoint < keep_end or (final_chunk and midpoint <= keep_end)
        )
        if in_owned_range and fragment.strip():
            cues.append(Cue(start, max(start + 0.08, end), fragment.strip()))
    return cues


def _restore_punctuation(text: str, tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        start = text.find(token, cursor)
        if start < 0:
            logging.warning(
                "could not map aligner tokens back to ASR punctuation; using bare tokens"
            )
            return tokens
        spans.append((start, start + len(token)))
        cursor = start + len(token)

    fragments: list[str] = []
    for index, (start, _end) in enumerate(spans):
        fragment_start = 0 if index == 0 else start
        fragment_end = spans[index + 1][0] if index + 1 < len(spans) else len(text)
        fragments.append(text[fragment_start:fragment_end])
    return fragments


def _media_duration(video: Path) -> float:
    ffprobe = require_command("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video.resolve()),
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True
        )
        duration = float(completed.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as exc:
        raise RuntimeError(f"could not determine media duration: {video}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"media has invalid duration: {duration}")
    return duration


def _extract_audio_chunk(
    video: Path, destination: Path, *, start: float, duration: float
) -> None:
    ffmpeg = require_command("ffmpeg")
    run(
        [
            ffmpeg,
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(video.resolve()),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(destination.resolve()),
        ]
    )


def _cache_signature(
    video: Path, duration: float, config: ASRConfig
) -> dict[str, object]:
    return {
        "video_name": video.name,
        "video_size": video.stat().st_size,
        "duration": round(duration, 3),
        "config": asdict(config),
    }


def _load_cache(path: Path, signature: dict[str, object]) -> dict[str, object]:
    empty: dict[str, object] = {
        "version": _CACHE_VERSION,
        "signature": signature,
        "chunks": {},
    }
    if not path.is_file():
        return empty
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logging.warning("ignoring unreadable Qwen3-ASR cache: %s", path)
        return empty
    if (
        not isinstance(value, dict)
        or value.get("version") != _CACHE_VERSION
        or value.get("signature") != signature
        or not isinstance(value.get("chunks"), dict)
    ):
        logging.info("Qwen3-ASR cache signature changed; starting a fresh cache")
        return empty
    return value


def _write_cache(path: Path, cache: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _decode_cached_cues(values: list[object], chunk_index: int) -> list[Cue]:
    cues: list[Cue] = []
    try:
        for value in values:
            if not isinstance(value, dict):
                raise TypeError
            cues.append(
                Cue(
                    float(value["start"]),
                    float(value["end"]),
                    str(value["text"]),
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Qwen3-ASR cache contains an invalid chunk {chunk_index}"
        ) from exc
    return cues


def _valid_cached_record(value: object) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("cues"), list):
        return False
    try:
        for cue in value["cues"]:
            if (
                not isinstance(cue, dict)
                or float(cue["end"]) <= float(cue["start"])
                or not str(cue["text"]).strip()
            ):
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True
