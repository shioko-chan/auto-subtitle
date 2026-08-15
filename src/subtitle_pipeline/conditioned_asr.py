from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .audio_analysis import AudioRegion, _overlap_intersections
from .audio_buffer import AudioBuffer
from .config import AudioAnalysisConfig
from .subtitles import Cue

logger = logging.getLogger(__name__)

_CACHE_VERSION = 2
_MAX_WINDOW_SECONDS = 30.0


@dataclass(frozen=True)
class ConditionedWindow:
    start: float
    end: float
    speakers: tuple[str, ...]
    turns: tuple[AudioRegion, ...]


def repair_long_overlaps(
    cues: list[Cue],
    diarization: list[AudioRegion],
    audio: AudioBuffer,
    job_dir: Path,
    config: AudioAnalysisConfig,
) -> list[Cue]:
    windows = _conditioned_windows(diarization, audio.duration, config)
    if not windows:
        return cues
    if config.conditioned_asr_backend == "disabled":
        first = windows[0]
        raise RuntimeError(
            "long overlapping speech requires conditioned ASR, but the backend is "
            f"disabled ({first.start:.3f}-{first.end:.3f}s)"
        )
    cache_path = job_dir / "conditioned-asr-cache.json"
    signature = {
        "version": _CACHE_VERSION,
        "model": config.conditioned_asr_model,
        "revision": config.conditioned_asr_revision,
        "windows": [_window_payload(window) for window in windows],
    }
    repaired = _load_cache(cache_path, signature)
    if repaired is None:
        repaired = _run_dicow(audio, windows, config)
        _write_cache(cache_path, signature, repaired)
    return _replace_windows(cues, repaired, windows)


def _conditioned_windows(
    diarization: list[AudioRegion],
    duration: float,
    config: AudioAnalysisConfig,
) -> list[ConditionedWindow]:
    intersections = [
        item
        for item in _overlap_intersections(diarization)
        if item.end - item.start > config.overlap_conditioned_asr_seconds
    ]
    windows = [
        _expand_overlap(item, diarization, duration, config.overlap_context_seconds)
        for item in intersections
    ]
    merged: list[ConditionedWindow] = []
    for window in windows:
        if merged and window.start <= merged[-1].end:
            previous = merged[-1]
            combined = ConditionedWindow(
                previous.start,
                max(previous.end, window.end),
                tuple(sorted({*previous.speakers, *window.speakers})),
                _unique_turns((*previous.turns, *window.turns)),
            )
            _validate_window_length(combined)
            merged[-1] = combined
        else:
            _validate_window_length(window)
            merged.append(window)
    return merged


def _expand_overlap(
    overlap: AudioRegion,
    diarization: list[AudioRegion],
    duration: float,
    context_seconds: float,
) -> ConditionedWindow:
    start = max(0.0, overlap.start - context_seconds)
    end = min(duration, overlap.end + context_seconds)
    turns = tuple(
        AudioRegion(
            max(start, region.start),
            min(end, region.end),
            "speech",
            region.speaker,
            anonymous_speaker=region.anonymous_speaker or region.speaker,
        )
        for region in diarization
        if region.end > start and region.start < end and region.speaker
    )
    speakers = tuple(
        sorted(
            {
                region.anonymous_speaker or region.speaker
                for region in turns
                if region.anonymous_speaker or region.speaker
            }
        )
    )
    return ConditionedWindow(round(start, 3), round(end, 3), speakers, turns)


def _validate_window_length(window: ConditionedWindow) -> None:
    if window.end - window.start > _MAX_WINDOW_SECONDS + 1e-3:
        raise RuntimeError(
            "conditioned ASR context exceeds the 30-second model window: "
            f"{window.start:.3f}-{window.end:.3f}s"
        )


def _unique_turns(regions: tuple[AudioRegion, ...]) -> tuple[AudioRegion, ...]:
    values = {
        (region.start, region.end, region.speaker, region.anonymous_speaker): region
        for region in regions
    }
    return tuple(
        sorted(
            values.values(), key=lambda item: (item.start, item.end, item.speaker or "")
        )
    )


def _run_dicow(
    audio: AudioBuffer,
    windows: list[ConditionedWindow],
    config: AudioAnalysisConfig,
) -> list[Cue]:
    uv = shutil.which("uv")
    project = Path(config.conditioned_asr_worker_project).resolve()
    worker = project / "worker.py"
    if uv is None or not worker.is_file():
        raise RuntimeError(f"DiCoW worker is unavailable at {worker}")
    payload = {
        "model": config.conditioned_asr_model,
        "revision": config.conditioned_asr_revision,
        "device": config.device,
        "language": "ja",
        "audio": audio.descriptor.as_dict(),
        "windows": [_window_payload(window) for window in windows],
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
            "DiCoW worker failed: " + (completed.stderr or completed.stdout)[-3000:]
        )
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("DiCoW worker returned malformed JSON") from exc
    if not isinstance(response, dict) or response.get("error"):
        raise RuntimeError(f"DiCoW inference failed: {response.get('error')}")
    return _decode_cues(response.get("cues"), windows)


def _window_payload(window: ConditionedWindow) -> dict[str, object]:
    return {
        "start": window.start,
        "end": window.end,
        "speakers": list(window.speakers),
        "turns": [
            {
                "start": region.start,
                "end": region.end,
                "speaker": region.anonymous_speaker or region.speaker,
            }
            for region in window.turns
        ],
    }


def _decode_cues(value: object, windows: list[ConditionedWindow]) -> list[Cue]:
    if not isinstance(value, list):
        raise TypeError("DiCoW worker returned no cues")
    label_to_character = {
        region.anonymous_speaker or region.speaker: region.speaker
        for window in windows
        for region in window.turns
        if region.speaker
    }
    cues: list[Cue] = []
    for item in value:
        if not isinstance(item, dict):
            raise TypeError("DiCoW worker returned an invalid cue")
        start = float(item["start"])
        end = float(item["end"])
        text = str(item["text"]).strip()
        label = str(item["speaker"])
        if (
            end <= start
            or not text
            or not any(
                start >= window.start - 0.1 and end <= window.end + 0.1
                for window in windows
            )
        ):
            raise RuntimeError("DiCoW worker returned an invalid cue range")
        cues.append(
            Cue(start, end, text, label_to_character.get(label, label), "speech")
        )
    if not cues:
        raise RuntimeError("DiCoW returned no speech for a long-overlap window")
    for window in windows:
        returned = {
            cue.speaker
            for cue in cues
            if cue.end > window.start and cue.start < window.end
        }
        expected = {
            label_to_character.get(speaker, speaker) for speaker in window.speakers
        }
        missing = sorted(expected - returned)
        if missing:
            raise RuntimeError(
                "DiCoW omitted active speakers from a repair window: "
                + ", ".join(missing)
            )
    return sorted(cues, key=lambda cue: (cue.start, cue.end, cue.speaker or ""))


def _replace_windows(
    baseline: list[Cue], repaired: list[Cue], windows: list[ConditionedWindow]
) -> list[Cue]:
    retained = [
        cue
        for cue in baseline
        if cue.kind == "singing"
        or not any(
            window.start <= (cue.start + cue.end) / 2 <= window.end
            for window in windows
        )
    ]
    return sorted(
        [*retained, *repaired], key=lambda cue: (cue.start, cue.end, cue.speaker or "")
    )


def _load_cache(path: Path, signature: dict[str, object]) -> list[Cue] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("signature") != signature:
            return None
        return [Cue(**item) for item in value["cues"]]
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        logger.warning("ignoring unreadable conditioned ASR cache: %s", path)
        return None


def _write_cache(path: Path, signature: dict[str, object], cues: list[Cue]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {"signature": signature, "cues": [asdict(cue) for cue in cues]},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
