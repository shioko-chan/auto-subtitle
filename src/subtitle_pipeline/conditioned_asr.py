from __future__ import annotations

import json
import logging
import math
import os
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from .cache import CacheStore, config_snapshot, restore_config
from .audio_analysis import AudioRegion, _overlap_intersections
from .audio_buffer import AudioBuffer
from .config import AudioAnalysisConfig
from .subtitles import Cue, cue_from_mapping

logger = logging.getLogger(__name__)

_MAX_WINDOW_SECONDS = 30.0
_CONDITIONED_CUE_KIND = "conditioned_speech"


@dataclass(frozen=True)
class ConditionedWindow:
    start: float
    end: float
    speakers: tuple[str, ...]
    turns: tuple[AudioRegion, ...]


@dataclass(frozen=True)
class ConditionedASRResult:
    cues: list[Cue]
    evidence: list[dict[str, object]]


@dataclass(frozen=True)
class ConditionedASRTranscription:
    windows: list[ConditionedWindow]
    cues: list[Cue]


def transcribe_long_overlaps(
    diarization: list[AudioRegion],
    audio: AudioBuffer,
    job_dir: Path,
    config: AudioAnalysisConfig,
) -> ConditionedASRTranscription:
    stage = CacheStore(job_dir / "cache.sqlite3").stage("conditioned_asr", lambda: {
        "config": config_snapshot(config),
        "windows": [asdict(window) for window in _conditioned_windows(diarization, audio.duration, config)],
    })
    config = restore_config(config, stage.plan["config"])
    windows = [ConditionedWindow(value["start"], value["end"], tuple(value["speakers"]),
               tuple(AudioRegion(**turn) for turn in value["turns"])) for value in stage.plan["windows"]]
    cached = {index: stage.get(str(index)) for index in range(len(windows))}
    missing = [index for index, value in cached.items() if value is None]

    def save_window(position: int, cues: list[Cue]) -> None:
        index = missing[position]
        raw = [Cue(cue.start, cue.end, cue.text, cue.speaker, _CONDITIONED_CUE_KIND, language="Japanese")
               for cue in cues]
        clean = _without_repetition_hallucinations(raw)
        reason = "repetition_loop" if len(clean) != len(raw) else None
        value = [asdict(cue) for cue in clean]
        stage.put(str(index), value, source="dicow", reason=reason)
        cached[index] = value

    if missing:
        try:
            if config.conditioned_asr_backend == "disabled":
                raise RuntimeError("long overlapping speech requires conditioned ASR, but the backend is disabled")
            _run_dicow(audio, [windows[index] for index in missing], config, save_window)
        except BaseException as exc:
            for index in missing:
                if cached[index] is None:
                    stage.failed(str(index), exc)
            raise
    transcribed = [
        cue_from_mapping(item)
        for index in range(len(windows))
        for item in cached[index]
    ]
    stage.finish([asdict(cue) for cue in transcribed])
    return ConditionedASRTranscription(windows, transcribed)


def reconcile_long_overlaps(
    baseline: list[Cue],
    repaired: list[Cue],
    windows: list[ConditionedWindow],
    *,
    qwen_windows: list[dict[str, object]] | None = None,
) -> ConditionedASRResult:
    evidence = _conditioned_evidence(windows, repaired, qwen_windows or [])
    return ConditionedASRResult(_replace_windows(baseline, repaired, windows), evidence)


def repair_long_overlaps(
    cues: list[Cue],
    diarization: list[AudioRegion],
    audio: AudioBuffer,
    job_dir: Path,
    config: AudioAnalysisConfig,
    qwen_windows: list[dict[str, object]] | None = None,
) -> ConditionedASRResult:
    transcription = transcribe_long_overlaps(diarization, audio, job_dir, config)
    return reconcile_long_overlaps(
        cues,
        transcription.cues,
        transcription.windows,
        qwen_windows=qwen_windows,
    )


def _without_repetition_hallucinations(cues: list[Cue]) -> list[Cue]:
    from .repetition import find_repetition_loop

    usable = []
    for cue in cues:
        repetition = find_repetition_loop(cue.text)
        if repetition is None:
            usable.append(cue)
            continue
        pattern, repeats = repetition.pattern, repetition.repeats
        logger.warning(
            "DiCoW repetition hallucination in %.3f-%.3fs speaker=%s "
            "pattern=%r repeats=%d; preserving Qwen baseline",
            cue.start,
            cue.end,
            cue.speaker or "unknown",
            pattern[:80],
            repeats,
        )
    return usable


def _conditioned_evidence(
    windows: list[ConditionedWindow],
    repaired: list[Cue],
    qwen_windows: list[dict[str, object]],
) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    for window in windows:
        qwen = [
            item
            for record in qwen_windows
            if (item := _qwen_window_evidence(record, window)) is not None
        ]
        dicow = [
            asdict(cue)
            for cue in repaired
            if cue.end > window.start and cue.start < window.end
        ]
        evidence.append(
            {
                "kind": "overlap_reconciliation",
                "start": window.start,
                "end": window.end,
                "qwen_mixed": qwen,
                "dicow": dicow,
                "turns": _window_payload(window)["turns"],
            }
        )
    return evidence


def _qwen_window_evidence(
    record: dict[str, object], window: ConditionedWindow
) -> dict[str, object] | None:
    if (
        float(record.get("core_end", 0.0)) <= window.start
        or float(record.get("core_start", 0.0)) >= window.end
    ):
        return None
    values = record.get("cues")
    if not isinstance(values, list):
        return None
    units = []
    for value in values:
        if not isinstance(value, dict):
            continue
        start = value.get("start")
        end = value.get("end")
        text = value.get("text")
        if (
            isinstance(start, (int, float))
            and isinstance(end, (int, float))
            and isinstance(text, str)
            and float(end) > window.start
            and float(start) < window.end
            and text.strip()
        ):
            units.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "text": text.strip(),
                }
            )
    if not units:
        return None
    return {
        "start": units[0]["start"],
        "end": units[-1]["end"],
        "text": "".join(str(unit["text"]) for unit in units),
        "units": units,
    }


def _conditioned_windows(
    diarization: list[AudioRegion],
    duration: float,
    config: AudioAnalysisConfig,
) -> list[ConditionedWindow]:
    intersections = [
        item
        for item in _overlap_intersections(diarization)
        if item.end - item.start >= config.overlap_conditioned_asr_seconds
    ]
    windows = [
        _expand_overlap(item, diarization, duration, config.overlap_context_seconds)
        for item in intersections
    ]
    merged: list[tuple[float, float]] = []
    for window in sorted(windows, key=lambda item: (item.start, item.end)):
        if merged and window.start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], window.end))
        else:
            merged.append((window.start, window.end))
    bounded = []
    for start, end in merged:
        count = max(1, math.ceil((end - start) / _MAX_WINDOW_SECONDS))
        for index in range(count):
            left = start + (end - start) * index / count
            right = start + (end - start) * (index + 1) / count
            bounded.append(
                _expand_overlap(
                    AudioRegion(left, right, "overlap"), diarization, duration, 0.0
                )
            )
    return bounded


def _expand_overlap(
    overlap: AudioRegion,
    diarization: list[AudioRegion],
    duration: float,
    context_seconds: float,
) -> ConditionedWindow:
    start = round(max(0.0, overlap.start - context_seconds), 3)
    end = round(min(duration, overlap.end + context_seconds), 3)
    turns = tuple(
        AudioRegion(
            max(start, region.start),
            min(end, region.end),
            "speech",
            region.speaker,
            anonymous_speaker=region.anonymous_speaker,
        )
        for region in diarization
        if region.end > start and region.start < end and region.speaker
    )
    speakers = tuple(
        sorted(
            {_condition_label(region) for region in turns if _condition_label(region)}
        )
    )
    return ConditionedWindow(round(start, 3), round(end, 3), speakers, turns)


def _run_dicow(
    audio: AudioBuffer,
    windows: list[ConditionedWindow],
    config: AudioAnalysisConfig,
    on_window: Callable[[int, list[Cue]], None],
) -> None:
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
        "batch_size": config.conditioned_asr_batch_size,
        "audio": audio.descriptor.as_dict(),
        "windows": [_window_payload(window) for window in windows],
    }
    with tempfile.TemporaryFile(mode="w+t") as errors:
        process = subprocess.Popen(
            [uv, "run", "--project", str(project), "python", str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
            start_new_session=True,
        )
        succeeded = False
        try:
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(json.dumps(payload, ensure_ascii=False))
            process.stdin.close()
            seen: set[int] = set()
            complete = False
            for line in process.stdout:
                try:
                    response = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("DiCoW worker returned malformed JSON") from exc
                if not isinstance(response, dict):
                    raise RuntimeError("DiCoW worker returned a non-object result")
                if response.get("error"):
                    raise RuntimeError(f"DiCoW inference failed: {response['error']}")
                if response.get("complete") is True:
                    complete = True
                    continue
                index = response.get("window_index")
                if complete or type(index) is not int or index in seen or not 0 <= index < len(windows):
                    raise RuntimeError("DiCoW worker returned an invalid window index")
                cues = _decode_cues(response.get("cues"), [windows[index]])
                on_window(index, cues)
                seen.add(index)
            return_code = process.wait()
            if return_code != 0:
                errors.seek(0)
                raise RuntimeError("DiCoW worker failed: " + errors.read()[-3000:])
            if not complete or len(seen) != len(windows):
                raise RuntimeError("DiCoW worker ended before all windows completed")
            succeeded = True
        finally:
            if not succeeded or process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()


def _window_payload(window: ConditionedWindow) -> dict[str, object]:
    return {
        "start": window.start,
        "end": window.end,
        "speakers": list(window.speakers),
        "turns": [
            {
                "start": region.start,
                "end": region.end,
                "speaker": _condition_label(region),
            }
            for region in window.turns
        ],
    }


def _decode_cues(value: object, windows: list[ConditionedWindow]) -> list[Cue]:
    if not isinstance(value, list):
        raise TypeError("DiCoW worker returned no cues")
    label_to_character = {
        _condition_label(region): region.speaker
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
            Cue(
                start,
                end,
                text,
                label_to_character.get(label, label),
                _CONDITIONED_CUE_KIND,
                language="Japanese",
            )
        )
    if not cues:
        raise RuntimeError("DiCoW returned no speech for a long-overlap window")
    for window in windows:
        returned = {
            cue.speaker
            for cue in cues
            if cue.end > window.start and cue.start < window.end
        }
        if not returned:
            raise RuntimeError(
                "DiCoW returned no speech for a long-overlap window: "
                f"{window.start:.3f}-{window.end:.3f}s"
            )
        expected = {
            label_to_character.get(speaker, speaker) for speaker in window.speakers
        }
        missing = sorted(expected - returned)
        if missing:
            logger.warning(
                "DiCoW omitted active speakers from %.3f-%.3fs; preserving their "
                "Qwen baseline cues: %s",
                window.start,
                window.end,
                ", ".join(missing),
            )
    return sorted(cues, key=lambda cue: (cue.start, cue.end, cue.speaker or ""))


def _replace_windows(
    baseline: list[Cue], repaired: list[Cue], windows: list[ConditionedWindow]
) -> list[Cue]:
    repaired_by_speaker: dict[str | None, list[Cue]] = {}
    for cue in sorted(repaired, key=lambda item: (item.start, item.end)):
        repaired_by_speaker.setdefault(cue.speaker, []).append(cue)
    retained = []
    for cue in baseline:
        if cue.kind == "singing":
            retained.append(cue)
            continue
        midpoint = (cue.start + cue.end) / 2
        matching = [
            window for window in windows if window.start <= midpoint <= window.end
        ]
        speakers = {cue.speaker} if cue.speaker is not None else {
            _condition_label(turn)
            for window in matching
            for turn in window.turns
            if turn.start <= midpoint < turn.end and _condition_label(turn)
        }
        if not matching or not speakers or not all(
            _covered_by_cues(cue, repaired_by_speaker.get(speaker, []))
            for speaker in speakers
        ):
            retained.append(cue)
    return sorted(
        [*retained, *repaired], key=lambda cue: (cue.start, cue.end, cue.speaker or "")
    )


def _covered_by_cues(baseline: Cue, repaired: list[Cue]) -> bool:
    """Only replace a baseline unit when usable output covers its whole span."""
    cursor = baseline.start
    for cue in repaired:
        if cue.end <= cursor:
            continue
        if cue.start > cursor + 1e-3:
            return False
        cursor = max(cursor, cue.end)
        if cursor >= baseline.end - 1e-3:
            return True
    return False


def _condition_label(region: AudioRegion) -> str | None:
    """Use resolved identity for DiCoW, retaining anonymous labels for audit."""
    return region.speaker or region.anonymous_speaker
