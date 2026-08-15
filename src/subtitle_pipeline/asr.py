from __future__ import annotations

import json
import logging
import math
import re
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

from .audio_analysis import AudioAnalysis, AudioRegion, analyze_audio
from .audio_buffer import AudioBuffer, AudioBufferPool, is_shared_audio_uri
from .commands import require_command, run
from .config import ASRConfig, AudioAnalysisConfig
from .subtitles import Cue, write_srt
from .telemetry import stage_metrics

_CACHE_VERSION = 3
_MIN_RETRY_CHUNK_SECONDS = 15.0
_MIN_REPETITION_SPAN_CHARACTERS = 160
_REPETITION_RE = re.compile(r"(.{12,200}?)\1{3,}", re.DOTALL)
_MIN_ASR_GENERATION_TOKENS = 128
_ASR_GENERATION_TOKENS_PER_SECOND = 32
_ASR_GENERATION_TOKEN_OVERHEAD = 32


class _StaleRuntimeAudio(RuntimeError):
    pass


def transcribe_with_qwen(
    video: Path,
    destination: Path,
    config: ASRConfig,
    analysis_config: AudioAnalysisConfig | None = None,
    metadata: dict[str, object] | None = None,
) -> Path:
    duration = _media_duration(video)
    with AudioBufferPool(video, destination.parent, duration) as audio_pool:
        if analysis_config is not None and analysis_config.enabled:
            analysis = analyze_audio(
                video,
                destination.parent,
                analysis_config,
                metadata=metadata,
                audio_pool=audio_pool,
            )
            try:
                return _transcribe_analyzed(
                    video,
                    destination,
                    config,
                    analysis_config,
                    analysis,
                    audio_pool,
                )
            except _StaleRuntimeAudio:
                logging.info(
                    "cached analysis needs ephemeral source tracks for missing ASR; "
                    "recomputing audio analysis"
                )
                analysis = analyze_audio(
                    video,
                    destination.parent,
                    analysis_config,
                    metadata=metadata,
                    audio_pool=audio_pool,
                    force_runtime_sources=True,
                )
                return _transcribe_analyzed(
                    video,
                    destination,
                    config,
                    analysis_config,
                    analysis,
                    audio_pool,
                )
        return _transcribe_unanalyzed(
            video, destination, config, duration, audio_pool
        )


def _transcribe_unanalyzed(
    video: Path,
    destination: Path,
    config: ASRConfig,
    duration: float,
    audio_pool: AudioBufferPool,
) -> Path:
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
        record = cached_chunks.get(str(index))
        if isinstance(record, dict) and _repetition_hallucination(
            str(record.get("text") or "")
        ):
            logging.warning(
                "discarding Qwen3-ASR chunk %d/%d with repeated-loop text",
                index + 1,
                chunk_count,
            )
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

    for index in missing:
        core_start = index * config.chunk_seconds
        core_end = min(duration, core_start + config.chunk_seconds)
        logging.info(
            "Qwen3-ASR chunk %d/%d: %.1f-%.1fs",
            index + 1,
            chunk_count,
            core_start,
            core_end,
        )
        assert model is not None
        record = _transcribe_range(
            model,
            video,
            None,
            config,
            core_start=core_start,
            core_end=core_end,
            media_duration=duration,
            final_chunk=index == chunk_count - 1,
            label=f"{index:05d}",
            audio_buffer=audio_pool.main(),
        )
        cached_chunks[str(index)] = record
        _write_cache(cache_path, cache)
        logging.info(
            "cached Qwen3-ASR chunk %d/%d: %d aligned units",
            index + 1,
            chunk_count,
            len(record["cues"]),
        )

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


def _transcribe_analyzed(
    video: Path,
    destination: Path,
    config: ASRConfig,
    analysis_config: AudioAnalysisConfig,
    analysis: AudioAnalysis,
    audio_pool: AudioBufferPool,
) -> Path:
    speech_windows = _speech_asr_windows(analysis, config)
    routed_regions = [
        region for region in _analysis_regions(analysis) if region.kind != "speech"
    ]
    regions = sorted([*speech_windows, *routed_regions], key=lambda item: item.start)
    if not regions:
        raise RuntimeError("audio analysis found no speech or singing regions")
    duration = _media_duration(video)
    cache_path = destination.parent / "asr-analysis-cache.json"
    signature = {
        **_cache_signature(video, duration, config),
        "analysis_version": 7,
        "analysis_config": asdict(analysis_config),
        "regions": [_analysis_region_signature(region) for region in regions],
    }
    cache = _load_cache(cache_path, signature)
    cached = cache["chunks"]
    assert isinstance(cached, dict)
    completed_ranges = cache.setdefault("completed_ranges", {})
    if not isinstance(completed_ranges, dict):
        completed_ranges = {}
        cache["completed_ranges"] = completed_ranges

    def persist_completed_ranges() -> None:
        _write_cache(cache_path, cache)

    missing = [
        index
        for index in range(len(regions))
        if not _valid_cached_record(cached.get(str(index)))
    ]
    if missing and any(
        is_shared_audio_uri(regions[index].source_path)
        and not audio_pool.contains(str(regions[index].source_path))
        for index in missing
    ):
        raise _StaleRuntimeAudio("missing ephemeral source audio")
    model = _load_qwen_model(config) if missing else None
    for index in missing:
        region = regions[index]
        assert model is not None
        if region.kind == "singing":
            record = _transcribe_song_range(
                model,
                Path(region.source_path) if region.source_path else video,
                None,
                config,
                region,
                label=f"{index:05d}",
                window_seconds=analysis_config.singing_asr_window_seconds,
                overlap_seconds=analysis_config.singing_asr_overlap_seconds,
                audio_buffer=_region_audio_buffer(
                    region, video, audio_pool
                ),
            )
        elif region.kind == "ambiguous":
            record = _transcribe_ambiguous_range(
                model,
                video,
                None,
                config,
                region,
                index=index,
                window_seconds=analysis_config.singing_asr_window_seconds,
                overlap_seconds=analysis_config.singing_asr_overlap_seconds,
                audio_pool=audio_pool,
            )
        else:
            record = _transcribe_range(
                model,
                video,
                None,
                config,
                core_start=region.start,
                core_end=region.end,
                media_duration=duration,
                final_chunk=True,
                label=f"{index:05d}",
                audio_buffer=audio_pool.main(),
                validate_timeline=True,
                completed_ranges=completed_ranges,
                completed_range_callback=persist_completed_ranges,
            )
            for cue in record["cues"]:
                cue["speaker"] = _speaker_for_aligned_cue(
                    float(cue["start"]), float(cue["end"]), analysis.diarization
                )
                cue["kind"] = "speech"
            record["window_kind"] = "mixed_speech"
        cached[str(index)] = record
        _write_cache(cache_path, cache)
        logging.info(
            "cached analyzed ASR region %d/%d kind=%s speaker=%s cues=%d",
            index + 1,
            len(regions),
            region.kind,
            region.speaker or "unknown",
            len(record["cues"]),
        )

    if model is not None:
        del model
        _release_cuda()

    cues: list[Cue] = []
    for index in range(len(regions)):
        record = cached.get(str(index))
        if not isinstance(record, dict) or not isinstance(record.get("cues"), list):
            raise RuntimeError(f"analyzed ASR cache is missing region {index}")
        cues.extend(_decode_cached_cues(record["cues"], index))
    cues.sort(key=lambda cue: (cue.start, cue.end, cue.speaker or ""))
    if not cues:
        raise RuntimeError("analyzed Qwen3-ASR did not produce any speech")
    if analysis.diarization:
        from .conditioned_asr import repair_long_overlaps

        conditioned = repair_long_overlaps(
            cues,
            analysis.diarization,
            audio_pool.main(),
            destination.parent,
            analysis_config,
            qwen_windows=[
                record
                for index in range(len(regions))
                if regions[index].kind == "speech"
                and isinstance((record := cached.get(str(index))), dict)
            ],
        )
        cues = conditioned.cues
        evidence = conditioned.evidence
    else:
        evidence = []
    write_srt(cues, destination)
    _write_cue_sidecar(
        cues, destination.with_suffix(".cues.json"), evidence=evidence
    )
    return destination


def _speech_asr_windows(
    analysis: AudioAnalysis, config: ASRConfig
) -> list[AudioRegion]:
    source = analysis.diarization or analysis.speech
    spans = _union_spans(
        [(region.start, region.end) for region in source if region.kind == "speech"]
    )
    if not spans:
        return []
    episodes: list[list[tuple[float, float]]] = []
    for span in spans:
        if episodes and span[0] - episodes[-1][-1][1] <= config.speech_window_max_gap_seconds:
            episodes[-1].append(span)
        else:
            episodes.append([span])

    target = max(
        0.1,
        config.speech_window_target_seconds - 2 * config.chunk_context_seconds,
    )
    maximum = max(
        target,
        config.speech_window_max_seconds - 2 * config.chunk_context_seconds,
    )
    windows: list[AudioRegion] = []
    for episode in episodes:
        cursor = episode[0][0]
        episode_end = episode[-1][1]
        while cursor < episode_end - 1e-6:
            hard_end = min(episode_end, cursor + maximum)
            boundaries = sorted(
                {
                    min(end, hard_end)
                    for start, end in episode
                    if end > cursor and start < hard_end
                }
                | {hard_end}
            )
            valid = [
                end
                for end in boundaries
                if _speech_window_density_ok(cursor, end, episode, config)
            ]
            if not valid:
                end = hard_end
            elif episode_end <= hard_end:
                end = valid[-1]
            else:
                after_target = [end for end in valid if end - cursor >= target]
                end = (after_target or valid)[0 if after_target else -1]
            windows.append(AudioRegion(round(cursor, 3), round(end, 3), "speech"))
            following = next((start for start, _ in episode if start >= end - 1e-6), None)
            cursor = max(end, following) if following is not None else end
    return windows


def _union_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _speech_window_density_ok(
    start: float,
    end: float,
    spans: list[tuple[float, float]],
    config: ASRConfig,
) -> bool:
    duration = end - start
    if duration <= 0:
        return False
    speech = sum(
        max(0.0, min(end, right) - max(start, left)) for left, right in spans
    )
    silence = duration - speech
    return (
        silence <= config.speech_window_max_silence_seconds + 1e-6
        and speech / duration >= config.speech_window_min_coverage - 1e-9
    )


def _speaker_for_aligned_cue(
    start: float, end: float, diarization: list[AudioRegion]
) -> str | None:
    matches = [
        region
        for region in diarization
        if region.speaker and region.end > start and region.start < end
    ]
    if len({region.speaker for region in matches}) > 1:
        return None
    scores: dict[str, float] = {}
    for region in matches:
        overlap = max(0.0, min(end, region.end) - max(start, region.start))
        scores[region.speaker or ""] = scores.get(region.speaker or "", 0.0) + overlap
    if not scores:
        return None
    speaker, overlap = max(scores.items(), key=lambda item: item[1])
    return speaker if overlap >= (end - start) * 0.5 else None


def _analysis_regions(analysis: AudioAnalysis) -> list[AudioRegion]:
    singing = sorted(analysis.singing, key=lambda region: region.start)
    ambiguous = sorted(analysis.ambiguous, key=lambda region: region.start)
    excluded = sorted([*singing, *ambiguous], key=lambda region: region.start)
    speech = [
        fragment
        for region in analysis.speech
        for fragment in _subtract_singing_regions(region, excluded)
    ]
    separated_tracks: dict[tuple[str | None, str, float], AudioRegion] = {}
    ordinary_speech: list[AudioRegion] = []
    for region in speech:
        if region.overlap and region.source_path:
            key = (region.speaker, region.source_path, region.source_offset)
            previous = separated_tracks.get(key)
            separated_tracks[key] = AudioRegion(
                min(previous.start, region.start) if previous else region.start,
                max(previous.end, region.end) if previous else region.end,
                "speech",
                region.speaker,
                confidence=region.confidence,
                overlap=True,
                source_path=region.source_path,
                source_offset=region.source_offset,
            )
        else:
            ordinary_speech.append(region)

    merged: list[AudioRegion] = []
    for region in sorted(
        [*ordinary_speech, *separated_tracks.values()],
        key=lambda item: (item.start, item.end, item.speaker or ""),
    ):
        if (
            merged
            and merged[-1].kind == "speech"
            and merged[-1].speaker == region.speaker
            and merged[-1].overlap == region.overlap
            and merged[-1].source_path == region.source_path
            and merged[-1].asr_route == region.asr_route
            and merged[-1].overlap_speakers == region.overlap_speakers
            and region.start - merged[-1].end <= 0.5
            and region.end - merged[-1].start <= 30.0
        ):
            previous = merged[-1]
            merged[-1] = replace(
                previous,
                end=max(previous.end, region.end),
                overlap_seconds=max(
                    previous.overlap_seconds, region.overlap_seconds
                ),
            )
        else:
            merged.append(region)
    return sorted(
        [*merged, *singing, *ambiguous],
        key=lambda item: (item.start, item.end),
    )


def _analysis_region_signature(region: AudioRegion) -> dict[str, object]:
    value = asdict(region)
    if is_shared_audio_uri(region.source_path):
        value["source_path"] = "shared-memory"
    return value


def _subtract_singing_regions(
    region: AudioRegion, singing: list[AudioRegion]
) -> list[AudioRegion]:
    fragments = [(region.start, region.end)]
    for song in singing:
        updated: list[tuple[float, float]] = []
        for start, end in fragments:
            if song.end <= start or song.start >= end:
                updated.append((start, end))
                continue
            if start < song.start:
                updated.append((start, song.start))
            if song.end < end:
                updated.append((song.end, end))
        fragments = updated
    return [
        replace(region, start=start, end=end)
        for start, end in fragments
        if end - start >= 0.08
    ]


def _transcribe_ambiguous_range(
    model: Any,
    video: Path,
    chunk_dir: Path | None,
    config: ASRConfig,
    region: AudioRegion,
    *,
    index: int,
    window_seconds: float,
    overlap_seconds: float,
    audio_pool: AudioBufferPool | None = None,
) -> dict[str, object]:
    speech_record: dict[str, object] | None = None
    song_record: dict[str, object] | None = None
    speech_error: RuntimeError | None = None
    song_error: RuntimeError | None = None
    try:
        speech_record = _transcribe_range(
            model,
            video,
            chunk_dir,
            config,
            core_start=region.start,
            core_end=region.end,
            media_duration=_media_duration(video),
            final_chunk=True,
            label=f"ambiguous-speech-{index:05d}",
            audio_buffer=audio_pool.main() if audio_pool is not None else None,
        )
        for cue in speech_record["cues"]:
            cue["speaker"] = region.speaker
            cue["kind"] = "speech"
    except RuntimeError as exc:
        speech_error = exc

    try:
        song_record = _transcribe_song_range(
            model,
            Path(region.source_path) if region.source_path else video,
            chunk_dir,
            config,
            region,
            label=f"ambiguous-song-{index:05d}",
            window_seconds=window_seconds,
            overlap_seconds=overlap_seconds,
            audio_buffer=(
                _region_audio_buffer(region, video, audio_pool)
                if audio_pool is not None
                else None
            ),
        )
    except RuntimeError as exc:
        song_error = exc

    if speech_record is not None and _record_timeline_is_healthy(
        speech_record, region
    ):
        speech_record["ambiguous_route"] = "speech"
        logging.info(
            "ambiguous %.3f-%.3fs selected forced-aligned speech",
            region.start,
            region.end,
        )
        return speech_record
    if song_record is not None and _valid_cached_record(song_record):
        song_record["ambiguous_route"] = "singing"
        logging.warning(
            "ambiguous %.3f-%.3fs selected sentence-level singing ASR",
            region.start,
            region.end,
        )
        return song_record
    details = "; ".join(
        str(error) for error in (speech_error, song_error) if error is not None
    )
    raise RuntimeError(
        f"both ASR routes failed for ambiguous region {region.start:.3f}-"
        f"{region.end:.3f}s{': ' + details if details else ''}"
    )


def _record_timeline_is_healthy(
    record: dict[str, object], region: AudioRegion
) -> bool:
    if not _valid_cached_record(record):
        return False
    text = str(record.get("text") or "")
    if _repetition_hallucination(text):
        return False
    values = record.get("cues")
    if not isinstance(values, list) or not values:
        return False
    try:
        starts = [float(cue["start"]) for cue in values]
        ends = [float(cue["end"]) for cue in values]
    except (KeyError, TypeError, ValueError):
        return False
    if any(end <= start for start, end in zip(starts, ends)):
        return False
    aligned_span = max(ends) - min(starts)
    duration = max(region.end - region.start, 1e-6)
    compact_length = len("".join(text.split()))
    if compact_length >= 20 and aligned_span < min(1.0, duration * 0.25):
        return False
    rounded_starts = [round(start, 3) for start in starts]
    if len(rounded_starts) >= 20:
        most_common = max(
            rounded_starts.count(start) for start in set(rounded_starts)
        )
        if most_common / len(rounded_starts) >= 0.8:
            return False
    return True


def _transcribe_song_range(
    model: Any,
    video: Path,
    chunk_dir: Path | None,
    config: ASRConfig,
    region: AudioRegion,
    *,
    label: str,
    window_seconds: float,
    overlap_seconds: float,
    audio_buffer: AudioBuffer | None = None,
) -> dict[str, object]:
    windows = _song_windows(region.start, region.end, window_seconds, overlap_seconds)
    texts: list[str] = []
    for window_index, (start, end) in enumerate(windows):
        chunk_path = (
            chunk_dir / f"song-{label}-{window_index}.wav"
            if chunk_dir is not None
            else None
        )
        if audio_buffer is not None:
            local_start = start - region.source_offset
            local_end = end - region.source_offset
            audio: object = (
                audio_buffer.slice(local_start, local_end, copy=True),
                audio_buffer.sample_rate,
            )
        else:
            if chunk_path is None:
                raise RuntimeError("song ASR requires an audio buffer or chunk directory")
            _extract_audio_chunk(video, chunk_path, start=start, duration=end - start)
            audio = str(chunk_path)
        try:
            with stage_metrics("asr.singing_chunk", config.device):
                results = model.transcribe(
                    audio=audio,
                    context=config.context,
                    language=config.language,
                    return_time_stamps=False,
                )
        finally:
            if chunk_path is not None:
                chunk_path.unlink(missing_ok=True)
        if len(results) != 1:
            raise RuntimeError(f"Qwen3-ASR returned {len(results)} song results")
        text = str(getattr(results[0], "text", "")).strip()
        if _repetition_hallucination(text):
            raise RuntimeError(
                f"Qwen3-ASR repetition loop in singing phrase {start:.3f}-{end:.3f}s"
            )
        texts.append(_remove_text_overlap(texts[-1] if texts else "", text))

    cues: list[Cue] = []
    for index, ((start, end), text) in enumerate(zip(windows, texts)):
        if not text:
            continue
        owned_start = region.start if index == 0 else (windows[index - 1][1] + start) / 2
        owned_end = region.end if index == len(windows) - 1 else (end + windows[index + 1][0]) / 2
        cues.append(Cue(owned_start, owned_end, text, region.speaker, "singing"))
    language = "Japanese"
    return {
        "core_start": region.start,
        "core_end": region.end,
        "language": language,
        "text": "\n".join(texts).strip(),
        "cues": [asdict(cue) for cue in cues],
    }


def _song_windows(
    start: float, end: float, window_seconds: float, overlap_seconds: float
) -> list[tuple[float, float]]:
    if end - start <= window_seconds:
        return [(start, end)]
    step = window_seconds - overlap_seconds
    windows: list[tuple[float, float]] = []
    cursor = start
    while cursor < end:
        window_end = min(end, cursor + window_seconds)
        windows.append((cursor, window_end))
        if window_end >= end:
            break
        cursor += step
    return windows


def _remove_text_overlap(previous: str, current: str) -> str:
    left = "".join(previous.split())
    right = "".join(current.split())
    maximum = min(len(left), len(right), 80)
    for size in range(maximum, 2, -1):
        if left[-size:] == right[:size]:
            compact_count = 0
            for position, character in enumerate(current):
                if not character.isspace():
                    compact_count += 1
                if compact_count == size:
                    return current[position + 1 :].lstrip()
    return current


def _transcribe_range(
    model: Any,
    video: Path,
    chunk_dir: Path | None,
    config: ASRConfig,
    *,
    core_start: float,
    core_end: float,
    media_duration: float,
    final_chunk: bool,
    label: str,
    audio_buffer: AudioBuffer | None = None,
    validate_timeline: bool = False,
    completed_ranges: dict[str, object] | None = None,
    completed_range_callback: Callable[[], None] | None = None,
) -> dict[str, object]:
    range_key = _completed_range_key(core_start, core_end, final_chunk)
    if completed_ranges is not None:
        cached_range = completed_ranges.get(range_key)
        if (
            isinstance(cached_range, dict)
            and _valid_cached_record(cached_range)
            and (
                not validate_timeline
                or _record_timeline_is_healthy(
                    cached_range, AudioRegion(core_start, core_end, "speech")
                )
            )
        ):
            return cached_range
    extract_start = max(0.0, core_start - config.chunk_context_seconds)
    extract_end = min(media_duration, core_end + config.chunk_context_seconds)
    chunk_path = chunk_dir / f"chunk-{label}.wav" if chunk_dir is not None else None
    if audio_buffer is not None:
        audio: object = (
            audio_buffer.slice(extract_start, extract_end, copy=True),
            audio_buffer.sample_rate,
        )
    else:
        if chunk_path is None:
            raise RuntimeError("ASR requires an audio buffer or chunk directory")
        _extract_audio_chunk(
            video,
            chunk_path,
            start=extract_start,
            duration=extract_end - extract_start,
        )
        audio = str(chunk_path)
    try:
        generation_token_limit = _asr_generation_token_limit(
            config,
            extract_end - extract_start,
        )
        previous_token_limit = getattr(model, "max_new_tokens", None)
        token_limit_changed = isinstance(previous_token_limit, int)
        if token_limit_changed:
            model.max_new_tokens = generation_token_limit
        with stage_metrics("asr.forced_aligned_chunk", config.device):
            try:
                results = model.transcribe(
                    audio=audio,
                    context=config.context,
                    language=config.language,
                    return_time_stamps=True,
                )
            finally:
                if token_limit_changed:
                    model.max_new_tokens = previous_token_limit
        if len(results) != 1:
            raise RuntimeError(
                f"Qwen3-ASR returned {len(results)} results for one audio chunk"
            )
        result = results[0]
        text = str(getattr(result, "text", "")).strip()
        repetition = _repetition_hallucination(text)
        if repetition is not None:
            pattern, repeats = repetition
            duration = core_end - core_start
            child_duration = duration / 2
            logging.warning(
                "Qwen3-ASR repetition loop in %.3f-%.3fs: pattern=%r repeats=%d; "
                "retrying with shorter chunks",
                core_start,
                core_end,
                pattern[:80],
                repeats,
            )
            if child_duration < _MIN_RETRY_CHUNK_SECONDS:
                raise RuntimeError(
                    "Qwen3-ASR repetition loop remains at minimum retry chunk "
                    f"{core_start:.3f}-{core_end:.3f}s"
                )
            midpoint = core_start + child_duration
            left = _transcribe_range(
                model,
                video,
                chunk_dir,
                config,
                core_start=core_start,
                core_end=midpoint,
                media_duration=media_duration,
                final_chunk=False,
                label=f"{label}-0",
                audio_buffer=audio_buffer,
                validate_timeline=validate_timeline,
                completed_ranges=completed_ranges,
                completed_range_callback=completed_range_callback,
            )
            right = _transcribe_range(
                model,
                video,
                chunk_dir,
                config,
                core_start=midpoint,
                core_end=core_end,
                media_duration=media_duration,
                final_chunk=final_chunk,
                label=f"{label}-1",
                audio_buffer=audio_buffer,
                validate_timeline=validate_timeline,
                completed_ranges=completed_ranges,
                completed_range_callback=completed_range_callback,
            )
            recovered = {
                "core_start": core_start,
                "core_end": core_end,
                "language": right["language"] or left["language"],
                "text": f'{left["text"]}\n{right["text"]}'.strip(),
                "cues": [*left["cues"], *right["cues"]],
                "recovered_from_repetition": True,
            }
            _store_completed_range(
                completed_ranges,
                range_key,
                recovered,
                completed_range_callback,
            )
            return recovered

        cues = _result_to_cues(
            result,
            offset=extract_start,
            keep_start=core_start,
            keep_end=core_end,
            final_chunk=final_chunk,
        )
        record: dict[str, object] = {
            "core_start": core_start,
            "core_end": core_end,
            "language": str(getattr(result, "language", "")),
            "text": text,
            "cues": [asdict(cue) for cue in cues],
            "generation_token_limit": generation_token_limit,
        }
        if validate_timeline and not _record_timeline_is_healthy(
            record, AudioRegion(core_start, core_end, "speech")
        ):
            duration = core_end - core_start
            child_duration = duration / 2
            if child_duration < _MIN_RETRY_CHUNK_SECONDS:
                raise RuntimeError(
                    "Qwen3 forced-alignment timeline remains invalid at minimum "
                    f"speech window {core_start:.3f}-{core_end:.3f}s"
                )
            midpoint = _timeline_retry_split(record, core_start, core_end)
            logging.warning(
                "Qwen3 timeline validation failed in %.3f-%.3fs; splitting at %.3fs",
                core_start,
                core_end,
                midpoint,
            )
            left = _transcribe_range(
                model,
                video,
                chunk_dir,
                config,
                core_start=core_start,
                core_end=midpoint,
                media_duration=media_duration,
                final_chunk=False,
                label=f"{label}-timeline-0",
                audio_buffer=audio_buffer,
                validate_timeline=True,
                completed_ranges=completed_ranges,
                completed_range_callback=completed_range_callback,
            )
            right = _transcribe_range(
                model,
                video,
                chunk_dir,
                config,
                core_start=midpoint,
                core_end=core_end,
                media_duration=media_duration,
                final_chunk=final_chunk,
                label=f"{label}-timeline-1",
                audio_buffer=audio_buffer,
                validate_timeline=True,
                completed_ranges=completed_ranges,
                completed_range_callback=completed_range_callback,
            )
            recovered = {
                "core_start": core_start,
                "core_end": core_end,
                "language": right["language"] or left["language"],
                "text": f'{left["text"]}\n{right["text"]}'.strip(),
                "cues": [*left["cues"], *right["cues"]],
                "recovered_from_timeline_failure": True,
            }
            _store_completed_range(
                completed_ranges,
                range_key,
                recovered,
                completed_range_callback,
            )
            return recovered
        _store_completed_range(
            completed_ranges,
            range_key,
            record,
            completed_range_callback,
        )
        return record
    finally:
        if chunk_path is not None:
            chunk_path.unlink(missing_ok=True)


def _completed_range_key(start: float, end: float, final_chunk: bool) -> str:
    return f"{start:.3f}:{end:.3f}:{int(final_chunk)}"


def _store_completed_range(
    completed_ranges: dict[str, object] | None,
    key: str,
    record: dict[str, object],
    callback: Callable[[], None] | None,
) -> None:
    if completed_ranges is None:
        return
    completed_ranges[key] = record
    if callback is not None:
        callback()


def _timeline_retry_split(
    record: dict[str, object], core_start: float, core_end: float
) -> float:
    midpoint = (core_start + core_end) / 2
    values = record.get("cues")
    if not isinstance(values, list):
        return midpoint
    candidates: list[tuple[float, float]] = []
    ordered = sorted(
        (value for value in values if isinstance(value, dict)),
        key=lambda value: float(value.get("start", core_start)),
    )
    for left, right in zip(ordered, ordered[1:]):
        try:
            gap_start = float(left["end"])
            gap_end = float(right["start"])
        except (KeyError, TypeError, ValueError):
            continue
        boundary = (gap_start + gap_end) / 2
        if (
            gap_end > gap_start
            and boundary - core_start >= _MIN_RETRY_CHUNK_SECONDS
            and core_end - boundary >= _MIN_RETRY_CHUNK_SECONDS
        ):
            candidates.append((gap_end - gap_start, boundary))
    if not candidates:
        return midpoint
    _, boundary = max(
        candidates,
        key=lambda item: (item[0], -abs(item[1] - midpoint)),
    )
    return boundary


def _asr_generation_token_limit(config: ASRConfig, audio_seconds: float) -> int:
    duration_limit = (
        math.ceil(max(0.0, audio_seconds) * _ASR_GENERATION_TOKENS_PER_SECOND)
        + _ASR_GENERATION_TOKEN_OVERHEAD
    )
    return min(
        config.max_new_tokens,
        max(_MIN_ASR_GENERATION_TOKENS, duration_limit),
    )


def _region_audio_buffer(
    region: AudioRegion,
    video: Path,
    audio_pool: AudioBufferPool,
) -> AudioBuffer:
    if region.source_path:
        if is_shared_audio_uri(region.source_path):
            return audio_pool.resolve(region.source_path)
        return audio_pool.source(Path(region.source_path))
    return audio_pool.source(video)


def _repetition_hallucination(text: str) -> tuple[str, int] | None:
    normalized = "".join(text.split())
    candidates = [
        match
        for match in _REPETITION_RE.finditer(normalized)
        if match.end() - match.start() >= _MIN_REPETITION_SPAN_CHARACTERS
    ]
    if candidates:
        match = max(candidates, key=lambda item: item.end() - item.start())
        pattern = match.group(1)
        repeats = (match.end() - match.start()) // len(pattern)
        return pattern, repeats
    return None


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


def _release_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


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
    cues: list[Cue] = []
    previous_start = -1.0
    for item in items:
        fragment = str(item.text)
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
            owned_start = max(start, keep_start)
            owned_end = min(end, keep_end)
            if owned_end > owned_start:
                cues.append(
                    Cue(
                        owned_start,
                        min(keep_end, max(owned_start + 0.08, owned_end)),
                        fragment.strip(),
                    )
                )
    return cues


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
    normalized_signature = json.loads(json.dumps(signature, ensure_ascii=False))
    empty: dict[str, object] = {
        "version": _CACHE_VERSION,
        "signature": normalized_signature,
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
        or value.get("signature") != normalized_signature
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
                    str(value["speaker"]) if value.get("speaker") else None,
                    str(value.get("kind") or "speech"),
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Qwen3-ASR cache contains an invalid chunk {chunk_index}"
        ) from exc
    return cues


def _write_cue_sidecar(
    cues: list[Cue],
    path: Path,
    *,
    evidence: list[dict[str, object]] | None = None,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "version": 2,
                "cues": [asdict(cue) for cue in cues],
                "evidence": evidence or [],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_cue_sidecar(path: Path) -> list[Cue]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("cues"), list):
        raise RuntimeError(f"invalid cue sidecar: {path}")
    return _decode_cached_cues(value["cues"], 0)


def read_cue_evidence(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    evidence = value.get("evidence") if isinstance(value, dict) else None
    if evidence is None:
        return []
    if not isinstance(evidence, list) or not all(
        isinstance(item, dict) for item in evidence
    ):
        raise RuntimeError(f"invalid cue evidence sidecar: {path}")
    return evidence


def _valid_cached_record(value: object) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("cues"), list):
        return False
    if _repetition_hallucination(str(value.get("text") or "")) is not None:
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
