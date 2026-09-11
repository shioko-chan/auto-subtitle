from __future__ import annotations

import json
import logging
import math
import re
import subprocess
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .cache import StageCache, CacheStore, config_snapshot, restore_config
from .audio_analysis import AudioAnalysis, AudioRegion, analyze_audio
from .audio_buffer import AudioBuffer, AudioBufferPool, is_shared_audio_uri
from .commands import require_command, run
from .config import ASRConfig, AudioAnalysisConfig
from .source_language import language_for_text, normalize_source_language
from .subtitles import Cue, write_srt
from .telemetry import stage_metrics
from .repetition import RepetitionLoopError

_CUE_SIDECAR_VERSION = 7
_MIN_RETRY_CHUNK_SECONDS = 15.0
_MIN_SONG_RETRY_CHUNK_SECONDS = 8.0
_MIN_ASR_GENERATION_TOKENS = 128
_ASR_GENERATION_TOKENS_PER_SECOND = 32
_ASR_GENERATION_TOKEN_OVERHEAD = 32
_MIN_SPEAKER_CUE_COVERAGE = 0.30
_MAX_ZERO_DURATION_ALIGNMENT_GAP_SECONDS = 2.0
_SONG_SUPPORT_MAX_SECONDS = 18.0

ASRTextCorrector = Callable[[list[dict[str, object]]], list[dict[str, object]]]
SongCueProcessor = Callable[[list[Cue]], list[Cue]]


class _StaleRuntimeAudio(RuntimeError):
    pass


@dataclass(frozen=True)
class _SpeakerAssignment:
    speaker: str | None
    reason: str | None
    fallback_speaker: str | None
    fallback_distance: float | None


@dataclass(frozen=True)
class _SongCutCandidate:
    time: float
    kind: str
    energy: float = 0.0


def transcribe_with_qwen(
    video: Path,
    destination: Path,
    config: ASRConfig,
    analysis_config: AudioAnalysisConfig | None = None,
    metadata: dict[str, object] | None = None,
    japanese_single_word_list: list[str] | None = None,
    asr_text_corrector: ASRTextCorrector | None = None,
    song_cue_processor: SongCueProcessor | None = None,
) -> Path:
    source_stage = CacheStore(destination.parent / "cache.sqlite3").existing("source_cues")
    saved = source_stage.get("__result__") if source_stage is not None else None
    if saved is not None:
        cues = [Cue(**value) for value in saved["cues"]]
        write_srt(cues, destination)
        _write_cue_sidecar(
            cues, destination.with_suffix(".cues.json"), evidence=saved["evidence"]
        )
        return destination
    japanese_single_word_list = sorted(set(japanese_single_word_list or []))
    duration = _media_duration(video)
    skip_conditioned_asr = False
    if (
        analysis_config is not None
        and analysis_config.skip_dicow_for_single_person_streams
    ):
        from .speakers import title_characters

        title_matches = title_characters(
            metadata or {}, analysis_config.character_styles_file
        )
        skip_conditioned_asr = len(title_matches) == 1
        if skip_conditioned_asr:
            logging.info(
                "skipping DiCoW because the video title identifies one member: %s",
                title_matches[0],
            )
        else:
            logging.info(
                "single-person title check found %d members; keeping DiCoW enabled",
                len(title_matches),
            )
    with AudioBufferPool(video, destination.parent, duration) as audio_pool:
        if analysis_config is not None and analysis_config.enabled:
            with stage_metrics("audio.analysis_total", analysis_config.device):
                analysis = analyze_audio(
                    video,
                    destination.parent,
                    analysis_config,
                    metadata=metadata,
                    audio_pool=audio_pool,
                )
            try:
                with stage_metrics("asr.transcription_total", config.device):
                    return _transcribe_analyzed(
                        video,
                        destination,
                        config,
                        analysis_config,
                        analysis,
                        audio_pool,
                        japanese_single_word_list,
                        asr_text_corrector,
                        song_cue_processor,
                        skip_conditioned_asr,
                    )
            except _StaleRuntimeAudio:
                logging.info(
                    "cached analysis needs ephemeral source tracks for missing ASR; "
                    "recomputing audio analysis"
                )
                with stage_metrics("audio.analysis_total", analysis_config.device):
                    analysis = analyze_audio(
                        video,
                        destination.parent,
                        analysis_config,
                        metadata=metadata,
                        audio_pool=audio_pool,
                        force_runtime_sources=True,
                    )
                with stage_metrics("asr.transcription_total", config.device):
                    return _transcribe_analyzed(
                        video,
                        destination,
                        config,
                        analysis_config,
                        analysis,
                        audio_pool,
                        japanese_single_word_list,
                        asr_text_corrector,
                        song_cue_processor,
                        skip_conditioned_asr,
                    )
        with stage_metrics("asr.transcription_total", config.device):
            return _transcribe_unanalyzed(
                video,
                destination,
                config,
                duration,
                audio_pool,
                japanese_single_word_list,
            )


def _transcribe_unanalyzed(
    video: Path,
    destination: Path,
    config: ASRConfig,
    duration: float,
    audio_pool: AudioBufferPool,
    japanese_single_word_list: list[str] | None = None,
) -> Path:
    chunk_count = max(1, math.ceil(duration / config.chunk_seconds))
    stage = CacheStore(destination.parent / "cache.sqlite3").stage("raw_speech", lambda: {
        "config": config_snapshot(config), "duration": duration,
        "words": japanese_single_word_list, "chunk_count": chunk_count,
    })
    config = restore_config(config, stage.plan["config"])
    duration, chunk_count = stage.plan["duration"], stage.plan["chunk_count"]
    japanese_single_word_list = stage.plan["words"]
    cached_chunks = {str(index): stage.get(str(index)) for index in range(chunk_count)}

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
        model = _load_qwen_model(config, japanese_single_word_list)
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
            cache=stage,
            empty_speech_audit_path=(
                destination.parent / "asr-empty-speech-audit.jsonl"
            ),
            repetition_audit_path=(destination.parent / "asr-repetition-audit.jsonl"),
        )
        cached_chunks[str(index)] = record
        stage.put(str(index), record, kind="aggregate")
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
        all_cues.extend(
            _decode_cached_cues(
                record["cues"], index, raw_text=str(record.get("text") or "")
            )
        )
    if not all_cues:
        raise RuntimeError("Qwen3-ASR did not produce any aligned speech")
    stage.finish([asdict(cue) for cue in all_cues])
    source_stage = stage.store.stage("source_cues", lambda: {})
    source_stage.finish({"cues": [asdict(cue) for cue in all_cues], "evidence": []})
    write_srt(all_cues, destination)
    _write_cue_sidecar(all_cues, destination.with_suffix(".cues.json"), evidence=[])
    logging.info("Qwen3-ASR wrote %d aligned units: %s", len(all_cues), destination)
    return destination


def _raw_song_support_cues(
    records: list[dict[str, object]],
) -> list[Cue]:
    cues: list[Cue] = []
    for record_index, record in enumerate(records):
        text = str(record.get("text") or "").strip()
        start = float(record.get("core_start") or 0.0)
        end = float(record.get("core_end") or start)
        if not text or end <= start:
            continue
        support_group = record.get("window_id", record_index)
        assignment = f"song_alignment_support:{support_group}"
        pieces = [
            value.strip()
            for value in re.findall(r".*?(?:[。！？!?]+|$)", text)
            if value.strip()
        ]
        if not pieces:
            pieces = [text]
        total_characters = sum(len(value) for value in pieces)
        cursor = start
        for piece_index, piece in enumerate(pieces):
            piece_duration = (end - start) * len(piece) / max(1, total_characters)
            piece_end = (
                end if piece_index == len(pieces) - 1 else cursor + piece_duration
            )
            subdivisions = max(
                1, math.ceil((piece_end - cursor) / _SONG_SUPPORT_MAX_SECONDS)
            )
            for subdivision in range(subdivisions):
                text_start = round(len(piece) * subdivision / subdivisions)
                text_end = round(len(piece) * (subdivision + 1) / subdivisions)
                cue_start = cursor + (piece_end - cursor) * subdivision / subdivisions
                cue_end = (
                    cursor + (piece_end - cursor) * (subdivision + 1) / subdivisions
                )
                fragment = piece[text_start:text_end].strip()
                if fragment and cue_end > cue_start:
                    cues.append(
                        Cue(
                            cue_start,
                            cue_end,
                            fragment,
                            kind="speech",
                            speaker_assignment=assignment,
                        )
                    )
            cursor = piece_end
    return cues


def _cached_song_support_cues(
    cached: dict[str, object],
    regions: list[AudioRegion],
    *,
    excluded_indices: set[int] | None = None,
) -> list[Cue]:
    excluded = excluded_indices or set()
    records: list[dict[str, object]] = []
    for index, region in enumerate(regions):
        if index in excluded or region.kind != "speech":
            continue
        record = cached.get(str(index))
        if not isinstance(record, dict) or not isinstance(record.get("cues"), list):
            continue
        records.append(
            {
                "core_start": region.start,
                "core_end": region.end,
                "text": str(record.get("text") or ""),
                "window_id": index,
            }
        )
    return _raw_song_support_cues(records)


def _conditioned_asr_records(
    cues: list[Cue], first_window_id: int
) -> tuple[list[dict[str, object]], list[AudioRegion]]:
    records: list[dict[str, object]] = []
    regions: list[AudioRegion] = []
    for position, cue in enumerate(cues):
        window_id = first_window_id + position
        records.append(
            {
                "core_start": cue.start,
                "core_end": cue.end,
                "language": cue.language or "Japanese",
                "text": cue.text,
                "speaker": cue.speaker,
                "window_id": window_id,
                "asr_source": "dicow",
            }
        )
        regions.append(
            AudioRegion(
                cue.start,
                cue.end,
                "speech",
                cue.speaker,
                overlap=True,
                asr_route="dicow",
            )
        )
    return records, regions


def _conditioned_aligned_cues(
    records: list[dict[str, object]], first_window_id: int
) -> list[Cue]:
    cues: list[Cue] = []
    for position, record in enumerate(records):
        speaker = str(record.get("speaker") or "") or None
        for cue in _decode_cached_cues(
            list(record.get("cues") or []), first_window_id + position
        ):
            cues.append(
                replace(
                    cue,
                    speaker=speaker,
                    kind="conditioned_speech",
                    speaker_assignment="dicow_overlap",
                )
            )
    return sorted(cues, key=lambda cue: (cue.start, cue.end, cue.speaker or ""))


def _transcribe_analyzed(
    video: Path,
    destination: Path,
    config: ASRConfig,
    analysis_config: AudioAnalysisConfig,
    analysis: AudioAnalysis,
    audio_pool: AudioBufferPool,
    japanese_single_word_list: list[str] | None = None,
    asr_text_corrector: ASRTextCorrector | None = None,
    song_cue_processor: SongCueProcessor | None = None,
    skip_conditioned_asr: bool = False,
) -> Path:
    speech_windows = _speech_asr_windows(analysis, config)
    routed_regions = [
        region for region in _analysis_regions(analysis) if region.kind != "speech"
    ]
    regions = sorted([*speech_windows, *routed_regions], key=lambda item: item.start)
    if not regions:
        raise RuntimeError("audio analysis found no speech or singing regions")
    duration = _media_duration(video)
    store = CacheStore(destination.parent / "cache.sqlite3")
    raw_stage = store.stage("raw_speech", lambda: {
        "config": config_snapshot(config), "analysis_config": config_snapshot(analysis_config),
        "regions": [asdict(region) for region in regions], "duration": duration,
        "words": japanese_single_word_list,
    })
    config = restore_config(config, raw_stage.plan["config"])
    analysis_config = restore_config(analysis_config, raw_stage.plan["analysis_config"])
    duration = raw_stage.plan["duration"]
    japanese_single_word_list = raw_stage.plan["words"]
    regions = [AudioRegion(**value) for value in raw_stage.plan["regions"]]
    singing_stage = store.stage("singing_asr", lambda: {
        "config": config_snapshot(config), "regions": [asdict(region) for region in regions],
    })
    alignment_stage = store.stage("speech_alignment", lambda: {
        "config": config_snapshot(config), "words": japanese_single_word_list,
    })
    cached = {str(index): (singing_stage if region.kind == "singing" else alignment_stage).get(str(index))
              for index, region in enumerate(regions)}
    missing = []
    for index, region in enumerate(regions):
        record = cached.get(str(index))
        if not _valid_cached_record(record) or (
            region.kind == "speech"
            and isinstance(record, dict)
            and not _record_timeline_is_healthy(record, region)
        ):
            missing.append(index)
    if missing and any(
        is_shared_audio_uri(regions[index].source_path)
        and not audio_pool.contains(str(regions[index].source_path))
        for index in missing
    ):
        raise _StaleRuntimeAudio("missing ephemeral source audio")
    speaker_timeline = _speaker_assignment_timeline(analysis)

    # Finish ALT and lyric verification before starting the local correction LLM.
    # This keeps all song discovery and acoustic work in the GPU-ASR phase.
    singing_missing = [index for index in missing if regions[index].kind == "singing"]
    singing_model = _load_heart_transcriptor(config) if singing_missing else None
    try:
        for index in singing_missing:
            region = regions[index]
            assert singing_model is not None
            record = _transcribe_song_range(
                singing_model,
                Path(region.source_path) if region.source_path else video,
                None,
                config,
                region,
                label=f"{index:05d}",
                window_config=analysis_config,
                audio_buffer=_region_audio_buffer(region, video, audio_pool),
                repetition_audit_path=(
                    destination.parent / "asr-repetition-audit.jsonl"
                ),
            )
            record["singing_asr_model"] = config.singing_model
            cached[str(index)] = record
            singing_stage.put(str(index), record)
            logging.info(
                "cached HeartTranscriptor region %d/%d speaker=%s cues=%d",
                index + 1,
                len(regions),
                region.speaker or "unknown",
                len(record["cues"]),
            )
    finally:
        if singing_model is not None:
            singing_model.close()
            del singing_model
            _release_cuda()

    singing_cues: list[Cue] = []
    for index, region in enumerate(regions):
        if region.kind != "singing":
            continue
        record = cached.get(str(index))
        if not isinstance(record, dict) or not isinstance(record.get("cues"), list):
            raise RuntimeError(f"analyzed ASR cache is missing singing region {index}")
        singing_cues.extend(
            _decode_cached_cues(
                record["cues"], index, raw_text=str(record.get("text") or "")
            )
        )
    processed_singing_cues = singing_cues
    song_cues_processed = False

    def process_songs(supporting_cues: list[Cue]) -> None:
        nonlocal processed_singing_cues, song_cues_processed
        if song_cues_processed:
            return
        song_cues_processed = True
        if song_cue_processor is None:
            return
        processed = song_cue_processor([*singing_cues, *supporting_cues])
        processed_singing_cues = [cue for cue in processed if cue.kind == "singing"]

    speech_missing = [index for index, region in enumerate(regions) if region.kind == "speech"]
    raw_records: list[dict[str, object]] = []
    if speech_missing:
        raw_chunks = {str(index): raw_stage.get(str(index)) for index in speech_missing}
        raw_uncached = [
            index for index in speech_missing if raw_chunks.get(str(index)) is None
        ]
        raw_model = (
            _load_qwen_model(config, japanese_single_word_list, with_aligner=False)
            if raw_uncached
            else None
        )
        try:
            for batch_start in range(
                0, len(speech_missing), config.max_inference_batch_size
            ):
                indices = speech_missing[
                    batch_start : batch_start + config.max_inference_batch_size
                ]
                uncached = [index for index in indices if raw_chunks.get(str(index)) is None]
                if not uncached:
                    continue
                assert raw_model is not None
                records = _transcribe_raw_speech_batch(
                    raw_model,
                    config,
                    [(index, regions[index]) for index in uncached],
                    cache=raw_stage,
                    media_duration=duration,
                    audio_buffer=audio_pool.main(),
                    repetition_audit_path=(
                        destination.parent / "asr-repetition-audit.jsonl"
                    ),
                )
                for index, record in records.items():
                    raw_chunks[str(index)] = record
                    raw_stage.put(str(index), record,
                                  kind="aggregate" if raw_stage.get(f"raw_retry:{record['core_start']}:{record['core_end']}") else "result",
                                  )
        finally:
            if raw_model is not None:
                del raw_model
                _release_cuda()

        for index in speech_missing:
            value = raw_chunks.get(str(index))
            if not isinstance(value, dict):
                raise RuntimeError(f"raw ASR cache is missing speech region {index}")
            raw_records.append({**value, "window_id": index})

    from .conditioned_asr import (
        ConditionedASRTranscription,
        reconcile_long_overlaps,
        transcribe_long_overlaps,
    )

    if skip_conditioned_asr:
        conditioned_transcription = ConditionedASRTranscription([], [])
    else:
        with stage_metrics("asr.conditioned_transcription", analysis_config.device):
            conditioned_transcription = transcribe_long_overlaps(
                analysis.diarization,
                audio_pool.main(),
                destination.parent,
                analysis_config,
            )

    if speech_missing:
        cached_supporting_cues = _cached_song_support_cues(
            cached,
            regions,
            excluded_indices=set(speech_missing),
        )
        process_songs(
            [
                *cached_supporting_cues,
                *_raw_song_support_cues(raw_records),
            ]
        )
    if not song_cues_processed:
        process_songs(_cached_song_support_cues(cached, regions))

    conditioned_cache = alignment_stage.get("conditioned")
    conditioned_cached_cues: list[Cue] | None = None
    if (
        isinstance(conditioned_cache, dict)
        and isinstance(conditioned_cache.get("cues"), list)
    ):
        conditioned_cached_cues = _decode_cached_cues(
            conditioned_cache["cues"], len(regions)
        )

    conditioned_records: list[dict[str, object]] = []
    conditioned_regions: list[AudioRegion] = []
    conditioned_records, conditioned_regions = _conditioned_asr_records(
        conditioned_transcription.cues, len(regions)
    )

    correctable_records = [
        record
        for record in raw_records
        if regions[int(record["window_id"])].asr_route != "song_speech_fallback"
    ]
    correction_input = sorted(
        [*correctable_records, *conditioned_records],
        key=lambda record: (
            float(record.get("core_start") or 0.0),
            int(record["window_id"]),
        ),
    )
    if asr_text_corrector is not None and correction_input:
        with stage_metrics("asr.text_correction"):
            corrected_records = asr_text_corrector(correction_input)
    else:
        corrected_records = correction_input
    if len(corrected_records) != len(correction_input):
        raise RuntimeError("ASR text corrector changed the speech window count")
    corrected_by_id = {int(record["window_id"]): record for record in corrected_records}
    records = [
        corrected_by_id.get(int(record["window_id"]), record) for record in raw_records
    ]
    corrected_conditioned_records = [
        corrected_by_id.get(int(record["window_id"]), record)
        for record in conditioned_records
    ]

    alignment_records = [*records, *corrected_conditioned_records]
    aligned_records = {}
    pending_alignment = []
    for record in alignment_records:
        index = int(record["window_id"])
        saved = alignment_stage.get(str(index))
        if saved is None:
            pending_alignment.append(record)
        else:
            aligned_records[index] = saved
    if pending_alignment:
        aligner = _load_qwen_aligner(config, japanese_single_word_list)
        try:
            for record in pending_alignment:
                completed = _align_speech_records(
                    aligner, [record], [*regions, *conditioned_regions],
                    audio_pool.main(), config, duration,
                    destination.parent / "asr-speech-quality-audit.jsonl",
                )
                for index, value in completed.items():
                    aligned_records[index] = value
                    alignment_stage.put(str(index), value)
        finally:
            del aligner
            _release_cuda()

    if speech_missing:
        for index in speech_missing:
            region = regions[index]
            record = aligned_records[index]
            for cue in record["cues"]:
                cue["speaker"] = _speaker_for_aligned_cue(
                    float(cue["start"]), float(cue["end"]), speaker_timeline
                )
                cue["kind"] = "speech"
                if region.asr_route == "song_speech_fallback":
                    cue["speaker_assignment"] = "acoustic_phrase_speech_fallback"
            record["window_kind"] = "mixed_speech"
            cached[str(index)] = record
            alignment_stage.put(str(index), record)
            logging.info(
                "cached analyzed ASR region %d/%d kind=%s speaker=%s cues=%d",
                index + 1,
                len(regions),
                region.kind,
                region.speaker or "unknown",
                len(record["cues"]),
            )

    if conditioned_cached_cues is None:
        conditioned_aligned_records = [
            aligned_records[int(record["window_id"])]
            for record in corrected_conditioned_records
        ]
        conditioned_cached_cues = _conditioned_aligned_cues(
            conditioned_aligned_records, len(regions)
        )
        conditioned_cache = {
            "cues": [asdict(cue) for cue in conditioned_cached_cues],
        }
        alignment_stage.put("conditioned", conditioned_cache)

    cues: list[Cue] = list(processed_singing_cues)
    for index in range(len(regions)):
        if regions[index].kind == "singing":
            continue
        record = cached.get(str(index))
        if not isinstance(record, dict) or not isinstance(record.get("cues"), list):
            raise RuntimeError(f"analyzed ASR cache is missing region {index}")
        cues.extend(
            _decode_cached_cues(
                record["cues"], index, raw_text=str(record.get("text") or "")
            )
        )
    if speaker_timeline:
        assigned_cues: list[Cue] = []
        for cue in cues:
            if cue.kind != "speech":
                assigned_cues.append(cue)
                continue
            assignment = _speaker_assignment_for_aligned_cue(
                cue.start, cue.end, speaker_timeline
            )
            assigned_cues.append(
                replace(
                    cue,
                    speaker=assignment.speaker,
                    speaker_assignment=assignment.reason,
                    speaker_fallback=assignment.fallback_speaker,
                    speaker_fallback_distance=assignment.fallback_distance,
                )
            )
        cues = assigned_cues
    cues.sort(key=lambda cue: (cue.start, cue.end, cue.speaker or ""))
    if not cues:
        raise RuntimeError("analyzed Qwen3-ASR did not produce any speech")
    if analysis.diarization:
        with stage_metrics("asr.conditioned_reconciliation"):
            conditioned = reconcile_long_overlaps(
                cues,
                conditioned_cached_cues or [],
                conditioned_transcription.windows,
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
    source_stage = store.stage("source_cues", lambda: {})
    raw_stage.finish(raw_records)
    singing_stage.finish({str(index): cached[str(index)] for index, region in enumerate(regions)
                          if region.kind == "singing"})
    alignment_stage.finish({"speech": {str(index): cached[str(index)] for index, region in enumerate(regions)
                                      if region.kind == "speech"}, "conditioned": conditioned_cache})
    source_stage.put("cues", {"cues": [asdict(cue) for cue in cues], "evidence": evidence})
    source_stage.finish({"cues": [asdict(cue) for cue in cues], "evidence": evidence})
    write_srt(cues, destination)
    _write_cue_sidecar(cues, destination.with_suffix(".cues.json"), evidence=evidence)
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
        if (
            episodes
            and span[0] - episodes[-1][-1][1] <= config.speech_window_max_gap_seconds
        ):
            episodes[-1].append(span)
        else:
            episodes.append([span])
    episodes = _merge_short_speech_episodes(episodes, config)

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
                following = next(
                    (start for start, _ in episode if start >= end - 1e-6), None
                )
                if (
                    following is not None
                    and episode_end - following < _MIN_RETRY_CHUNK_SECONDS
                ):
                    rebalanced = []
                    for candidate in valid:
                        tail_start = next(
                            (
                                start
                                for start, _ in episode
                                if start >= candidate - 1e-6
                            ),
                            None,
                        )
                        if (
                            tail_start is not None
                            and episode_end - tail_start >= _MIN_RETRY_CHUNK_SECONDS
                            and _speech_window_density_ok(
                                tail_start, episode_end, episode, config
                            )
                        ):
                            rebalanced.append(candidate)
                    if rebalanced:
                        end = rebalanced[-1]
            else:
                after_target = [end for end in valid if end - cursor >= target]
                end = (after_target or valid)[0 if after_target else -1]
            windows.append(AudioRegion(round(cursor, 3), round(end, 3), "speech"))
            following = next(
                (start for start, _ in episode if start >= end - 1e-6), None
            )
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


def _merge_short_speech_episodes(
    episodes: list[list[tuple[float, float]]], config: ASRConfig
) -> list[list[tuple[float, float]]]:
    merged = [list(episode) for episode in episodes]
    index = 0
    while index < len(merged) and len(merged) > 1:
        episode = merged[index]
        if episode[-1][1] - episode[0][0] >= _MIN_RETRY_CHUNK_SECONDS:
            index += 1
            continue
        candidates: list[tuple[float, int]] = []
        if index > 0:
            candidates.append((episode[0][0] - merged[index - 1][-1][1], index - 1))
        if index + 1 < len(merged):
            candidates.append((merged[index + 1][0][0] - episode[-1][1], index + 1))
        candidates = [
            candidate
            for candidate in candidates
            if candidate[0] <= config.speech_window_max_silence_seconds
        ]
        if not candidates:
            index += 1
            continue
        _, neighbor = min(candidates, key=lambda item: (item[0], item[1]))
        if neighbor < index:
            merged[neighbor].extend(episode)
            merged.pop(index)
            index = max(0, neighbor)
        else:
            episode.extend(merged.pop(neighbor))
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
    speech = sum(max(0.0, min(end, right) - max(start, left)) for left, right in spans)
    silence = duration - speech
    return (
        silence <= config.speech_window_max_silence_seconds + 1e-6
        and speech / duration >= config.speech_window_min_coverage - 1e-9
    )


def _speaker_for_aligned_cue(
    start: float, end: float, diarization: list[AudioRegion]
) -> str | None:
    return _speaker_assignment_for_aligned_cue(start, end, diarization).speaker


def _speaker_assignment_for_aligned_cue(
    start: float, end: float, diarization: list[AudioRegion]
) -> _SpeakerAssignment:
    duration = end - start
    if duration <= 0:
        return _SpeakerAssignment(None, None, None, None)
    scores = _speaker_overlap_scores(start, end, diarization)
    speaker, _ = _covered_speaker(scores, duration)
    if speaker is not None:
        return _SpeakerAssignment(speaker, "time_overlap", None, None)
    overlap_candidate = _unique_top_speaker(scores)
    if overlap_candidate is not None:
        return _SpeakerAssignment(None, None, overlap_candidate, 0.0)
    nearest_speaker, distance = _nearest_speaker(start, end, diarization)
    return _SpeakerAssignment(None, None, nearest_speaker, distance)


def _nearest_speaker(
    start: float, end: float, diarization: list[AudioRegion]
) -> tuple[str | None, float | None]:
    candidates: list[tuple[float, float, float, str]] = []
    midpoint = (start + end) / 2
    for region in diarization:
        if not region.speaker:
            continue
        distance = max(region.start - end, start - region.end, 0.0)
        center_distance = abs(midpoint - (region.start + region.end) / 2)
        candidates.append(
            (distance, center_distance, -(region.end - region.start), region.speaker)
        )
    if not candidates:
        return None, None
    distance, _, _, speaker = min(candidates)
    return speaker, distance


def _speaker_overlap_scores(
    start: float,
    end: float,
    diarization: list[AudioRegion],
    *,
    padding: float = 0.0,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for region in diarization:
        if not region.speaker:
            continue
        overlap = max(
            0.0,
            min(end, region.end + padding) - max(start, region.start - padding),
        )
        scores[region.speaker or ""] = scores.get(region.speaker or "", 0.0) + overlap
    return {speaker: overlap for speaker, overlap in scores.items() if overlap > 0}


def _unique_top_speaker(scores: dict[str, float]) -> str | None:
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ranked[0][0]


def _covered_speaker(
    scores: dict[str, float], duration: float
) -> tuple[str | None, bool]:
    if not scores:
        return None, False
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    speaker, overlap = ranked[0]
    if len(ranked) > 1:
        return speaker, False
    if overlap < duration * _MIN_SPEAKER_CUE_COVERAGE:
        return None, False
    return speaker, False


def _analysis_regions(analysis: AudioAnalysis) -> list[AudioRegion]:
    singing = sorted(analysis.singing, key=lambda region: region.start)
    speech = list(analysis.speech)
    supplemental_speech = [
        AudioRegion(
            phrase.start,
            phrase.end,
            "speech",
            confidence=phrase.singing_score,
            asr_route="song_speech_fallback",
            speech_confidence=phrase.speech_score,
            music_confidence=phrase.music_score,
        )
        for phrase in analysis.acoustic_phrases
        if phrase.route_speech and phrase.diarization_overlap_seconds < 0.08
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
                overlap_seconds=max(previous.overlap_seconds, region.overlap_seconds),
            )
        else:
            merged.append(region)
    return sorted(
        [*merged, *supplemental_speech, *singing],
        key=lambda item: (item.start, item.end),
    )


def _speaker_assignment_timeline(analysis: AudioAnalysis) -> list[AudioRegion]:
    return analysis.diarization or analysis.speech


def _record_timeline_is_healthy(record: dict[str, object], region: AudioRegion) -> bool:
    if not _valid_cached_record(record):
        return False
    if record.get("skipped_empty") is True:
        return record.get("cues") == []
    text = str(record.get("text") or "")
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
        most_common = max(rounded_starts.count(start) for start in set(rounded_starts))
        if most_common / len(rounded_starts) >= 0.8:
            return False
    return True


def _write_empty_speech_audit(
    path: Path,
    *,
    core_start: float,
    core_end: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now(UTC).isoformat(),
        "reason": "empty_aligned_cues",
        "core_start": core_start,
        "core_end": core_end,
        "duration": core_end - core_start,
        "action": "skip",
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
    logging.warning(
        "recorded and skipped empty speech window: %.3f-%.3fs audit=%s",
        core_start,
        core_end,
        path,
    )


def _write_repetition_discard_audit(
    path: Path | None,
    *,
    kind: str,
    start: float,
    end: float,
    pattern: str,
    repeats: int,
) -> dict[str, object]:
    event: dict[str, object] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "reason": "repetition_loop_at_minimum_window",
        "kind": kind,
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(end - start, 3),
        "pattern": pattern[:200],
        "repeats": repeats,
        "minimal_reproducer": True,
        "action": "discard",
    }
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    logging.warning(
        "discarded minimum %s repetition window %.3f-%.3fs: "
        "pattern=%r repeats=%d audit=%s",
        kind,
        start,
        end,
        pattern[:80],
        repeats,
        path or "record-only",
    )
    return event


def _transcribe_song_range(
    model: Any,
    video: Path,
    chunk_dir: Path | None,
    config: ASRConfig,
    region: AudioRegion,
    *,
    label: str,
    window_config: AudioAnalysisConfig,
    audio_buffer: AudioBuffer | None = None,
    repetition_audit_path: Path | None = None,
) -> dict[str, object]:
    candidates = _song_cut_candidates(region, audio_buffer, window_config)
    window_audit: list[dict[str, object]] = []
    windows = _song_windows(
        region.start,
        region.end,
        window_config.singing_asr_target_seconds,
        window_config.singing_asr_overlap_seconds,
        minimum_seconds=window_config.singing_asr_min_seconds,
        maximum_seconds=window_config.singing_asr_max_seconds,
        search_seconds=window_config.singing_asr_search_seconds,
        candidates=candidates,
        audit=window_audit,
    )
    texts: list[str] = []
    detected_languages: list[str] = []
    window_languages: list[str | None] = []
    resolved_windows: list[tuple[float, float]] = []

    def transcribe_window(
        start: float, end: float, window_label: str
    ) -> list[tuple[float, float, str, str]]:
        chunk_path = (
            chunk_dir / f"song-{window_label}.wav" if chunk_dir is not None else None
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
                raise RuntimeError(
                    "song ASR requires an audio buffer or chunk directory"
                )
            _extract_audio_chunk(video, chunk_path, start=start, duration=end - start)
            audio = str(chunk_path)
        repetition = None
        try:
            with stage_metrics("asr.singing_chunk", config.device):
                results = model.transcribe(
                    audio=audio,
                    context="",
                    language=None,
                    return_time_stamps=False,
                )
            if len(results) != 1:
                raise RuntimeError(f"ALT returned {len(results)} song results")
            result = results[0]
            text = str(getattr(result, "text", "")).strip()
            detected_language = str(getattr(results[0], "language", "")).strip()
        except RepetitionLoopError as exc:
            repetition = (exc.match.pattern, exc.match.repeats)
        finally:
            if chunk_path is not None:
                chunk_path.unlink(missing_ok=True)
        if repetition is not None:
            pattern, repeats = repetition
            child_duration = (end - start) / 2
            logging.warning(
                "ALT generation repetition loop in singing phrase %.3f-%.3fs: "
                "pattern=%r repeats=%d; retrying with shorter song windows",
                start,
                end,
                pattern[:80],
                repeats,
            )
            if child_duration < _MIN_SONG_RETRY_CHUNK_SECONDS:
                diagnostic = _write_repetition_discard_audit(
                    repetition_audit_path,
                    kind="singing_alt",
                    start=start,
                    end=end,
                    pattern=pattern,
                    repeats=repeats,
                )
                diagnostic["cut_reason"] = "repetition_discard"
                window_audit.append(diagnostic)
                return []
            midpoint = start + child_duration
            diagnostic: dict[str, object] = {
                "start": round(start, 3),
                "end": round(end, 3),
                "cut_reason": "repetition_retry",
                "retry_at": round(midpoint, 3),
                "pattern": pattern[:200],
                "repeats": repeats,
            }
            window_audit.append(diagnostic)
            child_audit_start = len(window_audit)
            children = [
                *transcribe_window(start, midpoint, f"{window_label}-0"),
                *transcribe_window(midpoint, end, f"{window_label}-1"),
            ]
            nested = [
                item
                for item in window_audit[child_audit_start:]
                if str(item.get("cut_reason", "")).startswith("repetition_")
            ]
            diagnostic["minimal_reproducer"] = not nested
            diagnostic["resolved_children"] = [
                {
                    "start": round(child_start, 3),
                    "end": round(child_end, 3),
                    "text_preview": child_text[:160],
                    "language": child_language,
                }
                for child_start, child_end, child_text, child_language in children
            ]
            return children
        return [(start, end, text, detected_language)]

    resolved: list[tuple[float, float, str, str]] = []
    for window_index, (start, end) in enumerate(windows):
        resolved.extend(transcribe_window(start, end, f"{label}-{window_index}"))

    for start, end, text, detected_language in resolved:
        if detected_language:
            detected_languages.append(detected_language)
        text = _remove_text_overlap(texts[-1] if texts else "", text)
        texts.append(text)
        window_languages.append(language_for_text(text, detected_language))
        resolved_windows.append((start, end))

    cues: list[Cue] = []
    for index, ((start, end), text) in enumerate(zip(resolved_windows, texts)):
        if not text:
            continue
        owned_start = (
            region.start if index == 0 else (resolved_windows[index - 1][1] + start) / 2
        )
        owned_end = (
            region.end
            if index == len(resolved_windows) - 1
            else (end + resolved_windows[index + 1][0]) / 2
        )
        cues.append(
            Cue(
                owned_start,
                owned_end,
                text,
                region.speaker,
                "singing",
                language=window_languages[index],
            )
        )
    unique_languages = list(dict.fromkeys(detected_languages))
    language = (
        unique_languages[0]
        if len(unique_languages) == 1
        else "mixed"
        if unique_languages
        else ""
    )
    return {
        "core_start": region.start,
        "core_end": region.end,
        "language": language,
        "text": "\n".join(texts).strip(),
        "cues": [asdict(cue) for cue in cues],
        "singing_windows": window_audit,
    }


def _song_windows(
    start: float,
    end: float,
    target_seconds: float,
    overlap_seconds: float,
    *,
    minimum_seconds: float = 20.0,
    maximum_seconds: float = 38.0,
    search_seconds: float = 5.0,
    candidates: list[_SongCutCandidate] | tuple[_SongCutCandidate, ...] = (),
    audit: list[dict[str, object]] | None = None,
) -> list[tuple[float, float]]:
    duration = end - start
    if duration <= maximum_seconds:
        if audit is not None:
            audit.append({"start": start, "end": end, "cut_reason": "region_end"})
        return [(start, end)]
    windows: list[tuple[float, float]] = []
    cursor = start
    while end - cursor > maximum_seconds:
        latest = min(cursor + maximum_seconds, end - minimum_seconds + overlap_seconds)
        earliest = cursor + minimum_seconds
        if end - cursor <= 2 * maximum_seconds - overlap_seconds:
            earliest = max(earliest, end - maximum_seconds + overlap_seconds)
        desired = min(cursor + target_seconds, latest)
        desired = max(earliest, desired)
        search_start = max(earliest, desired - search_seconds)
        search_end = min(latest, desired + search_seconds)
        options = [
            candidate
            for candidate in candidates
            if search_start <= candidate.time <= search_end
        ]
        selected = (
            min(options, key=lambda item: _song_cut_sort_key(item, desired))
            if options
            else None
        )
        window_end = selected.time if selected is not None else desired
        windows.append((cursor, window_end))
        if audit is not None:
            audit.append(
                {
                    "start": round(cursor, 3),
                    "end": round(window_end, 3),
                    "target": round(desired, 3),
                    "cut_reason": selected.kind if selected is not None else "hard",
                }
            )
        cursor = window_end - overlap_seconds
    windows.append((cursor, end))
    if audit is not None:
        audit.append(
            {
                "start": round(cursor, 3),
                "end": round(end, 3),
                "cut_reason": "region_end",
            }
        )
    return windows


def _song_cut_sort_key(
    candidate: _SongCutCandidate, target: float
) -> tuple[int, float, float]:
    priority = {
        "non_singing_gap": 0,
        "vocal_energy_valley": 1,
        "phrase_boundary": 2,
    }.get(candidate.kind, 3)
    distance = abs(candidate.time - target)
    if candidate.kind == "vocal_energy_valley":
        return priority, candidate.energy, distance
    return priority, distance, candidate.energy


def _song_cut_candidates(
    region: AudioRegion,
    audio_buffer: AudioBuffer | None,
    config: AudioAnalysisConfig,
) -> list[_SongCutCandidate]:
    if audio_buffer is None:
        return []
    import numpy as np

    local_start = region.start - region.source_offset
    local_end = region.end - region.source_offset
    samples = np.asarray(audio_buffer.slice(local_start, local_end), dtype=np.float32)
    if not len(samples):
        return []
    sample_rate = audio_buffer.sample_rate
    frame_length = min(max(1, round(0.05 * sample_rate)), len(samples))
    hop_length = min(max(1, round(0.025 * sample_rate)), frame_length)
    frame_starts = np.arange(0, len(samples), hop_length)
    frame_ends = np.minimum(frame_starts + frame_length, len(samples))
    squared_prefix = np.concatenate(
        ([0.0], np.cumsum(np.square(samples), dtype=np.float64))
    )
    rms = np.sqrt(
        (squared_prefix[frame_ends] - squared_prefix[frame_starts])
        / (frame_ends - frame_starts)
    )
    peak = max(float(rms.max()), 1e-9)
    low_energy = rms <= peak * (10 ** (-35 / 20))
    candidates: list[_SongCutCandidate] = []
    long_gap_seconds = max(0.8, config.singing_phrase_silence_seconds)
    run_start: int | None = None
    low_energy_runs: list[tuple[int, int]] = []
    for index, is_low in enumerate(low_energy):
        if is_low and run_start is None:
            run_start = index
        elif not is_low and run_start is not None:
            low_energy_runs.append((run_start, index))
            run_start = None
    if run_start is not None:
        low_energy_runs.append((run_start, len(low_energy)))
    for first_frame, after_last_frame in low_energy_runs:
        gap_start = frame_starts[first_frame] / sample_rate
        last_frame = after_last_frame - 1
        gap_end = min(
            len(samples) / sample_rate,
            (frame_starts[last_frame] + frame_length) / sample_rate,
        )
        duration = gap_end - gap_start
        if duration < config.singing_phrase_silence_seconds:
            continue
        kind = "non_singing_gap" if duration >= long_gap_seconds else "phrase_boundary"
        candidates.append(
            _SongCutCandidate(
                region.start + (gap_start + gap_end) / 2,
                kind,
                -duration,
            )
        )

    if len(rms) >= 3:
        smooth_width = max(1, round(0.4 * sample_rate / hop_length))
        kernel = np.ones(smooth_width, dtype=np.float32) / smooth_width
        smoothed = np.convolve(rms, kernel, mode="same")
        scale = max(float(smoothed.max()), 1e-9)
        for index in range(1, len(smoothed) - 1):
            if (
                smoothed[index] <= smoothed[index - 1]
                and smoothed[index] < smoothed[index + 1]
            ):
                candidates.append(
                    _SongCutCandidate(
                        region.start
                        + (frame_starts[index] + frame_length / 2) / sample_rate,
                        "vocal_energy_valley",
                        float(smoothed[index] / scale),
                    )
                )
    return candidates


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


def _transcribe_range(*args, cache: StageCache | None = None, **kwargs):
    if cache is None:
        return _transcribe_range_compute(*args, **kwargs)
    key = _completed_range_key(kwargs["core_start"], kwargs["core_end"], kwargs["final_chunk"])
    saved = cache.get("range:" + key)
    if saved is not None:
        return saved
    with cache.attempt("range:" + key):
        split = cache.get("split:" + key)
        if split is None:
            result = _transcribe_range_compute(*args, cache=cache, **kwargs)
        else:
            midpoint = split["midpoint"]
            left = _transcribe_range(*args, cache=cache, **{
                **kwargs, "core_end": midpoint, "final_chunk": False, "label": kwargs["label"] + "-0",
            })
            right = _transcribe_range(*args, cache=cache, **{
                **kwargs, "core_start": midpoint, "label": kwargs["label"] + "-1",
            })
            result = {
                "core_start": kwargs["core_start"], "core_end": kwargs["core_end"],
                "language": right["language"] or left["language"],
                "text": f"{left['text']}\n{right['text']}".strip(),
                "cues": [*left["cues"], *right["cues"]],
            }
        cache.put("range:" + key, result,
                  kind="aggregate" if cache.get("split:" + key) is not None else "result")
        return result


def _transcribe_range_compute(
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
    empty_speech_audit_path: Path | None = None,
    repetition_audit_path: Path | None = None,
    cache: StageCache | None = None,
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
        repetition = _generation_repetition(result)
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
                diagnostic = _write_repetition_discard_audit(
                    repetition_audit_path,
                    kind="forced_aligned_speech",
                    start=core_start,
                    end=core_end,
                    pattern=pattern,
                    repeats=repeats,
                )
                record = {
                    "core_start": core_start,
                    "core_end": core_end,
                    "language": str(getattr(result, "language", "")),
                    "text": "",
                    "cues": [],
                    "skipped_empty": True,
                }
                _store_completed_range(
                    completed_ranges,
                    range_key,
                    record,
                    completed_range_callback,
                )
                return record
            midpoint = core_start + child_duration
            if cache is not None:
                cache.remember("split:" + range_key, lambda: {
                    "midpoint": midpoint,
                })
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
                empty_speech_audit_path=empty_speech_audit_path,
                repetition_audit_path=repetition_audit_path,
                cache=cache,
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
                empty_speech_audit_path=empty_speech_audit_path,
                repetition_audit_path=repetition_audit_path,
                cache=cache,
            )
            recovered = {
                "core_start": core_start,
                "core_end": core_end,
                "language": right["language"] or left["language"],
                "text": f"{left['text']}\n{right['text']}".strip(),
                "cues": [*left["cues"], *right["cues"]],
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
        if validate_timeline and not cues:
            record["text"] = ""
            record["skipped_empty"] = True
            if empty_speech_audit_path is not None:
                _write_empty_speech_audit(
                    empty_speech_audit_path,
                    core_start=core_start,
                    core_end=core_end,
                )
            _store_completed_range(
                completed_ranges,
                range_key,
                record,
                completed_range_callback,
            )
            return record
        if validate_timeline and not _record_timeline_is_healthy(
            record, AudioRegion(core_start, core_end, "speech")
        ):
            duration = core_end - core_start
            child_duration = duration / 2
            midpoint = _timeline_retry_split(record, core_start, core_end)
            if child_duration < _MIN_RETRY_CHUNK_SECONDS:
                raise RuntimeError(
                    "Qwen3 forced-alignment timeline remains invalid at minimum "
                    f"speech window {core_start:.3f}-{core_end:.3f}s"
                )
            logging.warning(
                "Qwen3 timeline validation failed in %.3f-%.3fs; splitting at %.3fs",
                core_start,
                core_end,
                midpoint,
            )
            if cache is not None:
                cache.remember("split:" + range_key, lambda: {
                    "midpoint": midpoint, "recovery": "timeline_failure",
                })
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
                empty_speech_audit_path=empty_speech_audit_path,
                repetition_audit_path=repetition_audit_path,
                cache=cache,
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
                empty_speech_audit_path=empty_speech_audit_path,
                repetition_audit_path=repetition_audit_path,
                cache=cache,
            )
            recovered = {
                "core_start": core_start,
                "core_end": core_end,
                "language": right["language"] or left["language"],
                "text": f"{left['text']}\n{right['text']}".strip(),
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


def _transcribe_speech_batch(
    model: Any,
    video: Path,
    config: ASRConfig,
    indexed_regions: list[tuple[int, AudioRegion]],
    *,
    media_duration: float,
    audio_buffer: AudioBuffer,
    completed_ranges: dict[str, object] | None = None,
    completed_range_callback: Callable[[], None] | None = None,
    empty_speech_audit_path: Path | None = None,
) -> dict[int, dict[str, object]]:
    """Transcribe independent speech windows together, isolating bad results."""
    audio_inputs: list[object] = []
    extract_ranges: list[tuple[float, float]] = []
    token_limits: list[int] = []
    for _, region in indexed_regions:
        extract_start = max(0.0, region.start - config.chunk_context_seconds)
        extract_end = min(media_duration, region.end + config.chunk_context_seconds)
        extract_ranges.append((extract_start, extract_end))
        audio_inputs.append(
            (
                audio_buffer.slice(extract_start, extract_end, copy=True),
                audio_buffer.sample_rate,
            )
        )
        token_limits.append(
            _asr_generation_token_limit(config, extract_end - extract_start)
        )

    previous_token_limit = getattr(model, "max_new_tokens", None)
    token_limit_changed = isinstance(previous_token_limit, int)
    if token_limit_changed:
        model.max_new_tokens = max(token_limits)
    logging.info(
        "Qwen3-ASR speech batch size=%d ranges=%s",
        len(indexed_regions),
        ",".join(
            f"{region.start:.1f}-{region.end:.1f}" for _, region in indexed_regions
        ),
    )
    with stage_metrics("asr.forced_aligned_batch", config.device):
        try:
            results = model.transcribe(
                audio=audio_inputs,
                context=config.context,
                language=config.language,
                return_time_stamps=True,
            )
        finally:
            if token_limit_changed:
                model.max_new_tokens = previous_token_limit
    if len(results) != len(indexed_regions):
        raise RuntimeError(
            "Qwen3-ASR returned "
            f"{len(results)} results for {len(indexed_regions)} speech windows"
        )

    records: dict[int, dict[str, object]] = {}
    for position, ((index, region), result) in enumerate(zip(indexed_regions, results)):
        extract_start, _ = extract_ranges[position]
        text = str(getattr(result, "text", "")).strip()
        try:
            cues = _result_to_cues(
                result,
                offset=extract_start,
                keep_start=region.start,
                keep_end=region.end,
                final_chunk=True,
            )
            record: dict[str, object] = {
                "core_start": region.start,
                "core_end": region.end,
                "language": str(getattr(result, "language", "")),
                "text": text,
                "cues": [asdict(cue) for cue in cues],
                "generation_token_limit": token_limits[position],
            }
            valid = _generation_repetition(result) is None and _record_timeline_is_healthy(
                record, region
            )
        except RuntimeError:
            valid = False
            record = {}
        if not valid:
            logging.warning(
                "Qwen3-ASR batch result invalid in %.3f-%.3fs; "
                "retrying that window with recursive recovery",
                region.start,
                region.end,
            )
            record = _transcribe_range(
                model,
                video,
                None,
                config,
                core_start=region.start,
                core_end=region.end,
                media_duration=media_duration,
                final_chunk=True,
                label=f"{index:05d}-batch-retry",
                audio_buffer=audio_buffer,
                validate_timeline=True,
                completed_ranges=completed_ranges,
                completed_range_callback=completed_range_callback,
                empty_speech_audit_path=empty_speech_audit_path,
            )
        else:
            _store_completed_range(
                completed_ranges,
                _completed_range_key(region.start, region.end, True),
                record,
                completed_range_callback,
            )
        records[index] = record
    return records


def _transcribe_raw_speech_batch(
    model: Any,
    config: ASRConfig,
    indexed_regions: list[tuple[int, AudioRegion]],
    *,
    media_duration: float,
    audio_buffer: AudioBuffer,
    repetition_audit_path: Path | None = None,
    cache: StageCache | None = None,
) -> dict[int, dict[str, object]]:
    if cache is not None:
        resumable = [(index, region) for index, region in indexed_regions
                     if cache.get(f"raw_split:{region.start}:{region.end}") is not None
                     or cache.get(f"raw_retry:{region.start}:{region.end}") is not None]
        if resumable:
            resumed = {
                index: _transcribe_raw_range(model, config, region.start, region.end,
                                            media_duration, audio_buffer, repetition_audit_path,
                                            cache=cache)
                for index, region in resumable
            }
            pending = [(index, region) for index, region in indexed_regions if index not in resumed]
            if pending:
                resumed.update(_transcribe_raw_speech_batch(
                    model, config, pending, media_duration=media_duration,
                    audio_buffer=audio_buffer, repetition_audit_path=repetition_audit_path, cache=cache))
            return resumed
    audio_inputs: list[object] = []
    extract_ranges: list[tuple[float, float]] = []
    token_limits: list[int] = []
    for _, region in indexed_regions:
        extract_start = max(0.0, region.start - config.chunk_context_seconds)
        extract_end = min(media_duration, region.end + config.chunk_context_seconds)
        extract_ranges.append((extract_start, extract_end))
        audio_inputs.append(
            (
                audio_buffer.slice(extract_start, extract_end, copy=True),
                audio_buffer.sample_rate,
            )
        )
        token_limits.append(
            _asr_generation_token_limit(config, extract_end - extract_start)
        )
    previous_limit = getattr(model, "max_new_tokens", None)
    changed = isinstance(previous_limit, int)
    if changed:
        model.max_new_tokens = max(token_limits)
    try:
        with stage_metrics("asr.raw_speech_batch", config.device):
            results = model.transcribe(
                audio=audio_inputs,
                context=config.context,
                language=config.language,
                return_time_stamps=False,
            )
    finally:
        if changed:
            model.max_new_tokens = previous_limit
    if len(results) != len(indexed_regions):
        raise RuntimeError(
            f"Qwen3-ASR returned {len(results)} raw results for "
            f"{len(indexed_regions)} speech windows"
        )
    records: dict[int, dict[str, object]] = {}
    for position, ((index, region), result) in enumerate(zip(indexed_regions, results)):
        text = str(getattr(result, "text", "")).strip()
        repetition = _generation_repetition(result)
        if repetition is not None:
            pattern, repeats = repetition
            logging.warning(
                "Qwen3-ASR raw speech repetition loop in "
                "%.3f-%.3fs: pattern=%r repeats=%d; retrying with shorter chunks",
                region.start,
                region.end,
                pattern[:80],
                repeats,
            )
            if cache is not None:
                cache.remember(f"raw_retry:{region.start}:{region.end}", lambda: True)
            records[index] = _transcribe_raw_range(
                model,
                config,
                region.start,
                region.end,
                media_duration,
                audio_buffer,
                repetition_audit_path,
                cache=cache,
            )
            continue
        records[index] = {
            "core_start": region.start,
            "core_end": region.end,
            "extract_start": extract_ranges[position][0],
            "extract_end": extract_ranges[position][1],
            "language": str(getattr(result, "language", "")),
            "detected_languages": [str(getattr(result, "language", ""))],
            "text": text,
            "generation_token_limit": token_limits[position],
        }
        if cache is not None:
            cache.put(str(index), records[index])
    return records


def _transcribe_raw_range(model, config, core_start, core_end, media_duration, audio_buffer,
                          repetition_audit_path=None, *, cache: StageCache | None = None):
    if cache is None:
        return _transcribe_raw_range_compute(model, config, core_start, core_end, media_duration,
                                             audio_buffer, repetition_audit_path)
    key = f"raw_range:{core_start}:{core_end}"
    saved = cache.get(key)
    if saved is not None:
        return saved
    try:
        result = _transcribe_raw_range_compute(model, config, core_start, core_end, media_duration,
                                               audio_buffer, repetition_audit_path, cache=cache)
        cache.put(key, result,
                  kind="aggregate" if cache.get(f"raw_split:{core_start}:{core_end}") else "result")
        return result
    except BaseException as exc:
        cache.failed(key, exc)
        raise


def _transcribe_raw_range_compute(
    model: Any,
    config: ASRConfig,
    core_start: float,
    core_end: float,
    media_duration: float,
    audio_buffer: AudioBuffer,
    repetition_audit_path: Path | None = None,
    cache: StageCache | None = None,
) -> dict[str, object]:
    extract_start = max(0.0, core_start - config.chunk_context_seconds)
    extract_end = min(media_duration, core_end + config.chunk_context_seconds)
    audio = (
        audio_buffer.slice(extract_start, extract_end, copy=True),
        audio_buffer.sample_rate,
    )
    token_limit = _asr_generation_token_limit(config, extract_end - extract_start)
    previous_limit = getattr(model, "max_new_tokens", None)
    changed = isinstance(previous_limit, int)
    if changed:
        model.max_new_tokens = token_limit
    split_key = f"raw_split:{core_start}:{core_end}"
    split_plan = cache.get(split_key) if cache is not None else None
    repetition = None
    if split_plan is None:
        try:
            results = model.transcribe(
                audio=audio,
                context=config.context,
                language=config.language,
                return_time_stamps=False,
            )
        finally:
            if changed:
                model.max_new_tokens = previous_limit
        if len(results) != 1:
            raise RuntimeError("Qwen3-ASR raw retry returned an invalid result count")
        result = results[0]
        text = str(getattr(result, "text", "")).strip()
        repetition = _generation_repetition(result)
    elif changed:
        model.max_new_tokens = previous_limit
    if split_plan is not None or repetition is not None:
        duration = core_end - core_start
        if repetition is not None:
            pattern, repeats = repetition
            if duration / 2 < _MIN_RETRY_CHUNK_SECONDS:
                _write_repetition_discard_audit(
                    repetition_audit_path, kind="raw_speech", start=core_start,
                    end=core_end, pattern=pattern, repeats=repeats,
                )
                return {
                    "core_start": core_start, "core_end": core_end,
                    "extract_start": extract_start, "extract_end": extract_end,
                    "language": "", "detected_languages": [], "text": "",
                    "generation_token_limit": token_limit,
                }
            if cache is not None:
                cache.put(split_key, {"midpoint": (core_start + core_end) / 2}, kind="plan")
        midpoint = (core_start + core_end) / 2
        left = _transcribe_raw_range(
            model,
            config,
            core_start,
            midpoint,
            media_duration,
            audio_buffer,
            repetition_audit_path,
            cache=cache,
        )
        right = _transcribe_raw_range(
            model,
            config,
            midpoint,
            core_end,
            media_duration,
            audio_buffer,
            repetition_audit_path,
            cache=cache,
        )
        detected_languages = [
            str(language)
            for child in (left, right)
            for language in child.get("detected_languages", [child.get("language")])
            if language
        ]
        return {
            "core_start": core_start,
            "core_end": core_end,
            "extract_start": extract_start,
            "extract_end": extract_end,
            "language": right.get("language") or left.get("language") or "",
            "detected_languages": detected_languages,
            "text": f"{left.get('text', '')}\n{right.get('text', '')}".strip(),
            "generation_token_limit": token_limit,

        }
    return {
        "core_start": core_start,
        "core_end": core_end,
        "extract_start": extract_start,
        "extract_end": extract_end,
        "language": str(getattr(result, "language", "")),
        "detected_languages": [str(getattr(result, "language", ""))],
        "text": text,
        "generation_token_limit": token_limit,
    }


def _speech_candidate_quality(
    record: dict[str, object],
    region: AudioRegion,
    *,
    song_context: bool | None = None,
    strong_song_coverage: float = 0.0,
) -> dict[str, object]:
    if song_context is None:
        song_context = region.asr_route == "song_speech_fallback"
    text = str(record.get("text") or "").strip()
    duration = max(region.end - region.start, 1e-6)
    compact = "".join(character for character in text if not character.isspace())
    japanese = sum(
        "\u3040" <= character <= "\u30ff" or "\u3400" <= character <= "\u9fff"
        for character in compact
    )
    hangul = sum("\uac00" <= character <= "\ud7af" for character in compact)
    latin = sum(character.isascii() and character.isalpha() for character in compact)
    scripted = japanese + hangul + latin
    language_values = record.get("detected_languages", [record.get("language")])
    if not isinstance(language_values, list):
        language_values = [language_values]
    languages = sorted(
        {
            normalized.lower()
            for value in language_values
            if (normalized := str(value or "").strip())
        }
    )
    cues = record.get("cues")
    intervals: list[tuple[float, float]] = []
    if isinstance(cues, list):
        for cue in cues:
            if not isinstance(cue, dict):
                continue
            try:
                start = max(region.start, float(cue["start"]))
                end = min(region.end, float(cue["end"]))
            except (KeyError, TypeError, ValueError):
                continue
            if end > start:
                intervals.append((start, end))
    intervals.sort()
    merged: list[list[float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    aligned_seconds = sum(end - start for start, end in merged)
    reasons: list[str] = []
    if text and not _record_timeline_is_healthy(record, region):
        reasons.append("forced_alignment_unhealthy")
    density = len(compact) / duration
    if len(compact) >= 80 and density > 15:
        reasons.append("excessive_text_density")
    if (
        song_context
        and hangul >= 4
        and hangul / max(1, scripted) >= 0.08
        and japanese + latin >= 4
    ):
        reasons.append("unexpected_hangul_mixed_script")
    language_keys = {
        "korean" if value in {"ko", "kor", "korean", "한국어"} else value
        for value in languages
    }
    if (
        song_context
        and (
            len(language_keys) >= 3
            or ("korean" in language_keys and len(language_keys) >= 2)
        )
    ):
        reasons.append("unstable_split_languages")
    if strong_song_coverage >= 0.8:
        reasons.append("dominant_singing_music_region")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "metrics": {
            "duration_seconds": round(duration, 6),
            "compact_characters": len(compact),
            "characters_per_second": round(density, 6),
            "aligned_coverage": round(aligned_seconds / duration, 6),
            "japanese_characters": japanese,
            "hangul_characters": hangul,
            "latin_characters": latin,
            "detected_languages": languages,
            "singing_score": region.confidence,
            "speech_score": region.speech_confidence,
            "music_score": region.music_confidence,
            "song_context": song_context,
            "strong_song_coverage": round(strong_song_coverage, 6),
        },
    }


def _region_coverage(start: float, end: float, regions: list[AudioRegion]) -> float:
    intervals = sorted(
        (max(start, region.start), min(end, region.end))
        for region in regions
        if min(end, region.end) > max(start, region.start)
    )
    covered = 0.0
    current_start = current_end = 0.0
    for left, right in intervals:
        if current_end > current_start and left <= current_end:
            current_end = max(current_end, right)
            continue
        if current_end > current_start:
            covered += current_end - current_start
        current_start, current_end = left, right
    if current_end > current_start:
        covered += current_end - current_start
    return covered


def _write_speech_quality_audit(
    path: Path | None,
    record: dict[str, object],
    region: AudioRegion,
    quality: dict[str, object],
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now(UTC).isoformat(),
        "kind": "speech_quality",
        "start": round(region.start, 3),
        "end": round(region.end, 3),
        "action": "keep" if quality["accepted"] else "discard",
        "text": str(record.get("text") or ""),
        **quality,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def _align_speech_records(
    aligner: Any,
    corrected_records: list[dict[str, object]],
    regions: list[AudioRegion],
    audio_buffer: AudioBuffer,
    config: ASRConfig,
    media_duration: float,
    speech_quality_audit_path: Path | None = None,
) -> dict[int, dict[str, object]]:
    output: dict[int, dict[str, object]] = {}
    for batch_start in range(
        0, len(corrected_records), config.max_inference_batch_size
    ):
        batch = corrected_records[
            batch_start : batch_start + config.max_inference_batch_size
        ]
        empty = [
            record for record in batch if not str(record.get("text") or "").strip()
        ]
        for record in empty:
            index = int(record["window_id"])
            output[index] = {**record, "text": "", "cues": [], "skipped_empty": True}
        batch = [record for record in batch if str(record.get("text") or "").strip()]
        if not batch:
            continue
        audios: list[object] = []
        texts: list[str] = []
        languages: list[str] = []
        for record in batch:
            start = max(0.0, float(record["core_start"]) - config.chunk_context_seconds)
            end = min(
                media_duration,
                float(record["core_end"]) + config.chunk_context_seconds,
            )
            audios.append(
                (audio_buffer.slice(start, end, copy=True), audio_buffer.sample_rate)
            )
            texts.append(str(record.get("text") or ""))
            languages.append(str(record.get("language") or "Japanese"))
        with stage_metrics("asr.corrected_forced_alignment", config.device):
            alignments = aligner.align(audio=audios, text=texts, language=languages)
        if len(alignments) != len(batch):
            raise RuntimeError("Qwen forced aligner result count mismatch")
        for position, (record, alignment) in enumerate(zip(batch, alignments)):
            index = int(record["window_id"])
            region = regions[index]
            extract_start = max(
                0.0, float(record["core_start"]) - config.chunk_context_seconds
            )
            result = SimpleNamespace(
                text=str(record.get("text") or ""),
                language=str(record.get("language") or ""),
                time_stamps=alignment,
            )
            cues = _result_to_cues(
                result,
                offset=extract_start,
                keep_start=region.start,
                keep_end=region.end,
                final_chunk=True,
            )
            aligned: dict[str, object] = {
                **record,
                "cues": [asdict(cue) for cue in cues],
            }
            if not cues:
                aligned["text"] = ""
                aligned["skipped_empty"] = True
            elif not _record_timeline_is_healthy(aligned, region):
                original = str(record.get("original_text") or "").strip()
                corrected = str(record.get("text") or "").strip()
                if original and original != corrected:
                    logging.warning(
                        "corrected forced alignment invalid in %.3f-%.3fs; "
                        "falling back to the original ASR text",
                        region.start,
                        region.end,
                    )
                    fallback_alignment = aligner.align(
                        audio=audios[position],
                        text=original,
                        language=languages[position],
                    )[0]
                    fallback_result = SimpleNamespace(
                        text=original,
                        language=languages[position],
                        time_stamps=fallback_alignment,
                    )
                    fallback_cues = _result_to_cues(
                        fallback_result,
                        offset=extract_start,
                        keep_start=region.start,
                        keep_end=region.end,
                        final_chunk=True,
                    )
                    aligned = {
                        **record,
                        "text": original,
                        "correction_method": "alignment_fallback_original",
                        "cues": [asdict(cue) for cue in fallback_cues],
                    }
                if not _record_timeline_is_healthy(aligned, region):
                    logging.warning(
                        "discarding speech window %.3f-%.3fs because corrected "
                        "and original ASR text both produced unusable forced-"
                        "alignment timelines",
                        region.start,
                        region.end,
                    )
                    aligned = {
                        **record,
                        "text": "",
                        "cues": [],
                        "skipped_empty": True,
                        "correction_method": "alignment_unusable_discarded",
                        "alignment_error": "corrected_and_original_timeline_invalid",
                    }
            if str(aligned.get("text") or "").strip():
                strong_song_regions = [
                    other
                    for other in regions
                    if other.kind == "singing"
                    and float(other.music_confidence or 0.0) >= 0.05
                    and min(region.end, other.end) - max(region.start, other.start) > 0
                ]
                strong_song_coverage = _region_coverage(
                    region.start, region.end, strong_song_regions
                ) / max(1e-6, region.end - region.start)
                song_context = region.asr_route == "song_speech_fallback" or bool(
                    strong_song_regions
                )
                quality = _speech_candidate_quality(
                    aligned,
                    region,
                    song_context=song_context,
                    strong_song_coverage=strong_song_coverage,
                )
                _write_speech_quality_audit(
                    speech_quality_audit_path, aligned, region, quality
                )
                aligned["speech_quality"] = quality
                if not quality["accepted"]:
                    logging.warning(
                        "discarding low-quality speech candidate %.3f-%.3fs: %s",
                        region.start,
                        region.end,
                        ",".join(str(value) for value in quality["reasons"]),
                    )
                    aligned = {
                        **aligned,
                        "text": "",
                        "cues": [],
                        "skipped_empty": True,
                        "correction_method": "speech_quality_gate_discarded",
                    }
            output[index] = aligned
    return output


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


def _generation_repetition(result) -> tuple[str, int] | None:
    match = result.repetition
    return (match.pattern, match.repeats) if match is not None else None


def _load_qwen_model(
    config: ASRConfig,
    japanese_single_word_list: list[str] | None = None,
    *,
    with_aligner: bool = True,
) -> Any:
    try:
        import torch
        from .asr_generation import RepetitionGuardedASRModel
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
        "loading %s%s on %s (%s)",
        config.model,
        f" with aligner {config.aligner_model}" if with_aligner else " without aligner",
        config.device,
        config.dtype,
    )
    kwargs: dict[str, object] = {}
    if with_aligner:
        kwargs = {
            "forced_aligner": config.aligner_model,
            "forced_aligner_kwargs": {
                "dtype": dtype,
                "device_map": config.device,
                "attn_implementation": "sdpa",
                "japanese_single_word_list": japanese_single_word_list or [],
            },
        }
    return RepetitionGuardedASRModel.from_pretrained(
        config.model,
        dtype=dtype,
        device_map=config.device,
        attn_implementation="sdpa",
        max_inference_batch_size=config.max_inference_batch_size,
        max_new_tokens=config.max_new_tokens,
        **kwargs,
    )


def _load_qwen_aligner(
    config: ASRConfig,
    japanese_single_word_list: list[str] | None = None,
) -> Any:
    try:
        import torch
        from qwen_asr import Qwen3ForcedAligner
    except ImportError as exc:
        raise RuntimeError(
            "Qwen3-ASR is not installed; run `uv sync --extra asr`"
        ) from exc
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Qwen forced-aligner device is {config.device}, but CUDA is unavailable"
        )
    dtype = getattr(torch, config.dtype)
    logging.info(
        "loading standalone forced aligner %s on %s (%s)",
        config.aligner_model,
        config.device,
        config.dtype,
    )
    return Qwen3ForcedAligner.from_pretrained(
        config.aligner_model,
        dtype=dtype,
        device_map=config.device,
        attn_implementation="sdpa",
        japanese_single_word_list=japanese_single_word_list or [],
    )


class _HeartTranscriptorAdapter:
    def __init__(
        self,
        pipeline: Any,
        *,
        max_new_tokens: int,
        num_beams: int,
    ) -> None:
        self._pipeline = pipeline
        self._max_new_tokens = max_new_tokens
        self._num_beams = num_beams

    def transcribe(
        self,
        *,
        audio: object,
        context: str = "",
        language: str | None = None,
        return_time_stamps: bool = False,
    ) -> list[SimpleNamespace]:
        import numpy as np
        from transformers import StoppingCriteriaList
        from .alt_generation import ALTRepetitionStoppingCriteria

        del context, language, return_time_stamps
        source: object = audio
        if isinstance(audio, tuple) and len(audio) == 2:
            samples, sample_rate = audio
            source = {
                "raw": np.asarray(samples, dtype=np.float32),
                "sampling_rate": int(sample_rate),
            }
        result = self._pipeline(
            source,
            return_timestamps=False,
            generate_kwargs={
                "stopping_criteria": StoppingCriteriaList([
                    ALTRepetitionStoppingCriteria(self._pipeline.tokenizer),
                ]),
                "max_new_tokens": self._max_new_tokens,
                "num_beams": self._num_beams,
                "task": "transcribe",
                "condition_on_prev_tokens": False,
                "compression_ratio_threshold": 1.8,
                "temperature": (0.0, 0.1, 0.2, 0.4),
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.4,
            },
        )
        if not isinstance(result, dict):
            raise RuntimeError(
                "HeartTranscriptor returned a non-object transcription result"
            )
        text = str(result.get("text") or "").strip()
        return [
            SimpleNamespace(
                text=text,
                language=language_for_text(text, None) or "",
            )
        ]

    def close(self) -> None:
        self._pipeline = None


def _load_heart_transcriptor(config: ASRConfig) -> _HeartTranscriptorAdapter:
    try:
        import torch
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            pipeline,
        )
    except ImportError as exc:
        raise RuntimeError(
            "HeartTranscriptor dependencies are not installed; "
            "run `uv sync --extra asr`"
        ) from exc

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"HeartTranscriptor device is {config.device}, "
            "but PyTorch cannot access CUDA"
        )
    dtype = getattr(torch, config.dtype)
    logging.info(
        "loading singing ASR %s on %s (%s)",
        config.singing_model,
        config.device,
        config.dtype,
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        config.singing_model,
        dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(config.device)
    processor = AutoProcessor.from_pretrained(config.singing_model)
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=dtype,
        device=config.device,
    )
    return _HeartTranscriptorAdapter(
        transcriber,
        max_new_tokens=config.singing_max_new_tokens,
        num_beams=config.singing_num_beams,
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
    detected_language = normalize_source_language(str(getattr(result, "language", "")))
    items = list(getattr(alignment, "items", []) or [])
    if text and not items:
        raise RuntimeError(
            "Qwen3 forced aligner returned no timestamps for non-empty text"
        )
    cues: list[Cue] = []
    pending_prefix = ""
    previous_start = -1.0
    index = 0
    while index < len(items):
        item = items[index]
        fragment = str(item.text)
        start = round(float(item.start_time) + offset, 3)
        end = round(float(item.end_time) + offset, 3)
        if start < previous_start:
            raise RuntimeError("Qwen3 forced aligner returned non-monotonic timestamps")
        previous_start = start
        if fragment.strip() and end <= start:
            fragments = [fragment.strip()]
            positions = [getattr(item, "pos", None)]
            next_index = index + 1
            while next_index < len(items):
                following = items[next_index]
                following_fragment = str(following.text).strip()
                following_start = round(
                    float(following.start_time) + offset,
                    3,
                )
                following_end = round(float(following.end_time) + offset, 3)
                if not following_fragment:
                    next_index += 1
                    continue
                if following_end > following_start:
                    break
                fragments.append(following_fragment)
                positions.append(getattr(following, "pos", None))
                next_index += 1

            merged_fragment = "".join(fragments)
            in_owned_range = start >= keep_start and (
                start < keep_end or (final_chunk and start <= keep_end)
            )
            following_start = None
            if next_index < len(items):
                following_start = round(
                    float(items[next_index].start_time) + offset,
                    3,
                )
            available_gap = (
                following_start - start if following_start is not None else 0.0
            )
            assigned_end = (
                min(following_start, keep_end) if following_start is not None else start
            )
            if (
                in_owned_range
                and 0 < available_gap <= _MAX_ZERO_DURATION_ALIGNMENT_GAP_SECONDS
                and assigned_end > start
            ):
                cues.append(
                    Cue(
                        start,
                        assigned_end,
                        merged_fragment,
                        pos=(
                            str(positions[0])
                            if len(positions) == 1 and positions[0] is not None
                            else None
                        ),
                        language=language_for_text(
                            merged_fragment,
                            detected_language,
                        ),
                    )
                )
            elif in_owned_range and cues:
                previous = cues[-1]
                cues[-1] = replace(
                    previous,
                    text=previous.text + merged_fragment,
                    pos=None,
                    language=language_for_text(
                        previous.text + merged_fragment,
                        detected_language,
                    ),
                )
            elif in_owned_range:
                pending_prefix += merged_fragment
            index = next_index
            continue

        midpoint = (start + end) / 2
        in_owned_range = midpoint >= keep_start and (
            midpoint < keep_end or (final_chunk and midpoint <= keep_end)
        )
        if in_owned_range and fragment.strip():
            if pending_prefix:
                fragment = pending_prefix + fragment.strip()
                pending_prefix = ""
            owned_start = max(start, keep_start)
            owned_end = min(end, keep_end)
            if owned_end > owned_start:
                cues.append(
                    Cue(
                        owned_start,
                        min(keep_end, max(owned_start + 0.08, owned_end)),
                        fragment.strip(),
                        pos=(
                            str(item.pos)
                            if getattr(item, "pos", None) is not None
                            else None
                        ),
                        language=language_for_text(fragment, detected_language),
                    )
                )
        index += 1
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
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
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


def _decode_cached_cues(
    values: list[object], chunk_index: int, *, raw_text: str = ""
) -> list[Cue]:
    cues: list[Cue] = []
    try:
        for value in values:
            if not isinstance(value, dict):
                raise TypeError
            cues.append(
                Cue(
                    start=float(value["start"]),
                    end=float(value["end"]),
                    text=str(value["text"]),
                    speaker=(str(value["speaker"]) if value.get("speaker") else None),
                    kind=str(value.get("kind") or "speech"),
                    boundary_hint=(
                        str(value["boundary_hint"])
                        if value.get("boundary_hint")
                        else None
                    ),
                    pos=str(value["pos"]) if value.get("pos") else None,
                    source_text=(
                        str(value["source_text"]) if value.get("source_text") else None
                    ),
                    speaker_assignment=(
                        str(value["speaker_assignment"])
                        if value.get("speaker_assignment")
                        else None
                    ),
                    speaker_fallback=(
                        str(value["speaker_fallback"])
                        if value.get("speaker_fallback")
                        else None
                    ),
                    speaker_fallback_distance=(
                        float(value["speaker_fallback_distance"])
                        if value.get("speaker_fallback_distance") is not None
                        else None
                    ),
                    language=(
                        str(value["language"]) if value.get("language") else None
                    ),
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Qwen3-ASR cache contains an invalid chunk {chunk_index}"
        ) from exc
    return _add_punctuation_boundary_hints(raw_text, cues) if raw_text else cues


def _add_punctuation_boundary_hints(text: str, cues: list[Cue]) -> list[Cue]:
    """Project trustworthy ASR punctuation onto aligned-unit boundaries."""
    normalized_cues = [
        "".join(
            character
            for character in cue.text
            if not character.isspace()
            and not unicodedata.category(character).startswith("P")
        )
        for cue in cues
    ]
    aligned_text = "".join(normalized_cues)
    normalized_asr = "".join(
        character
        for character in text
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )
    if not aligned_text or not normalized_asr:
        return cues

    matcher = SequenceMatcher(None, normalized_asr, aligned_text, autojunk=False)
    if matcher.ratio() < 0.6:
        return cues
    matching_blocks = [block for block in matcher.get_matching_blocks() if block.size]

    cue_ends: list[int] = []
    cursor = 0
    for fragment in normalized_cues:
        cursor += len(fragment)
        cue_ends.append(cursor)

    hints: dict[int, str] = {}
    consumed = 0
    for character in text:
        if character.isspace():
            continue
        if not unicodedata.category(character).startswith("P"):
            consumed += 1
            continue
        strength = (
            "strong"
            if character in "。！？!?：:"
            else "weak"
            if character in "、,，"
            else None
        )
        if strength is None or consumed <= 0 or consumed >= len(normalized_asr):
            continue
        aligned_position = next(
            (
                block.b + consumed - block.a
                for block in matching_blocks
                if block.size >= 4 and block.a < consumed < block.a + block.size
            ),
            None,
        )
        if aligned_position is None:
            continue
        cue_index = next(
            (index for index, end in enumerate(cue_ends) if end >= aligned_position),
            None,
        )
        if cue_index is not None and (
            strength == "strong" or hints.get(cue_index) is None
        ):
            hints[cue_index] = strength

    return [
        replace(cue, boundary_hint=hints.get(index, cue.boundary_hint))
        for index, cue in enumerate(cues)
    ]


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
                "version": _CUE_SIDECAR_VERSION,
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
    if (
        not isinstance(value, dict)
        or value.get("version") != _CUE_SIDECAR_VERSION
        or not isinstance(value.get("cues"), list)
    ):
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
