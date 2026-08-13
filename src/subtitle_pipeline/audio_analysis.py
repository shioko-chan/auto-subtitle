from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .commands import require_command, run
from .config import AudioAnalysisConfig

logger = logging.getLogger(__name__)

_CACHE_VERSION = 7
_MIN_SEPARATED_SEGMENT_SECONDS = 0.08
_MIN_SEPARATED_PEAK = 1e-4
_MIN_SEPARATED_RMS = 1e-5
_SINGING_LABELS = {
    "chant",
    "child singing",
    "choir",
    "female singing",
    "male singing",
    "rapping",
    "singing",
    "synthetic singing",
    "yodeling",
}
_SPEECH_LABELS = {
    "child speech, kid speaking",
    "conversation",
    "female speech, woman speaking",
    "hubbub, speech noise, speech babble",
    "male speech, man speaking",
    "narration, monologue",
    "speech",
    "speech synthesizer",
}


@dataclass(frozen=True)
class AudioRegion:
    start: float
    end: float
    kind: str
    speaker: str | None = None
    confidence: float | None = None
    overlap: bool = False
    source_path: str | None = None
    source_offset: float = 0.0


@dataclass(frozen=True)
class AudioAnalysis:
    speech: list[AudioRegion]
    singing: list[AudioRegion]
    ambiguous: list[AudioRegion] = field(default_factory=list)


def analyze_audio(
    video: Path,
    job_dir: Path,
    config: AudioAnalysisConfig,
    metadata: dict[str, object] | None = None,
) -> AudioAnalysis:
    """Run reusable VAD/diarization and singing analysis before ASR."""
    job_dir.mkdir(parents=True, exist_ok=True)
    cache_path = job_dir / "audio-analysis.json"
    signature = _signature(video, config, metadata or {})
    cached = _load_cache(cache_path, signature)
    if cached is not None:
        logger.info(
            "audio analysis cache: %d speech turns, %d singing, %d ambiguous",
            len(cached.speech),
            len(cached.singing),
            len(cached.ambiguous),
        )
        return cached

    wav_path = job_dir / "source.analysis.wav"
    _extract_audio(video, wav_path)
    waveform, sample_rate = _load_waveform(wav_path)
    speech = _run_diarization(waveform, sample_rate, config)
    if any(region.overlap for region in speech):
        speech = _run_overlap_separation(
            waveform,
            sample_rate,
            speech,
            config,
            job_dir / "separated-speakers",
        )
    raw_scores = _score_singing_windows(waveform, sample_rate, config)
    raw_candidates = _singing_regions_from_scores(
        raw_scores,
        threshold=config.singing_threshold,
        smoothing_windows=config.singing_smoothing_windows,
        release_seconds=0.0,
        merge_gap_seconds=config.singing_merge_gap_seconds,
    )
    vocals_path = job_dir / "source.vocals.wav"
    singing_windows: list[AudioRegion] = []
    ambiguous_windows: list[AudioRegion] = []
    vocal_waveform = None
    vocal_rate = 0
    if raw_candidates:
        _separate_vocals(wav_path, vocals_path, config.device)
        vocal_waveform, vocal_rate = _load_waveform(vocals_path)
        vocal_scores = _score_singing_windows(vocal_waveform, vocal_rate, config)
        vocal_anchors = _singing_regions_from_scores(
            vocal_scores,
            threshold=config.singing_vocal_threshold,
            smoothing_windows=config.singing_smoothing_windows,
            release_seconds=0.0,
            merge_gap_seconds=config.singing_merge_gap_seconds,
        )
        singing_windows, ambiguous_windows = _arbitrate_singing_regions(
            raw_candidates,
            vocal_anchors,
            speech,
            speech_bgm_coverage=config.singing_speech_bgm_coverage,
            ambiguous_min_seconds=config.singing_ambiguous_min_seconds,
            minimum_singing_seconds=config.singing_min_phrase_seconds,
            release_seconds=config.singing_release_seconds,
        )
    from .speakers import identify_speakers, metadata_character

    known_character = metadata_character(metadata or {}, config.character_styles_file)
    speech = identify_speakers(
        waveform,
        sample_rate,
        speech,
        config,
        known_character=known_character,
        excluded_regions=singing_windows,
    )

    singing = singing_windows
    if singing_windows:
        assert vocal_waveform is not None
        singing = _singing_phrases(
            vocal_waveform,
            vocal_rate,
            singing_windows,
            silence_seconds=config.singing_phrase_silence_seconds,
            minimum_seconds=config.singing_min_phrase_seconds,
        )
        singing = [
            AudioRegion(
                **{
                    **asdict(region),
                    "speaker": known_character,
                    "source_path": str(vocals_path.resolve()),
                }
            )
            for region in singing
        ]

    ambiguous = [
        AudioRegion(
            **{
                **asdict(region),
                "kind": "ambiguous",
                "speaker": _dominant_speaker(region, speech),
                "source_path": str(vocals_path.resolve()),
            }
        )
        for region in ambiguous_windows
    ]
    result = AudioAnalysis(speech=speech, singing=singing, ambiguous=ambiguous)
    payload = {
        "version": _CACHE_VERSION,
        "signature": _signature(video, config, metadata or {}),
        "speech": [asdict(region) for region in speech],
        "singing": [asdict(region) for region in singing],
        "ambiguous": [asdict(region) for region in ambiguous],
    }
    temporary = cache_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(cache_path)
    logger.info(
        "audio analysis wrote %d speech turns, %d singing phrases, "
        "%d ambiguous regions",
        len(speech),
        len(singing),
        len(ambiguous),
    )
    return result


def _run_diarization(
    waveform: Any,
    sample_rate: int,
    config: AudioAnalysisConfig,
) -> list[AudioRegion]:
    try:
        import torch

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module=r"pyannote\.audio\.core\.io")
            from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "speaker diarization is unavailable; run `uv sync --extra asr`"
        ) from exc

    logger.info("loading speaker diarization model %s", config.diarization_model)
    pipeline = Pipeline.from_pretrained(config.diarization_model)
    if pipeline is None:
        raise RuntimeError(
            f"could not load gated diarization model {config.diarization_model}"
        )
    pipeline.to(torch.device(config.device))
    try:
        output = pipeline({"waveform": waveform, "sample_rate": sample_rate})
        annotation = getattr(output, "speaker_diarization", output)
        raw = [
            AudioRegion(
                round(float(segment.start), 3),
                round(float(segment.end), 3),
                "speech",
                str(speaker),
            )
            for segment, _track, speaker in annotation.itertracks(yield_label=True)
            if float(segment.end) > float(segment.start)
        ]
        return _mark_overlaps(raw)
    finally:
        del pipeline
        _release_cuda()


def _run_singing_detection(
    waveform: Any,
    sample_rate: int,
    config: AudioAnalysisConfig,
) -> list[AudioRegion]:
    return _singing_regions_from_scores(
        _score_singing_windows(waveform, sample_rate, config),
        threshold=config.singing_threshold,
        smoothing_windows=config.singing_smoothing_windows,
        release_seconds=config.singing_release_seconds,
        merge_gap_seconds=config.singing_merge_gap_seconds,
    )


def _score_singing_windows(
    waveform: Any,
    sample_rate: int,
    config: AudioAnalysisConfig,
) -> list[AudioRegion]:
    try:
        import torch
        from transformers import ASTForAudioClassification, AutoFeatureExtractor
    except ImportError as exc:
        raise RuntimeError(
            "singing detection is unavailable; run `uv sync --extra asr`"
        ) from exc

    mono_tensor = waveform.mean(dim=0).detach().cpu()
    target_rate = 16000
    if sample_rate != target_rate:
        import torchaudio.functional as audio_functional

        mono_tensor = audio_functional.resample(mono_tensor, sample_rate, target_rate)
        sample_rate = target_rate
    mono = mono_tensor.numpy()
    window_samples = max(1, round(config.singing_window_seconds * sample_rate))
    stride_samples = max(1, round(config.singing_stride_seconds * sample_rate))
    starts = list(range(0, max(1, len(mono) - window_samples + 1), stride_samples))
    if not starts or starts[-1] + window_samples < len(mono):
        starts.append(max(0, len(mono) - window_samples))

    logger.info("loading singing detector %s", config.singing_model)
    extractor = AutoFeatureExtractor.from_pretrained(config.singing_model)
    model = ASTForAudioClassification.from_pretrained(config.singing_model).to(
        config.device
    )
    model.eval()
    labels = {
        int(index): str(label).casefold()
        for index, label in model.config.id2label.items()
    }
    singing_ids = [index for index, label in labels.items() if label in _SINGING_LABELS]
    speech_ids = [index for index, label in labels.items() if label in _SPEECH_LABELS]
    if not singing_ids:
        raise RuntimeError("AST singing detector exposes no recognized singing labels")

    scored_windows: list[AudioRegion] = []
    try:
        for batch_start in range(0, len(starts), 16):
            batch_offsets = starts[batch_start : batch_start + 16]
            audio = [mono[offset : offset + window_samples] for offset in batch_offsets]
            inputs = extractor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(config.device) for key, value in inputs.items()}
            with torch.inference_mode():
                probabilities = torch.softmax(model(**inputs).logits, dim=-1).cpu()
            for offset, row in zip(batch_offsets, probabilities):
                singing_score = sum(float(row[index]) for index in singing_ids)
                speech_score = sum(float(row[index]) for index in speech_ids)
                score = _singing_evidence_score(singing_score, speech_score)
                scored_windows.append(
                    AudioRegion(
                        round(offset / sample_rate, 3),
                        round(min(len(mono), offset + window_samples) / sample_rate, 3),
                        "singing",
                        confidence=round(score, 4),
                    )
                )
    finally:
        del model
        _release_cuda()
    return scored_windows


def _singing_evidence_score(singing_score: float, speech_score: float) -> float:
    return max(0.0, singing_score - speech_score)


def _arbitrate_singing_regions(
    raw_candidates: list[AudioRegion],
    vocal_anchors: list[AudioRegion],
    speech: list[AudioRegion],
    *,
    speech_bgm_coverage: float,
    release_seconds: float,
    ambiguous_min_seconds: float = 15.0,
    minimum_singing_seconds: float = 30.0,
) -> tuple[list[AudioRegion], list[AudioRegion]]:
    """Confirm vocal singing and retain uncertain candidates for dual ASR."""
    confirmed: list[AudioRegion] = []
    ambiguous: list[AudioRegion] = []
    episodes = _merge_regions(raw_candidates, release_seconds)
    for candidate in episodes:
        matching_vocals = [
            vocal for vocal in vocal_anchors if _overlap_duration(candidate, vocal) > 0
        ]
        if matching_vocals:
            if candidate.end - candidate.start < minimum_singing_seconds:
                ambiguous.append(candidate)
                continue
            confirmed.append(
                AudioRegion(
                    candidate.start,
                    candidate.end,
                    "singing",
                    confidence=max(
                        float(region.confidence or 0.0) for region in matching_vocals
                    ),
                )
            )
            continue
        coverage = _region_coverage(candidate, speech)
        if (
            coverage < speech_bgm_coverage
            or candidate.end - candidate.start >= ambiguous_min_seconds
        ):
            ambiguous.append(candidate)

    ambiguous = [
        fragment
        for candidate in ambiguous
        for fragment in _subtract_regions(
            candidate, [(region.start, region.end) for region in confirmed]
        )
    ]
    return confirmed, ambiguous


def _overlap_duration(left: AudioRegion, right: AudioRegion) -> float:
    return max(0.0, min(left.end, right.end) - max(left.start, right.start))


def _region_coverage(region: AudioRegion, others: list[AudioRegion]) -> float:
    intersections = [
        AudioRegion(
            max(region.start, other.start),
            min(region.end, other.end),
            "speech",
        )
        for other in others
        if _overlap_duration(region, other) > 0
    ]
    covered = sum(item.end - item.start for item in _merge_regions(intersections, 0))
    return covered / max(region.end - region.start, 1e-6)


def _dominant_speaker(region: AudioRegion, speech: list[AudioRegion]) -> str | None:
    durations: dict[str, float] = {}
    for turn in speech:
        if turn.speaker:
            durations[turn.speaker] = durations.get(turn.speaker, 0.0) + (
                _overlap_duration(region, turn)
            )
    return max(durations, key=durations.get) if durations else None


def _singing_regions_from_scores(
    windows: list[AudioRegion],
    *,
    threshold: float,
    smoothing_windows: int,
    release_seconds: float,
    merge_gap_seconds: float = 0.0,
) -> list[AudioRegion]:
    """Turn noisy AST scores into song episodes with release hysteresis."""
    if not windows:
        return []

    radius = smoothing_windows // 2
    smoothed: list[AudioRegion] = []
    for index, window in enumerate(windows):
        nearby = windows[max(0, index - radius) : index + radius + 1]
        scores = sorted(float(item.confidence or 0.0) for item in nearby)
        scores.extend([0.0] * (smoothing_windows - len(scores)))
        scores.sort()
        score = scores[len(scores) // 2]
        smoothed.append(
            AudioRegion(
                window.start,
                window.end,
                "singing",
                confidence=round(score, 4),
            )
        )

    anchors = [
        raw
        for raw, smooth in zip(windows, smoothed)
        if float(smooth.confidence or 0.0) >= threshold
    ]
    # Offline release hysteresis: a negative run closes the song only when no
    # later singing anchor arrives within the configured release interval.
    episodes = _merge_regions(anchors, release_seconds)
    return _merge_regions(episodes, merge_gap_seconds)


def _run_overlap_separation(
    waveform: Any,
    sample_rate: int,
    diarized_regions: list[AudioRegion],
    config: AudioAnalysisConfig,
    output_dir: Path,
) -> list[AudioRegion]:
    """Separate two-speaker overlaps while retaining recording timestamps."""
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "overlap separation is unavailable; install the ASR optional dependencies"
        ) from exc

    overlap_spans = _overlap_spans(
        diarized_regions, float(waveform.shape[-1]) / sample_rate
    )
    if not overlap_spans:
        return diarized_regions
    eligible_spans: list[tuple[int, float, float]] = []
    for span_index, (span_start, span_end) in enumerate(overlap_spans):
        concurrent = _maximum_concurrent_speakers(
            diarized_regions, span_start, span_end
        )
        if concurrent > 2:
            logger.warning(
                "overlap %.3f-%.3f has %d concurrent speakers; "
                "MossFormer2 supports two, retaining original audio",
                span_start,
                span_end,
                concurrent,
            )
            continue
        eligible_spans.append((span_index, span_start, span_end))
    if not eligible_spans:
        return diarized_regions

    logger.info(
        "running overlap separator %s for %d two-speaker regions",
        config.overlap_separation_model,
        len(eligible_spans),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        outputs = _run_mossformer2_worker(
            waveform,
            sample_rate,
            eligible_spans,
            config,
            output_dir,
        )
    except RuntimeError as exc:
        logger.warning(
            "MossFormer2 separation failed; retaining original audio: %s", exc
        )
        return diarized_regions

    separated: list[AudioRegion] = []
    successful_spans: list[tuple[float, float]] = []
    for span_index, span_start, span_end in eligible_spans:
        paths = outputs.get(span_index, [])
        usable: list[Path] = []
        for path in paths:
            try:
                data, rate = sf.read(path, always_2d=True, dtype="float32")
            except (OSError, RuntimeError) as exc:
                logger.warning("could not read MossFormer2 output %s: %s", path, exc)
                continue
            source = data.mean(axis=1)
            duration = len(source) / max(int(rate), 1)
            if _separated_source_is_usable(
                source,
                [AudioRegion(0.0, duration, "speech")],
                int(rate),
            ):
                usable.append(path)
        if len(usable) != 2:
            logger.warning(
                "MossFormer2 overlap %d returned %d usable tracks; "
                "retaining original audio",
                span_index,
                len(usable),
            )
            continue
        successful_spans.append((span_start, span_end))
        separated.extend(
            AudioRegion(
                round(span_start, 3),
                round(span_end, 3),
                "speech",
                f"OVERLAP_{span_index:05d}_SOURCE_{source_index}",
                overlap=True,
                source_path=str(path.resolve()),
                source_offset=span_start,
            )
            for source_index, path in enumerate(usable)
        )

    if not successful_spans:
        return diarized_regions
    failed_spans = [
        span
        for span in overlap_spans
        if not any(span == successful for successful in successful_spans)
    ]
    untouched = [
        AudioRegion(
            **{
                **asdict(fragment),
                "overlap": any(
                    min(fragment.end, failed_end)
                    > max(fragment.start, failed_start)
                    for failed_start, failed_end in failed_spans
                ),
            }
        )
        for region in diarized_regions
        for fragment in _subtract_regions(region, successful_spans)
    ]
    return sorted([*untouched, *separated], key=lambda item: (item.start, item.end))


def _run_mossformer2_worker(
    waveform: Any,
    sample_rate: int,
    spans: list[tuple[int, float, float]],
    config: AudioAnalysisConfig,
    output_dir: Path,
) -> dict[int, list[Path]]:
    import soundfile as sf

    uv = shutil.which("uv")
    project = Path(config.overlap_separation_worker_project).resolve()
    worker = project / "worker.py"
    if uv is None or not worker.is_file():
        raise RuntimeError(f"MossFormer2 worker is unavailable at {worker}")

    with tempfile.TemporaryDirectory(prefix="mossformer2-input-") as temp:
        input_dir = Path(temp)
        items: list[dict[str, object]] = []
        for span_index, span_start, span_end in spans:
            start_sample = round(span_start * sample_rate)
            end_sample = round(span_end * sample_rate)
            mono = (
                waveform[:, start_sample:end_sample].mean(dim=0).detach().cpu().numpy()
            )
            input_path = input_dir / f"overlap-{span_index:05d}.wav"
            output_paths = [
                output_dir / f"mossformer2-overlap-{span_index:05d}-source-{index}.wav"
                for index in range(2)
            ]
            sf.write(input_path, mono, sample_rate)
            items.append(
                {
                    "id": span_index,
                    "input_path": str(input_path),
                    "output_paths": [str(path.resolve()) for path in output_paths],
                }
            )
        payload = {
            "model": config.overlap_separation_model,
            "items": items,
        }
        environment = os.environ.copy()
        if config.device.startswith("cuda:"):
            environment["CUDA_VISIBLE_DEVICES"] = config.device.split(":", 1)[1]
        else:
            environment["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(
            [
                uv,
                "run",
                "--frozen",
                "--project",
                str(project),
                "python",
                str(worker),
            ],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
            cwd=project,
            env=environment,
        )
    if result.returncode != 0:
        raise RuntimeError(
            "MossFormer2 worker failed: " + (result.stderr or result.stdout)[-3000:]
        )
    try:
        response = json.loads(result.stdout)
        records = response["items"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError("MossFormer2 worker returned malformed JSON") from exc
    if not isinstance(records, list):
        raise RuntimeError("MossFormer2 worker returned malformed items")
    outputs: dict[int, list[Path]] = {}
    for record in records:
        if not isinstance(record, dict) or record.get("error"):
            if isinstance(record, dict):
                logger.warning(
                    "MossFormer2 overlap %s failed: %s",
                    record.get("id"),
                    record.get("error"),
                )
            continue
        try:
            span_index = int(record["id"])
            paths = [Path(str(path)) for path in record["output_paths"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "MossFormer2 worker returned malformed output paths"
            ) from exc
        if len(paths) == 2 and all(path.is_file() for path in paths):
            outputs[span_index] = paths
    return outputs


def _separated_source_is_usable(
    source: Any, segments: list[Any], sample_rate: int
) -> bool:
    segments = [
        segment
        for segment in segments
        if float(segment.end) - float(segment.start) >= _MIN_SEPARATED_SEGMENT_SECONDS
    ]
    if not segments:
        return False
    squared_sum = 0.0
    sample_count = 0
    peak = 0.0
    for segment in segments:
        start = max(0, round(float(segment.start) * sample_rate))
        end = min(len(source), round(float(segment.end) * sample_rate))
        if end <= start:
            continue
        chunk = source[start:end]
        absolute = abs(chunk)
        peak = max(peak, float(absolute.max()))
        squared_sum += float((chunk * chunk).sum())
        sample_count += len(chunk)
    if sample_count == 0:
        return False
    rms = (squared_sum / sample_count) ** 0.5
    return peak >= _MIN_SEPARATED_PEAK and rms >= _MIN_SEPARATED_RMS


def _separate_vocals(source: Path, destination: Path, device: str) -> None:
    if destination.is_file():
        return
    try:
        import soundfile as sf
        from demucs.api import Separator
    except ImportError as exc:
        raise RuntimeError(
            "vocal separation requires the ASR optional dependencies"
        ) from exc

    logger.info("separating vocals with Demucs htdemucs")
    separator = Separator(model="htdemucs", device=device, progress=True)
    try:
        _origin, stems = separator.separate_audio_file(source)
        vocals = stems.get("vocals")
        if vocals is None:
            raise RuntimeError("Demucs did not return a vocals stem")
        data = vocals.detach().cpu().numpy().T
        sf.write(destination, data, separator.samplerate)
    finally:
        del separator
        _release_cuda()


def _singing_phrases(
    waveform: Any,
    sample_rate: int,
    regions: list[AudioRegion],
    *,
    silence_seconds: float,
    minimum_seconds: float = 30.0,
) -> list[AudioRegion]:
    """Use silence in the separated vocal stem as approximate lyric-line edges."""
    import librosa

    mono = waveform.mean(dim=0).detach().cpu().numpy()
    phrases: list[AudioRegion] = []
    for region in regions:
        region_phrases: list[AudioRegion] = []
        offset = max(0, round(region.start * sample_rate))
        end = min(len(mono), round(region.end * sample_rate))
        intervals = librosa.effects.split(
            mono[offset:end], top_db=35, frame_length=2048, hop_length=256
        )
        current: list[int] | None = None
        for local_start, local_end in intervals:
            if current is None:
                current = [int(local_start), int(local_end)]
                continue
            gap = (int(local_start) - current[1]) / sample_rate
            if gap <= silence_seconds:
                current[1] = int(local_end)
            else:
                region_phrases.append(
                    _vocal_interval(offset, current, sample_rate, region)
                )
                current = [int(local_start), int(local_end)]
        if current is not None:
            region_phrases.append(_vocal_interval(offset, current, sample_rate, region))
        blocks = _coalesce_singing_phrases(
            region_phrases,
            minimum_seconds=minimum_seconds,
        )
        if (
            blocks
            and len(blocks) == 1
            and blocks[0].end - blocks[0].start < minimum_seconds
        ):
            blocks = [
                AudioRegion(
                    region.start,
                    region.end,
                    "singing",
                    confidence=blocks[0].confidence,
                )
            ]
        phrases.extend(blocks)
    return phrases


def _coalesce_singing_phrases(
    phrases: list[AudioRegion], *, minimum_seconds: float
) -> list[AudioRegion]:
    if not phrases:
        return []
    blocks: list[AudioRegion] = []
    start = phrases[0].start
    confidence = phrases[0].confidence
    for phrase in phrases:
        confidence = max(float(confidence or 0.0), float(phrase.confidence or 0.0))
        if phrase.end - start >= minimum_seconds:
            blocks.append(
                AudioRegion(start, phrase.end, "singing", confidence=confidence)
            )
            start = phrase.end
            confidence = phrase.confidence
    tail_end = phrases[-1].end
    if tail_end > start:
        if blocks:
            previous = blocks[-1]
            blocks[-1] = AudioRegion(
                previous.start,
                tail_end,
                "singing",
                confidence=max(
                    float(previous.confidence or 0.0),
                    float(confidence or 0.0),
                ),
            )
        else:
            blocks.append(
                AudioRegion(start, tail_end, "singing", confidence=confidence)
            )
    return blocks


def _vocal_interval(
    offset: int, interval: list[int], sample_rate: int, parent: AudioRegion
) -> AudioRegion:
    return AudioRegion(
        round((offset + interval[0]) / sample_rate, 3),
        round((offset + interval[1]) / sample_rate, 3),
        "singing",
        confidence=parent.confidence,
    )


def _mark_overlaps(regions: list[AudioRegion]) -> list[AudioRegion]:
    marked: list[AudioRegion] = []
    for index, region in enumerate(regions):
        overlap = any(
            other.speaker != region.speaker
            and min(region.end, other.end) > max(region.start, other.start)
            for other in regions[index + 1 :]
            if other.start < region.end
        ) or any(
            other.speaker != region.speaker
            and min(region.end, other.end) > max(region.start, other.start)
            for other in regions[:index]
            if other.end > region.start
        )
        marked.append(AudioRegion(**{**asdict(region), "overlap": overlap}))
    return marked


def _overlap_spans(
    regions: list[AudioRegion], duration: float
) -> list[tuple[float, float]]:
    regions = sorted(regions, key=lambda item: (item.start, item.end))
    intersections: list[AudioRegion] = []
    for index, left in enumerate(regions):
        for right in regions[index + 1 :]:
            if right.start >= left.end:
                break
            if left.speaker == right.speaker:
                continue
            start = max(left.start, right.start)
            end = min(left.end, right.end)
            if end > start:
                intersections.append(
                    AudioRegion(
                        max(0.0, start - 0.75), min(duration, end + 0.75), "speech"
                    )
                )
    merged = _merge_regions(intersections, 0.25)
    spans: list[tuple[float, float]] = []
    for region in merged:
        cursor = region.start
        while cursor < region.end:
            end = min(region.end, cursor + 30.0)
            spans.append((cursor, end))
            if end >= region.end:
                break
            cursor = end
    return spans


def _maximum_concurrent_speakers(
    regions: list[AudioRegion], start: float, end: float
) -> int:
    events: list[tuple[float, int, str]] = []
    for region in regions:
        if not region.speaker:
            continue
        clipped_start = max(start, region.start)
        clipped_end = min(end, region.end)
        if clipped_end <= clipped_start:
            continue
        events.append((clipped_start, 1, region.speaker))
        events.append((clipped_end, -1, region.speaker))
    counts: dict[str, int] = {}
    maximum = 0
    # End events precede start events at the same timestamp.
    for _time, delta, speaker in sorted(events, key=lambda item: (item[0], item[1])):
        if delta < 0:
            count = counts.get(speaker, 0) - 1
            if count > 0:
                counts[speaker] = count
            else:
                counts.pop(speaker, None)
        else:
            counts[speaker] = counts.get(speaker, 0) + 1
            maximum = max(maximum, len(counts))
    return maximum


def _subtract_regions(
    region: AudioRegion, spans: list[tuple[float, float]]
) -> list[AudioRegion]:
    fragments = [(region.start, region.end)]
    for cut_start, cut_end in spans:
        updated: list[tuple[float, float]] = []
        for start, end in fragments:
            if cut_end <= start or cut_start >= end:
                updated.append((start, end))
                continue
            if start < cut_start:
                updated.append((start, cut_start))
            if cut_end < end:
                updated.append((cut_end, end))
        fragments = updated
    return [
        AudioRegion(
            start,
            end,
            region.kind,
            region.speaker,
            region.confidence,
            False,
            region.source_path,
            region.source_offset,
        )
        for start, end in fragments
        if end - start >= 0.08
    ]


def _merge_regions(regions: list[AudioRegion], maximum_gap: float) -> list[AudioRegion]:
    if not regions:
        return []
    ordered = sorted(regions, key=lambda item: (item.start, item.end))
    merged = [ordered[0]]
    for region in ordered[1:]:
        previous = merged[-1]
        if region.start <= previous.end + maximum_gap:
            scores = [
                score
                for score in (previous.confidence, region.confidence)
                if score is not None
            ]
            merged[-1] = AudioRegion(
                previous.start,
                max(previous.end, region.end),
                previous.kind,
                confidence=max(scores) if scores else None,
            )
        else:
            merged.append(region)
    return merged


def _load_waveform(path: Path) -> tuple[Any, int]:
    import soundfile as sf
    import torch

    data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    return torch.from_numpy(data.T.copy()), int(sample_rate)


def _extract_audio(video: Path, destination: Path) -> None:
    ffmpeg = require_command("ffmpeg")
    run(
        [
            ffmpeg,
            "-y",
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


def _release_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _signature(
    video: Path, config: AudioAnalysisConfig, metadata: dict[str, object]
) -> str:
    stat = video.stat()
    profile_dir = Path(config.speaker_profiles_dir).expanduser()
    profile_state = (
        [
            (path.name, path.stat().st_size, path.stat().st_mtime_ns)
            for path in sorted(profile_dir.glob("*.json"))
        ]
        if profile_dir.is_dir()
        else []
    )
    payload = {
        "version": _CACHE_VERSION,
        "video_size": stat.st_size,
        "video_mtime_ns": stat.st_mtime_ns,
        "config": asdict(config),
        "channel_identity": {
            key: metadata.get(key)
            for key in ("channel", "channel_id", "uploader", "uploader_id")
        },
        "speaker_profiles": profile_state,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_cache(path: Path, signature: str) -> AudioAnalysis | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("version") != _CACHE_VERSION
            or value.get("signature") != signature
        ):
            return None
        return AudioAnalysis(
            speech=[AudioRegion(**item) for item in value["speech"]],
            singing=[AudioRegion(**item) for item in value["singing"]],
            ambiguous=[AudioRegion(**item) for item in value.get("ambiguous", [])],
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("ignoring unreadable audio analysis cache %s: %s", path, exc)
        return None
