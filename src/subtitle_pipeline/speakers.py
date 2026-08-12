from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path
from typing import Any

from .audio_analysis import AudioRegion
from .config import AudioAnalysisConfig


@dataclass(frozen=True)
class CharacterStyle:
    id: str
    source_name: str
    primary_color: str
    outline_color: str


def load_character_styles(path_value: str | None = None) -> dict[str, CharacterStyle]:
    if path_value:
        value = json.loads(Path(path_value).expanduser().read_text(encoding="utf-8"))
    else:
        value = json.loads(
            resources.files("subtitle_pipeline")
            .joinpath("character_styles.json")
            .read_text(encoding="utf-8")
        )
    styles: dict[str, CharacterStyle] = {}
    for character in value.get("characters", []):
        style = character["subtitle_style"]
        item = CharacterStyle(
            str(character["id"]),
            str(character["source_name"]),
            _hex_color(str(style["primary_color"])),
            _hex_color(str(style.get("outline_color", "#000000"))),
        )
        styles[item.id] = item
    return styles


def metadata_character(metadata: dict[str, object], path_value: str | None = None) -> str | None:
    if path_value:
        value = json.loads(Path(path_value).expanduser().read_text(encoding="utf-8"))
    else:
        value = json.loads(
            resources.files("subtitle_pipeline")
            .joinpath("character_styles.json")
            .read_text(encoding="utf-8")
        )
    evidence = " ".join(
        str(metadata.get(key) or "")
        for key in ("channel", "channel_id", "uploader", "uploader_id")
    ).casefold()
    matches = [
        str(character["id"])
        for character in value.get("characters", [])
        if any(
            str(alias).casefold() in evidence
            for alias in character.get("identity_aliases", [])
        )
    ]
    return matches[0] if len(matches) == 1 else None


def identify_speakers(
    waveform: Any,
    sample_rate: int,
    regions: list[AudioRegion],
    config: AudioAnalysisConfig,
    *,
    known_character: str | None = None,
    excluded_regions: list[AudioRegion] | None = None,
) -> list[AudioRegion]:
    """Enroll clean solo audio and map anonymous clusters to member prototypes."""
    import numpy as np
    import torch
    from pyannote.audio import Inference, Model

    candidates = [
        region
        for region in regions
        if _can_embed_region(region)
        and not any(
            min(region.end, excluded.end) > max(region.start, excluded.start)
            for excluded in (excluded_regions or [])
        )
    ]
    if not candidates:
        return regions
    model = Model.from_pretrained(config.speaker_embedding_model)
    if model is None:
        raise RuntimeError(f"could not load speaker model {config.speaker_embedding_model}")
    inference = Inference(model, window="whole")
    inference.to(torch.device(config.device))
    try:
        by_label: dict[str, list[Any]] = {}
        separated_waveforms: dict[str, tuple[Any, int]] = {}
        for region in candidates:
            source_waveform, source_rate = waveform, sample_rate
            if region.source_path:
                if region.source_path not in separated_waveforms:
                    import soundfile as sf

                    data, loaded_rate = sf.read(
                        region.source_path, always_2d=True, dtype="float32"
                    )
                    separated_waveforms[region.source_path] = (
                        torch.from_numpy(data.T.copy()),
                        int(loaded_rate),
                    )
                source_waveform, source_rate = separated_waveforms[region.source_path]
            start = round((region.start - region.source_offset) * source_rate)
            end = round((region.end - region.source_offset) * source_rate)
            audio = source_waveform[:, start:end]
            embedding = np.asarray(
                inference({"waveform": audio, "sample_rate": source_rate})
            ).reshape(-1)
            if np.isfinite(embedding).all():
                by_label.setdefault(region.speaker or "unknown", []).append(embedding)
        centroids = {
            label: _normalize(np.mean(values, axis=0))
            for label, values in by_label.items()
            if values
        }
        profile_dir = Path(config.speaker_profiles_dir).expanduser()
        profiles = _load_profiles(profile_dir)
        enrollment_labels = {
            region.speaker or "unknown"
            for region in candidates
            if not region.overlap
        }
        enrollment = {
            label: values
            for label, values in by_label.items()
            if label in enrollment_labels
        }
        if known_character and enrollment:
            dominant = max(enrollment, key=lambda label: len(enrollment[label]))
            _update_profile(profile_dir, known_character, enrollment[dominant])
            profiles = _load_profiles(profile_dir)
        mapping: dict[str, tuple[str, float]] = {}
        for label, centroid in centroids.items():
            distances = sorted(
                (float(1.0 - np.dot(centroid, profile)), character)
                for character, profile in profiles.items()
            )
            if distances and distances[0][0] <= config.speaker_match_threshold:
                mapping[label] = (distances[0][1], distances[0][0])
        for label, (character, distance) in mapping.items():
            logging.info(
                "speaker identity %s -> %s cosine_distance=%.3f",
                label,
                character,
                distance,
            )
        return [
            replace(
                region,
                speaker=mapping.get(region.speaker or "", (region.speaker, 0.0))[0],
                confidence=(
                    round(1.0 - mapping[region.speaker or ""][1], 4)
                    if region.speaker in mapping
                    else region.confidence
                ),
            )
            for region in regions
        ]
    finally:
        del inference, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _can_embed_region(region: AudioRegion) -> bool:
    duration = region.end - region.start
    maximum = 30.0 if region.overlap and region.source_path else 15.0
    return 2.0 <= duration <= maximum and (not region.overlap or bool(region.source_path))


def _load_profiles(directory: Path) -> dict[str, Any]:
    import numpy as np

    profiles: dict[str, Any] = {}
    if not directory.is_dir():
        return profiles
    for path in directory.glob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            embeddings = np.asarray(value["embeddings"], dtype=np.float32)
            if embeddings.ndim == 2 and len(embeddings):
                profiles[path.stem] = _normalize(np.mean(embeddings, axis=0))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            logging.warning("ignoring malformed speaker profile %s", path)
    return profiles


def _update_profile(directory: Path, character: str, embeddings: list[Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{character}.json"
    existing: list[list[float]] = []
    if path.is_file():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            existing = value.get("embeddings", [])
        except (OSError, json.JSONDecodeError):
            pass
    additions = [embedding.astype(float).tolist() for embedding in embeddings]
    payload = {"version": 1, "character_id": character, "embeddings": (existing + additions)[-400:]}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(path)
    logging.info("speaker profile %s now contains %d samples", character, len(payload["embeddings"]))


def _normalize(vector: Any) -> Any:
    import numpy as np

    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _hex_color(value: str) -> str:
    normalized = value.strip().upper()
    if len(normalized) != 7 or not normalized.startswith("#"):
        raise ValueError(f"invalid character subtitle color: {value}")
    int(normalized[1:], 16)
    return normalized
