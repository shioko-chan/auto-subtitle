from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

from .config import SegmentationConfig


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


_TIMING_RE = re.compile(
    r"(?P<start>\d{1,3}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
    r"(?P<end>\d{1,3}:\d{2}:\d{2}[,.]\d{3})"
)
_TAG_RE = re.compile(r"<[^>]+>")
_SENTENCE_END_RE = re.compile(r"(?:[.!?。！？…]+|\.{3,})[\"'”’」』）)\]]*$")
_CJK_EDGE_RE = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_CJK_LEFT_EDGE_RE = re.compile(
    r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af][，、；：—]*$"
)


def _seconds(value: str) -> float:
    hours, minutes, rest = value.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(rest)


def _timestamp(value: float) -> str:
    millis = max(0, round(value * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    seconds, millis = divmod(millis, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def read_subtitles(path: Path) -> list[Cue]:
    content = path.read_text(encoding="utf-8-sig")
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", content)
    cues: list[Cue] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_index = next(
            (index for index, line in enumerate(lines) if _TIMING_RE.search(line)), None
        )
        if timing_index is None:
            continue
        match = _TIMING_RE.search(lines[timing_index])
        assert match is not None
        text = "\n".join(lines[timing_index + 1 :])
        text = _TAG_RE.sub("", text).strip()
        if text:
            cues.append(Cue(_seconds(match["start"]), _seconds(match["end"]), text))
    if not cues:
        raise ValueError(f"no subtitle cues found in {path}")
    return cues


def write_srt(cues: list[Cue], path: Path) -> None:
    parts = []
    for index, cue in enumerate(cues, 1):
        parts.append(
            f"{index}\n{_timestamp(cue.start)} --> {_timestamp(cue.end)}\n{cue.text.strip()}"
        )
    path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def merge_semantic_cues(cues: list[Cue], config: SegmentationConfig) -> list[Cue]:
    """Merge display-oriented cues into sentence-oriented translation units."""
    if not config.enabled or len(cues) < 2:
        return cues.copy()

    merged: list[Cue] = []
    current = cues[0]
    for following in cues[1:]:
        gap = following.start - current.end
        combined_text = _join_fragments(current.text, following.text)
        forced_boundary = (
            _ends_sentence(current.text)
            or gap > config.max_gap_seconds
            or following.end - current.start > config.max_duration_seconds
            or len(combined_text) > config.max_source_chars
        )
        if forced_boundary:
            merged.append(current)
            current = following
        else:
            current = Cue(current.start, following.end, combined_text)
    merged.append(current)
    return merged


def _ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END_RE.search(text.strip()))


def _join_fragments(left: str, right: str) -> str:
    left = " ".join(left.split())
    right = " ".join(right.split())
    if not left:
        return right
    if not right:
        return left
    if left.endswith("-") and not left.endswith("--"):
        return left[:-1] + right
    if _CJK_LEFT_EDGE_RE.search(left) and _CJK_EDGE_RE.search(right[0]):
        return left + right
    return left + " " + right


def translation_payload(cues: list[Cue]) -> str:
    return json.dumps(
        [{"id": index, "text": cue.text} for index, cue in enumerate(cues)],
        ensure_ascii=False,
    )


def apply_translations(cues: list[Cue], values: object) -> list[Cue]:
    if not isinstance(values, list):
        raise ValueError("LLM response field 'translations' must be a list")
    mapped: dict[int, str] = {}
    for item in values:
        if not isinstance(item, dict) or not isinstance(item.get("id"), int):
            raise ValueError("every translation must contain an integer id")
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("every translation must contain non-empty text")
        mapped[item["id"]] = text.strip()
    expected = set(range(len(cues)))
    if set(mapped) != expected:
        raise ValueError("LLM response ids do not match the requested subtitle cues")
    return [replace(cue, text=mapped[index]) for index, cue in enumerate(cues)]
