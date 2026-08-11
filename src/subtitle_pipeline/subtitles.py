from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path

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
_BRACKETED_MARKER_RE = re.compile(r"[\[［【](?P<label>[^\]］】\r\n]{1,30})[\]］】]")
_NON_SPEECH_MARKERS = {
    "applause",
    "background music",
    "bgm",
    "breathing",
    "cheering",
    "cheers",
    "inaudible",
    "laughter",
    "laughs",
    "music",
    "sigh",
    "sighs",
    "silence",
    "singing",
    "呼吸声",
    "呼吸聲",
    "喝彩",
    "噪音",
    "拍手",
    "掌声",
    "掌聲",
    "无声",
    "无音乐",
    "歌声",
    "歌聲",
    "欢呼",
    "歡呼",
    "笑",
    "笑声",
    "笑聲",
    "背景音乐",
    "背景音樂",
    "静音",
    "音樂",
    "音乐",
    "息",
    "歓声",
    "無音",
    "笑い",
    "音楽",
    "鼻息",
}


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


def clean_non_speech_markers(cues: list[Cue]) -> list[Cue]:
    cleaned: list[Cue] = []
    for cue in cues:
        text = _BRACKETED_MARKER_RE.sub(_remove_non_speech_marker, cue.text)
        lines = [" ".join(line.split()) for line in text.splitlines()]
        text = "\n".join(line for line in lines if line).strip()
        if text:
            cleaned.append(replace(cue, text=text))
    return cleaned


def _remove_non_speech_marker(match: re.Match[str]) -> str:
    label = " ".join(match.group("label").split()).casefold()
    return "" if label in _NON_SPEECH_MARKERS else match.group(0)


def merge_cues_at_boundaries(
    cues: list[Cue], boundary_after_ids: set[int]
) -> list[Cue]:
    """Merge aligned units using only explicitly approved boundary IDs."""
    if len(cues) < 2:
        return cues.copy()
    invalid = sorted(
        boundary for boundary in boundary_after_ids if not 0 <= boundary < len(cues)
    )
    if invalid:
        raise ValueError(f"subtitle boundary IDs are out of range: {invalid}")
    merged: list[Cue] = []
    current = cues[0]
    for index, following in enumerate(cues[1:]):
        if index in boundary_after_ids:
            merged.append(current)
            current = following
        else:
            current = Cue(
                current.start,
                following.end,
                _join_fragments(current.text, following.text),
            )
    merged.append(current)
    return merged


def trusted_sentence_boundaries(cues: list[Cue]) -> set[int]:
    return {index for index, cue in enumerate(cues) if _ends_sentence(cue.text)}


def trim_overlapping_cues(cues: list[Cue]) -> list[Cue]:
    """End each cue when the following cue starts so renderers never stack them."""
    if len(cues) < 2:
        return cues.copy()

    trimmed: list[Cue] = []
    for current, following in zip(cues, cues[1:]):
        if following.start < current.start:
            raise ValueError("subtitle cues must be ordered by start time")
        trimmed.append(replace(current, end=min(current.end, following.start)))
    trimmed.append(cues[-1])
    return trimmed


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


def text_display_width(text: str) -> float:
    """Estimate subtitle width in font-size units."""
    width = 0.0
    for character in text:
        if character.isspace():
            width += 0.35
        elif unicodedata.combining(character):
            continue
        elif unicodedata.east_asian_width(character) in {"W", "F"}:
            width += 1.0
        else:
            width += 0.55
    return width


def translation_payload(cues: list[Cue]) -> str:
    return json.dumps(
        [{"id": index, "text": cue.text} for index, cue in enumerate(cues)],
        ensure_ascii=False,
    )


def apply_translations(cues: list[Cue], values: object) -> list[Cue]:
    if not isinstance(values, list):
        raise ValueError("LLM response field 'translations' must be a list")
    mapped: dict[int, str] = {}
    duplicate_ids: set[int] = set()
    for item in values:
        if not isinstance(item, dict) or not isinstance(item.get("id"), int):
            raise ValueError("every translation must contain an integer id")
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("every translation must contain non-empty text")
        item_id = item["id"]
        if item_id in mapped:
            duplicate_ids.add(item_id)
        mapped[item_id] = text.strip()
    expected = set(range(len(cues)))
    if set(mapped) != expected:
        missing = sorted(expected - set(mapped))
        unexpected = sorted(set(mapped) - expected)
        raise ValueError(
            "LLM response ids do not match the requested subtitle cues: "
            f"missing={missing}, unexpected={unexpected}, duplicates={sorted(duplicate_ids)}"
        )
    if duplicate_ids:
        raise ValueError(
            "LLM response ids do not match the requested subtitle cues: "
            f"missing=[], unexpected=[], duplicates={sorted(duplicate_ids)}"
        )
    return [replace(cue, text=mapped[index]) for index, cue in enumerate(cues)]
