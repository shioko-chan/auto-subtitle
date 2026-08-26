from __future__ import annotations

import json
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

_TEXT_RE = re.compile(r"[A-Za-z0-9\u3040-\u30ff\u3400-\u9fff]")
_TERM_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9_+.-]{1,}|[\u30a0-\u30ff]{2,}|[\u3400-\u9fff]{1,}"
)
_URL_RE = re.compile(r"https?://\S+")
_CONTEXT_DEPENDENT_RE = re.compile(
    r"(?:これ|それ|あれ|ここ|そこ|あそこ|どういう|何の|なんの|this|that|it)"
)
_IGNORED_REPEATED_TERMS = frozenset(
    {"笑", "草", "www", "かわいい", "可愛い", "ありがとう", "おめでとう"}
)


@dataclass(frozen=True)
class YouTubeChatMessage:
    offset_seconds: float
    author: str
    text: str
    amount: str | None = None
    membership: bool = False
    message_id: str | None = None


@dataclass(frozen=True)
class ChatEvidence:
    text: str
    candidate_count: int
    selected_count: int
    repeated_terms: tuple[tuple[str, int], ...]


class CurrentVideoChatIndex:
    """Time-local, untrusted chat evidence for the video being processed."""

    def __init__(
        self,
        messages: list[YouTubeChatMessage],
        *,
        lookback_seconds: float = 30.0,
        lookahead_seconds: float = 15.0,
        maximum_chars: int = 900,
        audit_path: Path | None = None,
    ) -> None:
        self._messages = sorted(
            _deduplicate_messages(messages), key=lambda value: value.offset_seconds
        )
        self._lookback_seconds = lookback_seconds
        self._lookahead_seconds = lookahead_seconds
        self._maximum_chars = maximum_chars
        self._audit_path = audit_path
        self._audit_lock = threading.Lock()

    @classmethod
    def from_path(
        cls,
        path: Path,
        **kwargs: object,
    ) -> CurrentVideoChatIndex:
        return cls(read_youtube_live_chat(path), **kwargs)

    def __bool__(self) -> bool:
        return bool(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def evidence(
        self,
        start: float,
        end: float,
        query: str,
        *,
        stage: str,
        target_id: object,
    ) -> str:
        regular_start = max(0.0, start - self._lookback_seconds)
        paid_start = max(0.0, start - 900.0)
        search_end = end + self._lookahead_seconds
        candidates = [
            message
            for message in self._messages
            if message.offset_seconds <= search_end
            and message.offset_seconds
            >= (paid_start if message.amount else regular_start)
        ]
        evidence = _select_evidence(
            candidates,
            query,
            start,
            end,
            maximum_chars=self._maximum_chars,
        )
        self._append_audit(
            {
                "event": "current_video_chat_retrieval",
                "stage": stage,
                "target_id": target_id,
                "start": start,
                "end": end,
                "query": query,
                "candidate_count": evidence.candidate_count,
                "selected_count": evidence.selected_count,
                "repeated_terms": list(evidence.repeated_terms),
                "evidence": evidence.text,
            }
        )
        return evidence.text

    def _append_audit(self, value: dict[str, object]) -> None:
        if self._audit_path is None:
            return
        with self._audit_lock:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self._audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_youtube_live_chat(path: Path) -> list[YouTubeChatMessage]:
    messages: list[YouTubeChatMessage] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            replay = value.get("replayChatItemAction")
            if not isinstance(replay, dict):
                continue
            try:
                offset = max(0.0, float(replay.get("videoOffsetTimeMsec", 0)) / 1000)
            except (TypeError, ValueError):
                offset = 0.0
            for action in replay.get("actions", []):
                if not isinstance(action, dict):
                    continue
                add = action.get("addChatItemAction")
                item = add.get("item") if isinstance(add, dict) else None
                if not isinstance(item, dict):
                    continue
                renderer_name, renderer = _chat_renderer(item)
                if renderer is None:
                    continue
                text = _runs_text(renderer.get("message"))
                amount = _simple_text(renderer.get("purchaseAmountText")) or None
                membership = renderer_name == "liveChatMembershipItemRenderer"
                if not text and membership:
                    text = _runs_text(renderer.get("headerSubtext"))
                if not useful_chat_text(text):
                    continue
                messages.append(
                    YouTubeChatMessage(
                        offset,
                        _simple_text(renderer.get("authorName")),
                        text,
                        amount,
                        membership,
                        str(renderer.get("id") or "").strip() or None,
                    )
                )
    return messages


def remove_youtube_chat_files(directory: Path) -> None:
    for path in directory.glob("source.live_chat.json*"):
        if path.is_file():
            path.unlink()


def useful_chat_text(text: str) -> bool:
    without_emoji = re.sub(r":[A-Za-z0-9_+-]+:", "", text)
    return bool(_TEXT_RE.search(without_emoji))


def _select_evidence(
    candidates: list[YouTubeChatMessage],
    query: str,
    start: float,
    end: float,
    *,
    maximum_chars: int,
) -> ChatEvidence:
    if not candidates:
        return ChatEvidence("", 0, 0, ())
    repeated_terms = _repeated_terms(candidates)
    repeated_set = {term for term, _count in repeated_terms}
    normalized_query = _normalize(query)
    query_terms = set(_terms(query))
    center = (start + end) / 2
    ranked: list[tuple[float, float, YouTubeChatMessage]] = []
    for message in candidates:
        text = _clean_text(message.text)
        normalized_text = _normalize(text)
        terms = set(_terms(text))
        overlap = len(query_terms & terms) / max(1, len(query_terms))
        similarity = (
            SequenceMatcher(None, normalized_query, normalized_text).ratio()
            if normalized_query and normalized_text
            else 0.0
        )
        repeated = 0.3 if terms & repeated_set else 0.0
        paid = 0.45 if message.amount else 0.0
        distance = abs(message.offset_seconds - center)
        temporal = 1.0 / (1.0 + distance / 10.0)
        score = 2.0 * overlap + similarity + repeated + paid + 0.35 * temporal
        ranked.append((score, distance, message))
    ranked.sort(key=lambda value: (-value[0], value[1], value[2].offset_seconds))
    selected: list[YouTubeChatMessage] = []
    selected_ids: set[int] = set()
    if _CONTEXT_DEPENDENT_RE.search(query.casefold()):
        for _score, _distance, message in sorted(ranked, key=lambda value: value[1])[
            :2
        ]:
            selected.append(message)
            selected_ids.add(id(message))
    for score, _distance, message in ranked:
        if len(selected) >= 8:
            break
        terms = set(_terms(message.text))
        independently_relevant = bool(terms & repeated_set)
        if id(message) in selected_ids or (score < 0.7 and not independently_relevant):
            continue
        selected.append(message)
        selected_ids.add(id(message))
    selected.sort(key=lambda value: value.offset_seconds)

    lines = (
        [
            "repeated terms: "
            + ", ".join(f"{term} ({count} viewers)" for term, count in repeated_terms)
        ]
        if repeated_terms
        else []
    )
    lines.extend(_format_selected_messages(selected, start))
    text = "\n".join(lines)
    if len(text) > maximum_chars:
        text = text[: maximum_chars - 1].rstrip() + "…"
    return ChatEvidence(text, len(candidates), len(selected), repeated_terms)


def _deduplicate_messages(
    messages: list[YouTubeChatMessage],
) -> list[YouTubeChatMessage]:
    output: list[YouTubeChatMessage] = []
    seen: set[tuple[object, ...]] = set()
    for message in messages:
        key = (
            ("id", message.message_id)
            if message.message_id
            else (
                "fallback",
                round(message.offset_seconds, 1),
                message.author,
                _clean_text(message.text),
                message.amount,
                message.membership,
            )
        )
        if key not in seen:
            seen.add(key)
            output.append(message)
    return output


def _format_selected_messages(
    messages: list[YouTubeChatMessage], start: float
) -> list[str]:
    grouped: dict[tuple[str, str | None], list[YouTubeChatMessage]] = defaultdict(list)
    for message in messages:
        grouped[(_clean_text(message.text), message.amount)].append(message)
    lines: list[tuple[float, str]] = []
    for (text, amount), values in grouped.items():
        first = min(values, key=lambda value: value.offset_seconds)
        relative = first.offset_seconds - start
        timing = f"{relative:+.1f}s"
        kind = f"SC {amount}" if amount else "chat"
        authors = {value.author or value.message_id or "anonymous" for value in values}
        count = f" ×{len(authors)} viewers" if len(authors) > 1 else ""
        lines.append((first.offset_seconds, f"[{timing} {kind}{count}] {text}"))
    return [line for _offset, line in sorted(lines)]


def _repeated_terms(
    messages: list[YouTubeChatMessage],
) -> tuple[tuple[str, int], ...]:
    authors: dict[str, set[str]] = defaultdict(set)
    occurrences: Counter[str] = Counter()
    for index, message in enumerate(messages):
        author = message.author or f"anonymous:{index}"
        for term in set(_terms(message.text)):
            authors[term].add(author)
            occurrences[term] += 1
    values = [
        (term, len(values))
        for term, values in authors.items()
        if len(values) >= 3
        and occurrences[term] >= 3
        and term not in _IGNORED_REPEATED_TERMS
    ]
    values.sort(key=lambda value: (-value[1], -len(value[0]), value[0]))
    return tuple(values[:6])


def _terms(text: str) -> list[str]:
    return [value.casefold() for value in _TERM_RE.findall(text)]


def _normalize(text: str) -> str:
    return "".join(_terms(text))


def _clean_text(text: str) -> str:
    value = _URL_RE.sub("[link]", " ".join(text.split()))
    return value[:180]


def _chat_renderer(item: dict[str, object]) -> tuple[str, dict[str, object] | None]:
    for name in (
        "liveChatTextMessageRenderer",
        "liveChatPaidMessageRenderer",
        "liveChatPaidStickerRenderer",
        "liveChatMembershipItemRenderer",
    ):
        renderer = item.get(name)
        if isinstance(renderer, dict):
            return name, renderer
    return "", None


def _simple_text(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    simple = value.get("simpleText")
    return str(simple).strip() if isinstance(simple, str) else ""


def _runs_text(value: object) -> str:
    if not isinstance(value, dict) or not isinstance(value.get("runs"), list):
        return _simple_text(value)
    values: list[str] = []
    for run in value["runs"]:
        if not isinstance(run, dict):
            continue
        text = run.get("text")
        if isinstance(text, str):
            values.append(text)
            continue
        emoji = run.get("emoji")
        shortcuts = emoji.get("shortcuts") if isinstance(emoji, dict) else None
        if isinstance(shortcuts, list) and shortcuts:
            values.append(str(shortcuts[0]))
    return "".join(values).strip()
