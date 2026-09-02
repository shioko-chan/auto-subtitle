from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path

_TEXT_RE = re.compile(r"[A-Za-z0-9\u3040-\u30ff\u3400-\u9fff]")
_URL_RE = re.compile(r"https?://\S+")
_LOW_INFORMATION_RE = re.compile(
    r"(?:[wWｗＷ]+|[草笑]+|[8８]+|(?:拍手)+|(?:パチ)+|(?:ぱち)+)",
    re.IGNORECASE,
)
_REPEAT_GROUP_SECONDS = 10.0


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
    filtered_count: int


class CurrentVideoChatIndex:
    """Time-local, untrusted chat evidence for the video being processed."""

    def __init__(
        self,
        messages: list[YouTubeChatMessage],
        *,
        lookback_seconds: float = 30.0,
        lookahead_seconds: float = 15.0,
        audit_path: Path | None = None,
    ) -> None:
        self._messages = sorted(
            _deduplicate_messages(messages), key=lambda value: value.offset_seconds
        )
        self._lookback_seconds = lookback_seconds
        self._lookahead_seconds = lookahead_seconds
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
                "filtered_count": evidence.filtered_count,
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


def read_youtube_live_chat(
    path: Path, *, include_low_information: bool = False
) -> list[YouTubeChatMessage]:
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
                if not include_low_information and not useful_chat_text(text):
                    continue
                if not text and not amount and not membership:
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
) -> ChatEvidence:
    if not candidates:
        return ChatEvidence("", 0, 0, 0)
    selected = [message for message in candidates if not _low_information(message.text)]
    lines = _format_selected_messages(selected)
    return ChatEvidence(
        "\n".join(lines),
        len(candidates),
        len(lines),
        len(candidates) - len(selected),
    )


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


def _format_selected_messages(messages: list[YouTubeChatMessage]) -> list[str]:
    grouped: list[list[YouTubeChatMessage]] = []
    active: dict[tuple[str, str | None], list[YouTubeChatMessage]] = {}
    for message in messages:
        key = (_clean_text(message.text).casefold(), message.amount)
        values = active.get(key)
        if (
            values is None
            or message.offset_seconds - values[0].offset_seconds > _REPEAT_GROUP_SECONDS
        ):
            values = []
            active[key] = values
            grouped.append(values)
        values.append(message)
    lines: list[tuple[float, str]] = []
    for values in grouped:
        first = values[0]
        text = _clean_text(first.text)
        amount = first.amount
        authors = {value.author or value.message_id or "anonymous" for value in values}
        count = f" ×{len(authors)}" if len(authors) > 1 else ""
        if amount:
            author_text = ", ".join(sorted(authors))
            kind = f"SC {amount} author={author_text}"
        else:
            kind = "chat"
        lines.append((first.offset_seconds, f"[{kind}{count}] {text}"))
    return [line for _offset, line in sorted(lines)]


def _low_information(text: str) -> bool:
    value = re.sub(r":[A-Za-z0-9_+-]+:", "", _URL_RE.sub("", text))
    value = re.sub(r"[\s\W_]+", "", value, flags=re.UNICODE)
    return not value or _LOW_INFORMATION_RE.fullmatch(value) is not None


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
