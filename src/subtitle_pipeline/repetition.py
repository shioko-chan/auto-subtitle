from __future__ import annotations

import re
from dataclasses import dataclass

_MIN_REPETITION_SPAN_CHARACTERS = 160
_REPETITION_RE = re.compile(r"(.{1,200}?)\1{3,}", re.DOTALL)


@dataclass(frozen=True)
class RepetitionMatch:
    pattern: str
    repeats: int
    start: int
    end: int


class RepetitionLoopError(RuntimeError):
    def __init__(self, match: RepetitionMatch):
        self.match = match
        super().__init__(
            f"repetition loop pattern={match.pattern[:80]!r} "
            f"repeats={match.repeats} span={match.start}:{match.end}"
        )


def find_repetition_loop(text: str) -> RepetitionMatch | None:
    normalized = "".join(text.split())
    candidates = [
        match
        for match in _REPETITION_RE.finditer(normalized)
        if match.end() - match.start() >= _MIN_REPETITION_SPAN_CHARACTERS
    ]
    if not candidates:
        return None
    match = max(candidates, key=lambda item: item.end() - item.start())
    pattern = match.group(1)
    return RepetitionMatch(
        pattern=pattern,
        repeats=(match.end() - match.start()) // len(pattern),
        start=match.start(),
        end=match.end(),
    )
