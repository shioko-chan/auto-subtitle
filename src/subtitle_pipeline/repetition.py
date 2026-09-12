from __future__ import annotations

from dataclasses import dataclass

from .llm_errors import LLMResponseError

_MIN_REPETITION_SPAN_CHARACTERS = 160
_MAX_STREAM_PERIOD = 200
_MIN_STREAM_REPEATS = 4
_STREAM_TAIL_CHARACTERS = max(
    _MAX_STREAM_PERIOD * _MIN_STREAM_REPEATS,
    _MIN_REPETITION_SPAN_CHARACTERS + _MAX_STREAM_PERIOD - 1,
)


@dataclass(frozen=True)
class RepetitionMatch:
    pattern: str
    repeats: int
    start: int
    end: int


class RepetitionLoopError(LLMResponseError):
    def __init__(self, match: RepetitionMatch):
        self.match = match
        super().__init__(
            f"repetition loop pattern={match.pattern[:80]!r} "
            f"repeats={match.repeats} span={match.start}:{match.end}"
        )


def find_repetition_loop(text: str) -> RepetitionMatch | None:
    """Check complete text using the same detector as streaming generation."""
    return StreamingRepetitionDetector().feed(text)


class StreamingRepetitionDetector:
    """Bounded tail checks, independent of transport chunk boundaries."""

    def __init__(self) -> None:
        self._tail = ""
        self._length = 0

    def feed(self, text: str) -> RepetitionMatch | None:
        for character in text:
            if character.isspace():
                continue
            self._length += 1
            self._tail = (self._tail + character)[-_STREAM_TAIL_CHARACTERS:]
            n = len(self._tail)
            if n < _MIN_REPETITION_SPAN_CHARACTERS:
                continue
            for period in range(1, min(_MAX_STREAM_PERIOD, n // _MIN_STREAM_REPEATS) + 1):
                repeats = max(_MIN_STREAM_REPEATS, (_MIN_REPETITION_SPAN_CHARACTERS + period - 1) // period)
                span = repeats * period
                if _equal_tail_blocks(self._tail, period, repeats):
                    return RepetitionMatch(
                        self._tail[-period:], repeats, self._length - span, self._length
                    )
        return None


def _equal_tail_blocks(text: str, period: int, repeats: int) -> bool:
    """Compare corresponding characters from the tail; stop at the first mismatch."""
    if period * repeats > len(text):
        return False
    for offset in range(1, period + 1):
        character = text[-offset]
        for block in range(1, repeats):
            if text[-offset - block * period] != character:
                return False
    return True
