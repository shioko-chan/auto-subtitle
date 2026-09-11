"""Incremental repetition detection for ALT decoder generation steps."""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field

import torch
from transformers import StoppingCriteria

from .repetition import RepetitionLoopError, StreamingRepetitionDetector


@dataclass
class _PrefixState:
    detector: StreamingRepetitionDetector = field(default_factory=StreamingRepetitionDetector)
    pending: tuple[int, ...] = ()


class ALTRepetitionStoppingCriteria(StoppingCriteria):
    """Abort a looping ALT attempt; its caller retries shorter audio internally.

    Beam candidates can change order and fork on every step. Match each candidate
    to its previous token prefix, copying only the bounded detector state. Never
    associate persistent state with a beam's row index.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._states: dict[tuple[int, ...], _PrefixState] = {}
        self._position = 0

    def __call__(self, input_ids, scores, **kwargs):
        length = input_ids.shape[1]
        # Whisper can start another segment or decoding attempt with a new prompt.
        if length <= self._position:
            self._states = {}
            self._position = 0
        position = self._position or length - 1
        states = {}
        for values in input_ids.tolist():
            tokens = tuple(values)
            if tokens in states:
                continue
            parent = self._states[tokens[:position]] if self._states else _PrefixState()
            state = _PrefixState(copy(parent.detector), parent.pending + tokens[position:])
            text = self.tokenizer.decode(
                state.pending, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )
            # A byte-level token can end halfway through a UTF-8 character.
            if not text.endswith('\ufffd'):
                state.pending = ()
                match = state.detector.feed(text)
                if match is not None:
                    raise RepetitionLoopError(match)
            states[tokens] = state
        self._states = states
        self._position = length
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
