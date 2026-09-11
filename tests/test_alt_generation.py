import unittest

import torch

from subtitle_pipeline.alt_generation import ALTRepetitionStoppingCriteria
from subtitle_pipeline.repetition import RepetitionLoopError


class ByteTokenizer:
    def __init__(self):
        self.sizes = []

    def decode(self, tokens, **kwargs):
        self.sizes.append(len(tokens))
        return bytes(tokens).decode('utf-8', errors='replace')


class ALTGenerationTests(unittest.TestCase):
    def test_detects_during_generation_and_only_decodes_new_tokens(self):
        tokenizer = ByteTokenizer()
        guard = ALTRepetitionStoppingCriteria(tokenizer)
        with self.assertRaises(RepetitionLoopError) as error:
            for count in range(1, 201):
                guard(torch.tensor([[0] + [97] * count]), None)
        self.assertEqual(error.exception.match.end, 160)
        self.assertEqual(max(tokenizer.sizes), 1)

    def test_beams_follow_prefixes_after_reordering(self):
        guard = ALTRepetitionStoppingCriteria(ByteTokenizer())
        # A changes row on every step; it must still be stopped at 160 characters.
        a, b = [0], [0]
        with self.assertRaises(RepetitionLoopError) as error:
            for count in range(1, 161):
                a.append(97)
                b.append(33 + count % 90)
                guard(torch.tensor([a, b] if count % 2 else [b, a]), None)
        self.assertEqual(error.exception.match.pattern, 'a')
        self.assertEqual(error.exception.match.end, 160)

    def test_forked_beams_do_not_modify_parent_or_sibling(self):
        guard = ALTRepetitionStoppingCriteria(ByteTokenizer())
        for count in range(1, 159):
            guard(torch.tensor([[0] + [97] * count]), None)
        prefix = [0] + [97] * 158
        guard(torch.tensor([prefix + [98], prefix + [97]]), None)
        guard(torch.tensor([prefix + [98, 99]]), None)
        self.assertEqual(next(iter(guard._states.values())).detector._length, 160)

    def test_utf8_split_tokens_are_buffered(self):
        guard = ALTRepetitionStoppingCriteria(ByteTokenizer())
        tokens = [0]
        with self.assertRaises(RepetitionLoopError) as error:
            for value in ('あ' * 170).encode():
                tokens.append(value)
                guard(torch.tensor([tokens]), None)
        self.assertEqual(error.exception.match.pattern, 'あ')
        self.assertEqual(error.exception.match.end, 160)

    def test_new_segment_resets_previous_prefix(self):
        guard = ALTRepetitionStoppingCriteria(ByteTokenizer())
        for count in range(1, 159):
            guard(torch.tensor([[0] + [97] * count]), None)
        self.assertFalse(guard(torch.tensor([[0, 97]]), None).item())
        self.assertEqual(next(iter(guard._states.values())).detector._length, 1)
