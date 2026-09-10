import unittest

import torch

from subtitle_pipeline.asr_generation import RepetitionStoppingCriteria


class ByteTokenizer:
    def decode(self, tokens, **kwargs):
        return bytes(token for token in tokens if token < 256).decode('utf-8', errors='replace')


class ASRGenerationTests(unittest.TestCase):
    def test_rows_stop_independently_and_remain_stopped(self):
        guard = RepetitionStoppingCriteria(ByteTokenizer(), prompt_length=200, batch_size=2)
        ids = torch.full((2, 200), ord('x'), dtype=torch.long)
        for step in range(200):
            ids = torch.cat((ids, torch.tensor([[ord('a')], [33 + step % 90]])), dim=1)
            stopped = guard(ids, None).tolist()
            self.assertEqual(stopped, [step >= 159, False])
        self.assertEqual(guard.matches[0].repeats, 160)

    def test_utf8_split_tokens_and_fresh_batch_state(self):
        text = '不知道 ' * 54
        tokens = list(text.encode())
        guard = RepetitionStoppingCriteria(ByteTokenizer(), 0, 1)
        first_stop = None
        for end in range(1, len(tokens) + 1):
            if guard(torch.tensor([tokens[:end]]), None).item():
                first_stop = end
                break
        self.assertEqual(first_stop, len(tokens) - 1)  # final space is not needed
        self.assertEqual(guard.matches[0].pattern, '不知道')
        self.assertFalse(RepetitionStoppingCriteria(ByteTokenizer(), 0, 1)(torch.tensor([[97]]), None).item())

    def test_transformers_generate_finishes_healthy_row_after_loop_row(self):
        from transformers import GPT2Config, GPT2LMHeadModel, LogitsProcessor, LogitsProcessorList, StoppingCriteriaList

        class ScriptedTokens(LogitsProcessor):
            def __call__(self, input_ids, scores):
                step = input_ids.shape[1] - 1
                scores.fill_(-float('inf'))
                scores[0, 97] = 0
                scores[1, 256 if step == 170 else 33 + step % 90] = 0
                return scores

        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=258, n_positions=256, n_embd=8, n_layer=1, n_head=1,
            eos_token_id=256, pad_token_id=257, bos_token_id=256,
        )).eval()
        guard = RepetitionStoppingCriteria(ByteTokenizer(), 1, 2)
        with torch.no_grad():
            generated = model.generate(
                torch.tensor([[256], [256]]), attention_mask=torch.ones((2, 1), dtype=torch.long),
                max_new_tokens=200, do_sample=False,
                logits_processor=LogitsProcessorList([ScriptedTokens()]),
                stopping_criteria=StoppingCriteriaList([guard]),
            )
        self.assertTrue((generated[0, 1:161] == 97).all())
        self.assertTrue((generated[0, 161:] == 257).all())
        self.assertEqual(generated[1, -1].item(), 256)
        self.assertIsNone(guard.matches[1])


class ASRStopPropagationTests(unittest.TestCase):
    def test_chunk_stop_maps_to_original_sample_without_text_rescan_or_alignment(self):
        import numpy as np
        from unittest.mock import Mock, patch
        from subtitle_pipeline.asr_generation import RepetitionGuardedASRModel
        from subtitle_pipeline.repetition import RepetitionMatch
        from subtitle_pipeline.asr import _generation_repetition

        model = object.__new__(RepetitionGuardedASRModel)
        model.max_inference_batch_size = 2
        model.forced_aligner = Mock()
        model.forced_aligner.align.return_value = ['aligned']
        model._offset_align_result = lambda value, offset: value
        model._merge_align_results = lambda values: values
        stopped = RepetitionMatch('前', 160, 0, 160)

        def infer(contexts, wavs, languages):
            model._chunk_repetitions = [None, stopped, None]
            return ['language Japanese<asr_text>第一段', 'language Japanese<asr_text>中断文本', 'language Japanese<asr_text>正常']

        model._infer_asr_transformers = infer
        a, b, c = [np.zeros(1600, dtype=np.float32) for _ in range(3)]
        with patch('subtitle_pipeline.asr_generation.split_audio_into_chunks', side_effect=[[(a, 0), (b, 1)], [(c, 0)]]):
            results = model.transcribe([(a, 16000), (c, 16000)], return_time_stamps=True)
        self.assertEqual(results[0].repetition, stopped)
        self.assertIsNone(results[0].time_stamps)
        self.assertIsNone(results[1].repetition)
        self.assertEqual(model.forced_aligner.align.call_args.kwargs['text'], ['正常'])
        with patch('subtitle_pipeline.repetition.find_repetition_loop', side_effect=AssertionError('must not rescan')):
            self.assertEqual(_generation_repetition(results[0]), ('前', 160))
            self.assertIsNone(_generation_repetition(results[1]))
