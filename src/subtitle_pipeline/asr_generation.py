"""Qwen3-ASR generation with independent repetition stops for each batch row."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.qwen3_asr import ASRTranscription
from qwen_asr.inference.utils import (
    normalize_audios, normalize_language_name, validate_language, split_audio_into_chunks,
    parse_asr_output, merge_languages, SAMPLE_RATE, MAX_ASR_INPUT_SECONDS,
    MAX_FORCE_ALIGN_INPUT_SECONDS,
)
from transformers import StoppingCriteria, StoppingCriteriaList

from .repetition import RepetitionMatch, StreamingRepetitionDetector


class RepetitionStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length: int, batch_size: int):
        self.tokenizer = tokenizer
        self.position = prompt_length
        self.detectors = [StreamingRepetitionDetector() for _ in range(batch_size)]
        self.pending = [[] for _ in range(batch_size)]
        self.matches = [None] * batch_size

    def __call__(self, input_ids, scores, **kwargs):
        new_tokens = input_ids[:, self.position:].tolist()
        self.position = input_ids.shape[1]
        for index, tokens in enumerate(new_tokens):
            if self.matches[index] is not None:
                continue
            self.pending[index].extend(tokens)
            text = self.tokenizer.decode(
                self.pending[index], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            # Qwen's byte-level tokens can split a UTF-8 character across steps.
            if text.endswith('\ufffd'):
                continue
            self.pending[index].clear()
            self.matches[index] = self.detectors[index].feed(text)
            if self.matches[index] is not None:
                match = self.matches[index]
                logging.warning(
                    'Qwen3-ASR generation stopped repetition: batch_row=%d pattern=%r repeats=%d',
                    index, match.pattern[:80], match.repeats,
                )
        return torch.tensor(
            [match is not None for match in self.matches],
            dtype=torch.bool, device=input_ids.device,
        )


@dataclass
class GuardedASRTranscription(ASRTranscription):
    repetition: RepetitionMatch | None = None


class RepetitionGuardedASRModel(Qwen3ASRModel):
    def transcribe(self, audio, context="", language=None, return_time_stamps=False):
        if return_time_stamps and self.forced_aligner is None:
            raise ValueError("timestamps require a forced aligner")
        wavs = normalize_audios(audio)
        count = len(wavs)

        def expand(value):
            values = value if isinstance(value, list) else [value]
            if len(values) == 1:
                values = values * count
            if len(values) != count:
                raise ValueError("ASR audio/context/language batch size mismatch")
            return values

        contexts = expand(context)
        languages = [normalize_language_name(value) if value and value.strip() else None
                     for value in expand(language)]
        for value in languages:
            if value is not None:
                validate_language(value)
        limit = MAX_FORCE_ALIGN_INPUT_SECONDS if return_time_stamps else MAX_ASR_INPUT_SECONDS
        chunks = [(index, chunk, offset) for index, wav in enumerate(wavs)
                  for chunk, offset in split_audio_into_chunks(wav, SAMPLE_RATE, limit)]
        raw = self._infer_asr_transformers(
            [contexts[index] for index, _, _ in chunks],
            [chunk for _, chunk, _ in chunks],
            [languages[index] for index, _, _ in chunks],
        )
        parsed = [parse_asr_output(text, user_language=languages[index])
                  for text, (index, _, _) in zip(raw, chunks, strict=True)]
        matches = [None] * count
        for (index, _, _), match in zip(chunks, self._chunk_repetitions, strict=True):
            if match is not None and matches[index] is None:
                matches[index] = match
        alignments = [[] for _ in wavs]
        if return_time_stamps:
            selected = [position for position, ((index, _, _), (_, text))
                        in enumerate(zip(chunks, parsed, strict=True))
                        if matches[index] is None and text.strip()]
            batch_size = self.max_inference_batch_size
            if batch_size is None or batch_size < 0:
                batch_size = max(1, len(selected))
            for start in range(0, len(selected), batch_size):
                positions = selected[start:start + batch_size]
                aligned = self.forced_aligner.align(
                    audio=[(chunks[p][1], SAMPLE_RATE) for p in positions],
                    text=[parsed[p][1] for p in positions],
                    language=[parsed[p][0] for p in positions],
                )
                for position, value in zip(positions, aligned, strict=True):
                    index, _, offset = chunks[position]
                    alignments[index].append(self._offset_align_result(value, offset))
        texts = [[] for _ in wavs]
        detected_languages = [[] for _ in wavs]
        for (index, _, _), (lang, text) in zip(chunks, parsed, strict=True):
            texts[index].append(text)
            detected_languages[index].append(lang)
        return [GuardedASRTranscription(
            language=merge_languages(detected_languages[index]), text="".join(texts[index]),
            time_stamps=self._merge_align_results(alignments[index]) if alignments[index] else None,
            repetition=matches[index],
        ) for index in range(count)]

    def _infer_asr_transformers(self, contexts, wavs, languages):
        texts = [self._build_text_prompt(context=c, force_language=language)
                 for c, language in zip(contexts, languages)]
        batch_size = self.max_inference_batch_size
        if batch_size is None or batch_size < 0:
            batch_size = len(texts)
        outputs = []
        self._chunk_repetitions = []
        for start in range(0, len(texts), batch_size):
            inputs = self.processor(
                text=texts[start:start + batch_size], audio=wavs[start:start + batch_size],
                return_tensors='pt', padding=True,
            ).to(self.model.device).to(self.model.dtype)
            prompt_length = inputs['input_ids'].shape[1]
            guard = RepetitionStoppingCriteria(
                self.processor.tokenizer, prompt_length, inputs['input_ids'].shape[0],
            )
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                stopping_criteria=StoppingCriteriaList([guard]),
            )
            self._chunk_repetitions.extend(guard.matches)
            outputs.extend(self.processor.batch_decode(
                generated.sequences[:, prompt_length:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ))
        return outputs
