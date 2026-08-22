from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class LocalJapaneseTranslator:
    """Lazily loaded, process-local Japanese-to-Chinese fallback translator."""

    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._tokenizer = None
        self._model = None

    def translate(self, text: str) -> str:
        source = text.strip()
        if not source:
            return ""
        self._ensure_loaded()
        import torch

        with self._inference_lock, torch.inference_mode():
            self._tokenizer.src_lang = "ja"
            encoded = self._tokenizer(source, return_tensors="pt")
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            generated = self._model.generate(
                **encoded,
                forced_bos_token_id=self._tokenizer.get_lang_id("zh"),
                max_new_tokens=max(32, min(512, len(source) * 4)),
            )
            translated = self._tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )[0].strip()
        if not translated:
            raise RuntimeError("local Japanese translation returned empty text")
        return translated

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            import torch
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

            logger.info(
                "loading local fallback translation model %s on %s",
                self.model_name,
                self.device,
            )
            tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)
            model = M2M100ForConditionalGeneration.from_pretrained(self.model_name)
            model.to(torch.device(self.device))
            model.eval()
            self._tokenizer = tokenizer
            self._model = model
