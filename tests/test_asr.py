import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from subtitle_pipeline.asr import (
    _restore_punctuation,
    _valid_cached_record,
    transcribe_with_qwen,
)
from subtitle_pipeline.config import ASRConfig


class QwenASRTests(unittest.TestCase):
    def test_restores_punctuation_between_japanese_aligner_tokens(self):
        self.assertEqual(
            _restore_punctuation(
                "こんにちは、今日はライブです。",
                ["こんにちは", "今日", "は", "ライブ", "です"],
            ),
            ["こんにちは、", "今日", "は", "ライブ", "です。"],
        )

    def test_caches_each_completed_audio_chunk_and_reuses_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "source.mp4"
            destination = root / "source.qwen3-asr.srt"
            video.write_bytes(b"video")
            config = ASRConfig(chunk_seconds=10, chunk_context_seconds=1)
            item = SimpleNamespace(text="一", start_time=1, end_time=2)
            result = SimpleNamespace(
                language="Japanese",
                text="一。",
                time_stamps=SimpleNamespace(items=[item]),
            )
            model = SimpleNamespace(transcribe=lambda **_kwargs: [result])

            with patch(
                "subtitle_pipeline.asr._media_duration", return_value=15
            ), patch(
                "subtitle_pipeline.asr._load_qwen_model", return_value=model
            ) as load_model, patch(
                "subtitle_pipeline.asr._extract_audio_chunk"
            ):
                transcribe_with_qwen(video, destination, config)
                transcribe_with_qwen(video, destination, config)

            load_model.assert_called_once()
            self.assertTrue((root / "asr-cache.json").is_file())
            self.assertEqual(destination.read_text(encoding="utf-8").count("一。"), 2)

    def test_rejects_zero_duration_cached_cue(self):
        self.assertFalse(
            _valid_cached_record(
                {"cues": [{"start": 1.0, "end": 1.0, "text": "字幕"}]}
            )
        )


if __name__ == "__main__":
    unittest.main()
