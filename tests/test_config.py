import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from subtitle_pipeline.config import ConfigError, llm_api_key, load_config


class ConfigTests(unittest.TestCase):
    def test_rejects_unknown_llm_thinking_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text('[llm]\nthinking = "sometimes"\n', encoding="utf-8")

            with self.assertRaisesRegex(ConfigError, "llm.thinking"):
                load_config(path)

    def test_loads_defaults_and_overrides(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                'work_dir = "jobs"\n[llm]\nmodel = "test-model"\n'
                "[upload]\nenabled = true\n",
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.work_dir, Path("jobs"))
            self.assertEqual(config.llm.model, "test-model")
            self.assertTrue(config.upload.enabled)
            self.assertEqual(config.upload.description_max_chars, 1800)
            self.assertEqual(config.asr.model, "Qwen/Qwen3-ASR-1.7B")
            self.assertEqual(config.asr.aligner_model, "Qwen/Qwen3-ForcedAligner-0.6B")
            self.assertEqual(config.asr.dtype, "float16")
            self.assertEqual(config.asr.language, "Japanese")
            self.assertTrue(config.audio_analysis.enabled)
            self.assertFalse(config.audio_analysis.debug_audio_artifacts)
            self.assertEqual(config.audio_analysis.initial_analysis_concurrency, 1)
            self.assertEqual(config.audio_analysis.diarization_backend, "pyannote")
            self.assertEqual(
                config.audio_analysis.overlap_conditioned_asr_seconds, 1.5
            )
            self.assertEqual(config.audio_analysis.conditioned_asr_backend, "dicow")
            self.assertEqual(config.audio_analysis.moss_window_seconds, 480.0)
            self.assertEqual(config.audio_analysis.moss_max_window_seconds, 540.0)
            self.assertEqual(
                config.audio_analysis.speaker_embedding_backend, "eres2netv2"
            )
            self.assertEqual(
                config.audio_analysis.speaker_embedding_model,
                "iic/speech_eres2netv2_sv_zh-cn_16k-common",
            )
            self.assertEqual(
                config.audio_analysis.diarization_model,
                "pyannote/speaker-diarization-community-1",
            )
            self.assertEqual(
                config.audio_analysis.overlap_separation_model,
                "MossFormer2_SS_16K",
            )
            self.assertEqual(
                config.audio_analysis.overlap_separation_worker_project,
                "tools/mossformer2",
            )
            self.assertEqual(
                config.audio_analysis.speaker_enrollment_samples_per_video, 40
            )
            self.assertEqual(
                config.audio_analysis.speaker_overlap_match_threshold, 0.24
            )
            self.assertEqual(config.audio_analysis.speaker_match_margin, 0.03)
            self.assertEqual(config.audio_analysis.speaker_identity_trim_ratio, 0.15)
            self.assertEqual(
                config.audio_analysis.speaker_identity_max_weight_seconds, 10.0
            )
            self.assertEqual(config.audio_analysis.speaker_profile_max_centers, 5)
            self.assertEqual(
                config.audio_analysis.speaker_profile_min_samples_per_center, 20
            )
            self.assertEqual(config.audio_analysis.singing_threshold, 0.05)
            self.assertEqual(config.audio_analysis.singing_vocal_threshold, 0.5)
            self.assertEqual(config.audio_analysis.singing_speech_bgm_coverage, 0.35)
            self.assertEqual(config.audio_analysis.singing_ambiguous_min_seconds, 15.0)
            self.assertEqual(config.audio_analysis.singing_smoothing_windows, 3)
            self.assertEqual(config.audio_analysis.singing_release_seconds, 35.0)
            self.assertEqual(config.audio_analysis.singing_min_phrase_seconds, 30.0)
            self.assertFalse(config.song_identification.enabled)
            self.assertEqual(config.song_identification.device, "gpu:0")
            self.assertEqual(config.segmentation.model_window_cues, 600)
            self.assertEqual(config.llm.max_concurrency, 16)
            self.assertEqual(config.render.font_size_ratio, 0.066)
            self.assertEqual(config.render.portrait_font_size_ratio, 0.077)
            self.assertEqual(config.render.max_font_size, 144)
            self.assertEqual(config.render.margin_horizontal_ratio, 0.075)
            self.assertEqual(config.render.portrait_margin_horizontal_ratio, 0.025)
            self.assertEqual(config.render.margin_vertical_ratio, 0.05)
            self.assertEqual(config.render.outline_ratio, 0.003)
            self.assertEqual(config.render.backend, "auto")
            self.assertEqual(config.render.nvenc_preset, "p4")
            self.assertEqual(config.render.nvenc_cq, 23)
            self.assertEqual(config.upload.cooldown_min_seconds, 60)
            self.assertEqual(config.upload.cooldown_max_seconds, 120)
            self.assertEqual(
                config.upload.rate_limit_retry_delays_seconds,
                [120, 300, 600, 1200],
            )
            self.assertIn("vcodec^=vp9]", config.download.video_format)
            self.assertIn("vcodec^=vp09", config.download.video_format)

    def test_rejects_unknown_render_backend(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text('[render]\nbackend = "magic"\n', encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "render.backend"):
                load_config(path)

    def test_rejects_invalid_max_tokens(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text("[llm]\nmax_tokens = 0\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "max_tokens"):
                load_config(path)

    def test_rejects_invalid_llm_concurrency(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text("[llm]\nmax_concurrency = 0\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "max_concurrency"):
                load_config(path)

    def test_rejects_negative_context_cues(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text("[llm]\ncontext_cues = -1\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "context_cues"):
                load_config(path)

    def test_rejects_invalid_song_ocr_interval(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                "[song_identification]\nsample_interval_seconds = 0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "sample_interval_seconds"):
                load_config(path)

    def test_rejects_unknown_speaker_embedding_backend(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                '[audio_analysis]\nspeaker_embedding_backend = "mystery"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "speaker_embedding_backend"):
                load_config(path)

    def test_rejects_excessive_initial_audio_analysis_concurrency(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                "[audio_analysis]\ninitial_analysis_concurrency = 3\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "initial_analysis_concurrency"):
                load_config(path)

    def test_rejects_asr_chunk_longer_than_aligner_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                "[asr]\nchunk_seconds = 175\nchunk_context_seconds = 3\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "180 seconds"):
                load_config(path)

    def test_api_key_comes_from_named_environment_variable_when_pass_is_disabled(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                '[llm]\napi_key_pass_entry = ""\napi_key_env = "TEST_LLM_KEY"\n',
                encoding="utf-8",
            )
            config = load_config(path)
            with patch.dict(os.environ, {"TEST_LLM_KEY": "secret"}, clear=True):
                self.assertEqual(llm_api_key(config.llm), "secret")

    def test_api_key_comes_from_first_line_of_pass_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                '[llm]\napi_key_pass_entry = "api/deepseek"\n', encoding="utf-8"
            )
            config = load_config(path)
            completed = __import__("subprocess").CompletedProcess(
                ["pass", "show", "api/deepseek"],
                0,
                "secret-key\nmetadata: ignored\n",
                "",
            )
            with (
                patch(
                    "subtitle_pipeline.config.shutil.which",
                    return_value="/usr/bin/pass",
                ),
                patch(
                    "subtitle_pipeline.config.subprocess.run", return_value=completed
                ) as run,
            ):
                self.assertEqual(llm_api_key(config.llm), "secret-key")
            run.assert_called_once_with(
                ["/usr/bin/pass", "show", "api/deepseek"],
                check=True,
                capture_output=True,
                text=True,
            )


if __name__ == "__main__":
    unittest.main()
