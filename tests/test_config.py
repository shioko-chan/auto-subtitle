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
            self.assertEqual(
                config.asr.singing_model, "HeartMuLa/HeartTranscriptor-oss"
            )
            self.assertEqual(config.asr.singing_max_new_tokens, 256)
            self.assertEqual(config.asr.singing_num_beams, 2)
            self.assertEqual(config.song_identification.lyric_neighbor_max_lines, 12)
            self.assertEqual(
                config.song_identification.lyric_neighbor_min_coverage, 0.45
            )
            self.assertEqual(config.asr.dtype, "float16")
            self.assertIsNone(config.asr.language)
            self.assertTrue(config.audio_analysis.enabled)
            self.assertFalse(config.audio_analysis.debug_audio_artifacts)
            self.assertEqual(config.audio_analysis.initial_analysis_concurrency, 1)
            self.assertEqual(config.audio_analysis.diarization_backend, "pyannote")
            self.assertEqual(config.audio_analysis.overlap_conditioned_asr_seconds, 0.5)
            self.assertEqual(config.audio_analysis.conditioned_asr_backend, "dicow")
            self.assertFalse(config.audio_analysis.skip_dicow_for_single_person_streams)
            self.assertEqual(config.audio_analysis.conditioned_asr_batch_size, 4)
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
                config.audio_analysis.speaker_enrollment_samples_per_video, 40
            )
            self.assertEqual(config.audio_analysis.speaker_match_threshold, 0.42)
            self.assertEqual(config.audio_analysis.speaker_match_margin, 0.025)
            self.assertEqual(config.audio_analysis.speaker_identity_trim_ratio, 0.15)
            self.assertEqual(
                config.audio_analysis.speaker_identity_max_weight_seconds, 10.0
            )
            self.assertEqual(config.audio_analysis.speaker_profile_max_centers, 5)
            self.assertEqual(
                config.audio_analysis.speaker_profile_min_samples_per_center, 20
            )
            self.assertEqual(config.audio_analysis.singing_threshold, 0.015)
            self.assertEqual(config.audio_analysis.singing_vocal_threshold, 0.15)
            self.assertEqual(config.audio_analysis.singing_smoothing_windows, 3)
            self.assertEqual(config.audio_analysis.singing_asr_target_seconds, 10.0)
            self.assertEqual(config.audio_analysis.singing_asr_min_seconds, 6.0)
            self.assertEqual(config.audio_analysis.singing_asr_max_seconds, 15.0)
            self.assertEqual(config.audio_analysis.singing_asr_search_seconds, 4.0)
            self.assertFalse(config.song_identification.enabled)
            self.assertEqual(config.song_identification.device, "cuda:0")
            self.assertEqual(
                config.song_identification.song_search_group_gap_seconds, 35.0
            )
            self.assertEqual(config.song_identification.lyric_gap_recheck_seconds, 20.0)
            self.assertEqual(
                config.song_identification.lyric_gap_vocal_active_ratio, 0.08
            )
            self.assertEqual(config.segmentation.boundary_score_threshold, 3)
            self.assertEqual(config.segmentation.local_unit_max_seconds, 6.0)
            self.assertEqual(config.segmentation.model_window_units, 240)
            self.assertEqual(config.segmentation.model_window_chars, 3000)
            self.assertEqual(config.asr_correction.batch_windows, 6)
            self.assertEqual(config.asr_correction.batch_chars, 3000)
            self.assertEqual(config.translation.batch_cues, 32)
            self.assertEqual(config.translation.batch_chars, 3000)
            self.assertEqual(config.llm.max_concurrency, 16)
            self.assertFalse(config.llm.local_server_enabled)
            self.assertEqual(config.translation.local_model, "facebook/m2m100_418M")
            self.assertEqual(config.translation.local_device, "cpu")
            self.assertEqual(config.render.font_size_ratio, 0.066)
            self.assertEqual(config.render.portrait_font_size_ratio, 0.077)
            self.assertEqual(config.render.max_font_size, 144)
            self.assertEqual(config.render.margin_horizontal_ratio, 0.075)
            self.assertEqual(config.render.portrait_margin_horizontal_ratio, 0.025)
            self.assertEqual(config.render.margin_vertical_ratio, 0.05)
            self.assertEqual(config.render.outline_ratio, 0.0045)
            self.assertEqual(config.render.backend, "auto")
            self.assertEqual(config.render.nvenc_preset, "p4")
            self.assertEqual(config.render.nvenc_cq, 20)
            self.assertEqual(config.upload.cooldown_min_seconds, 60)
            self.assertEqual(config.upload.cooldown_max_seconds, 120)
            self.assertFalse(config.clips.upload)
            self.assertEqual(config.clips.max_speech_seconds, 480)
            self.assertEqual(config.clips.chat_peak_zscore, 3.0)
            self.assertEqual(config.clips.chat_min_unique_authors, 5)
            self.assertEqual(
                config.upload.rate_limit_retry_delays_seconds,
                [120, 300, 600, 1200],
            )
            self.assertIn("vcodec^=vp9]", config.download.video_format)
            self.assertIn("vcodec^=vp09", config.download.video_format)
            self.assertEqual(config.download.concurrent_fragments, 8)

    def test_enables_single_person_dicow_skip(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                "[audio_analysis]\nskip_dicow_for_single_person_streams = true\n",
                encoding="utf-8",
            )

            config = load_config(path)

        self.assertTrue(config.audio_analysis.skip_dicow_for_single_person_streams)

    def test_rejects_invalid_download_fragment_concurrency(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text("[download]\nconcurrent_fragments = 0\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "concurrent_fragments"):
                load_config(path)

    def test_loads_clip_upload_independently_from_full_upload(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                "[upload]\nenabled = false\n[clips]\nupload = true\n",
                encoding="utf-8",
            )

            config = load_config(path)

            self.assertFalse(config.upload.enabled)
            self.assertTrue(config.clips.upload)

    def test_rejects_invalid_clip_thresholds(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                "[clips]\nchat_min_unique_authors = 0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "chat_min_unique_authors"):
                load_config(path)

    def test_rejects_unknown_render_backend(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text('[render]\nbackend = "magic"\n', encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "render.backend"):
                load_config(path)

    def test_rejects_invalid_max_tokens(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text("[translation]\nmax_tokens = 0\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "max_tokens"):
                load_config(path)

    def test_rejects_invalid_llm_concurrency(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text("[llm]\nmax_concurrency = 0\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "max_concurrency"):
                load_config(path)

    def test_rejects_empty_local_translation_model(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text('[translation]\nlocal_model = ""\n', encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "translation.local_model"):
                load_config(path)

    def test_rejects_stage_fields_in_llm_section(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                "[llm]\nasr_correction_batch_windows = 6\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ConfigError, "unknown.*configuration field"):
                load_config(path)

    def test_rejects_invalid_song_ocr_score(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                "[song_identification]\nminimum_ocr_score = 2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "minimum_ocr_score"):
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

    def test_asr_language_defaults_to_automatic_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text("", encoding="utf-8")

            config = load_config(path)

            self.assertIsNone(config.asr.language)

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

    def test_local_server_uses_dummy_api_key_and_validates_endpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                "[llm]\n"
                'base_url = "http://127.0.0.1:8080/v1"\n'
                "local_server_enabled = true\n"
                'local_server_hf_repo = "unsloth/model:UD-Q4_K_M"\n',
                encoding="utf-8",
            )
            config = load_config(path)

            self.assertEqual(llm_api_key(config.llm), "local-llama-cpp")

            path.write_text(
                "[llm]\n"
                'base_url = "http://127.0.0.1:8081/v1"\n'
                "local_server_enabled = true\n"
                'local_server_hf_repo = "unsloth/model:UD-Q4_K_M"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "local HTTP server port"):
                load_config(path)

    def test_loads_openai_responses_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                "[llm]\n"
                'base_url = "https://api.openai.com/v1"\n'
                'api_style = "responses"\n'
                'api_key_pass_entry = ""\n'
                'api_key_env = "OPENAI_API_KEY"\n'
                'model = "gpt-5.6"\n'
                'reasoning_effort = "low"\n',
                encoding="utf-8",
            )

            config = load_config(path)

        self.assertEqual(config.llm.api_style, "responses")
        self.assertEqual(config.llm.reasoning_effort, "low")

    def test_rejects_provider_specific_llm_options_on_wrong_api_style(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                '[llm]\napi_style = "responses"\nthinking = "disabled"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "reasoning_effort"):
                load_config(path)

            path.write_text(
                '[llm]\napi_style = "chat_completions"\nreasoning_effort = "low"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "only supported"):
                load_config(path)

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
