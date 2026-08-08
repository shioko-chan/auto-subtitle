import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from subtitle_pipeline.config import ConfigError, llm_api_key, load_config


class ConfigTests(unittest.TestCase):
    def test_loads_defaults_and_overrides(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(
                'work_dir = "jobs"\n[llm]\nmodel = "test-model"\n[upload]\nenabled = true\n',
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.work_dir, Path("jobs"))
            self.assertEqual(config.llm.model, "test-model")
            self.assertTrue(config.upload.enabled)
            self.assertEqual(config.whisper.model, "small")

    def test_rejects_invalid_batch_size(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text("[llm]\nbatch_size = 0\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "batch_size"):
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
            with patch(
                "subtitle_pipeline.config.shutil.which", return_value="/usr/bin/pass"
            ), patch(
                "subtitle_pipeline.config.subprocess.run", return_value=completed
            ) as run:
                self.assertEqual(llm_api_key(config.llm), "secret-key")
            run.assert_called_once_with(
                ["/usr/bin/pass", "show", "api/deepseek"],
                check=True,
                capture_output=True,
                text=True,
            )


if __name__ == "__main__":
    unittest.main()
