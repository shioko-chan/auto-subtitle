from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from subtitle_pipeline.config import LLMConfig
from subtitle_pipeline.local_llm_server import LocalLLMServer, build_server_command


class LocalLLMServerTests(unittest.TestCase):
    def test_builds_hugging_face_server_command(self):
        config = LLMConfig(
            local_server_enabled=True,
            local_server_hf_repo="unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M",
            model="Qwen3.8-27B",
        )
        with patch(
            "subtitle_pipeline.local_llm_server.require_command",
            return_value="/nix/store/llama-server",
        ):
            command = build_server_command(config)

        self.assertIn("--hf-repo", command)
        self.assertIn("unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M", command)
        self.assertIn("--n-gpu-layers", command)
        self.assertIn("99", command)
        self.assertIn("--alias", command)
        self.assertIn("Qwen3.8-27B", command)
        self.assertIn("--reasoning", command)
        self.assertIn("off", command)
        context_index = command.index("--ctx-size")
        self.assertEqual(command[context_index + 1], "16384")
        self.assertIn("--no-mmproj", command)

    def test_starts_and_stops_server_process(self):
        config = LLMConfig(
            local_server_enabled=True,
            local_server_hf_repo="unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M",
        )
        process = Mock()
        process.poll.return_value = None
        with (
            tempfile.TemporaryDirectory() as temp,
            patch(
                "subtitle_pipeline.local_llm_server.require_command",
                return_value="/nix/store/llama-server",
            ),
            patch(
                "subtitle_pipeline.local_llm_server.subprocess.Popen",
                return_value=process,
            ) as popen,
            patch.object(LocalLLMServer, "_wait_until_ready"),
            patch.object(LocalLLMServer, "_release_cuda_cache"),
        ):
            server = LocalLLMServer(config, Path(temp) / "llama.log")
            server.start()
            server.stop()

        popen.assert_called_once()
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=30)

    def test_disabled_server_is_a_noop(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("subtitle_pipeline.local_llm_server.subprocess.Popen") as popen,
        ):
            server = LocalLLMServer(LLMConfig(), Path(temp) / "llama.log")
            server.start()
            server.stop()
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
