from __future__ import annotations

import gc
import logging
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .commands import require_command
from .config import LLMConfig

_LOGGER = logging.getLogger(__name__)


class LocalLLMServer:
    """Own a llama.cpp server for the LLM-only portion of one pipeline job."""

    def __init__(self, config: LLMConfig, log_path: Path):
        self.config = config
        self.log_path = log_path
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle = None

    def start(self) -> None:
        if not self.config.local_server_enabled or self._process is not None:
            return
        command = build_server_command(self.config)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._release_cuda_cache()
        self._log_handle = self.log_path.open("ab")
        _LOGGER.info(
            "starting local LLM server model=%s endpoint=%s",
            self.config.model,
            self.config.base_url,
        )
        try:
            self._process = subprocess.Popen(
                command,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
            )
            self._wait_until_ready()
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            _LOGGER.info("stopping local LLM server")
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _LOGGER.warning("local LLM server did not stop in 30s; killing it")
                process.kill()
                process.wait(timeout=10)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def _wait_until_ready(self) -> None:
        assert self._process is not None
        deadline = time.monotonic() + self.config.local_server_startup_timeout_seconds
        health_url = (
            f"http://{self.config.local_server_host}:"
            f"{self.config.local_server_port}/health"
        )
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            return_code = self._process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"local LLM server exited with code {return_code}; see {self.log_path}"
                )
            try:
                with urllib.request.urlopen(health_url, timeout=2) as response:
                    if response.status == 200:
                        _LOGGER.info("local LLM server is ready")
                        return
            except (OSError, urllib.error.URLError) as exc:
                last_error = exc
            time.sleep(1)
        raise TimeoutError(
            "local LLM server did not become ready within "
            f"{self.config.local_server_startup_timeout_seconds}s; "
            f"last error: {last_error}; see {self.log_path}"
        )

    @staticmethod
    def _release_cuda_cache() -> None:
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is None:
            return
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and cuda.is_available():
            cuda.empty_cache()
            if cuda.is_initialized():
                cuda.ipc_collect()


def build_server_command(config: LLMConfig) -> list[str]:
    command = [
        require_command(config.local_server_command),
        "--host",
        config.local_server_host,
        "--port",
        str(config.local_server_port),
        "--alias",
        config.model,
        "--ctx-size",
        str(config.local_server_context_size * config.local_server_parallel),
        "--n-gpu-layers",
        str(config.local_server_gpu_layers),
        "--parallel",
        str(config.local_server_parallel),
        "--reasoning",
        config.local_server_reasoning,
        "--no-mmproj",
        "--jinja",
    ]
    if config.local_server_model_path:
        model_path = Path(config.local_server_model_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"local LLM GGUF not found: {model_path}")
        command.extend(("--model", str(model_path)))
    else:
        assert config.local_server_hf_repo
        command.extend(("--hf-repo", config.local_server_hf_repo))
    return command
