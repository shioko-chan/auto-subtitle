from __future__ import annotations

import json
import logging
import math
import os
import subprocess
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

from .commands import require_command
from .telemetry import stage_metrics

logger = logging.getLogger(__name__)

_SHARED_URI_PREFIX = "shm://"


@dataclass(frozen=True)
class AudioBufferDescriptor:
    shared_memory: str
    dtype: str
    shape: tuple[int, ...]
    sample_rate: int

    def as_dict(self) -> dict[str, object]:
        return {
            "shared_memory": self.shared_memory,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "sample_rate": self.sample_rate,
        }


class AudioBuffer:
    def __init__(
        self,
        memory: shared_memory.SharedMemory,
        samples: Any,
        sample_rate: int,
        *,
        owner: bool,
    ) -> None:
        self._memory = memory
        self.samples = samples
        self.sample_rate = sample_rate
        self.owner = owner

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def uri(self) -> str:
        return _SHARED_URI_PREFIX + self._memory.name

    @property
    def descriptor(self) -> AudioBufferDescriptor:
        return AudioBufferDescriptor(
            self._memory.name,
            str(self.samples.dtype),
            tuple(self.samples.shape),
            self.sample_rate,
        )

    def slice(self, start: float, end: float, *, copy: bool = False) -> Any:
        start_sample = max(0, round(start * self.sample_rate))
        end_sample = min(len(self.samples), round(end * self.sample_rate))
        value = self.samples[start_sample:end_sample]
        return value.copy() if copy else value

    def close(self) -> None:
        import numpy as np

        self.samples = np.empty(0, dtype=np.float32)
        self._memory.close()
        if self.owner:
            try:
                self._memory.unlink()
            except FileNotFoundError:
                pass


class AudioBufferPool:
    """Own shared CPU audio for one pipeline job."""

    def __init__(self, video: Path, job_dir: Path, duration: float) -> None:
        self.video = video
        self.job_dir = job_dir
        self.duration = duration
        self.registry_path = job_dir / "audio-shared-memory.json"
        self._buffers: dict[str, AudioBuffer] = {}
        self._sources: dict[Path, AudioBuffer] = {}
        self._main: AudioBuffer | None = None
        self._cleanup_abandoned()

    def __enter__(self) -> AudioBufferPool:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def main(self) -> AudioBuffer:
        if self._main is None:
            with stage_metrics("audio.shared_memory_decode"):
                self._main = self._decode_main()
        return self._main

    def source(self, path: Path) -> AudioBuffer:
        import numpy as np

        resolved = path.resolve()
        if resolved == self.video.resolve():
            return self.main()
        if resolved not in self._sources:
            with stage_metrics("audio.secondary_source_decode"):
                ffmpeg = require_command("ffmpeg")
                result = subprocess.run(
                    [
                        ffmpeg,
                        "-v",
                        "error",
                        "-i",
                        str(resolved),
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-f",
                        "f32le",
                        "pipe:1",
                    ],
                    check=False,
                    capture_output=True,
                )
            if result.returncode != 0:
                raise RuntimeError(
                    f"could not decode audio source {resolved}: "
                    + result.stderr.decode("utf-8", errors="replace")[-2000:]
                )
            values = np.frombuffer(result.stdout, dtype=np.float32)
            if not len(values):
                raise RuntimeError(f"audio source contains no samples: {resolved}")
            self._sources[resolved] = self.add(values, 16000)
        return self._sources[resolved]

    def add(self, samples: Any, sample_rate: int) -> AudioBuffer:
        import numpy as np

        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        memory = shared_memory.SharedMemory(create=True, size=max(1, values.nbytes))
        target = np.ndarray(values.shape, dtype=np.float32, buffer=memory.buf)
        target[:] = values
        buffer = AudioBuffer(memory, target, sample_rate, owner=True)
        self._buffers[memory.name] = buffer
        self._write_registry()
        return buffer

    def resolve(self, uri: str) -> AudioBuffer:
        import numpy as np

        if not uri.startswith(_SHARED_URI_PREFIX):
            raise ValueError(f"not a shared audio URI: {uri}")
        name = uri.removeprefix(_SHARED_URI_PREFIX)
        if name in self._buffers:
            return self._buffers[name]
        memory = shared_memory.SharedMemory(name=name)
        metadata = self._registry_values().get(name)
        if not isinstance(metadata, dict):
            memory.close()
            raise RuntimeError(f"missing shared audio metadata for {name}")
        shape = tuple(int(item) for item in metadata["shape"])
        dtype = np.dtype(str(metadata["dtype"]))
        samples = np.ndarray(shape, dtype=dtype, buffer=memory.buf)
        buffer = AudioBuffer(
            memory, samples, int(metadata["sample_rate"]), owner=False
        )
        self._buffers[name] = buffer
        return buffer

    def contains(self, uri: str) -> bool:
        return uri.startswith(_SHARED_URI_PREFIX) and uri.removeprefix(
            _SHARED_URI_PREFIX
        ) in self._buffers

    def close(self) -> None:
        for buffer in list(self._buffers.values()):
            buffer.close()
        self._buffers.clear()
        self._sources.clear()
        self._main = None
        self.registry_path.unlink(missing_ok=True)

    def _decode_main(self) -> AudioBuffer:
        import numpy as np

        sample_rate = 16000
        reserve_samples = math.ceil(self.duration * sample_rate) + sample_rate * 10
        memory = shared_memory.SharedMemory(
            create=True, size=max(4, reserve_samples * np.dtype(np.float32).itemsize)
        )
        ffmpeg = require_command("ffmpeg")
        command = [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(self.video.resolve()),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ]
        logger.info("decoding 16 kHz mono audio once into shared memory")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        byte_view = memory.buf.cast("B")
        view_released = False
        written = 0
        try:
            assert process.stdout is not None
            while True:
                if written >= len(byte_view):
                    raise RuntimeError("decoded audio exceeded the allocated shared memory")
                count = process.stdout.readinto(byte_view[written:])
                if not count:
                    break
                written += count
            stderr = process.stderr.read() if process.stderr is not None else b""
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(
                    "ffmpeg audio decode failed: "
                    + stderr.decode("utf-8", errors="replace")[-2000:]
                )
            sample_count = written // np.dtype(np.float32).itemsize
            if sample_count == 0:
                raise RuntimeError("ffmpeg decoded no audio samples")
            samples = np.ndarray(
                (sample_count,), dtype=np.float32, buffer=memory.buf
            )
            buffer = AudioBuffer(memory, samples, sample_rate, owner=True)
            self._buffers[memory.name] = buffer
            self._write_registry()
            logger.info(
                "decoded %.1fs audio into shared memory %s (%.1f MiB)",
                buffer.duration,
                memory.name,
                samples.nbytes / (1024 * 1024),
            )
            return buffer
        except Exception:
            process.kill()
            process.wait()
            byte_view.release()
            view_released = True
            memory.close()
            memory.unlink()
            raise
        finally:
            if not view_released:
                byte_view.release()
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()

    def _cleanup_abandoned(self) -> None:
        payload = self._registry_payload()
        owner_pid = payload.get("owner_pid")
        if isinstance(owner_pid, int) and owner_pid != os.getpid():
            try:
                os.kill(owner_pid, 0)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                raise RuntimeError(
                    f"audio shared memory is owned by inaccessible process {owner_pid}"
                ) from exc
            else:
                raise RuntimeError(
                    f"audio shared memory is still owned by running process {owner_pid}"
                )
        buffers = payload.get("buffers", {})
        for name in buffers if isinstance(buffers, dict) else {}:
            try:
                memory = shared_memory.SharedMemory(name=name)
                memory.close()
                memory.unlink()
                logger.warning("removed abandoned shared audio segment %s", name)
            except FileNotFoundError:
                pass
        self.registry_path.unlink(missing_ok=True)

    def _registry_values(self) -> dict[str, object]:
        value = self._registry_payload()
        buffers = value.get("buffers")
        return buffers if isinstance(buffers, dict) else {}

    def _registry_payload(self) -> dict[str, object]:
        if not self.registry_path.is_file():
            return {}
        try:
            value = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _write_registry(self) -> None:
        self.job_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "owner_pid": os.getpid(),
            "buffers": {
                name: buffer.descriptor.as_dict()
                for name, buffer in self._buffers.items()
                if buffer.owner
            },
        }
        temporary = self.registry_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.registry_path)


def is_shared_audio_uri(value: str | None) -> bool:
    return bool(value and value.startswith(_SHARED_URI_PREFIX))
