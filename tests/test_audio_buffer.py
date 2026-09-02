import json
import subprocess
import sys
import tempfile
import unittest
from multiprocessing import shared_memory
from pathlib import Path
from unittest.mock import patch

import numpy as np

from subtitle_pipeline.audio_buffer import (
    AudioBufferPool,
    _decode_process_once,
    _decode_process_with_retries,
)


class AudioBufferTests(unittest.TestCase):
    def test_audio_decode_process_times_out(self):
        target = memoryview(bytearray(16))
        try:
            with self.assertRaises(subprocess.TimeoutExpired):
                _decode_process_once(
                    [sys.executable, "-c", "import time; time.sleep(5)"],
                    target,
                    0.05,
                )
        finally:
            target.release()

    def test_audio_decode_retries_once_after_timeout(self):
        target = memoryview(bytearray(16))
        timeout = subprocess.TimeoutExpired(["ffmpeg"], 60)
        try:
            with patch(
                "subtitle_pipeline.audio_buffer._decode_process_once",
                side_effect=[timeout, 8],
            ) as decode:
                written = _decode_process_with_retries(
                    ["ffmpeg"], target, timeout_seconds=60, attempts=2
                )
        finally:
            target.release()

        self.assertEqual(written, 8)
        self.assertEqual(decode.call_count, 2)

    def test_shared_buffer_registry_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "source.mp4"
            video.write_bytes(b"video")
            pool = AudioBufferPool(video, root, 1.0)
            buffer = pool.add(np.asarray([0.1, 0.2], dtype=np.float32), 16000)
            name = buffer.descriptor.shared_memory

            registry = json.loads(
                (root / "audio-shared-memory.json").read_text(encoding="utf-8")
            )
            self.assertIn(name, registry["buffers"])
            self.assertTrue(np.allclose(pool.resolve(buffer.uri).samples, [0.1, 0.2]))

            pool.close()
            self.assertFalse((root / "audio-shared-memory.json").exists())
            with self.assertRaises(FileNotFoundError):
                shared_memory.SharedMemory(name=name)

    def test_abandoned_registry_is_cleaned_on_next_job(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "source.mp4"
            video.write_bytes(b"video")
            memory = shared_memory.SharedMemory(create=True, size=16)
            name = memory.name
            memory.close()
            (root / "audio-shared-memory.json").write_text(
                json.dumps({"buffers": {name: {}}}), encoding="utf-8"
            )

            pool = AudioBufferPool(video, root, 1.0)
            pool.close()

            with self.assertRaises(FileNotFoundError):
                shared_memory.SharedMemory(name=name)


if __name__ == "__main__":
    unittest.main()
