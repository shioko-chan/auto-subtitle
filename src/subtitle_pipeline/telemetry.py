from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)


@contextmanager
def stage_metrics(name: str, device: str | None = None) -> Iterator[None]:
    torch = None
    cuda = bool(device and device.startswith("cuda"))
    if cuda:
        try:
            import torch as torch_module

            torch = torch_module
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            else:
                torch = None
        except (ImportError, RuntimeError):
            torch = None
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        peak = None
        if torch is not None:
            try:
                peak = torch.cuda.max_memory_allocated(device) / (1024**3)
            except RuntimeError:
                pass
        logger.info(
            "stage=%s elapsed_seconds=%.3f peak_vram_gib=%s",
            name,
            elapsed,
            f"{peak:.3f}" if peak is not None else "unknown",
        )
