from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_ACTIVE_MONITOR: PerformanceMonitor | None = None
_ACTIVE_MONITOR_LOCK = threading.Lock()


class PerformanceMonitor:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started_at = datetime.now(UTC)
        self.started_perf = time.perf_counter()
        self._records: list[dict[str, object]] = []
        self._lock = threading.Lock()

    def record(
        self,
        name: str,
        started_perf: float,
        elapsed: float,
        peak_vram_gib: float | None,
        error: BaseException | None,
    ) -> None:
        record: dict[str, object] = {
            "name": name,
            "started_offset_seconds": round(
                max(0.0, started_perf - self.started_perf), 3
            ),
            "elapsed_seconds": round(elapsed, 3),
            "status": "failed" if error is not None else "completed",
            "peak_vram_gib": (
                round(peak_vram_gib, 3) if peak_vram_gib is not None else None
            ),
        }
        if error is not None:
            record["error"] = f"{type(error).__name__}: {error}"[:1000]
        with self._lock:
            self._records.append(record)

    def write(self, status: str, error: BaseException | None = None) -> None:
        finished_at = datetime.now(UTC)
        elapsed = max(0.0, time.perf_counter() - self.started_perf)
        with self._lock:
            records = sorted(
                (dict(record) for record in self._records),
                key=lambda record: (
                    float(record["started_offset_seconds"]),
                    str(record["name"]),
                ),
            )
        payload: dict[str, object] = {
            "version": 1,
            "status": status,
            "started_at": self.started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "elapsed_seconds": round(elapsed, 3),
            "stages": records,
            "summary": _summarize(records),
        }
        if error is not None:
            payload["error"] = f"{type(error).__name__}: {error}"[:1000]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
        logger.info(
            "performance report: status=%s elapsed_seconds=%.3f path=%s",
            status,
            elapsed,
            self.path,
        )


def _summarize(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["name"]), []).append(record)
    summary: dict[str, dict[str, object]] = {}
    for name, values in sorted(grouped.items()):
        elapsed = [float(value["elapsed_seconds"]) for value in values]
        peaks = [
            float(value["peak_vram_gib"])
            for value in values
            if isinstance(value.get("peak_vram_gib"), (int, float))
        ]
        summary[name] = {
            "calls": len(values),
            "completed": sum(value.get("status") == "completed" for value in values),
            "failed": sum(value.get("status") == "failed" for value in values),
            "total_elapsed_seconds": round(sum(elapsed), 3),
            "max_elapsed_seconds": round(max(elapsed), 3),
            "peak_vram_gib": round(max(peaks), 3) if peaks else None,
        }
    return summary


@contextmanager
def pipeline_metrics(path: Path) -> Iterator[PerformanceMonitor]:
    global _ACTIVE_MONITOR
    monitor = PerformanceMonitor(path)
    with _ACTIVE_MONITOR_LOCK:
        if _ACTIVE_MONITOR is not None:
            raise RuntimeError("a pipeline performance monitor is already active")
        _ACTIVE_MONITOR = monitor
    error: BaseException | None = None
    try:
        yield monitor
    except BaseException as exc:
        error = exc
        raise
    finally:
        with _ACTIVE_MONITOR_LOCK:
            _ACTIVE_MONITOR = None
        try:
            monitor.write("failed" if error is not None else "completed", error)
        except Exception:
            logger.exception("could not write pipeline performance report: %s", path)
            if error is None:
                raise


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
    error: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        error = exc
        raise
    finally:
        elapsed = time.perf_counter() - started
        peak = None
        if torch is not None:
            try:
                peak = torch.cuda.max_memory_allocated(device) / (1024**3)
            except RuntimeError:
                pass
        monitor = _ACTIVE_MONITOR
        if monitor is not None:
            monitor.record(name, started, elapsed, peak, error)
        logger.info(
            "stage=%s elapsed_seconds=%.3f peak_vram_gib=%s status=%s",
            name,
            elapsed,
            f"{peak:.3f}" if peak is not None else "unknown",
            "failed" if error is not None else "completed",
        )
