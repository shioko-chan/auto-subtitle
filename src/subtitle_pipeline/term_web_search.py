from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class TermWebSearcher:
    def __init__(self, worker_project: Path, *, maximum_results: int = 4) -> None:
        self._worker_project = worker_project.resolve()
        self._maximum_results = maximum_results

    def search(self, query: str) -> dict[str, object]:
        worker = self._worker_project / "worker.py"
        uv = shutil.which("uv")
        if uv is None or not worker.is_file():
            raise RuntimeError(f"term search worker is unavailable at {worker}")
        completed = subprocess.run(
            [uv, "run", "--project", str(self._worker_project), "python", str(worker)],
            input=json.dumps(
                {
                    "action": "search",
                    "query": query,
                    "limit": self._maximum_results,
                },
                ensure_ascii=False,
            ),
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr[-500:] or "term search worker failed")
        response = json.loads(completed.stdout)
        if not isinstance(response, dict):
            raise TypeError("term search worker returned malformed JSON")
        return response
