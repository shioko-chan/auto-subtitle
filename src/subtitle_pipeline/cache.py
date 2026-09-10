"""Job-local, explicitly versioned computation results.

Change a stage version when its processing contract changes. Inputs and code are
deliberately not hashed: an existing plan is the execution snapshot on resume.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageDefinition:
    version: int
    parents: tuple[str, ...] = ()


STAGES = {
    "download": StageDefinition(1),
    "audio_analysis": StageDefinition(1, ("download",)),
    "raw_speech": StageDefinition(3, ("audio_analysis",)),
    "singing_asr": StageDefinition(1, ("audio_analysis",)),
    "conditioned_asr": StageDefinition(1, ("audio_analysis",)),
    "song_identification": StageDefinition(4, ("raw_speech", "singing_asr")),
    "asr_correction": StageDefinition(3, ("raw_speech", "conditioned_asr")),
    "speech_alignment": StageDefinition(2, ("asr_correction",)),
    "source_cues": StageDefinition(1, ("speech_alignment", "song_identification", "conditioned_asr")),
    "lyrics_translation": StageDefinition(1, ("song_identification",)),
    "translation": StageDefinition(2, ("source_cues", "lyrics_translation")),
    "metadata": StageDefinition(1, ("source_cues", "song_identification")),
    "render": StageDefinition(1, ("translation",)),
    "clip_analysis": StageDefinition(1, ("render", "metadata")),
    "clip_render": StageDefinition(1, ("clip_analysis", "render")),
}

_STORE_LOCKS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_STORE_LOCKS_GUARD = threading.Lock()


def json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_encode, ensure_ascii=False))


def _encode(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not a cache value: {type(value).__name__}")


def config_snapshot(config: Any) -> dict[str, Any]:
    """Exclude credential locations as well as credentials from durable plans."""
    def scrub(value):
        if isinstance(value, dict):
            return {key: scrub(item) for key, item in value.items()
                    if not any(part in key.lower() for part in ("api_key", "cookie", "token", "password", "secret", "credential"))
                    or key in {"max_tokens", "max_new_tokens"}}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value
    return scrub(json_value(config))


def restore_config(config: Any, snapshot: dict[str, Any]) -> Any:
    from dataclasses import replace
    return replace(config, **snapshot)


class CacheStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _STORE_LOCKS_GUARD:
            key = str(path.resolve())
            self._lock = _STORE_LOCKS.setdefault(key, threading.RLock())
        with self._connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS stages (
                    name TEXT PRIMARY KEY, version INTEGER NOT NULL,
                    generation INTEGER NOT NULL DEFAULT 1,
                    parents TEXT NOT NULL DEFAULT '{}', plan TEXT,
                    complete INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS units (
                    stage TEXT NOT NULL, id TEXT NOT NULL, kind TEXT NOT NULL,
                    status TEXT NOT NULL, source TEXT NOT NULL, reason TEXT,
                    payload TEXT NOT NULL, retry INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(stage, id)
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    stage TEXT NOT NULL, unit TEXT NOT NULL, error TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
            """)
            changed = [row["name"] for row in db.execute("SELECT * FROM stages")
                       if row["name"] not in STAGES or row["version"] != STAGES[row["name"]].version]
            if changed:
                self._invalidate(db, self.descendants(changed))

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            db = sqlite3.connect(self.path, timeout=30)
            db.row_factory = sqlite3.Row
            try:
                with db:
                    yield db
            finally:
                db.close()

    @staticmethod
    def descendants(names: list[str]) -> set[str]:
        selected = set(names)
        while True:
            expanded = selected | {name for name, definition in STAGES.items()
                                   if selected.intersection(definition.parents)}
            if expanded == selected:
                return selected
            selected = expanded

    def _invalidate(self, db: sqlite3.Connection, names: set[str]) -> None:
        for name in names:
            db.execute("DELETE FROM units WHERE stage=?", (name,))
            db.execute("UPDATE stages SET version=?, generation=generation+1, plan=NULL, complete=0 WHERE name=?",
                       (STAGES[name].version if name in STAGES else 0, name))

    def reset(self, name: str | None = None) -> None:
        if name is not None and name not in STAGES:
            raise ValueError(f"unknown cache stage: {name}")
        with self._connection() as db:
            self._invalidate(db, self.descendants([name]) if name else set(STAGES))

    def retry_degraded(self, name: str) -> int:
        if name not in STAGES:
            raise ValueError(f"unknown cache stage: {name}")
        with self._connection() as db:
            count = db.execute("SELECT count(*) FROM units WHERE stage=? AND status='degraded' AND kind='result'", (name,)).fetchone()[0]
            if count:
                db.execute("UPDATE units SET retry=1 WHERE stage=? AND status='degraded' AND kind='result'", (name,))
                db.execute("DELETE FROM units WHERE stage=? AND kind='aggregate'", (name,))
                db.execute("UPDATE stages SET complete=0, generation=generation+1 WHERE name=?", (name,))
                self._invalidate(db, self.descendants([name]) - {name})
            return count

    def restore_interrupted_retries(self) -> None:
        """Called once after acquiring the job lock, before a new run starts."""
        with self._connection() as db:
            db.execute("INSERT INTO attempts(stage,unit,error) SELECT stage,id,'explicit retry interrupted before replacement' FROM units WHERE retry=1")
            db.execute("UPDATE units SET retry=0 WHERE retry=1")

    @classmethod
    def inspect(cls, path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        try:
            return cls._status(db)
        finally:
            db.close()

    def stage(self, name: str, make_plan: Callable[[], Any]) -> StageCache:
        definition = STAGES[name]
        with self._connection() as db:
            row = db.execute("SELECT * FROM stages WHERE name=?", (name,)).fetchone()
            plan = json.loads(row["plan"]) if row is not None and row["plan"] is not None else None
        # Plan construction can perform expensive preparation; never inside a transaction.
        if plan is None:
            plan = json_value(make_plan())
            with self._connection() as db:
                parents = {parent: {"version": STAGES[parent].version,
                           "generation": (db.execute("SELECT generation FROM stages WHERE name=?", (parent,)).fetchone() or [0])[0]}
                           for parent in definition.parents}
                db.execute("INSERT INTO stages(name, version, parents, plan) VALUES(?,?,?,?) ON CONFLICT(name) DO UPDATE SET version=excluded.version, parents=excluded.parents, plan=excluded.plan, complete=0",
                           (name, definition.version, json.dumps(parents), json.dumps(plan, ensure_ascii=False)))
        return StageCache(self, name, plan)

    def existing(self, name: str) -> StageCache | None:
        with self._connection() as db:
            row = db.execute("SELECT plan FROM stages WHERE name=?", (name,)).fetchone()
        if row is None or row[0] is None:
            return None
        return StageCache(self, name, json.loads(row[0]))

    def status(self) -> list[dict[str, Any]]:
        with self._connection() as db:
            return self._status(db)

    @staticmethod
    def _status(db) -> list[dict[str, Any]]:
        result = []
        for row in db.execute("SELECT * FROM stages ORDER BY name"):
            values = dict(row)
            values.pop("plan")
            values["parents"] = json.loads(values["parents"])
            values["units"] = [dict(unit) for unit in db.execute(
                "SELECT id,status,source,reason,retry FROM units WHERE stage=? AND kind='result' ORDER BY id", (row["name"],))]
            values["completed_units"] = sum(not unit["retry"] for unit in values["units"])
            values["current_version"] = STAGES[row["name"]].version if row["name"] in STAGES else None
            result.append(values)
        return result


class StageCache:
    def __init__(self, store: CacheStore, name: str, plan: Any):
        self.store, self.name, self.plan = store, name, plan

    def record(self, unit: str, *, include_retry: bool = False) -> dict[str, Any] | None:
        with self.store._connection() as db:
            row = db.execute("SELECT * FROM units WHERE stage=? AND id=?", (self.name, str(unit))).fetchone()
        if row is None or (row["retry"] and not include_retry):
            return None
        value = dict(row)
        value["payload"] = json.loads(value["payload"])
        return value

    def get(self, unit: str) -> Any | None:
        value = self.record(unit)
        return value["payload"] if value is not None else None

    def put(self, unit: str, payload: Any, *, source: str = "computed", reason: str | None = None,
            kind: str = "result") -> None:
        encoded = json.dumps(payload, default=_encode, ensure_ascii=False)
        with self.store._connection() as db:
            db.execute("INSERT OR REPLACE INTO units(stage,id,kind,status,source,reason,payload,retry) VALUES(?,?,?,?,?,?,?,0)",
                       (self.name, str(unit), kind, "degraded" if reason else "success", source, reason, encoded))

    def remember(self, unit: str, create: Callable[[], Any]) -> Any:
        value = self.get(unit)
        if value is None:
            value = json_value(create())
            self.put(unit, value, kind="plan")
        return value

    def failed(self, unit: str, error: BaseException) -> None:
        with self.store._connection() as db:
            db.execute("INSERT INTO attempts(stage,unit,error) VALUES(?,?,?)", (self.name, str(unit), f"{type(error).__name__}: {error}"))
            db.execute("UPDATE units SET retry=0 WHERE stage=? AND id=?", (self.name, str(unit)))

    @contextmanager
    def attempt(self, unit: str):
        try:
            yield
        except BaseException as exc:
            self.failed(unit, exc)
            raise

    def finish(self, result: Any) -> None:
        with self.store._connection() as db:
            db.execute("INSERT OR REPLACE INTO units(stage,id,kind,status,source,payload) VALUES(?, '__result__', 'aggregate', 'success', 'computed', ?)",
                       (self.name, json.dumps(result, default=_encode, ensure_ascii=False)))
            db.execute("UPDATE stages SET complete=1 WHERE name=?", (self.name,))


@contextmanager
def job_lock(directory: Path) -> Iterator[None]:
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".job.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"job is already running: {directory}") from exc
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
