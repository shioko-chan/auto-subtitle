from __future__ import annotations

import hashlib
import json
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class LyricLine:
    line_no: int
    text: str
    reading: str | None = None
    translation: str | None = None
    translation_source: str | None = None


@dataclass(frozen=True)
class LibrarySong:
    song_id: str
    title: str
    artist: str
    aliases: tuple[str, ...]
    source_url: str
    source_hash: str
    lines: tuple[LyricLine, ...]


class LyricsLibrary:
    """Canonical released-song lyrics and translations.

    ASR hypotheses are deliberately absent from this schema. Only a fetched,
    structured source may create or replace Japanese lyric lines.
    """

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._database = sqlite3.connect(path)
        self._database.row_factory = sqlite3.Row
        self._database.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS songs (
                song_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                artist TEXT NOT NULL,
                aliases_json TEXT NOT NULL,
                source_url TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                fetched_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS lyric_lines (
                song_id TEXT NOT NULL REFERENCES songs(song_id) ON DELETE CASCADE,
                line_no INTEGER NOT NULL,
                text TEXT NOT NULL,
                reading TEXT,
                translation TEXT,
                translation_source TEXT,
                translation_model TEXT,
                translation_prompt_hash TEXT,
                translation_source_hash TEXT,
                PRIMARY KEY (song_id, line_no)
            );
            """
        )

    def close(self) -> None:
        self._database.close()

    def songs(self) -> list[LibrarySong]:
        return [
            self._load_song(row)
            for row in self._database.execute("SELECT * FROM songs")
        ]

    def get(self, song_id: str) -> LibrarySong | None:
        row = self._database.execute(
            "SELECT * FROM songs WHERE song_id = ?", (song_id,)
        ).fetchone()
        return self._load_song(row) if row is not None else None

    def store_canonical_song(
        self,
        *,
        title: str,
        artist: str,
        aliases: list[str] | tuple[str, ...],
        source_url: str,
        lines: list[tuple[str, str | None]],
    ) -> LibrarySong:
        cleaned = [
            (text.strip(), reading.strip() if reading else None)
            for text, reading in lines
        ]
        cleaned = [item for item in cleaned if item[0]]
        if len(cleaned) < 3 or not source_url.startswith(("http://", "https://")):
            raise ValueError(
                "canonical lyrics require a structured external source and at least 3 lines"
            )
        title = title.strip()
        artist = artist.strip()
        if not title or not artist:
            raise ValueError("canonical lyrics require a song title and artist")
        song_id = _song_id(title, artist)
        source_hash = _lyrics_hash([item[0] for item in cleaned])
        previous = self.get(song_id)
        previous_translations = (
            {line.line_no: line for line in previous.lines}
            if previous is not None and previous.source_hash == source_hash
            else {}
        )
        with self._database:
            self._database.execute(
                """INSERT INTO songs VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(song_id) DO UPDATE SET
                     title=excluded.title, artist=excluded.artist,
                     aliases_json=excluded.aliases_json,
                     source_url=excluded.source_url, source_hash=excluded.source_hash,
                     fetched_at=excluded.fetched_at""",
                (
                    song_id,
                    title,
                    artist,
                    json.dumps(sorted(set(aliases)), ensure_ascii=False),
                    source_url,
                    source_hash,
                    datetime.now(UTC).isoformat(),
                ),
            )
            self._database.execute(
                "DELETE FROM lyric_lines WHERE song_id = ?", (song_id,)
            )
            self._database.executemany(
                """INSERT INTO lyric_lines
                   (song_id, line_no, text, reading, translation, translation_source,
                    translation_model, translation_prompt_hash, translation_source_hash)
                   VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?)""",
                [
                    (
                        song_id,
                        index,
                        text,
                        reading,
                        previous_translations.get(index).translation
                        if index in previous_translations
                        else None,
                        previous_translations.get(index).translation_source
                        if index in previous_translations
                        else None,
                        source_hash if index in previous_translations else None,
                    )
                    for index, (text, reading) in enumerate(cleaned)
                ],
            )
        result = self.get(song_id)
        assert result is not None
        return result

    def store_translations(
        self,
        song_id: str,
        translations: dict[int, str],
        *,
        source: str,
        model: str | None = None,
        prompt_hash: str | None = None,
    ) -> None:
        song = self.get(song_id)
        if song is None:
            raise KeyError(song_id)
        with self._database:
            for line_no, text in translations.items():
                if not text.strip() or line_no < 0 or line_no >= len(song.lines):
                    continue
                current = song.lines[line_no]
                if _translation_priority(source) < _translation_priority(
                    current.translation_source
                ):
                    continue
                self._database.execute(
                    """UPDATE lyric_lines SET translation=?, translation_source=?,
                       translation_model=?, translation_prompt_hash=?, translation_source_hash=?
                       WHERE song_id=? AND line_no=?""",
                    (
                        text.strip(),
                        source,
                        model,
                        prompt_hash,
                        song.source_hash,
                        song_id,
                        line_no,
                    ),
                )

    def _load_song(self, row: sqlite3.Row) -> LibrarySong:
        lines = self._database.execute(
            "SELECT * FROM lyric_lines WHERE song_id = ? ORDER BY line_no",
            (row["song_id"],),
        ).fetchall()
        return LibrarySong(
            song_id=row["song_id"],
            title=row["title"],
            artist=row["artist"],
            aliases=tuple(json.loads(row["aliases_json"])),
            source_url=row["source_url"],
            source_hash=row["source_hash"],
            lines=tuple(
                LyricLine(
                    line_no=value["line_no"],
                    text=value["text"],
                    reading=value["reading"],
                    translation=value["translation"],
                    translation_source=value["translation_source"],
                )
                for value in lines
            ),
        )


def _song_id(title: str, artist: str) -> str:
    identity = f"{unicodedata.normalize('NFKC', title).casefold()}\0{unicodedata.normalize('NFKC', artist).casefold()}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _lyrics_hash(lines: list[str]) -> str:
    normalized = "\n".join(
        unicodedata.normalize("NFKC", line).strip() for line in lines
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _translation_priority(source: str | None) -> int:
    return {
        None: 0,
        "machine": 1,
        "llm": 2,
        "external": 3,
        "official": 4,
    }.get(source, 0)
