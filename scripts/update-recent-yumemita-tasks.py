#!/usr/bin/env python3
"""Refresh the Yumemita upload queue from recent public YouTube streams."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_SCRIPT = ROOT / "scripts" / "upload-recent-yumemita.sh"
DEFAULT_UPLOADED = ROOT / "work" / "yumemita-2026-08-10-uploaded.txt"
JST = ZoneInfo("Asia/Tokyo")
MEMBERS_ONLY_MARKERS = ("メン限", "メンバー限定", "members only", "member only")
CHANNELS = (
    ("arale", "@arale_yumemita"),
    ("nonoka", "@nonoka_yumemita"),
    ("ritsu", "@ritsu_yumemita"),
    ("miyako", "@miyako_yumemita"),
    ("yuno", "@yuno_yumemita"),
    ("group", "@BDP_yumemita"),
)


@dataclass(frozen=True)
class Record:
    published: datetime
    channel: str
    video_id: str
    title: str = ""

    def shell_line(self) -> str:
        return f'    "{self.published.astimezone(JST):%Y-%m-%d}|{self.channel}|{self.video_id}"'


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def parse_feed(xml_text: str) -> dict[str, datetime]:
    namespace = {
        "atom": "http://www.w3.org/2005/Atom",
        "yt": "http://www.youtube.com/xml/schemas/2015",
    }
    root = ET.fromstring(xml_text)
    published: dict[str, datetime] = {}
    for entry in root.findall("atom:entry", namespace):
        video_id = entry.findtext("yt:videoId", namespaces=namespace)
        timestamp = entry.findtext("atom:published", namespaces=namespace)
        if video_id and timestamp:
            published[video_id] = parse_timestamp(timestamp)
    return published


def select_recent_streams(
    playlist: dict[str, object],
    feed_dates: dict[str, datetime],
    *,
    channel: str,
    cutoff: datetime,
) -> list[Record]:
    records: list[Record] = []
    for entry in playlist.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        video_id = entry.get("id")
        title = str(entry.get("title") or "")
        if not isinstance(video_id, str) or video_id not in feed_dates:
            continue
        if entry.get("live_status") != "was_live":
            continue
        availability = entry.get("availability")
        if availability not in (None, "public"):
            continue
        if any(marker in title.casefold() for marker in MEMBERS_ONLY_MARKERS):
            continue
        published = feed_dates[video_id]
        if published >= cutoff:
            records.append(Record(published, channel, video_id, title))
    return records


def run_flat_playlist(handle: str, browser: str, playlist_end: int) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--cookies-from-browser",
        browser,
        "--flat-playlist",
        "--playlist-end",
        str(playlist_end),
        "--dump-single-json",
        f"https://www.youtube.com/{handle}/streams",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def fetch_feed(channel_id: str) -> str:
    url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    request = urllib.request.Request(url, headers={"User-Agent": "auto-subtitle/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def read_uploaded_records(batch_script: Path, uploaded_path: Path) -> list[Record]:
    uploaded = {
        line.strip()
        for line in uploaded_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    records: list[Record] = []
    inside = False
    for line in batch_script.read_text(encoding="utf-8").splitlines():
        if line == "RECORDS=(":
            inside = True
            continue
        if inside and line == ")":
            break
        if not inside:
            continue
        value = line.strip().strip('"')
        parts = value.split("|")
        if len(parts) == 3 and parts[2] in uploaded:
            published = datetime.strptime(parts[0], "%Y-%m-%d").replace(tzinfo=JST)
            records.append(Record(published, parts[1], parts[2]))
    missing = uploaded - {record.video_id for record in records}
    if missing:
        raise RuntimeError(
            "Uploaded records are missing from the existing queue: " + ", ".join(sorted(missing))
        )
    return records


def replace_records(batch_script: Path, records: list[Record]) -> None:
    text = batch_script.read_text(encoding="utf-8")
    lines = text.splitlines()
    try:
        start = lines.index("RECORDS=(")
        end = lines.index(")", start + 1)
    except ValueError as error:
        raise RuntimeError(f"Could not locate RECORDS block in {batch_script}") from error

    recent = [record for record in records if record.title]
    if recent:
        first = min(record.published for record in recent).astimezone(JST).date()
        last = max(record.published for record in recent).astimezone(JST).date()
        comment = f"# Successful history plus public archived streams published from {first} through {last}."
    else:
        comment = "# Successfully uploaded history; no recent public archived streams were found."
    if start >= 2 and lines[start - 2].startswith("# Public archived streams"):
        lines[start - 2] = comment
    elif start >= 1 and lines[start - 1].startswith("# Public archived streams"):
        lines[start - 1] = comment

    block = ["RECORDS=(", *(record.shell_line() for record in records), ")"]
    updated = "\n".join([*lines[:start], *block, *lines[end + 1 :]]) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=batch_script.parent, delete=False
    ) as temporary:
        temporary.write(updated)
        temporary_path = Path(temporary.name)
    os.chmod(temporary_path, batch_script.stat().st_mode)
    os.replace(temporary_path, batch_script)


def deduplicate(records: list[Record]) -> list[Record]:
    by_video: dict[str, Record] = {}
    for record in records:
        current = by_video.get(record.video_id)
        if current is None or (not current.title and record.title):
            by_video[record.video_id] = record
    channel_order = {channel: index for index, (channel, _) in enumerate(CHANNELS)}
    return sorted(
        by_video.values(),
        key=lambda record: (
            record.published,
            channel_order.get(record.channel, len(channel_order)),
            record.video_id,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--browser", default="chromium")
    parser.add_argument("--playlist-end", type=int, default=100)
    parser.add_argument("--batch-script", type=Path, default=DEFAULT_BATCH_SCRIPT)
    parser.add_argument("--uploaded", type=Path, default=DEFAULT_UPLOADED)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.days <= 0:
        raise SystemExit("--days must be positive")
    cutoff = datetime.now(UTC) - timedelta(days=args.days)
    historical = read_uploaded_records(args.batch_script, args.uploaded)
    recent: list[Record] = []

    for channel, handle in CHANNELS:
        playlist = run_flat_playlist(handle, args.browser, args.playlist_end)
        channel_id = playlist.get("channel_id") or playlist.get("uploader_id")
        if not isinstance(channel_id, str) or not channel_id.startswith("UC"):
            raise RuntimeError(f"Could not determine channel ID for {handle}")
        feed_dates = parse_feed(fetch_feed(channel_id))
        if not feed_dates:
            raise RuntimeError(f"YouTube feed was empty for {handle}")
        selected = select_recent_streams(
            playlist, feed_dates, channel=channel, cutoff=cutoff
        )
        recent.extend(selected)
        print(f"{channel:7s} {len(selected):2d} recent public archived streams")

    records = deduplicate([*historical, *recent])
    new_ids = {record.video_id for record in recent} - {
        record.video_id for record in historical
    }
    print(
        f"Queue: {len(historical)} uploaded history + {len(new_ids)} new "
        f"= {len(records)} unique videos"
    )
    for record in records:
        if record.video_id in new_ids:
            print(f"  + {record.shell_line().strip()}")

    if args.dry_run:
        print("Dry run: batch script was not changed.")
    else:
        replace_records(args.batch_script, records)
        print(f"Updated {args.batch_script}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
