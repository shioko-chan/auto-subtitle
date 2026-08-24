from __future__ import annotations

import html
import ipaddress
import json
import re
import socket
import sys
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlsplit


def public_http_url(url: str) -> bool:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    try:
        literal = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        literal = None
    if literal is not None:
        return public_address(literal)
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443)
    except socket.gaierror:
        return False
    return all(
        public_address(address) or address in ipaddress.ip_network("198.18.0.0/15")
        for address in (ipaddress.ip_address(item[4][0]) for item in addresses)
    )


def public_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
    )


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        if not public_http_url(newurl):
            raise urllib.error.URLError("redirected to a non-public URL")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def search(query: str, limit: int) -> dict[str, object]:
    from ddgs import DDGS

    results: list[dict[str, object]] = []
    errors: list[str] = []
    for backend in ("startpage", "yahoo", "duckduckgo", "brave"):
        try:
            batch = DDGS().text(query, backend=backend, max_results=limit)
        except Exception as exc:
            errors.append(f"search {backend}: {str(exc)[:300]}")
            continue
        for item in batch:
            if isinstance(item, dict) and item not in results:
                results.append(item)
        if results:
            break
    return {"results": results[:limit], "errors": errors}


def fetch_lyrics(url: str) -> dict[str, object]:
    if not public_http_url(url):
        return {"error": "URL is not public"}
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    opener = urllib.request.build_opener(SafeRedirectHandler())
    with opener.open(request, timeout=15) as response:
        raw = response.read(2_000_000)
        charset = response.headers.get_content_charset() or "utf-8"
    try:
        document = raw.decode(charset, errors="replace")
    except LookupError:
        document = raw.decode("utf-8", errors="replace")
    hostname = (urlsplit(url).hostname or "").casefold()
    parsers = []
    if hostname.endswith("utaten.com"):
        parsers.append(_parse_utaten)
    elif hostname.endswith("oricon.co.jp"):
        parsers.append(_parse_oricon)
    elif hostname.endswith("awa.fm"):
        parsers.append(_parse_awa)
    elif hostname.endswith("uta-net.com"):
        parsers.append(_parse_utanet)
    parsers.extend(
        parser
        for parser in (_parse_utaten, _parse_oricon, _parse_awa, _parse_utanet)
        if parser not in parsers
    )
    values = next((value for parser in parsers if (value := parser(document))), None)
    if values is None:
        return {"error": "page has no supported structured canonical lyrics"}
    return values


def _parse_utaten(document: str) -> dict[str, object] | None:
    if "utaten.com" not in document.lower() and 'class="hiragana"' not in document:
        return None
    title = ""
    artist = ""
    for raw in re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        document,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        try:
            value = json.loads(html.unescape(raw))
        except (json.JSONDecodeError, TypeError):
            continue
        objects = value if isinstance(value, list) else [value]
        for item in objects:
            if not isinstance(item, dict):
                continue
            if item.get("@type") == "MusicComposition":
                title = str(item.get("name") or "").strip()
                recording = item.get("recordedAs")
                by_artist = (
                    recording.get("byArtist")
                    if isinstance(recording, dict)
                    else item.get("byArtist")
                )
                if isinstance(by_artist, dict):
                    artist = str(by_artist.get("name") or "").strip()
    match = re.search(
        r'<div[^>]+class=["\'][^"\']*hiragana[^"\']*["\'][^>]*>(.*?)</div>',
        document,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None
    body = re.sub(
        r'<span[^>]+class=["\'][^"\']*rt[^"\']*["\'][^>]*>.*?</span>',
        "",
        match.group(1),
        flags=re.IGNORECASE | re.DOTALL,
    )
    body = re.sub(
        r"<rt\b[^>]*>.*?</rt>",
        "",
        body,
        flags=re.IGNORECASE | re.DOTALL,
    )
    body = re.sub(r"<(?:br|p|li)\b[^>]*>", "\n", body, flags=re.IGNORECASE)
    body = re.sub(r"</(?:p|li)>", "\n", body, flags=re.IGNORECASE)
    body = re.sub(r"<[^>]+>", "", body)
    lines = [html.unescape(line).strip() for line in body.splitlines()]
    lines = [line for line in lines if line]
    if not title or len(lines) < 3:
        return None
    return {"title": title, "artist": artist, "lines": lines, "readings": []}


def _parse_oricon(document: str) -> dict[str, object] | None:
    title, artist = _json_ld_song_identity(document)
    match = re.search(
        r'<div[^>]+class=["\'][^"\']*all-lyrics[^"\']*["\'][^>]*>(.*?)</div>',
        document,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None
    lines = _html_lyric_lines(match.group(1))
    if not title or len(lines) < 3:
        return None
    return {"title": title, "artist": artist, "lines": lines, "readings": []}


def _parse_awa(document: str) -> dict[str, object] | None:
    title_match = re.search(
        r"<h1\b[^>]*>(.*?)</h1>", document, re.IGNORECASE | re.DOTALL
    )
    artist_match = re.search(
        r"<span\b[^>]*>\s*Track by\s*</span>\s*<a\b[^>]*>(.*?)</a>",
        document,
        re.IGNORECASE | re.DOTALL,
    )
    lyrics_match = re.search(
        r"<h2\b[^>]*>\s*歌詞\s*</h2>\s*<p\b[^>]*>(.*?)</p>",
        document,
        re.IGNORECASE | re.DOTALL,
    )
    if title_match is None or lyrics_match is None:
        return None
    title = _plain_html_text(title_match.group(1))
    artist = _plain_html_text(artist_match.group(1)) if artist_match else ""
    lines = _html_lyric_lines(lyrics_match.group(1))
    if not title or len(lines) < 3:
        return None
    return {"title": title, "artist": artist, "lines": lines, "readings": []}


def _parse_utanet(document: str) -> dict[str, object] | None:
    title, artist = _json_ld_song_identity(document)
    match = re.search(
        r'<div[^>]+(?:id|class)=["\'][^"\']*kashi_area[^"\']*["\'][^>]*>(.*?)</div>',
        document,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None or "Enable JavaScript and cookies" in document:
        return None
    lines = _html_lyric_lines(match.group(1))
    if not title or len(lines) < 3:
        return None
    return {"title": title, "artist": artist, "lines": lines, "readings": []}


def _json_ld_song_identity(document: str) -> tuple[str, str]:
    title = ""
    artist = ""
    for raw in re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        document,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        try:
            value = json.loads(html.unescape(raw))
        except (json.JSONDecodeError, TypeError):
            continue
        objects = value if isinstance(value, list) else [value]
        for item in objects:
            if not isinstance(item, dict):
                continue
            item_type = item.get("@type")
            if item_type in {"MusicGroup", "Person"} and not artist:
                artist = str(item.get("name") or "").strip()
                continue
            if item_type not in {"MusicComposition", "MusicRecording"}:
                continue
            title = str(item.get("name") or title).strip()
            by_artist = item.get("byArtist")
            if isinstance(by_artist, list):
                by_artist = next(
                    (
                        candidate
                        for candidate in by_artist
                        if isinstance(candidate, dict)
                    ),
                    None,
                )
            if isinstance(by_artist, dict):
                artist = str(by_artist.get("name") or artist).strip()
    return title, artist


def _plain_html_text(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def _html_lyric_lines(value: str) -> list[str]:
    body = re.sub(r"<(?:br|p|li)\b[^>]*>", "\n", value, flags=re.IGNORECASE)
    body = re.sub(r"</(?:p|li)>", "\n", body, flags=re.IGNORECASE)
    body = re.sub(r"<[^>]+>", "", body)
    return [
        line
        for raw in html.unescape(body).splitlines()
        if (line := re.sub(r"\s+", " ", raw).strip())
    ]


def main() -> None:
    request = json.load(sys.stdin)
    if request.get("action") == "search":
        result = search(str(request.get("query") or ""), int(request.get("limit", 5)))
    elif request.get("action") == "fetch_lyrics":
        result = fetch_lyrics(str(request.get("url") or ""))
    else:
        result = {"error": "unknown action"}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
