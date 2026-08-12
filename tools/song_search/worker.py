from __future__ import annotations

import ipaddress
import json
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
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443)
    except socket.gaierror:
        return False
    return all(
        not (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
        )
        for address in (ipaddress.ip_address(item[4][0]) for item in addresses)
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
    for backend in ("duckduckgo", "brave", "yahoo"):
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


def fetch(url: str, limit: int) -> dict[str, object]:
    if not public_http_url(url):
        return {"error": "URL is not public"}
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    opener = urllib.request.build_opener(SafeRedirectHandler())
    with opener.open(request, timeout=15) as response:
        raw = response.read(2_000_000)
    import trafilatura

    text = trafilatura.extract(raw, include_comments=False, include_tables=False) or ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    numbered = "\n".join(f"L{i + 1}: {line}" for i, line in enumerate(lines))
    return {"text": numbered[:limit]}


def main() -> None:
    request = json.load(sys.stdin)
    if request.get("action") == "search":
        result = search(str(request.get("query") or ""), int(request.get("limit", 5)))
    elif request.get("action") == "fetch":
        result = fetch(str(request.get("url") or ""), int(request.get("limit", 12000)))
    else:
        result = {"error": "unknown action"}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
