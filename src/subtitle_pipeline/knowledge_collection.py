from __future__ import annotations

import json
import logging
import re
import ssl
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit

import certifi

from .commands import require_command

logger = logging.getLogger(__name__)


def collect_official_documents(
    urls: list[str],
    *,
    source_type: str = "official_news",
    timeout_seconds: int = 30,
    follow_links: bool = False,
    link_pattern: str | None = None,
    maximum_documents: int = 1000,
    required_terms: tuple[str, ...] = (),
    maximum_depth: int | None = None,
    strict_errors: bool = False,
    known_source_urls: set[str] | None = None,
    conditional_headers: dict[str, dict[str, str]] | None = None,
    response_validators: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, object]]:
    documents: list[dict[str, object]] = []
    queue = [(url, 0) for url in dict.fromkeys(urls)]
    seen: set[str] = set()
    allowed_hosts = {urlsplit(url).netloc for url in urls}
    pattern = re.compile(link_pattern) if link_pattern else None
    while queue and len(documents) < maximum_documents:
        url, depth = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            if conditional_headers is not None and url in conditional_headers:
                downloaded = _download_conditional(
                    url, timeout_seconds, conditional_headers[url]
                )
                if downloaded is None:
                    continue
                body, content_type, validators = downloaded
                if response_validators is not None and validators:
                    response_validators[url] = validators
            else:
                body, content_type = _download(url, timeout_seconds)
        except OSError as exc:
            if strict_errors:
                raise RuntimeError(
                    f"official page collection failed: {url}: {exc}"
                ) from exc
            logger.warning("skipping unavailable official page %s: %s", url, exc)
            continue
        if "xml" in content_type or body.lstrip().startswith(
            ("<?xml", "<rss", "<feed")
        ):
            values = _feed_documents(body, url, source_type)
            documents.extend(values)
            links = [str(value.get("source_url") or "") for value in values]
        else:
            documents.append(_html_document(body, url, source_type))
            links = _html_links(body, url)
        if follow_links and (maximum_depth is None or depth < maximum_depth):
            for link in links:
                if not link or urlsplit(link).netloc not in allowed_hosts:
                    continue
                if pattern is not None and pattern.search(link) is None:
                    continue
                if known_source_urls is not None and link in known_source_urls:
                    continue
                if link not in seen:
                    queue.append((link, depth + 1))
    return [
        document
        for document in documents
        if str(document.get("text") or "").strip()
        and (
            not required_terms
            or any(
                term in f"{document.get('title', '')}\n{document.get('text', '')}"
                for term in required_terms
            )
        )
    ]


def collect_sns_documents(
    urls: list[str],
    *,
    cookies_from_browser: str | None = None,
    known_external_ids: set[str] | None = None,
) -> list[dict[str, object]]:
    if not urls:
        return []
    command = [
        require_command("gallery-dl"),
        "--no-download",
        "--no-skip",
        "--quiet",
        "-P",
        "metadata",
        "-O",
        "mode=jsonl",
        "-O",
        "filename=-",
        "-O",
        "event=post",
    ]
    if cookies_from_browser:
        command.extend(["--cookies-from-browser", cookies_from_browser])
    command.extend(urls)
    if known_external_ids is None:
        try:
            output_lines = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            ).stdout.splitlines()
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip()[-500:]
            raise RuntimeError(
                f"gallery-dl metadata collection failed ({exc.returncode}): {detail}"
            ) from exc
    else:
        output_lines = _incremental_sns_listing(command, known_external_ids)
    documents: dict[tuple[str, str], dict[str, object]] = {}
    for line in output_lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        document = normalize_sns_metadata(value)
        if document is None:
            continue
        key = (str(document["source_type"]), str(document["external_id"]))
        documents[key] = document
    return list(documents.values())


def _incremental_sns_listing(
    command: list[str], known_external_ids: set[str]
) -> list[str]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    lines: list[str] = []
    reached_known = False
    for line in process.stdout:
        lines.append(line)
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        external_id = _first(value, "tweet_id", "post_id", "shortcode", "id")
        if external_id in known_external_ids:
            reached_known = True
            process.terminate()
            break
    _stdout, stderr = process.communicate()
    if not reached_known and process.returncode:
        raise RuntimeError(
            f"incremental SNS collection failed ({process.returncode}): "
            f"{(stderr or '').strip()[-500:]}"
        )
    return lines


def normalize_sns_metadata(value: dict[str, object]) -> dict[str, object] | None:
    category = str(value.get("category") or value.get("extractor") or "").casefold()
    if "twitter" in category or "x.com" in str(value.get("post_url") or ""):
        source_type = "x_post"
        external_id = _first(value, "tweet_id", "id", "post_id")
        text = _first(value, "content", "text", "description")
        author = _nested_first(value, ("author", "name"), ("author", "nick"))
        username = _nested_first(value, ("author", "username"), ("user", "name"))
        source_url = str(value.get("post_url") or "")
        if not source_url and external_id:
            source_url = f"https://x.com/{username or 'i'}/status/{external_id}"
    elif "instagram" in category:
        source_type = "instagram_post"
        external_id = _first(value, "post_id", "shortcode", "id")
        text = _first(value, "description", "caption", "content", "text")
        author = _nested_first(value, ("user", "full_name"), ("owner", "full_name"))
        username = _first(value, "username", "owner_username") or _nested_first(
            value, ("user", "username"), ("owner", "username")
        )
        shortcode = _first(value, "shortcode")
        source_url = str(value.get("post_url") or "")
        if not source_url and shortcode:
            source_url = f"https://www.instagram.com/p/{shortcode}/"
    else:
        return None
    if not external_id or not text.strip():
        return None
    reply_id = _first(value, "reply_id", "in_reply_to_status_id")
    quote_id = _first(value, "quote_id", "quoted_status_id")
    repost_id = _first(value, "retweet_id", "repost_id", "retweeted_status_id")
    relation = (
        "repost"
        if repost_id
        else "quote"
        if quote_id
        else "reply"
        if reply_id and reply_id != "0"
        else "original"
    )
    related_post_id = repost_id or quote_id or reply_id
    return {
        "source_type": source_type,
        "external_id": external_id,
        "source_url": source_url or None,
        "title": f"{author or username or source_type}: {text[:80]}",
        "text": text,
        "author": author or username or None,
        "published_at": _date_value(value),
        "language": _first(value, "lang", "language") or None,
        "reliability": 0.8,
        "metadata": {
            "username": username or None,
            "reply_id": reply_id or None,
            "conversation_id": value.get("conversation_id"),
            "post_relation": relation,
            "related_post_id": related_post_id or None,
        },
    }


def write_documents_jsonl(documents: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in documents),
        encoding="utf-8",
    )


class _ArticleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.meta: dict[str, str] = {}
        self.canonical = ""
        self.blocks: list[str] = []
        self.json_ld: list[str] = []
        self.links: list[str] = []
        self._tag = ""
        self._skip_depth = 0
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.casefold(): value or "" for key, value in attrs}
        if tag == "a" and values.get("href"):
            self.links.append(values["href"])
        if tag in {"script", "style", "nav", "footer", "noscript", "svg"}:
            self._skip_depth += 1
        if tag == "meta":
            key = values.get("property") or values.get("name")
            if key and values.get("content"):
                self.meta[key.casefold()] = values["content"].strip()
        elif tag == "link" and "canonical" in values.get("rel", "").casefold():
            self.canonical = values.get("href", "")
        if self._skip_depth == 0 and tag in {
            "title",
            "h1",
            "h2",
            "h3",
            "p",
            "li",
            "time",
        }:
            self._tag = tag
            self._buffer = []
        elif tag == "script" and values.get("type") == "application/ld+json":
            self._tag = "json-ld"
            self._buffer = []

    def handle_endtag(self, tag: str) -> None:
        if self._tag == "json-ld" and tag == "script":
            self.json_ld.append("".join(self._buffer))
            self._tag = ""
            self._buffer = []
        elif self._tag == tag:
            text = " ".join("".join(self._buffer).split())
            if text:
                if tag == "title":
                    self.title = text
                else:
                    self.blocks.append(text)
            self._tag = ""
            self._buffer = []
        if (
            tag in {"script", "style", "nav", "footer", "noscript", "svg"}
            and self._skip_depth
        ):
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._tag and (self._skip_depth == 0 or self._tag == "json-ld"):
            self._buffer.append(data)


def _download(url: str, timeout_seconds: int) -> tuple[str, str]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "auto-subtitle-knowledge/1.0"},
    )
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(
        request, timeout=timeout_seconds, context=context
    ) as response:
        content_type = response.headers.get_content_type()
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace"), content_type


def _download_conditional(
    url: str, timeout_seconds: int, validators: dict[str, str]
) -> tuple[str, str, dict[str, str]] | None:
    headers = {"User-Agent": "auto-subtitle-knowledge/1.0"}
    if validators.get("etag"):
        headers["If-None-Match"] = validators["etag"]
    if validators.get("last_modified"):
        headers["If-Modified-Since"] = validators["last_modified"]
    request = urllib.request.Request(url, headers=headers)
    context = ssl.create_default_context(cafile=certifi.where())
    try:
        response = urllib.request.urlopen(
            request, timeout=timeout_seconds, context=context
        )
    except HTTPError as exc:
        if exc.code == 304:
            return None
        raise
    with response:
        content_type = response.headers.get_content_type()
        charset = response.headers.get_content_charset() or "utf-8"
        return (
            response.read().decode(charset, errors="replace"),
            content_type,
            {
                key: value
                for key, value in {
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                }.items()
                if value
            },
        )


def _html_document(body: str, url: str, source_type: str) -> dict[str, object]:
    parser = _ArticleParser()
    parser.feed(body)
    structured = _json_ld_metadata(parser.json_ld)
    title = (
        str(structured.get("headline") or structured.get("name") or "").strip()
        or parser.meta.get("og:title", "")
        or parser.title
        or url
    )
    description = parser.meta.get("og:description") or parser.meta.get(
        "description", ""
    )
    blocks = list(dict.fromkeys([description, *parser.blocks]))
    text = "\n".join(value for value in blocks if value)
    canonical = urljoin(url, parser.canonical) if parser.canonical else url
    return {
        "source_type": source_type,
        "external_id": canonical,
        "source_url": canonical,
        "title": title,
        "text": text,
        "author": _structured_author(structured),
        "published_at": structured.get("datePublished")
        or parser.meta.get("article:published_time"),
        "language": structured.get("inLanguage"),
        "reliability": 0.95,
        "metadata": {"collector": "official_html"},
    }


def _html_links(body: str, base_url: str) -> list[str]:
    parser = _ArticleParser()
    parser.feed(body)
    return list(dict.fromkeys(urljoin(base_url, link) for link in parser.links))


def _feed_documents(
    body: str, base_url: str, source_type: str
) -> list[dict[str, object]]:
    root = ET.fromstring(body)
    entries = [*root.findall(".//item"), *root.findall(".//{*}entry")]
    documents: list[dict[str, object]] = []
    for entry in entries:
        link = _xml_text(entry, "link")
        if not link:
            link_node = entry.find("{*}link")
            link = str(link_node.get("href") or "") if link_node is not None else ""
        url = urljoin(base_url, link) if link else base_url
        title = _xml_text(entry, "title") or url
        text = _strip_html(
            _xml_text(entry, "description")
            or _xml_text(entry, "summary")
            or _xml_text(entry, "content")
        )
        documents.append(
            {
                "source_type": source_type,
                "external_id": _xml_text(entry, "guid")
                or _xml_text(entry, "id")
                or url,
                "source_url": url,
                "title": title,
                "text": text,
                "author": _xml_text(entry, "author") or None,
                "published_at": _xml_text(entry, "pubDate")
                or _xml_text(entry, "published")
                or None,
                "reliability": 0.95,
                "metadata": {"collector": "official_feed", "feed_url": base_url},
            }
        )
    return documents


def _json_ld_metadata(values: list[str]) -> dict[str, Any]:
    for value in values:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            continue
        candidates = parsed if isinstance(parsed, list) else [parsed]
        for candidate in candidates:
            if isinstance(candidate, dict) and any(
                key in candidate for key in ("headline", "datePublished", "startDate")
            ):
                return candidate
    return {}


def _structured_author(value: dict[str, Any]) -> str | None:
    author = value.get("author") or value.get("publisher")
    if isinstance(author, dict):
        name = author.get("name")
        return str(name).strip() if name else None
    return str(author).strip() if isinstance(author, str) else None


def _xml_text(entry: ET.Element, name: str) -> str:
    node = entry.find(name)
    if node is None:
        node = entry.find(f"{{*}}{name}")
    return "" if node is None or node.text is None else node.text.strip()


def _strip_html(value: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", value).split())


def _first(value: dict[str, object], *keys: str) -> str:
    for key in keys:
        item = value.get(key)
        if item not in (None, ""):
            return str(item).strip()
    return ""


def _nested_first(value: dict[str, object], *paths: tuple[str, str]) -> str:
    for parent, child in paths:
        nested = value.get(parent)
        if isinstance(nested, dict) and nested.get(child) not in (None, ""):
            return str(nested[child]).strip()
    return ""


def _date_value(value: dict[str, object]) -> str | None:
    raw = value.get("date") or value.get("created_at") or value.get("timestamp")
    return str(raw).strip() if raw not in (None, "") else None
