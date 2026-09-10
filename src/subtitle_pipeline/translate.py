from __future__ import annotations

import json
import logging
import math
import random
import ssl
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import certifi
from copy import copy

from .config import LLMConfig, SegmentationConfig, TranslationConfig
from .fan_knowledge import KnowledgeHit
from .llm_response import finish_reason as _finish_reason
from .llm_response import parse_json_object as _parse_json_object
from .llm_response import (
    strip_markdown_code_fence,
    structured_request_body,
    structured_response_content,
)
from .local_segmentation import build_speaker_tracks
from .local_translation import LocalJapaneseTranslator
from .llm_stream import read_chat_stream, read_responses_stream
from .repetition import RepetitionLoopError
from .prompt_budget import count_llama_prompt_tokens, estimate_prompt_tokens, validate_request_budget
from .prompt_templates import render_user_prompt
from .reference_context import (
    compact_lyrics_reference_context,
    compact_reference_context,
)
from .subtitles import Cue
from .cache import CacheStore, config_snapshot, restore_config
from .telemetry import stage_metrics


class TranslationError(RuntimeError):
    pass


class LLMHTTPError(TranslationError):
    def __init__(
        self,
        status: int,
        detail: str,
        retry_after_seconds: float | None = None,
    ):
        self.status = status
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"LLM API returned HTTP {status}: {detail}")


class LocalLLMError(TranslationError):
    pass


def is_llm_quota_exhausted(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, LLMHTTPError) and current.status == 402:
            return True
        current = current.__cause__ or current.__context__
    return False


_HONORIFIC_TRANSLATION_RULES = (
    "Apply these Japanese-honorific rules when translating into Chinese. さん may be "
    "rendered as 桑 in fan dialogue or a stable affectionate nickname; otherwise usually "
    "omit it or translate it as 先生, 女士, or 老师 according to the person's role. "
    "Translate ちゃん as 酱, 小 followed by the name, or another natural affectionate "
    "form. Usually omit くん; use 君 or 同学 only when context requires it. "
    "Translate さま or 様 as 大人, 阁下, 先生, or another status-appropriate form. "
    "Translate 先生 as 老师, 医生, or 先生 according to the person's actual role. An explicit "
    "REFERENCE mapping for a complete name-plus-honorific form overrides these defaults. "
)


class OpenAICompatibleTranslator:
    def __init__(
        self,
        config: LLMConfig,
        translation: TranslationConfig,
        api_key: str,
        *,
        audit_path: Path | None = None,
    ):
        self.config = config
        self.translation = translation
        self.api_key = api_key
        self.before_request = None
        self.audit_path = audit_path
        self._audit_lock = threading.Lock()
        self._audit_request_ids: dict[int, str] = {}
        self.ssl_context = _create_ssl_context()
        self.local_translator = LocalJapaneseTranslator(
            translation.local_model,
            translation.local_device,
        )

    def request(self, body: dict[str, object]) -> dict[str, object]:
        """Send an auxiliary agent request using the configured model settings."""
        payload = dict(body)
        payload.setdefault("model", self.config.model)
        payload.setdefault("max_tokens", self.translation.max_tokens)
        if self.config.thinking is not None:
            payload.setdefault("thinking", {"type": self.config.thinking})
        return self._request(payload)

    def _stage_sender(self, cache_path: Path | None, name: str):
        stage = CacheStore(cache_path).existing(name) if cache_path is not None else None
        if stage is None:
            return self
        snapshot = stage.plan.get("llm") or stage.remember(
            "request_config", lambda: config_snapshot(self.config)
        )
        sender = copy(self)
        sender.config = restore_config(self.config, snapshot)
        return sender

    def stage_request(self, cache_path: Path | None, name: str):
        return lambda body: self._stage_sender(cache_path, name)._request(body)

    def stage_budget_validator(self, cache_path: Path | None, name: str):
        return lambda body: self._stage_sender(cache_path, name).validate_request(body)

    def validate_request(self, body: dict[str, object]) -> None:
        if not self.config.local_server_enabled:
            return
        if self.before_request is not None:
            self.before_request(self.config)
        if callable(self.api_key):
            self.api_key = self.api_key()
        validate_request_budget(
            body, context_size=self.config.local_server_context_size,
            count_tokens=lambda value: count_llama_prompt_tokens(value, self._budget_post),
        )

    def _budget_post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        base = self.config.base_url.rstrip('/').removesuffix('/v1')
        request = urllib.request.Request(
            base + path, data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
            headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'},
            method='POST',
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout_seconds, context=self.ssl_context) as response:
            return json.loads(response.read().decode('utf-8'))

    def _stage_executor(self, cache_path: Path | None, name: str):
        stage = CacheStore(cache_path).existing(name) if cache_path is not None else None
        if stage is None:
            return self
        execution = stage.remember("execution", lambda: {
            "llm": stage.plan.get("llm", config_snapshot(self.config)),
            "translation": stage.plan.get("translation", config_snapshot(self.translation)),
        })
        sender = copy(self)
        sender.config = restore_config(self.config, execution["llm"])
        sender.translation = restore_config(self.translation, execution["translation"])
        if (sender.translation.local_model, sender.translation.local_device) != (self.translation.local_model, self.translation.local_device):
            sender.local_translator = LocalJapaneseTranslator(sender.translation.local_model, sender.translation.local_device)
        return sender

    def segment_and_translate_cues(
        self, cues: list[Cue], config: SegmentationConfig, *,
        max_line_units: float, translation_context: dict[str, object] | None = None,
        cache_path: Path | None = None, audit_path: Path | None = None,
        local_audit_path: Path | None = None,
        retrieve_knowledge: Callable[[list[Cue], str], list[KnowledgeHit]] | None = None,
        retrieve_chat: Callable[[list[Cue]], str] | None = None,
    ) -> tuple[list[Cue], list[Cue]]:
        from .staged_translation import run_joint_translation

        self = self._stage_executor(cache_path, "translation")
        tracks = []
        if cache_path is None or CacheStore(cache_path).existing("translation") is None:
            if not cues:
                return [], []
            with stage_metrics("subtitle.local_segmentation"):
                tracks, _ = build_speaker_tracks(
                    cues, config, audit_path=local_audit_path,
                    source_maximum_units=max_line_units * 1.25 / 2,
                )
        return run_joint_translation(
            tracks=tracks, source_cues=cues, segmentation=config,
            translation=self.translation, llm=self.config,
            request=self.stage_request(cache_path, "translation"),
            translation_context=translation_context or {}, cache_path=cache_path,
            maximum_units=max_line_units, honorific_rules=_HONORIFIC_TRANSLATION_RULES,
            parse_content=_parse_cue_records, finish_reason=_finish_reason,
            retry_delay=_transient_retry_delay, is_nontransient=_is_nontransient_http_error,
            log_invalid_response=self._log_invalid_response,
            local_translate=self.local_translator.translate,
            retrieve_knowledge=retrieve_knowledge, retrieve_chat=retrieve_chat,
            audit_path=audit_path,
        )

    def translate_lyrics(
        self,
        title: str,
        artist: str,
        lines: list[str],
        *,
        translation_context: dict[str, object] | None = None,
        cache_path: Path | None = None,
    ) -> tuple[dict[int, str], str]:
        """Translate canonical lyric lines without changing their correspondence."""
        self = self._stage_executor(cache_path, "lyrics_translation")
        if not lines:
            return {}, "llm"
        prompt = render_user_prompt(
            "lyrics-translate.md",
            SONG_TITLE=title,
            ARTIST=artist or "(unknown)",
            REFERENCE_TEXT=json.dumps(
                compact_lyrics_reference_context(translation_context or {}),
                ensure_ascii=False,
            ),
            LYRICS_TEXT="\n".join(
                f"<{index}>{line}" for index, line in enumerate(lines)
            ),
        )
        body = structured_request_body(
            model=self.config.model,
            prompt_name="lyrics-translate.md",
            prompt=prompt,
            max_tokens=self.translation.max_tokens,
            temperature=0.2,

            thinking=self.config.thinking,
        )
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            response: object = None
            content: object = None
            try:
                response = self._request(body)
                content = structured_response_content(
                    response, finish_reason=_finish_reason
                )
                parsed = _parse_json_object(content)
                values = parsed.get("lines")
                if not isinstance(values, list):
                    raise ValueError("lyrics response requires a lines array")
                translated: dict[int, str] = {}
                for value in values:
                    if not isinstance(value, dict):
                        raise ValueError("lyrics response line is not an object")
                    line_id = value.get("line_id")
                    text = value.get("text")
                    if isinstance(line_id, str) and line_id.isdigit():
                        line_id = int(line_id)
                    if not isinstance(line_id, int) or not isinstance(text, str):
                        raise ValueError("invalid lyrics line_id or text")
                    translated[line_id] = text.strip()
                if set(translated) != set(range(len(lines))) or any(
                    not value for value in translated.values()
                ):
                    raise ValueError("lyrics translation did not cover every line")
                return translated, "llm"
            except Exception as exc:
                last_error = exc
                self._log_invalid_response(
                    "lyrics_translation", exc, content, body, response
                )
                if _is_nontransient_http_error(exc):
                    raise
                delay = _transient_retry_delay(exc, attempt)
                if delay is not None:
                    time.sleep(delay)
        raise RuntimeError("lyrics translation exhausted LLM retries") from last_error

    def translate_metadata(
        self,
        title: str,
        description: str,
        *,
        youtube_context: dict[str, object] | None = None,
        subtitle_evidence: str = "",
        ip_aliases: dict[str, object] | None = None,
        bilibili_tag_catalog: dict[str, object] | None = None,
        translation_context: dict[str, object] | None = None,
        cache_path: Path | None = None,
    ) -> tuple[str, str, str, list[str]]:
        self = self._stage_executor(cache_path, "metadata")
        source = {
            "title": title,
            "description": description[
                : self.translation.metadata_description_max_chars
            ],
            "youtube_context": youtube_context or {},
            "subtitle_evidence": subtitle_evidence[
                : self.translation.metadata_subtitle_max_chars
            ],
            "known_ip_aliases": ip_aliases or {},
            "bilibili_tag_catalog": bilibili_tag_catalog or {},
            "translation_context": compact_reference_context(translation_context or {}),
        }
        prompt = render_user_prompt(
            "metadata-translate.md",
            TARGET_LANGUAGE=self.translation.target_language,
            TAG_COUNT=self.translation.metadata_tag_count,
            SOURCE_TEXT=json.dumps(source, ensure_ascii=False),
        )
        body = structured_request_body(
            model=self.config.model,
            prompt_name="metadata-translate.md",
            prompt=prompt,
            max_tokens=self.translation.max_tokens,
            temperature=0.2,

            thinking=self.config.thinking,
        )

        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            content: object = None
            response: object = None
            try:
                response = self._request(body)
                content = structured_response_content(
                    response, finish_reason=_finish_reason
                )
                finish_reason = _finish_reason(response)
                logging.info(
                    "metadata response attempt %d finish_reason=%s",
                    attempt,
                    finish_reason or "unknown",
                )
                parsed = _parse_json_object(content)
                translated_title = parsed.get("title")
                translated_description = parsed.get("description")
                content_summary = parsed.get("content_summary")
                translated_tags = parsed.get("tags")
                if (
                    not isinstance(translated_title, str)
                    or not translated_title.strip()
                ):
                    raise ValueError("translated metadata title must be non-empty text")
                if not isinstance(translated_description, str):
                    raise ValueError("translated metadata description must be text")
                if not isinstance(content_summary, str) or not content_summary.strip():
                    raise ValueError("metadata content_summary must be non-empty text")
                if not isinstance(translated_tags, list):
                    raise ValueError("translated metadata tags must be a list")
                tags = _clean_tags(translated_tags, self.translation.metadata_tag_count)
                if not tags:
                    raise ValueError("translated metadata tags must not be empty")
                return (
                    translated_title.strip(),
                    translated_description.strip(),
                    content_summary.strip(),
                    tags,
                )
            except (
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                urllib.error.URLError,
                TimeoutError,
                TranslationError,
            ) as exc:
                last_error = exc
                self._log_invalid_response("metadata", exc, content, body, response)
                if _is_nontransient_http_error(exc):
                    raise
                if attempt < self.config.max_retries:
                    delay = _transient_retry_delay(exc, attempt)
                    if delay is not None:
                        logging.warning(
                            "metadata translation attempt %d hit a transient "
                            "failure (%s); retrying in %ss",
                            attempt,
                            exc,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        logging.warning(
                            "metadata translation attempt %d failed validation "
                            "(%s); retrying immediately",
                            attempt,
                            exc,
                        )
        if isinstance(last_error, LLMHTTPError):
            raise last_error
        raise TranslationError(
            "metadata translation failed after "
            f"{self.config.max_retries} attempts: {last_error}"
        )

    def _log_invalid_response(
        self,
        kind: str,
        error: Exception,
        content: object,
        request_body: dict[str, object] | None = None,
        response: object = None,
    ) -> None:
        _log_invalid_response(kind, error, content)
        if self.audit_path is None:
            return
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": "invalid_llm_response",
            "request_id": (
                self._audit_request_ids.get(id(request_body))
                if request_body is not None
                else None
            ),
            "kind": kind,
            "error_type": type(error).__name__,
            "error": str(error),
            "request": request_body,
            "response": response,
            "response_content": content,
        }
        for name in (
            "patch_start",
            "patch_end",
            "expected_start",
            "expected_end",
            "expected_next",
            "received_ranges",
        ):
            if hasattr(error, name):
                entry[name] = getattr(error, name)
        with self._audit_lock:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, default=repr) + "\n")

    def _request(self, body: dict[str, object]) -> dict[str, object]:
        self.validate_request(body)
        if self.before_request is not None:
            self.before_request(self.config)
        if callable(self.api_key):
            self.api_key = self.api_key()
        request_id = uuid.uuid4().hex
        self._audit_request_ids[id(body)] = request_id
        url, request_body = _prepare_api_request(self.config, body)
        request_body = {**request_body, "stream": True}
        if self.config.api_style == "chat_completions":
            request_body["stream_options"] = {"include_usage": True}
        read_stream = read_chat_stream if self.config.api_style == "chat_completions" else read_responses_stream
        request = urllib.request.Request(
            url,
            data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        options = {"context": self.ssl_context}
        if not self.config.local_server_enabled:
            options["timeout"] = self.config.timeout_seconds
        try:
            with urllib.request.urlopen(request, **options) as response:
                payload = read_stream(response)
        except RepetitionLoopError as exc:
            self._log_invalid_response("stream_repetition", exc, None, body)
            raise
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            if self.config.local_server_enabled:
                raise LocalLLMError(f"local LLM request failed: {detail}") from exc
            retry_after = _parse_retry_after(exc.headers.get("Retry-After")) if exc.headers else None
            raise LLMHTTPError(exc.code, detail, retry_after) from exc
        except urllib.error.URLError as exc:
            if self.config.local_server_enabled:
                raise LocalLLMError(f"local LLM request failed: {exc.reason}") from exc
            raise
        normalized = _normalize_api_response(self.config, payload)
        normalized["_audit_request_id"] = request_id
        _log_response_usage(normalized)
        self._log_request_context(body, normalized)
        self._log_successful_response(request_id, body, normalized)
        return normalized

    def _log_successful_response(
        self,
        request_id: str,
        request_body: dict[str, object],
        response: dict[str, object],
    ) -> None:
        if self.audit_path is None:
            return
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": "llm_response",
            "request_id": request_id,
            "request": request_body,
            "response": response,
        }
        with self._audit_lock:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, default=repr) + "\n")

    def _log_request_context(
        self, body: dict[str, object], response: dict[str, object]
    ) -> None:
        if self.audit_path is None:
            return
        messages = body.get("messages")
        contents = (
            [
                str(message.get("content") or "")
                for message in messages
                if isinstance(message, dict) and isinstance(message.get("content"), str)
            ]
            if isinstance(messages, list)
            else []
        )
        prompt = "\n\n".join(contents)
        usage = response.get("usage")
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        completion_tokens = (
            usage.get("completion_tokens") if isinstance(usage, dict) else None
        )
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": "llm_request_context",
            "model": body.get("model", self.config.model),
            "input_characters": len(prompt),
            "estimated_input_tokens": estimate_prompt_tokens(prompt),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "max_output_tokens": body.get("max_tokens"),
            "context_capacity": (
                self.config.local_server_context_size
                if self.config.local_server_enabled
                else None
            ),
            "sections": _prompt_section_sizes(prompt),
        }
        with self._audit_lock:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _prepare_api_request(
    config: LLMConfig, body: dict[str, object]
) -> tuple[str, dict[str, object]]:
    base_url = config.base_url.rstrip("/")
    if config.api_style == "chat_completions":
        return f"{base_url}/chat/completions", body
    return f"{base_url}/responses", _responses_request_body(config, body)


def _responses_request_body(
    config: LLMConfig, body: dict[str, object]
) -> dict[str, object]:
    converted: dict[str, object] = {
        "model": body.get("model", config.model),
        "max_output_tokens": body["max_tokens"],
        "store": False,
    }
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise TypeError("LLM request messages must be a list")
    instructions: list[str] = []
    inputs: list[dict[str, object]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError("LLM request message must be an object")
        role = message.get("role")
        content = message.get("content")
        if role in {"system", "developer"}:
            if isinstance(content, str) and content:
                instructions.append(content)
            continue
        if role == "tool":
            inputs.append(
                {
                    "type": "function_call_output",
                    "call_id": str(message.get("tool_call_id") or ""),
                    "output": str(content or ""),
                }
            )
            continue
        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            if isinstance(content, str) and content:
                inputs.append({"role": "assistant", "content": content})
            for call in message["tool_calls"]:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if not isinstance(function, dict):
                    continue
                inputs.append(
                    {
                        "type": "function_call",
                        "call_id": str(call.get("id") or ""),
                        "name": str(function.get("name") or ""),
                        "arguments": str(function.get("arguments") or "{}"),
                    }
                )
            continue
        if role not in {"user", "assistant"}:
            raise ValueError(f"unsupported Responses API message role: {role!r}")
        inputs.append({"role": role, "content": str(content or "")})
    if instructions:
        converted["instructions"] = "\n\n".join(instructions)
    converted["input"] = inputs

    response_format = body.get("response_format")
    if isinstance(response_format, dict):
        if response_format.get("type") == "json_schema":
            converted["text"] = {"format": {"type": "json_schema", **response_format["json_schema"]}}
        else:
            converted["text"] = {"format": response_format}
    tools = body.get("tools")
    if isinstance(tools, list):
        converted["tools"] = [_responses_tool_definition(tool) for tool in tools]
    if "tool_choice" in body:
        converted["tool_choice"] = body["tool_choice"]
    if config.reasoning_effort is not None:
        converted["reasoning"] = {"effort": config.reasoning_effort}
    return converted


def _responses_tool_definition(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or value.get("type") != "function":
        raise ValueError("Responses API supports only function tools in this pipeline")
    function = value.get("function")
    if not isinstance(function, dict):
        raise ValueError("function tool definition is malformed")
    return {
        "type": "function",
        "name": function.get("name"),
        "description": function.get("description", ""),
        "parameters": function.get("parameters", {}),
        "strict": function.get("strict", False),
    }


def _normalize_api_response(config: LLMConfig, payload: object) -> dict[str, object]:
    if config.api_style == "chat_completions":
        if not isinstance(payload, dict):
            raise TypeError("LLM response must be an object")
        return payload
    if not isinstance(payload, dict):
        raise TypeError("Responses API response must be an object")
    output = payload.get("output")
    if not isinstance(output, list):
        raise ValueError("Responses API response has no output array")
    text_parts: list[str] = []
    tool_calls: list[dict[str, object]] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            tool_calls.append(
                {
                    "id": str(item.get("call_id") or item.get("id") or ""),
                    "type": "function",
                    "function": {
                        "name": str(item.get("name") or ""),
                        "arguments": str(item.get("arguments") or "{}"),
                    },
                }
            )
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "output_text":
                text_parts.append(str(part.get("text") or ""))
    status = payload.get("status")
    incomplete = payload.get("incomplete_details")
    reason = incomplete.get("reason") if isinstance(incomplete, dict) else None
    if status == "completed":
        finish_reason = "stop"
    elif status == "incomplete" and reason == "max_output_tokens":
        finish_reason = "length"
    else:
        finish_reason = str(reason or status or "unknown")
    message: dict[str, object] = {"content": "".join(text_parts)}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": payload.get("id"),
        "choices": [{"finish_reason": finish_reason, "message": message}],
        "usage": _normalize_responses_usage(payload.get("usage")),
    }


def _normalize_responses_usage(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    prompt = value.get("input_tokens")
    completion = value.get("output_tokens")
    details = value.get("input_tokens_details")
    normalized: dict[str, object] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": value.get("total_tokens"),
    }
    if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
        normalized["prompt_tokens_details"] = {
            "cached_tokens": details["cached_tokens"]
        }
    return normalized


def _transient_retry_delay(exc: Exception, attempt: int) -> float | None:
    if isinstance(exc, LLMHTTPError):
        if exc.status == 429:
            if exc.retry_after_seconds is not None:
                return _retry_after_delay(exc.retry_after_seconds)
            return _jittered_exponential_backoff(attempt)
        if 500 <= exc.status < 600:
            return _jittered_exponential_backoff(attempt)
        return None
    if isinstance(exc, (urllib.error.URLError, TimeoutError)):
        return _jittered_exponential_backoff(attempt)
    return None


def _is_transient_failure(exc: Exception) -> bool:
    return (
        isinstance(exc, (urllib.error.URLError, TimeoutError))
        or isinstance(exc, LLMHTTPError)
        and (exc.status == 429 or 500 <= exc.status < 600)
    )


def _jittered_exponential_backoff(attempt: int) -> float:
    base_delay = float(2 ** (attempt - 1))
    return random.uniform(base_delay * 0.75, base_delay * 1.25)


def _retry_after_delay(server_delay: float) -> float:
    jitter_ceiling = max(0.25, min(2.0, server_delay * 0.25))
    return server_delay + random.uniform(0.0, jitter_ceiling)


def _parse_retry_after(
    value: str | None,
    *,
    now: datetime | None = None,
) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    try:
        return max(0.0, float(stripped))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(stripped)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    current = now or datetime.now(UTC)
    return float(max(0, math.ceil((retry_at - current).total_seconds())))


def _is_nontransient_http_error(exc: Exception) -> bool:
    return (
        isinstance(exc, LocalLLMError)
        or isinstance(exc, LLMHTTPError)
        and not (exc.status == 429 or 500 <= exc.status < 600)
    )


def _parse_cue_records(content: object) -> list[object]:
    value = strip_markdown_code_fence(content)
    if not value:
        raise ValueError("empty response")

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = _parse_json_sequence(value)
    if isinstance(parsed, dict):
        if "cues" in parsed:
            cues = parsed["cues"]
            if not isinstance(cues, list):
                raise ValueError('cue response field "cues" must be an array')
            return cues
        if {"start_id", "end_id", "text"}.issubset(parsed):
            return [parsed]
        raise ValueError('cue response object requires a "cues" array')
    if isinstance(parsed, list):
        return parsed
    raise ValueError("cue response must be an object or array")


def _parse_json_sequence(value: str) -> list[object]:
    decoder = json.JSONDecoder()
    records: list[object] = []
    position = 0
    while position < len(value):
        while position < len(value) and (
            value[position].isspace() or value[position] == ","
        ):
            position += 1
        if position >= len(value):
            break
        try:
            record, position = decoder.raw_decode(value, position)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid cue JSON at character {exc.pos}: {exc.msg}"
            ) from exc
        records.append(record)
    if not records:
        raise ValueError("empty response")
    return records


def _log_invalid_response(kind: str, error: Exception, content: object) -> None:
    if isinstance(content, str):
        tail = content[-500:]
    else:
        tail = repr(content)
    logging.warning("invalid %s response (%s); response_tail=%r", kind, error, tail)


def _log_response_usage(response: object) -> None:
    if not isinstance(response, dict):
        return
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return

    def integer(name: str) -> int | None:
        value = usage.get(name)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    prompt = integer("prompt_tokens")
    completion = integer("completion_tokens")
    total = integer("total_tokens")
    cache_hit = integer("prompt_cache_hit_tokens")
    cache_miss = integer("prompt_cache_miss_tokens")
    if cache_hit is None:
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached = details.get("cached_tokens")
            if isinstance(cached, int) and not isinstance(cached, bool):
                cache_hit = cached
    if cache_miss is None and prompt is not None and cache_hit is not None:
        cache_miss = max(0, prompt - cache_hit)
    cache_rate = None
    if cache_hit is not None and cache_miss is not None and cache_hit + cache_miss > 0:
        cache_rate = cache_hit / (cache_hit + cache_miss)
    logging.info(
        "LLM usage prompt=%s cache_hit=%s cache_miss=%s cache_hit_rate=%s "
        "completion=%s total=%s",
        prompt if prompt is not None else "unknown",
        cache_hit if cache_hit is not None else "unknown",
        cache_miss if cache_miss is not None else "unknown",
        f"{cache_rate:.1%}" if cache_rate is not None else "unknown",
        completion if completion is not None else "unknown",
        total if total is not None else "unknown",
    )


def _prompt_section_sizes(prompt: str) -> dict[str, dict[str, int]]:
    marker_names = {
        "ENTITY_REFERENCE:": "entity_reference",
        "REFERENCE:": "reference",
        "CURRENT_VIDEO_CHAT:": "current_video_chat",
        "DIALOGUE_CONTEXT:": "dialogue_context",
        "SOURCE:": "source",
        "TARGET:": "target",
    }
    sections: dict[str, list[str]] = {}
    fixed: list[str] = []
    current: str | None = None
    for line in prompt.splitlines(keepends=True):
        marker = marker_names.get(line.strip())
        if marker is not None:
            current = marker
            sections.setdefault(marker, [])
        elif current is None:
            fixed.append(line)
        else:
            sections[current].append(line)
    values = {
        name: {
            "characters": len(text),
            "estimated_tokens": estimate_prompt_tokens(text),
        }
        for name, lines in sections.items()
        if (text := "".join(lines).strip())
    }
    fixed_text = "".join(fixed).strip()
    values["fixed_prompt"] = {
        "characters": len(fixed_text),
        "estimated_tokens": estimate_prompt_tokens(fixed_text),
    }
    return values


def _create_ssl_context() -> ssl.SSLContext:
    """Trust platform/user CAs and supplement them with certifi's CA bundle."""
    context = ssl.create_default_context()
    context.load_verify_locations(cafile=certifi.where())
    return context


def _clean_tags(values: list[object], limit: int) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        for candidate in value.split(","):
            tag = candidate.strip().lstrip("#").strip()
            if not tag:
                continue
            tag = tag[:20]
            key = tag.casefold()
            if key not in seen:
                seen.add(key)
                tags.append(tag)
            if len(tags) >= limit:
                return tags
    return tags
