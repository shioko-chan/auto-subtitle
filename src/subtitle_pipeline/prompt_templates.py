from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import cache
from importlib import resources

_PLACEHOLDER_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")


@dataclass(frozen=True)
class PromptTemplate:
    system: str
    user: str


@cache
def load_prompt_template(name: str) -> PromptTemplate:
    text = (
        resources.files("subtitle_pipeline")
        .joinpath("prompts", name)
        .read_text(encoding="utf-8")
    )
    return PromptTemplate(
        _marked_section(text, "SYSTEM_PROMPT"),
        _marked_section(text, "USER_PROMPT"),
    )


def prompt_system(name: str) -> str:
    return load_prompt_template(name).system


def render_user_prompt(name: str, **values: object) -> str:
    template = load_prompt_template(name).user
    placeholders = set(_PLACEHOLDER_RE.findall(template))
    supplied = set(values)
    missing = sorted(placeholders - supplied)
    unexpected = sorted(supplied - placeholders)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        raise RuntimeError(f"invalid prompt values for {name}: {'; '.join(details)}")
    return _PLACEHOLDER_RE.sub(lambda match: str(values[match.group(1)]), template)


def prompt_templates_digest(*names: str) -> str:
    digest = hashlib.sha256()
    for name in names:
        template = load_prompt_template(name)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(template.system.encode("utf-8"))
        digest.update(b"\0")
        digest.update(template.user.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _marked_section(text: str, label: str) -> str:
    start_marker = f"<!-- {label}_START -->"
    end_marker = f"<!-- {label}_END -->"
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(f"prompt template requires {start_marker} and {end_marker}")
    content_start = start + len(start_marker)
    return text[content_start:end].strip()
