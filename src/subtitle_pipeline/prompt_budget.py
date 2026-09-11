from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from typing import TypeVar


class PromptBudgetExceeded(RuntimeError):
    pass


def estimate_prompt_tokens(text: str) -> int:
    cjk = sum(
        "\u3040" <= character <= "\u30ff"
        or "\u3400" <= character <= "\u9fff"
        for character in text
    )
    return cjk + math.ceil((len(text) - cjk) / 4)


Item = TypeVar("Item")


def validate_request_budget(
    body: dict[str, object], *, context_size: int,
    count_tokens: Callable[[dict[str, object]], int],
) -> None:
    """Validate the complete rendered request, including its output allowance."""
    reserve = int(body['max_tokens'])
    if reserve < 1 or context_size <= reserve:
        raise PromptBudgetExceeded(f"output_reserve={reserve} leaves no input capacity in context={context_size}")
    tokens = count_tokens(body)
    if tokens + reserve > context_size:
        raise PromptBudgetExceeded(
            f"prompt tokens={tokens} + output_reserve={reserve} exceed context={context_size}"
        )



def estimate_request_tokens(body: dict[str, object]) -> int:
    """Estimate all message contents and their serialized structure."""
    return estimate_prompt_tokens(json.dumps(body["messages"], ensure_ascii=False))


def request_budget_validator(
    context_size: int, *,
    count_tokens: Callable[[dict[str, object]], int] = estimate_request_tokens,
    input_token_limit: int | None = None,
    validate_request: Callable[[dict[str, object]], None] | None = None,
) -> Callable[[dict[str, object]], None]:
    """Share capacity arithmetic and optional provider validation across tasks.

    count_tokens may be a conservative estimator or the serving tokenizer.
    input_token_limit is a batching target, separate from the output allowance.
    """
    def validate(body):
        capacity = context_size
        if input_token_limit is not None:
            capacity = min(capacity, input_token_limit + int(body["max_tokens"]))
        validate_request_budget(body, context_size=capacity, count_tokens=count_tokens)
        if validate_request is not None:
            validate_request(body)
    return validate


def request_fits(body: dict[str, object], validate_request: Callable) -> bool:
    """Only budget overflow means 'does not fit'; other failures propagate."""
    try:
        validate_request(body)
    except PromptBudgetExceeded:
        return False
    return True


def count_llama_prompt_tokens(body: dict[str, object], post: Callable) -> int:
    """Use the serving model's template and tokenizer, without generation."""
    formatted = post('/apply-template', body)
    prompt = formatted.get('prompt')
    if not isinstance(prompt, str):
        raise ValueError('llama-server /apply-template returned no prompt')
    tokenized = post('/tokenize', {'content': prompt, 'add_special': True, 'parse_special': True})
    tokens = tokenized.get('tokens')
    if not isinstance(tokens, list):
        raise ValueError('llama-server /tokenize returned no tokens')
    return len(tokens)


def batch_requests(
    items: Sequence[Item], *, render_request: Callable[[Sequence[Item]], dict[str, object]],
    validate_request: Callable[[dict[str, object]], None],
) -> list[tuple[Item, ...]]:
    """Group complete requests; never silently drop an oversized single item."""
    batches = []
    current: list[Item] = []
    for item in items:
        proposed = [*current, item]
        try:
            validate_request(render_request(proposed))
        except PromptBudgetExceeded:
            if not current:
                raise
            batches.append(tuple(current))
            current = [item]
            validate_request(render_request(current))
        else:
            current = proposed
    if current:
        batches.append(tuple(current))
    return batches


def fit_optional_text(text: str, *, render_request: Callable[[str], dict[str, object]],
                      validate_request: Callable[[dict[str, object]], None]) -> str:
    """Trim only optional context, after proving the mandatory request fits."""
    try:
        validate_request(render_request(text))
        return text
    except PromptBudgetExceeded:
        validate_request(render_request(''))
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        try:
            validate_request(render_request(text[:middle]))
            low = middle
        except PromptBudgetExceeded:
            high = middle - 1
    fitted = text[:low]
    validate_request(render_request(fitted))
    return fitted


def chunk_text(
    text: str, *, render_request: Callable[[str], dict[str, object]],
    validate_request: Callable[[dict[str, object]], None],
) -> list[str]:
    """Partition all text without loss, validating each complete request first."""
    validate_request(render_request(''))
    chunks = []
    remaining = text
    while remaining:
        fitted = fit_optional_text(remaining, render_request=render_request,
                                   validate_request=validate_request)
        low = len(fitted)
        if low == len(remaining):
            chunks.append(remaining)
            break
        if not low:
            raise PromptBudgetExceeded('one source character cannot fit the request budget')
        # Prefer complete lines, preserving their separators in the source chunks.
        boundary = remaining.rfind('\n', 0, low)
        if boundary >= low // 2:
            low = boundary + 1
        part = remaining[:low]
        validate_request(render_request(part))
        chunks.append(part)
        remaining = remaining[low:]
    return chunks
