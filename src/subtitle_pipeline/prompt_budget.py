from __future__ import annotations

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


def prompt_input_budget(context_size: int, max_output_tokens: int) -> int:
    return max(0, context_size - max_output_tokens)


def validate_prompt_budget(
    prompt: str,
    *,
    context_size: int,
    max_output_tokens: int,
    estimate_tokens: Callable[[str], int] = estimate_prompt_tokens,
) -> None:
    estimated = estimate_tokens(prompt)
    budget = prompt_input_budget(context_size, max_output_tokens)
    if estimated > budget:
        raise PromptBudgetExceeded(
            f"estimated prompt tokens={estimated} exceed budget={budget} "
            f"(context={context_size}, output_reserve={max_output_tokens})"
        )


Item = TypeVar("Item")


def batch_prompt_items(
    items: Sequence[Item],
    *,
    render_prompt: Callable[[Sequence[Item]], str],
    context_size: int,
    max_output_tokens: Callable[[int], int],
    estimate_tokens: Callable[[str], int] = estimate_prompt_tokens,
) -> list[tuple[Item, ...]]:
    return batch_requests(
        items,
        render_request=lambda values: {
            "prompt": render_prompt(values), "max_tokens": max_output_tokens(len(values))
        },
        validate_request=lambda body: validate_prompt_budget(
            body["prompt"], context_size=context_size,
            max_output_tokens=body["max_tokens"], estimate_tokens=estimate_tokens,
        ),
    )


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
