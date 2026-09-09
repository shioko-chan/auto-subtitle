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
    batches: list[tuple[Item, ...]] = []
    current: list[Item] = []
    for item in items:
        proposed = [*current, item]
        try:
            validate_prompt_budget(
                render_prompt(proposed),
                context_size=context_size,
                max_output_tokens=max_output_tokens(len(proposed)),
                estimate_tokens=estimate_tokens,
            )
        except PromptBudgetExceeded:
            if not current:
                raise
            batches.append(tuple(current))
            current = [item]
            validate_prompt_budget(
                render_prompt(current),
                context_size=context_size,
                max_output_tokens=max_output_tokens(1),
                estimate_tokens=estimate_tokens,
            )
        else:
            current = proposed
    if current:
        batches.append(tuple(current))
    return batches
