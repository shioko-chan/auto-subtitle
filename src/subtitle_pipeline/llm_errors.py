"""Expected model-response failures, separate from programming errors."""


class LLMResponseError(RuntimeError):
    """The model did not produce a usable, complete response."""


class LLMStreamError(LLMResponseError):
    """The response stream failed before its terminal event."""
