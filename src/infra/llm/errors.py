"""Domain exceptions raised by `src.infra.llm`."""
from __future__ import annotations


class LLMError(Exception):
    """Base class for all errors raised by the LLM infrastructure module."""


class EngineTimeoutError(LLMError):
    """The sglang engine did not respond within the timeout (or returned 5xx)."""


class EngineHTTPError(LLMError):
    """Non-retryable 4xx response from the sglang engine."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body


class SchemaParseError(LLMError):
    """Model output failed Pydantic validation after retries."""

    def __init__(self, message: str, *, raw_output: str | None = None) -> None:
        super().__init__(message)
        self.raw_output = raw_output


class ModeMismatchError(LLMError):
    """`LLM_MODE` value is unknown or incompatible with the requested operation."""
