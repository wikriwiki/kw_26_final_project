"""Pydantic-validated structured output for LLM calls.

Two-tier reliability: sglang's `response_format=json_schema` constrains generation
at the decode level, AND we re-validate the result through Pydantic. On the first
parse miss we retry once at low temperature for determinism.
"""
from __future__ import annotations

import logging
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .engine_client import EngineClient
from .errors import SchemaParseError
from .prompt_layers import PromptLayers

T = TypeVar("T", bound=BaseModel)

_log = logging.getLogger(__name__)


def build_response_format(schema: type[BaseModel]) -> dict[str, Any]:
    """OpenAI-style `response_format` from a Pydantic model.

    sglang understands the `json_schema` type and will constrain decoding.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": schema.model_json_schema(),
            "strict": True,
        },
    }


async def generate_structured(
    client: EngineClient,
    layers: PromptLayers,
    schema: type[T],
    *,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    retry_temperature: float = 0.1,
) -> T:
    """Generate and validate against a Pydantic schema.

    On first parse failure, retry once at `retry_temperature` (deterministic).
    Raises `SchemaParseError` if both attempts fail.
    """
    response_format = build_response_format(schema)

    raw = await client.generate(
        layers,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format=response_format,
    )
    try:
        return schema.model_validate_json(raw)
    except ValidationError as first:
        _log.warning(
            "Structured parse failed (attempt 1), retrying at temp=%s: %s",
            retry_temperature,
            first,
        )

    raw_retry = await client.generate(
        layers,
        max_tokens=max_tokens,
        temperature=retry_temperature,
        response_format=response_format,
    )
    try:
        return schema.model_validate_json(raw_retry)
    except ValidationError as second:
        raise SchemaParseError(
            f"Pydantic validation failed after retry: {second}",
            raw_output=raw_retry,
        ) from second
