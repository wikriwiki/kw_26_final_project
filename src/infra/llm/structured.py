"""Pydantic-validated structured output for LLM calls.

Two-tier reliability: sglang's `response_format=json_schema` constrains generation
at the decode level, AND we re-validate the result through Pydantic. On the first
parse miss we retry once at low temperature for determinism.

[한글 요약]
LLM에게 "이런 JSON 구조로 응답해라"고 요구하고, 응답이 정말 그 구조인지 확인하는 모듈.

이중 안전망:
1차 — sglang 서버가 디코딩 단계에서 JSON 스키마를 강제 (토큰을 뽑을 때부터 형식 맞춤)
2차 — 받은 응답을 Pydantic으로 다시 검증 (1차에서 빠져나간 케이스 잡음)

왜 둘 다 필요한가:
- 1차만 하면 sglang 버그/edge case로 잘못된 JSON이 나올 수 있음
- 2차만 하면 잘못된 출력을 너무 많이 만들어서 비용 낭비
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
    # Pydantic 모델 → JSON Schema → OpenAI response_format 형식
    # sglang은 이걸 받으면 디코딩 단계에서 토큰을 제약 (잘못된 토큰을 생성 못 하게)
    # strict=True → 스키마에 없는 필드 추가 금지
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

    # 1차 시도 — 기본 temperature (0.3 정도, 약간의 다양성)
    raw = await client.generate(
        layers,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format=response_format,
    )
    try:
        return schema.model_validate_json(raw)
    except ValidationError as first:
        # 1차 실패 → 경고 로그 남기고 재시도로 진입
        # 호출자에게 즉시 던지지 않음 (재시도로 풀릴 가능성 있어서)
        _log.warning(
            "Structured parse failed (attempt 1), retrying at temp=%s: %s",
            retry_temperature,
            first,
        )

    # 2차 시도 — temperature 낮춰서 더 결정적인 출력 유도
    # 같은 프롬프트 + 같은 caching 효과로 비용 거의 0에 가까움
    raw_retry = await client.generate(
        layers,
        max_tokens=max_tokens,
        temperature=retry_temperature,
        response_format=response_format,
    )
    try:
        return schema.model_validate_json(raw_retry)
    except ValidationError as second:
        # 2차도 실패 → 포기. 원본 응답을 함께 전달해 디버깅 가능하게.
        raise SchemaParseError(
            f"Pydantic validation failed after retry: {second}",
            raw_output=raw_retry,
        ) from second
