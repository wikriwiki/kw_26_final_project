"""
llm_client.py
=============
정책 추출용 LLM 호출 래퍼.

`scripts.sim.llm_client` 의 OpenAI(SGLang) 클라이언트를 재사용하되, 정책 추출에 필요한
**Structured Output** 메서드를 추가.

두 경로 지원:
  1. `client.beta.chat.completions.parse(response_format=PydanticModel)`
     — OpenAI 클라우드 (모델이 strict structured output 지원 시).
  2. `response_format={"type":"json_object"}` 폴백 — SGLang / 구버전 호환.
     응답을 직접 Pydantic 검증.

LLM 호출 실패 시 재시도:
  - RateLimit / Timeout / 5xx → 지수 백오프 2회
  - 400-class → 즉시 raise (요청 자체가 잘못)
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)

DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF_BASE = 1.5


class StructuredCompletionError(RuntimeError):
    """structured output 호출 / 파싱 단계의 베이스 에러."""


class StructuredRefusalError(StructuredCompletionError):
    """모델이 응답을 거부함."""


class StructuredValidationError(StructuredCompletionError):
    """응답이 스키마와 안 맞음."""


# ---------------------------------------------------------------------------
# 메인 API
# ---------------------------------------------------------------------------
def call_chat_structured(
    response_model: type[T],
    system_prompt: str,
    user_prompt: str,
    *,
    mode: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> T:
    """LLM 에 system+user 보내고 `response_model` 인스턴스를 받아온다.

    구현 우선순위:
      1. `beta.chat.completions.parse()` — OpenAI 클라우드용 strict schema.
      2. response_format=json_object + 수동 파싱 — SGLang 폴백.

    `OPENAI_PROVIDER=openai` (env) 또는 SGLang base_url 이 OpenAI 클라우드를 가리킬 때
    1번 경로를 시도. 그 외엔 2번 폴백.
    """
    from scripts.sim.llm_client import (
        get_client,
        get_spec,
        resolve_mode,
        _extra_body_for,
    )

    spec = get_spec(resolve_mode(mode))
    client = get_client()

    use_strict = _supports_strict_structured(client)

    def _attempt() -> T:
        if use_strict:
            return _call_via_strict_parse(
                client, spec, system_prompt, user_prompt, response_model,
                temperature, max_tokens,
            )
        return _call_via_json_object(
            client, spec, system_prompt, user_prompt, response_model,
            temperature, max_tokens, _extra_body_for,
        )

    return _with_retry(_attempt, max_retries=max_retries)


# ---------------------------------------------------------------------------
# Strict parse 경로 (OpenAI 클라우드)
# ---------------------------------------------------------------------------
def _supports_strict_structured(client) -> bool:
    """base_url 이 OpenAI 클라우드면 True. 환경변수로 강제 가능."""
    forced = os.getenv("POLICY_STRUCTURED_MODE")
    if forced == "strict":
        return True
    if forced == "json_object":
        return False
    base = str(getattr(client, "base_url", "") or "").lower()
    return "api.openai.com" in base


def _call_via_strict_parse(
    client, spec, system_prompt, user_prompt, response_model, temperature, max_tokens,
):
    completion = client.beta.chat.completions.parse(
        model=spec.hf_id,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=response_model,
    )
    choice = completion.choices[0].message
    if getattr(choice, "refusal", None):
        raise StructuredRefusalError(f"LLM refused: {choice.refusal}")
    if choice.parsed is None:
        raise StructuredCompletionError("LLM returned no parsed content")
    return choice.parsed


# ---------------------------------------------------------------------------
# JSON object 폴백 (SGLang / 자체 호스팅 LLM)
# ---------------------------------------------------------------------------
def _call_via_json_object(
    client, spec, system_prompt, user_prompt, response_model, temperature, max_tokens,
    extra_body_for,
):
    sys_with_schema = (
        f"{system_prompt}\n\n"
        f"Return JSON only matching this schema:\n"
        f"{json.dumps(response_model.model_json_schema(), ensure_ascii=False)}"
    )
    response = client.chat.completions.create(
        model=spec.hf_id,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": sys_with_schema},
            {"role": "user", "content": user_prompt},
        ],
        extra_body=extra_body_for(spec.family) or None,
    )
    text = response.choices[0].message.content or ""
    payload = _parse_json_loose(text)
    try:
        return response_model.model_validate(payload)
    except ValidationError as exc:
        raise StructuredValidationError(str(exc)) from exc


def _parse_json_loose(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise StructuredValidationError("LLM JSON root must be object")
    return parsed


# ---------------------------------------------------------------------------
# Retry helper — RateLimit/Timeout/5xx 재시도, 4xx 즉시 raise
# ---------------------------------------------------------------------------
def _with_retry(call, *, max_retries: int):
    try:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            BadRequestError,
            InternalServerError,
            RateLimitError,
        )
    except ImportError:  # pragma: no cover
        return call()

    transient = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
    permanent = (BadRequestError,)

    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return call()
        except permanent:
            raise
        except transient as exc:
            last_err = exc
            if attempt >= max_retries:
                break
            time.sleep(DEFAULT_BACKOFF_BASE * (2 ** attempt) + random.random())
    assert last_err is not None
    raise last_err
