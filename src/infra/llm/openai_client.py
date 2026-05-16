"""
openai_client.py
================
OpenAI API 얇은 래퍼. 두 가지 모드 제공:

  1. `complete(prompt)` — JSON object 모드 (구버전 호환).
  2. `complete_structured(prompt, response_model)` — **Structured Output**.
     Pydantic v2 모델을 그대로 넘기면 OpenAI 가 strict JSON schema 로 강제하여
     스키마를 어긋난 응답이 원천 차단됨. 호출 결과는 검증된 Pydantic 인스턴스.

호출 실패 시 2회 지수 백오프로 재시도. APIError 종류에 따라 분기:
  - RateLimit / Timeout / 5xx → 재시도
  - 400-class (BadRequest, Auth) → 즉시 raise (요청 자체가 잘못)
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

if TYPE_CHECKING:
    from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF_BASE_SECONDS = 1.5

T = TypeVar("T", bound="BaseModel")


# 일시 장애 — 재시도 대상.
_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)
# 영구 장애 — 즉시 raise (요청 자체가 잘못).
_PERMANENT_ERRORS: tuple[type[Exception], ...] = (BadRequestError,)


class OpenAIChatClient:
    def __init__(
        self,
        model: str | None = None,
        env_path: Path = DEFAULT_ENV_PATH,
        temperature: float = 0.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
    ) -> None:
        load_dotenv(env_path)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ------------------------------------------------------------------
    # JSON object 모드 — 구버전 호환. 가능하면 complete_structured 를 쓸 것.
    # ------------------------------------------------------------------
    def complete(self, prompt: str) -> str:
        def _call() -> str:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "Return valid JSON only. Do not include markdown.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content or ""

        return self._with_retry(_call)

    # ------------------------------------------------------------------
    # Structured Output — Pydantic 모델 = JSON schema, 응답이 모델 인스턴스로 옴.
    # ------------------------------------------------------------------
    def complete_structured(
        self,
        prompt: str,
        response_model: type[T],
        *,
        system_prompt: str | None = None,
    ) -> T:
        """OpenAI structured output 호출.

        - `response_model` 은 Pydantic v2 BaseModel 서브클래스.
        - 응답은 OpenAI 가 strict 모드로 스키마를 강제한 뒤 파싱해 돌려준다.
        - `refusal` 이 채워져 오면 모델이 응답을 거부한 것이므로 RuntimeError 로 raise.
        """
        sys_msg = system_prompt or (
            "Extract structured data per the provided schema. "
            "Use null for unknown scalars, [] for unknown lists. "
            "Set requires_human_review=true when ambiguous or low-confidence. "
            "Do not include markdown."
        )

        def _call() -> T:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": prompt},
                ],
                response_format=response_model,
            )
            choice = completion.choices[0].message
            if choice.refusal:
                raise RuntimeError(f"LLM refused: {choice.refusal}")
            if choice.parsed is None:
                raise RuntimeError("LLM returned no parsed content")
            return choice.parsed

        return self._with_retry(_call)

    # ------------------------------------------------------------------
    # Retry helper
    # ------------------------------------------------------------------
    def _with_retry(self, call):
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return call()
            except _PERMANENT_ERRORS:
                raise
            except _TRANSIENT_ERRORS as exc:
                last_err = exc
                if attempt >= self.max_retries:
                    break
                wait = self.backoff_base_seconds * (2 ** attempt) + random.random()
                time.sleep(wait)
            except APIError as exc:
                # 알 수 없는 APIError 도 일단 재시도하되, 한 번만.
                last_err = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.backoff_base_seconds + random.random())

        assert last_err is not None
        raise last_err
