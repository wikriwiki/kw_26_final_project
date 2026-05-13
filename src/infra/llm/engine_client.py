"""Async HTTP client for an sglang server (OpenAI-compatible API).

One client wraps one server. Load balancing across multiple servers is out of
scope for v1 — single sglang instance hosting the currently-active model.

[한글 요약]
sglang 서버(다른 프로세스/머신에서 돌아감)에 HTTP POST를 보내는 비동기 클라이언트.
OpenAI Chat Completions API와 호환되는 엔드포인트를 호출.

설계 의도:
- 클라이언트 1개 = 서버 1대 (LB는 v1 범위 외)
- 비동기 — 동시에 수백 요청을 처리해야 throughput이 나옴
- 모드별 차이(Qwen thinking off 등)는 _build_payload에서만 처리 → 호출자 무지
- 4xx와 5xx 구분 — 5xx는 자동 재시도, 4xx는 즉시 raise (호출자 코드 버그)
"""
from __future__ import annotations

import os
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .errors import EngineHTTPError, EngineTimeoutError
from .models import LLMMode, ModelProfile, get_profile, resolve_mode
from .prompt_layers import PromptLayers


class EngineClient:
    """Async sglang client. Use as an async context manager or call `aclose()`."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        mode: str | LLMMode | None = None,
        timeout: float = 300.0,
        max_retries: int = 3,
    ) -> None:
        # endpoint 우선순위: 인자 > LLM_ENDPOINT 환경변수 > localhost 기본값
        raw_endpoint = endpoint or os.getenv("LLM_ENDPOINT") or "http://localhost:30000/v1"
        self.endpoint: str = raw_endpoint.rstrip("/")
        self.mode: LLMMode = resolve_mode(mode)
        self.profile: ModelProfile = get_profile(self.mode)
        self._max_retries = max_retries
        # timeout 300s — Plan 생성처럼 긴 응답까지 허용
        # connect=10s — 서버 자체가 다운된 경우 빠르게 실패 인지
        # max_connections 512 — Qwen 256 동시 요청 + 여유분
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=10.0),
            limits=httpx.Limits(max_connections=512, max_keepalive_connections=128),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> EngineClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # ─── public API ────────────────────────────────────────────────

    async def generate(
        self,
        layers: PromptLayers,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Single chat completion. Returns the assistant message content string."""
        payload = self._build_payload(
            layers,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )
        data = await self._post_with_retry(f"{self.endpoint}/chat/completions", payload)
        return data["choices"][0]["message"]["content"]

    # ─── internals ─────────────────────────────────────────────────

    def _build_payload(
        self,
        layers: PromptLayers,
        *,
        max_tokens: int,
        temperature: float,
        response_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        # OpenAI Chat Completions 형식의 페이로드를 조립.
        # model 필드는 모드에 따라 자동 결정 — 호출자는 모드를 모름.
        payload: dict[str, Any] = {
            "model": self.profile.hf_id,
            "messages": layers.to_messages(),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        # Qwen3.5 emits <think>...</think> by default. Disable for structured tasks.
        # ※ 모드별 차이 처리의 핵심 지점. Qwen은 기본적으로 <think>...</think>로
        #   사고 과정을 출력하는데, JSON 생성에는 방해됨 + 토큰 낭비.
        #   chat_template_kwargs로 sglang 측에 "thinking 끄고 가라"는 신호 전달.
        # EXAONE은 thinking 모드가 없으므로 추가 처리 불필요.
        if self.mode == LLMMode.QWEN:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        return payload

    async def _post_with_retry(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with exponential-backoff retry on transient errors.

        Retries: network errors and 5xx responses. NOT retried: 4xx (caller bug).
        """
        # 지수 백오프 재시도 — 1초, 2초, 4초… 최대 10초 간격
        # 재시도 대상: 네트워크 오류 + 5xx (서버 일시 장애)
        # 재시도 안 함: 4xx (요청 자체가 잘못 — 재시도해도 똑같이 실패)
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, max=10),
            retry=retry_if_exception_type((httpx.TransportError, EngineTimeoutError)),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                try:
                    resp = await self._client.post(url, json=payload)
                except httpx.TimeoutException as e:
                    # 타임아웃 → 재시도 대상 예외로 변환
                    raise EngineTimeoutError(str(e)) from e

                if resp.status_code >= 500:
                    # 서버 일시 장애 → 재시도 가능한 예외로 raise
                    raise EngineTimeoutError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                if resp.status_code >= 400:
                    # 4xx → 재시도하지 않는 예외로 raise (즉시 호출자에게 전달)
                    raise EngineHTTPError(resp.status_code, resp.text)

                return resp.json()

        # 도달 불가능 — AsyncRetrying은 성공/실패 둘 중 하나로 끝남
        raise AssertionError("AsyncRetrying loop exited without raising")  # pragma: no cover
