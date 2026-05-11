"""Async HTTP client for an sglang server (OpenAI-compatible API).

One client wraps one server. Load balancing across multiple servers is out of
scope for v1 — single sglang instance hosting the currently-active model.
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
        raw_endpoint = endpoint or os.getenv("LLM_ENDPOINT") or "http://localhost:30000/v1"
        self.endpoint: str = raw_endpoint.rstrip("/")
        self.mode: LLMMode = resolve_mode(mode)
        self.profile: ModelProfile = get_profile(self.mode)
        self._max_retries = max_retries
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
        payload: dict[str, Any] = {
            "model": self.profile.hf_id,
            "messages": layers.to_messages(),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        # Qwen3.5 emits <think>...</think> by default. Disable for structured tasks.
        if self.mode == LLMMode.QWEN:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        return payload

    async def _post_with_retry(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with exponential-backoff retry on transient errors.

        Retries: network errors and 5xx responses. NOT retried: 4xx (caller bug).
        """
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
                    raise EngineTimeoutError(str(e)) from e

                if resp.status_code >= 500:
                    raise EngineTimeoutError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                if resp.status_code >= 400:
                    raise EngineHTTPError(resp.status_code, resp.text)

                return resp.json()

        raise AssertionError("AsyncRetrying loop exited without raising")  # pragma: no cover
