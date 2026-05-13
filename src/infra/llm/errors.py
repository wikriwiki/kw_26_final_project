"""Domain exceptions raised by `src.infra.llm`.

[한글 요약]
LLM 모듈에서 발생할 수 있는 예외 5종을 정의. 호출자가 어떤 예외를 잡아야 할지
명확하게 구분하기 위해 분리. 특히 EngineTimeoutError(재시도 가능)와
EngineHTTPError(호출자 버그라 재시도 불가)를 구분하는 것이 중요.
"""
from __future__ import annotations


class LLMError(Exception):
    """Base class for all errors raised by the LLM infrastructure module."""
    # 모든 LLM 예외의 부모. 호출자가 LLM 관련 에러만 한꺼번에 잡고 싶을 때 사용.


class EngineTimeoutError(LLMError):
    """The sglang engine did not respond within the timeout (or returned 5xx)."""
    # 서버 일시 장애 — 재시도하면 풀릴 가능성 있음.
    # engine_client._post_with_retry가 자동으로 재시도하고, 다 실패하면 이 예외를 raise.


class EngineHTTPError(LLMError):
    """Non-retryable 4xx response from the sglang engine."""
    # 호출자 코드 버그(잘못된 모델명, 잘못된 스키마 등) — 재시도해도 똑같이 실패.
    # 즉시 raise해서 디버깅을 빠르게 유도.

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body  # 응답 본문 보존 — 디버깅 시 필요


class SchemaParseError(LLMError):
    """Model output failed Pydantic validation after retries."""
    # 모델이 JSON 스키마를 어긴 채로 응답함. 재시도 후에도 실패한 경우.
    # raw_output을 보존해서 어떤 응답이 왔는지 추적 가능.

    def __init__(self, message: str, *, raw_output: str | None = None) -> None:
        super().__init__(message)
        self.raw_output = raw_output


class ModeMismatchError(LLMError):
    """`LLM_MODE` value is unknown or incompatible with the requested operation."""
    # LLM_MODE에 'exaone'/'qwen' 외 값이 들어왔거나, 모드와 서버 측 모델이 다를 때.
