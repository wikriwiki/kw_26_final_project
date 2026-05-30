"""SGLang/vLLM OpenAI-호환 엔드포인트 통합 클라이언트.

원본: prototype `sglang_client.py` (feat/sglang-migration 브랜치)를 우리 코드에 맞게 확장.
주요 변경:
  - sync `generate_chat` 추가 (우리 ThreadPoolExecutor 기반 메인 루프와 호환)
  - chat.completions raw 응답까지 반환 (token usage 메타 필요)
  - SGLang 기본 포트 30000, vLLM 호환 8000도 자동 감지
  - Qwen family enable_thinking=False 자동 주입

사용 예:
    from llm_client import call_chat, get_active_mode
    mode = get_active_mode()         # CLI/env/default 우선순위
    resp = call_chat(mode, system, user, max_tokens=300)
    text = resp.choices[0].message.content
    usage = resp.usage
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


# ═══════════════════════════════════════════
# Model Registry (prototype과 동일)
# ═══════════════════════════════════════════
@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    family: str
    description: str


MODELS: dict[str, ModelSpec] = {
    "qwen32b": ModelSpec(
        key="qwen32b",
        hf_id="Qwen/Qwen3-32B-AWQ",
        family="qwen",
        description="기본값. AWQ 4-bit 양자화. RTX 5090 / A100 80GB 1장에서 동작.",
    ),
    "qwen8b": ModelSpec(
        key="qwen8b",
        hf_id="Qwen/Qwen3-8B",
        family="qwen",
        description="텍스트 전용 8B 모델. BF16, RTX 5090 32GB에 여유. "
                    "Qwen3-14B-AWQ 대비 ~30% 빠름. 페르소나 요약·시뮬 기본값.",
    ),
    "qwen9b": ModelSpec(
        key="qwen9b",
        hf_id="Qwen/Qwen3-8B",
        family="qwen",
        description="(deprecated) qwen8b alias — Qwen3.5-9B 멀티모달 회피. qwen8b 사용 권장.",
    ),
    "qwen14b": ModelSpec(
        key="qwen14b",
        hf_id="Qwen/Qwen3-14B-AWQ",
        family="qwen",
        description="중간 사이즈 14B AWQ. RTX 5090 32GB에 여유롭게 fit, "
                    "Qwen3-32B 대비 추론 ~2배 빠름. 한국어 OK.",
    ),
    "exaone": ModelSpec(
        key="exaone",
        hf_id="LGAI-EXAONE/EXAONE-4.5-33B-FP8",
        family="exaone",
        description="국내 대회용 EXAONE 33B FP8.",
    ),
}

DEFAULT_MODE = "qwen8b"
DEFAULT_BASE_URL = "http://localhost:30000/v1"   # SGLang 기본 포트
VLLM_FALLBACK_URL = "http://localhost:8000/v1"   # vLLM 기존 포트 (호환)


# ═══════════════════════════════════════════
# 모드 해결
# ═══════════════════════════════════════════
def resolve_mode(cli_arg: str | None = None) -> str:
    """우선순위: CLI > LLM_MODE env > DEFAULT_MODE."""
    mode = cli_arg or os.getenv("LLM_MODE") or DEFAULT_MODE
    if mode not in MODELS:
        raise ValueError(
            f"Unknown LLM_MODE={mode!r}. Choose: {', '.join(MODELS)}"
        )
    return mode


def get_spec(mode: str | None = None) -> ModelSpec:
    return MODELS[resolve_mode(mode)]


def get_active_mode() -> str:
    """현재 활성 모드 (CLI 없을 때 env/default)."""
    return resolve_mode(None)


# ═══════════════════════════════════════════
# 클라이언트 (싱글톤, thread-safe)
# ═══════════════════════════════════════════
_CLIENT: OpenAI | None = None
_CLIENT_LOCK = threading.Lock()


def make_client(base_url: str | None = None) -> OpenAI:
    """OpenAI 호환 클라이언트. SGLang(30000) 또는 vLLM(8000) 자동.

    base_url 우선순위:
      1. 인자
      2. env SGLANG_BASE_URL
      3. env LLM_BASE_URL
      4. SGLang 기본 (30000) — 안 떠 있으면 vLLM (8000)
    """
    if base_url is None:
        base_url = os.getenv("SGLANG_BASE_URL") or os.getenv("LLM_BASE_URL")
    if base_url is None:
        base_url = _autodetect_base_url()
    return OpenAI(base_url=base_url, api_key="EMPTY")


def _autodetect_base_url() -> str:
    """SGLang(30000) 우선 — 안 뜨면 vLLM(8000) 폴백."""
    import socket
    for url, port in [(DEFAULT_BASE_URL, 30000), (VLLM_FALLBACK_URL, 8000)]:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return url
        except OSError:
            continue
    return DEFAULT_BASE_URL


def get_client() -> OpenAI:
    """싱글톤 클라이언트. double-checked locking 으로 thread-safe.

    workers=32+ 의 첫 호출에서 race condition 으로 다중 client 생성 방지.
    OpenAI SDK 내부 httpx 클라이언트가 connection pool 을 가지므로 단일
    인스턴스 재사용이 HTTP keep-alive 효과 극대화.
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = make_client()
    return _CLIENT


# ═══════════════════════════════════════════
# 모델군별 extra_body
# ═══════════════════════════════════════════
def _extra_body_for(family: str) -> dict[str, Any]:
    """Qwen3 family는 <think> 토큰 낭비를 막기 위해 thinking 강제 끔."""
    if family == "qwen":
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


# ═══════════════════════════════════════════
# 동기 호출 (메인 LLM 콜) — ThreadPoolExecutor 호환
# ═══════════════════════════════════════════
def call_chat(
    mode: str | None,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 1200,
    client: OpenAI | None = None,
    response_format: dict | None = None,
) -> Any:
    """동기 호출. response 객체 그대로 반환 (usage·choices 등 메타 필요).

    response_format: vLLM `response_format` 전달 — strict JSON schema 강제.
    예: {"type":"json_schema","json_schema":{"name":"...","strict":True,"schema":{...}}}
    """
    spec = get_spec(mode)
    cli = client or get_client()
    extra = _extra_body_for(spec.family)
    kwargs: dict = dict(
        model=spec.hf_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra or None,
    )
    if response_format:
        kwargs["response_format"] = response_format
    return cli.chat.completions.create(**kwargs)


# ═══════════════════════════════════════════
# (옵션) 비동기 — prototype 호환
# ═══════════════════════════════════════════
async def generate_chat(
    mode: str | None,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.85,
    max_tokens: int = 2000,
) -> str:
    """prototype 시그니처 호환 — text only 반환."""
    import asyncio
    resp = await asyncio.to_thread(
        call_chat, mode, system_prompt, user_prompt,
        temperature=temperature, max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


# ═══════════════════════════════════════════
# Health check
# ═══════════════════════════════════════════
def healthcheck() -> dict:
    """현재 활성 서버 + 모드 + 응답 가능 여부."""
    try:
        cli = get_client()
        url = str(cli.base_url)
        models = list(cli.models.list().data)
        served = [m.id for m in models]
        spec = get_spec(None)
        return {
            "base_url": url,
            "active_mode": spec.key,
            "active_model": spec.hf_id,
            "served_models": served,
            "served_match": spec.hf_id in served,
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import json
    print(json.dumps(healthcheck(), indent=2, ensure_ascii=False))
