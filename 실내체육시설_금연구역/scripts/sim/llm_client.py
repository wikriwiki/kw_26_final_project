"""SGLang/vLLM OpenAI-호환 엔드포인트 통합 클라이언트.

원본: prototype `sglang_client.py` (feat/sglang-migration 브랜치)를 우리 코드에 맞게 확장.
주요 변경:
  - sync `generate_chat` 추가 (우리 ThreadPoolExecutor 기반 메인 루프와 호환)
  - chat.completions raw 응답까지 반환 (token usage 메타 필요)
  - SGLang 기본 포트 30000, vLLM 호환 8000도 자동 감지
  - LG EXAONE enable_thinking=False 자동 주입

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
    "midm": ModelSpec(
        key="midm",
        hf_id="K-intelligence/Midm-2.0-Base-Instruct",
        family="midm",
        description="KT Midm 2.0 Base Instruct — 한국어 특화 instruct 모델. "
                    "vLLM 0.11 호환 (Llama/Mistral 호환 아키텍처 가정). "
                    "served_model_name=midm-2.0-base-instruct.",
    ),
    "midm_awq": ModelSpec(
        key="midm_awq",
        hf_id="jinkyeongk/Midm-2.0-Base-Instruct-AWQ",
        family="midm",
        description="Midm 2.0 Base Instruct AWQ 4-bit (community quant). "
                    "served_model_name=midm-2.0-base-instruct (BF16과 동일 이름으로 호환).",
    ),
    "exaone": ModelSpec(
        key="exaone",
        hf_id="LGAI-EXAONE/EXAONE-4.0-32B-AWQ",
        family="exaone",
        description="EXAONE 4.0 32B AWQ (4-bit). RTX 5090 32GB single-GPU fit. "
                    "text-only. served_model_name=exaone-4.0-32b-awq. "
                    "WSL Ubuntu venv (uv) + vllm 0.11.0 + transformers 4.55 + flashinfer 비활성.",
    ),
    "exaone_4_5": ModelSpec(
        key="exaone_4_5",
        hf_id="LGAI-EXAONE/EXAONE-4.5-33B-AWQ",
        family="exaone",
        description="기본 LG EXAONE 4.5 33B AWQ. 단일 GPU의 메모리 효율을 고려한 선택이며 "
                    "동일 GPU에서 가장 빠른 변형이라는 실측 주장은 아니다. 자동 모델 폴백 없음.",
    ),
    "exaone_fp8": ModelSpec(
        key="exaone_fp8",
        hf_id="LGAI-EXAONE/EXAONE-4.5-33B-FP8",
        family="exaone",
        description="(레거시) EXAONE 33B FP8 — RTX 5090 단일 GPU에 빡빡함.",
    ),
}

DEFAULT_MODE = "exaone_4_5"
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


def require_supported_model_id(model_id: str) -> str:
    """Validate direct HTTP experiment configs without rewriting frozen evidence."""
    if model_id not in {spec.hf_id for spec in MODELS.values()}:
        raise ValueError(f"Unsupported model ID {model_id!r}; use a model registered in llm_client. "
                         "Historical configs are retained as evidence, not executable model aliases.")
    return model_id


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
    # Application stages own bounded retries. The timeout is recorded in the
    # experiment manifest so a slow, loaded server cannot silently turn valid
    # agent-days into skips under a fixed 180-second client limit.
    raw_timeout = os.getenv("SIM_LLM_TIMEOUT_SECONDS", "180")
    try:
        timeout_seconds = int(raw_timeout)
    except ValueError as exc:
        raise ValueError("SIM_LLM_TIMEOUT_SECONDS must be an integer") from exc
    if not 30 <= timeout_seconds <= 1800:
        raise ValueError("SIM_LLM_TIMEOUT_SECONDS must be between 30 and 1800")
    return OpenAI(base_url=base_url, api_key="EMPTY", timeout=float(timeout_seconds), max_retries=0)


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
    """추론(thinking) 토큰 낭비·JSON 파싱 파손을 막기 위해 강제로 끔.

    EXAONE 4.x는 chat_template의 enable_thinking 플래그를 지원한다.
    EXAONE-4.5는 기본이 thinking ON이라 끄지 않으면 매 호출이 수백~수천 추론
    토큰을 내고 Stage 응답의 JSON이 추론 서문에 묻혀 파싱 실패한다.
    (SGLang 실측: enable_thinking=False → 완성 토큰 300+ → 16, 순수 JSON.)
    """
    if family == "exaone":
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
    # (G6, 선택) 상생 백테스트 결정성 보강 — POLICY_BACKTEST_DETERMINISTIC=1 이면
    # 모든 LLM 호출을 temperature=0 으로 강제해 A/B런 잔여 노이즈를 줄인다(§5.4·§9 G6).
    # 부동소수점 비결합성까지는 못 없애므로 '완전 결정론'이 아니라 '노이즈 축소'.
    if os.environ.get("POLICY_BACKTEST_DETERMINISTIC", "0") == "1":
        temperature = 0.0
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
        grammar_mode = os.environ.get('SIM_JSON_GRAMMAR_MODE', 'json_schema')
        if grammar_mode not in {'json_schema', 'json_object'}:
            raise ValueError('Unsupported SIM_JSON_GRAMMAR_MODE')
        # Outlines' JSON-object grammar avoids the observed XGrammar loop that
        # emitted only whitespace. Python still validates all grounded fields,
        # source references and POI choices before any graph transaction.
        kwargs["response_format"] = (
            {'type': 'json_object'}
            if grammar_mode == 'json_object' and response_format.get('type') == 'json_schema'
            else response_format
        )
    from no_smoking_context import next_llm_seed
    paired_seed = next_llm_seed()
    if paired_seed is not None:
        kwargs["seed"] = paired_seed
    from interview_evidence import record_chat_call
    def dispatch():
        from prompt_budget import check_request_budget
        check_request_budget(kwargs)
        return cli.chat.completions.create(**kwargs)
    return record_chat_call(kwargs, dispatch)


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
