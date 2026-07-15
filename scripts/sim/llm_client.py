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
    "qwen35_9b_awq": ModelSpec(
        key="qwen35_9b_awq",
        hf_id="QuantTrio/Qwen3.5-9B-AWQ",
        family="qwen",
        description="Qwen3.5-9B AWQ 4-bit (community 빌드). VRAM ~5GB → KV cache 대폭 여유. "
                    "Qwen3-8B BF16 대비 환각·품질 개선 + workers 80+ 가능.",
    ),
    "qwen3_8b_awq": ModelSpec(
        key="qwen3_8b_awq",
        hf_id="Qwen/Qwen3-8B-AWQ",
        family="qwen",
        description="Qwen3-8B 공식 AWQ. Qwen3.5-9B-AWQ swap 실패 시 fallback. "
                    "VRAM ~5GB. 같은 8B family라 8B BF16 대비 분석 mix 영향 최소.",
    ),
    "qwen36_35b_a3b_awq": ModelSpec(
        key="qwen36_35b_a3b_awq",
        hf_id="QuantTrio/Qwen3.6-35B-A3B-AWQ",
        family="qwen",
        description="Qwen3.6-35B-A3B AWQ (MoE 35B 총, 3B active). text-only 모드. "
                    "VRAM ~18GB. throughput 25~40 agents/min 기대. workers 32~48.",
    ),
    "qwen3_30b_a3b_awq": ModelSpec(
        key="qwen3_30b_a3b_awq",
        hf_id="stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ",
        family="qwen",
        description="Qwen3-30B-A3B-Instruct-2507 AWQ (MoE 30B, 3B active). Qwen3.6 fallback. "
                    "VRAM ~15GB. throughput 25~35 agents/min 기대.",
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
        description="EXAONE 4.5 33B AWQ — EXP-001(GPU LIVE, A100×2 TP2) 채택 모델. "
                    "서빙: scripts/serve/serve_exaone45_awq_a100x2.sh (vllm 최신 필요 — "
                    "0.11에서 quantization schema 미지원 이력, 실패 시 동일 모델 FP8 폴백).",
    ),
    "exaone_fp8": ModelSpec(
        key="exaone_fp8",
        hf_id="LGAI-EXAONE/EXAONE-4.5-33B-FP8",
        family="exaone",
        description="(레거시) EXAONE 33B FP8 — RTX 5090 단일 GPU에 빡빡함.",
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
