"""
sglang_client.py
================
SGLang OpenAI-호환 엔드포인트를 부르는 얇은 래퍼.

- 3-way 모델 선택 (qwen32b / qwen9b / exaone)
- Qwen 계열은 `enable_thinking=False`를 자동으로 주입해 <think> 토큰 낭비를 막음
- 우선순위: CLI `--model` > 환경변수 `LLM_MODE` > 기본값 `qwen32b`

서버는 `scripts/serve_<mode>.sh` 로 별도 기동한다 (SGLang은 OpenAI 호환 API를 노출).
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    key: str           # CLI/env에서 사용할 짧은 키
    hf_id: str         # HuggingFace 모델 ID (SGLang `--model` 값과 동일)
    family: str        # "qwen" | "exaone" — 모델별 옵션 분기에 사용
    description: str


MODELS: dict[str, ModelSpec] = {
    "qwen32b": ModelSpec(
        key="qwen32b",
        hf_id="Qwen/Qwen3-32B-AWQ",
        family="qwen",
        description="기존 기본값. AWQ 4-bit 양자화. A100 80GB 1장에서 동작.",
    ),
    "qwen9b": ModelSpec(
        key="qwen9b",
        hf_id="Qwen/Qwen3.5-9B",
        family="qwen",
        description="빠른 개발/디버깅용 9B 모델.",
    ),
    "exaone": ModelSpec(
        key="exaone",
        hf_id="LGAI-EXAONE/EXAONE-4.5-33B-FP8",
        family="exaone",
        description="국내 대회용 EXAONE 33B FP8.",
    ),
}

DEFAULT_MODE = "qwen32b"
DEFAULT_BASE_URL = "http://localhost:30000/v1"  # SGLang 기본 포트


def resolve_mode(cli_arg: str | None = None) -> str:
    """우선순위: CLI > LLM_MODE 환경변수 > 기본값."""
    mode = cli_arg or os.getenv("LLM_MODE") or DEFAULT_MODE
    if mode not in MODELS:
        raise ValueError(
            f"Unknown model mode: {mode!r}. "
            f"Choose one of: {', '.join(MODELS)}"
        )
    return mode


def get_spec(mode: str) -> ModelSpec:
    return MODELS[resolve_mode(mode)]


def make_client(base_url: str | None = None) -> OpenAI:
    """SGLang OpenAI-호환 클라이언트.

    base_url 기본값: env `SGLANG_BASE_URL` > `http://localhost:30000/v1`.
    """
    url = base_url or os.getenv("SGLANG_BASE_URL") or DEFAULT_BASE_URL
    return OpenAI(base_url=url, api_key="EMPTY")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _extra_body_for(family: str) -> dict[str, Any]:
    """모델군별 SGLang 확장 옵션."""
    if family == "qwen":
        # Qwen3 계열은 chat template에서 thinking 모드를 끄지 않으면
        # <think>...</think> 블록을 길게 뱉어 토큰을 낭비한다.
        return {"chat_template_kwargs": {"enable_thinking": False}}
    if family == "exaone":
        # EXAONE은 기본적으로 thinking 토큰을 내지 않음. 추가 옵션 없음.
        return {}
    return {}


async def generate_chat(
    client: OpenAI,
    mode: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.85,
    max_tokens: int = 2000,
) -> str:
    """system / user 메시지를 SGLang에 보내고 raw text를 반환.

    프리픽스 캐시 정렬을 위해 호출자는 system_prompt는 호출 간 동일하게,
    user_prompt 는 '공유 → 고유' 순으로 조립된 layered 문자열을 넣는다.
    """
    spec = get_spec(mode)
    extra_body = _extra_body_for(spec.family)

    response = await asyncio.to_thread(
        client.chat.completions.create,
        model=spec.hf_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body or None,
    )
    return response.choices[0].message.content or ""
