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
from pathlib import Path
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
    default_base_url: str | None = None
    api_key_envs: tuple[str, ...] = ()
    model_id_envs: tuple[str, ...] = ()


FRIENDLI_DEDICATED_BASE_URL = "https://api.friendli.ai/dedicated/v1"
K_EXAONE_DEFAULT_ENDPOINT_ID = "depmkuykpfon9lg"
FRIENDLI_API_KEY_ENVS = (
    "LG_EXAONE_KEY",
    "K_EXAONE_API_KEY",
    "EXAONE_API_KEY",
    "FRIENDLI_API_KEY",
    "FRIENDLI_TOKEN",
)
K_EXAONE_ENDPOINT_ENVS = (
    "K_EXAONE_ENDPOINT_ID",
    "EXAONE_ENDPOINT_ID",
    "FRIENDLI_ENDPOINT_ID",
)


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
    "k_exaone": ModelSpec(
        key="k_exaone",
        hf_id=K_EXAONE_DEFAULT_ENDPOINT_ID,
        family="k_exaone",
        description="K-EXAONE-236B-A23B Friendli Dedicated Endpoint. "
                    "model에는 endpoint-id를 넣고, Authorization은 LG_EXAONE_KEY/flp_* 키를 사용.",
        default_base_url=FRIENDLI_DEDICATED_BASE_URL,
        api_key_envs=FRIENDLI_API_KEY_ENVS,
        model_id_envs=K_EXAONE_ENDPOINT_ENVS,
    ),
    "exaone_4_5": ModelSpec(
        key="exaone_4_5",
        hf_id="LGAI-EXAONE/EXAONE-4.5-33B-AWQ",
        family="exaone",
        description="(미사용) EXAONE 4.5 33B AWQ — quantization schema 호환 이슈로 vllm 0.11 미지원.",
    ),
    "exaone_fp8": ModelSpec(
        key="exaone_fp8",
        hf_id="LGAI-EXAONE/EXAONE-4.5-33B-FP8",
        family="exaone",
        description="(레거시) EXAONE 33B FP8 — RTX 5090 단일 GPU에 빡빡함.",
    ),
}

MODE_ALIASES = {
    "k-exaone": "k_exaone",
    "exaone_api": "k_exaone",
}

DEFAULT_MODE = "qwen8b"
DEFAULT_BASE_URL = "http://localhost:30000/v1"   # SGLang 기본 포트
VLLM_FALLBACK_URL = "http://localhost:8000/v1"   # vLLM 기존 포트 (호환)

_ENV_LOADED = False
_ENV_LOCK = threading.Lock()


def _load_dotenv_once() -> None:
    """Load local .env files without overriding shell-provided variables."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    with _ENV_LOCK:
        if _ENV_LOADED:
            return
        for path in _dotenv_candidates():
            _load_dotenv(path)
        _ENV_LOADED = True


def _dotenv_candidates() -> list[Path]:
    root = Path(__file__).resolve().parents[2]
    candidates: list[Path] = []
    explicit = os.getenv("LLM_ENV_FILE") or os.getenv("DOTENV_PATH")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend([
        Path.cwd() / ".env",
        root / ".env",
        root / "scripts" / "report" / ".env",
    ])

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _first_env(names: tuple[str, ...]) -> str | None:
    _load_dotenv_once()
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


# ═══════════════════════════════════════════
# 모드 해결
# ═══════════════════════════════════════════
def resolve_mode(cli_arg: str | None = None) -> str:
    """우선순위: CLI > LLM_MODE env > DEFAULT_MODE."""
    _load_dotenv_once()
    mode = cli_arg or os.getenv("LLM_MODE") or DEFAULT_MODE
    mode = MODE_ALIASES.get(mode, mode)
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
@dataclass(frozen=True)
class ClientConfig:
    base_url: str
    api_key: str


_CLIENTS: dict[tuple[str, str], OpenAI] = {}
_CLIENT_LOCK = threading.Lock()


def make_client(base_url: str | None = None, mode: str | None = None) -> OpenAI:
    """OpenAI 호환 클라이언트. SGLang/vLLM 또는 Friendli Dedicated 자동.

    base_url 우선순위:
      1. 인자
      2. k_exaone: K_EXAONE_BASE_URL / FRIENDLI_BASE_URL / Friendli 기본 URL
      3. 로컬 모델: SGLANG_BASE_URL / LLM_BASE_URL
      4. 로컬 모델 자동감지: SGLang 기본 (30000) — 안 떠 있으면 vLLM (8000)
    """
    cfg = _client_config(mode, base_url)
    return OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)


def _client_config(mode: str | None = None, base_url: str | None = None) -> ClientConfig:
    spec = get_spec(mode)
    resolved_base_url = _base_url_for(spec, base_url)
    return ClientConfig(
        base_url=resolved_base_url,
        api_key=_api_key_for(spec, resolved_base_url),
    )


def _base_url_for(spec: ModelSpec, base_url: str | None = None) -> str:
    _load_dotenv_once()
    if base_url:
        return base_url
    if spec.default_base_url:
        return (
            os.getenv("K_EXAONE_BASE_URL")
            or os.getenv("FRIENDLI_BASE_URL")
            or spec.default_base_url
        )
    return (
        os.getenv("SGLANG_BASE_URL")
        or os.getenv("LLM_BASE_URL")
        or _autodetect_base_url()
    )


def _api_key_for(spec: ModelSpec, base_url: str) -> str:
    if spec.api_key_envs:
        key = _first_env(spec.api_key_envs)
        if not key:
            env_names = ", ".join(spec.api_key_envs)
            raise RuntimeError(f"{spec.key} API key not found. Set one of: {env_names}")
        return key
    if "api.openai.com" in base_url:
        key = _first_env(("OPENAI_API_KEY",))
        if not key:
            raise RuntimeError("OPENAI_API_KEY not found for api.openai.com base_url")
        return key
    if "friendli.ai" in base_url:
        key = _first_env(FRIENDLI_API_KEY_ENVS)
        if not key:
            env_names = ", ".join(FRIENDLI_API_KEY_ENVS)
            raise RuntimeError(f"Friendli API key not found. Set one of: {env_names}")
        return key
    return os.getenv("LLM_API_KEY") or "EMPTY"


def _model_id_for(spec: ModelSpec) -> str:
    if spec.model_id_envs:
        return _first_env(spec.model_id_envs) or spec.hf_id
    return spec.hf_id


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


def get_client(mode: str | None = None, base_url: str | None = None) -> OpenAI:
    """싱글톤 클라이언트. double-checked locking 으로 thread-safe.

    workers=32+ 의 첫 호출에서 race condition 으로 다중 client 생성 방지.
    OpenAI SDK 내부 httpx 클라이언트가 connection pool 을 가지므로 단일
    인스턴스 재사용이 HTTP keep-alive 효과 극대화.
    """
    resolved_mode = resolve_mode(mode)
    cfg = _client_config(resolved_mode, base_url)
    cache_key = (resolved_mode, cfg.base_url)
    if cache_key in _CLIENTS:
        return _CLIENTS[cache_key]
    with _CLIENT_LOCK:
        if cache_key not in _CLIENTS:
            _CLIENTS[cache_key] = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    return _CLIENTS[cache_key]


# ═══════════════════════════════════════════
# 모델군별 extra_body
# ═══════════════════════════════════════════
def _extra_body_for(family: str) -> dict[str, Any]:
    """Qwen3 family는 <think> 토큰 낭비를 막기 위해 thinking 강제 끔."""
    if family in {"qwen", "k_exaone"}:
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
    resolved_mode = resolve_mode(mode)
    spec = get_spec(resolved_mode)
    cli = client or get_client(resolved_mode)
    extra = _extra_body_for(spec.family)
    kwargs: dict = dict(
        model=_model_id_for(spec),
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
        spec = get_spec(None)
        cli = get_client(spec.key)
        url = str(cli.base_url)
        model_id = _model_id_for(spec)
        result = {
            "base_url": url,
            "active_mode": spec.key,
            "active_model": model_id,
        }
        if spec.default_base_url == FRIENDLI_DEDICATED_BASE_URL:
            result["served_models"] = None
            result["served_match"] = None
            result["models_list_note"] = "Friendli Dedicated uses endpoint-id as model; /models is not used."
            return result
        try:
            models = list(cli.models.list().data)
            served = [m.id for m in models]
            result["served_models"] = served
            result["served_match"] = model_id in served
        except Exception as e:  # noqa: BLE001
            result["served_models"] = None
            result["served_match"] = None
            result["models_list_error"] = str(e)
        return result
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import json
    print(json.dumps(healthcheck(), indent=2, ensure_ascii=False))
