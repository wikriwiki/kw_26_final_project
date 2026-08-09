"""보고서 해설용 LLM 클라이언트 — Gemini / OpenAI 호환 엔드포인트.

설계 원칙
---------
1. **표준 라이브러리만 쓴다.** ``urllib`` 로 REST 를 직접 호출한다.
   보고서 생성 job 이 무거운 SDK 의존성 때문에 실패하지 않게 한다.
2. **키가 없으면 조용히 실패하지 않는다.** 어떤 제공자도 설정되지 않았으면
   ``configured=False`` 와 이유를 그대로 돌려주고, 호출부는 결정론적 문장으로 되돌아간다.
   "AI 가 썼다"고 표시된 자리에 규칙 기반 문장이 몰래 들어가지 않는다.
3. **모델이 숫자를 새로 만들지 못하게 한다.** 프롬프트로만 부탁하지 않고,
   생성된 문장의 숫자를 계산 결과에서 나온 숫자 집합과 **대조**한다
   (``numeric_guard``). 근거 없는 숫자가 있으면 그 해설은 채택하지 않는다.

.env 설정 (사용자가 나중에 채운다)
----------------------------------
``GEMINI_API_KEY=...``            ← 이 값만 넣으면 바로 동작한다
``GEMINI_MODEL=gemini-2.5-flash`` (선택)
``REPORT_LLM_PROVIDER=auto``      auto | gemini | openai | none
``REPORT_LLM_BASE_URL=...``       사내 vLLM/SGLang 등 OpenAI 호환 서버
``REPORT_LLM_MODEL=...``
``REPORT_LLM_API_KEY=...``

``.env`` 는 저장소 루트와 ``data/neo4j_load/.env`` 를 모두 읽는다.
환경변수가 이미 있으면 파일보다 우선한다.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILES = (REPO_ROOT / ".env", REPO_ROOT / "data" / "neo4j_load" / ".env", REPO_ROOT / "web" / ".env")

GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"
GEMINI_DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_TOKENS = 900
DEFAULT_TEMPERATURE = 0.2

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
# 문장 속 숫자: 1,234 / 12.5 / -3 / 45% 모두 잡는다.
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def load_dotenv(paths: Iterable[Path] = ENV_FILES, *, override: bool = False) -> dict[str, str]:
    """.env 를 읽어 ``os.environ`` 에 채운다. 이미 있는 값은 기본적으로 보존한다."""
    loaded: dict[str, str] = {}
    for path in paths:
        try:
            if not Path(path).is_file():
                continue
            text = Path(path).read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            match = _ENV_LINE.match(line)
            if not match:
                continue
            key, value = match.group(1), match.group(2).strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            loaded[key] = value
            if override or key not in os.environ:
                os.environ[key] = value
    return loaded


@dataclass
class LlmResult:
    ok: bool
    text: str = ""
    provider: str = "none"
    model: str = ""
    error: str | None = None
    latency_ms: int | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "provider": self.provider,
            "model": self.model,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "usage": self.usage,
        }


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _float_env(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(float(_env(name) or default))
    except ValueError:
        return default


def resolve_provider() -> str:
    """어떤 제공자를 쓸지 결정한다. ``auto`` 는 Gemini → OpenAI호환 → none 순이다."""
    choice = _env("REPORT_LLM_PROVIDER", "auto").lower()
    if choice in {"gemini", "openai", "none"}:
        return choice
    if _env("GEMINI_API_KEY"):
        return "gemini"
    if _env("REPORT_LLM_BASE_URL"):
        return "openai"
    return "none"


def provider_status(*, load_env: bool = True) -> dict[str, Any]:
    """UI/API 가 그대로 보여줄 수 있는 설정 상태. **키 값 자체는 절대 내보내지 않는다.**"""
    if load_env:
        load_dotenv()
    provider = resolve_provider()
    gemini_key = _env("GEMINI_API_KEY")
    base_url = _env("REPORT_LLM_BASE_URL")
    status: dict[str, Any] = {
        "provider": provider,
        "configured": False,
        "model": "",
        "reason": None,
        "env_files": [str(p) for p in ENV_FILES if Path(p).is_file()],
        "expects": {
            "gemini": ["GEMINI_API_KEY", "GEMINI_MODEL(선택)"],
            "openai": ["REPORT_LLM_BASE_URL", "REPORT_LLM_MODEL", "REPORT_LLM_API_KEY(선택)"],
        },
        "key_present": {
            "GEMINI_API_KEY": bool(gemini_key),
            "REPORT_LLM_BASE_URL": bool(base_url),
            "REPORT_LLM_API_KEY": bool(_env("REPORT_LLM_API_KEY")),
        },
        "unknown": [],
    }
    if provider == "gemini":
        status["model"] = _env("GEMINI_MODEL", GEMINI_DEFAULT_MODEL)
        status["configured"] = bool(gemini_key)
        if not gemini_key:
            status["reason"] = "GEMINI_API_KEY 가 .env 에 없습니다. 키를 넣으면 해설이 자동으로 켜집니다."
            status["unknown"].append("GEMINI_API_KEY")
    elif provider == "openai":
        status["model"] = _env("REPORT_LLM_MODEL")
        status["configured"] = bool(base_url and status["model"])
        if not status["configured"]:
            status["reason"] = "REPORT_LLM_BASE_URL 과 REPORT_LLM_MODEL 이 모두 필요합니다."
            status["unknown"].append("REPORT_LLM_BASE_URL")
    else:
        status["reason"] = (
            "해설 LLM 이 설정되지 않았습니다. .env 에 GEMINI_API_KEY 를 넣으면 켜집니다. "
            "지금은 보고서가 계산 결과만으로 생성되며 해설 문장은 결정론적 서술로 대체됩니다."
        )
    return status


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json; charset=utf-8")
    for key, value in headers.items():
        request.add_header(key, value)
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - 고정된 API 엔드포인트
        raw = response.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw else {}


def _gemini(system: str, user: str, *, timeout: float, max_tokens: int, temperature: float) -> LlmResult:
    key = _env("GEMINI_API_KEY")
    model = _env("GEMINI_MODEL", GEMINI_DEFAULT_MODEL)
    base = _env("GEMINI_API_BASE", GEMINI_DEFAULT_BASE).rstrip("/")
    if not key:
        return LlmResult(False, provider="gemini", model=model, error="GEMINI_API_KEY 가 없습니다")
    url = f"{base}/models/{model}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "text/plain",
        },
    }
    started = time.time()
    try:
        data = _post_json(url, payload, {"x-goog-api-key": key}, timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400] if hasattr(exc, "read") else str(exc)
        return LlmResult(False, provider="gemini", model=model, error=f"HTTP {exc.code}: {detail}")
    except Exception as exc:  # noqa: BLE001 - 네트워크/파싱 실패를 그대로 보고한다
        return LlmResult(False, provider="gemini", model=model, error=f"{type(exc).__name__}: {exc}")
    candidates = data.get("candidates") or []
    parts = (candidates[0].get("content", {}).get("parts") if candidates else []) or []
    text = "".join(part.get("text", "") for part in parts).strip()
    usage = data.get("usageMetadata") or {}
    if not text:
        reason = (candidates[0].get("finishReason") if candidates else None) or "빈 응답"
        return LlmResult(False, provider="gemini", model=model, error=f"본문이 비었습니다 ({reason})", usage=usage)
    return LlmResult(
        True,
        text=text,
        provider="gemini",
        model=model,
        latency_ms=int((time.time() - started) * 1000),
        usage={
            "input_tokens": usage.get("promptTokenCount"),
            "output_tokens": usage.get("candidatesTokenCount"),
            "total_tokens": usage.get("totalTokenCount"),
        },
    )


def _openai_compatible(
    system: str, user: str, *, timeout: float, max_tokens: int, temperature: float
) -> LlmResult:
    base = _env("REPORT_LLM_BASE_URL").rstrip("/")
    model = _env("REPORT_LLM_MODEL")
    api_key = _env("REPORT_LLM_API_KEY", "not-needed")
    if not base or not model:
        return LlmResult(False, provider="openai", model=model, error="REPORT_LLM_BASE_URL/MODEL 이 없습니다")
    url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    started = time.time()
    try:
        data = _post_json(url, payload, {"Authorization": f"Bearer {api_key}"}, timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400] if hasattr(exc, "read") else str(exc)
        return LlmResult(False, provider="openai", model=model, error=f"HTTP {exc.code}: {detail}")
    except Exception as exc:  # noqa: BLE001
        return LlmResult(False, provider="openai", model=model, error=f"{type(exc).__name__}: {exc}")
    choices = data.get("choices") or []
    text = (choices[0].get("message", {}).get("content") if choices else "") or ""
    text = text.strip()
    if not text:
        return LlmResult(False, provider="openai", model=model, error="본문이 비었습니다")
    usage = data.get("usage") or {}
    return LlmResult(
        True,
        text=text,
        provider="openai",
        model=model,
        latency_ms=int((time.time() - started) * 1000),
        usage={
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
    )


def complete(
    system: str,
    user: str,
    *,
    timeout: float | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    load_env: bool = True,
) -> LlmResult:
    """설정된 제공자로 한 번 호출한다. 실패는 예외가 아니라 결과로 돌려준다."""
    if load_env:
        load_dotenv()
    provider = resolve_provider()
    timeout = timeout if timeout is not None else _float_env("REPORT_LLM_TIMEOUT", DEFAULT_TIMEOUT)
    max_tokens = max_tokens if max_tokens is not None else _int_env("REPORT_LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS)
    temperature = (
        temperature if temperature is not None else _float_env("REPORT_LLM_TEMPERATURE", DEFAULT_TEMPERATURE)
    )
    if provider == "gemini":
        return _gemini(system, user, timeout=timeout, max_tokens=max_tokens, temperature=temperature)
    if provider == "openai":
        return _openai_compatible(system, user, timeout=timeout, max_tokens=max_tokens, temperature=temperature)
    return LlmResult(
        False,
        provider="none",
        error="해설 LLM 이 설정되지 않았습니다 (.env 의 GEMINI_API_KEY 또는 REPORT_LLM_BASE_URL 확인)",
    )


def ping() -> dict[str, Any]:
    """실제 왕복 1회로 연결을 확인한다. UI 의 '연결 확인' 버튼이 부르는 함수."""
    status = provider_status()
    if not status["configured"]:
        return {**status, "reachable": False, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    result = complete(
        "너는 연결 확인용 응답기다. 사용자가 무엇을 묻든 정확히 'OK' 한 단어만 출력한다.",
        "연결 확인",
        max_tokens=16,
        temperature=0.0,
        load_env=False,
    )
    return {
        **status,
        "reachable": result.ok,
        "sample": result.text[:40] if result.ok else None,
        "error": result.error,
        "latency_ms": result.latency_ms,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# --------------------------------------------------------------------------- #
# 숫자 가드
# --------------------------------------------------------------------------- #


def _canonical(token: str) -> str:
    token = token.replace(",", "")
    if token.endswith("."):
        token = token[:-1]
    try:
        value = float(token)
    except ValueError:
        return token
    if value == int(value):
        return str(int(abs(value)))
    return f"{abs(value):.10g}"


def allowed_number_set(payload: Any, *, extra: Iterable[float] = ()) -> set[str]:
    """계산 결과에 실제로 등장하는 숫자들의 정규화 집합.

    사람이 읽는 축약(억/만)과 반올림도 허용하도록, 각 값에 대해
    원값·0/1/2자리 반올림·만 단위·억 단위 표현을 모두 넣는다.
    """
    allowed: set[str] = set()

    def add(value: float) -> None:
        for candidate in (
            value,
            round(value),
            round(value, 1),
            round(value, 2),
            value / 1e4,
            round(value / 1e4, 1),
            round(value / 1e4, 2),
            value / 1e8,
            round(value / 1e8, 1),
            round(value / 1e8, 2),
            value * 100,
            round(value * 100, 1),
            round(value * 100, 2),
        ):
            try:
                allowed.add(_canonical(str(float(candidate))))
            except (TypeError, ValueError):
                continue

    def walk(node: Any) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            add(float(node))
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(payload)
    for value in extra:
        add(float(value))
    # 0~100 의 작은 정수는 순번·개수 표현에 흔히 쓰이므로 허용한다.
    allowed.update(str(i) for i in range(0, 101))
    return allowed


def numeric_guard(text: str, allowed: set[str]) -> tuple[bool, list[str]]:
    """문장 속 숫자가 모두 계산 결과에서 나왔는지 확인한다."""
    offenders: list[str] = []
    for token in _NUMBER.findall(text):
        canonical = _canonical(token)
        if canonical not in allowed:
            offenders.append(token)
    return (not offenders), offenders
