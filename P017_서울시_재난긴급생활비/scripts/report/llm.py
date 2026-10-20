"""보고서 해설용 LLM 클라이언트 — K-EXAONE (Friendli 전용 엔드포인트, OpenAI 호환).

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
``REPORT_LLM_BASE_URL=https://api.friendli.ai/dedicated/v1``  (기본값)
``REPORT_LLM_MODEL=<K-EXAONE 엔드포인트 ID>``
``REPORT_LLM_API_KEY=<키>``
``REPORT_LLM_PROVIDER=auto``      auto | exaone | none

``REPORT_LLM_*`` 이 비어 있으면 1대1 인터뷰의 ``INTERVIEW_*`` 값을 그대로 쓴다.
키를 두 벌 넣지 않아도 해설과 인터뷰가 함께 켜진다.

국내 AI 트랙 제출물이라 **국외 AI 엔드포인트는 설정돼 있어도 부르지 않는다**
(``ai_call_log.is_foreign_ai_endpoint``).

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

from scripts import ai_call_log

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILES = (REPO_ROOT / ".env", REPO_ROOT / "data" / "neo4j_load" / ".env", REPO_ROOT / "web" / ".env")

DEFAULT_BASE_URL = "https://api.friendli.ai/dedicated/v1"
DEFAULT_TIMEOUT = 60.0
#: 넘어갈 수 있는 실패에 한해 다시 걸어 보는 횟수 (첫 시도 포함).
DEFAULT_RETRIES = 3
RETRY_BACKOFF_SEC = 1.5
#: 다시 걸면 달라질 수 있는 실패들. 인증·요청 형식 오류는 여기에 없다.
TRANSIENT_MARKERS = ("HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504", "timed out", "TimeoutError", "URLError")
DEFAULT_MAX_TOKENS = 900
DEFAULT_TEMPERATURE = 0.2

#: 호출 로그 파일 이름이 되는 기능명 (AI API 호출 로그 제출 가이드: 파일명 = 기능명).
DEFAULT_FEATURE = "보고서_해설"

#: 운영자에게만 보여 줄 안내. 고객 화면에는 `reason` 만 나가고 이 문장은 나가지 않는다.
#: 두 문장을 하나로 합치면 시연 화면에 `.env` 같은 내부 사정이 그대로 뜬다.
OPERATOR_HINT = (
    "저장소 루트의 `.env` 파일에 K-EXAONE 엔드포인트 ID(`REPORT_LLM_MODEL`)와 키(`REPORT_LLM_API_KEY`)를 "
    "넣고 서버를 다시 시작하십시오. `.env.example` 을 복사해 쓰면 됩니다."
)

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


def _setting(name: str) -> str:
    """``REPORT_LLM_<name>`` → 없으면 ``INTERVIEW_<name>``."""
    return _env(f"REPORT_LLM_{name}") or _env(f"INTERVIEW_{name}")


def endpoint_settings() -> tuple[str, str, str]:
    """(base_url, model, api_key). 값이 없으면 빈 문자열."""
    return (_setting("BASE_URL") or DEFAULT_BASE_URL).rstrip("/"), _setting("MODEL"), _setting("API_KEY")


def resolve_provider() -> str:
    """``exaone`` 또는 ``none``. ``auto`` 는 엔드포인트 ID 가 있으면 exaone 이다."""
    choice = _env("REPORT_LLM_PROVIDER", "auto").lower()
    if choice == "none":
        return "none"
    if choice == "exaone":
        return "exaone"
    return "exaone" if _setting("MODEL") else "none"


def provider_status(*, load_env: bool = True) -> dict[str, Any]:
    """UI/API 가 그대로 보여줄 수 있는 설정 상태. **키 값 자체는 절대 내보내지 않는다.**"""
    if load_env:
        load_dotenv()
    provider = resolve_provider()
    base_url, model, api_key = endpoint_settings()
    status: dict[str, Any] = {
        "provider": provider,
        "configured": False,
        "model": "",
        "reason": None,
        "env_files": [str(p) for p in ENV_FILES if Path(p).is_file()],
        "expects": {
            "exaone": ["REPORT_LLM_MODEL", "REPORT_LLM_API_KEY", "REPORT_LLM_BASE_URL(선택)"],
        },
        "key_present": {
            "REPORT_LLM_MODEL": bool(model),
            "REPORT_LLM_API_KEY": bool(api_key),
        },
        "unknown": [],
    }
    if provider == "exaone":
        status["model"] = model
        status["configured"] = bool(model and api_key)
        if ai_call_log.is_foreign_ai_endpoint(base_url):
            status["configured"] = False
            status["reason"] = "해설 모델 주소가 국내 AI 엔드포인트가 아닙니다."
            status["operator_hint"] = "`REPORT_LLM_BASE_URL` 을 K-EXAONE 엔드포인트로 바꾸십시오."
        elif not status["configured"]:
            status["reason"] = "해설 모델이 아직 연결되지 않았습니다."
            status["operator_hint"] = OPERATOR_HINT
            status["unknown"] += [k for k, present in status["key_present"].items() if not present]
    else:
        status["reason"] = (
            "해설 모델이 아직 연결되지 않았습니다. "
            "지금은 보고서가 계산 결과만으로 생성되며 해설 문장은 결정론적 서술로 대체됩니다."
        )
        status["operator_hint"] = OPERATOR_HINT
    return status


def _is_transient(error: str | None) -> bool:
    """다시 걸어 볼 만한 실패인가. 키가 틀린 실패를 재시도하면 시간만 버린다."""
    if not error:
        return False
    return any(marker in error for marker in TRANSIENT_MARKERS)


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json; charset=utf-8")
    for key, value in headers.items():
        request.add_header(key, value)
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - 고정된 API 엔드포인트
        raw = response.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw else {}


def _exaone(
    system: str, user: str, *, timeout: float, max_tokens: int, temperature: float, feature: str = DEFAULT_FEATURE
) -> LlmResult:
    base, model, api_key = endpoint_settings()
    if not model or not api_key:
        return LlmResult(False, provider="exaone", model=model, error="REPORT_LLM_MODEL/API_KEY 가 없습니다")
    if ai_call_log.is_foreign_ai_endpoint(base):
        return LlmResult(False, provider="exaone", model=model, error="국내 AI 엔드포인트가 아닙니다")
    url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        # 추론(thinking)을 켜 두면 해설 한 문단에 수십 초가 걸리고, 그 사고 과정이
        # 본문에 섞여 나오기도 한다. 보고서 해설은 계산 결과를 **옮겨 적는** 일이라
        # 추론이 필요 없다. 이 옵션을 모르는 서버는 그냥 무시한다.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.time()
    requested_at = ai_call_log.now_kst()
    try:
        data = _post_json(url, payload, {"Authorization": f"Bearer {api_key}"}, timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400] if hasattr(exc, "read") else str(exc)
        error = f"HTTP {exc.code}: {detail}"
        ai_call_log.record(feature, endpoint=url, model=model, request_body=payload, requested_at=requested_at, error=error)
        return LlmResult(False, provider="exaone", model=model, error=error)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        ai_call_log.record(feature, endpoint=url, model=model, request_body=payload, requested_at=requested_at, error=error)
        return LlmResult(False, provider="exaone", model=model, error=error)
    ai_call_log.record(feature, endpoint=url, model=model, request_body=payload, requested_at=requested_at, response_body=data)
    choices = data.get("choices") or []
    text = (choices[0].get("message", {}).get("content") if choices else "") or ""
    text = text.strip()
    if not text:
        return LlmResult(False, provider="exaone", model=model, error="본문이 비었습니다")
    usage = data.get("usage") or {}
    return LlmResult(
        True,
        text=text,
        provider="exaone",
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
    feature: str = DEFAULT_FEATURE,
) -> LlmResult:
    """설정된 제공자로 한 번 호출한다. 실패는 예외가 아니라 결과로 돌려준다.

    ``feature`` 는 호출 로그 파일 이름이다 (``scripts.ai_call_log``).
    """
    if load_env:
        load_dotenv()
    provider = resolve_provider()
    timeout = timeout if timeout is not None else _float_env("REPORT_LLM_TIMEOUT", DEFAULT_TIMEOUT)
    max_tokens = max_tokens if max_tokens is not None else _int_env("REPORT_LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS)
    temperature = (
        temperature if temperature is not None else _float_env("REPORT_LLM_TEMPERATURE", DEFAULT_TEMPERATURE)
    )
    call = _exaone if provider == "exaone" else None
    if call is not None:
        # 분당 요청 수를 넘기면 429 로 거절된다. 보고서 한 편은
        # 해설을 여러 번 부르므로 그 거절이 곧 "해설 없는 보고서"가 된다.
        # 넘어갈 수 있는 실패(429·5xx·시간초과)만 짧게 물러났다 다시 시도한다.
        # 키가 틀렸거나 모델 이름이 없는 실패는 다시 걸어도 같으므로 그대로 돌려준다.
        attempts = max(1, _int_env("REPORT_LLM_RETRIES", DEFAULT_RETRIES))
        result = LlmResult(False, provider=provider, error="호출을 시도하지 못했습니다")
        for attempt in range(attempts):
            result = call(
                system, user, timeout=timeout, max_tokens=max_tokens, temperature=temperature, feature=feature
            )
            if result.ok or not _is_transient(result.error) or attempt == attempts - 1:
                return result
            time.sleep(RETRY_BACKOFF_SEC * (2 ** attempt))
        return result
    return LlmResult(
        False,
        provider="none",
        error="해설 LLM 이 설정되지 않았습니다 (.env 의 REPORT_LLM_MODEL·REPORT_LLM_API_KEY 확인)",
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
        feature="연결_점검",
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
