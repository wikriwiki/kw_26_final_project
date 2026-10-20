"""AI API 호출 로그 — 국내 AI 트랙 「AI API 호출 로그 제출 가이드 v1.0」 형식.

가이드가 요구하는 것
--------------------
- 요청과 응답을 **한 건의 기록**으로 묶는다.
- 한 건에는 요청 시각(KST)·엔드포인트(도메인 포함)·요청 모델·요청 내용(전체 메시지)과
  응답 시각(KST)·응답 본문(응답 ID·모델명·답변·토큰 사용량)이 모두 들어간다.
- API 키·Authorization 헤더·비밀번호 같은 민감정보는 넣지 않는다.
- 최종 결과 보고서의 **기능 단위로 파일을 나눈다** (파일명 = 기능명).

그래서 이 모듈은
----------------
- 기능 하나당 ``<로그폴더>/<기능명>.jsonl`` 한 파일에 한 줄씩 덧붙인다.
  기능이 끝날 때까지 누적된다(가이드의 기록 절차 4).
- 헤더는 애초에 받지 않는다. 키를 담을 자리가 없으면 새어 나갈 길도 없다.
  그래도 본문에 키 모양 문자열이 섞여 들어오면 ``mask_secrets`` 가 지운다.
- 로그 쓰기가 실패해도 본 기능은 멈추지 않는다. 로그 때문에 시연이 깨지면 안 된다.

로그 폴더는 ``AI_CALL_LOG_DIR`` 로 바꿀 수 있고, ``AI_CALL_LOG=0`` 이면 남기지 않는다.
제출 파일은 ``python -m scripts.ai_call_log_export`` 로 만든다.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "output" / "ai_call_logs"
KST = timezone(timedelta(hours=9), "KST")

#: 키·토큰으로 보이는 값. 본문에 섞여 들어와도 그 부분만 지우고 나머지 구조는 둔다.
_SECRET_PATTERNS = (
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"\bflp_[A-Za-z0-9]{8,}"),  # Friendli 개인 토큰
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"\bup_[A-Za-z0-9]{16,}"),  # Upstage
)
_SECRET_KEYS = {"authorization", "api_key", "apikey", "x-api-key", "x-goog-api-key", "password", "token"}
MASK = "***MASKED***"

#: 국내 AI 연계 기업이 아닌 곳의 AI API 도메인. 국내 AI 트랙에서는 부르면 안 된다.
FOREIGN_AI_HOSTS = (
    "generativelanguage.googleapis.com",
    "aiplatform.googleapis.com",
    "api.openai.com",
    "openai.azure.com",
    "api.anthropic.com",
    "api.mistral.ai",
    "api.deepseek.com",
    "dashscope.aliyuncs.com",
)

_LOCK = threading.Lock()


def now_kst() -> str:
    """밀리초까지 담은 KST ISO 8601 시각."""
    return datetime.now(KST).isoformat(timespec="milliseconds")


def log_dir() -> Path:
    return Path(os.environ.get("AI_CALL_LOG_DIR") or DEFAULT_DIR)


def enabled() -> bool:
    return os.environ.get("AI_CALL_LOG", "1").strip().lower() not in {"0", "false", "off", "no"}


def mask_secrets(value: Any) -> Any:
    """키 모양 문자열과 민감한 이름의 필드만 가린다. 구조는 그대로 둔다."""
    if isinstance(value, dict):
        return {
            k: (MASK if str(k).lower() in _SECRET_KEYS else mask_secrets(v)) for k, v in value.items()
        }
    if isinstance(value, list):
        return [mask_secrets(v) for v in value]
    if isinstance(value, str):
        for pattern in _SECRET_PATTERNS:
            value = pattern.sub(MASK, value)
        return value
    return value


def is_foreign_ai_endpoint(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in FOREIGN_AI_HOSTS)


def _safe_name(feature: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", feature.strip()) or "unnamed"


def record(
    feature: str,
    *,
    endpoint: str,
    model: str,
    request_body: Any,
    requested_at: str,
    response_body: Any = None,
    responded_at: str | None = None,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path | None:
    """요청 한 번과 그 응답을 한 줄로 남긴다. 실패해도 예외를 올리지 않는다."""
    if not enabled():
        return None
    entry: dict[str, Any] = {
        "feature": feature,
        "request": {
            "requested_at": requested_at,
            "endpoint": endpoint,
            "model": model,
            "body": request_body,
        },
        "response": {
            "responded_at": responded_at or now_kst(),
            "body": response_body,
        },
    }
    if error:
        entry["response"]["error"] = error
    if extra:
        entry["extra"] = extra
    try:
        path = log_dir() / f"{_safe_name(feature)}.jsonl"
        line = json.dumps(mask_secrets(entry), ensure_ascii=False)
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return path
    except Exception as exc:  # noqa: BLE001 - 로그 실패가 본 기능을 멈추면 안 된다
        print(f"[ai_call_log] 기록 실패 ({feature}): {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
