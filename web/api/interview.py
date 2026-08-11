"""1대1 인터뷰 — 시뮬레이션 대상자 한 명에게 직접 묻는다.

무엇을 하는가
-------------
대상자의 **실제 기록**(프로필·소비·기억·정책 지갑)을 근거로 묶어 K-EXAONE 에 넘기고,
그 사람의 입장에서 답하게 한다. 모델은 기록에 없는 사실을 만들어 낼 수 없도록
"주어진 기록에만 근거하라"는 제약을 받고, 모르면 모른다고 말하게 한다.

왜 서버에서 부르는가
--------------------
API 키가 브라우저로 나가면 안 된다. 화면은 질문만 보내고, 키는 서버에만 있다.

모델
----
Friendli Dedicated Endpoint 위의 K-EXAONE (OpenAI 호환 `chat/completions`).
설정은 `.env` 로만 받는다 — 키를 코드에 적지 않는다.

    INTERVIEW_BASE_URL=https://api.friendli.ai/dedicated/v1
    INTERVIEW_MODEL=<endpoint-id>
    INTERVIEW_API_KEY=<flp_...>
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from scripts.report.llm import load_dotenv

#: 추론(thinking)은 끈다. 인터뷰는 한 문단 대화라 지연이 곧 체감 품질이다.
THINKING = False
TIMEOUT = float(os.environ.get("INTERVIEW_TIMEOUT", "60"))
MAX_TOKENS = int(os.environ.get("INTERVIEW_MAX_TOKENS", "700"))

SYSTEM = """당신은 서울시 상권 정책 시뮬레이션에 참여한 시민 한 명입니다.
아래 '내 기록'에 적힌 사실만 근거로, 그 사람 본인의 말투로 답하세요.

지켜야 할 것:
- 기록에 없는 사실을 지어내지 마세요. 모르면 "그건 기억에 없어요"라고 말하세요.
- 금액·날짜·장소는 기록에 적힌 값을 그대로 쓰세요. 어림잡지 마세요.
- 3~5문장으로 짧게, 존댓말로 답하세요.
- 자신이 인공지능이라거나 시뮬레이션이라는 말은 하지 마세요. 그냥 시민으로 답하세요.
"""


class InterviewError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def settings() -> dict[str, Any]:
    """설정 상태. **키 값 자체는 절대 담지 않는다.**"""
    load_dotenv()
    base = os.environ.get("INTERVIEW_BASE_URL", "").strip()
    model = os.environ.get("INTERVIEW_MODEL", "").strip()
    key = os.environ.get("INTERVIEW_API_KEY", "").strip()
    ready = bool(base and model and key)
    return {
        "ready": ready,
        "model_label": "K-EXAONE",
        "reason": None
        if ready
        else "대화 모델이 아직 연결되지 않았습니다. 준비되면 바로 물어볼 수 있습니다.",
    }


def _profile_block(agent: dict[str, Any]) -> str:
    """대상자 기록을 모델이 읽을 수 있는 짧은 한국어 블록으로 만든다.

    통째로 넘기지 않는다. 기록이 길수록 모델이 중요한 사실을 놓치고,
    화면에 없는 내부 필드까지 답변에 흘러나온다.
    """
    lines: list[str] = []
    add = lines.append
    add(f"- 나이대·성별: {agent.get('age_band') or '기록 없음'} {agent.get('gender') or ''}".strip())
    add(f"- 사는 곳: {agent.get('residence') or '기록 없음'}")
    if agent.get("job"):
        add(f"- 하는 일: {agent['job']}")
    if agent.get("spend_decile") is not None:
        add(f"- 소비 분위: {agent['spend_decile']}분위 (1이 가장 적게 쓰는 쪽)")
    if agent.get("grant_total"):
        add(f"- 받은 소비쿠폰: {int(agent['grant_total']):,}원")
    if agent.get("grant_used") is not None:
        add(f"- 그중 쓴 금액: {int(agent['grant_used']):,}원")
    spend = agent.get("spend_by_category") or {}
    if spend:
        top = sorted(spend.items(), key=lambda kv: -kv[1])[:6]
        add("- 주로 쓴 곳: " + ", ".join(f"{k} {int(v):,}원" for k, v in top))
    visits = agent.get("recent_visits") or []
    if visits:
        add("- 최근 다녀온 곳:")
        for v in visits[:8]:
            add(f"    {v.get('day')} {v.get('place') or ''} {v.get('category') or ''} "
                f"{int(v.get('amount') or 0):,}원".rstrip())
    memories = agent.get("memories") or []
    if memories:
        add("- 기억하는 일:")
        for m in memories[:8]:
            add(f"    {m}")
    return "\n".join(lines)


def ask(agent: dict[str, Any], question: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    conf = settings()
    if not conf["ready"]:
        raise InterviewError(503, conf["reason"] or "대화 모델이 연결되지 않았습니다.")
    question = (question or "").strip()
    if not question:
        raise InterviewError(400, "질문을 입력해 주세요.")
    if len(question) > 500:
        raise InterviewError(400, "질문이 너무 깁니다. 500자 안으로 줄여 주세요.")

    base = os.environ["INTERVIEW_BASE_URL"].rstrip("/")
    messages = [{"role": "system", "content": SYSTEM + "\n\n[내 기록]\n" + _profile_block(agent)}]
    # 직전 대화만 넘긴다. 길어질수록 모델이 기록보다 제 말을 근거로 삼는다
    for turn in (history or [])[-6:]:
        role = turn.get("role")
        if role in ("user", "assistant") and turn.get("content"):
            messages.append({"role": role, "content": str(turn["content"])[:1000]})
    messages.append({"role": "user", "content": question})

    payload = {
        "model": os.environ["INTERVIEW_MODEL"],
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "chat_template_kwargs": {"enable_thinking": THINKING},
    }
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {os.environ['INTERVIEW_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # 원문 오류를 그대로 올리지 않는다 — 키가 섞여 나올 수 있고 사용자도 못 고친다
        raise InterviewError(
            502, f"대화 모델이 응답하지 않았습니다 (HTTP {exc.code}). 잠시 후 다시 시도해 주세요."
        ) from exc
    except Exception as exc:  # 네트워크·시간초과
        raise InterviewError(
            502, "대화 모델에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요."
        ) from exc

    choices = body.get("choices") or []
    text = (choices[0].get("message", {}).get("content") or "").strip() if choices else ""
    if not text:
        raise InterviewError(502, "대화 모델이 빈 응답을 보냈습니다. 다시 물어봐 주세요.")
    usage = body.get("usage") or {}
    return {
        "answer": text,
        "model_label": "K-EXAONE",
        "tokens": {"in": usage.get("prompt_tokens"), "out": usage.get("completion_tokens")},
    }
