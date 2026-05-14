"""Stage 1 — 의도·카테고리·anchor 선택 LLM 호출.

입력: DawnContext (페르소나 + 어제 State + Memory + 약속 + 정책 + 지인 + KNOWS_POI 요약)
출력: List[Stage1Event] — 시간순 이벤트 시퀀스 (poi_id 없음, category + anchor만)

설계: docs/schedule_generation_plan/prompt.md §1
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Literal

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dawn_context import DawnContext, build_dawn_context  # noqa: E402
from llm_client import call_chat as _llm_call  # noqa: E402


# =========================================================
# Pydantic 검증
# =========================================================
try:
    from pydantic import BaseModel, Field, field_validator
except ImportError:
    print("[install] pydantic v2 필요. pip install pydantic")
    raise


class Stage1Event(BaseModel):
    time: str = Field(..., description="HH:MM 24시간제")
    anchor: str = Field(..., description="residence | workplace | zone:<dong_code>")
    category: str = Field(..., description="L1 카테고리 (식사·카페·…·집·직장)")
    sub_category: str | None = None
    intent: str = Field(..., description="짧은 의도 표현")
    pinned_poi: str | None = None
    with_agents: list[str] | None = None

    @field_validator("time")
    @classmethod
    def _check_time(cls, v):
        if not re.fullmatch(r"\d{2}:\d{2}", v):
            raise ValueError(f"time format HH:MM required, got {v!r}")
        hh, mm = int(v[:2]), int(v[3:])
        if not (0 <= hh < 24 and 0 <= mm < 60):
            raise ValueError(f"invalid time {v}")
        return v

    @field_validator("anchor")
    @classmethod
    def _check_anchor(cls, v):
        if v in {"residence", "workplace"}:
            return v
        if v.startswith("zone:") and len(v) > 5:
            return v
        raise ValueError(f"anchor must be residence|workplace|zone:<code>, got {v!r}")


class Stage1Output(BaseModel):
    events: list[Stage1Event]

    @field_validator("events")
    @classmethod
    def _check_events(cls, evs):
        if len(evs) < 3:
            raise ValueError(f"too few events ({len(evs)})")
        if len(evs) > 14:
            raise ValueError(f"too many events ({len(evs)})")
        # 첫·마지막은 residence
        if evs[0].anchor != "residence":
            raise ValueError("first event must anchor=residence")
        if evs[-1].anchor != "residence":
            raise ValueError("last event must anchor=residence")
        # 시간 단조 증가
        prev = -1
        for e in evs:
            cur = int(e.time[:2]) * 60 + int(e.time[3:])
            if cur <= prev:
                raise ValueError(f"non-monotonic time at {e.time}")
            prev = cur
        return evs


# =========================================================
# 프롬프트 빌더
# =========================================================
SYSTEM_PROMPT = """당신은 서울 시민 에이전트의 하루 동선을 설계하는 Daily Planner입니다.
출력은 반드시 주어진 JSON 스키마만 따르며, 자연어 해설을 덧붙이지 않습니다.

[이벤트 규칙]
- 하루 이벤트 수: 평일 5~9개, 주말/공휴일 3~8개.
- 첫 이벤트와 마지막 이벤트는 반드시 anchor='residence' (집).
- 평일 + 직장 있음: anchor='workplace' 체류가 09~18시 사이 누적 4시간 이상.
- 이벤트 간 최소 체류 20분.
- 시간은 24시간제 "HH:MM", 단조 증가.
- 카테고리 운영시간을 넘는 방문 금지.

[카테고리 어휘]
L1: 식사 · 카페 · 디저트 · 주점 · 편의점 · 마트 · 미용 · 쇼핑 · 여가 · 건강 · 교육 · 기타 · 집 · 직장
- anchor='residence'일 때 category='집' (식사 등 집 안에서의 활동도 '집'으로 표기 가능)
- anchor='workplace'일 때 category='직장'
- 외출 이벤트: category는 L1 어휘 중 선택, sub_category는 더 구체 (예: 한식·일식·헬스장 …)

[anchor 규칙]
- "residence": 거주지 (집 근처 외출 포함, anchor=residence + category=식사 등 가능)
- "workplace": 직장 근처
- "zone:<dong_code>": 거주·직장 외 특정 행정동으로 이동 (지인 약속·관광 등)

[정책·기억·소식 반영]
- 정책 대상 카테고리(혜택 환급) 방문은 페르소나 성향에 따라 가중. 소비분위 1~4는 민감, 9~10은 둔감.
- 어제 만족도 낮은 카테고리/장소는 회피.
- 지인 약속(appointment)이 있으면 해당 시간·장소(anchor=zone:<dong>, pinned_poi)에 강제 진입.

[pinned_poi]
- appointment의 meeting_poi_id가 있으면 해당 event에 pinned_poi 설정.
- 그 외엔 pinned_poi 생략 (POI 결정은 Stage 2에 위임).

[출력 형식]
다음 JSON 스키마만 출력. 다른 텍스트 금지.
{"events": [
  {"time":"08:10","anchor":"residence","category":"집","intent":"기상"},
  {"time":"08:50","anchor":"residence","category":"편의점","intent":"출근길 음료"},
  ...
]}"""


def _format_dawn_blocks(ctx: DawnContext, today: date, day_type: str) -> str:
    blocks = ctx.to_prompt_blocks()
    return f"""## 페르소나
{blocks['persona']}

## 어제 상태
{blocks['state']}

## 오늘
- 날짜: {today.isoformat()} ({_dow_kr(today)})
- 요일유형: {day_type}

## 최근 30일 기억 (Memory Top-N)
{blocks['memory']}

## 오늘 예정 약속
{blocks['appointment']}

## 거주·직장 동에 적용 정책
{blocks['policy']}

## 지인 풀
{blocks['social']}

## 사전 인지 POI 요약 (카테고리별)
{blocks['knows_poi']}

→ 위 정보로 오늘 하루 이벤트 시퀀스를 JSON으로 생성하세요. **JSON만 출력**. /no_think"""


_DOW_KR = ["월", "화", "수", "목", "금", "토", "일"]


def _dow_kr(d: date) -> str:
    return _DOW_KR[d.weekday()]


def _day_type(d: date) -> Literal["weekday", "weekend"]:
    return "weekend" if d.weekday() >= 5 else "weekday"


# =========================================================
# LLM 호출 (SGLang/vLLM auto-detect via llm_client) + 재시도
# =========================================================
# 모델·서버는 llm_client가 환경변수 LLM_MODE / SGLANG_BASE_URL로 동적 선택


def _extract_json(text: str) -> str:
    """LLM 응답에서 첫 JSON 객체 추출. <think> 블록 등 제거."""
    # <think>...</think> 제거
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # ```json ... ``` 블록 우선
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    # 첫 { 부터 마지막 } 까지
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        raise ValueError(f"no JSON object found in: {text[:200]}")
    return text[s : e + 1]


import os
_FAILURE_LOG = Path(os.environ.get("SIM_OUTPUT_DIR",
                                   os.path.expanduser("~/sim_output"))) / "stage1_failures.jsonl"
_FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)


def call_stage1(
    aid: str,
    today: date,
    ctx: DawnContext | None = None,
    max_retry: int = 2,
    verbose: bool = False,
    log_failures: bool = True,
) -> tuple[Stage1Output, dict]:
    """1명 1일 Stage 1 호출. (검증된 출력, 메타) 반환.

    log_failures=True: 검증 실패한 첫 시도 raw 응답을 jsonl에 append.
    """
    if ctx is None:
        ctx = build_dawn_context(aid, today)

    day_type = _day_type(today)
    user_block = _format_dawn_blocks(ctx, today, day_type)

    last_err = None
    last_raw = None
    for attempt in range(max_retry + 1):
        temp = 0.7 + 0.2 * attempt
        try:
            resp = _llm_call(
                None, SYSTEM_PROMPT, user_block,
                temperature=temp, max_tokens=1200,
            )
            raw = resp.choices[0].message.content
            last_raw = raw
            finish = resp.choices[0].finish_reason
            if verbose:
                print(f"--- attempt {attempt} (temp={temp}, finish={finish}) ---")
                print(raw[:500])
            json_str = _extract_json(raw)
            data = json.loads(json_str)
            parsed = Stage1Output.model_validate(data)
            meta = {
                "attempt": attempt,
                "temp": temp,
                "tokens_in": resp.usage.prompt_tokens,
                "tokens_out": resp.usage.completion_tokens,
            }
            return parsed, meta
        except Exception as e:
            last_err = e
            if log_failures and last_raw is not None:
                try:
                    with _FAILURE_LOG.open("a", encoding="utf-8") as fp:
                        fp.write(json.dumps({
                            "aid": aid, "day": today.isoformat(),
                            "attempt": attempt, "temp": temp,
                            "error_type": type(e).__name__,
                            "error": str(e)[:300],
                            "finish_reason": finish if 'finish' in dir() else None,
                            "raw_excerpt": raw[:800] if last_raw else "",
                        }, ensure_ascii=False) + "\n")
                except Exception:
                    pass
            if verbose:
                print(f"[attempt {attempt}] failed: {e}")

    raise RuntimeError(f"Stage1 failed after {max_retry+1} attempts: {last_err}")


# =========================================================
# CLI 테스트
# =========================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--aid", default="AGT_11110515_F_20대_001")
    ap.add_argument("--day", default="2026-05-01")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    today = date.fromisoformat(args.day)
    output, meta = call_stage1(args.aid, today, verbose=args.verbose)
    print("\n=== Stage 1 출력 ===")
    print(output.model_dump_json(indent=2))
    print("\n=== meta ===")
    print(meta)
