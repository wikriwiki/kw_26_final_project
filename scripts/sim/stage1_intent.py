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


# ─── trigger 정규화 ───
# 행동 동기에서 '습관(habit)' 과 '라이프스타일(lifestyle)' 은 분리하지 않고
# **lifestyle 로 통일**. habit / life_style / routine 등은 모두 lifestyle 에 흡수.
# 페르소나의 lifestyle 필드는 별개의 트레잇으로 reasoning 에 인용될 뿐 — trigger 라벨과
# 의미가 겹치는 게 자연스럽다는 판단.
# 또한 LLM 이 가끔 환각으로 만들어내는 비표준 라벨(workplace, neighbor, campaign,
# health 등)은 정규화 시점에 'none' 으로 흡수.
CANONICAL_TRIGGERS = {
    "appointment", "rumor", "policy", "lifestyle",
    "top_category", "mood", "none",
}
_TRIGGER_ALIASES = {
    "habit":      "lifestyle",
    "life_style": "lifestyle",
    "life-style": "lifestyle",
    "routine":    "lifestyle",
}


def normalize_trigger(t):
    """트리거 라벨을 표준 enum 으로 정규화.

    - None / 빈 문자열 → None (변경 없음)
    - 'habit'/'life_style' → 'lifestyle'
    - 표준 enum 에 속하면 그대로
    - 그 외 비표준 (LLM 환각) → 'none'
    """
    if not t:
        return t
    t = str(t).strip().lower()
    if not t:
        return None
    if t in _TRIGGER_ALIASES:
        return _TRIGGER_ALIASES[t]
    if t in CANONICAL_TRIGGERS:
        return t
    return "none"


class Stage1Event(BaseModel):
    time: str = Field(..., description="HH:MM 24시간제")
    anchor: str = Field(..., description="residence | workplace | zone:<dong_code>")
    category: str = Field(..., description="L1 카테고리 (식사·카페·…·집·직장)")
    sub_category: str | None = None
    intent: str = Field(..., description="짧은 의도 표현")
    # ───────────────────── 사고과정 흔적 (인터뷰용) ─────────────────────
    # 왜 이 시간·카테고리·anchor를 골랐는지 페르소나·기억·정책·약속·소문 중
    # 무엇이 결정 요인인지 1~3문장으로. trigger는 아래 enum.
    reasoning: str | None = Field(default=None, description="이 이벤트를 선택한 이유 (1~3문장)")
    trigger: str | None = Field(default=None, description="appointment | rumor | policy | lifestyle | top_category | mood | none")
    # ──────────────────────────────────────────────────────────────────
    pinned_poi: str | None = None
    with_agents: list[str] | None = None

    @field_validator("trigger")
    @classmethod
    def _normalize_trigger(cls, v):
        return normalize_trigger(v)

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
        if len(evs) > 20:   # 외출 적극 권장 SYSTEM과 일치하도록 상한 완화 (14→20)
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

[이벤트 규칙 — 중요]
- 하루 이벤트 수: **평일 6~10개**, 주말/공휴일 4~8개.
- 첫 이벤트와 마지막 이벤트는 반드시 anchor='residence' (집).
- 평일 + 직장 있음: anchor='workplace' 체류가 09~18시 사이 누적 4시간 이상.
- 이벤트 간 최소 체류 20분.
- 시간은 24시간제 "HH:MM", 단조 증가.
- 카테고리 운영시간을 넘는 방문 금지.

[외출 적극 권장 — 매우 중요]
- 사람들은 평일에도 일상적 외출(점심·간식·간단 쇼핑·운동·약 처방)을 한다. **외출을 너무 보수적으로 줄이지 말 것**.
- 평일 + 직장 있음: 점심(12~13시) 직장 동 식사 외출 최소 1회 + 퇴근길/저녁(18~20시) 외출 1회 권장. anchor='zone:<work_dong>' 또는 'zone:<home_dong>'.
- 평일 + 직장 없음(은퇴·전업·학생): Top 카테고리 기반 외출(병원·헬스장·학원·마트) 1~2회 필수. 종일 집에만 있는 plan은 비현실적.
- 주말: 외식·카페·쇼핑·여가 외출 2~4회 자연스러움.
- 외출 카테고리는 반드시 anchor='zone:<dong_code>' (residence/workplace 아님).

[페르소나 Top 카테고리 활용 — 매우 중요]
- 페르소나의 "평일 Top 카테고리"·"주말 Top 카테고리"는 그 agent의 실제 소비 패턴.
- **Top 1~2위 카테고리(특히 30%+ 비중)는 일주일에 2~3회 plan에 포함**.
  - 예: Top "종합병원 78%" → 평일 중 1번 의료 외출.
  - 예: Top "학원 60%" → 자녀 학원 동반 외출.
  - 예: Top "헬스장 30%" → 운동 외출.
- 페르소나가 명확한 라이프스타일을 보이면 그 패턴을 plan에 반영.

[카테고리 어휘]
L1: 식사 · 카페 · 디저트 · 주점 · 편의점 · 마트 · 미용 · 쇼핑 · 여가 · 건강 · 교육 · 기타 · 집 · 직장
- anchor='residence'일 때 category='집' (수면·휴식·재택·집안일만)
- anchor='workplace'일 때 category='직장' (회의·근무·직장 내 체류만)
- 외출 이벤트(식사·카페·편의점·미용·쇼핑 등 commerce 카테고리): **반드시 zone anchor** 사용

[anchor 규칙 — 매우 중요]
- "residence": 집 안에서만 일어나는 활동. category는 '집'만.
- "workplace": 직장 빌딩 내부에서만 일어나는 활동. category는 '직장'만.
- "zone:<dong_code>": **모든 외출 활동의 anchor**. 집·직장 외 카테고리(식사·카페·편의점·미용·쇼핑·여가·건강·교육·마트·주점·디저트·기타)는 반드시 zone anchor.
  - 거주 동 근처 외출(예: 집 앞 편의점·식당) → zone:<home_dong_code>
  - 직장 동 근처 외출(예: 점심 식당·퇴근길 카페) → zone:<work_dong_code>
  - 그 외 자치구 이동(주말 나들이·약속) → zone:<other_dong_code>
- 절대 금지: anchor='residence' + category='편의점/식사/카페/한식/...' 같은 조합. 외출 카테고리면 무조건 zone.

[정책·기억·소식 반영]
- 정책 type별 메커니즘:
  * subsidy (쿠폰·환급): 정책 블록의 "남은 잔액 N원" 확인. 잔액 있으면 대상 카테고리 우선, 잔액 0원이면 일반 카테고리로 전환. 무한 사용 금지.
  * regulation (규제): 해당 카테고리·시간대 회피.
  * facility (시설): 해당 시설 방문 권장.
  * campaign (홍보): description 자율 해석 (예: "걷기 좋은 거리" → 도보 외출).

- 페르소나에 따라 정책 반응 정도가 달라야 함 (모든 agent가 동일하게 반응하면 안 됨):
  * 소비분위 1~4 (저소득·절약형): 쿠폰 매우 적극 활용, 작은 혜택도 행동 변화 큼.
  * 소비분위 5~8 (중산): 합리적 활용, 본인 선호 카테고리에 쿠폰 매칭되면 사용.
  * 소비분위 9~10 (고소득·소비형): 정책에 둔감. 쿠폰 있어도 평소 패턴 유지.
  * 라이프스타일에 "건강 우선·운동·자기관리" 키워드: 건강·교육 정책에 민감, 외식·주점 정책에는 무관심.
  * 라이프스타일에 "환경친화·미니멀·검소": environment·campaign 정책에 민감.
  * 라이프스타일에 "여가·문화·트렌드": 카페·쇼핑·여가 카테고리 정책에 적극.
  * **정책에 아예 관심 없는 페르소나도 있음** (예: 라이프스타일이 가족 중심·일 중심이고 외부 정보 차단 성향). 이 경우 정책 무시하고 평소 루틴 유지.
  * 같은 정책이라도 agent마다 사용 시점·강도·횟수 달라야 함 (예: 어떤 agent는 첫날 쿠폰 한 번에 소진, 어떤 agent는 며칠에 걸쳐 분산).

- 어제 만족도 낮은 카테고리/장소는 회피.
- 지인 약속(appointment)이 있으면 해당 시간·장소(anchor=zone:<dong>, pinned_poi)에 강제 진입.

[pinned_poi]
- appointment의 meeting_poi_id가 있으면 해당 event에 pinned_poi 설정.
- 그 외엔 pinned_poi 생략 (POI 결정은 Stage 2에 위임).

[reasoning + trigger — 매우 중요, 인터뷰 가능성을 위한 핵심]
각 이벤트마다 **왜 이 결정을 했는지** 1~3문장으로 reasoning에, 그리고 결정 요인 1개를 trigger에 적는다.

trigger enum (반드시 이 중 하나):
- "appointment"  : 약속 블록의 항목 때문 (다음날 만남으로 잡힌 약속)
- "rumor"        : 어제·그제 들은 소문/추천 때문 (Memory에 source/topic_value 있는 rumor 항목)
- "policy"       : 정책 블록의 쿠폰·바우처·캠페인 때문 (해당 카테고리·동 우선 방문)
- "lifestyle"    : 페르소나의 정형적 루틴 + 라이프스타일에서 비롯된 일상 패턴 (이 둘은 분리하지 않고 lifestyle 하나로 통일. 'habit' 같은 라벨도 lifestyle 로 정규화됨).
- "top_category" : 페르소나 Top 카테고리 강하게 반영 (lifestyle 보다 더 명시적, 예: 헬스장 30%)
- "mood"         : 어제 만족도·mood·fatigue 같은 컨디션 반영 (피곤해서 단축, 기분 좋아서 외출 추가)
- "none"         : 집·직장 같은 자동 anchor 이벤트 또는 특별 사유 없음

reasoning 작성 규칙:
- 페르소나의 **구체적 속성을 인용**한다 (예: "라이프스타일=가족중심 + 평일 Top 한식 14% → 점심 단골 한식집").
- 정책 사용 시 **잔액·할인율·카테고리**를 reasoning에 명시 (예: "강남 카페 바우처 잔액 45,000원 + 카페 좋아함 → 점심 후 카페 이용").
- 약속 진입 시 **상대 agent_id와 약속 잡힌 사유**를 명시 (예: "동료 AGT_..._40대_006이 어제 권유한 점심 약속").
- 소문 따라간 경우 **출처 agent와 topic**을 명시 (예: "이웃이 두부마을찬 평이 좋다고 함 → 처음 방문").
- 같은 카테고리라도 agent마다 reasoning이 달라야 함. **추상적 진술 금지** ("그냥 좋아서" X).

예시:
{"time":"12:00","anchor":"zone:11680670","category":"식사","sub_category":"한식","intent":"점심",
 "reasoning":"평일 Top 1위가 한식(14%)이고 어제 두부마을찬 sat 0.65로 만족했음. 직장 같은 동이라 도보 5분.",
 "trigger":"lifestyle"}

{"time":"15:00","anchor":"zone:11680670","category":"카페","sub_category":"카페","intent":"오후 휴식",
 "reasoning":"강남 여름 카페 바우처 잔액 45,000원 남았고, 라이프스타일이 '가족 중심·실속형'이라 30% 환급 매력적.",
 "trigger":"policy"}

{"time":"18:30","anchor":"zone:11680670","category":"식사","sub_category":"한식","intent":"동료와 저녁",
 "reasoning":"어제 night에 동료 AGT_11680670_M_40대_006이 두부마을찬 저녁 약속 제안. 친밀도 0.6이라 응함.",
 "trigger":"appointment"}

{"time":"19:30","anchor":"zone:11680670","category":"카페","sub_category":"카페","intent":"새 카페 탐방",
 "reasoning":"3일 전 이웃 AGT_..._F_30대가 '개포타임스터디카페 분위기 좋다'고 추천. KNOWS_POI에 없는 신규.",
 "trigger":"rumor"}

[출력 형식]
다음 JSON 스키마만 출력. 다른 텍스트 금지.
zone anchor의 dong_code는 **반드시 8자리 숫자** (행정동 표준 코드). 페르소나 블록의 거주 동 코드·직장 동 코드를 그대로 복사할 것.
플레이스홀더 텍스트 (`<home_dong_code>` 등)는 **금지**. 실제 숫자만.

**모든 이벤트는 reasoning + trigger 필드를 반드시 포함**. 절대 생략 금지.

예시 (실제 dong_code는 페르소나 블록 참조):
{"events": [
  {"time":"08:10","anchor":"residence","category":"집","intent":"기상",
   "reasoning":"평일 아침 기상. 라이프스타일=가족중심이라 출근 전 가벼운 식사 위해 일찍 일어남.",
   "trigger":"lifestyle"},
  {"time":"08:50","anchor":"zone:11680103","category":"편의점","sub_category":"편의점","intent":"출근길 음료",
   "reasoning":"평일 Top 카테고리에 편의점 5% 포함. 출근 전 음료 사는 루틴.",
   "trigger":"lifestyle"},
  {"time":"12:00","anchor":"zone:11680111","category":"식사","sub_category":"한식","intent":"점심",
   "reasoning":"평일 Top 1위 한식 14% + 어제 두부마을찬 만족도 0.65 좋았음. 직장 같은 동.",
   "trigger":"top_category"},
  {"time":"15:00","anchor":"zone:11680111","category":"카페","sub_category":"카페","intent":"오후 휴식",
   "reasoning":"강남 카페 바우처 잔액 45,000원 + 라이프스타일 실속형이라 30% 환급 매력.",
   "trigger":"policy"},
  ...
]}"""


def _format_dawn_blocks(ctx: DawnContext, today: date, day_type: str) -> str:
    blocks = ctx.to_prompt_blocks()
    # zone anchor에 쓸 실제 dong code를 명시적으로 추출 (LLM이 placeholder 출력 방지)
    home_dong = ctx.persona.get("home_dong_code") or ""
    work_dong = ctx.persona.get("work_dong_code") or ""
    dong_codes = f"- 거주 동 코드 (zone:으로 사용 시): {home_dong}\n"
    if work_dong:
        dong_codes += f"- 직장 동 코드 (zone:으로 사용 시): {work_dong}\n"
    return f"""## 페르소나
{blocks['persona']}

## zone anchor 코드 (반드시 이 값들 중 하나만 사용)
{dong_codes}
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
                temperature=temp, max_tokens=2200,  # reasoning 필드 추가로 출력량 ↑
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

            # Post-validation: 평일 보수성 검증 (외출 의무)
            has_work = bool(ctx.persona.get("work_poi_id"))
            n_events = len(parsed.events)
            n_zone = sum(1 for e in parsed.events if e.anchor.startswith("zone:"))
            min_events = 6 if day_type == "weekday" else 4
            min_zone = 1 if (day_type == "weekday" or has_work) else 0
            problems = []
            if n_events < min_events:
                problems.append(f"events={n_events} < min {min_events}")
            if n_zone < min_zone:
                problems.append(f"zone_anchor_events={n_zone} < min {min_zone}")
            if problems:
                raise ValueError(f"plan too conservative — {', '.join(problems)}")

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
