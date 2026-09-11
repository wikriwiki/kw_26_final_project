"""Stage 1 — 의도·카테고리·anchor 선택 LLM 호출.

입력: DawnContext (페르소나 + 어제 State + Memory + 약속 + 정책 + 지인 + KNOWS_POI 요약)
출력: List[Stage1Event] — 시간순 이벤트 시퀀스 (poi_id 없음, category + anchor만)

설계: docs/schedule_generation_plan/prompt.md §1
"""
from __future__ import annotations

import json
import re
import sys
import time
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
import prompts as _prompts  # noqa: E402
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
    "mood", "none",
}
_TRIGGER_ALIASES = {
    "habit":      "lifestyle",
    "life_style": "lifestyle",
    "life-style": "lifestyle",
    "routine":    "lifestyle",
}


# sub_category → L1 대분류 정규화 맵
# Stage 1 LLM이 세부업종('한식', '약국' 등)을 category로 출력하면 fallback fetch 실패 → 드롭.
# 아래 맵으로 대분류('식사', '건강')로 올려준다.
_CAT_TO_L1: dict[str, str] = {
    # 식사
    "한식": "식사", "중식": "식사", "일식": "식사", "양식": "식사",
    "분식": "식사", "패스트푸드": "식사", "뷔페": "식사", "도시락": "식사",
    "기타요식": "식사", "음식점": "식사", "점심": "식사", "저녁": "식사",
    # 카페
    "커피": "카페", "카페·커피": "카페",
    # 디저트
    "제과": "디저트", "베이커리": "디저트", "아이스크림": "디저트",
    "제과점": "디저트", "케이크": "디저트",
    # 건강
    "병원": "건강", "의원": "건강", "한의원": "건강", "약국": "건강",
    "치과": "건강", "일반병원": "건강", "피부과": "건강", "안과": "건강",
    "정형외과": "건강", "내과": "건강", "의료기관": "건강",
    # 마트
    "슈퍼마켓": "마트", "슈퍼": "마트", "할인점": "마트", "대형마트": "마트",
    "이마트": "마트", "홈플러스": "마트",
    # 미용
    "헤어샵": "미용", "미용실": "미용", "네일": "미용", "피부관리": "미용",
    "이발소": "미용", "헤어": "미용",
    # 여가
    "영화": "여가", "공연": "여가", "스포츠": "여가", "헬스": "여가",
    "수영": "여가", "운동": "여가", "볼링": "여가", "독서실": "여가",
    # 교육
    "학원": "교육", "과외": "교육", "학교": "교육", "보습": "교육",
    # 내구재 — 그래프 실제 Category: 쇼핑 아래 '가전·통신'·'가구'·'안경'
    "전자제품": "쇼핑", "가전제품": "쇼핑", "가전": "쇼핑", "가전·통신": "쇼핑",
    "휴대폰": "쇼핑", "컴퓨터": "쇼핑", "가구점": "쇼핑", "침구": "쇼핑", "안경점": "쇼핑",
    # 주점
    "술집": "주점", "바": "주점", "호프": "주점", "포차": "주점",
}

L1_CATEGORIES = {
    "식사", "카페", "디저트", "건강", "마트", "미용",
    "여가", "교육", "주점", "쇼핑", "편의점", "기타",
    "집", "직장",
}


def normalize_category(cat: str | None, sub: str | None = None) -> tuple[str | None, str | None]:
    """Stage 1 category 출력을 L1 대분류로 정규화.

    1. cat이 L1이면 그대로 통과
    2. 세부업종이면 _CAT_TO_L1 매핑으로 L1 승격 (원본 세부업종은 sub_category로 보존)
    3. 매핑에 없으면 cat 원본 유지 (Stage2 fallback Cypher가 Category.name으로 매칭 시도)
       → '기타' 강등으로 정보 손실 방지

    sub_category가 비어있고 cat이 L1이 아니면 cat을 sub로 복사해
    Stage2 candidate fetch가 세부업종 매칭 → L1 매칭 → district L1 매칭 순으로 시도 가능.
    """
    if not cat:
        return cat, sub
    if cat in L1_CATEGORIES:
        return cat, sub
    # 세부업종 → L1 매핑 (정확한 매핑 존재)
    mapped = _CAT_TO_L1.get(cat)
    if mapped:
        return mapped, sub or cat
    # 매핑 없음 — 원본 cat 유지, sub=cat 복사
    # 후처리 fallback에서 Category.name 매칭으로 처리하게 둠 (기타 강등 X)
    return cat, sub or cat


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
    trigger: str | None = Field(default=None, description="appointment | rumor | policy | lifestyle | mood | none")
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


# 낱말 태세 → 오늘 쓸 수 있는 결제 중 이 지갑으로 내는 비율.
# 연속값(grant_use)으로 물으면 계층 구분 없이 균일한 값이 나왔으므로(R84 전원 0.19 /
# R85 전원 0.87) 분기를 강제하는 범주로 묻고, 낱말의 뜻대로 비율을 해석한다.
GRANT_STYLE_MAP: dict[str, float] = {"빠짐없이": 1.0, "섞어서": 0.5, "가끔만": 0.18}


def grant_style_to_use(style: str | None) -> float | None:
    """낱말 태세를 비율로. 알 수 없는 값이면 None(태세 미지정)."""
    if not style:
        return None
    return GRANT_STYLE_MAP.get(str(style).strip())


class Stage1Output(BaseModel):
    events: list[Stage1Event]
    # 오늘 소비성향 p∈[0,1] — 평상 소비예산 대비 오늘의 소비 의향.
    # 정책지갑 인출률이 아니며, consumption 모델이 prior 밴드 안으로 클램프한다.
    daily_propensity: float | None = Field(default=None, description="오늘 소비성향 0~1")
    # 오늘 정책지갑(지원금) 사용의향 g∈[0,1] — 가진 지원금을 오늘 얼마나 적극 쓸지.
    # 유동성 제약이 큰 사람(쓸 현금이 빠듯한 사람)은 지원금이 미뤄둔 필요를 풀 여력이라
    # 높고, 현금 여유가 있고 사용기한이 넉넉하면 아껴 나눠 쓰므로 낮다. 지급액 크기나
    # 소득분위로 고정하지 말고 이 사람의 오늘 형편에서 판단한다. 지원금이 없으면 무의미.
    grant_use: float | None = Field(default=None, description="오늘 지원금 사용의향 0~1")
    # 낱말 선택형 태세. 연속값(grant_use)은 계층 구분 없이 균일한 값이 나왔으므로
    # (R84 전원 0.19 / R85 전원 0.87), 분기를 강제하는 범주로 묻는다.
    grant_style: str | None = Field(default=None, description="빠짐없이 | 섞어서 | 가끔만")

    # 남은 지원금을 앞으로 며칠에 걸쳐 쓸 생각인지(일). 사람이 실제로 하는 판단 형태.
    # 짧으면 오늘 많이, 길면 오늘 조금 쓰게 된다. consumption 모델이 이 계획을 그대로 따른다.
    grant_spread_days: int | None = Field(default=None, description="남은 지원금을 며칠에 걸쳐 쓸지")
    # 위 일수를 왜 그렇게 잡았는지 한 줄. 습관적으로 같은 숫자를 내지 않고 자기 상황을
    # 실제로 들여다보게 하는 장치(사고 흔적). 분석·인터뷰에도 쓰인다.
    grant_plan_reason: str | None = Field(default=None, description="그 일수로 잡은 이유 한 줄")
    # 쿠폰으로 결제해 굳은 현금 덕에 오늘 평소보다 더 쓰게 되는 정도(0~1).
    # 굳은 돈을 그대로 아끼면 0, 그 돈만큼 다른 데 더 쓰면 1에 가깝다.
    grant_extra_spend: float | None = Field(default=None, description="굳은 현금으로 더 쓰는 정도 0~1")
    # 쿠폰으로 굳은 현금 중 그대로 통장에 남는 몫(0~1). 유동성 제약이 큰 사람일수록 남지 않는다.
    # 지정되면 이 값이 우선하며 (1 - 남는 몫)이 오늘 추가 소비로 이어진다.
    grant_kept_share: float | None = Field(default=None, description="굳은 현금이 통장에 남는 몫 0~1")
    # 오늘 지출 중 집에서 주문해 배송으로 받는 몫(0~1). 가게 방문(events)이 없는 지출이며
    # 소비쿠폰은 온라인 결제에 쓸 수 없다(P010 사용처 조건). 이 채널이 없으면 하루 지출
    # 전액이 동네 가맹점으로 흘러 1인당 가맹점 지출이 실제보다 크게 잡힌다.
    online_share: float | None = Field(default=None, description="오늘 지출 중 배송으로 받는 몫 0~1")

    @field_validator(
        "daily_propensity", "grant_use", "grant_extra_spend", "grant_kept_share", "online_share"
    )
    @classmethod
    def _clip_propensity(cls, v):
        if v is None:
            return v
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return None

    @field_validator("grant_spread_days")
    @classmethod
    def _clip_spread(cls, v):
        if v is None:
            return v
        try:
            iv = int(round(float(v)))
        except (TypeError, ValueError):
            return None
        return max(1, min(180, iv))

    @field_validator("events")
    @classmethod
    def _check_events(cls, evs):
        # 이벤트 개수 제한 제거 (사용자 결정) — 최소 1개만 보장
        if len(evs) < 1:
            raise ValueError("no events")
        # 첫 이벤트가 residence 아니면 자동 보정 — 06:30 기상 event 앞에 prepend
        if evs[0].anchor != "residence":
            evs.insert(0, Stage1Event(
                time="06:30", anchor="residence", category="집",
                intent="기상", reasoning="자동 보정: 첫 이벤트 누락",
                trigger=None, pinned_poi=None,
            ))
        # 마지막 이벤트가 residence 아니면 자동 보정 — 23:30 취침 event append
        if evs[-1].anchor != "residence":
            # 마지막 이벤트 시간 이후로 — 23:30 보장 (불가능하면 23:59)
            last_min = int(evs[-1].time[:2]) * 60 + int(evs[-1].time[3:])
            tgt = "23:30" if last_min < 23*60 + 30 else "23:59"
            evs.append(Stage1Event(
                time=tgt, anchor="residence", category="집",
                intent="취침", reasoning="자동 보정: 마지막 이벤트 누락",
                trigger=None, pinned_poi=None,
            ))
        # 시간 단조 증가 — 위반 시 자동 보정 (이전 + 20분으로 강제)
        prev = -1
        for i, e in enumerate(evs):
            cur = int(e.time[:2]) * 60 + int(e.time[3:])
            if cur <= prev:
                # 자동 보정: 이전 시간 + 20분으로 밀기
                new_min = min(prev + 20, 23*60 + 59)
                evs[i] = e.model_copy(update={"time": f"{new_min//60:02d}:{new_min%60:02d}"})
                cur = new_min
            prev = cur
        return evs


# =========================================================
# 프롬프트 빌더
# =========================================================
# 소비행동 지시문은 정책군별 프롬프트 모듈에 있다 (scripts/sim/prompts/).
# P010 검증본이 기본이며, SIM_PROMPT_VARIANT 로 바꾼다.
SYSTEM_PROMPT = _prompts.get().SYSTEM_PROMPT


def _format_dawn_blocks(ctx: DawnContext, today: date, day_type: str) -> str:
    """Dawn 컨텍스트를 사용자 메시지로. 본문은 정책군별 프롬프트 모듈이 가진다."""
    blocks = ctx.to_prompt_blocks(today)
    return _prompts.get().format_dawn_blocks(
        blocks, today, day_type, _dow_kr(today)
    )


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
    total_started = time.perf_counter()
    timing: dict[str, float | int | list] = {
        "t_context_fetch": 0.0,
        "t_prompt_build": 0.0,
        "t_retry_prompt": 0.0,
        "t_llm": 0.0,
        "t_json_extract": 0.0,
        "t_json_parse": 0.0,
        "t_model_validate": 0.0,
        "t_category_normalize": 0.0,
        "t_rule_validate": 0.0,
        "t_failure_log": 0.0,
        "n_llm_calls": 0,
        "attempts": [],
    }

    if ctx is None:
        started = time.perf_counter()
        ctx = build_dawn_context(aid, today)
        timing["t_context_fetch"] = time.perf_counter() - started

    day_type = _day_type(today)
    started = time.perf_counter()
    user_block = _format_dawn_blocks(ctx, today, day_type)
    timing["t_prompt_build"] = time.perf_counter() - started

    last_err = None
    last_raw = None
    total_tokens_in = 0
    total_tokens_out = 0
    for attempt in range(max_retry + 1):
        temp = 0.7 + 0.2 * attempt
        attempt_timing: dict[str, float | int | str] = {"attempt": attempt}
        attempt_started = time.perf_counter()
        # retry 시 피드백 첨부 — LLM에게 직전 실수 알림
        started = time.perf_counter()
        user_block_now = user_block
        if attempt > 0 and last_err:
            user_block_now = user_block + (
                f"\n\n[직전 시도 검증 실패] {str(last_err)[:300]}\n"
                f"위 규칙을 어겼습니다. 이번엔 반드시 따를 것.\n"
            )
        elapsed = time.perf_counter() - started
        timing["t_retry_prompt"] += elapsed
        attempt_timing["t_retry_prompt"] = elapsed
        error_stage = "llm"
        try:
            started = time.perf_counter()
            resp = _llm_call(
                None, SYSTEM_PROMPT, user_block_now,
                temperature=temp, max_tokens=2200,
            )
            elapsed = time.perf_counter() - started
            timing["t_llm"] += elapsed
            timing["n_llm_calls"] += 1
            attempt_timing["t_llm"] = elapsed
            raw = resp.choices[0].message.content
            last_raw = raw
            finish = resp.choices[0].finish_reason
            tokens_in = int(getattr(resp.usage, "prompt_tokens", 0) or 0)
            tokens_out = int(getattr(resp.usage, "completion_tokens", 0) or 0)
            total_tokens_in += tokens_in
            total_tokens_out += tokens_out
            attempt_timing["tokens_in"] = tokens_in
            attempt_timing["tokens_out"] = tokens_out
            if verbose:
                print(f"--- attempt {attempt} (temp={temp}, finish={finish}) ---")
                print(raw[:500])

            error_stage = "json_extract"
            started = time.perf_counter()
            json_str = _extract_json(raw)
            elapsed = time.perf_counter() - started
            timing["t_json_extract"] += elapsed
            attempt_timing["t_json_extract"] = elapsed

            error_stage = "json_parse"
            started = time.perf_counter()
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                # Midm AWQ 특유 실수: `sub_category: "한식"` 처럼 key 앞 큰따옴표 누락 자동 보정
                import re as _re
                fixed = _re.sub(
                    r'(?<=[,\{\s\n])([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)',
                    r'"\1"\2', json_str
                )
                # 이미 큰따옴표로 감싸진 key는 이중 보정 방지
                fixed = _re.sub(r'""([a-zA-Z_][a-zA-Z0-9_]*)""', r'"\1"', fixed)
                data = json.loads(fixed)
            elapsed = time.perf_counter() - started
            timing["t_json_parse"] += elapsed
            attempt_timing["t_json_parse"] = elapsed

            error_stage = "model_validate"
            started = time.perf_counter()
            parsed = Stage1Output.model_validate(data)
            elapsed = time.perf_counter() - started
            timing["t_model_validate"] += elapsed
            attempt_timing["t_model_validate"] = elapsed

            # category 정규화 — 세부업종('한식') → L1('식사')
            error_stage = "category_normalize"
            started = time.perf_counter()
            for ev in parsed.events:
                norm_cat, norm_sub = normalize_category(ev.category, ev.sub_category)
                if norm_cat != ev.category:
                    ev.category = norm_cat
                if norm_sub != ev.sub_category:
                    ev.sub_category = norm_sub
            elapsed = time.perf_counter() - started
            timing["t_category_normalize"] += elapsed
            attempt_timing["t_category_normalize"] = elapsed

            # Post-validation: 평일 보수성 검증 (외출 의무)
            error_stage = "rule_validate"
            started = time.perf_counter()
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
            elapsed = time.perf_counter() - started
            timing["t_rule_validate"] += elapsed
            attempt_timing["t_rule_validate"] = elapsed
            attempt_timing["status"] = "ok"
            attempt_timing["t_total"] = time.perf_counter() - attempt_started
            timing["attempts"].append({
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in attempt_timing.items()
            })
            timing["t_total"] = time.perf_counter() - total_started

            meta = {
                "attempt": attempt,
                "temp": temp,
                "tokens_in": resp.usage.prompt_tokens,
                "tokens_out": resp.usage.completion_tokens,
                "tokens_in_total": total_tokens_in,
                "tokens_out_total": total_tokens_out,
                "s1_timing": {
                    k: (
                        round(v, 6)
                        if isinstance(v, float)
                        else v
                    )
                    for k, v in timing.items()
                },
                "prompt_timing": dict(ctx.prompt_timing),
            }
            return parsed, meta
        except Exception as e:
            # 현재 단계가 예외로 끝나도 그 단계에서 소비한 시간을 누락하지 않는다.
            stage_key = {
                "llm": "t_llm",
                "json_extract": "t_json_extract",
                "json_parse": "t_json_parse",
                "model_validate": "t_model_validate",
                "category_normalize": "t_category_normalize",
                "rule_validate": "t_rule_validate",
            }.get(error_stage)
            if stage_key and stage_key not in attempt_timing:
                elapsed = time.perf_counter() - started
                timing[stage_key] += elapsed
                attempt_timing[stage_key] = elapsed
                if error_stage == "llm":
                    timing["n_llm_calls"] += 1
            last_err = e
            attempt_timing["status"] = "error"
            attempt_timing["error_stage"] = error_stage
            attempt_timing["error_type"] = type(e).__name__
            attempt_timing["t_total"] = time.perf_counter() - attempt_started
            timing["attempts"].append({
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in attempt_timing.items()
            })
            if log_failures and last_raw is not None:
                started_log = time.perf_counter()
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
                timing["t_failure_log"] += time.perf_counter() - started_log
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
