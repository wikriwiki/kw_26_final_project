"""Stage 1 — 의도·카테고리·anchor 선택 LLM 호출.

입력: DawnContext (페르소나 + 어제 State + Memory + 약속 + 정책 + 지인 + KNOWS_POI 요약)
출력: List[Stage1Event] — 시간순 이벤트 시퀀스 (poi_id 없음, category + anchor만)

설계: docs/schedule_generation_plan/prompt.md §1
"""
from __future__ import annotations

import json
import hashlib
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Literal

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dawn_context import DawnContext, build_dawn_context  # noqa: E402
import prompts as _prompts  # noqa: E402
from llm_client import call_chat as _llm_call  # noqa: E402
from prompt_grounding import validate_stated_reason


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
    evidence_ref: str | None = Field(default=None, description="입력에 표시된 근거 줄 번호")
    evidence_quote: str | None = Field(default=None, description="입력 맥락에서 그대로 인용한 근거 문구")
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
    # Validate optional appraisals after generation; malformed metadata must not retry an otherwise valid day plan.
    policy_appraisals: Any = Field(default_factory=list)
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
    from experience import prompt_block
    rendered = _prompts.get().format_dawn_blocks(
        blocks, today, day_type, _dow_kr(today)
    )
    rendered += prompt_block(ctx.state, today)
    if _prompts.active_name() == 'no_smoking_v1':
        rendered = _number_evidence_lines(rendered)
    return rendered


_EVIDENCE_LINE = re.compile(r'^\[(E\d{4})\] (.+)$')


def _number_evidence_lines(text: str, *, exclude_prefixes: tuple[str, ...] = ()) -> str:
    """Number factual input lines so the model can select a source without copying it."""
    result = []
    number = 0
    for line in text.splitlines():
        value = line.strip()
        citeable = (
            4 <= len(value) <= 1000
            and not value.startswith('##')
            and not value.startswith(('- 날짜:', '- 요일유형:'))
            and not value.startswith('위 입력에서 관련 근거를')
            and not value.startswith(exclude_prefixes)
        )
        if citeable:
            number += 1
            if number > 9999:
                raise ValueError('too many evidence lines')
            result.append(f'[E{number:04d}] {value}')
        else:
            result.append(line)
    return '\n'.join(result)


def _evidence_lines(user_text: str) -> dict[str, str]:
    result = {}
    for line in user_text.splitlines():
        match = _EVIDENCE_LINE.fullmatch(line)
        if match:
            result[match.group(1)] = match.group(2)
    return result


_DOW_KR = ["월", "화", "수", "목", "금", "토", "일"]


def _dow_kr(d: date) -> str:
    # 평일에 낀 공휴일이면 요일 옆에 이름을 붙인다(그 외 날은 렌더가 그대로). kr_holidays 참조.
    from kr_holidays import holiday_name
    name = holiday_name(d)
    return _DOW_KR[d.weekday()] + (f", {name} 공휴일" if name else "")


def _day_type(d: date) -> Literal["weekday", "weekend"]:
    from kr_holidays import day_type_of   # 공휴일은 주말과 같이 쉬는 날
    return day_type_of(d)


# =========================================================
# LLM 호출 (SGLang/vLLM auto-detect via llm_client) + 재시도
# =========================================================
# 모델·서버는 llm_client가 환경변수 LLM_MODE / SGLANG_BASE_URL로 동적 선택


def _extract_json(text: str) -> str:
    """Extract the first complete object, respecting braces inside quoted strings."""
    # <think>...</think> 제거
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start = text.find('{')
    if start < 0:
        raise ValueError(f"no JSON object found in: {text[:200]}")
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise ValueError('incomplete JSON object')


import os
_FAILURE_LOG = Path(os.environ.get("SIM_OUTPUT_DIR",
                                   os.path.expanduser("~/sim_output"))) / "stage1_failures.jsonl"
_FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)


class Stage1Exhausted(RuntimeError):
    """A single agent's bounded, feedback-driven Stage1 attempts are used up."""

    def __init__(self, attempts: int, reason: Exception | None):
        self.attempts = attempts
        super().__init__(f"Stage1 failed after {attempts} attempts: {reason}")


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
    system_prompt = _prompts.get().SYSTEM_PROMPT
    grounded_experiment = _prompts.active_name() == 'no_smoking_v1'
    retry_limit = max_retry
    started = time.perf_counter()
    user_block = _format_dawn_blocks(ctx, today, day_type)
    evidence_lines = _evidence_lines(user_block) if grounded_experiment else {}
    from grounded_schema import canonical_evidence_ref, rejected_response_feedback, stage1_format
    from execution_errors import fatal_dispatch_error
    anchors = None
    pinned_pois = None
    peers = None
    if grounded_experiment and hasattr(ctx, 'zone_candidates'):
        anchors = ['residence', *('zone:' + z['code'] for z in ctx.zone_candidates)]
        if ctx.persona.get('work_poi_id'):
            anchors.append('workplace')
        pinned_pois = [a['meeting_poi_id'] for a in ctx.appointment if a.get('meeting_poi_id')]
        peers = [aid for a in ctx.appointment for aid in (a.get('with_agents') or [])]
    response_schema = stage1_format(evidence_lines, anchors, pinned_pois, peers) if grounded_experiment else None
    timing["t_prompt_build"] = time.perf_counter() - started

    last_err = None
    last_raw = None
    total_tokens_in = 0
    total_tokens_out = 0
    for attempt in range(retry_limit + 1):
        previous_raw = last_raw
        temp = (0.2 if attempt == 0 else (0.1 if attempt < 3 else 0.3)) if grounded_experiment else 0.7 + 0.2 * attempt
        effective_temp = 0.0 if os.environ.get('POLICY_BACKTEST_DETERMINISTIC') == '1' else temp
        attempt_timing: dict[str, float | int | str] = {
            "attempt": attempt, "temp_requested": temp, "temp": effective_temp,
        }
        attempt_started = time.perf_counter()
        # retry 시 피드백 첨부 — LLM에게 직전 실수 알림
        started = time.perf_counter()
        user_block_now = user_block
        if attempt > 0 and last_err:
            if grounded_experiment:
                user_block_now += rejected_response_feedback(
                    previous_raw, last_err,
                    f'재시도 {attempt}/{retry_limit}입니다. 지적된 항목을 고친 뒤 전체 계획을 검증하세요. '
                    + 'evidence_ref는 쉼표 없이 입력의 E0001 형식 번호 하나만 쓰세요. '
                    + ('허용된 anchor는 ' + ', '.join(dict.fromkeys(anchors)) + '입니다. '
                       if anchors else '입력에 나온 anchor만 사용하세요. ')
                    + '그 조건에 맞춰 전체 JSON 계획을 새로 작성하세요.',
                )
            else:
                user_block_now += f'\n\n[직전 시도 검증 실패] {str(last_err)[:300]}\n위 규칙을 지키세요.'
        elapsed = time.perf_counter() - started
        timing["t_retry_prompt"] += elapsed
        attempt_timing["t_retry_prompt"] = elapsed
        error_stage = "llm"
        last_raw = None
        finish = None
        try:
            started = time.perf_counter()
            resp = _llm_call(
                None, system_prompt, user_block_now,
                temperature=temp, max_tokens=2200,
                **({'response_format': response_schema} if grounded_experiment else {}),
            )
            elapsed = time.perf_counter() - started
            timing["t_llm"] += elapsed
            timing["n_llm_calls"] += 1
            attempt_timing["t_llm"] = elapsed
            raw = resp.choices[0].message.content
            last_raw = raw
            finish = getattr(resp.choices[0], "finish_reason", None)
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
                # [뒤에 덧붙인 말을 버린다]
                # 모델이 완전한 JSON 을 낸 뒤 설명이나 두 번째 객체를 덧붙이면
                # "Extra data: line N" 으로 죽는다. 라이브에서 같은 시민이 이 이유로
                # 사흘 연속 실패해 하루치를 세 번씩 다시 돌렸다(벽시계 2배).
                # 첫 객체만 읽는다 — **모델의 판단을 바꾸지 않고** 덧붙인 말만 버린다.
                try:
                    data = json.JSONDecoder().raw_decode(json_str.lstrip())[0]
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
            if grounded_experiment:
                for event_index, event in enumerate(data.get('events', [])):
                    location = f'events[{event_index}] time={event.get("time")!r}: '
                    raw_ref = event.get('evidence_ref')
                    ref = canonical_evidence_ref(raw_ref, evidence_lines)
                    if not isinstance(ref, str) or ref not in evidence_lines:
                        raise ValueError(
                            location + f'evidence_ref {str(raw_ref)[:60]!r} must identify exactly one '
                            'numbered factual input line'
                        )
                    if ref != raw_ref:
                        attempt_timing['evidence_ref_format_repairs'] = (
                            attempt_timing.get('evidence_ref_format_repairs', 0) + 1
                        )
                    event['evidence_ref'] = ref
                    event['evidence_quote'] = evidence_lines[ref]
                    try:
                        validate_stated_reason(event, user_block)
                    except ValueError as exc:
                        raise ValueError(location + str(exc)) from exc
                    if anchors is not None and event.get('anchor') not in anchors:
                        raise ValueError(
                            location + f'anchor {str(event.get("anchor"))[:60]!r} must be one of the '
                            'supplied residence/workplace/zone values'
                        )
                    if pinned_pois is not None and event.get('pinned_poi') not in (None, *pinned_pois):
                        raise ValueError(location + 'pinned_poi must be a supplied appointment location')
                    companions = event.get('with_agents')
                    if peers is not None and companions is not None and (
                            not isinstance(companions, list)
                            or any(peer not in peers for peer in companions)):
                        raise ValueError(location + 'with_agents must contain only supplied appointment agent IDs; '
                                         + 'allowed=' + repr(sorted(set(peers))))
                    if not isinstance(event.get('time'), str) or not re.fullmatch(
                            r'([01][0-9]|2[0-3]):[0-5][0-9]', event['time']):
                        raise ValueError(location + 'time must be HH:MM from 00:00 to 23:59')
                raw_events = data.get('events', [])
                if (not raw_events or raw_events[0].get('anchor') != 'residence'
                        or raw_events[-1].get('anchor') != 'residence'):
                    raise ValueError('First and last events must use residence; no automatic uncited events')
                times = [int(ev['time'][:2])*60 + int(ev['time'][3:]) for ev in raw_events]
                if any(b <= a for a, b in zip(times, times[1:])):
                    raise ValueError('Event times must strictly increase')
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
            # The regulation experiment must permit staying home. A minimum
            # number of outings would impose behavior before measuring policy.
            # [2026-10-07 EXP_NO_MIN_OUTING] 정책 본런도 집에 머무는 하루를 허용한다(사용자 결정 — 이후 정책 전체).
            _free = grounded_experiment or os.environ.get("EXP_NO_MIN_OUTING", "0") == "1"
            min_events = 1 if _free else (6 if day_type == "weekday" else 4)
            min_zone = 0 if _free else (1 if (day_type == "weekday" or has_work) else 0)
            problems = []
            if n_events < min_events:
                problems.append(f"events={n_events} < min {min_events}")
            if n_zone < min_zone:
                problems.append(f"zone_anchor_events={n_zone} < min {min_zone}")
            if problems:
                raise ValueError(f"plan too conservative — {', '.join(problems)}")
            if os.environ.get("EXP_CLEAN_PROMPT", "0") == "1":
                # [2026-10-07b 상황 시험] 입력에 없는 계기를 지어내지 않는다: 들은 이야기(rumor 기억)가 없는데 trigger=rumor,
                # 오늘 약속이 없는데 trigger=appointment 면 다시 묻는다(첫날 '이웃에게 들었다'를 지어낸 사례).
                _shown_types = {str((m or {}).get("type") or "") for m in (ctx.memory or [])}
                _ungrounded = 0
                for _i, _e in enumerate(parsed.events):
                    _t = (getattr(_e, "trigger", None) or "")
                    _bad_t = ((_t == "rumor" and "rumor" not in _shown_types)
                              or (_t == "appointment" and not (ctx.appointment or [])))
                    if not _bad_t:
                        continue
                    if attempt < 2:
                        raise ValueError(f"events[{_i}]: trigger={_t} 인데 입력에 그 근거(들은 이야기 기억 / 오늘 약속)가 없다. "
                                         "입력에 없는 이야기·대화·약속을 만들지 말고 실제 계기로 고친다")
                    # 세 번째 시도부터는 그 사람의 하루를 실패시키지 않고 계기 표시만 바로잡고 센다(이유 문장은 남는다).
                    _e.trigger = "lifestyle"
                    _ungrounded += 1
                timing["trigger_ungrounded_fixed"] = _ungrounded
                # 같은 끼니를 두 번 먹지 않는다: 가게 식사 둘이 90분 안에 붙으면 다시 묻는다(약속 전에 따로 점심을 먹은 사례).
                # 세 번째 시도부터는 약속이 아닌 쪽을 뺀다.
                _meals = [e for e in parsed.events if e.category == "식사"]
                _mins = lambda e: int(e.time[:2]) * 60 + int(e.time[3:])
                _dup = [(a, b) for a, b in zip(_meals, _meals[1:]) if _mins(b) - _mins(a) < 90]
                if _dup:
                    if attempt < 2:
                        a, b = _dup[0]
                        raise ValueError(f"식사 {a.time}·{b.time} 이 90분 안에 두 번이다 — 같은 끼니를 두 번 먹지 않는다"
                                         "(약속이 끼니 때면 그 자리에서 먹는다)")
                    _drop = {id(a if (getattr(b, "trigger", None) == "appointment") else b) for a, b in _dup}
                    parsed.events = [e for e in parsed.events if id(e) not in _drop]
                    timing["double_meal_dropped"] = len(_drop)
                # 약속 장소가 정해진 약속은 그 가게로 고정한다(합의된 사실 — 한쪽만 다른 가게로 가던 사례).
                _pinned = 0
                for _ap in (ctx.appointment or []):
                    _pid = _ap.get("meeting_poi_id")
                    if not _pid:
                        continue
                    _tt = str(_ap.get("target_time") or "")[:5]
                    _cands = [e for e in parsed.events
                              if (getattr(e, "trigger", None) or "") == "appointment" and str(e.anchor).startswith("zone:")]
                    if _cands and _tt[:2].isdigit():
                        _tm = int(_tt[:2]) * 60 + int(_tt[3:5] or 0)
                        _cands.sort(key=lambda e: abs(int(e.time[:2]) * 60 + int(e.time[3:]) - _tm))
                    if _cands and not _cands[0].pinned_poi:
                        _cands[0].pinned_poi = _pid
                        _pinned += 1
                timing["appointment_pinned"] = _pinned
                # 이유·의도가 '집에서' 하는 일이라고 적었는데 가게 업종으로 분류된 일정은 집으로 바로잡는다
                # ('집에서 책 읽기'가 여가로 분류돼 여행사에 붙은 사례). 집 근처·집에서 나와 같은 말은 제외한다.
                _home_fix = 0
                for _e in parsed.events:
                    if _e.category in ("집", "직장"):
                        continue
                    _home = r"집에서|집 안에서|자택에서|집안에서"
                    _near = r"집에서 (가까|나와|나가|출발|멀|걸어|가는)|집 근처|집 앞"
                    _out = r"방문|식당|외식|가게|들러|들름|들렀|가서|사러|매장|구매|주문해 받|포장"
                    _in_intent = re.search(_home, _e.intent or "") and not re.search(_near, _e.intent or "")
                    _in_reason = (re.search(_home, _e.reasoning or "") and not re.search(_near, _e.reasoning or "")
                                  and not re.search(_out, _e.reasoning or ""))
                    if _in_intent or _in_reason:
                        _e.category, _e.anchor, _e.sub_category = "집", "residence", None
                        _home_fix += 1
                for _e in parsed.events:
                    # 재택근무를 직장으로 적은 일정은 집으로(재택근무는 집이라고 규칙에 적었는데도 직장으로 둔 사례).
                    if _e.category == "직장" and re.search(r"재택", _e.intent or "") and not re.search(r"출근", _e.intent or ""):
                        _e.category, _e.anchor, _e.sub_category = "집", "residence", None
                        _home_fix += 1
                timing["home_activity_recategorized"] = _home_fix
                # 이번 달 이미 낸 학원비·회비를 '내러 가는' 일정은 외출이 아니다(사용자 지적 2026-10-07: '학원비 0원 ·
                # 어제 학원비 냈음' 같은 일정). 수업·운동처럼 이용하러 가는 일정은 남긴다.
                _mp = set((ctx.persona or {}).get("monthly_paid") or [])
                _pay_re = r"학원비|수강료|교습비|회비|등록비|납부|결제하러|이용료 결제|레슨비|원비"
                _use_re = r"수업|강의|레슨|운동|이용|픽업|데려|등원|하원"
                _before = len(parsed.events)
                parsed.events = [e for e in parsed.events if not (
                    (e.sub_category or "") in _mp
                    and re.search(_pay_re, e.intent or "")
                    and not re.search(_use_re, e.intent or ""))]
                timing["paid_monthly_dropped"] = _before - len(parsed.events)
            # [EXP_CLINIC_COOLDOWN] 최근 3일 안에 진료받은 사람의 새 진료 일정은 뺀다(다시 묻지 않는다 — 고집하면
            # 그 사람의 하루가 실패하므로). 뺀 수는 기록한다.
            _recent_clinic = (ctx.persona or {}).get("recent_clinic_days") or []
            if _recent_clinic:
                _before = len(parsed.events)
                parsed.events = [e for e in parsed.events
                                 if (getattr(e, "sub_category", None) or "") not in ("의원", "치과", "한의원")]
                timing["clinic_cooldown_dropped"] = _before - len(parsed.events)
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
                "prompt_sha256": hashlib.sha256((system_prompt + "\n" + user_block_now).encode("utf-8")).hexdigest(),
                "model_id": getattr(resp, "model", None),
                "attempt": attempt,
                "temp": effective_temp,
                "temp_requested": temp,
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
            if fatal_dispatch_error(e):
                raise
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
                            "attempt": attempt, "temp": effective_temp, "temp_requested": temp,
                            "error_type": type(e).__name__,
                            "error": str(e)[:300],
                            "finish_reason": finish,
                            "raw_excerpt": raw[:800] if last_raw else "",
                        }, ensure_ascii=False) + "\n")
                except Exception:
                    pass
                timing["t_failure_log"] += time.perf_counter() - started_log
            if verbose:
                print(f"[attempt {attempt}] failed: {e}")

    raise Stage1Exhausted(int(timing['n_llm_calls']), last_err)


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
