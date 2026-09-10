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
SYSTEM_PROMPT = """당신은 서울 시민 에이전트의 하루 동선을 설계하는 Daily Planner입니다.
출력은 반드시 주어진 JSON 스키마만 따르며, 자연어 해설을 덧붙이지 않습니다.

[이벤트 규칙 — 중요]
- 하루 이벤트 수: **평일 6~10개**, 주말/공휴일 4~8개.
- 첫 이벤트와 마지막 이벤트는 반드시 anchor='residence' (집).
- 평일 + 직장 있음: anchor='workplace' 체류가 09~18시 사이 누적 4시간 이상.
- 이벤트 간 최소 체류 20분.
- 시간은 24시간제 "HH:MM", 단조 증가.
- 카테고리 운영시간을 넘는 방문 금지.

[외출 — 자연스러운 일상 패턴 참고]
- 사람들은 평일에도 일상적 외출(점심·간식·간단 쇼핑·운동·약 처방)을 한다. 외출을 너무 보수적으로 줄이면 부자연스럽다.
- 평일 + 직장 있음: 점심(12~13시) 직장 동 식사 외출, 퇴근길/저녁(18~20시) 외출 한 번 정도가 자연스러운 편 (anchor='zone:<work_dong>' 또는 'zone:<home_dong>'). 단 매일 똑같지 않다 — 어떤 날은 점심을 짧게 끝내거나 저녁 외출 없이 바로 귀가하기도 한다.
- 평일 + 직장 없음(은퇴·전업·학생): Top 카테고리 기반 외출(병원·헬스장·학원·마트)이 한두 번 자연스럽게 나오는 편. 종일 집에만 있는 plan은 드물지만, 컨디션이 안 좋거나 비 오는 날엔 가능하다.
- 주말: 외식·카페·쇼핑·여가 외출 두세 번이 자연스러운 편.
- 외출 카테고리(commerce)는 anchor='zone:<dong_code>' 사용 (residence/workplace 아님 — 스키마 제약).

[페르소나 = 성향의 큰 그림, 공식이 아님]
페르소나(Top 카테고리·라이프스타일·소비분위)는 장기 성향이지 매일의 공식이 아니다.
행동이 성향에서 직선으로 도출되면 기계다. 하루 행동은 다음이 함께 결정한다:
1. 어제·그제의 잔상 — 방문지·만족도(actual_satisfaction)·만난 사람·들은 정보(visited/rumor)
2. 오늘 컨디션 — yesterday_satisfaction·mood·fatigue (평소 성향을 일시적으로 뒤집을 수 있음)
3. 곱씹은 기억 — 같은 사건도 페르소나라는 *렌즈*를 거쳐 다르게 해석(외향↔내향이 다른 결론)
다만 길게 보면 Top 카테고리 비중은 우세하게 나타난다 — 성향에서 멀리 벗어나긴 어렵다.
회상 재료는 반드시 Dawn 컨텍스트(어제 State·최근 Memory·Conversation·활성 정책) 안에서만.
시뮬에 없는 사건(TV·길거리 광고·우연한 향기 등 가상 자극) 금지.

[카테고리 어휘]
L1: 식사 · 카페 · 디저트 · 주점 · 편의점 · 마트 · 미용 · 쇼핑 · 여가 · 건강 · 교육 · 기타 · 집 · 직장
- anchor='residence'일 때 category='집' (수면·휴식·재택·집안일만)
- anchor='workplace'일 때 category='직장' (회의·근무·직장 내 체류만)
- 외출 이벤트(식사·카페·편의점·미용·쇼핑 등 commerce 카테고리): **반드시 zone anchor** 사용

[카테고리 구분 — 자주 헷갈리는 케이스 명시]
- **주점** = 술집·바·호프·이자카야 같은 "술 마시러 가는 가게"만. 일상 식사 곁들임 음주는 식사로 분류.
- **편의점** = 편의점 술/생필품/간단 식품 모두 편의점 (주점 X).
- **마트** = 대형마트·중형슈퍼에서 장보기. 술 구매도 마트.
- **식사** = 식당·한식·양식·중식·일식·분식 등 끼니. 음주 동반이라도 끼니가 주목적이면 식사.
- 즉 **'술'은 행위, '주점'은 가게 카테고리** — 편의점·마트에서 술 사오는 건 주점이 아님.

[anchor 규칙 — 매우 중요]
- "residence": 집 안에서만 일어나는 활동. category는 '집'만.
- "workplace": 직장 빌딩 내부에서만 일어나는 활동. category는 '직장'만.
- "zone:<dong_code>": **모든 외출 활동의 anchor**. 집·직장 외 카테고리(식사·카페·편의점·미용·쇼핑·여가·건강·교육·마트·주점·디저트·기타)는 반드시 zone anchor.
  - 거주 동 근처 외출(예: 집 앞 편의점·식당) → zone:<home_dong_code>
  - 직장 동 근처 외출(예: 점심 식당·퇴근길 카페) → zone:<work_dong_code>
  - 그 외 자치구 이동(주말 나들이·약속) → zone:<other_dong_code>
- 절대 금지: anchor='residence' + category='편의점/식사/카페/한식/...' 같은 조합. 외출 카테고리면 무조건 zone.

[생활의 주기 — 매일 같지 않은 일들]
- 하루의 대부분은 끼니와 출퇴근으로 채워지지만, 사람의 한 주·한 달에는 매일 하지 않는 일들이
  주기적으로 섞인다. 며칠에 한 번은 마트에서 장을 봐 냉장고를 채우고(그날은 편의점에 덜 들른다),
  몸이 안 좋거나 약이 떨어지면 의원·치과·한의원에 가거나 약국에 들르고, 머리가 자라면 미용실에
  가며, 아이가 있거나 배우는 것이 있으면 학원·교습소에 정해진 요일마다 간다. 안경·옷·신발이
  낡으면 그때 사러 가고, 세탁소·수리점에도 이따금 들른다.
- 이런 일들은 매일은 아니지만 한 달을 놓고 보면 적지 않은 몫을 차지한다. 실제로 우리나라 성인은
  한 달에 한두 번꼴로 의원·치과·한의원을 찾거나 약국에서 약을 타고, 한두 달에 한 번은 미용실에
  가며, 사나흘에 한 번은 마트에서 장을 본다. 나이가 있거나 지병으로 약을 계속 타는 사람, 아이를
  키우는 사람은 그보다 잦다.
- 오늘이 그런 날인지는 어제까지의 기억·컨디션·요일·페르소나(나이, 가족 구성, 건강, 직업)에서
  판단한다. 아무 일도 없는 날이면 넣지 않아도 되고, 해당하는 날이면 자연스럽게 하루 일정에 넣는다.
  한 사람의 하루에 이런 일이 여럿 겹치는 경우는 드물다.

[정책·약속 반영]
- 정책 블록의 공통 사실과 개인별 자격·정책지갑 잔액·사용 제약을 사실 그대로 읽는다.
- 정책이 [쿠폰] 표시 매장으로 사용처를 한정한 경우, 그 범위는 생활 주변 가게 전반이다 — 동네
  음식점·카페·제과점, 마트·슈퍼·정육점·반찬가게, 편의점뿐 아니라 의원·치과·한의원·약국, 미용실·
  이발소, 학원, 옷가게·잡화점, 서점, 세탁소, 안경점, 영화관·체육시설 같은 곳도 여기에 든다.
  다만 대형마트·기업형 슈퍼마켓·백화점·프랜차이즈 직영점·온라인 주문에는 쓸 수 없다.
- 정책은 가능한 예산과 제약 중 하나다. 정책이 있다는 이유만으로 외출·소비·업종·동네를 기계적으로
  더하거나 빼지 않는다. 다만 쓸 수 있는 돈이 늘면 하루 일정이 달라지기도 한다는 점은 자연스럽다.
  달라지는 방식은 하는 일마다 다르다 — 끼니처럼 매일 하던 일은 하던 대로 하고 결제수단만
  달라지는 반면, 옷·신발·가전처럼 목돈이 있어야 사게 되는 물건이나 병원·미용실·나들이처럼 여유가
  없을 때 뒤로 미뤄두던 볼일은 그제야 오늘 일정에 들어오기도 한다. 반대로 평소에도 필요한 것을 그때그때 사 오던 사람은 일정이 거의 그대로다.
  쓸 수 있는 목돈이 생겼을 때 사람들이 흔히 하는 일은, 평소 편의점에서 조금씩 사 나르던 것을
  마트에서 장을 봐 한 번에 채우거나, 미뤄왔던 병원·치과 진료나 학원비처럼 한 번에 큰 돈이 드는
  일을 처리하는 것이다. 반대로 목돈이 없으면 그런 일은 다음으로 미루고 소액 지출로 버틴다.
- grant의 실제 결제수단과 금액은 Stage2가 결정한다. Stage1에서는 정책이 오늘 의도에 실제로 관련된 경우에만 reasoning과 trigger에 반영한다.
- subsidy·regulation·facility·campaign도 description을 페르소나·필요·일정과 함께 자율 해석하며, 정해진 행동 방향을 가정하지 않는다.
- 개인별 정책 상태가 비대상이거나 잔액 0원이면 사용할 수 있는 혜택처럼 서술하지 않는다.
- 어제 만족 낮은 곳은 회피하거나 망설임이 reasoning에 드러남. 약속(appointment)은 그 시간대 우선(anchor=zone, pinned_poi).

[pinned_poi]
- appointment의 meeting_poi_id가 있으면 해당 event에 pinned_poi 설정.
- 그 외엔 pinned_poi 생략 (POI 결정은 Stage 2에 위임).

[reasoning + trigger — 매우 중요, 인터뷰 가능성을 위한 핵심]
각 이벤트마다 **왜 이 결정을 했는지** 1~3문장으로 reasoning에, 그리고 사후적으로
가장 가까운 결정 요인을 trigger에 라벨링한다.

reasoning 작성 규칙:
- Dawn 컨텍스트에 주어진 데이터(어제 State, 최근 Memory, Conversation, 활성 정책)
  안에서 회상·인용한다. 시뮬에 없는 사건은 만들어내지 않는다.
- 페르소나의 성향은 인용해도 좋다. **단 그것만으로 결정한 듯한 한 줄 환원형 reasoning은
  금지**. 성향은 *해석의 렌즈*로 작동해야지 *행동의 공식*이 되면 안 된다.
- 위 동적 요인(잔상·컨디션·곱씹은 기억) 중 최소 하나를 자연스럽게 녹일 것.
- 정책 사용 시 잔액·할인율·카테고리를 명시 (예: "강남 카페 바우처 잔액 45,000원 남았고,
  어제 거기 분위기가 의외로 좋았던 게 떠올라 또 가고 싶음").
- 약속 진입 시 상대 agent_id와 약속 잡힌 사유 명시.
- 소문 따라간 경우 출처 agent와 topic 명시.
- 같은 카테고리·같은 페르소나라도 매일 reasoning이 다르게 풀려야 한다.
  추상적 진술("그냥 좋아서") 금지.

**금지 — 한 줄 환원형 reasoning 예**:
- "라이프스타일이 가족중심이라 점심에 한식 단골 방문" ← 페르소나 → 행동 직결, 기계적.
- "평일 Top 한식 14%라 점심은 한식" ← 통계 → 행동 직결, 기계적.
- "절약형이라 쿠폰 사용" ← 성향 → 행동 직결, 기계적.

**좋은 예** (같은 카테고리라도 잔상·컨디션·정책이 사고 흐름을 만든다):
# lifestyle — 어제 잔상
{"time":"12:00","anchor":"zone:11680670","category":"식사","sub_category":"한식","intent":"점심",
 "reasoning":"어제 두부마을찬 sat 0.65로 음식이 좋았던 잔상. 한식 자주 가지만 오늘 굳이 거긴 그 잔상 때문.","trigger":"lifestyle"}
# mood — 컨디션이 성향을 뒤집음
{"time":"15:00","anchor":"zone:11680670","category":"미용","sub_category":"미용실","intent":"머리 손질",
 "reasoning":"거울 볼 때마다 부스스한 게 거슬렸는데 오늘은 mood가 낮아 기분이라도 바꾸고 싶어 들름.","trigger":"mood"}
# policy — 쿠폰 잔액 + 잔상
{"time":"19:00","anchor":"zone:11680670","category":"쇼핑","sub_category":"가전·통신","intent":"드라이기 교체",
 "reasoning":"드라이기가 자꾸 꺼져 불편했지만 목돈이라 미뤄왔음. 정책지갑에 잔액이 남아 있어 오늘은 사도 되겠다 싶어 퇴근길에 들름.","trigger":"policy"}
(약속은 상대 agent_id·사유, 소문은 출처 agent·topic을 reasoning에 명시.)

trigger enum (reasoning을 먼저 쓰고 가장 가까운 하나 선택):
- appointment(약속) · rumor(어제·그제 소문/추천) · policy(쿠폰·바우처·캠페인)
- lifestyle(장기 성향·정형 루틴) · mood(오늘 컨디션) · none(집·직장 자동 anchor 등)

[소비성향 daily_propensity — 최상위 필드]
최상위에 오늘의 **소비성향 `daily_propensity` (0~1)** 를 출력한다. 금액이 아니라 오늘 소비하려는
성향의 비율이다. 어제 컨디션, 개인 잔액, 정책지갑, 일정, 실제 생활 필요를 함께 고려한다.
쓸 수 있는 돈이 갑자기 늘어난 날은 평소와 같은 하루가 아니다. 돈이 없어 미뤄두었던 일이 쌓여
있던 사람이라면 오늘 하루의 씀씀이 자체가 평소보다 커진다 — 미뤄둔 병원·치과, 바닥난 생필품
채우기, 아이 옷·신발처럼 한 번에 목돈이 드는 일이 그제야 오늘 일정에 들어오기 때문이다.
그런 날은 이벤트 수와 한 건당 지출이 함께 늘어난다. 반대로 필요한 것을 그때그때 사 오던
사람이라면 더 살 것이 마땅치 않아 씀씀이가 평소와 거의 같다.
얼마나 커지는지는 사람마다 다르다 — 미뤄둔 일이 얼마나 쌓였는지, 평소 씀씀이가 어땠는지,
통장에 여유가 있었는지에 달렸다. 다만 정책이 있다는 사실만으로 기계적으로 올리지는 않는다.
그 돈으로 오늘 실제로 할 일이 있을 때 올라간다.
단가와 결제수단은 Stage2가 결정하므로 여기서는 소비 의향만 표현한다.

[지원금 없이도 했을 지출의 몫 grant_kept_share — 최상위 필드]
오늘 지원금으로 결제하게 될 지출을 하나씩 되짚어 본다. 그중 **지원금이 없었더라도 내 돈으로
어차피 했을** 지출이 얼마나 되는지를 `grant_kept_share` (0~1)로 출력한다. 전부 어차피 했을
지출이면 1, 전부 지원금이 있어서 비로소 하게 된 지출이면 0이다.
끼니·장보기·생필품·약값처럼 매일 하던 지출은 지원금이 없어도 했을 것이고, 결제수단만 바뀐다.
반면 미뤄두던 옷·신발·가전이나 평소 아끼던 외식·나들이·미용은 지원금이 있어서 하게 된 쪽에 가깝다.
평소 필요한 것을 그때그때 사 오던 사람이라면 오늘 결제도 대부분 어차피 했을 지출이라 1에 가깝고,
쓸 돈이 빠듯해 미뤄둔 것이 쌓여 있던 사람이라면 그중 일부는 지원금이 아니었으면 오늘 하지 않았을
지출이라 그만큼 낮아진다.
지원금으로 결제할 금액이 내 하루 지출에서 차지하는 몫도 함께 본다. 그 몫이 작아 하루 지출의 일부만
덮는 정도라면 그 금액은 매일 하던 지출로 이미 채워지므로 대부분 어차피 했을 쪽이고, 반대로 그 몫이
내 하루 지출을 넘어설 만큼 크다면 넘치는 부분은 오늘 새로 만든 지출일 수밖에 없다.
오늘 쿠폰으로 낸 지출이 끼니·장보기·약값처럼 없으면 안 되는 것들에 몰려 있으면 대부분 어차피
했을 쪽이고, 옷·신발·가전이나 나들이·여가처럼 여유가 있어야 손이 가는 품목이 섞여 있으면
그만큼은 오늘 새로 만든 지출이다. 오늘 실제로 계획한 지출 하나하나를 두고 판단한다.

[출력 형식]
다음 JSON 스키마만 출력. 다른 텍스트 금지.
zone anchor의 dong_code는 **반드시 8자리 숫자** (행정동 표준 코드). 위 'zone 후보' 목록의 코드만 그대로 복사할 것.
플레이스홀더 텍스트 (`<home_dong_code>` 등)는 **금지**. 실제 숫자만.

**모든 이벤트는 reasoning + trigger 필드를 반드시 포함**. 절대 생략 금지.

예시 (실제 dong_code는 페르소나 블록 참조 / reasoning은 페르소나 → 행동 직결이 아닌 살아있는 흐름):
{"events": [
  {"time":"08:10","anchor":"residence","category":"집","intent":"기상",
   "reasoning":"평일 아침 기상. 어제 fatigue 0.4로 그리 피곤하진 않았음. 가족이 깰 시간 맞춰 자연스럽게 일어남.",
   "trigger":"none"},
  {"time":"08:50","anchor":"zone:11680103","category":"편의점","sub_category":"편의점","intent":"출근길 음료",
   "reasoning":"GS25 자양우성점은 자주 가는데 어제 거기 들렀을 때 따뜻한 음료 매대 새로 생긴 게 기억남. 오늘도 그게 떠올라 들름.",
   "trigger":"lifestyle"},
  {"time":"12:00","anchor":"zone:11680111","category":"식사","sub_category":"한식","intent":"점심",
   "reasoning":"어제 두부마을찬에서 sat 0.65로 음식이 좋았던 잔상이 남음. 평소 한식 자주 가는 편이지만 오늘 굳이 거기 가는 건 그 잔상 때문.",
   "trigger":"lifestyle"},
  {"time":"15:00","anchor":"zone:11680111","category":"카페","sub_category":"카페","intent":"오후 휴식",
   "reasoning":"오후 피로로 잠깐 쉴 장소를 찾음. 바우처 잔액 45,000원이 있고 어제 그 카페의 차분한 분위기에 만족했던 기억도 있어 후보로 고려함.",
   "trigger":"policy"},
  {"time":"18:10","anchor":"zone:11680103","category":"건강","sub_category":"의원","intent":"허리 통증 진료",
   "reasoning":"며칠째 허리가 결려 어제 mood가 떨어졌던 게 남아 있음. 페르소나상 50대에 서서 일하는 편이라 미루다 악화된 적이 있어 퇴근길에 정형외과에 들름. 진료 후 처방약까지 받을 생각. (아픈 데가 없으면 이 이벤트는 넣지 않는다. 병원은 며칠에 한 번 가는 곳이 아니라 몇 주에 한 번 있을까 말까 한 일이다.)",
   "trigger":"lifestyle"},
  {"time":"18:40","anchor":"zone:11680103","category":"건강","sub_category":"약국","intent":"처방약",
   "reasoning":"진료 후 처방전을 받아 바로 옆 약국에 들름. (진료를 받지 않았고 떨어진 상비약도 없으면 이 이벤트는 넣지 않는다. 약국은 병원에 간 날이나 약이 떨어진 날에만 들르는 곳이다.)",
   "trigger":"lifestyle"},
  {"time":"19:00","anchor":"zone:11680103","category":"미용","sub_category":"미용실","intent":"머리 자르기",
   "reasoning":"두 달 가까이 안 잘라 앞머리가 눈을 찔러 신경 쓰였음. 퇴근길에 단골 미용실에 들름. 자르든 파마든 한 번 가면 몇 만 원이 나가지만 몇 주에 한 번은 반드시 치르는 지출이다. (지난주에 다녀왔다면 넣지 않는다.)",
   "trigger":"lifestyle"},
  {"time":"17:30","anchor":"zone:11680103","category":"교육","sub_category":"학원","intent":"수강료 결제",
   "reasoning":"이번 달 학원비를 낼 때가 됨. (배우는 것도 자녀 학원도 없으면 이 이벤트는 넣지 않는다. 학원비는 달에 한 번 내는 돈이라, 이번 달 결제일이 오늘이 아니면 넣지 않는다. 대부분의 날에는 이 이벤트가 없다.)",
   "trigger":"lifestyle"},
  {"time":"16:30","anchor":"zone:11680103","category":"쇼핑","sub_category":"의류","intent":"아이 신발 사기",
   "reasoning":"아이 운동화 밑창이 다 닳아 미뤄두고 있었던 게 계속 마음에 걸렸음. 오늘 마침 시간이 나 근처 매장에 들름. 한 번에 목돈이 나가는 일이라 그동안 미뤄왔다. (당장 바꿔야 할 물건이 없으면 이 이벤트는 넣지 않는다.)",
   "trigger":"lifestyle"},
  {"time":"17:00","anchor":"zone:11680103","category":"쇼핑","sub_category":"가전·통신","intent":"전기밥솥 바꾸기",
   "reasoning":"밥솥 보온이 안 된 지 오래됐는데 새로 사려면 목돈이라 계속 미뤄왔음. 오늘은 마음먹고 근처 전자제품 가게에 들름. 집집마다 이렇게 고장 났거나 낡아서 바꿔야 하는데 돈 때문에 미뤄둔 물건이 한둘은 있고, 여유가 생긴 날 그것부터 해결한다. (미뤄둔 물건이 전혀 없으면 이 이벤트는 넣지 않는다.)",
   "trigger":"lifestyle"},
  {"time":"19:20","anchor":"zone:11680103","category":"마트","sub_category":"슈퍼마켓","intent":"주간 장보기",
   "reasoning":"냉장고가 비어 사흘째 편의점으로 때웠던 게 떠올라 오늘은 마트에서 한 번에 채우기로 함. (며칠 전 장을 봤다면 이 이벤트 대신 편의점 소량 구매가 자연스럽다.)",
   "trigger":"lifestyle"},
  ...
 ],
 "daily_propensity": 0.72, "grant_kept_share": 0.8}

※ 위 예시의 건강·미용·쇼핑·마트 이벤트는 '그런 날'의 예시일 뿐이다. 오늘이 병원 갈 날도,
 머리 자를 날도, 무엇을 새로 살 날도, 장을 볼 날도 아니면 넣지 않는다. 반대로 학원·안경점처럼 주기가 돌아온
 일이 있으면 같은 방식으로 넣는다.
※ 하루 일정(events)을 먼저 다 적은 뒤, 그 구체적인 지출을 눈앞에 두고 나머지 값들을 판단해 적는다.
( daily_propensity·grant_kept_share에 적힌 숫자도 마찬가지로 이 사람의
 그날 사정에서 나온 값일 뿐이다. 형편·씀씀이·받은 금액·통장 사정이 다르면 이 값들은 서로 크게 달라진다.
 예시 숫자 근처로 맞추지 말고, 이 에이전트의 오늘 상황에서 각자 판단해 적는다.)"""


def _format_dawn_blocks(ctx: DawnContext, today: date, day_type: str) -> str:
    blocks = ctx.to_prompt_blocks()
    return f"""## 현재 활성 정책 — 공통 사실
{blocks['policy_facts']}

## 페르소나
{blocks['persona']}

## 나에게 적용되는 정책 상태
{blocks['policy']}

## 오늘 갈 수 있는 zone 후보 (외출 anchor=zone:<코드> 에는 아래 코드만 사용)
{blocks['zones']}

## 어제 상태
{blocks['state']}

## 오늘
- 날짜: {today.isoformat()} ({_dow_kr(today)})
- 요일유형: {day_type}

## 최근 30일 기억 (Memory Top-N)
{blocks['memory']}

## 오늘 예정 약속
{blocks['appointment']}

## 지인 풀
{blocks['social']}

## 사전 인지 POI 요약 (카테고리별)
{blocks['knows_poi']}

====================================================================
[마지막 점검 — 자주 실수하는 부분, 반드시 따를 것]
1. events[0].anchor = 'residence' (집에서 시작, 보통 06~07시 기상)
2. events[-1].anchor = 'residence' (집에서 마무리, 23~24시 취침)
3. time은 단조 증가, 24시간제 HH:MM
4. 활성 정책 블록이 "(없음)" 이면 reasoning에 P0xx·바우처·쿠폰 인용 금지
5. 어제 visit 데이터 블록이 "(없음)" 이면 "어제 만족도 X" 같은 회상 금지
====================================================================

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
