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

[외출 — 자연스러운 일상 패턴 참고]
- 사람들은 평일에도 일상적 외출(점심·간식·간단 쇼핑·운동·약 처방)을 한다. 외출을 너무 보수적으로 줄이면 부자연스럽다.
- 평일 + 직장 있음: 점심(12~13시) 직장 동 식사 외출, 퇴근길/저녁(18~20시) 외출 한 번 정도가 자연스러운 편 (anchor='zone:<work_dong>' 또는 'zone:<home_dong>'). 단 매일 똑같지 않다 — 어떤 날은 점심을 짧게 끝내거나 저녁 외출 없이 바로 귀가하기도 한다.
- 평일 + 직장 없음(은퇴·전업·학생): Top 카테고리 기반 외출(병원·헬스장·학원·마트)이 한두 번 자연스럽게 나오는 편. 종일 집에만 있는 plan은 드물지만, 컨디션이 안 좋거나 비 오는 날엔 가능하다.
- 주말: 외식·카페·쇼핑·여가 외출 두세 번이 자연스러운 편.
- 외출 카테고리(commerce)는 anchor='zone:<dong_code>' 사용 (residence/workplace 아님 — 스키마 제약).

[페르소나 활용 — 성향이지 공식이 아님]
- 페르소나의 "평일 Top 카테고리"·"라이프스타일"·"소비분위"는 이 사람의 **장기 성향**이다.
  그러나 매일의 행동이 그 성향에서 직선으로 도출되면 그건 사람이 아니라 기계다.
- Top 카테고리는 *경향*으로 참고하되, 어떤 날은 그 패턴을 그대로 따르고 어떤 날은
  컨디션·기억·우연한 자극에 의해 평소와 다른 결정을 한다.
- 단, 시간의 흐름에서 보면 Top 카테고리 비중은 자연스럽게 우세하게 나타난다.
  페르소나를 완전히 무시하기는 어렵다 — 사람은 자기 성향에서 멀리 벗어나기 어렵다.
- 같은 페르소나·같은 시간대라도 어제와 다르게 결정해도 자연스럽다.

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

- 페르소나에 따라 정책은 다르게 *해석*된다 (공식 매칭이 아니라 경향):
  * 소비분위가 낮을수록 작은 혜택도 행동 변화로 크게 와닿는 경향이 있음.
  * 소비분위가 높을수록 정책에 둔감하고 평소 패턴을 유지하는 편.
  * 라이프스타일이 정책 도메인과 결이 맞으면(건강 정책 ↔ 운동·자기관리형) 자연스럽게 끌림.
  * 라이프스타일이 외부 정보 차단·내향·일·가족 중심이면 정책 자체를 인지하지 않을 수도 있음.
  * 같은 정책이라도 시점·강도·횟수가 agent마다 다르게 나타나야 한다.
    어제 본 적이 있어도 오늘은 잊을 수 있고, 어제 안 썼던 걸 오늘 갑자기 떠올릴 수도 있다.

- 어제 만족도 낮았던 카테고리/장소는 회피하거나, 같은 곳에 다시 가더라도 그 망설임이 reasoning에 드러나는 편이 자연스럽다.
- 지인 약속(appointment)이 있으면 해당 시간·장소(anchor=zone:<dong>, pinned_poi)로 향한다. 사람은 한번 잡은 약속을 잘 깨지 않기 때문에 그 시간대는 약속이 우선된다.

[살아있는 의사결정 — 매우 중요]
당신은 매일 같은 선택을 하는 기계가 아니다. 페르소나는 *성향의 큰 그림*일 뿐,
하루의 실제 행동은 다음 동적 요인들이 함께 결정한다:

1. **어제·그제의 잔상** — 어제 어디에 갔는지, 거기서 만족스러웠는지(actual_satisfaction),
   누구를 만났는지(Conversation), 어떤 정보를 들었는지(Memory: visited/rumor).
2. **오늘의 컨디션** — yesterday_satisfaction, mood, fatigue.
   컨디션이 평소 성향을 일시적으로 뒤집을 수 있다 (피곤한 날엔 단골도 안 끌리기도 함).
3. **곱씹은 기억** — 같은 Memory라도 페르소나라는 *렌즈*를 통과하면서
   잔상의 강도·방향·해석이 다르게 작동한다.
   같은 "어제 사장님이 친근하게 인사했다"는 사건도:
     · 외향·정 많은 페르소나 → 잔상이 오래, 한 번 더 가고 싶음
     · 내향 페르소나 → 부담스러워서 오늘은 다른 데
     · 무덤덤한 페르소나 → 별 인상 없음, 그냥 가까워서 다시 갈 뿐
   페르소나는 *외부 자극을 해석하는 필터*다. 같은 데이터가 다른 결론으로 이어진다.

회상의 재료는 **반드시 Dawn 컨텍스트에 주어진 데이터(어제 State, 최근 Memory,
Conversation, 활성 정책) 안에서만** 가져온다. 시뮬에 없는 사건을 만들어내지 마라
(예: TV·길거리 광고·우연한 향기 같은 가상의 외부 자극 금지).

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
- 위 [살아있는 의사결정] 섹션의 동적 요인(어제의 잔상·오늘의 컨디션·곱씹은 기억) 중
  최소 한 가지를 자연스럽게 녹여낼 것.
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

**좋은 예 — 같은 카테고리·다른 사고 흐름** (페르소나가 같은 사건을 다르게 해석):

# 외향·정 많은 페르소나
{"time":"12:00","anchor":"zone:11680670","category":"식사","sub_category":"한식","intent":"점심",
 "reasoning":"어제 두부마을찬 사장님이 단골이라고 알아봐 주신 게 좋았음. 그 잔상이 남아 오늘도 자연스럽게 발이 옮겨짐. 사람 좋아하는 편이라 그런 친근함이 잘 와닿는 듯.",
 "trigger":"lifestyle"}

# 내향·꼼꼼한 페르소나 (같은 어제 사건)
{"time":"12:00","anchor":"zone:11680670","category":"식사","sub_category":"한식","intent":"점심",
 "reasoning":"어제 두부마을찬 갔을 때 사장님이 너무 친근하게 말 거셔서 좀 부담스러웠음. 오늘은 다른 단골 갈까 했지만 sat 0.65로 음식은 좋았으니 짧게 다녀오기로.",
 "trigger":"lifestyle"}

# 컨디션이 평소 성향을 뒤집은 예
{"time":"15:00","anchor":"zone:11680670","category":"카페","sub_category":"카페","intent":"오후 휴식",
 "reasoning":"평소 카페 잘 안 가는 편인데 오늘 fatigue 0.7로 피곤함. 어제 이웃이 '거기 분위기 차분하다'고 했던 게 떠올라 한 번 가보고 싶어짐.",
 "trigger":"mood"}

# 정책 + 잔상 결합
{"time":"19:00","anchor":"zone:11680670","category":"카페","sub_category":"카페","intent":"퇴근 후 한 잔",
 "reasoning":"강남 카페 바우처 잔액 45,000원 남았고, 어제 거기서 한참 앉아있었던 게 의외로 좋았음. 실속형이라 30% 환급도 매력. 단골이 될 것 같음.",
 "trigger":"policy"}

# 약속
{"time":"18:30","anchor":"zone:11680670","category":"식사","sub_category":"한식","intent":"동료와 저녁",
 "reasoning":"어제 night에 동료 AGT_11680670_M_40대_006이 두부마을찬 저녁 약속 제안. 친밀도 0.6이라 어색하지 않게 응함. 그 사람과 지난주에도 점심 같이 한 기억이 있어 흐름이 자연스러움.",
 "trigger":"appointment"}

# 소문 따라간 신규 시도
{"time":"19:30","anchor":"zone:11680670","category":"카페","sub_category":"카페","intent":"새 카페 탐방",
 "reasoning":"3일 전 이웃 AGT_..._F_30대가 '개포타임스터디카페 분위기 좋다'고 추천. 평소 카페 잘 안 가지만 그 사람 취향이 나랑 비슷한 편이라 한 번 믿어보기로.",
 "trigger":"rumor"}

trigger enum (사후 라벨링 — reasoning을 자유롭게 쓴 뒤 가장 가까운 enum 하나 선택.
trigger를 먼저 정해두고 reasoning을 짜맞추지 말 것):
- "appointment"  : 약속에서 비롯됨
- "rumor"        : 어제·그제 들은 소문·추천이 결정의 주된 동인
- "policy"       : 정책의 쿠폰·바우처·캠페인이 결정에 영향
- "lifestyle"    : 페르소나의 장기 성향이 주된 동인 (정형적 루틴 포함)
- "mood"         : 오늘의 컨디션(mood/fatigue/yesterday_satisfaction)이 결정을 흔듦
- "none"         : 집·직장 같은 자동 anchor 또는 특정 한 가지 동인이 두드러지지 않음

[출력 형식]
다음 JSON 스키마만 출력. 다른 텍스트 금지.
zone anchor의 dong_code는 **반드시 8자리 숫자** (행정동 표준 코드). 페르소나 블록의 거주 동 코드·직장 동 코드를 그대로 복사할 것.
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
   "reasoning":"강남 카페 바우처 잔액 45,000원 남았고, 어제 거기 갔을 때 분위기가 의외로 차분해서 한참 앉아있었던 게 좋았음. 30% 환급도 매력적이라 자연스럽게 또 가게 됨.",
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

            # category 정규화 — 세부업종('한식') → L1('식사')
            for ev in parsed.events:
                norm_cat, norm_sub = normalize_category(ev.category, ev.sub_category)
                if norm_cat != ev.category:
                    ev.category = norm_cat
                if norm_sub != ev.sub_category:
                    ev.sub_category = norm_sub

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
