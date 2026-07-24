"""Stage 2 — Stage 1 이벤트 시퀀스 → 각 이벤트의 poi_id 결정.

입력:
  - Stage1Output (시간순 events)
  - DawnContext.persona (거주·직장 동 코드)
  - 각 이벤트별 candidate POI list (Cypher 사전 조회)

출력:
  Stage2Output — [{order, poi_id}, ...]. residence/workplace anchor는 home_poi/work_poi 그대로,
  pinned_poi가 있으면 그대로, 그 외는 LLM이 candidate 중 선택.

설계: docs/schedule_generation_plan/runtime_ontology.md §4.3
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dawn_context import (  # noqa: E402
    DawnContext, build_dawn_context,
    build_stage2_candidates,
    build_stage2_candidates_l1_dong,
    build_stage2_candidates_l1_district,
)
from stage1_intent import Stage1Output, call_stage1, _extract_json  # noqa: E402
from llm_client import call_chat as _llm_call  # noqa: E402
from poi_price import poi_price, price_icon, unit_price_anchor, band_factor  # noqa: E402
from coupon_eligibility import is_coupon_eligible  # noqa: E402
from poi_review_lookup import lookup_reviews_batch, format_review_block  # noqa: E402


try:
    from pydantic import BaseModel, Field
except ImportError:
    raise


class Stage2Pick(BaseModel):
    order: int
    poi_id: str
    actual_spent: float | None = None        # LLM이 설정 (원, 양수, 총 소비액)
    actual_satisfaction: float | None = None # LLM이 설정 (0~1)
    # actual_spent 중 정책 지원금에서 사용한 금액 — {"P009": 5000} 형태.
    # 평소 잔액으로 쓴 부분 = actual_spent - sum(policy_spend.values())
    policy_spend: dict[str, float] | None = None
    pick_reason: str | None = None
    pick_factor: str | None = None  # known | distance | satisfaction | rumor | appointment | random


class Stage2Output(BaseModel):
    picks: list[Stage2Pick]
    # LLM이 신중한 결정을 위해 별점·리뷰 추가 확인을 원하는 POI id 목록.
    # 비어 있거나 누락이면 첫 picks 그대로 채택. 채워져 있으면 별점·리뷰 첨부해서 한 번 재호출.
    review_lookup_requests: list[str] | None = None


# commerce 이벤트에 actual_spent가 0/None이면 카테고리·소득별 fallback 값 부여.
# 가급적 LLM이 직접 정하게 하되, 환각·누락 시 cap 추적 무력화 방지용 안전망.
_SPEND_FALLBACK_BY_L1 = {
    "편의점": 5000, "마트": 25000,
    "식사": 12000, "카페": 6000, "디저트": 8000, "주점": 30000,
    "미용": 30000, "쇼핑": 50000,
    "여가": 20000, "건강": 15000, "교육": 50000, "기타": 10000,
}


def _ensure_positive_spend(
    pick: "Stage2Pick", category: str | None, daily_wd: float | int | None,
    price_factor: float = 1.0,
    base_won: int | None = None,
    band: int | None = None,
) -> None:
    """LLM이 actual_spent 누락 / 0 / 음수로 출력했을 때 fallback 부여.

    track_policy_usage가 spend<=0 이면 cap 추적을 skip하므로,
    여기서 최소값을 강제해 정책 효과 측정 신뢰도 확보.

    base_won: 실측 단가 앵커(unit_price_anchor — 동계수 이미 포함). 있으면
      밴드 배율만 곱한다(동계수 이중계상 방지). 없으면 구 방식(표×price_factor).
    """
    cur = pick.actual_spent or 0
    if cur > 0:
        return
    if base_won:
        base = int(base_won * band_factor(category, band or 2))
    else:
        base = _SPEND_FALLBACK_BY_L1.get((category or "기타"), 10000)
        base = int(base * (price_factor or 1.0))
    if daily_wd and daily_wd > 0:
        # daily_wd가 매우 작은 경우 비율 보정 (예: 절약형 페르소나)
        base = min(base, int(daily_wd * 0.4))
    pick.actual_spent = max(1000, base)


# =========================================================
# Helper: Stage 1 이벤트 → 행정동 코드 결정
# =========================================================
def resolve_dong(event_anchor: str, persona: dict, stats: dict | None = None) -> str | None:
    """zone:DONG_CODE → dong code 추출. invalid면 persona fallback + stats 카운트."""
    if event_anchor == "residence":
        return persona.get("home_dong_code")
    if event_anchor == "workplace":
        return persona.get("work_dong_code")
    if event_anchor.startswith("zone:"):
        dong = event_anchor.split(":", 1)[1].strip()
        # Neo4j Dong 노드 코드는 8자리 (행정안전부 표준 8자리, KOSIS 코드).
        # 5자리(district)로 와도 fallback 처리(persona 동코드 사용).
        if dong.isdigit() and len(dong) == 8:
            return dong
        # placeholder/invalid — persona fallback
        if stats is not None:
            stats["resolve_dong_placeholder_fallback"] = stats.get("resolve_dong_placeholder_fallback", 0) + 1
        if "work" in dong.lower():
            return persona.get("work_dong_code") or persona.get("home_dong_code")
        return persona.get("home_dong_code")
    return None


# =========================================================
# 각 이벤트별 candidate 수집 (commerce만, residence/workplace/직장/집 제외)
# =========================================================
INTERNAL_CATS = {"집", "직장"}  # residence/workplace anchor에서 사용. POI 미고정


# 쿠폰(사용처 제한 지원금) 활성 시 사용 가능 매장의 정렬 보너스 — "매력도 재산출".
# 실제 소비쿠폰 기간에 사용가능 매장으로 수요가 이동하는 것을 후보 노출 순위로 반영.
COUPON_SORT_BONUS = 0.05


def _score_and_sort_by_desire(cands: list[dict], today: date, coupon_boost: bool = False) -> list[dict]:
    """avg_satisfaction 내림차순 → km 오름차순 정렬. coupon_boost 시 쿠폰가능 매장 가점."""
    for c in cands:
        sat = c.get("avg_satisfaction")
        c["desire"] = float(sat) if sat is not None else 0.0
        if coupon_boost and c.get("coupon_eligible"):
            c["desire"] += COUPON_SORT_BONUS
    # 1순위: avg_satisfaction 내림차순, 2순위: km 오름차순 (None은 뒤)
    cands.sort(key=lambda c: (-c["desire"], c.get("km") or 9999))
    return cands


def fetch_candidates_for_events(
    aid: str, events: list, persona: dict, today: date,
    k_per_event: int = 12,
    stats: dict | None = None,
    timing: dict | None = None,
) -> dict[int, list[dict]]:
    """이벤트별 candidate POI dict. key=order, value=list of candidate dicts.

    정렬 (단순화 — 2026-05-30):
      avg_satisfaction 내림차순 → km 오름차순.
      복잡한 desire 4요인 곱셈(affinity·recency·saturation·novelty)은 폐기.
      반복 억제는 Stage 2 프롬프트의 '최근 3일 방문 POI' 헤더로 LLM이 자율 처리.

    같은 날 반복 차단:
      같은 (dong, sub_category) 이벤트가 N개면 후보를 N×k_per_event 크기로 한 번에
      fetch + 정렬 후 round-robin 분할. 같은 POI가 두 이벤트 풀에 동시 등장 못 함.

    stats: fallback 카운트 dict (mutate). 누적 키:
      - resolve_dong_placeholder_fallback
      - cand_sub_match / cand_fallback_l1_dong / cand_fallback_l1_district / cand_all_empty
      - pool_split_groups : 분할이 일어난 그룹 수
      - pool_split_events : 분할 적용된 이벤트 수
    """
    from collections import defaultdict
    from neo4j_load._common import driver_session

    out: dict[int, list[dict]] = {}
    s = stats if stats is not None else {}
    tm = timing if timing is not None else {}
    tm.update({
        "t_group_resolve": 0.0,
        "t_query_exact": 0.0,
        "t_query_l1_dong": 0.0,
        "t_query_l1_district": 0.0,
        "t_enrich": 0.0,
        "t_sort_split": 0.0,
        "n_groups": 0,
        "n_query_exact": 0,
        "n_query_l1_dong": 0,
        "n_query_l1_district": 0,
    })

    # 1) 각 이벤트 → 그룹 키 (dong_code, sub_cat) 결정. 스킵은 즉시 빈 풀.
    group_started = time.perf_counter()
    group_key_for: dict[int, tuple[str, str]] = {}
    l1_for: dict[int, str] = {}
    for i, ev in enumerate(events):
        if ev.category in INTERNAL_CATS or ev.pinned_poi:
            out[i] = []
            continue
        sub_cat = ev.sub_category or _guess_sub_from_l1(ev.category)
        if sub_cat is None:
            out[i] = []
            continue
        dong_code = resolve_dong(ev.anchor, persona, stats=s)
        if not dong_code:
            out[i] = []
            continue
        group_key_for[i] = (dong_code, sub_cat)
        l1_for[i] = ev.category

    # 2) 같은 (dong, sub_cat) 그룹화. dict 삽입 순서 = 이벤트 시간 순.
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, key in group_key_for.items():
        groups[key].append(i)
    tm["t_group_resolve"] = time.perf_counter() - group_started
    tm["n_groups"] = len(groups)

    if not groups:
        return out

    # 3) 그룹별 fetch + round-robin 분할 (fallback 체인 그룹 단위 1회)
    #     [perf] agent-day의 모든 후보 조회를 단일 세션으로 — 그룹마다 세션 생성 제거.
    with driver_session() as sess:
        for (dong_code, sub_cat), event_idxs in groups.items():
            n = len(event_idxs)
            pool_size = k_per_event if n == 1 else n * k_per_event
            l1 = l1_for[event_idxs[0]]   # 같은 sub_cat ⇒ 같은 L1

            started = time.perf_counter()
            cands = build_stage2_candidates(aid, dong_code, sub_cat, limit=pool_size, session=sess)
            tm["t_query_exact"] += time.perf_counter() - started
            tm["n_query_exact"] += 1
            if cands:
                s["cand_sub_match"] = s.get("cand_sub_match", 0) + n
            else:
                if l1 and l1 not in INTERNAL_CATS:
                    started = time.perf_counter()
                    cands = build_stage2_candidates_l1_dong(aid, dong_code, l1, limit=pool_size, session=sess)
                    tm["t_query_l1_dong"] += time.perf_counter() - started
                    tm["n_query_l1_dong"] += 1
                    if cands:
                        s["cand_fallback_l1_dong"] = s.get("cand_fallback_l1_dong", 0) + n
                if not cands and l1:
                    district_code = dong_code[:5] if len(dong_code) >= 5 else None
                    if district_code:
                        started = time.perf_counter()
                        cands = build_stage2_candidates_l1_district(
                            aid, district_code, l1, limit=pool_size, session=sess,
                        )
                        tm["t_query_l1_district"] += time.perf_counter() - started
                        tm["n_query_l1_district"] += 1
                        if cands:
                            s["cand_fallback_l1_district"] = s.get("cand_fallback_l1_district", 0) + n
                if not cands:
                    s["cand_all_empty"] = s.get("cand_all_empty", 0) + n

            # POI 가격대 부착 (결정론, O(1)) — Stage2 프롬프트 표기·소비 반영용.
            # district fallback 후보는 자기 동 미상 → anchor 동 prior로 근사.
            # unit_anchor: 이 동네×업종 평균 결제단가(실측 기반) — 프롬프트 스케일 앵커.
            started = time.perf_counter()
            anchor_won = unit_price_anchor(dong_code, l1)
            # 사용처 제한 지원금(쿠폰) 잔액 보유 여부 — run_simulation이 persona에 세팅
            # 정책 사용 가능 여부는 후보 정보로만 제공한다. 후보 정렬 가점은 결과를
            # 사전 유도하므로 기본 0이며, 별도 민감도 실험에서만 명시적으로 켠다.
            coupon_active = bool(persona.get("coupon_poi_restricted"))
            coupon_boost = (
                coupon_active
                and os.environ.get("POLICY_POI_SORT_BOOST", "0") == "1"
            )
            for c in cands or []:
                c["price_band"], c["price_factor"] = poi_price(c["poi_id"], dong_code, l1)
                c["unit_anchor"] = anchor_won
                # 쿠폰 사용처 판정 — DB 백필값(p.coupon_eligible) 우선, 없으면 룰 fallback
                el = c.get("coupon_eligible")
                if el is None:
                    el = is_coupon_eligible(c.get("name"), sub_cat, l1)[0]
                c["coupon_eligible"] = bool(el)
                # 프롬프트 마커: 쿠폰 활성 시에만 표기 (평시 토큰 0)
                c["coupon_tag"] = "[쿠폰]" if (coupon_active and c["coupon_eligible"]) else ""
            tm["t_enrich"] += time.perf_counter() - started

            # desire 점수 계산 + 정렬 (분할·할당 전에 1회) — 쿠폰가능 매장 가점(매력도 재산출)
            started = time.perf_counter()
            cands = _score_and_sort_by_desire(cands or [], today, coupon_boost=coupon_boost)

            if n == 1:
                out[event_idxs[0]] = cands[:k_per_event]
            else:
                buckets = _split_pool_round_robin(cands, n, k_per_event)
                for bucket, ev_i in zip(buckets, event_idxs):
                    out[ev_i] = bucket
                if cands:
                    s["pool_split_groups"] = s.get("pool_split_groups", 0) + 1
                    s["pool_split_events"] = s.get("pool_split_events", 0) + n
            tm["t_sort_split"] += time.perf_counter() - started

    return out


def _split_pool_round_robin(
    cands: list[dict], n: int, k_per_event: int,
) -> list[list[dict]]:
    """정렬된 풀(avg_satisfaction DESC, km ASC)을 N개 이벤트 풀로 round-robin 분할.

    같은 POI 가 여러 버킷에 들어가지 않음 — 한 cand 는 idx % n 한 곳만 들어감.
    상위 → 하위 순서대로 라운드로빈이라 각 버킷이 만족도 분포를 골고루 받는다.
    가장 이른 시간 이벤트(idx 0)가 만족도 1순위를 받음.
    """
    buckets: list[list[dict]] = [[] for _ in range(n)]
    for idx, c in enumerate(cands):
        buckets[idx % n].append(c)
    return [b[:k_per_event] for b in buckets]


# L1 → 대표 sub 매핑 (sub_category 누락 시 fallback)
_L1_TO_SUB_DEFAULT = {
    "식사": "한식", "카페": "카페", "디저트": "베이커리", "주점": "일반주점",
    "편의점": "편의점", "마트": "슈퍼마켓", "미용": "미용실",
    "쇼핑": "의류", "여가": "노래방", "건강": "약국", "교육": "학원", "기타": "기타개인",
}


def _guess_sub_from_l1(l1: str) -> str | None:
    return _L1_TO_SUB_DEFAULT.get(l1)


# =========================================================
# 프롬프트 빌더
# =========================================================
SYSTEM_S2 = """당신은 에이전트의 오늘 외출 이벤트에 대해 구체적인 방문 장소(POI)를 결정하고,
소비 금액과 만족도를 설정하는 Daily Planner Stage 2입니다.

## 핵심 규칙

**POI 선택**
- 각 이벤트는 반드시 자기 자신의 candidates 풀에서만 선택합니다.
- 후보 ID를 절대 지어내지 마세요. 목록에 있는 poi_id만 사용합니다.
- order는 0-base 정수이고, 각 외출 이벤트(residence/workplace/pinned 제외) 모두에 정확히 1개의 pick을 만듭니다.
- 픽 누락 금지: events에 표시된 모든 외출 order 각각에 대해 반드시 1개의 pick을 생성합니다.
- 같은 order에 대해 중복 pick 금지.
- residence/workplace/집/직장 이벤트, pinned_poi 이벤트는 picks에 포함하지 않습니다.

**페르소나 기반 선택 (핵심)**
- 에이전트의 라이프스타일·성향·직업·생활 패턴을 고려해 자연스럽게 어울리는 장소를 선택합니다.
- 과거 만족도(avg_sat)가 높은 곳을 선호하되, 페르소나가 탐색형이면 새 곳도 도전합니다.
- avg_sat이 없는 신규 장소는 거리(km)가 가까운 곳을 우선합니다.

**가격대와 예산 (핵심)**
- 각 후보에는 가격대가 표시됩니다: ₩(저가) / ₩₩(중간) / ₩₩₩(고가).
- 이벤트 제목의 '동네 평균단가'는 그 동네·업종의 실제 카드 결제단가 기준 참고값입니다.
  actual_spent는 이 스케일에서 시작해 가격대·상황에 맞게 조정하세요 (₩는 그보다 낮게, ₩₩₩는 높게).
- 헤더의 잔액·평소 소비규모·소비성향에 맞는 가격대의 장소를 고르세요.
  잔액이 빠듯하거나 절약형이면 ₩ 위주로, 여유가 있거나 특별한 상황(기념일·약속 등)이면 ₩₩₩도 선택할 수 있습니다.
- actual_spent 단가는 선택한 POI의 가격대와 정합되게: 같은 카테고리에서 ₩₩₩는 ₩의 대략 1.5~2배.

**단순 반복 억제**
- 최근 3일 이내 방문한 POI(⚠️ 표시)는 특별한 사유 없이 재선택하지 마세요.
- 같은 날 여러 이벤트가 있을 때 동일 POI를 두 번 선택하지 마세요.

**소비액 설정 (actual_spent + policy_spend)**

`actual_spent` = 이 거래의 총 소비액 (양수). 얼마나 소비할지는 페르소나 성향대로.
`policy_spend` = 그 거래를 **어느 지갑으로 결제했는지** — 정부 지원금(grant)에서 낸 금액. `{"P009": 5000}` 형태.
제약: sum(policy_spend) ≤ actual_spent, 그리고 ≤ 지원금 잔액.

[지원금(grant) 회계와 제약]
- 지원금은 개인 잔액과 분리된 정책 지갑이다. 지원금으로 결제한 금액은 개인 잔액에서 차감되지 않고 정책 지갑에서 차감된다.
- 소비 필요·POI·actual_spent는 정책지갑 잔액만으로 만들지 말고 페르소나의 필요와 상황에 따라 정한다.
- 선택한 거래가 지원금 사용 가능 매장이면 정책지갑을 자기자금보다 먼저 결제한다. `policy_spend`에는 예상 결제액을 기록하되 최종값은 시스템이 사용처·거래액·잔액 범위에서 우선 정산한다.
- `sum(policy_spend) ≤ actual_spent`이고, 정책별 잔액을 넘을 수 없다.
- 사용처 제한 정책은 후보에 `[쿠폰]` 표시가 있는 매장에서만 사용할 수 있다. 표시가 없는 매장은 개인 잔액으로만 결제한다.
- 정책 존재만으로 소비 필요, 소비액, POI 선택을 미리 정하지 않는다. 페르소나의 필요·습관·자산·일정과 후보 특성을 함께 고려해 판단한다.
- 모든 commerce 이벤트에 양의 actual_spent를 반드시 부여 (0원·음수 금지).

**만족도 설정 (actual_satisfaction)**
- 0.0 ~ 1.0 범위의 실수입니다.
- 과거 방문 기록(avg_sat)이 있으면 그 근처에서 페르소나 성향을 반영해 조정합니다.
- 처음 가는 곳은 페르소나·카테고리·거리 등을 고려해 자유롭게 설정합니다.
- 값이 높을수록 만족, 낮을수록 불만족입니다.

**카카오 별점·리뷰 조회 (선택)**
- 후보 정보만으로 판단하기 어렵고 페르소나가 리뷰를 확인할 상황이면 `review_lookup_requests`에 후보 poi_id를 넣습니다.
- 요청한 리뷰가 제공되면 같은 후보 안에서 최종 선택을 다시 판단합니다. 필요하지 않으면 비우거나 생략합니다.
- 리뷰 확인 여부와 리뷰 반영 정도도 페르소나와 상황에 따라 자율적으로 결정합니다.

## 출력 형식 (JSON만, 다른 텍스트 금지)
{"picks": [
  {
    "order": 0,
    "poi_id": "C_xxxxxx",
    "actual_spent": 12000,
    "policy_spend": null,
    "actual_satisfaction": 0.71,
    "pick_reason": "단골 한식집. 어제 sat 0.72로 만족도 높음. 직장 0.05km. 평소 한식 즐겨 찾는 성향.",
    "pick_factor": "satisfaction"
  },
  {
    "order": 2,
    "poi_id": "C_yyyyyy",
    "actual_spent": 25000,
    "policy_spend": {"P009": 15000},
    "actual_satisfaction": 0.68,
    "pick_reason": "오늘 카페 휴식 의도와 가까운 후보가 맞았고, 사용 가능한 P009 정책지갑에서 15,000원을 결제하기로 선택.",
    "pick_factor": "satisfaction"
  }
],
"review_lookup_requests": ["C_aaa", "C_bbb"]  // 별점·리뷰 확인이 필요한 POI id (선택). 없으면 [] 또는 누락.
}

pick_factor enum 정의 (가장 결정적이었던 단일 요인 1개):
- `known`         : 단골/방문 경험 있는 곳 (KNOWS_POI 매칭) — visit_count > 0 이고 그 기억이 결정 좌우
- `distance`      : 거리 가까움이 결정적 — 어제 만족도·리뷰 데이터 없고 그냥 가깝다
- `satisfaction`  : 본인 어제 만족도(avg_sat) 높음이 결정적 — 본인 경험치 기반
- `review`        : 외부 카카오 별점·리뷰가 결정적 — review_lookup_requests 발동 후 결정 바꾼/굳힌 경우 (★ satisfaction과 명확 구분)
- `rumor`         : 어제 들은 소문·추천(KNOWS Conversation rumor)이 결정적
- `appointment`   : 약속(pinned_poi or 다른 agent와 만남 약속)이 결정적
- `random`        : 위 어느 단서도 결정적이지 않고 페르소나 성향으로 다양화 시도
/no_think"""


def _format_event_with_candidates(
    i: int, ev, cands: list[dict], recent_poi_ids: set[str] | None = None
) -> str:
    if not cands:
        return ""
    # 동네×업종 평균 결제단가(실측 카드 데이터 기반) — actual_spent 스케일 앵커
    anchor = cands[0].get("unit_anchor")
    anchor_s = f" | 동네 평균단가 ~{anchor:,}원" if anchor else ""
    lines = [
        f"### 이벤트 {i} | {ev.time} | {ev.anchor} | "
        f"{ev.category}/{ev.sub_category or _guess_sub_from_l1(ev.category)} | {ev.intent}{anchor_s}"
    ]
    recent = recent_poi_ids or set()
    for c in cands:
        known_mark = "★" if c["known"] else " "
        recent_mark = "⚠️" if c["poi_id"] in recent else ""
        sat = c.get("avg_satisfaction")
        km = c.get("km")
        visit_count = c.get("visit_count") or 0

        sat_s = f"avg_sat={sat:.2f}" if sat is not None else "신규"
        km_s = f"{km:.2f}km" if km is not None else ""
        visit_s = f"({visit_count}회)" if visit_count > 0 else ""
        price_s = price_icon(c.get("price_band"))
        coupon_s = c.get("coupon_tag") or ""

        lines.append(
            f"  {known_mark}{recent_mark} {c['poi_id']} | {c.get('name') or '(이름없음)'} | "
            f"{km_s} | {price_s} | {sat_s} {visit_s}{coupon_s}"
        )
    return "\n".join(lines)


def build_stage2_prompt(
    events: list,
    cands_by_order: dict[int, list[dict]],
    persona: dict | None = None,
    recent_poi_ids: set[str] | None = None,
    state: dict | None = None,
) -> str:
    # 페르소나 헤더
    header_parts = []
    if persona:
        daily_wd = persona.get("daily_wd") or 0
        daily_we = persona.get("daily_we") or 0
        tendency = persona.get("tendency") or ""
        lifestyle = (persona.get("lifestyle") or "").strip()
        income = persona.get("income") or ""
        budget_info = f"평소 1일 소비규모(스케일 참고, 총액 아님): 평일 {daily_wd:,}원 / 주말 {daily_we:,}원"
        # 가용 자산 — 가격대(₩~₩₩₩) 선택의 예산 근거
        balance = (state or {}).get("balance")
        if balance is not None:
            budget_info += f" / 현재 잔액: {int(balance):,}원"
        header_parts.append(f"## 에이전트 정보\n{lifestyle}\n{budget_info} / 소비성향: {tendency} / 소득분위: {income}")
        # 활성 정책 (grant 위주, LLM이 policy_spend 책정 시 참조)
        policy_budget = persona.get("policy_budget_summary") or ""
        if policy_budget:
            header_parts.append(f"## 활성 정책 (policy_spend 책정 시 참조)\n{policy_budget}")

    if recent_poi_ids:
        header_parts.append(
            f"## 최근 3일 방문 POI (⚠️ 표시 — 단순 반복 자제)\n"
            + ", ".join(list(recent_poi_ids)[:20])
        )

    blocks = []
    for i, ev in enumerate(events):
        cs = cands_by_order.get(i) or []
        if not cs:
            continue
        blocks.append(_format_event_with_candidates(i, ev, cs, recent_poi_ids))

    if not blocks:
        return "(외부 POI 결정 필요한 이벤트 없음)"

    header = "\n\n".join(header_parts) + "\n\n" if header_parts else ""
    body = "\n\n".join(blocks)
    return (
        f"{header}"
        f"다음 이벤트별 candidates 중에서 POI를 선택하고 소비액·만족도를 설정하세요.\n\n"
        f"{body}\n\n"
        f"각 이벤트의 order·poi_id·actual_spent·actual_satisfaction·pick_reason·pick_factor를 JSON으로 출력하세요. /no_think"
    )


# =========================================================
# LLM 호출 (SGLang/vLLM auto-detect via llm_client)
# =========================================================


def call_stage2(
    aid: str,
    stage1: Stage1Output,
    persona: dict,
    today: date,
    max_retry: int = 2,
    verbose: bool = False,
    state: dict | None = None,
    # 정책 정보는 persona["policy_budget_summary"]로 build_stage2_prompt에 전달됨 — 아래 두 인자는 호출부 호환용(미사용)
    active_policies: list[dict] | None = None,  # noqa: ARG001
    grant_remaining: dict[str, int] | None = None,  # noqa: ARG001
) -> tuple[Stage2Output, dict[int, list[dict]], dict]:
    """Stage 2 LLM 호출. (picks, 사용된 candidates, meta) 반환.

    today: 오늘 날짜. desire 계산의 days_since_visit 산출에 사용.
    state: State 노드 dict (balance 등) — 가격대 선택의 예산 근거로 프롬프트에 노출.
    """
    total_started = time.perf_counter()
    timing: dict[str, object] = {
        "t_candidates": 0.0,
        "t_price_maps": 0.0,
        "t_recent_memory": 0.0,
        "t_prompt_build": 0.0,
        "t_schema_build": 0.0,
        "t_retry_prompt": 0.0,
        "t_llm": 0.0,
        "t_llm_initial": 0.0,
        "t_llm_review": 0.0,
        "t_llm_retry": 0.0,
        "t_json_extract": 0.0,
        "t_json_parse": 0.0,
        "t_model_validate": 0.0,
        "t_review_lookup": 0.0,
        "t_candidate_validate": 0.0,
        "t_postprocess": 0.0,
        "n_llm_calls": 0,
        "attempts": [],
    }

    def timing_snapshot() -> dict:
        timing["t_total"] = time.perf_counter() - total_started
        return {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in timing.items()
        }

    fb_stats: dict[str, int] = {}
    candidate_timing: dict[str, float | int] = {}
    started = time.perf_counter()
    cands_by_order = fetch_candidates_for_events(
        aid, stage1.events, persona, today, stats=fb_stats, timing=candidate_timing,
    )
    timing["t_candidates"] = time.perf_counter() - started
    timing["candidate_detail"] = {
        k: round(v, 6) if isinstance(v, float) else v
        for k, v in candidate_timing.items()
    }
    need_llm = any(cs for cs in cands_by_order.values())

    # POI → (price_band, price_factor) 맵 — merge/소비모델에서 금액 반영용
    started = time.perf_counter()
    price_by_poi: dict[str, tuple[int, float]] = {
        c["poi_id"]: (c.get("price_band"), c.get("price_factor", 1.0))
        for cs in cands_by_order.values() for c in cs
    }
    # POI → 쿠폰 사용처 여부 — merge/정책사용 하드검증용
    coupon_by_poi: dict[str, bool] = {
        c["poi_id"]: bool(c.get("coupon_eligible"))
        for cs in cands_by_order.values() for c in cs
    }
    timing["t_price_maps"] = time.perf_counter() - started

    if not need_llm:
        # 외부 POI 결정 필요 없음 (전부 residence/workplace/pinned)
        return Stage2Output(picks=[]), cands_by_order, {
            "skipped": True,
            "price_by_poi": price_by_poi,
            "coupon_by_poi": coupon_by_poi,
            "s2_timing": timing_snapshot(),
            **fb_stats,
        }

    # 최근 3일 방문 POI (억제용)
    recent_poi_ids: set[str] = set()
    started = time.perf_counter()
    try:
        from neo4j_load._common import driver_session
        from datetime import timedelta
        three_days_ago = (today - timedelta(days=3)).isoformat()
        with driver_session() as s:
            rows = s.run(
                "MATCH (a:Agent {id:$aid})-[:REMEMBERS]->(m:Memory {type:'visited'})-[:ABOUT_POI]->(p:POI) "
                "WHERE m.day >= date($since) RETURN p.id AS pid",
                aid=aid, since=three_days_ago
            )
            recent_poi_ids = {r["pid"] for r in rows}
    except Exception:
        pass
    timing["t_recent_memory"] = time.perf_counter() - started

    started = time.perf_counter()
    user_block = build_stage2_prompt(
        stage1.events, cands_by_order,
        persona=persona,
        recent_poi_ids=recent_poi_ids,
        state=state,
    )
    timing["t_prompt_build"] = time.perf_counter() - started

    # 환각 차단용 JSON schema — poi_id는 전체 후보풀 union enum 강제.
    # order-별 enum은 아니지만 후보풀 외 POI는 0건 보장. order_mismatch는 fallback에서 처리.
    started = time.perf_counter()
    all_pids = sorted({c["poi_id"] for cs in cands_by_order.values() for c in cs})
    expected_orders = [
        i for i, ev in enumerate(stage1.events)
        if ev.category not in INTERNAL_CATS and not ev.pinned_poi and cands_by_order.get(i)
    ]
    s2_schema = None
    if all_pids and expected_orders:
        s2_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "stage2_picks", "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "picks": {
                            "type": "array",
                            "minItems": len(expected_orders),
                            "maxItems": len(expected_orders),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "order": {"type": "integer", "enum": expected_orders},
                                    "poi_id": {"type": "string", "enum": all_pids},
                                    "actual_spent": {"type": "number", "minimum": 0},
                                    "actual_satisfaction": {"type": "number", "minimum": 0, "maximum": 1},
                                    "policy_spend": {"type": ["object", "null"]},
                                    "pick_reason": {"type": ["string", "null"]},
                                    "pick_factor": {"type": ["string", "null"]},
                                },
                                "required": ["order", "poi_id", "actual_satisfaction", "actual_spent"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["picks"],
                    "additionalProperties": False,
                },
            },
        }
        # review_lookup_requests 선택적 출력 허용 — POI id 목록 (전체 cand pool union)
        s2_schema["json_schema"]["schema"]["properties"]["review_lookup_requests"] = {
            "type": ["array", "null"],
            "items": {"type": "string", "enum": all_pids},
        }
    timing["t_schema_build"] = time.perf_counter() - started

    last_err = None
    review_lookup_used: dict[str, dict] = {}  # 첨부됐던 lookup 결과 (meta 출력용)
    pre_review_picks: dict[int, str] = {}     # 리뷰 보기 전(1차) 선택 {order: poi_id} — 사고변화 추적
    for attempt in range(max_retry + 1):
        temp = 0.7 + 0.1 * attempt
        attempt_started = time.perf_counter()
        attempt_timing: dict[str, float | int | str] = {"attempt": attempt}
        call_kind = "review" if review_lookup_used else ("initial" if attempt == 0 else "retry")
        attempt_timing["call_kind"] = call_kind
        # review_lookup 결과가 있으면 prompt에 추가 컨텍스트 첨부
        started = time.perf_counter()
        prompt_now = user_block
        if review_lookup_used:
            review_block_lines = ["", "## 추가로 조회된 카카오 별점·리뷰 (요청한 POI만)"]
            for pid, info in review_lookup_used.items():
                review_block_lines.append(format_review_block(pid, info))
            prompt_now = user_block + "\n" + "\n".join(review_block_lines) + (
                "\n\n위 정보를 참고해 최종 picks를 결정하세요. "
                "이번 응답에서는 review_lookup_requests를 비워 두세요(이미 조회 완료).\n"
            )
        elapsed = time.perf_counter() - started
        timing["t_retry_prompt"] += elapsed
        attempt_timing["t_retry_prompt"] = elapsed
        error_stage = "llm"
        try:
            started = time.perf_counter()
            resp = _llm_call(
                None, SYSTEM_S2, prompt_now,
                temperature=temp, max_tokens=1400,  # review_lookup_requests 필드 + 추가 컨텍스트
                response_format=s2_schema,
            )
            elapsed = time.perf_counter() - started
            timing["t_llm"] += elapsed
            timing[f"t_llm_{call_kind}"] += elapsed
            timing["n_llm_calls"] += 1
            attempt_timing["t_llm"] = elapsed
            raw = resp.choices[0].message.content
            attempt_timing["tokens_in"] = int(getattr(resp.usage, "prompt_tokens", 0) or 0)
            attempt_timing["tokens_out"] = int(getattr(resp.usage, "completion_tokens", 0) or 0)
            if verbose:
                print(f"--- attempt {attempt} (temp={temp}) ---")
                print(raw[:600])

            error_stage = "json_extract"
            started = time.perf_counter()
            json_str = _extract_json(raw)
            elapsed = time.perf_counter() - started
            timing["t_json_extract"] += elapsed
            attempt_timing["t_json_extract"] = elapsed

            error_stage = "json_parse"
            started = time.perf_counter()
            data = json.loads(json_str)
            elapsed = time.perf_counter() - started
            timing["t_json_parse"] += elapsed
            attempt_timing["t_json_parse"] = elapsed

            error_stage = "model_validate"
            started = time.perf_counter()
            parsed = Stage2Output.model_validate(data)
            elapsed = time.perf_counter() - started
            timing["t_model_validate"] += elapsed
            attempt_timing["t_model_validate"] = elapsed

            # === review_lookup_requests 처리 (한 번만, 첫 호출에서만) ===
            if not review_lookup_used and parsed.review_lookup_requests:
                # 후보 풀 안에 있는 poi_id만 채택 (환각 방지)
                valid_lookup_ids = [pid for pid in parsed.review_lookup_requests
                                    if pid in set(all_pids)]
                # 총 LLM 호출 상한(max_retry+1)은 유지한다. 마지막 허용 호출에서
                # 리뷰를 요청하면 조회만 하고 최종판단을 못 하는 문제를 피하기 위해
                # 남은 호출 예산이 있을 때만 리뷰를 가져온다.
                if valid_lookup_ids and attempt < max_retry:
                    error_stage = "review_lookup"
                    started = time.perf_counter()
                    fetched = lookup_reviews_batch(valid_lookup_ids[:8], max_reviews=3)
                    elapsed = time.perf_counter() - started
                    timing["t_review_lookup"] += elapsed
                    attempt_timing["t_review_lookup"] = elapsed
                    if fetched:
                        review_lookup_used = fetched
                        # 리뷰 보기 전(1차) 선택 보존 — 추가 LLM 호출 없이 '사고 변화' 추적용
                        pre_review_picks = {p.order: p.poi_id for p in parsed.picks}
                        # 같은 attempt에서 재호출이 아니라 다음 attempt에 첨부해서 한 번 더 시도
                        # (max_retry 안 쓰고 별도 1회 — temp 0.7 그대로)
                        if verbose:
                            print(f"[review_lookup] fetched {len(fetched)} POIs, retrying with context")
                        attempt_timing["status"] = "review_retry"
                        attempt_timing["t_total"] = time.perf_counter() - attempt_started
                        timing["attempts"].append({
                            k: round(v, 6) if isinstance(v, float) else v
                            for k, v in attempt_timing.items()
                        })
                        continue  # 다음 iteration에서 prompt_now에 첨부됨
                elif valid_lookup_ids:
                    fb_stats["review_skipped_no_call_budget"] = (
                        fb_stats.get("review_skipped_no_call_budget", 0) + 1
                    )

            # 후보 풀 안에 있는지 검증 — 반드시 해당 order의 candidates 안에서만 valid.
            # (이전 버그: valid_pois = 전체 cands flat → 다른 order의 POI도 통과되어 카테고리 매칭이 깨짐)
            error_stage = "candidate_validate"
            started = time.perf_counter()
            import random as _random
            corrected_picks = []
            hallucinations = 0          # 보정 (해당 order의 cands에 없지만 cands는 존재)
            hallucinations_dropped = 0  # 드롭 (해당 order에 cands 자체 없음)
            order_mismatch = 0          # LLM이 다른 order의 POI를 가져옴 (보정 카운트에 포함)
            rng = _random.Random(hash(aid))
            # 전체 cands flat — order 추적용 (어느 다른 order에 속하는지 진단)
            poi_to_orders: dict[str, list[int]] = {}
            for ord_i, cs in cands_by_order.items():
                for c in cs:
                    poi_to_orders.setdefault(c["poi_id"], []).append(ord_i)
            for pick in parsed.picks:
                cands_for_this_order = cands_by_order.get(pick.order, [])
                valid_for_order = {c["poi_id"] for c in cands_for_this_order}
                if pick.poi_id in valid_for_order:
                    corrected_picks.append(pick)
                else:
                    # 다른 order의 cands에 있는 POI? (order 매핑 흐트러짐 진단)
                    if pick.poi_id in poi_to_orders:
                        order_mismatch += 1
                    if cands_for_this_order:
                        top = cands_for_this_order[:5]
                        chosen = rng.choice(top)["poi_id"]
                        corrected_picks.append(Stage2Pick(order=pick.order, poi_id=chosen, actual_spent=None, actual_satisfaction=0.5))
                        hallucinations += 1
                    else:
                        # 해당 order에 candidates 자체 없음 — drop
                        hallucinations_dropped += 1
            parsed = Stage2Output(picks=corrected_picks)
            elapsed = time.perf_counter() - started
            timing["t_candidate_validate"] += elapsed
            attempt_timing["t_candidate_validate"] = elapsed

            # 후처리: LLM이 답 안 한 외출 이벤트에 candidates 자동 fill
            error_stage = "postprocess"
            started = time.perf_counter()
            picks_before_fill = len(parsed.picks)
            parsed = _fill_missing_picks(parsed, stage1.events, cands_by_order, aid=aid)
            missing_filled = len(parsed.picks) - picks_before_fill

            # 후처리: actual_spent 0/None인 commerce 이벤트에 fallback (cap 추적 무력화 방지)
            # 실측 단가 앵커(동네×업종) × 밴드 배율 우선, 없으면 구 표 방식.
            daily_wd = persona.get("daily_wd") or 0
            cat_by_order = {i: ev.category for i, ev in enumerate(stage1.events)}
            anchor_by_order = {
                i: (cs[0].get("unit_anchor") if cs else None)
                for i, cs in cands_by_order.items()
            }
            for pick in parsed.picks:
                cat = cat_by_order.get(pick.order)
                if cat and cat not in INTERNAL_CATS:
                    pb, pf = price_by_poi.get(pick.poi_id) or (None, 1.0)
                    _ensure_positive_spend(
                        pick, cat, daily_wd, price_factor=pf,
                        base_won=anchor_by_order.get(pick.order), band=pb,
                    )
            elapsed = time.perf_counter() - started
            timing["t_postprocess"] += elapsed
            attempt_timing["t_postprocess"] = elapsed
            attempt_timing["status"] = "ok"
            attempt_timing["t_total"] = time.perf_counter() - attempt_started
            timing["attempts"].append({
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in attempt_timing.items()
            })

            meta = {
                "attempt": attempt,
                "temp": temp,
                "tokens_in": resp.usage.prompt_tokens,
                "tokens_out": resp.usage.completion_tokens,
                "hallucinations_corrected": hallucinations,
                "hallucinations_dropped": hallucinations_dropped,
                "order_mismatch": order_mismatch,
                "missing_picks_filled": missing_filled,
                "price_by_poi": price_by_poi,
                "coupon_by_poi": coupon_by_poi,
                "review_lookup_count": len(review_lookup_used),
                # 리뷰 흔적 — 추가 LLM 호출 없이 기존 2-pass 데이터에서 캡처
                "review_lookup_used": review_lookup_used,   # {poi_id: {rating, rating_count, reviews, category}}
                "pre_review_picks": pre_review_picks,       # {order: 리뷰 전 선택 poi_id}
                "s2_timing": timing_snapshot(),
                **fb_stats,
            }
            return parsed, cands_by_order, meta
        except Exception as e:
            stage_key = {
                "llm": "t_llm",
                "json_extract": "t_json_extract",
                "json_parse": "t_json_parse",
                "model_validate": "t_model_validate",
                "review_lookup": "t_review_lookup",
                "candidate_validate": "t_candidate_validate",
                "postprocess": "t_postprocess",
            }.get(error_stage)
            if stage_key and stage_key not in attempt_timing:
                elapsed = time.perf_counter() - started
                timing[stage_key] += elapsed
                attempt_timing[stage_key] = elapsed
                if error_stage == "llm":
                    timing[f"t_llm_{call_kind}"] += elapsed
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
            if verbose:
                print(f"[attempt {attempt}] failed: {e}")

    # 최종 retry 실패: LLM picks 빈 상태에서 candidates 첫 거 강제 fill
    fallback = _fill_missing_picks(Stage2Output(picks=[]), stage1.events, cands_by_order, aid=aid)
    if fallback.picks:
        return fallback, cands_by_order, {
            "fallback_only": True,
            "price_by_poi": price_by_poi,
            "coupon_by_poi": coupon_by_poi,
            "last_err": str(last_err)[:200],
            "s2_timing": timing_snapshot(),
        }
    raise RuntimeError(f"Stage2 failed after {max_retry+1} attempts: {last_err}")


def _fill_missing_picks(
    stage2: Stage2Output, stage1_events: list, cands_by_order: dict[int, list[dict]],
    aid: str = "",
) -> Stage2Output:
    """LLM이 picks에 안 만든 외출 이벤트에 candidates Top 5 중 random POI 자동 채움."""
    import random as _random
    picked_orders = {p.order for p in stage2.picks}
    new_picks = list(stage2.picks)
    rng = _random.Random(hash(aid) if aid else 42)
    for i, ev in enumerate(stage1_events):
        if i in picked_orders:
            continue
        if ev.category in INTERNAL_CATS or ev.pinned_poi:
            continue
        cs = cands_by_order.get(i) or []
        if not cs:
            continue
        top = cs[:5]
        chosen = rng.choice(top)["poi_id"]
        new_picks.append(Stage2Pick(order=i, poi_id=chosen, actual_spent=None, actual_satisfaction=0.5))
    return Stage2Output(picks=new_picks)


# =========================================================
# Stage 1 + Stage 2 병합 → 최종 events
# =========================================================
def merge_to_final_events(
    stage1: Stage1Output, stage2: Stage2Output, persona: dict,
    price_by_poi: dict[str, tuple] | None = None,
    coupon_by_poi: dict[str, bool] | None = None,
    review_lookup_used: dict | None = None,
    pre_review_picks: dict | None = None,
) -> list[dict]:
    """Stage 1 + Stage 2 picks → 최종 events with poi_id.

    price_by_poi: call_stage2 meta의 {poi_id: (price_band, price_factor)} —
    이벤트에 가격대를 부착해 소비모델(apply_consumption_model)이 금액에 반영.
    coupon_by_poi: {poi_id: 쿠폰 사용처 여부} — 정책사용 하드검증·INCLUDES 기록용.

    카테고리 기준 우선 (anchor는 출발지 표시일 뿐):
      - pinned_poi 있으면 그대로
      - cat ∈ {집, 직장} (머무름) → anchor에 따라 home/work POI
      - cat ∈ 외출 카테고리 (식사·카페·편의점 등) → Stage 2 pick (commerce POI)
        - Stage 2 pick 누락 시 fallback으로 anchor POI 사용
    """
    review_lookup_used = review_lookup_used or {}   # {poi_id: {rating, rating_count, reviews, ...}}
    pre_review_picks = pre_review_picks or {}        # {order: 리뷰 전 선택 poi_id}
    pick_by_order = {p.order: p for p in stage2.picks}
    out = []
    for i, ev in enumerate(stage1.events):
        poi_id = None
        pick_obj = pick_by_order.get(i)
        if ev.pinned_poi:
            poi_id = ev.pinned_poi
        elif ev.category in INTERNAL_CATS:
            # 머무름 — anchor POI 사용
            if ev.anchor == "residence":
                poi_id = persona.get("home_poi_id")
            elif ev.anchor == "workplace":
                poi_id = persona.get("work_poi_id")
            else:  # cat=집/직장인데 zone: anchor — LLM 흔한 패턴, category 기준으로 처리
                if ev.category == "집":
                    poi_id = persona.get("home_poi_id")
                else:  # 직장
                    poi_id = persona.get("work_poi_id")
        else:
            # 외출 카테고리 — Stage 2 pick (anchor의 동에서 commerce POI 결정)
            poi_id = pick_obj.poi_id if pick_obj else None
            if not poi_id:
                # Stage 2 pick 누락 시 anchor POI fallback
                if ev.anchor == "residence":
                    poi_id = persona.get("home_poi_id")
                elif ev.anchor == "workplace":
                    poi_id = persona.get("work_poi_id")

        # POI 가격대 (commerce pick만 — anchor/내부 이벤트는 band None·factor 1.0)
        _pb = (price_by_poi or {}).get(poi_id) if (poi_id and ev.category not in INTERNAL_CATS) else None

        # 리뷰 노출·사고변화 흔적 (추가 호출 0 — 기존 2-pass 캡처 데이터만 사용)
        _seen = review_lookup_used.get(poi_id) if poi_id else None
        _pre_poi = pre_review_picks.get(i)
        _review_changed = bool(_pre_poi and poi_id and _pre_poi != poi_id)
        _seen_rv = (_seen.get("reviews") if _seen else None) or []
        _snippet = (_seen_rv[0].get("contents") if _seen_rv else None)
        out.append({
            "order": i,
            "time": ev.time,
            "duration_min": None,  # 다음 이벤트 시간으로 계산 또는 기본 60
            "anchor": ev.anchor,
            "category": ev.category,
            "sub_category": ev.sub_category,
            "intent": ev.intent,
            "poi_id": poi_id,
            "with_agents": ev.with_agents or [],
            "actual_satisfaction": pick_obj.actual_satisfaction if pick_obj else None,
            "actual_spent": pick_obj.actual_spent if pick_obj else 0,
            "price_band": _pb[0] if _pb else None,
            "price_factor": float(_pb[1]) if _pb else 1.0,
            # 쿠폰 사용처 여부 (후보풀 밖 POI(anchor 등)는 None = 판정 불가)
            "coupon_eligible": (coupon_by_poi or {}).get(poi_id) if poi_id else None,
            # 정책별 사용액 dict ({"P009": 5000}) — 분석 시 정책 사용처 추적
            "policy_spend": (pick_obj.policy_spend if pick_obj else None) or {},
            # ───── 사고과정 흔적 (인터뷰용) ─────
            "reasoning": ev.reasoning,                 # Stage 1: 왜 이 의도·카테고리·anchor
            "trigger": ev.trigger,                     # Stage 1: appointment/rumor/policy/...
            "pick_reason": pick_obj.pick_reason if pick_obj else None,   # Stage 2: 왜 이 POI
            "pick_factor": pick_obj.pick_factor if pick_obj else None,   # Stage 2: known/distance/...
            # ───── 리뷰 노출·사고변화 흔적 (추가 LLM 호출 0) ─────
            "review_seen": _seen is not None,                        # 이 POI 카카오 리뷰를 봤나
            "seen_rating": (_seen or {}).get("rating"),              # 본 평균 별점
            "seen_rating_count": (_seen or {}).get("rating_count"),
            "review_snippet": _snippet,                              # 본 리뷰 한 줄
            "pre_review_poi": _pre_poi if _review_changed else None, # 리뷰 전(1차) 선택 — 바뀐 경우만
            "review_changed": _review_changed,                       # 리뷰가 최종 선택을 바꿨나
        })
    return out


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
    ctx = build_dawn_context(args.aid, today)
    s1, m1 = call_stage1(args.aid, today, ctx=ctx, verbose=args.verbose)
    print("\n=== Stage 1 ===")
    print(s1.model_dump_json(indent=2))
    print(f"\nmeta: {m1}")

    s2, cands, m2 = call_stage2(args.aid, s1, ctx.persona, today, verbose=args.verbose)
    print("\n=== Stage 2 ===")
    print(s2.model_dump_json(indent=2))
    print(f"\nmeta: {m2}")

    final = merge_to_final_events(s1, s2, ctx.persona)
    print("\n=== 최종 이벤트 ===")
    for e in final:
        print(f"  [{e['order']}] {e['time']} {e['anchor']:30} {e['category']:6} → {e['poi_id']} ({e['intent']})")
