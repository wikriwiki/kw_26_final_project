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

import copy
import json
import hashlib
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
    _format_policy_facts, _format_policy_status, _json_dict,
)
from stage1_intent import Stage1Output, call_stage1, _extract_json, _number_evidence_lines, _evidence_lines  # noqa: E402
from llm_client import call_chat as _llm_call  # noqa: E402
from prompt_grounding import validate_stated_reason
from no_smoking_prompts import SYSTEM_STAGE2
from poi_price import poi_price, price_icon, unit_price_anchor, band_factor  # noqa: E402
import price_ticket as _pt  # noqa: E402  [2026-10-07] 실측 결제 1건당·메뉴×인분 금액

# [2026-10-11] 장보기 후보에 가까운 대형 형태 매장(대형마트·기업형 슈퍼)을 브랜드와 상관없이 더한다(P016).
# 반경·개수·형태는 실행 설정으로 고정한다. 참여 체인 중 GS더프레시·농협하나로마트는 대부분 기업형 슈퍼 형태라
# 대형마트만 넣으면 이들이 빠진다. 형태 둘·가까운 3곳이면 명부 2,000명 중 85.4%, 2곳이면 73.3% 의 장보기 후보에
# 참여 매장이 하나 이상 들어간다(/data/ab3w/audit_tools/p016_reach2.py, 2026-10-11, 마트 적재 고친 뒤).
_MART_REACH = os.environ.get("EXP_MART_REACH", "0") == "1"
_MART_REACH_KM = float(os.environ.get("EXP_MART_REACH_KM", "3"))
_MART_REACH_N = int(os.environ.get("EXP_MART_REACH_N", "2"))
_MART_REACH_FORMATS = [f.strip() for f in os.environ.get("EXP_MART_REACH_FORMATS", "hypermarket").split(",") if f.strip()]
_GROCERY_SUBS = frozenset({"슈퍼마켓", "식료품", "청과", "정육", "수산", "장보기"})
_MART_REACH_CYPHER = """
MATCH (a:Agent {id: $aid})-[:LIVES_AT|WORKS_AT]->(anchor:POI)
WHERE anchor.lon IS NOT NULL AND anchor.lat IS NOT NULL
MATCH (p:POI) WHERE p.mart_format IN $formats
  AND p.type = 'commerce' AND p.lon IS NOT NULL AND p.lat IS NOT NULL
WITH a, p, min(point.distance(point({longitude: p.lon, latitude: p.lat}),
                              point({longitude: anchor.lon, latitude: anchor.lat})) / 1000.0) AS km
WHERE km <= $radius_km
OPTIONAL MATCH (a)-[kp:KNOWS_POI]->(p)
RETURN p.id AS poi_id, p.name AS name,
       (kp IS NOT NULL) AS known,
       coalesce(kp.visit_count, 0) AS visit_count,
       kp.avg_satisfaction AS avg_satisfaction,
       kp.last_visit AS last_visit,
       p.coupon_eligible AS coupon_eligible,
       p.sangsaeng_eligible AS sangsaeng_eligible,
       p.upjong_l3 AS upjong_l3,
       head([(p)-[:IN_CATEGORY]->(oc:Category) | oc.name]) AS poi_sub_category,
       head([(p)-[:IN_CATEGORY]->(oc:Category) | oc.parent]) AS poi_l1,
       km
ORDER BY km ASC, poi_id ASC LIMIT $n
"""

# [2026-10-11] 장보기 결제의 농축산물 금액을 모델에게 묻는다(P016 품목 할인·인터뷰용). 켜면 두 갈래가 같은 질문을 받는다.
_PRODUCE_FIELD = os.environ.get("EXP_PRODUCE_FIELD", "0") == "1"
_PRODUCE_NOTE = (
    "- `produce_spent`: 마트·슈퍼·식료품·청과·정육에서 장을 본 결제라면, 그 결제 중 국산 신선 농축산물"
    "(과일·채소·고기·쌀·잡곡·계란)에 쓴 금액(원)입니다. 장보기가 아니면 null 입니다. `actual_spent`보다 클 수 없습니다.\n")
FREE_OUTDOOR_SUBS = frozenset({"산책", "공원", "공원 산책", "산책로", "등산", "걷기", "둘레길", "한강공원"})
_FREE_OUTDOOR_RE = re.compile(r"산책|등산|공원|둘레길|휴양림|걷기|숲길|하이킹|한강")
_PINNED_CYPHER = """
MATCH (p:POI {id: $pid})
OPTIONAL MATCH (p)-[:IN_DONG]->(dg:Dong)
OPTIONAL MATCH (a:Agent {id: $aid})-[kp:KNOWS_POI]->(p)
RETURN p.id AS poi_id, p.name AS name, (kp IS NOT NULL) AS known,
       coalesce(kp.visit_count, 0) AS visit_count, kp.avg_satisfaction AS avg_satisfaction,
       kp.last_visit AS last_visit, p.coupon_eligible AS coupon_eligible,
       p.sangsaeng_eligible AS sangsaeng_eligible, p.upjong_l3 AS upjong_l3,
       head([(p)-[:IN_CATEGORY]->(oc:Category) | oc.name]) AS poi_sub_category,
       head([(p)-[:IN_CATEGORY]->(oc:Category) | oc.parent]) AS poi_l1,
       NULL AS km, dg.code AS dong_code
LIMIT 1
"""
from coupon_eligibility import is_coupon_eligible  # noqa: E402
from sangsaeng_eligibility import is_sangsaeng_eligible  # noqa: E402
from poi_review_lookup import lookup_reviews_batch, format_review_block  # noqa: E402


try:
    from pydantic import BaseModel, Field, model_serializer
except ImportError:
    raise


class Stage2Pick(BaseModel):
    order: int
    poi_id: str
    actual_spent: float | None = None        # LLM이 설정 (원, 양수, 총 소비액)
    # [EXP_PRICE_MODE] 무엇을 얼마에: 메뉴·물건, 1인분·1개 가격, 내가 계산하는 인분·개수. 금액 = unit_price × pay_count
    menu: str | None = None
    unit_price: float | None = None
    pay_count: int | None = None
    # [2026-10-11 EXP_PRODUCE_FIELD] 장보기 결제 중 국산 신선 농축산물에 쓴 금액(원). 장보기가 아니면 None.
    # 품목 할인(P016)의 기준액과 인터뷰 근거로만 쓴다 — 두 갈래 모두 같은 질문을 받는다.
    produce_spent: int | None = None
    actual_satisfaction: float | None = None # LLM이 설정 (0~1)
    # actual_spent 중 정책 지원금에서 사용한 금액 — {"P009": 5000} 형태.
    # 평소 잔액으로 쓴 부분 = actual_spent - sum(policy_spend.values())
    policy_spend: dict[str, float] | None = None
    # 이 지출이 지원금이 없었어도 했을 것인지(true) / 지원금이 있어서 비로소 한 것인지(false).
    # 참고3 ④의 서베이 문항 형태(품목별 0/1)를 그대로 옮긴 것. 금액 가중평균이 MPC가 된다.
    would_buy_anyway: bool | None = None
    extra_spent: int | None = None
    pick_reason: str | None = None
    evidence_ref: str | None = None
    evidence_quote: str | None = None
    pick_factor: str | None = None  # known | distance | satisfaction | rumor | appointment | random

    @model_serializer(mode='wrap')
    def _serialize_optional_evidence(self, handler):
        value = handler(self)
        if self.evidence_ref is None:
            value.pop('evidence_ref', None)
        if self.evidence_quote is None:
            value.pop('evidence_quote', None)
        for _k in ('menu', 'unit_price', 'pay_count', 'produce_spent'):
            if getattr(self, _k) is None:
                value.pop(_k, None)
        return value


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
    durable: bool = False,
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
    if daily_wd and daily_wd > 0 and not durable:
        # daily_wd가 매우 작은 경우 비율 보정 (예: 절약형 페르소나)
        # 내구재는 예외 — 냉장고는 하루 예산의 0.4배로 살 수 있는 물건이 아니다.
        # 이 클램프를 그대로 두면 내구재 앵커를 넣어도 4만원 페르소나는 1.6만원이 된다.
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

    neutral_rules = None
    if active_stage2_is_neutral() and persona.get("coupon_poi_restricted"):
        from eligibility import validated_restricted_rules
        neutral_rules = validated_restricted_rules(persona.get("poi_eligibility_spec"))
        if not persona.get("poi_eligible_marker"):
            raise ValueError("사용처 제한 정책에 eligible_marker가 없습니다")

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
        if ev.category in INTERNAL_CATS or (ev.pinned_poi and not _pt.ON):
            out[i] = []
            continue
        if ev.pinned_poi:
            # [2026-10-07b] 약속 장소가 정해진 일정: 후보를 그 한 곳으로 두고 2단계가 메뉴·금액·만족도를 정한다.
            # 예전엔 pinned 일정이 2단계를 건너뛰어 금액·만족도가 비었다(가격 판에서 0원·기억 없음).
            group_key_for[i] = ("", "__PIN__" + str(ev.pinned_poi))
            l1_for[i] = ev.category
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
            # [2026-10-07b] 무료 야외 활동(산책·공원 등)은 세부 업종을 무엇으로 적었든(스포츠·유원지·오락 등) 돈 내는 가게로
            # 보내지 않는다 — 10명 시험에서 '동네 공원에서 가벼운 산책'(sub 스포츠)이 골프존에 붙었다.
            _free_grp = (_pt.ON and not sub_cat.startswith("__PIN__") and (
                sub_cat in FREE_OUTDOOR_SUBS or _FREE_OUTDOOR_RE.search(sub_cat or "")
                or all(_FREE_OUTDOOR_RE.search(str(getattr(events[_ix], "intent", "") or "")) for _ix in event_idxs)))
            if _free_grp:
                cands = []
                for _alt in ("공원", "산책"):
                    cands = build_stage2_candidates(aid, dong_code, _alt, limit=pool_size, session=sess)
                    if cands:
                        break
                if not cands:
                    s["cand_free_outdoor_none"] = s.get("cand_free_outdoor_none", 0) + n
                    for idx in event_idxs:
                        out[idx] = []
                    continue
            elif sub_cat.startswith("__PIN__"):
                cands = [dict(r) for r in sess.run(_PINNED_CYPHER, aid=aid, pid=sub_cat[len("__PIN__"):])]
                dong_code = (cands[0].get("dong_code") if cands else "") or ""
                sub_cat = (cands[0].get("poi_sub_category") if cands else None) or (l1 or "")
            else:
                cands = build_stage2_candidates(aid, dong_code, sub_cat, limit=pool_size, session=sess)
            tm["t_query_exact"] += time.perf_counter() - started
            tm["n_query_exact"] += 1
            if not cands and _pt.ON and (sub_cat in FREE_OUTDOOR_SUBS or _FREE_OUTDOOR_RE.search(sub_cat or "")
                                          or any(_FREE_OUTDOOR_RE.search(str(getattr(events[_ix], "intent", "") or ""))
                                                 for _ix in event_idxs)):
                # [2026-10-07b] 산책·공원은 돈 내는 업종이 아니다. 대분류(여가) 가게로 넘기면 골프연습장·탁구클럽·여행사가
                # 붙었다(산책 의도의 40~45%). 공원 POI 만 찾고, 없으면 가게 없이 둔다.
                for _alt in ("공원", "산책"):
                    if _alt != sub_cat:
                        cands = build_stage2_candidates(aid, dong_code, _alt, limit=pool_size, session=sess)
                        if cands:
                            break
                if not cands:
                    s["cand_free_outdoor_none"] = s.get("cand_free_outdoor_none", 0) + n
                    for idx in event_idxs:
                        out[idx] = []
                    continue
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

            # [2026-10-11 EXP_MART_REACH] 장보기라면 집·직장에서 가까운 대형마트(동 밖이라도)를 후보에 더한다.
            # 후보를 사건 동 하나에서만 찾으면 대형마트가 없는 동(대부분)의 시민에게 대형마트가 보이지 않는다.
            # 두 갈래 모두 같은 규칙이고, 정책 여부와 상관없이 실제 거리로만 고른다.
            if (_MART_REACH and cands is not None and not sub_cat.startswith("__PIN__")
                    and (l1 == "마트" or sub_cat in _GROCERY_SUBS)):
                _have = {c.get("poi_id") for c in cands}
                _extra = [dict(r) for r in sess.run(_MART_REACH_CYPHER, aid=aid, radius_km=_MART_REACH_KM, formats=_MART_REACH_FORMATS,
                                                     n=_MART_REACH_N + len(_have))]
                _extra = [c for c in _extra if c.get("poi_id") not in _have][:_MART_REACH_N]
                if _extra:
                    cands = list(cands) + _extra
                    s["cand_mart_reach_added"] = s.get("cand_mart_reach_added", 0) + len(_extra)

            # POI 가격대 부착 (결정론, O(1)) — Stage2 프롬프트 표기·소비 반영용.
            # district fallback 후보는 자기 동 미상 → anchor 동 prior로 근사.
            # unit_anchor: 이 동네×업종 평균 결제단가(실측 기반) — 프롬프트 스케일 앵커.
            started = time.perf_counter()
            anchor_won = unit_price_anchor(dong_code, l1)
            # [내구재 앵커] 이 사람이 이 업종에서 미뤄 둔 물건이 있으면 그 시세를 쓴다.
            # 동네x업종 평균단가(쇼핑 ~5만원)만 보면 90만원짜리 냉장고가 나올 수 없다.
            # 7차 실측: 내구재 채널은 작동했으나 건당 16,863원에 그쳤다.
            # 대기 목록이 있는 사람에게만 붙으므로 가전 결제가 '드물고 큰' 형태가 된다.
            _dur_anchor = None
            try:
                from durables import anchor_for as _dur_anchor_for
                _dur_anchor = _dur_anchor_for(
                    persona.get("id"), persona.get("life_stage"),
                    persona.get("age_group"), sub_cat)
            except Exception:
                _dur_anchor = None
            if _dur_anchor:
                anchor_won = _dur_anchor
            # 사용처 제한 지원금(쿠폰) 잔액 보유 여부 — run_simulation이 persona에 세팅
            # 정책 사용 가능 여부는 후보 정보로만 제공한다. 후보 정렬 가점은 결과를
            # 사전 유도하므로 기본 0이며, 별도 민감도 실험에서만 명시적으로 켠다.
            coupon_active = bool(persona.get("coupon_poi_restricted"))
            # 적격 판정은 정책이 정한 룰로 한다. 룰이 없으면 기존 쿠폰 룰(P010)로
            # 떨어진다 — 정책마다 여기에 분기를 더하면 1:1 결합이 되살아난다.
            _mk = persona.get("poi_eligible_marker") or "[쿠폰]"
            # 장소 조건이 있는 정책(지역화폐 — 사는 곳 자치구 안에서만)을 위해
            # 이 후보군이 거주 자치구 안인지 계산해 넣는다. 후보는 이 dong_code
            # (또는 그 자치구) 로 조회한 것이므로 추가 조회가 필요하지 않다.
            # 이것이 반영되지 않아 구 밖 가게에도 사용 표시가 붙었다(2026-09-18).
            _home_gu = str(persona.get("home_dong_code") or "")[:5]
            _same_gu = (bool(_home_gu)
                        and str(dong_code or "")[:5] == _home_gu)
            # [2026-10-06] 발행 구 상품권(P014): 어느 구 상품권이든 살 수 있어 직장 구도 쓸 수 있다(서울시 2019-12-19).
            if persona.get("voucher_scope") == "issuing_district":
                _work_gu = str(persona.get("work_dong_code") or "")[:5]
                _same_gu = str(dong_code or "")[:5] in {g for g in (_home_gu, _work_gu) if g}
            _rules = neutral_rules
            _spec = persona.get("poi_eligibility_spec")
            if _rules is None and _spec:
                try:
                    from eligibility import Rules as _ERules
                    _rules = _ERules(_spec)
                except Exception:
                    _rules = None
            coupon_boost = (
                coupon_active
                and os.environ.get("POLICY_POI_SORT_BOOST", "0") == "1"
            )
            # 상생 캐시백 활성 시 적립업종에 [적립] 사실 표시 (정렬 가점 아님 — 표시만)
            sangsaeng_active = bool(persona.get("sangsaeng_active"))
            for c in cands or []:
                # 판정은 고른 가게 자신의 업종으로 한다 — 상위 업종으로 넓혀 가져온 후보는 계획 업종과 다를 수 있다.
                _own_sub = c.get("poi_sub_category") or sub_cat
                _own_l1 = (c.get("poi_l1") or l1) if _own_sub != sub_cat else l1
                c["poi_sub_category"], c["poi_l1"] = _own_sub, _own_l1
                c["poi_same_gu"] = bool(_same_gu) if _home_gu else None
                c["poi_gu"] = str(dong_code or "")[:5] or None
                c["price_band"], c["price_factor"] = poi_price(c["poi_id"], dong_code, l1)
                c["unit_anchor"] = anchor_won
                c["durable_anchor"] = bool(_dur_anchor)
                if _pt.ON:
                    # 경험 상수(내구재 시세·가격대 배율)를 쓰지 않는다. 금액 근거는 실측 결제 1건당뿐.
                    c["price_ticket"] = _pt.info(dong_code, _own_sub)
                    c["durable_anchor"] = False
                    c["price_factor"] = 1.0
                    c["unit_anchor"] = unit_price_anchor(dong_code, l1)
                # 쿠폰 사용처 판정 — DB 백필값(p.coupon_eligible) 우선, 없으면 룰 fallback
                if _rules is not None:
                    el = _rules.eligible(c.get("name"), _own_sub, _own_l1,
                                         c.get("upjong_l3"), _same_gu)[0]
                else:
                    el = c.get("coupon_eligible")
                    if el is None:
                        el = is_coupon_eligible(c.get("name"), _own_sub, _own_l1)[0]
                c["coupon_eligible"] = bool(el)
                # 프롬프트 마커: 정책이 정한 표시. 활성 시에만 표기 (평시 토큰 0)
                c["coupon_tag"] = _mk if (coupon_active and c["coupon_eligible"]) else ""
                # 상생 적립 판정 — DB 백필값(p.sangsaeng_eligible) 우선, 없으면 룰 fallback
                sel = c.get("sangsaeng_eligible")
                if sel is None:
                    sel = is_sangsaeng_eligible(c.get("name"), _own_sub, _own_l1)[0]
                c["sangsaeng_eligible"] = bool(sel)
                # 프롬프트 마커: 캐시백 활성 시 적립업종만 표기 (평시 토큰 0)
                c["sangsaeng_tag"] = "[적립]" if (sangsaeng_active and c["sangsaeng_eligible"]) else ""
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
# 적립 문턱을 Stage2 의 금액 판단 자리로 보낼지 — experiments/plan_channel/s2_threshold.md
EXP_S2_THRESHOLD = os.environ.get("EXP_S2_THRESHOLD", "0") == "1"
# Stage1 이 붙인 trigger 를 Stage2 이벤트 줄로 넘길지 —
# experiments/plan_channel/s2_trigger_candidate.md
# **reasoning 은 안 넘긴다.** 거기엔 "캐시백까지 받겠다" 같은 문장이 있어
# 방향을 지시하지 않아도 문장 자체가 미는 힘을 갖는다. trigger 는 에이전트가
# 고른 분류 낱말 하나(policy·lifestyle·appointment·rumor·mood)라 그 위험이 작다.
EXP_S2_TRIGGER = os.environ.get("EXP_S2_TRIGGER", "0") == "1"


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
- **결제 하나하나를 무엇으로 낼지는 이 사람이 정한다.** 쓸 수 있는 매장이라고 자동으로 이 지갑에서
  나가지 않는다. `policy_spend`에 적은 금액이 실제로 이 지갑에서 나간 돈이다.
- 판단의 근거는 **남은 금액이 내 씀씀이로 며칠치인가** 하나다(정책 지갑 정보에 적혀 있다).
  여러 주치가 되는 금액이면 아껴 둘 이유가 없다 — 쓸 수 있는 자리에서 자기 돈을 먼저 낼 까닭이
  없으니 대체로 이 지갑으로 낸다. 반대로 며칠치에 그치는 금액이면 자리마다 다 얹으면 금방 사라진다.
  기한이 가을 끝까지로 남아 있으니 서둘러 없앨 이유가 없어, 꼭 필요한 결제에서만 꺼내 쓰게 된다.
- **결제 금액이 작다는 것은 건너뛸 이유가 되지 않는다.** 이 지갑은 큰 결제를 위해 아껴 두는
  상품권이 아니라 끼니·장보기·약값처럼 매일 하는 결제에 그대로 얹혀 나가는 돈이다.
  "소액이니 평소 쓰던 카드로"라는 습관을 이유로 이 지갑을 건너뛰지 않는다.
- 왜 그렇게 냈는지는 `pick_reason`에 자연스럽게 적는다.
- 한 결제에서 일부만 이 지갑으로 낼 수도 있다. `sum(policy_spend) ≤ actual_spent`이고, 정책별
  잔액을 넘을 수 없다.

**지원금으로 결제한 건마다 `would_buy_anyway` (true/false)**
이 지갑으로 낸 결제 하나하나에 대해, **지원금이 없었어도 이 지출을 했을지**를 판단해 적는다.
- `true`  — 지원금이 없었어도 내 돈으로 어차피 했을 지출이다.
- `false` — 지원금이 있어서 비로소 하게 된 지출이다. 없었으면 오늘 하지 않았을 것이다.
판단의 기준은 **그 지출이 원래 예정돼 있었는지**다. 없으면 안 되어서 어차피 했을 지출이면
결제수단만 바뀐 것이므로 `true`다. 반대로 오늘 이 돈이 있어서 비로소 하기로 한 것이면 `false`다.
업종으로 정해지는 것이 아니다 — 같은 병원 진료도 원래 가려던 날이면 `true`, 이 돈이 생겨
앞당긴 것이면 `false`다. 같은 외식도 늘 하던 끼니면 `true`, 안 하려던 것을 하게 된 것이면
`false`다. 이 사람의 평소 지출 습관에 그 지출이 들어 있었는지를 보고 건별로 정한다.
위 '평소 업종별 지출 구성'과 견주어 본다. 오늘 지출이 그 구성 안에서 늘 하던 만큼이면
어차피 했을 지출(`true`)이고, 평소 그 업종에 거의 쓰지 않던 사람이 오늘 쓴 것이거나 평소보다
훨씬 큰 금액이면 이 돈이 있어서 하게 된 지출(`false`)이다.
이 판단은 품목의 종류만으로 정하지 않는다. **같은 품목도 사람에 따라 다르다.** 평소 필요한 것을
그때그때 사 오던 사람이면 오늘 결제도 대부분 어차피 했을 지출이고, 쓸 돈이 빠듯해 미뤄둔 것이
쌓여 있던 사람이면 그중 상당수가 이 돈이 아니었으면 오늘 하지 않았을 지출이다. 이 사람의 형편과
평소 씀씀이를 보고 건별로 정한다.
지원금으로 결제하지 않은 건에는 이 필드가 무의미하다(생략).
- 사용처 제한 정책은 후보에 `[쿠폰]` 표시가 있는 매장에서만 사용할 수 있다. 표시가 없는 매장은 개인 잔액으로만 결제한다.
- 정책 존재만으로 소비 필요, 소비액, POI 선택을 미리 정하지 않는다. 페르소나의 필요·습관·자산·일정과 후보 특성을 함께 고려해 판단한다.
- 모든 commerce 이벤트에 양의 actual_spent를 반드시 부여 (0원·음수 금지).

**그 결제에서 `extra_spent` (원)**
같은 결제 안에도 평소 쓰던 만큼이 있고, 이 돈이 있어서 더 쓴 만큼이 있다. `extra_spent`는
그 결제 금액 중 **정책지갑이 없었다면 쓰지 않았을 금액**이다.
- 원래 하려던 지출을 결제수단만 바꾼 것이면 `0`이다.
- 늘 가던 곳인데 오늘은 이 돈이 있어 평소보다 좋은 것을 고르거나 양을 늘렸다면,
  **그 늘어난 금액만** 적는다(결제액 전부가 아니다).
- 이 돈이 없었으면 오늘 아예 하지 않았을 지출이면 결제 금액 전부를 적는다.
`0 ≤ extra_spent ≤ actual_spent`. `would_buy_anyway`가 `false`면 대개 결제액 전부이고,
`true`여도 평소보다 더 쓴 것이 있으면 `0`이 아니다.


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
    "policy_spend": {"P009": 12000},
    "would_buy_anyway": true,
    "extra_spent": 2000,
    "actual_satisfaction": 0.71,
    "pick_reason": "단골 한식집. 어제 sat 0.72로 만족도 높음. 직장 0.05km. 평소 한식 즐겨 찾는 성향. [쿠폰] 매장이고 남은 지원금이 몇 주치라 굳이 자기 돈 쓸 것 없이 정책지갑으로 계산.",
    "pick_factor": "satisfaction"
  },
  {
    "order": 2,
    "poi_id": "C_yyyyyy",
    "actual_spent": 25000,
    "policy_spend": null,
    "actual_satisfaction": 0.68,
    "pick_reason": "오늘 카페 휴식 의도와 가까운 후보가 맞음. 계산할 때는 습관대로 늘 쓰던 카드를 먼저 꺼내 지원금은 그대로 뒀다.",
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


def _build_neutral_stage2_system() -> str:
    """Preserve the legacy spatial contract while removing its grant assumptions."""
    start = "**소비액 설정 (actual_spent + policy_spend)**"
    end = "**만족도 설정 (actual_satisfaction)**"
    assert SYSTEM_S2.count(start) == SYSTEM_S2.count(end) == 1
    before, rest = SYSTEM_S2.split(start, 1)
    _, after = rest.split(end, 1)
    neutral_money = """**소비액과 정책 결제**
- 실제 방문·금액·결제수단은 오늘의 필요, 평소 습관, 잔액, 일정, 후보 가격과 정책 블록의 적용 조건을 함께 보고 건별로 정합니다. 정책이 있다는 사실만으로 구매나 방문을 만들지 않습니다.
- `actual_spent`는 거래 총액(원)입니다. 모든 commerce 이벤트에 양수를 적습니다.
- `policy_spend`는 별도로 지급된 정책 지갑에서 이 거래에 실제로 사용하기로 한 금액만 `{정책ID: 금액}`으로 적습니다. 그런 지갑이 없거나 쓰지 않으면 null 또는 빈 객체로 적습니다. 나중에 돌려받는 혜택이나 가격 할인액을 정책 지갑 지출로 적지 않습니다.
- 정책 지갑을 쓴다면 입력에 나온 해당 정책 ID·잔액·사용 자격·장소 조건을 확인합니다. 정책별 사용액은 잔액을 넘지 않고 합계는 `actual_spent`를 넘지 않습니다. 후보의 자격 표시는 그 후보에만 적용합니다.
- `would_buy_anyway`는 이 정책이 없었어도 오늘 이 구매를 했을지에 대한 건별 판단입니다. `extra_spent`는 같은 조건에서 오늘 쓰지 않았을 것으로 판단한 거래액 부분이며 0 이상 `actual_spent` 이하입니다. 정책과 무관하거나 근거가 없으면 null로 둡니다. 이 두 자기보고값을 먼저 정해 구매·장소를 맞추지 않습니다.
- `pick_reason`에는 시민의 구체적인 필요·예산·기억·후보 특성과, 실제 관련될 때만 정책 조건을 적습니다. 입력에 없는 사용처나 결과를 만들지 않습니다.

"""
    text = before + neutral_money + end + after
    example_start = '## 출력 형식 (JSON만, 다른 텍스트 금지)'
    example_end = 'pick_factor enum 정의'
    assert text.count(example_start) == text.count(example_end) == 1
    prefix, suffix = text.split(example_start, 1)
    _, tail = suffix.split(example_end, 1)
    example = """## 출력 형식 (JSON만, 다른 텍스트 금지)
{"picks": [
  {"order": 0, "poi_id": "C_xxxxxx", "actual_spent": 12000,
   "policy_spend": null, "would_buy_anyway": null, "extra_spent": null,
   "actual_satisfaction": 0.71,
   "pick_reason": "오늘 필요한 방문에 맞고 가깝고 평소 예산에도 맞음.",
   "pick_factor": "distance"}
], "review_lookup_requests": []}

"""
    return prefix + example + example_end + tail


SYSTEM_S2_NEUTRAL = _build_neutral_stage2_system()


_AUTO_PAY_OLD = (
    "- 실제 방문·금액·결제수단은 오늘의 필요, 평소 습관, 잔액, 일정, 후보 가격과 정책 블록의 적용 조건을 "
    "함께 보고 건별로 정합니다. 정책이 있다는 사실만으로 구매나 방문을 만들지 않습니다.\n")
_AUTO_PAY_NEW = (
    "- 실제 방문·금액은 오늘의 필요, 평소 습관, 잔액, 일정, 후보 가격과 정책 블록의 적용 조건을 "
    "함께 보고 건별로 정합니다. 정책이 있다는 사실만으로 구매나 방문을 만들지 않습니다.\n"
    "- 결제: 정책 지갑이 있는 시민이 그 지갑의 사용 가능 매장에서 결제하면 지갑에서 자동으로 먼저 "
    "빠져나가고 모자란 만큼만 본인 돈으로 냅니다. 이 경우 `policy_spend`는 null 로 두어도 됩니다.\n")


_PRICE_SECTION_NEW = """**무엇을 얼마에 (핵심)**
- 결제마다 오늘 실제로 사거나 먹을 것을 `menu`에 짧게 적습니다(메뉴 이름이나 물건 이름).
- `unit_price`는 **한 사람 몫의 결제 금액(원)**입니다. 식당이면 1인분, 가게에서 여러 물건을 사면 그 물건 값의 합입니다.
  고른 가게와 이 사람이 아는 물가로 정합니다.
- `pay_count`는 **이 사람이 몇 명 몫을 계산하는지**입니다. 혼자면 1, 가족·일행 몫까지 내면 그 사람 수, 각자 내면 1.
  물건 개수가 아닙니다.
- `actual_spent`는 `unit_price × pay_count`입니다(엔진이 같은 식으로 다시 계산합니다).
- 돈을 내지 않는 방문(둘러보기·상담만 하고 나옴, 이번 달 이미 낸 곳을 이용만 함 등)은 `unit_price` 0 입니다.
"""
_TICKET_NOTE = """- 이벤트 제목의 '카드 결제 1건당'은 그 업종에서 실제 카드 결제 한 건의 평균(2025 서울 실측)과 서울 동별 하위 10%~상위 10% 범위입니다.
  여럿이 함께 낸 결제가 섞여 있어 1인분 가격이 아닙니다.
"""


def _price_mode_system(text: str) -> str:
    start, end = "**가격대와 예산 (핵심)**", "**단순 반복 억제**"
    assert text.count(start) == text.count(end) == 1
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    sec = _PRICE_SECTION_NEW + (_PRODUCE_NOTE if _PRODUCE_FIELD else "") + (_TICKET_NOTE if _pt.MODE == "ticket" else "")
    text = before + sec + "\n" + end + after
    # [2026-10-07b] 0원 방문을 금지하던 문장·만족도 예시 숫자·반복 금지 지시를 뺀다(본런 검수: 0원이 지어낸 금액으로,
    # 만족도가 0.71 근처 계단값으로, 어제 간 가게 재방문이 5~6% 로 눌렸다).
    text = text.replace("- `actual_spent`는 거래 총액(원)입니다. 모든 commerce 이벤트에 양수를 적습니다.\n",
                        "- `actual_spent`는 거래 총액(원)입니다.\n")
    text = text.replace('"actual_satisfaction": 0.71,', '"actual_satisfaction": <0~1, 이 방문이 실제로 어땠을지>,')
    text = text.replace('"pick_reason": "오늘 필요한 방문에 맞고 가깝고 평소 예산에도 맞음.",', '"pick_reason": "<이 가게를 고른 이유>",')
    text = text.replace("- 최근 3일 이내 방문한 POI(⚠️ 표시)는 특별한 사유 없이 재선택하지 마세요.\n",
                        "- ⚠️ 는 최근 3일 안에 간 곳이라는 표시입니다.\n")
    text = text.replace("- residence/workplace/집/직장 이벤트, pinned_poi 이벤트는 picks에 포함하지 않습니다.\n",
                        "- residence/workplace/집/직장 이벤트는 picks에 포함하지 않습니다. 약속 장소가 정해진 이벤트는 후보가 그 한 곳뿐입니다.\n")
    # 출력 예시의 금액 숫자(12000·25000)는 모델이 따라 쓰는 기준값이 된다 — 자리표시로 바꾼다.
    text = text.replace('"actual_spent": 12000,', '"menu": "<먹거나 살 것>", "unit_price": <한 사람 몫 결제 금액>, "pay_count": <몇 명 몫을 내는지>, "actual_spent": <unit_price×pay_count>,')
    text = text.replace('"actual_spent": 25000,', '"menu": "<먹거나 살 것>", "unit_price": <한 사람 몫 결제 금액>, "pay_count": <몇 명 몫을 내는지>, "actual_spent": <unit_price×pay_count>,')
    if _PRODUCE_FIELD:
        text = text.replace('"actual_spent": <unit_price×pay_count>,',
                            '"actual_spent": <unit_price×pay_count>, "produce_spent": <장보기면 그중 농축산물 금액, 아니면 null>,')
    text = text.replace('"policy_spend": {"P009": 12000},', '"policy_spend": {"P009": <지원금으로 낸 금액>},')
    text = text.replace('"extra_spent": 2000,', '"extra_spent": <이 돈 때문에 더 쓴 금액>,')
    return text


def active_stage2_system() -> str:
    """Historical variants keep their original Stage2 prompt byte for byte.

    결제 규칙이 사용처 자동 차감인 정책(EXP_PAYMENT_CHOICE=0)에서는 중립 프롬프트의
    결제 문장만 그 규칙으로 바꾼다. 기본값에서는 아무것도 바뀌지 않는다.
    """
    text = SYSTEM_S2_NEUTRAL if active_stage2_is_neutral() else SYSTEM_S2
    if _pt.ON:
        text = _price_mode_system(text)
    from mechanisms import payment_choice_mode
    if not payment_choice_mode() and text is SYSTEM_S2_NEUTRAL:
        assert text.count(_AUTO_PAY_OLD) == 1
        text = text.replace(_AUTO_PAY_OLD, _AUTO_PAY_NEW, 1)
    return text


def active_stage2_is_neutral() -> bool:
    """Use the variant's declared Stage2 contract, not its version label."""
    from prompts import get
    return bool(getattr(get(), "STAGE2_NEUTRAL", False))


def _format_event_with_candidates(
    i: int, ev, cands: list[dict], recent_poi_ids: set[str] | None = None,
    modeled_prices: bool = False,
) -> str:
    if not cands:
        return ""
    # 동네×업종 평균 결제단가(실측 카드 데이터 기반) — actual_spent 스케일 앵커
    anchor = cands[0].get("unit_anchor")
    # 내구재 앵커는 동네 평균이 아니라 그 물건의 시세다 — 라벨을 구분한다.
    if anchor and cands[0].get("durable_anchor"):
        anchor_s = f" | 바꾸려는 물건 시세 ~{anchor:,}원"
    elif anchor:
        anchor_s = f" | 동네 평균단가 ~{anchor:,}원"
    else:
        anchor_s = ""
    # Stage1 이 왜 이 이벤트를 넣었는지 — 한 낱말만 넘긴다. 새 사실이 아니라
    # 이미 적힌 값을 다음 단계로 보내는 것이다(s2_trigger_candidate.md).
    trig_s = ""
    if EXP_S2_TRIGGER:
        _t = str(getattr(ev, "trigger", "") or "")
        if _t and _t != "none":
            trig_s = f" | 계기:{_t}"
    if _pt.ON:
        anchor_s = _pt.label(cands[0].get("price_ticket")) if _pt.MODE == "ticket" else ""
    if modeled_prices:
        anchor_s = anchor_s.replace('동네 평균단가', '모형 참고단가').replace('바꾸려는 물건 시세', '모형 물품 참고단가')
    lines = [
        f"### 이벤트 {i} | {ev.time} | {ev.anchor} | "
        f"{ev.category}/{ev.sub_category or _guess_sub_from_l1(ev.category)} | {ev.intent}{trig_s}{anchor_s}"
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
        price_s = "" if _pt.ON else price_icon(c.get("price_band"))   # 가격 판: 가격대는 POI id 해시라 보이지 않는다
        coupon_s = c.get("coupon_tag") or ""
        sangsaeng_s = c.get("sangsaeng_tag") or ""

        lines.append(
            f"  {known_mark}{recent_mark} {c['poi_id']} | {c.get('name') or '(이름없음)'} | "
            f"{km_s} | {price_s} | {sat_s} {visit_s}{coupon_s}{sangsaeng_s}"
        )
    return "\n".join(lines)


def build_stage2_prompt(
    events: list,
    cands_by_order: dict[int, list[dict]],
    persona: dict | None = None,
    recent_poi_ids: set[str] | None = None,
    state: dict | None = None,
    active_policies: list[dict] | None = None,
    today: date | None = None,
    neutral: bool = False,
) -> str:
    # 페르소나 헤더
    grounded_experiment = bool((persona or {}).get('_no_smoking_prompt'))
    header_parts = []
    if grounded_experiment:
        from personal_context import render_personal_context
        header_parts.append('## 에이전트 정보\n'+render_personal_context(
            persona, state if state is not None else {}, include_smoking_rule=False))
    if persona and not grounded_experiment:
        daily_wd = persona.get("daily_wd") or 0
        daily_we = persona.get("daily_we") or 0
        tendency = persona.get("tendency") or ""
        lifestyle = (persona.get("lifestyle") or "").strip()
        income = persona.get("income") or ""
        budget_info = f"평소 1일 소비규모(스케일 참고, 총액 아님): 평일 {daily_wd:,}원 / 주말 {daily_we:,}원"
        if _pt.ON and today is not None:
            from kr_holidays import is_day_off as _off
            budget_info = f"평소 {'쉬는 날' if _off(today) else '평일'} 하루 소비규모(실측, 참고): {(daily_we if _off(today) else daily_wd):,}원"
        # [적립 문턱을 금액 판단 자리로 보낸다] experiments/plan_channel/s2_threshold.md
        # 금액(actual_spent)은 **여기서** 정해지는데 문턱 정보는 Stage1 에만 있었다.
        # 새 사실이 아니라 Stage1 이 이미 받은 그 줄을 그대로 옮긴다 — 방향도
        # 목표 수치도 붙이지 않는다. 받아들이는 방식은 형편에 달렸다고만 적는다.
        if EXP_S2_THRESHOLD:
            _ss = (
                (persona.get("sangsaeng_status_line") or "") if not neutral else ""
            ).strip().lstrip("- ")
            if _ss:
                budget_info += (
                    "\n적립 정책 상태: " + _ss +
                    "\n  (돌아오는 돈은 다음 달이므로 오늘 쓸 수 있는 돈이 는 것은 아니다."
                    " 어차피 할 지출로 문턱이 저절로 넘어가는 사람도, 넘길 일이 없어"
                    " 신경 쓰지 않는 사람도 있다.)"
                )
        # 가용 자산 — 가격대(₩~₩₩₩) 선택의 예산 근거
        balance = (state or {}).get("balance")
        if balance is not None:
            budget_info += f" / 현재 잔액: {int(balance):,}원"
        # 평소 업종별 지출 구성(BDC 실측) — would_buy_anyway 판정의 근거.
        # 오늘 지출이 이 사람의 평소 패턴 안이었는지 밖이었는지를 볼 수 있어야 한다.
        # Stage1 과 같은 줄을 쓴다 — 어휘가 갈리면 두 단계가 다른 기준으로 움직인다.
        _cat = ""
        try:
            from bdc_category_map import line as _fold_line
            _b = _fold_line(persona.get("cat_ratio_wd"))
            if _b:
                _cat = "\n" + _b
        except Exception:
            _cat = ""
        # 집안 내구재 상태 — Stage1 이 '냉장고 교체'를 계획했을 때 Stage2 가
        # actual_spent 를 그 물건의 시세로 잡을 수 있어야 한다. 동네 평균단가
        # 앵커(쇼핑 ~5만원)만 보면 90만원짜리 냉장고가 나올 수 없다.
        _dur = ""
        try:
            from durables import format_block as _durfmt
            _dur = _durfmt(persona.get("id") or "", persona.get("life_stage"),
                           persona.get("age_group"))
            if _dur:
                _dur = "\n" + _dur
        except Exception:
            _dur = ""
        # [2026-10-06] 오늘이 어떤 날인지 — 2단계는 날짜를 몰라 평일에도 '주말 예산'으로 금액을 잡는 일이
        # 약 1% 있었다(가게 이유에 '예산(주말 58,388원)'). 1단계와 같은 판정(kr_holidays)으로 사실만 적는다.
        _today_line = ""
        if today is not None:
            from kr_holidays import holiday_name as _hn
            _h = _hn(today)
            _kind = f"공휴일({_h})" if _h else ("주말" if today.weekday() >= 5 else "평일")
            _today_line = f"오늘: {today.isoformat()} ({'월화수목금토일'[today.weekday()]}요일, {_kind})\n"
        _monthly = ""
        if persona.get("monthly_paid") is not None:
            # 학원비·헬스장 회원권은 달마다 한 번 낸다. 이번 달 이미 냈으면 오늘은 이용만(0원).
            _mp = set(persona.get("monthly_paid") or ())
            _subs = sorted({str(getattr(e, "sub_category", "") or "") for e in (events or [])} & {"학원", "기타교육", "헬스장"})
            if _subs:
                def _due(x):
                    _i = _pt.info(persona.get("home_dong_code") or None, x) if _pt.ON else None
                    _r = (f" (서울 {x} 카드 결제 1건, 2025 실측 하위 10%~상위 10%: {int(_i['p10']):,}~{int(_i['p90']):,}원)"
                          if _i and _i.get('p10') else "")
                    return '오늘 이번 달 치를 낼 차례' + _r
                _monthly = "\n월 납부: " + " / ".join(
                    f"{x} — {'이번 달 이미 냄(오늘은 이용만, 0원)' if x in _mp else _due(x)}" for x in _subs)
        header_parts.append(f"## 에이전트 정보\n{lifestyle}\n{_today_line}{budget_info} / 소비성향: {tendency} / 소득분위: {income}{_cat}{_dur}{_monthly}")
    if persona:
        if neutral:
            policies = active_policies or []
            status = (
                _format_policy_status(
                    policies, policy_used=_json_dict((state or {}).get("policy_used")),
                    persona=persona, state=state, today=today,
                ) if policies else "(오늘 적용 정책 없음)"
            )
            header_parts.append(
                "## 오늘 활성 정책 — 공통 사실\n"
                + _format_policy_facts(policies)
                + "\n\n## 나에게 적용되는 정책 상태\n" + status
            )
        else:
            # Historical variants retain their original wallet-oriented block.
            policy_budget = persona.get("policy_budget_summary") or ""
            if policy_budget:
                header_parts.append(f"## 활성 정책 (policy_spend 책정 시 참조)\n{policy_budget}")
        from no_smoking_context import configured_context, prompt_for_persona
        smoking_context = prompt_for_persona(persona)
        if smoking_context:
            header_parts.append(f"## 흡연 상태와 시설 이용 규칙\n{smoking_context}")
            smoking_runtime = configured_context()
            if smoking_runtime:
                facts = smoking_runtime.candidate_facts(
                    c["poi_id"] for candidates in cands_by_order.values() for c in candidates)
                if facts:
                    header_parts.append(f"## 후보 시설 분류\n{facts}")

    if recent_poi_ids:
        from no_smoking_context import configured_context
        recent_ordered = sorted(recent_poi_ids) if configured_context() else list(recent_poi_ids)
        header_parts.append(
            ("## 최근 3일 방문 POI (반복 방문 여부는 입력 상황으로 판단)\n" if grounded_experiment
             else ("## 최근 3일 방문 POI (⚠️ 표시)\n" if _pt.ON else "## 최근 3일 방문 POI (⚠️ 표시 — 단순 반복 자제)\n"))
            + ", ".join(recent_ordered[:20])
        )

    blocks = []
    for i, ev in enumerate(events):
        cs = cands_by_order.get(i) or []
        if not cs:
            continue
        blocks.append(_format_event_with_candidates(i, ev, cs, recent_poi_ids, modeled_prices=grounded_experiment))

    if not blocks:
        return "(외부 POI 결정 필요한 이벤트 없음)"

    header = "\n\n".join(header_parts) + "\n\n" if header_parts else ""
    body = "\n\n".join(blocks)
    return (
        f"{header}"
        f"다음 이벤트별 candidates 중에서 POI를 선택하고 소비액·만족도를 설정하세요.\n\n"
        f"{body}\n\n"
        + ("각 이벤트의 order·poi_id·menu·unit_price·pay_count·actual_spent·actual_satisfaction·pick_reason·pick_factor를 JSON으로 출력하세요."
           if _pt.ON else "각 이벤트의 order·poi_id·actual_spent·actual_satisfaction·pick_reason·pick_factor를 JSON으로 출력하세요.")
        + (" 각 pick의 evidence_ref에는 입력의 [E0001] 형태로 표시된 사실 줄 번호를 넣으세요."
           " 프로그램이 해당 원문을 evidence_quote로 기록하므로 evidence_quote는 출력하지 마세요."
           if grounded_experiment else " /no_think")
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
    # v53 reads policy facts from active_policies; older variants keep the legacy summary.
    active_policies: list[dict] | None = None,
    grant_remaining: dict[str, int] | None = None,  # noqa: ARG001
    decision_context: DawnContext | None = None,
) -> tuple[Stage2Output, dict[int, list[dict]], dict]:
    """Stage 2 LLM 호출. (picks, 사용된 candidates, meta) 반환.

    today: 오늘 날짜. desire 계산의 days_since_visit 산출에 사용.
    state: State 노드 dict (balance 등) — 가격대 선택의 예산 근거로 프롬프트에 노출.
    decision_context: opt-in 기록용 Dawn 상태. 경량 결과는 실제 결정에 반영하지 않음.
    """
    total_started = time.perf_counter()
    grounded_experiment = bool(persona.get('_no_smoking_prompt'))
    from grounded_schema import canonical_evidence_ref, rejected_response_feedback, stage2_format
    from execution_errors import fatal_dispatch_error
    # One follow-up is needed when a valid grounded response selects only a
    # subset of required orders. Keep the earlier partial picks and ask solely
    # for the missing orders; the agent-day dispatcher bounds full retries.
    retry_limit = max(max_retry, 1) if grounded_experiment else max_retry
    system_prompt = SYSTEM_STAGE2 if grounded_experiment else SYSTEM_S2
    if grounded_experiment:
        system_prompt = system_prompt.replace(
            '- 각 외출 이벤트의 order에 대해 해당 이벤트의 후보 poi_id 중 하나만 선택한다.',
            '- 이번 응답에서 요청한 order에 대해서만 해당 이벤트의 후보 poi_id 중 하나를 선택한다. '
            '전체 이벤트 목록은 문맥 자료이며, 이미 검증된 order는 다시 선택하지 않는다.',
        )
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
    # POI → (가게 자신의 업종, 그 상위 업종, 가게 이름, 세부업종 코드) — 즉시 할인 판정이 계획 업종이 아니라 이것을 쓴다.
    poi_cat_by_poi: dict[str, tuple] = {
        c["poi_id"]: (c.get("poi_sub_category"), c.get("poi_l1"), c.get("name"), c.get("upjong_l3"),
                      c.get("poi_same_gu"), c.get("poi_gu"))
        for cs in cands_by_order.values() for c in cs
    }
    timing["t_price_maps"] = time.perf_counter() - started

    if not need_llm:
        # 외부 POI 결정 필요 없음 (전부 residence/workplace/pinned)
        return Stage2Output(picks=[]), cands_by_order, {
            "skipped": True,
            "price_by_poi": price_by_poi,
            "coupon_by_poi": coupon_by_poi,
            "poi_cat_by_poi": poi_cat_by_poi,
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
                "WHERE m.day >= date($since) AND m.day < date($today) RETURN p.id AS pid",
                aid=aid, since=three_days_ago, today=today.isoformat()
            )
            recent_poi_ids = {r["pid"] for r in rows}
    except Exception:
        pass
    timing["t_recent_memory"] = time.perf_counter() - started

    started = time.perf_counter()
    # [2026-10-06] grounded(금연 실험)는 위에서 고른 SYSTEM_STAGE2 를 지킨다 — 예전에는 여기서 덮어썼다(doinggyu 검토 3.4).
    if not grounded_experiment:
        system_prompt = active_stage2_system()
    neutral_stage2 = active_stage2_is_neutral()
    user_block = build_stage2_prompt(
        stage1.events, cands_by_order,
        persona=persona,
        recent_poi_ids=recent_poi_ids,
        state=state,
        active_policies=active_policies,
        today=today,
        neutral=neutral_stage2,
    )
    if grounded_experiment:
        user_block = _number_evidence_lines(
            user_block,
            exclude_prefixes=('다음 이벤트별 candidates', '각 이벤트의 order'),
        )
    evidence_lines = _evidence_lines(user_block) if grounded_experiment else {}
    if grounded_experiment:
        # This is an unnumbered instruction, so existing evidence IDs stay put.
        user_block = user_block.replace(
            '각 이벤트의 order·poi_id·actual_spent·actual_satisfaction·pick_reason·pick_factor를 JSON으로 출력하세요.',
            '이번 응답에서 요청한 order의 poi_id·actual_spent·actual_satisfaction·pick_reason·pick_factor만 JSON으로 출력하세요.',
        )
    if grounded_experiment and not evidence_lines:
        raise ValueError('Stage2 grounded prompt has no factual evidence lines')
    timing["t_prompt_build"] = time.perf_counter() - started

    # 환각 차단용 JSON schema — poi_id는 전체 후보풀 union enum 강제.
    # order-별 enum은 아니지만 후보풀 외 POI는 0건 보장. order_mismatch는 fallback에서 처리.
    started = time.perf_counter()
    all_pids = sorted({c["poi_id"] for cs in cands_by_order.values() for c in cs})
    expected_orders = [
        i for i, ev in enumerate(stage1.events)
        if ev.category not in INTERNAL_CATS and (not ev.pinned_poi or _pt.ON) and cands_by_order.get(i)
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
                                    "would_buy_anyway": {"type": ["boolean", "null"]},
                                    "extra_spent": {"type": ["integer", "null"], "minimum": 0},
                                    "pick_reason": {"type": ["string", "null"]},
                                    "pick_factor": {"type": ["string", "null"]},
                                    **({"menu": {"type": "string"},
                                        "unit_price": {"type": "number", "minimum": 0},
                                        "pay_count": {"type": "integer", "minimum": 1, "maximum": 8}}
                                       if _pt.ON else {}),
                                    **({"produce_spent": {"type": ["integer", "null"], "minimum": 0}}
                                       if _PRODUCE_FIELD else {}),
                                },
                                # policy_spend·would_buy_anyway를 필수로 둔다. 선택 필드였을 때 모델이 대부분 생략했고
                                # (측정: 40만 tier가 5일 중 4일 쿠폰 0건), 생략은 곧 "미사용"으로 처리돼
                                # 결제 판단이 이뤄지지 않았다. null을 허용하므로 "안 쓴다"도 표현 가능하며,
                                # 다만 건별로 명시적으로 답해야 한다.
                                # [되돌림] policy_spend를 필수로 두니 모델이 건별로 판단은 하되
                                # 사용 수준이 0.42로 뛰어 전체 소진이 7.3%/일(목표 2.73)로 과속했고
                                # MPC도 0.174→0.085로 무너졌다. 선택 필드로 되돌린다.
                                # [2026-10-11] produce_spent 는 null 을 허용하되 필수로 둔다 — 선택 필드는 모델이 생략한다(위 주석).
                                "required": ((["order", "poi_id", "actual_satisfaction", "actual_spent", "menu", "unit_price", "pay_count"]
                                              if _pt.ON else ["order", "poi_id", "actual_satisfaction", "actual_spent"])
                                             + (["produce_spent"] if _PRODUCE_FIELD else [])),
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
        if grounded_experiment:
            pick_schema = s2_schema["json_schema"]["schema"]["properties"]["picks"]["items"]
            pick_schema["properties"]["evidence_ref"] = {
                "type": "string", "enum": sorted(evidence_lines),
            }
            pick_schema["required"].append("evidence_ref")
    timing["t_schema_build"] = time.perf_counter() - started

    # Opt-in capture uses the exact pre-decision candidates and prompt. It may
    # observe a small model in shadow mode, but never replaces the teacher pick.
    # Deep copies keep logging/shadow failures or mutations out of simulation state.
    capture = None
    capture_error = None
    fast_mode = os.environ.get("SIM_FAST_MODE", "off").strip().lower()
    if fast_mode != "off":
        started = time.perf_counter()
        try:
            from fast_decision.runtime import start_capture

            capture = start_capture(
                aid=aid, today=today,
                stage1=stage1.model_copy(deep=True),
                persona=copy.deepcopy(persona), state=copy.deepcopy(state),
                candidates=copy.deepcopy(cands_by_order),
                system_prompt=system_prompt, user_prompt=user_block,
                recent_poi_ids=set(recent_poi_ids),
                active_policies=copy.deepcopy(active_policies),
                grant_remaining=copy.deepcopy(grant_remaining),
                context=copy.deepcopy(decision_context),
            )
            if capture is None:
                capture_error = {"mode": fast_mode, "status": "capture_unavailable",
                                 "applied": False, "teacher_preserved": True}
        except Exception as exc:
            capture_error = {"mode": fast_mode, "status": "capture_error",
                             "error_type": type(exc).__name__, "teacher_preserved": True}
        timing["t_fast_capture"] = time.perf_counter() - started

    def finish_capture(output: Stage2Output, meta: dict) -> dict:
        if capture is not None:
            started = time.perf_counter()
            try:
                import copy

                meta["acceleration"] = capture.finish(
                    output.model_copy(deep=True), copy.deepcopy(meta),
                )
            except Exception as exc:
                meta["acceleration"] = {
                    "mode": fast_mode, "status": "capture_error",
                    "error_type": type(exc).__name__, "teacher_preserved": True,
                }
            timing["t_fast_finish"] = time.perf_counter() - started
            meta["s2_timing"] = timing_snapshot()
        elif capture_error is not None:
            meta["acceleration"] = capture_error
        return meta

    last_err = None
    collected_picks: dict[int, Stage2Pick] = {}
    raw = None
    review_lookup_used: dict[str, dict] = {}  # 첨부됐던 lookup 결과 (meta 출력용)
    pre_review_picks: dict[int, str] = {}     # 리뷰 보기 전(1차) 선택 {order: poi_id} — 사고변화 추적
    prior_output_limited = False
    total_tokens_in = 0
    total_tokens_out = 0
    for attempt in range(retry_limit + 1):
        previous_raw = raw
        temp = (0.2 if attempt == 0 else (0.1 if attempt < 3 else 0.3)) if grounded_experiment else 0.7 + 0.1 * attempt
        token_cap = 3200 if prior_output_limited else 2200
        attempt_started = time.perf_counter()
        effective_temp = 0.0 if os.environ.get('POLICY_BACKTEST_DETERMINISTIC') == '1' else temp
        attempt_timing: dict[str, float | int | str] = {
            "attempt": attempt, "temp_requested": temp, "temp": effective_temp,
        }
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
        response_schema = s2_schema
        if grounded_experiment and s2_schema is not None:
            remaining_orders = [order for order in expected_orders if order not in collected_picks]
            response_schema = stage2_format(s2_schema, remaining_orders, cands_by_order, evidence_lines)
            prompt_now = ('[이번 응답에서 요청한 order] ' + ', '.join(map(str, remaining_orders))
                          + '\n이 order만 각각 한 번 출력하세요. 아래 전체 후보 목록은 문맥 자료입니다.\n\n'
                          + prompt_now)
            if collected_picks:
                prompt_now += (
                    '\n\n이미 검증된 선택은 반복하지 마세요. 이번 picks에는 누락된 order '
                    + ', '.join(map(str, remaining_orders))
                    + '만 출력하고 각각 짧은 선택 이유와 입력 근거를 적으세요.'
                )
            if attempt > 0 and last_err:
                correction = f'재시도 {attempt}/{retry_limit}입니다. '
                bad_order = re.search(r'order (\d+): poi_id must be from that order candidates', str(last_err))
                if bad_order:
                    order = int(bad_order.group(1))
                    allowed = [c['poi_id'] for c in cands_by_order.get(order, [])]
                    correction += f"order {order}에 허용된 poi_id는 다음뿐입니다: {', '.join(allowed)}. "
                prompt_now += rejected_response_feedback(
                    previous_raw, last_err,
                    correction + '이번 응답에서 허용된 남은 order는 '
                    + ', '.join(map(str, remaining_orders)) + '입니다. '
                    + '이미 검증된 order는 다시 출력하지 말고, 각 order를 한 번씩만 쓰세요. '
                    + '해당 order의 후보 poi_id와 입력의 E0001 형식 evidence_ref 하나만 사용하세요.',
                )
        if _pt.ON and not grounded_experiment and attempt > 0 and last_err and previous_raw:
            _corr = f'재시도 {attempt}/{retry_limit}입니다. '
            _bo = re.search(r'order (\d+): poi_id must be from that order candidates', str(last_err))
            if _bo:
                _o = int(_bo.group(1))
                _corr += f"order {_o}에 허용된 poi_id는 다음뿐입니다: {', '.join(c['poi_id'] for c in cands_by_order.get(_o, []))}. "
            _corr += '외출 order 마다 pick 을 하나씩 빠짐없이 쓰고, poi_id 는 그 order 의 후보에서만, unit_price·pay_count 는 정수로 쓰세요.'
            prompt_now += rejected_response_feedback(previous_raw, last_err, _corr)
        elapsed = time.perf_counter() - started
        timing["t_retry_prompt"] += elapsed
        attempt_timing["t_retry_prompt"] = elapsed
        attempt_timing["max_tokens"] = token_cap
        error_stage = "llm"
        finish = None
        tokens_out = 0
        raw = None
        try:
            started = time.perf_counter()
            resp = _llm_call(
                None, system_prompt, prompt_now,
                temperature=temp, max_tokens=(2400 if grounded_experiment else token_cap),
                response_format=response_schema,
            )
            elapsed = time.perf_counter() - started
            timing["t_llm"] += elapsed
            timing[f"t_llm_{call_kind}"] += elapsed
            timing["n_llm_calls"] += 1
            attempt_timing["t_llm"] = elapsed
            raw = resp.choices[0].message.content
            finish = getattr(resp.choices[0], "finish_reason", None)
            attempt_timing["finish_reason"] = finish
            attempt_timing["tokens_in"] = int(getattr(resp.usage, "prompt_tokens", 0) or 0)
            tokens_out = int(getattr(resp.usage, "completion_tokens", 0) or 0)
            attempt_timing["tokens_out"] = tokens_out
            total_tokens_in += attempt_timing["tokens_in"]
            total_tokens_out += tokens_out
            if verbose:
                print(f"--- attempt {attempt} (temp={temp}) ---")
                print(raw[:600])

            error_stage = "json_extract"
            started = time.perf_counter()
            # A length-terminated response can be syntactically valid while
            # omitting later events. Do not let the missing-pick filler turn
            # that partial generation into a successful citizen decision.
            if finish == "length":
                raise ValueError("Stage2 output reached the generation limit")
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
            if grounded_experiment:
                for pick in data.get('picks', []):
                    raw_ref = pick.get('evidence_ref')
                    ref = canonical_evidence_ref(raw_ref, evidence_lines)
                    if not isinstance(ref, str) or ref not in evidence_lines:
                        raise ValueError(
                            f'order {pick.get("order")}: evidence_ref {str(raw_ref)[:60]!r} must identify exactly one '
                            'numbered factual input line'
                        )
                    if ref != raw_ref:
                        attempt_timing['evidence_ref_format_repairs'] = (
                            attempt_timing.get('evidence_ref_format_repairs', 0) + 1
                        )
                    pick['evidence_ref'] = ref
                    pick['evidence_quote'] = evidence_lines[ref]
                    try:
                        validate_stated_reason(pick, user_block, reason_key='pick_reason')
                    except ValueError as exc:
                        raise ValueError(f'order {pick.get("order")}: {exc}') from exc
            parsed = Stage2Output.model_validate(data)
            elapsed = time.perf_counter() - started
            timing["t_model_validate"] += elapsed
            attempt_timing["t_model_validate"] = elapsed

            # === review_lookup_requests 처리 (한 번만, 첫 호출에서만) ===
            if not grounded_experiment and not review_lookup_used and parsed.review_lookup_requests:
                # 후보 풀 안에 있는 poi_id만 채택 (환각 방지)
                valid_lookup_ids = [pid for pid in parsed.review_lookup_requests
                                    if pid in set(all_pids)]
                # 총 LLM 호출 상한(max_retry+1)은 유지한다. 마지막 허용 호출에서
                # 리뷰를 요청하면 조회만 하고 최종판단을 못 하는 문제를 피하기 위해
                # 남은 호출 예산이 있을 때만 리뷰를 가져온다.
                if valid_lookup_ids and attempt < max_retry:
                    error_stage = "review_lookup"
                    started = time.perf_counter()
                    try:
                        fetched = lookup_reviews_batch(valid_lookup_ids[:8], max_reviews=3)
                    except Exception as review_error:
                        # Review is optional. A local SQLite fault is not a
                        # reason to discard the already valid LLM picks.
                        fetched = {}
                        fb_stats["review_lookup_error"] = fb_stats.get("review_lookup_error", 0) + 1
                        attempt_timing["review_lookup_error_type"] = type(review_error).__name__
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
            if grounded_experiment:
                actual_orders = [pick.order for pick in parsed.picks]
                remaining_orders = set(expected_orders) - collected_picks.keys()
                # An exact replay cannot revise an accepted decision. Ignore it
                # idempotently; changed or conflicting repeats are still errors.
                new_picks = []
                if len(actual_orders) == len(set(actual_orders)):
                    for pick in parsed.picks:
                        saved = collected_picks.get(pick.order)
                        if saved is not None and pick.model_dump() == saved.model_dump():
                            attempt_timing['already_saved_picks_ignored'] = (
                                attempt_timing.get('already_saved_picks_ignored', 0) + 1)
                        else:
                            new_picks.append(pick)
                    parsed = Stage2Output(picks=new_picks)
                    actual_orders = [pick.order for pick in new_picks]
                if (len(actual_orders) != len(set(actual_orders))
                        or not set(actual_orders).issubset(remaining_orders)):
                    raise ValueError(
                        'picks must contain each remaining order at most once; got '
                        + ', '.join(map(str, actual_orders)) + '; remaining '
                        + ', '.join(map(str, sorted(remaining_orders)))
                    )
                for pick in parsed.picks:
                    if pick.poi_id not in {c['poi_id'] for c in cands_by_order[pick.order]}:
                        raise ValueError(f'order {pick.order}: poi_id must be from that order candidates')
                collected_picks.update((pick.order, pick) for pick in parsed.picks)
                missing_orders = [order for order in expected_orders if order not in collected_picks]
                if missing_orders:
                    attempt_timing['partial_picks_saved'] = len(collected_picks)
                    raise ValueError('missing required orders: ' + ', '.join(map(str, missing_orders)))
                parsed = Stage2Output(picks=[collected_picks[order] for order in expected_orders])
            if _pt.ON and not grounded_experiment:
                # [2026-10-07b] 가격 판에서는 엔진이 가게·금액을 대신 정하지 않는다. 예전에는 후보 밖 가게를 상위 5개 중
                # 무작위 가게(만족도 0.5)로, 빠진 외출을 무작위 가게로 채우고 금액은 실측 하위 10%로 채웠다(결제의 약 7%).
                # 이제는 모델에게 다시 묻는다. 끝까지 못 하면 그 사람의 하루는 실패로 남는다(지어낸 하루보다 낫다).
                _bad = [p.order for p in parsed.picks
                        if p.poi_id not in {c['poi_id'] for c in cands_by_order.get(p.order, [])}]
                if _bad:
                    raise ValueError(f'order {_bad[0]}: poi_id must be from that order candidates')
                _have = {p.order for p in parsed.picks}
                _miss = [i for i, ev in enumerate(stage1.events)
                         if ev.category not in INTERNAL_CATS and cands_by_order.get(i) and i not in _have]
                if _miss:
                    raise ValueError('missing required orders: ' + ', '.join(map(str, _miss)))
                _noprice = [p.order for p in parsed.picks if p.unit_price is None or p.pay_count is None]
                if _noprice:
                    raise ValueError(f'order {_noprice[0]}: unit_price and pay_count are required')
            import random as _random
            corrected_picks = []
            hallucinations = 0          # 보정 (해당 order의 cands에 없지만 cands는 존재)
            hallucinations_dropped = 0  # 드롭 (해당 order에 cands 자체 없음)
            order_mismatch = 0          # LLM이 다른 order의 POI를 가져옴 (보정 카운트에 포함)
            rng = _random.Random(int.from_bytes(hashlib.sha256(aid.encode("utf-8")).digest()[:8], "big"))
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
            if grounded_experiment and (
                hallucinations or hallucinations_dropped or missing_filled
                or any(not pick.evidence_ref or not pick.evidence_quote for pick in parsed.picks)
            ):
                raise ValueError('grounded Stage2 cannot accept repaired or uncited picks')

            # 후처리: actual_spent 0/None인 commerce 이벤트에 fallback (cap 추적 무력화 방지)
            # 실측 단가 앵커(동네×업종) × 밴드 배율 우선, 없으면 구 표 방식.
            daily_wd = persona.get("daily_wd") or 0
            cat_by_order = {i: ev.category for i, ev in enumerate(stage1.events)}
            anchor_by_order = {
                i: (cs[0].get("unit_anchor") if cs else None)
                for i, cs in cands_by_order.items()
            }
            spend_imputed = 0
            durable_by_order = {
                i: bool(cs[0].get("durable_anchor")) if cs else False
                for i, cs in cands_by_order.items()
            }
            spend_amount_fallbacks = 0
            for pick in parsed.picks:
                cat = cat_by_order.get(pick.order)
                if cat and cat not in INTERNAL_CATS:
                    if _pt.ON:
                        # [2026-10-07b] 금액 = 모델이 답한 한 사람 몫 × 인원. 0원(산책·상담·이번 달 이미 낸 학원 등)은 0원.
                        # 예전엔 0원을 '누락'으로 보고 같은 업종 값·실측 하위 10%·업종 상수로 채웠다(산책 → 133,716원).
                        pick.actual_spent = float(max(0, int(round(pick.unit_price or 0))) * max(1, int(pick.pay_count or 1)))
                        if pick.produce_spent is not None:
                            # 모델 답을 결제액 안으로만 자른다(값을 채우거나 바꾸지 않는다).
                            pick.produce_spent = max(0, min(int(pick.produce_spent), int(pick.actual_spent)))
                        continue
                    if pick.actual_spent is None or pick.actual_spent <= 0:
                        spend_amount_fallbacks += 1
                        if _pt.ON:
                            # 빠진 금액은 같은 응답에서 같은 업종의 다른 결제(1인 몫)를 먼저 쓰고, 없으면 실측 결제 1건당의
                            # 하위 10%를 쓴다(결제 1건당 중앙·평균은 여럿이 함께 낸 결제가 섞여 혼자 몫으로는 크다).
                            _same = [float(x.unit_price) for x in parsed.picks
                                     if x is not pick and (x.unit_price or 0) > 0 and cat_by_order.get(x.order) == cat]
                            _cs = cands_by_order.get(pick.order) or []
                            _ti = _cs[0].get("price_ticket") if _cs else None
                            _fb = (sorted(_same)[len(_same) // 2] if _same else (_ti["p10"] if _ti else None))
                            if _fb:
                                pick.actual_spent = float(int(round(_fb)))
                    pb, pf = price_by_poi.get(pick.poi_id) or (None, 1.0)
                    previous_spend = pick.actual_spent
                    _ensure_positive_spend(
                        pick, cat, daily_wd, price_factor=pf,
                        base_won=anchor_by_order.get(pick.order), band=pb,
                        durable=bool(durable_by_order.get(pick.order)),
                    )
                    if pick.actual_spent != previous_spend:
                        spend_imputed += 1
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
                "temp": effective_temp,
                "temp_requested": temp,
                "tokens_in": total_tokens_in,
                "tokens_out": total_tokens_out,
                "tokens_in_total": sum(row.get('tokens_in', 0) for row in timing['attempts']),
                "tokens_out_total": sum(row.get('tokens_out', 0) for row in timing['attempts']),
                "hallucinations_corrected": hallucinations,
                "hallucinations_dropped": hallucinations_dropped,
                "order_mismatch": order_mismatch,
                "missing_picks_filled": missing_filled,
                "spend_amount_fallbacks": spend_amount_fallbacks,
                "spend_imputed": spend_imputed,
                "price_by_poi": price_by_poi,
                "coupon_by_poi": coupon_by_poi,
                "poi_cat_by_poi": poi_cat_by_poi,
                "review_lookup_count": len(review_lookup_used),
                # 리뷰 흔적 — 추가 LLM 호출 없이 기존 2-pass 데이터에서 캡처
                "review_lookup_used": review_lookup_used,   # {poi_id: {rating, rating_count, reviews, category}}
                "pre_review_picks": pre_review_picks,       # {order: 리뷰 전 선택 poi_id}
                "s2_timing": timing_snapshot(),
                **fb_stats,
            }
            return parsed, cands_by_order, finish_capture(parsed, meta)
        except Exception as e:
            if fatal_dispatch_error(e):
                raise
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
            if error_stage in ("json_extract", "json_parse") and (
                finish == "length" or tokens_out >= token_cap
            ):
                prior_output_limited = True
                attempt_timing["output_limited"] = True
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

    if grounded_experiment:
        raise RuntimeError(f"Stage2 grounded decision failed after {timing['n_llm_calls']} attempts: {last_err}")
    # Full LLM failure cannot be scored as a citizen choice. The old runner
    # silently used a top-5 POI fallback (106/500 citizens on P012 Oct16),
    # making a completed day look valid despite having no Stage2 decision.
    # Preserve that behavior only when explicitly requested for exploration.
    if os.environ.get("SIM_ALLOW_STAGE2_FALLBACK") == "1":
        fallback = _fill_missing_picks(Stage2Output(picks=[]), stage1.events,
                                       cands_by_order, aid=aid)
        if fallback.picks:
            meta = {
                "fallback_only": True,
                "price_by_poi": price_by_poi,
                "coupon_by_poi": coupon_by_poi,
                "poi_cat_by_poi": poi_cat_by_poi,
                "last_err": str(last_err)[:200],
                "tokens_in": total_tokens_in,
                "tokens_out": total_tokens_out,
                "s2_timing": timing_snapshot(),
            }
            return fallback, cands_by_order, finish_capture(fallback, meta)
    raise RuntimeError(f"Stage2 failed after {max_retry+1} attempts: {last_err}")


def _fill_missing_picks(
    stage2: Stage2Output, stage1_events: list, cands_by_order: dict[int, list[dict]],
    aid: str = "",
) -> Stage2Output:
    """LLM이 picks에 안 만든 외출 이벤트에 candidates Top 5 중 random POI 자동 채움."""
    import random as _random
    picked_orders = {p.order for p in stage2.picks}
    new_picks = list(stage2.picks)
    rng = _random.Random(int.from_bytes(hashlib.sha256(aid.encode("utf-8")).digest()[:8], "big") if aid else 42)
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
    poi_cat_by_poi: dict[str, tuple] | None = None,
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
            # [EXP_PRICE_MODE] 무엇을 얼마에 — 인터뷰·기억에 남긴다(옛 모드에서는 None)
            "menu": (pick_obj.menu if pick_obj else None),
            "unit_price": (pick_obj.unit_price if pick_obj else None),
            "pay_count": (pick_obj.pay_count if pick_obj else None),
            # [2026-10-11] 장보기 중 농축산물 금액과 그 비율 — 금액 조정(감당 범위로 줄이기) 뒤에도 비율로 할인 기준액을 다시 낸다.
            "produce_spent": (pick_obj.produce_spent if pick_obj else None),
            "produce_share": ((pick_obj.produce_spent / pick_obj.actual_spent)
                              if pick_obj and pick_obj.produce_spent is not None and (pick_obj.actual_spent or 0) > 0
                              else None),
            "price_band": _pb[0] if _pb else None,
            "price_factor": float(_pb[1]) if _pb else 1.0,
            # 쿠폰 사용처 여부 (후보풀 밖 POI(anchor 등)는 None = 판정 불가)
            "coupon_eligible": (coupon_by_poi or {}).get(poi_id) if poi_id else None,
            # 고른 가게 자신의 업종(후보풀 밖 POI 는 None) — 즉시 할인 판정용
            "poi_sub_category": ((poi_cat_by_poi or {}).get(poi_id) or (None,) * 4)[0] if poi_id else None,
            "poi_category": ((poi_cat_by_poi or {}).get(poi_id) or (None,) * 4)[1] if poi_id else None,
            "poi_name": ((poi_cat_by_poi or {}).get(poi_id) or (None,) * 4)[2] if poi_id else None,
            "upjong_l3": ((poi_cat_by_poi or {}).get(poi_id) or (None,) * 5)[3] if poi_id else None,
            "poi_same_gu": (((poi_cat_by_poi or {}).get(poi_id) or (None,) * 6) + (None,) * 2)[4] if poi_id else None,
            "poi_gu": (((poi_cat_by_poi or {}).get(poi_id) or (None,) * 6) + (None,) * 2)[5] if poi_id else None,
            # 정책별 사용액 dict ({"P009": 5000}) — 분석 시 정책 사용처 추적
            "policy_spend": (pick_obj.policy_spend if pick_obj else None) or {},
            # 지원금 결제건의 "없었어도 했을 지출인가"(참고3 ④ 문항 형태). MPC 산출 입력.
            "would_buy_anyway": (pick_obj.would_buy_anyway if pick_obj else None),
            "extra_spent": (pick_obj.extra_spent if pick_obj else None),
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

    s2, cands, m2 = call_stage2(
        args.aid, s1, ctx.persona, today, verbose=args.verbose,
        state=ctx.state, active_policies=ctx.policy,
    )
    print("\n=== Stage 2 ===")
    print(s2.model_dump_json(indent=2))
    print(f"\nmeta: {m2}")

    final = merge_to_final_events(s1, s2, ctx.persona)
    print("\n=== 최종 이벤트 ===")
    for e in final:
        print(f"  [{e['order']}] {e['time']} {e['anchor']:30} {e['category']:6} → {e['poi_id']} ({e['intent']})")
