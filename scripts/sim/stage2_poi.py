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
import re
import sys
from datetime import date
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dawn_context import (  # noqa: E402
    DawnContext, build_dawn_context,
    build_stage2_candidates,
    build_stage2_candidates_l1_dong,
    build_stage2_candidates_l1_district,
)
from stage1_intent import Stage1Output, call_stage1, _extract_json  # noqa: E402
from llm_client import call_chat as _llm_call  # noqa: E402


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


# commerce 이벤트에 actual_spent가 0/None이면 카테고리·소득별 fallback 값 부여.
# 가급적 LLM이 직접 정하게 하되, 환각·누락 시 cap 추적 무력화 방지용 안전망.
_SPEND_FALLBACK_BY_L1 = {
    "편의점": 5000, "마트": 25000,
    "식사": 12000, "카페": 6000, "디저트": 8000, "주점": 30000,
    "미용": 30000, "쇼핑": 50000,
    "여가": 20000, "건강": 15000, "교육": 50000, "기타": 10000,
}


def _ensure_positive_spend(pick: "Stage2Pick", category: str | None, daily_wd: float | int | None) -> None:
    """LLM이 actual_spent 누락 / 0 / 음수로 출력했을 때 카테고리 fallback 부여.

    track_policy_usage가 spend<=0 이면 cap 추적을 skip하므로,
    여기서 최소값을 강제해 정책 효과 측정 신뢰도 확보.
    """
    cur = pick.actual_spent or 0
    if cur > 0:
        return
    base = _SPEND_FALLBACK_BY_L1.get((category or "기타"), 10000)
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


def _score_and_sort_by_desire(cands: list[dict], today: date) -> list[dict]:
    """avg_satisfaction 내림차순 → km 오름차순 정렬."""
    for c in cands:
        sat = c.get("avg_satisfaction")
        c["desire"] = float(sat) if sat is not None else 0.0
    # 1순위: avg_satisfaction 내림차순, 2순위: km 오름차순 (None은 뒤)
    cands.sort(key=lambda c: (-c["desire"], c.get("km") or 9999))
    return cands


def fetch_candidates_for_events(
    aid: str, events: list, persona: dict, today: date,
    k_per_event: int = 12,
    stats: dict | None = None,
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

    out: dict[int, list[dict]] = {}
    s = stats if stats is not None else {}

    # 1) 각 이벤트 → 그룹 키 (dong_code, sub_cat) 결정. 스킵은 즉시 빈 풀.
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

    # 3) 그룹별 fetch + round-robin 분할 (fallback 체인 그룹 단위 1회)
    for (dong_code, sub_cat), event_idxs in groups.items():
        n = len(event_idxs)
        pool_size = k_per_event if n == 1 else n * k_per_event
        l1 = l1_for[event_idxs[0]]   # 같은 sub_cat ⇒ 같은 L1

        cands = build_stage2_candidates(aid, dong_code, sub_cat, limit=pool_size)
        if cands:
            s["cand_sub_match"] = s.get("cand_sub_match", 0) + n
        else:
            if l1 and l1 not in INTERNAL_CATS:
                cands = build_stage2_candidates_l1_dong(aid, dong_code, l1, limit=pool_size)
                if cands:
                    s["cand_fallback_l1_dong"] = s.get("cand_fallback_l1_dong", 0) + n
            if not cands and l1:
                district_code = dong_code[:5] if len(dong_code) >= 5 else None
                if district_code:
                    cands = build_stage2_candidates_l1_district(
                        aid, district_code, l1, limit=pool_size,
                    )
                    if cands:
                        s["cand_fallback_l1_district"] = s.get("cand_fallback_l1_district", 0) + n
            if not cands:
                s["cand_all_empty"] = s.get("cand_all_empty", 0) + n

        # desire 점수 계산 + 정렬 (분할·할당 전에 1회)
        cands = _score_and_sort_by_desire(cands or [], today)

        if n == 1:
            out[event_idxs[0]] = cands[:k_per_event]
        else:
            buckets = _split_pool_round_robin(cands, n, k_per_event)
            for bucket, ev_i in zip(buckets, event_idxs):
                out[ev_i] = bucket
            if cands:
                s["pool_split_groups"] = s.get("pool_split_groups", 0) + 1
                s["pool_split_events"] = s.get("pool_split_events", 0) + n

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

**단순 반복 억제**
- 최근 3일 이내 방문한 POI(⚠️ 표시)는 특별한 사유 없이 재선택하지 마세요.
- 같은 날 여러 이벤트가 있을 때 동일 POI를 두 번 선택하지 마세요.

**소비액 설정 (actual_spent + policy_spend)**

`actual_spent` = 이 거래의 총 소비액 (양수).
`policy_spend` = 그중 grant에서 쓴 부분 분리. `{"P009": 5000}` 형태. grant 안 쓰면 null/`{}`.
제약: sum(policy_spend) ≤ actual_spent / 잔액.

소비 결정 방식:
- 본인 예산 부분은 페르소나·POI·기분 따라 자연스럽게.
- grant는 **별도 지갑(가계 부담 0)에서 차감되는 추가 소비 자금**. 본인 잔액 영향 없음.
- **actual_spent는 평소 평균에 묶일 필요는 없지만**, **페르소나·라이프스타일에 일관되게** 결정.
- 매일 grant 풀 사용 같은 비현실적 폭주는 X — 사람은 기분·상황에 따라 변동.
- POI·기분·페르소나 조합이 결정 요인. 평소 평균에 맞출 필요는 없지만 페르소나 성격은 일관되게.
- 모든 commerce 이벤트에 양의 actual_spent를 반드시 부여 (0원·음수 금지).

★★ **JSON 필수 일치**: pick_reason에 "P009"/"지원금"/"활용" 등 정책 활용 의도를 적었다면 **반드시 `policy_spend: {"P009": N}` 필드를 채울 것**. 의도만 적고 빈 dict로 두는 누락은 절대 금지. `policy_spend`는 null 또는 실제 사용 금액 dict. 모호한 경우 actual_spent 전액을 policy_spend로 (잔액 한도 내).

**만족도 설정 (actual_satisfaction)**
- 0.0 ~ 1.0 범위의 실수입니다.
- 과거 방문 기록(avg_sat)이 있으면 그 근처에서 페르소나 성향을 반영해 조정합니다.
- 처음 가는 곳은 페르소나·카테고리·거리 등을 고려해 자유롭게 설정합니다.
- 값이 높을수록 만족, 낮을수록 불만족입니다.

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
    "pick_reason": "P009 지원금으로 평소 못 가본 카페 시도. 페르소나 소비형이라 grant 적극 활용.",
    "pick_factor": "satisfaction"
  }
]}

pick_factor enum: known | distance | satisfaction | rumor | appointment | random
/no_think"""


def _format_event_with_candidates(
    i: int, ev, cands: list[dict], recent_poi_ids: set[str] | None = None
) -> str:
    if not cands:
        return ""
    lines = [
        f"### 이벤트 {i} | {ev.time} | {ev.anchor} | "
        f"{ev.category}/{ev.sub_category or _guess_sub_from_l1(ev.category)} | {ev.intent}"
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

        lines.append(
            f"  {known_mark}{recent_mark} {c['poi_id']} | {c.get('name') or '(이름없음)'} | "
            f"{km_s} | {sat_s} {visit_s}"
        )
    return "\n".join(lines)


def build_stage2_prompt(
    events: list,
    cands_by_order: dict[int, list[dict]],
    persona: dict | None = None,
    recent_poi_ids: set[str] | None = None,
    active_policies: list[dict] | None = None,
    grant_remaining: dict[str, int] | None = None,
) -> str:
    import json as _json
    # 페르소나 헤더
    header_parts = []
    if persona:
        daily_wd = persona.get("daily_wd") or 0
        daily_we = persona.get("daily_we") or 0
        tendency = persona.get("tendency") or ""
        lifestyle = (persona.get("lifestyle") or "").strip()
        income = persona.get("income") or ""
        budget_info = f"일일 예산: 평일 {daily_wd:,}원 / 주말 {daily_we:,}원"
        header_parts.append(f"## 에이전트 정보\n{lifestyle}\n{budget_info} / 소비성향: {tendency} / 소득분위: {income}")

        # 활성 정책 명시 (grant 위주, LLM이 policy_spend 책정 시 참조)
        rem = grant_remaining or {}
        pol_lines = []
        for pol in (active_policies or []):
            pid = pol.get("id") or ""
            ptype = pol.get("type") or ""
            pname = pol.get("name") or ""
            if ptype == "grant":
                # income_grants에서 본인 분위 수령액 lookup
                try:
                    grants = _json.loads(pol.get("income_grants") or "{}")
                except Exception:
                    grants = {}
                my_amount = int(grants.get(income, 0))
                cur_rem = int(rem.get(pid, my_amount))
                if my_amount > 0:
                    used = max(0, my_amount - cur_rem)
                    pol_lines.append(
                        f"{pid} {pname} [grant 추가 소비 자금] — 수령액 {my_amount:,}원, 사용 {used:,}원, "
                        f"잔액 {cur_rem:,}원. 별도 지갑(가계 부담 0). "
                        f"actual_spent는 평소 평균에 맞출 필요 없음 — POI·기분·grant 활용 의지로 자유 결정"
                    )
                else:
                    pol_lines.append(f"{pid} {pname} — 본인 분위 '{income}' 제외 대상")
            elif ptype == "subsidy":
                cap = pol.get("cap") or 0
                if cap > 0:
                    pol_lines.append(f"🎫 {pid} {pname} [subsidy] — 한도 {cap:,}원, 잔액 정보 별도")
        if pol_lines:
            header_parts.append("## 활성 정책 (policy_spend 책정 시 참조)\n" + "\n".join(pol_lines))

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
    active_policies: list[dict] | None = None,
    grant_remaining: dict[str, int] | None = None,
) -> tuple[Stage2Output, dict[int, list[dict]], dict]:
    """Stage 2 LLM 호출. (picks, 사용된 candidates, meta) 반환.

    today: 오늘 날짜. desire 계산의 days_since_visit 산출에 사용.
    """
    fb_stats: dict[str, int] = {}
    cands_by_order = fetch_candidates_for_events(
        aid, stage1.events, persona, today, stats=fb_stats,
    )
    need_llm = any(cs for cs in cands_by_order.values())

    if not need_llm:
        # 외부 POI 결정 필요 없음 (전부 residence/workplace/pinned)
        return Stage2Output(picks=[]), cands_by_order, {"skipped": True, **fb_stats}

    # 최근 3일 방문 POI (억제용)
    recent_poi_ids: set[str] = set()
    try:
        from _common import driver_session
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

    user_block = build_stage2_prompt(
        stage1.events, cands_by_order,
        persona=persona,
        recent_poi_ids=recent_poi_ids,
        active_policies=active_policies,
        grant_remaining=grant_remaining,
    )

    # 환각 차단용 JSON schema — poi_id는 전체 후보풀 union enum 강제.
    # order-별 enum은 아니지만 후보풀 외 POI는 0건 보장. order_mismatch는 fallback에서 처리.
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
                                    "actual_spent": {"type": "number"},
                                    "actual_satisfaction": {"type": "number"},
                                    "policy_spend": {"type": ["object", "null"]},
                                    "pick_reason": {"type": ["string", "null"]},
                                    "pick_factor": {"type": ["string", "null"]},
                                },
                                "required": ["order", "poi_id", "actual_spent", "actual_satisfaction"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["picks"],
                    "additionalProperties": False,
                },
            },
        }

    last_err = None
    for attempt in range(max_retry + 1):
        temp = 0.7 + 0.1 * attempt
        try:
            resp = _llm_call(
                None, SYSTEM_S2, user_block,
                temperature=temp, max_tokens=1200,  # pick_reason 필드 추가로 출력량 ↑
                response_format=s2_schema,
            )
            raw = resp.choices[0].message.content
            if verbose:
                print(f"--- attempt {attempt} (temp={temp}) ---")
                print(raw[:600])
            json_str = _extract_json(raw)
            data = json.loads(json_str)
            parsed = Stage2Output.model_validate(data)

            # 후보 풀 안에 있는지 검증 — 반드시 해당 order의 candidates 안에서만 valid.
            # (이전 버그: valid_pois = 전체 cands flat → 다른 order의 POI도 통과되어 카테고리 매칭이 깨짐)
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
                        chosen_c = rng.choice(top)
                        chosen = chosen_c["poi_id"]
                        sat_prior = chosen_c.get("avg_satisfaction")
                        if sat_prior is None:
                            sat_prior = 0.5
                        corrected_picks.append(Stage2Pick(
                            order=pick.order, poi_id=chosen,
                            actual_spent=pick.actual_spent,  # LLM이 준 값 유지(없으면 None → 후처리)
                            actual_satisfaction=float(sat_prior),
                        ))
                        hallucinations += 1
                    else:
                        # 해당 order에 candidates 자체 없음 — drop
                        hallucinations_dropped += 1
            parsed = Stage2Output(picks=corrected_picks)

            # 후처리: LLM이 답 안 한 외출 이벤트에 candidates 자동 fill
            picks_before_fill = len(parsed.picks)
            parsed = _fill_missing_picks(parsed, stage1.events, cands_by_order, aid=aid)
            missing_filled = len(parsed.picks) - picks_before_fill

            # 후처리: actual_spent 0/None인 commerce 이벤트에 카테고리 fallback (cap 추적 무력화 방지)
            daily_wd = persona.get("daily_wd") or 0
            cat_by_order = {i: ev.category for i, ev in enumerate(stage1.events)}
            for pick in parsed.picks:
                cat = cat_by_order.get(pick.order)
                if cat and cat not in INTERNAL_CATS:
                    _ensure_positive_spend(pick, cat, daily_wd)

            meta = {
                "attempt": attempt,
                "temp": temp,
                "tokens_in": resp.usage.prompt_tokens,
                "tokens_out": resp.usage.completion_tokens,
                "hallucinations_corrected": hallucinations,
                "hallucinations_dropped": hallucinations_dropped,
                "order_mismatch": order_mismatch,
                "missing_picks_filled": missing_filled,
                **fb_stats,
            }
            return parsed, cands_by_order, meta
        except Exception as e:
            last_err = e
            if verbose:
                print(f"[attempt {attempt}] failed: {e}")

    # 최종 retry 실패: LLM picks 빈 상태에서 candidates 첫 거 강제 fill
    fallback = _fill_missing_picks(Stage2Output(picks=[]), stage1.events, cands_by_order, aid=aid)
    if fallback.picks:
        return fallback, cands_by_order, {"fallback_only": True, "last_err": str(last_err)[:200]}
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
        chosen_c = rng.choice(top)
        chosen = chosen_c["poi_id"]
        sat_prior = chosen_c.get("avg_satisfaction")
        if sat_prior is None:
            sat_prior = 0.5
        new_picks.append(Stage2Pick(order=i, poi_id=chosen, actual_spent=None, actual_satisfaction=float(sat_prior)))
    return Stage2Output(picks=new_picks)


# =========================================================
# Stage 1 + Stage 2 병합 → 최종 events
# =========================================================
def merge_to_final_events(stage1: Stage1Output, stage2: Stage2Output, persona: dict) -> list[dict]:
    """Stage 1 + Stage 2 picks → 최종 events with poi_id.

    카테고리 기준 우선 (anchor는 출발지 표시일 뿐):
      - pinned_poi 있으면 그대로
      - cat ∈ {집, 직장} (머무름) → anchor에 따라 home/work POI
      - cat ∈ 외출 카테고리 (식사·카페·편의점 등) → Stage 2 pick (commerce POI)
        - Stage 2 pick 누락 시 fallback으로 anchor POI 사용
    """
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
            # 정책별 사용액 dict ({"P009": 5000}) — 분석 시 정책 사용처 추적
            "policy_spend": (pick_obj.policy_spend if pick_obj else None) or {},
            # ───── 사고과정 흔적 (인터뷰용) ─────
            "reasoning": ev.reasoning,                 # Stage 1: 왜 이 의도·카테고리·anchor
            "trigger": ev.trigger,                     # Stage 1: appointment/rumor/policy/...
            "pick_reason": pick_obj.pick_reason if pick_obj else None,   # Stage 2: 왜 이 POI
            "pick_factor": pick_obj.pick_factor if pick_obj else None,   # Stage 2: known/distance/...
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
