"""Plan 적재 + 정책 cap 추적 + Night Phase (visited Memory + KNOWS_POI 갱신).

흐름:
  1. write_plan: Stage 1+2 병합 결과를 :Plan + [:INCLUDES]로 적재
  2. track_policy_usage: 정책 subsidy cap 잔액 추적 (만족도·소비액 설정은 Stage2 LLM이 담당)
  3. night_finalize: 어제 INCLUDES → :Memory{type:'visited'} CREATE + KNOWS_POI MERGE/UPDATE
  4. night_state_update: 오늘 :State CREATE (잔액·에너지·mood·fatigue 갱신)

Conversation 적재(Night Phase 2)는 별도 night_phase.py로 (LLM 의도 분류 호출 포함).
"""
from __future__ import annotations

import sys
import uuid
from datetime import date, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neo4j_load._common import driver_session


# =========================================================
# Plan 적재
# =========================================================
WRITE_PLAN_CYPHER = """
MATCH (a:Agent {id: $aid})
MERGE (p:Plan {id: $plan_id})
SET p.agent_id = $aid, p.day = date($day), p.day_type = $day_type,
    p.generated_at = datetime(),
    p.llm_tokens_in = $tokens_in, p.llm_tokens_out = $tokens_out,
    // 리뷰 노출 기록 — 추가 LLM 호출 없이 기존 2-pass 데이터에서 캡처
    p.review_lookup_count = $review_lookup_count,     // 오늘 조회한 POI 수
    p.review_changed_count = $review_changed_count,   // 리뷰로 선택이 바뀐 pick 수
    p.reviews_seen = $reviews_seen                    // 본 리뷰 전체 JSON {poi_id:{rating,reviews,...}}
MERGE (a)-[:HAS_PLAN {day: date($day)}]->(p)
// 재시뮬 시 기존 INCLUDES 삭제 (idempotent)
WITH p
OPTIONAL MATCH (p)-[i:INCLUDES]->()
DELETE i
"""

WRITE_INCLUDES_CYPHER = """
MATCH (p:Plan {id: $plan_id})
UNWIND $events AS ev
MATCH (poi:POI {id: ev.poi_id})
CREATE (p)-[:INCLUDES {
  order: ev.order,
  time: time(ev.time),
  intent: ev.intent,
  category: ev.category,
  sub_category: ev.sub_category,
  anchor: ev.anchor,
  with_agents: ev.with_agents,
  actual_satisfaction: ev.actual_satisfaction,
  actual_spent: coalesce(ev.actual_spent, 0),
  // 정책 지원금에서 사용한 금액 JSON 형태 ({"P009": 5000})
  // 분석 시: 정책별 사용처/누적 사용액 추적 가능
  spent_from_policy: coalesce(ev.spent_from_policy_json, '{}'),
  // 사고과정 흔적 (인터뷰 가능성 확보용)
  reasoning: ev.reasoning,           // Stage 1: 왜 이 시간·카테고리·anchor
  trigger: ev.trigger,               // Stage 1: appointment/rumor/policy/lifestyle/mood/none
  pick_reason: ev.pick_reason,       // Stage 2: 왜 후보풀 중 이 POI
  pick_factor: ev.pick_factor,       // Stage 2: known/distance/satisfaction/rumor/novelty/random
  price_band: ev.price_band,         // POI 가격대 1(₩)/2(₩₩)/3(₩₩₩) — poi_price.py
  price_factor: ev.price_factor,     // 적용 가격배율 (검증·판매자 가격 채널 분석용)
  coupon_eligible: ev.coupon_eligible, // 쿠폰 사용처 여부 (백테스트 T2 업종/사용처 분석용)
  // ───── 리뷰 노출·사고변화 흔적 (추가 LLM 호출 0) ─────
  review_seen: ev.review_seen,           // 이 POI 카카오 리뷰를 봤나
  seen_rating: ev.seen_rating,           // 본 평균 별점
  seen_rating_count: ev.seen_rating_count,
  review_snippet: ev.review_snippet,     // 본 리뷰 한 줄
  pre_review_poi: ev.pre_review_poi,     // 리뷰 전(1차) 선택 — 바뀐 경우만
  review_changed: ev.review_changed      // 리뷰가 최종 선택을 바꿨나
}]->(poi)
"""


def write_plan(
    aid: str, today: date, events: list[dict], day_type: str,
    tokens_in: int = 0, tokens_out: int = 0,
    reviews_seen: dict | None = None, review_lookup_count: int = 0,
):
    import json as _json
    plan_id = f"{aid}_{today.isoformat()}"
    # poi_id가 None인 이벤트는 적재 스킵 (직장 POI 비어있는 경우 등)
    valid_events = [dict(e) for e in events if e.get("poi_id")]
    # policy_spend dict → JSON 문자열로 변환 (Neo4j relationship property는 nested dict 미지원)
    for ev in valid_events:
        ps = ev.get("policy_spend") or {}
        ev["spent_from_policy_json"] = _json.dumps(ps, ensure_ascii=False) if ps else "{}"
    # 리뷰 노출 기록(어떤 리뷰를 봤나) + 사고변화 건수 — O(events), 추가 호출 없음
    reviews_seen_json = _json.dumps(reviews_seen, ensure_ascii=False) if reviews_seen else "{}"
    review_changed_count = sum(1 for ev in valid_events if ev.get("review_changed"))
    with driver_session() as s:
        s.run(WRITE_PLAN_CYPHER,
              aid=aid, plan_id=plan_id, day=today.isoformat(), day_type=day_type,
              tokens_in=tokens_in, tokens_out=tokens_out,
              reviews_seen=reviews_seen_json, review_lookup_count=review_lookup_count,
              review_changed_count=review_changed_count)
        if valid_events:
            s.run(WRITE_INCLUDES_CYPHER, plan_id=plan_id, events=valid_events)
    return plan_id, len(valid_events)


# =========================================================
# 정책 cap 추적
# =========================================================
INTERNAL_CATS = {"집", "직장"}


def _policy_match(ev: dict, pol: dict, home_dist5: str, work_dist5: str) -> bool:
    """이벤트가 정책 적용 대상인지 (자치구 + 카테고리).

    POLICY_CYPHER가 dist.code(5자리)를 region_codes로 반환. 빈 list면 전 자치구 (서울 전체).
    """
    region_codes = [c for c in (pol.get("region_codes") or []) if c]
    target_l1s = [t for t in (pol.get("target_l1s") or []) if t]
    region_ok = (not region_codes) or (home_dist5 in region_codes) or (work_dist5 in region_codes)
    if not region_ok:
        return False
    cat = ev.get("category"); sub = ev.get("sub_category") or ""
    if cat in INTERNAL_CATS:
        return False
    cat_ok = (not target_l1s) or (cat in target_l1s) or (sub in target_l1s)
    return cat_ok


def apply_grant_to_prev_state(aid: str, today: date, grant: int) -> None:
    """grant 정책 effective_from 당일, 전날 State 잔액에 지원금 추가.

    Stage 1 Dawn 컨텍스트가 이 업데이트를 반영해 LLM이 추가 예산을 인식한다.
    """
    yesterday = (today - timedelta(days=1)).isoformat()
    with driver_session() as s:
        s.run(
            "MATCH (a:Agent {id:$aid})-[:HAS_STATE]->(st:State) "
            "WHERE toString(st.day) = $yest "
            "SET st.balance = coalesce(st.balance, 0) + $grant",
            aid=aid, yest=yesterday, grant=grant,
        )


def _parse_income_grants(raw) -> dict[str, int]:
    """policy.income_grants 파싱. dict / JSON string / None 처리."""
    import json as _json
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(k): int(v) for k, v in raw.items()}
    if isinstance(raw, str):
        try:
            d = _json.loads(raw)
            return {str(k): int(v) for k, v in d.items()} if isinstance(d, dict) else {}
        except Exception:
            return {}
    return {}


def _parse_excluded_income(raw) -> set[str]:
    """policy.excluded_income 파싱. list / JSON string / None 처리."""
    import json as _json
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple, set)):
        return {str(x) for x in raw}
    if isinstance(raw, str):
        try:
            d = _json.loads(raw)
            return {str(x) for x in d} if isinstance(d, (list, tuple)) else set()
        except Exception:
            return set()
    return set()


def grant_lookup_key(pol: dict, income: str, spend_decile=None) -> str:
    """정책의 income_grants를 조회할 키.

    policy.grant_key == 'spend_decile' 이면 소비 10분위(1~10)로, 아니면 소득 5분위로 조회.
    (P010 민생회복 소비쿠폰처럼 계층별 차등을 10분위로 표현하는 정책용)
    """
    if (pol.get("grant_key") or "income") == "spend_decile":
        try:
            return str(int(spend_decile))
        except (TypeError, ValueError):
            return ""
    return income or ""


def _grant_for_single_policy(income: str, pol: dict, spend_decile=None) -> int:
    """단일 grant 정책 지급액.

    신규 decile_grants가 있으면 소비 10분위를 사용하고, 없으면 레거시
    income_grants/grant_key를 사용한다.
    """
    if pol.get("type") != "grant":
        return 0
    decile_grants = _parse_income_grants(pol.get("decile_grants"))
    if decile_grants:
        try:
            key = str(int(spend_decile))
        except (TypeError, ValueError):
            return 0
        if key in _parse_excluded_income(pol.get("excluded_deciles")):
            return 0
        return int(decile_grants.get(key, 0))

    excluded = _parse_excluded_income(pol.get("excluded_income"))
    if income in excluded:
        return 0
    grants = _parse_income_grants(pol.get("income_grants"))
    return grants.get(grant_lookup_key(pol, income, spend_decile), 0)


def get_grant_amount(income: str, policies: list[dict], spend_decile=None) -> int:
    """오늘 적용 가능한 모든 grant 정책의 지급액 합계. 해당 없으면 0.

    여러 grant 정책이 동시에 활성이면 누적 합산.
    각 정책의 income_grants(소득별 지급액 dict)·excluded_income(제외 소득 list)을
    동적으로 읽어서 처리. P009 같은 차등 지급 정책 외에도 income_grants가 있는 어떤
    grant 정책이든 자동 처리.
    """
    total = 0
    for pol in policies:
        total += _grant_for_single_policy(income, pol, spend_decile=spend_decile)
    return total


def aggregate_policy_spend(events: list[dict]) -> dict[str, int]:
    """오늘 모든 INCLUDES의 policy_spend 합계 → {정책ID: 오늘 사용액}.

    LLM이 Stage2에서 거래별로 분리해 출력한 policy_spend를 정책별로 누적.
    grant 잔여액 계산용 (grant_remaining = received - 누적 spend).
    """
    out: dict[str, int] = {}
    for e in events:
        ps = e.get("policy_spend") or {}
        if not isinstance(ps, dict):
            continue
        for pid, amt in ps.items():
            try:
                v = int(amt)
                if v > 0:
                    out[str(pid)] = out.get(str(pid), 0) + v
            except (TypeError, ValueError):
                continue
    return out


def validate_policy_spend(
    events: list[dict],
    policy_remaining: dict[str, int] | None = None,
    restricted_pids: set[str] | None = None,
) -> int:
    """LLM 환각 보정 — 세 가지 제약 강제:
      (0) 사용처 제한: restricted_pids 정책은 coupon_eligible=True 매장에서만 사용 가능
          (민생회복 소비쿠폰류 — LLM이 규칙을 어기면 해당 정책분 제거 = 자기부담 전환)
      (1) 거래 단위: sum(policy_spend.values()) ≤ actual_spent
      (2) 정책 단위: 누적 policy_spend[pid] ≤ policy_remaining[pid]

    각 이벤트를 시간순으로 처리하면서 잔여 가용액을 차감. 초과분은 잘라냄.
    policy_remaining=None이면 (1)만 검증 (하위호환).
    반환: 보정된 거래 수.
    """
    corrected = 0
    restricted = restricted_pids or set()
    # 정책별 잔여 가용액 (시간순 차감)
    pol_rem = dict(policy_remaining or {})
    for e in events:
        ps = e.get("policy_spend") or {}
        # 결제수단 미선택은 그대로 0원이다. reasoning에 정책 단어가 있다는 이유로
        # policy_spend를 자동 생성하면 에이전트 판단을 덮어쓰고 강제 소진을 만든다.
        if not isinstance(ps, dict) or not ps:
            continue
        # 정수 정규화 + 양수만
        ps = {str(k): int(v) for k, v in ps.items()
              if isinstance(v, (int, float)) and v > 0}
        if not ps:
            e["policy_spend"] = {}
            continue
        changed = False
        # (0) 사용처 제한 정책 — 쿠폰 불가 매장(False)이거나 판정 불가(None) POI면 제거.
        #     None(후보풀 밖 anchor 등)도 보수적으로 차단 — 쿠폰 사용은 검증된 매장에서만.
        if restricted and e.get("coupon_eligible") is not True:
            before = set(ps.keys())
            ps = {k: v for k, v in ps.items() if k not in restricted}
            if set(ps.keys()) != before:
                changed = True
        # (2) 정책 단위 잔여액 cap
        if policy_remaining is not None:
            for pid in list(ps.keys()):
                avail = pol_rem.get(pid, 0)
                if ps[pid] > avail:
                    ps[pid] = max(0, avail)
                    changed = True
            ps = {k: v for k, v in ps.items() if v > 0}
        # (1) 거래 단위 actual_spent cap (정책 총합 ≤ 거래 총액)
        spent = e.get("actual_spent") or 0
        total = sum(ps.values())
        if total > spent and total > 0:
            ratio = spent / total
            for pid in list(ps.keys()):
                ps[pid] = max(0, int(ps[pid] * ratio))
            ps = {k: v for k, v in ps.items() if v > 0}
            changed = True
        # 잔여 차감
        for pid, amt in ps.items():
            pol_rem[pid] = max(0, pol_rem.get(pid, 0) - amt)
        e["policy_spend"] = ps
        if changed:
            corrected += 1
    return corrected


def track_policy_usage(
    events: list[dict],
    persona: dict,
    active_policies: list[dict] | None = None,
    policy_used: dict[str, int] | None = None,
) -> dict[str, int]:
    """정책 subsidy cap 잔액 추적 (만족도·소비액 설정은 Stage2 LLM이 담당).

    actual_spent가 Stage2에서 설정됐으면 그 값 기준으로 cap 차감 추적.
    아직 설정 안 됐으면 skip.
    """
    home_dist5 = (persona.get("home_dong_code") or "")[:5]
    work_dist5 = (persona.get("work_dong_code") or "")[:5]
    policies = active_policies or []
    used = dict(policy_used or {})

    for e in events:
        cat = e.get("category")
        if cat in INTERNAL_CATS or not e.get("poi_id"):
            continue
        spend = e.get("actual_spent") or 0
        if spend <= 0:
            continue
        # event.policy_spend (Stage2 LLM 산출) 우선 — grant 적용 분 누적
        ps = e.get("policy_spend") or {}
        if isinstance(ps, dict):
            for pid, amt in ps.items():
                if isinstance(amt, (int, float)) and amt > 0:
                    used[str(pid)] = used.get(str(pid), 0) + int(amt)

        # subsidy 정책 cap-rate 처리 (legacy)
        for pol in policies:
            cap = pol.get("cap") or 0
            rate = pol.get("rate") or 0.0
            if cap <= 0 or rate <= 0:
                continue
            if not _policy_match(e, pol, home_dist5, work_dist5):
                continue
            pid = pol.get("id")
            remaining = max(0, cap - used.get(pid, 0))
            if remaining > 0:
                refund = min(int(spend * rate), remaining)
                used[pid] = used.get(pid, 0) + refund
    return used


# =========================================================
# Night Phase 3 (0): 어제 INCLUDES → Memory{visited} + KNOWS_POI 갱신
# =========================================================
NIGHT_VISITED_CYPHER = """
MATCH (a:Agent {id: $aid})-[:HAS_PLAN {day: date($yesterday)}]->(p:Plan)-[i:INCLUDES]->(poi:POI)
WHERE i.actual_satisfaction IS NOT NULL
WITH a, p, i, poi
// commerce/외출 이벤트만 Memory 적재 (집·직장 anchor는 일상이라 skip)
WHERE i.anchor STARTS WITH 'zone:' OR (i.category IS NOT NULL AND NOT i.category IN ['집','직장'])
WITH a, p, i, poi,
     0.5 + 1.5 * i.actual_satisfaction AS importance,
     poi.name + '(' + coalesce(i.sub_category, i.category) + ') 방문, 만족도 ' +
       toString(round(i.actual_satisfaction * 100) / 100.0) AS summary,
     // 결정적 id (agent + poi + day + order) — resume 재실행 시 중복 방지
     'mem_vis_' + a.id + '_' + poi.id + '_' + $yesterday + '_' + toString(i.order) AS mem_id
MERGE (m:Memory {id: mem_id})
  ON CREATE SET
    m.type = 'visited',
    m.day = date($yesterday),
    m.importance = importance,
    m.summary = summary,
    m.satisfaction = i.actual_satisfaction
MERGE (a)-[:REMEMBERS {day: date($yesterday)}]->(m)
MERGE (m)-[:ABOUT_POI]->(poi)

// KNOWS_POI MERGE + 집계 갱신
// recent_visit_dates: 30일 슬라이딩 윈도우 (saturation 계산용).
// Python 등가: scripts.sim.visit_window.trim_and_push_visit
MERGE (a)-[kp:KNOWS_POI]->(poi)
ON CREATE SET
  kp.since = date($yesterday), kp.source = 'visited',
  kp.visit_count = 1, kp.avg_satisfaction = i.actual_satisfaction,
  kp.last_visit = date($yesterday),
  kp.recent_visit_dates = [date($yesterday)]
ON MATCH SET
  kp.visit_count = coalesce(kp.visit_count, 0) + 1,
  kp.avg_satisfaction = (coalesce(kp.avg_satisfaction, 0.5) * coalesce(kp.visit_count, 0) + i.actual_satisfaction)
                         / (coalesce(kp.visit_count, 0) + 1),
  kp.last_visit = date($yesterday),
  kp.recent_visit_dates =
    [d IN coalesce(kp.recent_visit_dates, [])
     WHERE duration.inDays(d, date($yesterday)).days < 30]
    + [date($yesterday)]
RETURN count(m) AS n_memories
"""


def night_finalize_yesterday(aid: str, today: date) -> int:
    """어제 INCLUDES → Memory{visited} CREATE + KNOWS_POI 갱신."""
    yesterday = today - timedelta(days=1)
    with driver_session() as s:
        r = s.run(NIGHT_VISITED_CYPHER, aid=aid, yesterday=yesterday.isoformat()).single()
        return r["n_memories"] if r else 0


# =========================================================
# Night Phase 3 (4): 오늘 State CREATE
# =========================================================
NIGHT_STATE_CYPHER = """
MATCH (a:Agent {id: $aid})
OPTIONAL MATCH (a)-[:HAS_STATE {day: date($yesterday)}]->(prev:State)
WITH a, prev,
     coalesce(prev.balance, 1500000) AS prev_balance,
     coalesce(prev.energy, 0.8) AS prev_energy,
     coalesce(prev.mood, 0.5) AS prev_mood,
     coalesce(prev.fatigue, 0.3) AS prev_fatigue,
     coalesce(prev.month_spent, 0) AS prev_month_spent

// 오늘 INCLUDES 누적: 실제 actual_spent 합산 (외출 commerce만)
OPTIONAL MATCH (a)-[:HAS_PLAN {day: date($today)}]->(today_plan:Plan)-[i:INCLUDES]->()
WITH a, prev_balance, prev_energy, prev_mood, prev_fatigue, prev_month_spent,
     count(i) AS n_events,
     avg(i.actual_satisfaction) AS avg_sat,
     sum(coalesce(i.actual_spent, 0)) AS today_spent

// mood EMA: 0.7 * prev + 0.3 * avg_sat
// fatigue: 0.5 * prev + 0.05 * n_events + (0.2 if low_sat else 0) - (0.1 if home_dominant else 0)
WITH a, prev_balance, prev_energy, prev_month_spent,
     toInteger(today_spent) AS today_spent,
     coalesce(avg_sat, prev_mood) AS today_avg_sat,
     n_events,
     0.7 * prev_mood + 0.3 * coalesce(avg_sat, prev_mood) AS new_mood,
     CASE
       WHEN coalesce(avg_sat, 0.5) < 0.3 THEN 0.5 * prev_fatigue + 0.05 * n_events + 0.2
       ELSE 0.5 * prev_fatigue + 0.05 * n_events
     END AS new_fatigue_raw

WITH a, prev_balance, prev_energy, prev_month_spent, today_spent, today_avg_sat, n_events,
     new_mood,
     CASE WHEN new_fatigue_raw > 1.0 THEN 1.0
          WHEN new_fatigue_raw < 0.0 THEN 0.0
          ELSE new_fatigue_raw END AS new_fatigue

MERGE (s:State {id: $aid + '_' + $today})
SET s.agent_id = $aid,
    s.day = date($today),
    s.balance = prev_balance - (today_spent - $today_policy_spent),
    s.energy = 0.8,
    s.yesterday_satisfaction = today_avg_sat,
    s.mood = new_mood,
    s.fatigue = new_fatigue,
    s.month_spent = prev_month_spent + (today_spent - $today_policy_spent),
    s.policy_lifecycle = $policy_lifecycle_json,
    s.policy_used = $policy_used_json,   // 정책별 누적 사용액 JSON {"P007": 87000, ...}
    s.grant_received = $grant_received_json,  // 정책별 누적 grant 수령액 JSON {"P009": 250000, ...}
    s.grant_remaining = $grant_remaining_json // 정책별 grant 잔여액 JSON ({"P009": 175000} — 받은액 - 누적 사용)
MERGE (a)-[:HAS_STATE {day: date($today)}]->(s)
RETURN s.id AS state_id, s.balance AS balance, s.mood AS mood, s.fatigue AS fatigue
"""


def night_create_state(
    aid: str,
    today: date,
    policy_used: dict[str, int] | None = None,
    policy_lifecycle: dict[str, bool] | str | None = None,
    grant_received: dict[str, int] | str | None = None,
    grant_remaining: dict[str, int] | str | None = None,
    today_policy_spent: int = 0,
) -> dict:
    """오늘 State 노드 CREATE.

    policy_used: 정책별 누적 사용액 dict. 그래프에 JSON string으로 저장.
    policy_lifecycle: 정책 인지 상태 dict ({"P009": true} 형식).
        Dawn에서 주입된 정책 ID들이 들어옴. 외부에서 안 주면 빈 dict.
    grant_received: 정책별 누적 grant 수령액 dict ({"P009": 250000} 형식).
        resume 시 중복 적용 방지용 — 어제 State에 이미 기록된 정책은 skip.
    grant_remaining: 정책별 grant 잔여액 dict ({"P009": 175000} 형식).
        분석용: 받은 grant 중 아직 안 쓴 잔액 추적.
    """
    import json as _json
    yesterday = today - timedelta(days=1)
    used_json = _json.dumps(policy_used or {}, ensure_ascii=False)
    if isinstance(policy_lifecycle, str):
        lifecycle_json = policy_lifecycle
    else:
        lifecycle_json = _json.dumps(policy_lifecycle or {}, ensure_ascii=False)
    if isinstance(grant_received, str):
        grant_json = grant_received
    else:
        grant_json = _json.dumps(grant_received or {}, ensure_ascii=False)
    if isinstance(grant_remaining, str):
        grant_rem_json = grant_remaining
    else:
        grant_rem_json = _json.dumps(grant_remaining or {}, ensure_ascii=False)
    with driver_session() as s:
        r = s.run(NIGHT_STATE_CYPHER,
                  aid=aid, today=today.isoformat(), yesterday=yesterday.isoformat(),
                  policy_used_json=used_json,
                  policy_lifecycle_json=lifecycle_json,
                  grant_received_json=grant_json,
                  grant_remaining_json=grant_rem_json,
                  today_policy_spent=int(today_policy_spent or 0)).single()
        return dict(r) if r else {}


# =========================================================
# CLI 테스트 (1명 1일 end-to-end + Night)
# =========================================================
if __name__ == "__main__":
    import argparse, json as _json
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dawn_context import build_dawn_context
    from stage1_intent import call_stage1
    from stage2_poi import call_stage2, merge_to_final_events

    ap = argparse.ArgumentParser()
    ap.add_argument("--aid", default="AGT_11110515_F_20대_001")
    ap.add_argument("--day", default="2026-05-02")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    today = date.fromisoformat(args.day)
    day_type = "weekend" if today.weekday() >= 5 else "weekday"

    print(f"[1/5] Dawn 컨텍스트 빌드 ({args.aid}, {today})")
    ctx = build_dawn_context(args.aid, today)

    print("[2/5] Stage 1")
    s1, m1 = call_stage1(args.aid, today, ctx=ctx, verbose=args.verbose)

    print("[3/5] Stage 2")
    s2, cands, m2 = call_stage2(args.aid, s1, ctx.persona, today, verbose=args.verbose)
    events = merge_to_final_events(s1, s2, ctx.persona)

    print("[4/5] 정책 cap 추적 (만족도·소비액은 Stage2 LLM이 설정)")
    policy_used = track_policy_usage(events, ctx.persona)

    print("[5/5] Plan 적재 + Night Phase")
    tokens_in = m1["tokens_in"] + (m2.get("tokens_in") or 0)
    tokens_out = m1["tokens_out"] + (m2.get("tokens_out") or 0)
    plan_id, n_inc = write_plan(args.aid, today, events, day_type, tokens_in, tokens_out)
    print(f"  Plan id={plan_id}, INCLUDES x {n_inc}")

    state = night_create_state(args.aid, today, policy_used=policy_used)
    print(f"  State: {state}")

    print("\n=== 최종 events ===")
    for e in events:
        print(f"  [{e['order']}] {e['time']} {e['anchor']:25} {(e['category'] or ''):6} sat={e.get('actual_satisfaction')} → {e['poi_id']} ({e['intent']})")

    print(f"\n총 tokens: in={tokens_in}, out={tokens_out}")
