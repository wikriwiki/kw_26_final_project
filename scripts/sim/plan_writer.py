"""Plan 적재 + 만족도 룰 + Night Phase (visited Memory + KNOWS_POI 갱신).

흐름:
  1. write_plan: Stage 1+2 병합 결과를 :Plan + [:INCLUDES]로 적재
  2. simulate_satisfaction: 각 이벤트의 actual_satisfaction을 룰로 부여 (낮 시뮬 대체)
  3. night_finalize: 어제 INCLUDES → :Memory{type:'visited'} CREATE + KNOWS_POI MERGE/UPDATE
  4. night_state_update: 오늘 :State CREATE (잔액·에너지·mood·fatigue 갱신)

Conversation 적재(Night Phase 2)는 별도 night_phase.py로 (LLM 의도 분류 호출 포함).
"""
from __future__ import annotations

import random
import sys
import uuid
from datetime import date, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neo4j_load"))
from _common import driver_session


# =========================================================
# Plan 적재
# =========================================================
WRITE_PLAN_CYPHER = """
MATCH (a:Agent {id: $aid})
MERGE (p:Plan {id: $plan_id})
SET p.agent_id = $aid, p.day = date($day), p.day_type = $day_type,
    p.generated_at = datetime(),
    p.llm_tokens_in = $tokens_in, p.llm_tokens_out = $tokens_out
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
  actual_satisfaction: ev.actual_satisfaction
}]->(poi)
"""


def write_plan(
    aid: str, today: date, events: list[dict], day_type: str,
    tokens_in: int = 0, tokens_out: int = 0,
):
    plan_id = f"{aid}_{today.isoformat()}"
    # poi_id가 None인 이벤트는 적재 스킵 (직장 POI 비어있는 경우 등)
    valid_events = [e for e in events if e.get("poi_id")]
    with driver_session() as s:
        s.run(WRITE_PLAN_CYPHER,
              aid=aid, plan_id=plan_id, day=today.isoformat(), day_type=day_type,
              tokens_in=tokens_in, tokens_out=tokens_out)
        if valid_events:
            s.run(WRITE_INCLUDES_CYPHER, plan_id=plan_id, events=valid_events)
    return plan_id, len(valid_events)


# =========================================================
# 만족도 룰 (낮 시뮬 대체)
# =========================================================
INTERNAL_CATS = {"집", "직장"}


def simulate_satisfaction(
    persona: dict, events: list[dict],
    policy_target_cats: set[str] | None = None,
    policy_district_code: str | None = None,
    seed: int | None = None,
) -> list[dict]:
    """각 이벤트의 actual_satisfaction을 룰 기반으로 부여. 반환 = 변경된 events."""
    rng = random.Random(seed)
    top_wd = _parse_top_cats(persona.get("top_wd_json"))
    top_we = _parse_top_cats(persona.get("top_we_json"))
    home_dist5 = (persona.get("home_dong_code") or "")[:5]
    work_dist5 = (persona.get("work_dong_code") or "")[:5]
    tendency = persona.get("tendency") or ""

    for e in events:
        cat = e.get("category")
        if cat in INTERNAL_CATS or not e.get("poi_id"):
            e["actual_satisfaction"] = round(0.6 + rng.uniform(-0.1, 0.1), 2)
            continue

        score = 0.5
        # 페르소나 top categories 매칭
        sub = e.get("sub_category") or ""
        if sub in top_wd or sub in top_we or cat in top_wd or cat in top_we:
            score += 0.10
        # 성향 가중
        if tendency == "소비형":
            score += 0.05
        elif tendency == "절약형":
            score -= 0.03
        # 정책 적용 카테고리 + 거주·직장 자치구
        if policy_target_cats and policy_district_code:
            if (cat in policy_target_cats or sub in policy_target_cats) and \
               (home_dist5 == policy_district_code or work_dist5 == policy_district_code):
                score += 0.10
        # 노이즈
        score += rng.uniform(-0.10, 0.10)
        e["actual_satisfaction"] = round(max(0.0, min(1.0, score)), 2)
    return events


def _parse_top_cats(raw_json: str | None) -> set[str]:
    if not raw_json:
        return set()
    try:
        import json
        d = json.loads(raw_json)
        return set(d.keys())
    except Exception:
        return set()


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
       toString(round(i.actual_satisfaction * 100) / 100.0) AS summary
CREATE (m:Memory {
  id: 'mem_' + randomUUID(),
  type: 'visited',
  day: date($yesterday),
  importance: importance,
  summary: summary,
  satisfaction: i.actual_satisfaction
})
CREATE (a)-[:REMEMBERS {day: date($yesterday)}]->(m)
CREATE (m)-[:ABOUT_POI]->(poi)

// KNOWS_POI MERGE + 집계 갱신
MERGE (a)-[kp:KNOWS_POI]->(poi)
ON CREATE SET
  kp.since = date($yesterday), kp.source = 'visited',
  kp.visit_count = 1, kp.avg_satisfaction = i.actual_satisfaction,
  kp.last_visit = date($yesterday),
  kp.affinity = 0.3 + 0.4 * i.actual_satisfaction
ON MATCH SET
  kp.visit_count = coalesce(kp.visit_count, 0) + 1,
  kp.avg_satisfaction = (coalesce(kp.avg_satisfaction, 0.5) * coalesce(kp.visit_count, 0) + i.actual_satisfaction)
                         / (coalesce(kp.visit_count, 0) + 1),
  kp.last_visit = date($yesterday),
  kp.affinity = coalesce(kp.affinity, 0.5) * 0.7 + i.actual_satisfaction * 0.3
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
     coalesce(prev.month_spent, 0) AS prev_month_spent,
     coalesce(prev.policy_lifecycle, '{}') AS prev_lc

// 오늘 INCLUDES 누적 이벤트 수 + 평균 만족도 계산
OPTIONAL MATCH (a)-[:HAS_PLAN {day: date($today)}]->(today_plan:Plan)-[i:INCLUDES]->()
WITH a, prev_balance, prev_energy, prev_mood, prev_fatigue, prev_month_spent, prev_lc,
     count(i) AS n_events,
     avg(i.actual_satisfaction) AS avg_sat,
     sum(CASE WHEN i.anchor STARTS WITH 'zone' OR NOT i.category IN ['집','직장']
              THEN $event_spend ELSE 0 END) AS today_spent

// mood EMA: 0.7 * prev + 0.3 * avg_sat
// fatigue: 0.5 * prev + 0.05 * n_events + (0.2 if low_sat else 0) - (0.1 if home_dominant else 0)
WITH a, prev_balance, prev_energy, prev_lc, prev_month_spent,
     toInteger(today_spent) AS today_spent,
     coalesce(avg_sat, prev_mood) AS today_avg_sat,
     n_events,
     0.7 * prev_mood + 0.3 * coalesce(avg_sat, prev_mood) AS new_mood,
     CASE
       WHEN coalesce(avg_sat, 0.5) < 0.3 THEN 0.5 * prev_fatigue + 0.05 * n_events + 0.2
       ELSE 0.5 * prev_fatigue + 0.05 * n_events
     END AS new_fatigue_raw

WITH a, prev_balance, prev_energy, prev_lc, prev_month_spent, today_spent, today_avg_sat, n_events,
     new_mood,
     CASE WHEN new_fatigue_raw > 1.0 THEN 1.0
          WHEN new_fatigue_raw < 0.0 THEN 0.0
          ELSE new_fatigue_raw END AS new_fatigue

MERGE (s:State {id: $aid + '_' + $today})
SET s.agent_id = $aid,
    s.day = date($today),
    s.balance = prev_balance - today_spent,
    s.energy = 0.8,
    s.yesterday_satisfaction = today_avg_sat,
    s.mood = new_mood,
    s.fatigue = new_fatigue,
    s.month_spent = prev_month_spent + today_spent,
    s.policy_lifecycle = prev_lc
MERGE (a)-[:HAS_STATE {day: date($today)}]->(s)
RETURN s.id AS state_id, s.balance AS balance, s.mood AS mood, s.fatigue AS fatigue
"""


def night_create_state(aid: str, today: date, event_spend: int = 12000) -> dict:
    """오늘 State 노드 CREATE. event_spend = 외출 이벤트당 평균 지출 (KRW)."""
    yesterday = today - timedelta(days=1)
    with driver_session() as s:
        r = s.run(NIGHT_STATE_CYPHER,
                  aid=aid, today=today.isoformat(), yesterday=yesterday.isoformat(),
                  event_spend=event_spend).single()
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
    s2, cands, m2 = call_stage2(args.aid, s1, ctx.persona, verbose=args.verbose)
    events = merge_to_final_events(s1, s2, ctx.persona)

    print("[4/5] 만족도 룰 적용")
    # POC 정책 P001 = 강남구(11680) + 식사/카페/디저트 L1
    events = simulate_satisfaction(
        ctx.persona, events,
        policy_target_cats={"식사", "카페", "디저트"},
        policy_district_code="11680",
        seed=hash(args.aid + str(today)),
    )

    print("[5/5] Plan 적재 + Night Phase")
    tokens_in = m1["tokens_in"] + (m2.get("tokens_in") or 0)
    tokens_out = m1["tokens_out"] + (m2.get("tokens_out") or 0)
    plan_id, n_inc = write_plan(args.aid, today, events, day_type, tokens_in, tokens_out)
    print(f"  Plan id={plan_id}, INCLUDES x {n_inc}")

    state = night_create_state(args.aid, today)
    print(f"  State: {state}")

    print("\n=== 최종 events ===")
    for e in events:
        print(f"  [{e['order']}] {e['time']} {e['anchor']:25} {(e['category'] or ''):6} sat={e['actual_satisfaction']} → {e['poi_id']} ({e['intent']})")

    print(f"\n총 tokens: in={tokens_in}, out={tokens_out}")
