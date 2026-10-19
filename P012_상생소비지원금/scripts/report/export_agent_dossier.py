"""시뮬 뒤 반드시 남아야 하는 것을 그래프에서 **복원 없이 읽을 수 있게** 뽑는다.

    python scripts/report/export_agent_dossier.py \
        --roster out/roster.json --start 2021-10-01 --end 2021-10-31 \
        --arm on --out out/on/dossier.jsonl

## 왜 따로 뽑는가

그래프 덤프(neo4j.dump)에도 다 들어 있다. 그러나 덤프는 **복원해야** 읽을 수 있고,
두 팔을 돌리면 뒤쪽 팔이 앞쪽 팔의 그래프를 덮는다. 1대1 인터뷰는 앞쪽 팔(정책
있음)의 기억을 읽어야 하므로, 팔이 끝난 그 자리에서 사람 단위로 빼 둔다.

## 한 줄에 무엇이 들어가는가 (에이전트 한 명)

    memories   [:REMEMBERS]->(:Memory)  — 방문 기억·소문. 무엇을 왜 골랐고 만족했나
    plans      (:Plan)-[:INCLUDES]->(:POI) — 스케줄(시각·순서·의도)과 실제 지출
    states     (:State)                 — 날짜별 지갑·기분·피로·정책 단계
    totals     위 세 개에서 센 합계 — 원장과 맞대는 자리

## 손실을 '없기를 바라는 것' 이 아니라 '검사' 로 막는다

에이전트마다 요구한 날 수만큼 State 가 있어야 하고, 명부의 사람이 하나도 빠지면
안 된다. 어긋나면 파일을 쓰지 않고 멈춘다 — 조용히 반쪽짜리를 남기지 않는다.
`--allow-missing-days` 는 그 검사를 **완화가 아니라 기록**으로 바꾼다(빠진 날을
파일에 적는다). 기억은 그날 외출이 없으면 없을 수 있으므로 날짜 검사를 하지
않고, 대신 0 인 사람 수를 세어 요약에 낸다.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def days_between(a: str, b: str) -> list[str]:
    d0, d1 = date.fromisoformat(a), date.fromisoformat(b)
    if d1 < d0:
        raise ValueError("end before start")
    return [(d0 + timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]


def as_str(v):
    """Neo4j Date/DateTime 을 문자열로 — json 이 삼키게."""
    return v if v is None or isinstance(v, (str, int, float, bool)) else str(v)


def clean(m: dict) -> dict:
    return {k: as_str(v) for k, v in m.items() if v is not None}


Q_MEM = """
MATCH (a:Agent {id: $aid})-[:REMEMBERS]->(m:Memory)
WHERE m.day >= date($start) AND m.day <= date($end)
RETURN m.day AS day, m.type AS type, m.summary AS summary, m.why AS why,
       m.pick_why AS pick_why, m.category AS category, m.spent AS spent,
       m.paid_policy AS paid_policy, m.satisfaction AS satisfaction,
       m.importance AS importance, m.trigger AS trigger, m.id AS id
ORDER BY m.day, m.id
"""

Q_PLAN = """
MATCH (a:Agent {id: $aid})-[:HAS_PLAN]->(p:Plan)
WHERE p.day >= date($start) AND p.day <= date($end)
OPTIONAL MATCH (p)-[i:INCLUDES]->(poi:POI)
RETURN p.day AS day, p.day_type AS day_type, p.generated_at AS generated_at,
       collect({
         order: i.order, time: i.time, intent: i.intent,
         poi: poi.name, poi_id: poi.id, dong_code: poi.dong_code,
         category: i.category, sub_category: i.sub_category,
         desired_spent: i.desired_spent, actual_spent: i.actual_spent,
         spent_from_policy: i.spent_from_policy,
         purchase_status: i.purchase_status, coupon_eligible: i.coupon_eligible,
         instant_discount: i.instant_discount, price_band: i.price_band,
         pick_reason: i.pick_reason, reasoning: i.reasoning,
         trigger: i.trigger, with_agents: i.with_agents,
         actual_satisfaction: i.actual_satisfaction
       }) AS items
ORDER BY p.day
"""

Q_STATE = """
MATCH (a:Agent {id: $aid})-[:HAS_STATE]->(st:State)
WHERE st.day >= date($start) AND st.day <= date($end)
RETURN st.day AS day, st.balance AS balance, st.month_spent AS month_spent,
       st.sangsaeng_month_spent AS sangsaeng_month_spent, st.mood AS mood,
       st.fatigue AS fatigue, st.energy AS energy,
       st.policy_lifecycle AS policy_lifecycle,
       st.yesterday_satisfaction AS yesterday_satisfaction
ORDER BY st.day
"""

# 속성명은 그래프에 물어서 맞췄다 — Agent 는 p_*/personal_*/residence_* 를 쓴다.
Q_PROFILE = """
MATCH (a:Agent {id: $aid})
RETURN a.id AS id, a.personal_age AS age, a.p_age_group AS age_group,
       a.p_gender AS gender, a.personal_job_raw AS job,
       a.p_life_stage AS life_stage, a.p_income_level AS income_level,
       a.personality_lifestyle_raw AS lifestyle,
       a.personality_spending_tendency AS spending_tendency,
       a.residence_dong_name AS residence_dong, a.residence_gu AS gu,
       a.workplace_dong_name AS workplace_dong,
       a.workplace_commute_min AS commute_min,
       a.spending_level_wd AS spending_level_wd, a.s_daily_wd AS s_daily_wd,
       a.spending_top_wd_json AS spending_top_wd_json,
       a.behavior_mobility_level AS mobility_level
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--arm", required=True, choices=("on", "off"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    ap.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD", ""))
    ap.add_argument("--allow-missing-days", action="store_true",
                    help="빠진 날에 멈추지 않고 파일에 적는다(완화가 아니라 기록).")
    a = ap.parse_args()

    from neo4j import GraphDatabase          # 서버에서만 필요하다

    roster = json.loads(Path(a.roster).read_text(encoding="utf-8"))
    if isinstance(roster, dict):
        roster = list(roster)
    if not roster or len(set(roster)) != len(roster):
        raise SystemExit("roster is empty or has duplicates")
    want = days_between(a.start, a.end)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    drv = GraphDatabase.driver(a.uri, auth=(a.user, a.password))
    gaps, no_mem, tot = [], 0, {"memories": 0, "plan_items": 0, "states": 0, "spent": 0.0}
    tmp = out.with_suffix(out.suffix + ".part")
    try:
        with drv.session() as s, io.open(tmp, "w", encoding="utf-8", newline="\n") as f:
            for aid in roster:
                prof = s.run(Q_PROFILE, aid=aid).single()
                if prof is None:
                    raise SystemExit("agent missing from graph: %s" % aid)
                mem = [clean(dict(r)) for r in s.run(Q_MEM, aid=aid, start=a.start, end=a.end)]
                sts = [clean(dict(r)) for r in s.run(Q_STATE, aid=aid, start=a.start, end=a.end)]
                plans = []
                for r in s.run(Q_PLAN, aid=aid, start=a.start, end=a.end):
                    row = clean({k: v for k, v in dict(r).items() if k != "items"})
                    row["items"] = [clean(it) for it in (r["items"] or [])
                                    if it.get("poi_id") is not None]
                    plans.append(row)
                miss = sorted(set(want) - {d["day"] for d in sts})
                if miss:
                    gaps.append({"aid": aid, "missing_state_days": miss})
                if not mem:
                    no_mem += 1
                spent = sum(float(it.get("actual_spent") or 0)
                            for p in plans for it in p["items"])
                tot["memories"] += len(mem)
                tot["plan_items"] += sum(len(p["items"]) for p in plans)
                tot["states"] += len(sts)
                tot["spent"] += spent
                f.write(json.dumps({
                    "aid": aid, "arm": a.arm, "window": [a.start, a.end],
                    "profile": clean(dict(prof)),
                    "memories": mem, "plans": plans, "states": sts,
                    "totals": {"memories": len(mem), "states": len(sts),
                               "plan_days": len(plans),
                               "plan_items": sum(len(p["items"]) for p in plans),
                               "actual_spent": round(spent, 2)},
                }, ensure_ascii=False) + "\n")
    finally:
        drv.close()

    if gaps and not a.allow_missing_days:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            "state days missing for %d agent(s); nothing written. 첫 사례: %s\n"
            "빠진 날을 기록하고 진행하려면 --allow-missing-days"
            % (len(gaps), json.dumps(gaps[0], ensure_ascii=False)))
    tmp.replace(out)

    man = {"arm": a.arm, "window": [a.start, a.end], "agents": len(roster),
           "expected_days": len(want), "totals": tot,
           "agents_without_memory": no_mem,
           "state_day_gaps": gaps,
           "sha256": __import__("hashlib").sha256(out.read_bytes()).hexdigest()}
    io.open(str(out) + ".manifest.json", "w", encoding="utf-8", newline="\n").write(
        json.dumps(man, ensure_ascii=False, indent=1))

    print("에이전트 %d명 · 요구 일수 %d" % (len(roster), len(want)))
    print("  기억 {:,}건 · 계획 항목 {:,}건 · 상태 {:,}건 · 실지출 {:,.0f}원"
          .format(tot["memories"], tot["plan_items"], tot["states"], tot["spent"]))
    print("  기억이 0 건인 사람 %d명 (외출이 없던 사람은 정상)" % no_mem)
    if gaps:
        print("  **상태 날짜 빠짐 %d명** — manifest 에 적었다" % len(gaps))
    print("→ %s" % out)
    print("→ %s.manifest.json" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
