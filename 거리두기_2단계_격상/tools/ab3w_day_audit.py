"""본런 하루 검수 (2026-10-06) — 결제 원장·선택 이유·기억이 제대로 적재됐는지, 이유가 그날 배경과 맞는지.

    python tools/ab3w_day_audit.py <BASE> <arm: pre|on|off> <YYYY-MM-DD> [--json out.json] [--samples 6]

그래프(그 갈래의 Neo4j)에서 명부 사람들의 그날 계획(Plan -INCLUDES-> POI)·상태(State)·기억(Memory)을 읽는다.
읽기만 한다. 실행 중인 런과 같은 그래프를 읽어도 된다(그날이 끝난 뒤에 부를 것).

하드 오류(있으면 종료 코드 2): 계획·상태가 없는 사람, 음수 결제, 정책 결제 > 결제액, 정책 없는 갈래(pre/off)의
정책 결제·할인·환급, 이유(reasoning)·계기(trigger) 빈 외출, 업종 없는 가게, 실행 ID 섞임.
주의(종료 코드 0, 보고만): 0원 외출, 평소 하루 소비의 10배 넘는 결제, 정책 없는 갈래의 trigger=policy,
그날 규제와 어긋나 보이는 방문(매장 취식 마감 뒤 식사·카페, 집합금지 업종), 이유에 규제를 언급한 비율.
표본: 이유문 몇 개(사람·시각·업종·금액·계기·이유·가게 고른 이유)를 그대로 보여 준다 — 사람이 읽고 배경과 맞는지 본다.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "sim"))

TRIGGERS = {"appointment", "rumor", "policy", "lifestyle", "mood", "none"}
HOME_WORK = {"집", "직장"}
DINE = {"식사", "카페", "주점", "디저트"}
CLOSED_HINT = re.compile(r"유흥|단란|클럽|나이트|감성주점|헌팅|콜라텍|무도장|홀덤")
RULE_WORDS = re.compile(r"거리두기|방역|확진|코로나|포장|배달만|매장 ?취식|영업 ?시간|\d{1,2}시 ?이후|\d{1,2}:00|모임|인원")

Q_PLANS = """
UNWIND $aids AS aid
MATCH (a:Agent {id: aid})
OPTIONAL MATCH (a)-[:HAS_PLAN {day: date($day)}]->(p:Plan)
OPTIONAL MATCH (p)-[i:INCLUDES]->(poi:POI)
OPTIONAL MATCH (poi)-[:IN_CATEGORY]->(c:Category)
WITH a, p, i, poi, head(collect(c)) AS c
RETURN a.id AS aid, a.s_daily_wd AS daily, p.id AS plan,
       i.order AS ord, toString(i.time) AS t, i.category AS cat, i.sub_category AS sub, i.anchor AS anchor,
       i.actual_spent AS amt, i.spent_from_policy AS sfp, i.instant_discount AS disc, i.policy_rebate AS reb,
       i.reasoning AS why, i.trigger AS trig, i.pick_reason AS pick, i.pick_factor AS factor,
       poi.id AS poi_id, poi.name AS poi_name, c.name AS poi_sub, c.parent AS poi_l1
"""
Q_STATE = """
UNWIND $aids AS aid
OPTIONAL MATCH (s:State {id: aid + '_' + $day})
RETURN aid, s IS NOT NULL AS has_state, s.today_spent AS today_spent, s.run_id AS run_id
"""
Q_MEM = """
UNWIND $aids AS aid
MATCH (a:Agent {id: aid})
OPTIONAL MATCH (a)--(m:Memory) WHERE m.day = date($day)
RETURN aid, count(m) AS n, collect(DISTINCT m.type) AS types
"""


def jd(v):
    if not v:
        return {}
    if isinstance(v, dict):
        return v
    try:
        d = json.loads(v)
        return d if isinstance(d, dict) else {}
    except (TypeError, ValueError):
        return {}


def env_rules(base: Path, arm: str, day: str) -> dict:
    """그날 그 갈래의 사회 배경 규칙(실행 기록의 환경 이름으로 다시 만든다)."""
    m = json.loads((base / "run_manifest.json").read_text(encoding="utf-8"))
    env = m.get({"pre": "env_pre", "on": "env_on", "off": "env_off"}[arm]) or ""
    if not env:
        return {"env": "", "cutoff": None, "closed": []}
    from environments import covid_2021  # noqa: E402
    rule_day = date(2020, 11, 23) if env == "covid_2020_hold_1123" else None
    reg = covid_2021._regime_for(rule_day or date.fromisoformat(day))  # noqa: SLF001
    if not reg:
        return {"env": env, "cutoff": None, "closed": []}
    cutoff = covid_2021._effective(reg, "dine_in_cutoff")  # noqa: SLF001
    closed = covid_2021._effective(reg, "closed_facilities") or []  # noqa: SLF001
    return {"env": env, "level": reg.get("level"), "cutoff": cutoff, "closed": list(closed)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("arm", choices=["pre", "on", "off"]); ap.add_argument("day")
    ap.add_argument("--json"); ap.add_argument("--samples", type=int, default=6)
    a = ap.parse_args()
    base = Path(a.base)
    aids = [str(x) for x in json.loads((base / "roster.json").read_text(encoding="utf-8"))]
    man = json.loads((base / "run_manifest.json").read_text(encoding="utf-8"))
    policy_arm = a.arm == "on" and bool(man.get("policy_id"))
    from neo4j import GraphDatabase
    drv = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=("neo4j", os.environ["NEO4J_PASSWORD"]))
    with drv.session() as s:
        rows = [dict(r) for r in s.run(Q_PLANS, aids=aids, day=a.day)]
        states = {r["aid"]: dict(r) for r in s.run(Q_STATE, aids=aids, day=a.day)}
        mems = {r["aid"]: dict(r) for r in s.run(Q_MEM, aids=aids, day=a.day)}
    drv.close()

    rules = env_rules(base, a.arm, a.day)
    hard, warn = {}, {}

    def bump(d, k, n=1):
        d[k] = d.get(k, 0) + n

    plans = {r["aid"] for r in rows if r["plan"]}
    no_plan = [x for x in aids if x not in plans]
    no_state = [x for x in aids if not states.get(x, {}).get("has_state")]
    if no_plan: bump(hard, "계획 없는 사람", len(no_plan))
    if no_state: bump(hard, "상태(State) 없는 사람", len(no_state))
    run_ids = {states[x].get("run_id") for x in aids if states.get(x, {}).get("has_state")} - {None}
    if len(run_ids) > 1: bump(hard, "실행 ID 가 섞임", len(run_ids))

    ev = [r for r in rows if r["ord"] is not None]
    outings = [r for r in ev if (r["cat"] or "") not in HOME_WORK]
    paid = [r for r in outings if (r["amt"] or 0) > 0]
    total = sum(int(r["amt"] or 0) for r in ev)
    rule_mention = 0
    cutoff_h = None
    if rules.get("cutoff"):
        m = re.match(r"(\d{1,2})", str(rules["cutoff"]))
        cutoff_h = int(m.group(1)) if m else None
    for r in ev:
        amt = r["amt"]
        if amt is None or not isinstance(amt, int) or amt < 0: bump(hard, "결제액이 음수·비정수")
        sfp, disc, reb = jd(r["sfp"]), jd(r["disc"]), jd(r["reb"])
        pol_amt = sum(int(v or 0) for v in sfp.values())
        if pol_amt > int(amt or 0): bump(hard, "정책 결제 > 결제액")
        if not policy_arm and (pol_amt or any(disc.values()) or any(reb.values())):
            bump(hard, "정책 없는 갈래에서 정책 결제·할인·환급")
        if (r["cat"] or "") in HOME_WORK:
            continue
        if not (r["why"] or "").strip(): bump(hard, "이유(reasoning) 빈 외출")
        if (r["trig"] or "") not in TRIGGERS: bump(hard, "계기(trigger) 없음·잘못된 값")
        if not policy_arm and r["trig"] == "policy": bump(warn, "정책 없는 갈래인데 trigger=policy")
        # 가게 고르는 단계를 건너뛴 결제(약속 장소·이미 정한 가게)는 pick_reason 이 비는 것이 정상이다 — 주의로만 센다
        if (amt or 0) > 0 and not (r["pick"] or "").strip(): bump(warn, "가게 고른 이유 없음(가게 고르기 생략)")
        if r["poi_id"] and not r["poi_sub"]: bump(hard, "업종 없는 가게")
        if (amt or 0) == 0: bump(warn, "0원 외출")
        daily = float(r["daily"] or 0)
        if daily > 0 and (amt or 0) > 10 * daily: bump(warn, "평소 하루 소비의 10배 넘는 결제")
        if RULE_WORDS.search(r["why"] or ""): rule_mention += 1
        hh = int((r["t"] or "00")[:2]) if r["t"] else None
        if cutoff_h is not None and hh is not None and hh >= cutoff_h and (r["cat"] in DINE):
            bump(warn, f"매장 취식 마감({rules['cutoff']}) 뒤 식사·카페·주점 방문")
        if rules.get("closed") and CLOSED_HINT.search(f"{r['poi_sub'] or ''} {r['poi_name'] or ''}"):
            bump(warn, "집합금지 업종으로 보이는 가게 방문")
    mem_n = sum(mems.get(x, {}).get("n", 0) for x in aids)
    no_mem_people = sum(1 for x in aids if not mems.get(x, {}).get("n"))
    big = sorted((r for r in paid), key=lambda r: -(r["amt"] or 0))[:3]
    rnd = random.Random(f"{a.day}{a.arm}")
    sample = rnd.sample(paid, min(a.samples, len(paid))) if paid else []
    pol_sample = [r for r in paid if r["trig"] == "policy"][:2]

    out = {
        "base": str(base), "arm": a.arm, "day": a.day, "people": len(aids), "policy_arm": policy_arm,
        "background": rules, "events": len(ev), "outings": len(outings), "paid_outings": len(paid),
        "total_spent": total, "per_person": round(total / max(1, len(aids))),
        "memory_nodes_dated_today": mem_n, "people_without_memory_today": no_mem_people,
        "reasons_mentioning_rules": f"{rule_mention}/{len(outings)}",
        "hard_errors": hard, "warnings": warn,
        "largest": [{k: r[k] for k in ("aid", "t", "cat", "sub", "amt", "daily", "poi_name")} for r in big],
        "samples": [{k: r[k] for k in ("aid", "t", "cat", "sub", "amt", "trig", "why", "pick", "poi_name")}
                    for r in sample + pol_sample],
    }
    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"== {base.name} {a.arm} {a.day}: 사람 {len(aids)} · 외출 {len(outings)} · 결제 {len(paid)} · 1인 {out['per_person']:,}원 · "
          f"기억 {mem_n}(오늘 기억 없는 사람 {no_mem_people}) · 규제 언급 {out['reasons_mentioning_rules']} · 배경 {rules.get('env') or '없음'} {rules.get('level') or ''}")
    print("  하드 오류:", hard or "없음")
    print("  주의:", warn or "없음")
    for r in out["largest"]:
        print(f"  큰 결제: {r['aid']} {r['t']} {r['cat']}/{r['sub']} {r['amt']:,}원 (평소 하루 {r['daily']}) {r['poi_name']}")
    for r in out["samples"]:
        print(f"  표본 {r['aid']} {r['t']} {r['cat']}/{r['sub']} {r['amt']:,}원 [{r['trig']}] {r['poi_name']}\n"
              f"     이유: {(r['why'] or '')[:220]}\n     가게: {(r['pick'] or '')[:160]}")
    sys.exit(2 if hard else 0)


if __name__ == "__main__":
    main()
