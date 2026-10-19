"""P012 캐시백 문턱의 기준(2분기 월평균에 해당)을 그 사람 자신의 정책 전 주 실제 적립업종 결제로 정한다 (2026-10-07b).

실제 제도: 문턱 = 본인 2분기 카드 사용 월평균 × 1.03. 예전 엔진은 사람별 기준이 없어 (앵커 × 0.268) 근사를 썼고,
새 엔진에서는 정책 없이도 76% 가 문턱을 넘었다. 여기서는 정책 전 주(두 갈래 공통) 의 적립업종(POI.sangsaeng_eligible)
결제를 평일·쉬는 날로 나눠 평균하고 (평일×5 + 쉬는 날×2)/7 을 하루 기준으로 Agent.sangsaeng_base_daily 에 넣는다.
엔진(dawn_context._sangsaeng_monthly_anchor)과 내보내기(export_cashback_month)가 이 속성을 먼저 읽는다.
복제 전에 정책 있음 그래프에 넣으므로 두 갈래가 같은 값을 갖는다. 한 칸이 비면 다른 칸 값을 쓴다.
"""
import argparse, json, sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "sim"))
from scripts.neo4j_load._common import driver_session
from kr_holidays import is_day_off

ap = argparse.ArgumentParser()
ap.add_argument("--roster", required=True)
ap.add_argument("--days", required=True, help="정책 전 주 날짜, 쉼표")
ap.add_argument("--out", required=True)
a = ap.parse_args()
ids = [r["agent_id"] if isinstance(r, dict) else r for r in json.load(open(a.roster, encoding="utf-8"))]
days = [d for d in a.days.split(",") if d]
Q = """UNWIND $ids AS aid
MATCH (ag:Agent {id: aid})
UNWIND $days AS d
OPTIONAL MATCH (ag)-[:HAS_PLAN]->(p:Plan {day: date(d)})-[i:INCLUDES]->(x:POI)
WITH aid, d, sum(CASE WHEN x.sangsaeng_eligible = true THEN coalesce(i.actual_spent, 0) ELSE 0 END) AS won
RETURN aid, d, won"""
per = {}
with driver_session() as s:
    for r in s.run(Q, ids=ids, days=days):
        per.setdefault(r["aid"], {})[r["d"]] = int(r["won"] or 0)
rows, base = [], {}
for aid in ids:
    m = per.get(aid, {})
    wd = [v for d, v in m.items() if not is_day_off(date.fromisoformat(d))]
    we = [v for d, v in m.items() if is_day_off(date.fromisoformat(d))]
    mw = sum(wd) / len(wd) if wd else None
    me = sum(we) / len(we) if we else None
    if mw is None: mw = me
    if me is None: me = mw
    b = round((5 * (mw or 0) + 2 * (me or 0)) / 7, 1)
    base[aid] = b
    rows.append({"aid": aid, "weekday_mean": mw, "dayoff_mean": me, "base_daily": b, "n_wd": len(wd), "n_off": len(we)})
with driver_session() as s:
    s.run("UNWIND $rows AS r MATCH (ag:Agent {id: r.aid}) SET ag.sangsaeng_base_daily = r.b",
          rows=[{"aid": k, "b": v} for k, v in base.items()]).consume()
    n = s.run("MATCH (ag:Agent) WHERE ag.id IN $ids AND ag.sangsaeng_base_daily IS NOT NULL RETURN count(ag) AS n", ids=ids).single()["n"]
json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
vals = sorted(base.values())
print(f"sangsaeng_base_daily 적재 {n}/{len(ids)} · 0원 {sum(1 for v in vals if v <= 0)} · 중앙 {vals[len(vals)//2]:,.0f}원")
if n != len(ids):
    sys.exit("기준을 못 넣은 사람이 있다")
