"""P016 검출력 관문(2026-10-11). 정책 전 주가 끝난 뒤, 두 갈래를 돌리기 전에 실행기(4c 단계)가 부른다.

정답 지표 C1(참여 유통업체 매장의 국산 신선 농축산물 결제)을 같은 사람 정책 있음−없음으로 잴 때,
명부 n명·두 갈래 일수 D일로 80% 검출(양측 5%) 가능한 최소 차이를 정책 전 주 기록으로 미리 잰다.

  사람 i 의 D일 합 Y = μ_i + e. 같은 사람 두 갈래의 차 = e_on − e_off → 분산 2·Var(e).
  Var(e) 는 사람 안 하루 분산(정책 전 주 날끼리)을 D 배 한 것(날끼리 독립 가정, 사람 평균 차이는 짝 비교에서 빠진다).
  최소 차이 = 2.8 × sqrt(2·D·σ²_within) / sqrt(n), 기준(D일 1인 평균)에 대한 % 로 낸다.

가정의 방향: 두 갈래가 같은 사람·같은 이력에서 출발하므로 e_on, e_off 는 양(+)의 상관이 있을 수 있다 → 실제 최소 차이는
이 값보다 작을 수 있다(보수적). 장보기는 며칠에 한 번이라 날끼리 음(−)의 상관(어제 봤으면 오늘 안 봄)이 있으면 D일 합의
분산은 D·σ² 보다 작다 → 이것도 보수적이다.

값을 지어내지 않는다: 품목 금액은 엔진이 Stage 2 에서 받은 produce_spent 그대로, 참여 매장 판정은 정책 파일의 이름 규칙 그대로.
통과 기준(--max-mde)은 실행 전에 설계 문서에 적은 값이다(data/experiments/P016_DESIGN_20261011.md).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from neo4j_load._common import driver_session  # noqa: E402

Q = """
UNWIND $ids AS aid
MATCH (a:Agent {id: aid})-[h:HAS_PLAN]->(p:Plan)-[i:INCLUDES]->(poi:POI)
WHERE toString(h.day) IN $days AND coalesce(i.actual_spent, 0) > 0
RETURN aid, toString(h.day) AS day, poi.name AS name, i.actual_spent AS spent, i.produce_spent AS prod,
       head([(poi)-[:IN_CATEGORY]->(c:Category) | c.name]) AS sub
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--start", required=True, help="정책 전 주 첫날")
    ap.add_argument("--days", type=int, required=True, help="정책 전 주 일수")
    ap.add_argument("--arm-days", type=int, required=True, help="두 갈래 일수 D")
    ap.add_argument("--max-mde", type=float, default=25.0, help="C1 최소 차이(%%) 상한 — 넘으면 통과 못 함")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ids = json.load(open(a.roster, encoding="utf-8"))
    ids = ids if isinstance(ids, list) else ids.get("ids") or ids.get("agents")
    pol = json.load(open(a.policy, encoding="utf-8"))
    rx = re.compile(pol["eligibility"]["include"]["name_regex"])
    subs = set(pol["eligibility"]["include"]["subs"])
    s0 = date.fromisoformat(a.start)
    days = [str(s0 + timedelta(i)) for i in range(a.days)]
    with driver_session() as s:
        rows = s.run(Q, ids=ids, days=days).data()
    # 지표별 사람×날 값
    M = {"C1 참여 매장 농축산물": lambda r: (r["prod"] or 0) if (r["sub"] in subs and rx.match(r["name"] or "")) else 0,
         "C2 참여 매장 결제 전체": lambda r: (r["spent"] or 0) if (r["sub"] in subs and rx.match(r["name"] or "")) else 0,
         "장보기 농축산물 전체(모든 가게)": lambda r: (r["prod"] or 0),
         "장보기 업종 결제 전체": lambda r: (r["spent"] or 0) if r["sub"] in subs else 0}
    cell = {k: defaultdict(float) for k in M}
    n_prod = sum(1 for r in rows if r["prod"] is not None)
    for r in rows:
        for k, f in M.items():
            cell[k][(r["aid"], r["day"])] += f(r)
    n, D = len(ids), a.arm_days
    out = {"made": str(date.today()), "roster": len(ids), "pre_days": days, "arm_days": D,
           "max_mde_pct": a.max_mde, "payments": len(rows), "payments_with_produce": n_prod, "metrics": {}}
    for k in M:
        per = {i: [cell[k][(i, d)] for d in days] for i in ids}
        mean_day = st.mean(v for xs in per.values() for v in xs)
        within = st.mean(st.pvariance(xs) * len(xs) / max(1, len(xs) - 1) for xs in per.values())
        sd_diff = math.sqrt(2 * D * within)
        base = mean_day * D
        mde = 2.8 * sd_diff / math.sqrt(n)
        users = sum(1 for xs in per.values() if sum(xs) > 0)
        out["metrics"][k] = {"base_won_per_person": round(base), "mde_won": round(mde),
                             "mde_pct": round(100 * mde / base, 1) if base > 0 else None,
                             "people_with_any": users}
        print(f"  {k}: {D}일 1인 기준 {base:,.0f}원 · 정책 전 주에 한 번이라도 산 사람 {users}/{n} · "
              f"80% 검출 최소 차이 {mde:,.0f}원 = {out['metrics'][k]['mde_pct']}%")
    c1 = out["metrics"]["C1 참여 매장 농축산물"]["mde_pct"]
    out["pass"] = bool(c1 is not None and c1 <= a.max_mde)
    out["reason"] = (f"C1 최소 차이 {c1}% {'≤' if out['pass'] else '>'} 상한 {a.max_mde}%"
                     if c1 is not None else "정책 전 주에 참여 매장 농축산물 결제가 하나도 없다")
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"검출력 관문: {'통과' if out['pass'] else '통과 못 함'} — {out['reason']} · 정답지 +6.957% 와 견줄 것")
    return 0 if out["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
