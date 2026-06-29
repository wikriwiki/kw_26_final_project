"""L2 spillover 정밀 측정 (4-case 분리).

day_health_check 10-D 의 L2 그룹은 거주지만 본다 — 거주 L2 + 직장 L1 (사업 구간) 케이스가
direct + spillover 혼재. POLICY_CYPHER 가 LIVES_AT OR WORKS_AT 둘 다 fetch 하므로
LLM 도 양쪽 신호를 다 받음.

이 스크립트는 거주 L2 를 직장 위치에 따라 4 case 로 분리해 1인당 매출과 Control 대비
격차, baseline vs 정책 활성 DID 를 계산한다.

CLI:
  python scripts/sim/precise_l2_did.py \\
      --start 2026-05-18 --days 6 --policy-from 2026-05-20 \\
      --out output/sim/report/L2_PRECISE_6D.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neo4j_load"))

from _common import driver_session  # noqa: E402


L1_DONGS = ["역삼1동", "역삼2동", "도곡1동"]
L2_DONGS = ["도곡2동", "삼성1동", "삼성2동", "논현1동", "논현2동"]

# 거주 L2 의 4 case (직장 위치별)
CASES = {
    "pure_L2":          "거주 L2 ∩ 직장 비강남(또는 직장 없음) — 순수 spillover",
    "mixed_L2_L1work":  "거주 L2 ∩ 직장 L1 — direct + spillover 혼재",
    "both_walk":        "거주 L2 ∩ 직장 L2 — 양쪽 도보권",
    "L2_gangnam_other": "거주 L2 ∩ 직장 강남 비도보권 — spillover (직장 동선 일부 강남)",
}


def _classify_workplace(home_dong: str, work_dong: str | None, work_gu: str | None) -> str | None:
    if home_dong not in L2_DONGS:
        return None
    if work_dong in L1_DONGS:
        return "mixed_L2_L1work"
    if work_dong in L2_DONGS:
        return "both_walk"
    if work_gu == "강남구":
        return "L2_gangnam_other"
    return "pure_L2"  # 비강남 직장 또는 직장 없음


def measure_day(s, day: str) -> dict:
    """주어진 날짜의 거주 L2 4-case 1인당 매출 + Control."""
    # 분모 계산: 거주 L2 agent 의 직장 그룹별 카운트
    denom_rows = s.run("""
        MATCH (a:Agent)-[:LIVES_AT]->(:POI)-[:IN_DONG]->(hd:Dong)
        WHERE hd.name IN $l2
        OPTIONAL MATCH (a)-[:WORKS_AT]->(:POI)-[:IN_DONG]->(wd:Dong)
                       <-[:HAS_DONG]-(wgu:District)
        WITH a, hd.name AS home_dong, wd.name AS work_dong, wgu.name AS work_gu
        RETURN home_dong, work_dong, work_gu, count(a) AS n
    """, l2=L2_DONGS).data()

    denoms = {k: 0 for k in CASES}
    aid_to_case: dict[str, str] = {}  # not needed since we do separate aggregation; left for clarity
    for row in denom_rows:
        case = _classify_workplace(row["home_dong"], row["work_dong"], row["work_gu"])
        if case:
            denoms[case] += row["n"]

    # Control 분모 — 비강남 거주
    n_control = s.run("""
        MATCH (a:Agent)-[:LIVES_AT]->(:POI)-[:IN_DONG]->(:Dong)
              <-[:HAS_DONG]-(gu:District)
        WHERE gu.name <> '강남구'
        RETURN count(a) AS n
    """).single()["n"]

    # 매출 — 각 agent 의 카페·디저트 외출 매출 합 → case 별 집계
    sales_rows = s.run("""
        MATCH (a:Agent)-[:LIVES_AT]->(:POI)-[:IN_DONG]->(hd:Dong)
        WHERE hd.name IN $l2
        OPTIONAL MATCH (a)-[:WORKS_AT]->(:POI)-[:IN_DONG]->(wd:Dong)
                       <-[:HAS_DONG]-(wgu:District)
        OPTIONAL MATCH (a)-[:HAS_PLAN {day: date($d)}]->(:Plan)
                       -[i:INCLUDES]->(:POI {type:'commerce'})
        WITH a, hd.name AS home_dong, wd.name AS work_dong, wgu.name AS work_gu,
             sum(CASE WHEN i.category IN ['카페','디저트']
                      THEN coalesce(i.actual_spent, 0) ELSE 0 END) AS spent
        RETURN home_dong, work_dong, work_gu, sum(spent) AS s, count(a) AS n
    """, d=day, l2=L2_DONGS).data()

    spends = {k: 0 for k in CASES}
    for row in sales_rows:
        case = _classify_workplace(row["home_dong"], row["work_dong"], row["work_gu"])
        if case:
            spends[case] += row["s"]

    # Control 매출
    s_control = s.run("""
        MATCH (a:Agent)-[:LIVES_AT]->(:POI)-[:IN_DONG]->(:Dong)
              <-[:HAS_DONG]-(gu:District)
        WHERE gu.name <> '강남구'
        OPTIONAL MATCH (a)-[:HAS_PLAN {day: date($d)}]->(:Plan)
                       -[i:INCLUDES]->(:POI {type:'commerce'})
        WITH sum(CASE WHEN i.category IN ['카페','디저트']
                      THEN coalesce(i.actual_spent, 0) ELSE 0 END) AS spent
        RETURN spent AS s
    """, d=day).single()["s"]

    per_case = {k: (spends[k] / denoms[k] if denoms[k] else 0) for k in CASES}
    per_control = s_control / n_control if n_control else 0

    return {
        "day": day,
        "denoms": denoms,
        "n_control": n_control,
        "spends": spends,
        "s_control": s_control,
        "per_case": per_case,
        "per_control": per_control,
    }


def aggregate(records: list[dict], days: list[str]) -> dict:
    """주어진 날짜 집합의 평균 1인당 매출."""
    selected = [r for r in records if r["day"] in days]
    if not selected:
        return {}
    out = {"days": days, "n_days": len(selected)}
    for case in CASES:
        spends = sum(r["spends"][case] for r in selected)
        denoms = sum(r["denoms"][case] for r in selected)
        out[case] = spends / denoms if denoms else 0
    sc = sum(r["s_control"] for r in selected)
    nc = sum(r["n_control"] for r in selected)
    out["control"] = sc / nc if nc else 0
    return out


def diff_vs_control(group_val: float, control_val: float) -> float:
    if control_val <= 0:
        return 0.0
    return (group_val - control_val) / control_val * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-18")
    ap.add_argument("--days", type=int, default=6)
    ap.add_argument("--policy-from", default="2026-05-20")
    ap.add_argument("--baseline-days", default="2026-05-19",
                    help="baseline 일자 (콤마 구분). 기본=5/19 단일")
    ap.add_argument("--out", default="output/sim/report/L2_PRECISE_6D.md")
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    all_days = [(start + timedelta(days=i)).isoformat() for i in range(args.days)]
    baseline_days = [d.strip() for d in args.baseline_days.split(",") if d.strip()]
    policy_start = date.fromisoformat(args.policy_from)
    policy_days = [d for d in all_days if date.fromisoformat(d) >= policy_start]

    print(f"[precise_l2_did] start={args.start} days={args.days} policy_from={args.policy_from}")
    print(f"  baseline_days={baseline_days}")
    print(f"  policy_days={policy_days}")

    with driver_session() as s:
        # 한 번에 모든 날짜 측정
        records = []
        for d in all_days:
            r = measure_day(s, d)
            records.append(r)
            print(f"  measured {d}: pure_L2={r['per_case']['pure_L2']:,.0f}원 "
                  f"control={r['per_control']:,.0f}원")

    base = aggregate(records, baseline_days)
    pol = aggregate(records, policy_days)

    # ─ Markdown 작성 ─────────────────────────────────────
    L: list[str] = []
    L.append("# L2 spillover 정밀 측정 (4-case 분리)")
    L.append("")
    L.append(f"- 시뮬 기간: {args.start} ~ {all_days[-1]} ({args.days}일)")
    L.append(f"- 정책 시행일: {args.policy_from}")
    L.append(f"- baseline 일자: {', '.join(baseline_days)}")
    L.append(f"- 정책 활성 일자: {', '.join(policy_days)}")
    L.append("")
    L.append("## 1. 분석 동기")
    L.append("")
    L.append("`day_health_check.py` 10-D 의 L2 (도보권 거주) 그룹은 거주지만 보고 분류한다. "
             "그러나 P008 정책의 `POLICY_CYPHER`는 `LIVES_AT OR WORKS_AT` 둘 다 fetch하므로, "
             "거주 L2 + 직장 L1(사업 구간)인 agent는 *direct + spillover 혼재* 신호를 받는다. "
             "이 스크립트는 거주 L2 를 직장 위치별 4 case 로 분리해, *진정한 spillover* 만 추출한다.")
    L.append("")
    L.append("## 2. 분모 (거주 L2 agent의 직장 분포)")
    L.append("")
    L.append("| Case | 의미 | n |")
    L.append("|---|---|---:|")
    for case, desc in CASES.items():
        L.append(f"| **{case}** | {desc} | {base.get(case, 0) and records[0]['denoms'][case]:,} |")
    L.append(f"| (Control) | 비강남 거주 | {records[0]['n_control']:,} |")
    L.append("")
    L.append("> 분모는 시뮬 첫날 기준 (agent population은 변하지 않음).")
    L.append("")

    L.append("## 3. baseline 일자 평균 1인당 매출 (카페·디저트)")
    L.append("")
    L.append("| Case | 1인당 매출 (원) | vs Control |")
    L.append("|---|---:|---:|")
    base_ctrl = base.get("control", 0)
    for case in CASES:
        v = base.get(case, 0)
        diff = diff_vs_control(v, base_ctrl)
        L.append(f"| {case} | {v:,.0f} | {diff:+.1f}% |")
    L.append(f"| Control (비강남 거주) | {base_ctrl:,.0f} | — |")
    L.append("")

    L.append("## 4. 정책 활성 일자 평균 1인당 매출")
    L.append("")
    L.append("| Case | 1인당 매출 (원) | vs Control |")
    L.append("|---|---:|---:|")
    pol_ctrl = pol.get("control", 0)
    for case in CASES:
        v = pol.get(case, 0)
        diff = diff_vs_control(v, pol_ctrl)
        L.append(f"| {case} | {v:,.0f} | {diff:+.1f}% |")
    L.append(f"| Control | {pol_ctrl:,.0f} | — |")
    L.append("")

    L.append("## 5. DID — 정책 효과 (baseline 격차 → 정책 활성 격차)")
    L.append("")
    L.append("| Case | baseline 격차 | 정책 활성 격차 | **DID 효과** |")
    L.append("|---|---:|---:|---:|")
    for case in CASES:
        d_base = diff_vs_control(base.get(case, 0), base_ctrl)
        d_pol = diff_vs_control(pol.get(case, 0), pol_ctrl)
        L.append(f"| {case} | {d_base:+.1f}% | {d_pol:+.1f}% | **{d_pol - d_base:+.1f}%p** |")
    L.append("")
    L.append("> **pure_L2 의 DID 효과**가 P008 의 진정한 spillover 지표. "
             "`day_health_check.py` 의 L2 통합 수치는 이 4 case 의 가중평균으로, "
             "mixed_L2_L1work 의 direct 효과가 일부 섞여있다.")
    L.append("")

    # JSON 부산물도 저장
    payload = {
        "config": {
            "start": args.start, "days": args.days, "policy_from": args.policy_from,
            "baseline_days": baseline_days, "policy_days": policy_days,
        },
        "denoms": records[0]["denoms"] if records else {},
        "n_control": records[0]["n_control"] if records else 0,
        "by_day": records,
        "baseline_avg": base,
        "policy_avg": pol,
    }
    out_md = Path(args.out)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")
    out_json = out_md.with_suffix(".json")
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    print(f"\n[precise_l2_did] wrote {out_md}")
    print(f"[precise_l2_did] wrote {out_json}")


if __name__ == "__main__":
    main()
