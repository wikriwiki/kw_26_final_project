"""정책 지원금 사용처 분석 — INCLUDES.spent_from_policy 집계.

옵션 A로 적재된 거래별 정책 사용액을 분석:
1. 정책별 누적 사용액 (총합)
2. 정책 × 카테고리 분포 (어느 업종에 grant가 흘러갔나)
3. 정책 × 자치구 분포 (어느 지역 매출에 기여)
4. 잔여액 분포 (덜 쓴 페르소나 분석)
5. 거래별 평균 policy_spend 비중 (= policy_spend / actual_spent)

CLI:
  python scripts/sim/analyze_policy_spend.py --start 2026-05-25 --days 4 \\
      --out output/sim/report/POLICY_SPEND_ANALYSIS.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neo4j_load"))

from _common import driver_session


def fetch_policy_spend(start: date, days: int) -> dict:
    """모든 거래의 spent_from_policy JSON 파싱 → 정책별 사용처 집계."""
    end = start + timedelta(days=days - 1)
    with driver_session() as s:
        rows = s.run(
            """
            MATCH (a:Agent)-[:HAS_PLAN]->(p:Plan)-[i:INCLUDES]->(poi:POI)
            WHERE p.day >= date($start) AND p.day <= date($end)
              AND i.spent_from_policy IS NOT NULL
              AND i.spent_from_policy <> '{}'
            OPTIONAL MATCH (poi)-[:IN_DONG]->(d:Dong)<-[:HAS_DONG]-(dist:District)
            RETURN a.id AS aid,
                   toString(p.day) AS day,
                   i.category AS category,
                   i.sub_category AS sub,
                   i.actual_spent AS actual_spent,
                   i.spent_from_policy AS spent_json,
                   poi.id AS poi_id, poi.name AS poi_name,
                   d.name AS dong, dist.name AS gu
            """,
            start=start.isoformat(), end=end.isoformat(),
        ).data()

    by_policy_total: dict[str, int] = defaultdict(int)
    by_policy_cat: dict[tuple[str, str], int] = defaultdict(int)
    by_policy_gu: dict[tuple[str, str], int] = defaultdict(int)
    by_policy_day: dict[tuple[str, str], int] = defaultdict(int)
    spend_ratios: list[float] = []
    n_tx_with_policy = 0
    total_actual = 0

    for r in rows:
        try:
            ps = json.loads(r["spent_json"] or "{}")
        except Exception:
            continue
        if not ps:
            continue
        actual = r.get("actual_spent") or 0
        total_actual += int(actual)
        sum_policy = sum(int(v) for v in ps.values() if isinstance(v, (int, float)))
        if sum_policy > 0 and actual > 0:
            spend_ratios.append(sum_policy / actual)
            n_tx_with_policy += 1
        for pid, amt in ps.items():
            v = int(amt) if isinstance(amt, (int, float)) else 0
            if v <= 0:
                continue
            by_policy_total[pid] += v
            cat = r.get("category") or "기타"
            by_policy_cat[(pid, cat)] += v
            gu = r.get("gu") or "?"
            by_policy_gu[(pid, gu)] += v
            by_policy_day[(pid, r["day"])] += v

    return {
        "total_actual_spent_with_policy": total_actual,
        "n_transactions_with_policy": n_tx_with_policy,
        "avg_policy_ratio": (sum(spend_ratios) / len(spend_ratios)) if spend_ratios else 0,
        "by_policy_total": dict(by_policy_total),
        "by_policy_cat": dict(by_policy_cat),
        "by_policy_gu": dict(by_policy_gu),
        "by_policy_day": dict(by_policy_day),
    }


def fetch_grant_state(start: date, days: int) -> dict:
    """마지막 날 State에서 grant_received / grant_remaining 집계."""
    end = start + timedelta(days=days - 1)
    with driver_session() as s:
        rows = s.run(
            """
            MATCH (a:Agent)-[:HAS_STATE {day: date($end)}]->(st:State)
            WHERE st.grant_received IS NOT NULL OR st.grant_remaining IS NOT NULL
            RETURN a.id AS aid,
                   a.p_income_level AS income,
                   st.grant_received AS received,
                   st.grant_remaining AS remaining
            """,
            end=end.isoformat(),
        ).data()

    by_policy_received: dict[str, int] = defaultdict(int)
    by_policy_remaining: dict[str, int] = defaultdict(int)
    by_policy_income_received: dict[tuple[str, str], int] = defaultdict(int)
    by_policy_income_remaining: dict[tuple[str, str], int] = defaultdict(int)
    n_recipients_per_policy: dict[str, int] = defaultdict(int)
    n_full_recipients: dict[str, int] = defaultdict(int)  # 잔여 0원 = 다 씀

    for r in rows:
        income = r.get("income") or "?"
        try:
            recv = json.loads(r["received"] or "{}") if r["received"] else {}
        except Exception:
            recv = {}
        try:
            rem = json.loads(r["remaining"] or "{}") if r["remaining"] else {}
        except Exception:
            rem = {}
        for pid, amt in recv.items():
            v = int(amt) if isinstance(amt, (int, float)) else 0
            by_policy_received[pid] += v
            by_policy_income_received[(pid, income)] += v
            if v > 0:
                n_recipients_per_policy[pid] += 1
        for pid, amt in rem.items():
            v = int(amt) if isinstance(amt, (int, float)) else 0
            by_policy_remaining[pid] += v
            by_policy_income_remaining[(pid, income)] += v
            if v == 0 and pid in recv:
                n_full_recipients[pid] += 1

    return {
        "by_policy_received": dict(by_policy_received),
        "by_policy_remaining": dict(by_policy_remaining),
        "by_policy_income_received": dict(by_policy_income_received),
        "by_policy_income_remaining": dict(by_policy_income_remaining),
        "n_recipients_per_policy": dict(n_recipients_per_policy),
        "n_full_recipients": dict(n_full_recipients),
    }


def build_markdown(spend_stats: dict, state_stats: dict, start: date, days: int) -> str:
    L: list[str] = []
    L.append("# 정책 지원금 사용처 분석 (옵션 A)")
    L.append("")
    L.append(f"- 기간: {start.isoformat()} ~ {(start+timedelta(days=days-1)).isoformat()} ({days}일)")
    L.append(f"- 정책 사용 거래 수: **{spend_stats['n_transactions_with_policy']:,}건**")
    L.append(f"- 거래당 평균 정책 비중: **{spend_stats['avg_policy_ratio']*100:.1f}%** (정책 지원금 사용 거래 중)")
    L.append("")

    # 1. 정책별 누적
    L.append("## 1. 정책별 누적 사용액 / 수령액 / 잔여액")
    L.append("")
    L.append("| 정책 | 누적 수령 (a) | 누적 사용 (b) | 잔여 (a−b) | 수혜자 수 | 다 쓴 수혜자 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    all_pids = set(state_stats["by_policy_received"]) | set(spend_stats["by_policy_total"])
    for pid in sorted(all_pids):
        recv = state_stats["by_policy_received"].get(pid, 0)
        used = spend_stats["by_policy_total"].get(pid, 0)
        rem = state_stats["by_policy_remaining"].get(pid, 0)
        n_recip = state_stats["n_recipients_per_policy"].get(pid, 0)
        n_full = state_stats["n_full_recipients"].get(pid, 0)
        L.append(f"| {pid} | {recv:,}원 | {used:,}원 | {rem:,}원 | {n_recip:,} | {n_full:,} |")
    L.append("")

    # 2. 정책 × 카테고리
    L.append("## 2. 정책별 카테고리 사용처")
    L.append("")
    for pid in sorted(spend_stats["by_policy_total"]):
        L.append(f"### {pid}")
        cat_amts = [(c, a) for (p, c), a in spend_stats["by_policy_cat"].items() if p == pid]
        cat_amts.sort(key=lambda x: -x[1])
        total = sum(a for _, a in cat_amts) or 1
        L.append("| 카테고리 | 사용액 | 비중 |")
        L.append("|---|---:|---:|")
        for cat, amt in cat_amts[:15]:
            L.append(f"| {cat or '?'} | {amt:,}원 | {amt/total*100:.1f}% |")
        L.append("")

    # 3. 정책 × 자치구
    L.append("## 3. 정책별 자치구 사용처 (Top 10)")
    L.append("")
    for pid in sorted(spend_stats["by_policy_total"]):
        L.append(f"### {pid}")
        gu_amts = [(g, a) for (p, g), a in spend_stats["by_policy_gu"].items() if p == pid]
        gu_amts.sort(key=lambda x: -x[1])
        total = sum(a for _, a in gu_amts) or 1
        L.append("| 자치구 | 사용액 | 비중 |")
        L.append("|---|---:|---:|")
        for gu, amt in gu_amts[:10]:
            L.append(f"| {gu} | {amt:,}원 | {amt/total*100:.1f}% |")
        L.append("")

    # 4. 일별 추이
    L.append("## 4. 정책별 일별 사용액 추이")
    L.append("")
    days_set = sorted({d for _, d in spend_stats["by_policy_day"].keys()})
    pids_set = sorted({p for p, _ in spend_stats["by_policy_day"].keys()})
    L.append("| 일자 | " + " | ".join(pids_set) + " |")
    L.append("|---|" + "---:|" * len(pids_set))
    for d in days_set:
        row = [d]
        for pid in pids_set:
            amt = spend_stats["by_policy_day"].get((pid, d), 0)
            row.append(f"{amt:,}원")
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # 5. 소득별 분포
    L.append("## 5. 정책 × 소득별 수령액 vs 잔여액")
    L.append("")
    for pid in sorted(state_stats["by_policy_received"]):
        L.append(f"### {pid}")
        L.append("| 소득 | 누적 수령 | 누적 잔여 | 소진율 |")
        L.append("|---|---:|---:|---:|")
        for income in ["상", "중상", "중", "중하", "하"]:
            recv = state_stats["by_policy_income_received"].get((pid, income), 0)
            rem = state_stats["by_policy_income_remaining"].get((pid, income), 0)
            ratio = (recv - rem) / recv * 100 if recv else 0
            L.append(f"| {income} | {recv:,}원 | {rem:,}원 | {ratio:.1f}% |")
        L.append("")

    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-25")
    ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--out", default="output/sim/report/POLICY_SPEND_ANALYSIS.md")
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    print(f"[analyze_policy_spend] {args.start} + {args.days}d")

    spend = fetch_policy_spend(start, args.days)
    print(f"  거래 {spend['n_transactions_with_policy']:,}건 / 정책 {len(spend['by_policy_total'])}개")

    state = fetch_grant_state(start, args.days)
    print(f"  State 수혜자 {sum(state['n_recipients_per_policy'].values()):,}명")

    md = build_markdown(spend, state, start, args.days)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"  → {out}")


if __name__ == "__main__":
    main()
