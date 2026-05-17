"""풀런 후 fb_* (fallback) 카운터 집계.

run_simulation.py 가 매 agent×day 마다 metrics jsonl에 기록한 fb_* 필드를
day별로 합계/비율로 묶어낸다. stage2_poi.py 의 다단 fallback이 실제 풀런에서
얼마나 자주 발동했는지 보고용.

사용:
  python scripts/sim/aggregate_fallback_stats.py --start 2026-05-01 --days 3 \
      [--out docs/SIM_QWEN14B_P008_3D_FALLBACK.md]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

METRICS_DIR = Path(os.environ.get("SIM_OUTPUT_DIR",
                                  os.path.expanduser("~/sim_output"))) / "metrics"

FB_FIELDS = [
    "fb_resolve_dong",
    "fb_cand_sub_match",
    "fb_cand_l1_dong",
    "fb_cand_l1_district",
    "fb_cand_all_empty",
    "fb_hallucinations_corrected",
    "fb_hallucinations_dropped",
    "fb_order_mismatch",
    "fb_missing_picks_filled",
]


def load_day(d: str) -> list[dict]:
    p = METRICS_DIR / f"day_{d}.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def aggregate(days: list[str]) -> dict:
    out: dict = {"by_day": {}, "total": {}}
    grand = {k: 0 for k in FB_FIELDS}
    grand_agents = 0
    grand_events = 0  # n_includes sum
    grand_cand_calls = 0  # sub+l1d+l1dist+all_empty
    for d in days:
        rows = load_day(d)
        if not rows:
            out["by_day"][d] = {"empty": True}
            continue
        ok = [r for r in rows if r.get("status") == "ok"]
        sums = {k: sum(r.get(k, 0) for r in ok) for k in FB_FIELDS}
        n_includes = sum(r.get("n_includes", 0) for r in ok)
        cand_calls = (sums["fb_cand_sub_match"]
                      + sums["fb_cand_l1_dong"]
                      + sums["fb_cand_l1_district"]
                      + sums["fb_cand_all_empty"])
        out["by_day"][d] = {
            "agents_ok": len(ok),
            "n_includes_total": n_includes,
            "cand_calls": cand_calls,
            **sums,
            "sub_match_rate": round(sums["fb_cand_sub_match"] / max(cand_calls, 1) * 100, 2),
            "l1_dong_rate":   round(sums["fb_cand_l1_dong"]   / max(cand_calls, 1) * 100, 2),
            "l1_district_rate": round(sums["fb_cand_l1_district"] / max(cand_calls, 1) * 100, 2),
            "all_empty_rate": round(sums["fb_cand_all_empty"] / max(cand_calls, 1) * 100, 2),
            "hallu_corrected_per_1k_inc": round(sums["fb_hallucinations_corrected"] / max(n_includes, 1) * 1000, 3),
            "hallu_dropped_per_1k_inc":   round(sums["fb_hallucinations_dropped"]   / max(n_includes, 1) * 1000, 3),
            "missing_filled_per_1k_inc":  round(sums["fb_missing_picks_filled"]    / max(n_includes, 1) * 1000, 3),
            "resolve_dong_fallback_per_agent": round(sums["fb_resolve_dong"] / max(len(ok), 1) * 100, 2),
        }
        for k in FB_FIELDS:
            grand[k] += sums[k]
        grand_agents += len(ok)
        grand_events += n_includes
        grand_cand_calls += cand_calls
    out["total"] = {
        "agents_ok": grand_agents,
        "n_includes_total": grand_events,
        "cand_calls": grand_cand_calls,
        **grand,
        "sub_match_rate":   round(grand["fb_cand_sub_match"] / max(grand_cand_calls, 1) * 100, 2),
        "l1_dong_rate":     round(grand["fb_cand_l1_dong"]   / max(grand_cand_calls, 1) * 100, 2),
        "l1_district_rate": round(grand["fb_cand_l1_district"] / max(grand_cand_calls, 1) * 100, 2),
        "all_empty_rate":   round(grand["fb_cand_all_empty"] / max(grand_cand_calls, 1) * 100, 2),
        "hallu_corrected_per_1k_inc": round(grand["fb_hallucinations_corrected"] / max(grand_events, 1) * 1000, 3),
        "hallu_dropped_per_1k_inc":   round(grand["fb_hallucinations_dropped"]   / max(grand_events, 1) * 1000, 3),
        "missing_filled_per_1k_inc":  round(grand["fb_missing_picks_filled"]    / max(grand_events, 1) * 1000, 3),
        "resolve_dong_fallback_per_agent": round(grand["fb_resolve_dong"] / max(grand_agents, 1) * 100, 2),
    }
    return out


def to_markdown(agg: dict) -> str:
    lines = ["# Stage 2 Fallback Stats — 풀런 집계", ""]
    lines.append("## Day별")
    lines.append("")
    lines.append("| day | agents_ok | n_inc | cand_calls | sub_match | L1_dong | L1_dist | all_empty | hallu_corr | hallu_drop | miss_fill | resolve_dong_FB |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d, s in agg["by_day"].items():
        if s.get("empty"):
            lines.append(f"| {d} | (empty) |  |  |  |  |  |  |  |  |  |  |")
            continue
        lines.append(
            f"| {d} | {s['agents_ok']:,} | {s['n_includes_total']:,} | {s['cand_calls']:,} "
            f"| {s['fb_cand_sub_match']:,} ({s['sub_match_rate']}%) "
            f"| {s['fb_cand_l1_dong']:,} ({s['l1_dong_rate']}%) "
            f"| {s['fb_cand_l1_district']:,} ({s['l1_district_rate']}%) "
            f"| {s['fb_cand_all_empty']:,} ({s['all_empty_rate']}%) "
            f"| {s['fb_hallucinations_corrected']:,} ({s['hallu_corrected_per_1k_inc']}‰) "
            f"| {s['fb_hallucinations_dropped']:,} ({s['hallu_dropped_per_1k_inc']}‰) "
            f"| {s['fb_missing_picks_filled']:,} ({s['missing_filled_per_1k_inc']}‰) "
            f"| {s['fb_resolve_dong']:,} ({s['resolve_dong_fallback_per_agent']}%) |"
        )
    t = agg["total"]
    lines.append("")
    lines.append("## 전체 (3-day total)")
    lines.append("")
    lines.append(f"- 처리 agent×day OK: **{t['agents_ok']:,}**")
    lines.append(f"- 적재된 INCLUDES 이벤트 수: **{t['n_includes_total']:,}**")
    lines.append(f"- Stage 2 후보 fetch 호출: **{t['cand_calls']:,}** (= 외출 이벤트 수)")
    lines.append("")
    lines.append("### 후보 풀 매칭 단계 (어디서 후보가 채워졌나)")
    lines.append(f"- 1차 sub_cat 매칭: **{t['fb_cand_sub_match']:,}** ({t['sub_match_rate']}%)")
    lines.append(f"- 2차 L1 dong fallback: **{t['fb_cand_l1_dong']:,}** ({t['l1_dong_rate']}%)")
    lines.append(f"- 3차 L1 district fallback: **{t['fb_cand_l1_district']:,}** ({t['l1_district_rate']}%)")
    lines.append(f"- 4차 모두 비어 있음 (드롭): **{t['fb_cand_all_empty']:,}** ({t['all_empty_rate']}%)")
    lines.append("")
    lines.append("### 적재 직전 LLM 산출 보정")
    lines.append(f"- 환각 자동 보정 (random Top-5): **{t['fb_hallucinations_corrected']:,}** ({t['hallu_corrected_per_1k_inc']}‰)")
    lines.append(f"- 환각 드롭 (후보 자체 없음): **{t['fb_hallucinations_dropped']:,}** ({t['hallu_dropped_per_1k_inc']}‰)")
    lines.append(f"- 미선택 이벤트 채움: **{t['fb_missing_picks_filled']:,}** ({t['missing_filled_per_1k_inc']}‰)")
    lines.append("")
    lines.append("### Persona 동코드 placeholder 보정")
    lines.append(f"- resolve_dong placeholder fallback: **{t['fb_resolve_dong']:,}** agent×day ({t['resolve_dong_fallback_per_agent']}%)")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--out", default=None, help="markdown 파일로 저장")
    ap.add_argument("--json", default=None, help="json 으로도 저장")
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    day_strs = [(start + timedelta(days=i)).isoformat() for i in range(args.days)]

    agg = aggregate(day_strs)
    md = to_markdown(agg)
    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"\n[saved md] {args.out}", file=sys.stderr)
    if args.json:
        Path(args.json).write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved json] {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
