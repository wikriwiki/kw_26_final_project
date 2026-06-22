"""Run threshold_sweep across a date range and summarize totals.

This still does not call the intent LLM. It queries Neo4j day by day and reuses
the same scoring logic as the real Night interaction selection.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.night_threshold_sweep import score_all_pairs, sweep_thresholds
from scripts.sim import night_interaction
from scripts.sim.night_interaction import DEFAULT_TEMPERATURE, MAX_PAIRS_PER_AGENT


def _dates(start: date, end: date) -> list[date]:
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD, inclusive")
    ap.add_argument("--thresholds", default="0.30,0.32,0.34,0.36,0.38,0.40")
    ap.add_argument("--max-pairs", type=int, default=MAX_PAIRS_PER_AGENT)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-agents", type=int, default=None,
                    help="Randomly sample this many outed agents per day before scoring")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    per_day = []
    totals = {
        th: {"total_candidates": 0, "above_threshold": 0, "below_threshold": 0, "selected_pairs": 0}
        for th in thresholds
    }

    for d in _dates(start, end):
        print(f"\n=== {d.isoformat()} ===")
        scored = score_all_pairs(
            d,
            (night_interaction.W_EXPOSURE,
             night_interaction.W_RELATION,
             night_interaction.W_URGENCY),
            sample_agents=args.sample_agents,
            seed=args.seed,
        )
        rows = sweep_thresholds(scored, thresholds, args.max_pairs, args.temperature, args.seed)
        per_day.append({"day": d.isoformat(), "raw_candidates": len(scored), "rows": rows})
        print(f"raw candidate pairs: {len(scored):,}")
        for r in rows:
            totals[r["threshold"]]["above_threshold"] += r["above_threshold"]
            totals[r["threshold"]]["below_threshold"] += r["below_threshold"]
            totals[r["threshold"]]["total_candidates"] += r["total_candidates"]
            totals[r["threshold"]]["selected_pairs"] += r["selected_pairs"]
            print(
                f"  th={r['threshold']:.2f} "
                f"above={r['above_threshold']:,} ({r['above_rate']*100:.1f}%) "
                f"selected={r['selected_pairs']:,} "
                f"selected/above={r['selected_rate_of_above']*100:.1f}% "
                f"score_p50={r['score']['p50']} "
                f"rel_p50={r['relationship']['p50']}"
            )

    n_days = len(per_day)
    print("\n=== Range summary ===")
    print(f"days: {n_days}, start={start.isoformat()}, end={end.isoformat()}")
    base_selected = totals[thresholds[0]]["selected_pairs"] if thresholds else 0
    print("threshold | above% | below% | total_selected | avg/day | selected change | selected/above")
    for th in thresholds:
        total_sel = totals[th]["selected_pairs"]
        total_above = totals[th]["above_threshold"]
        total_cands = totals[th]["total_candidates"]
        above_rate = (total_above / total_cands) if total_cands else 0.0
        below_rate = (totals[th]["below_threshold"] / total_cands) if total_cands else 0.0
        selected_above = (total_sel / total_above) if total_above else 0.0
        change = ((total_sel - base_selected) / base_selected) if base_selected else 0.0
        print(
            f"{th:>9.2f} | "
            f"{above_rate*100:>6.1f}% | "
            f"{below_rate*100:>6.1f}% | "
            f"{total_sel:>14,} | "
            f"{total_sel / n_days:>7,.1f} | "
            f"{change*100:>+14.1f}% | "
            f"{selected_above*100:>13.1f}%"
        )

    if args.out:
        out = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "thresholds": thresholds,
            "totals": totals,
            "per_day": per_day,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
