"""Sweep Night interaction thresholds without calling the intent LLM.

This reuses the same fetch/scoring/matching logic as night_interaction.py, then
prints how many candidate pairs survive each threshold. It is meant for tuning
THRESHOLD before spending time on LLM intent classification.
"""
from __future__ import annotations

import argparse
import json
import statistics
import random
from collections import defaultdict
from datetime import date
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sim.night_interaction import (
    DEFAULT_TEMPERATURE,
    MAX_PAIRS_PER_AGENT,
    W_EXPOSURE,
    W_RELATION,
    W_URGENCY,
    _softmax_select,
    calc_exposure,
    calc_relationship,
    calc_urgency,
    fetch_all,
    find_candidate_pairs,
)


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p25": None, "p50": None, "p75": None, "p90": None, "max": None, "avg": None}
    xs = sorted(values)

    def pct(p: float) -> float:
        idx = min(len(xs) - 1, max(0, round((len(xs) - 1) * p)))
        return round(xs[idx], 4)

    return {
        "min": round(xs[0], 4),
        "p25": pct(0.25),
        "p50": pct(0.50),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "max": round(xs[-1], 4),
        "avg": round(statistics.fmean(xs), 4),
    }


def score_all_pairs(
    day: date,
    weights: tuple[float, float, float],
    sample_agents: int | None = None,
    seed: int = 42,
) -> list[dict]:
    data = fetch_all(day)
    if sample_agents and sample_agents > 0 and len(data["outed"]) > sample_agents:
        rng = random.Random(seed)
        keep = set(rng.sample(sorted(data["outed"]), sample_agents))
        data["outed"] = set(keep)
        data["visits"] = defaultdict(list, {
            aid: visits for aid, visits in data["visits"].items()
            if aid in keep
        })
        data["state"] = {aid: st for aid, st in data["state"].items() if aid in keep}
        data["info_count"] = {aid: n for aid, n in data["info_count"].items() if aid in keep}
        data["knows"] = {
            pair: rel for pair, rel in data["knows"].items()
            if pair[0] in keep and pair[1] in keep
        }
        data["conv_history"] = {
            pair: hist for pair, hist in data["conv_history"].items()
            if pair[0] in keep and pair[1] in keep
        }
        print(f"  [sample] agents: {sample_agents:,} / outed sampled")
    cands = find_candidate_pairs(data)
    w_e, w_r, w_u = weights
    scored = []
    for a, b in cands:
        exp = calc_exposure(a, b, data)
        rel = calc_relationship(a, b, data, current_day=day)
        urg = calc_urgency(a, b, data)
        total = w_e * exp + w_r * rel + w_u * urg
        scored.append({
            "aid_a": a,
            "aid_b": b,
            "score": round(total, 4),
            "exposure": round(exp, 4),
            "relationship": round(rel, 4),
            "urgency": round(urg, 4),
        })
    return scored


def sweep_thresholds(
    scored: list[dict],
    thresholds: list[float],
    max_pairs_per_agent: int,
    temperature: float,
    seed: int | None,
) -> list[dict]:
    rows = []
    total_candidates = len(scored)
    for th in thresholds:
        above = [r for r in scored if r["score"] >= th]
        selected = _softmax_select(above, max_pairs_per_agent, temperature, __import__("random").Random(seed))
        rows.append({
            "threshold": th,
            "total_candidates": total_candidates,
            "above_threshold": len(above),
            "below_threshold": total_candidates - len(above),
            "above_rate": round(len(above) / total_candidates, 4) if total_candidates else 0.0,
            "below_rate": round((total_candidates - len(above)) / total_candidates, 4) if total_candidates else 0.0,
            "selected_pairs": len(selected),
            "selected_rate_of_candidates": round(len(selected) / total_candidates, 4) if total_candidates else 0.0,
            "selected_rate_of_above": round(len(selected) / len(above), 4) if above else 0.0,
            "score": _quantiles([r["score"] for r in above]),
            "exposure": _quantiles([r["exposure"] for r in above]),
            "relationship": _quantiles([r["relationship"] for r in above]),
            "urgency": _quantiles([r["urgency"] for r in above]),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True, help="YYYY-MM-DD")
    ap.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50")
    ap.add_argument("--max-pairs", type=int, default=MAX_PAIRS_PER_AGENT)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-agents", type=int, default=None,
                    help="Randomly sample this many outed agents before candidate-pair scoring")
    ap.add_argument("--out", default=None, help="Optional JSON output path")
    args = ap.parse_args()

    day = date.fromisoformat(args.day)
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    scored = score_all_pairs(day, (W_EXPOSURE, W_RELATION, W_URGENCY), args.sample_agents, args.seed)
    rows = sweep_thresholds(scored, thresholds, args.max_pairs, args.temperature, args.seed)

    print(f"\nNight threshold sweep: {day.isoformat()}")
    print(f"raw candidate pairs: {len(scored):,}")
    print(f"max_pairs_per_agent={args.max_pairs}, temperature={args.temperature}, seed={args.seed}")
    base_selected = rows[0]["selected_pairs"] if rows else 0
    print("\nthreshold | above% | below% | selected | selected/above | selected change | score p50/p75/p90 | rel p50/p75")
    for r in rows:
        q = r["score"]
        rq = r["relationship"]
        change = 0.0 if not base_selected else (r["selected_pairs"] - base_selected) / base_selected
        print(
            f"{r['threshold']:>9.2f} | "
            f"{r['above_rate']*100:>6.1f}% | "
            f"{r['below_rate']*100:>6.1f}% | "
            f"{r['selected_pairs']:>8,} | "
            f"{r['selected_rate_of_above']*100:>13.1f}% | "
            f"{change*100:>+14.1f}% | "
            f"{q['p50']}/{q['p75']}/{q['p90']} | "
            f"{rq['p50']}/{rq['p75']}"
        )

    # Quick weak-tie view for deciding whether a threshold mostly removes low-signal overlaps.
    buckets = defaultdict(int)
    for r in scored:
        if r["relationship"] == 0 and r["urgency"] == 0:
            buckets["rel0_urg0"] += 1
        elif r["relationship"] == 0:
            buckets["rel0"] += 1
        elif r["relationship"] < 0.2:
            buckets["rel_lt_0.2"] += 1
        else:
            buckets["rel_ge_0.2"] += 1
    print("\nraw candidate relationship buckets:")
    for k in ["rel0_urg0", "rel0", "rel_lt_0.2", "rel_ge_0.2"]:
        print(f"- {k}: {buckets[k]:,}")

    if args.out:
        out = {"day": day.isoformat(), "raw_candidates": len(scored), "rows": rows}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
