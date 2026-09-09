"""Compare optimized functions to frozen c684f83 algorithms, without DB/LLM.

Run: python scripts/analysis/benchmark_complexity.py --output PATH
Timings are measurements, not timing assertions; output equality is asserted.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.sim import export_visualization as export
from scripts.sim import night_interaction as night
from tests.unit.sim import performance_reference as reference

social = importlib.import_module("scripts.neo4j_load.06_social")
with patch.dict(sys.modules, {"_common": importlib.import_module("scripts.persona._common")}):
    rank = importlib.import_module("scripts.persona.build_rank_coupling")


def measure(fn, repeats):
    timings = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            result = fn()
        timings.append(time.perf_counter() - started)
    return result, statistics.median(timings)


def compare(name, dimensions, original, optimized, repeats):
    expected, before = measure(original, repeats)
    actual, after = measure(optimized, repeats)
    assert expected == actual, name
    row = {"operation": name, **dimensions, "original_seconds": before,
           "optimized_seconds": after, "speedup": before / after, "equal": True}
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--operations", nargs="+", choices=["social", "exposure", "timeline", "rank"],
                        default=["social", "exposure", "timeline", "rank"])
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    rows = []
    for size in (200, 500, 1000):
        if "social" not in args.operations:
            continue
        ids = [f"a{i}" for i in range(size)]
        work = {"w": ids}
        home = {"h": ids[::-1]}
        rows.append(compare("social_graph", {"agents": size},
                            lambda: reference.social_pairs(work, home, random.Random(42)),
                            lambda: social.build_social_pairs(work, home, random.Random(42)), args.repeats))

    for visits_per_agent in (6, 30, 120):
        if "exposure" not in args.operations:
            continue
        rng = random.Random(42)
        ids = [f"a{i}" for i in range(100)]
        data = {"visits": {a: [(f"d{rng.randrange(3)}", rng.randrange(24))
                               for _ in range(visits_per_agent)] for a in ids}}
        pairs = [tuple(rng.sample(ids, 2)) for _ in range(3000)]

        def indexed():
            counts = {a: night._count_visits(v) for a, v in data["visits"].items()}
            return [night.calc_exposure(a, b, data, visit_counts=counts) for a, b in pairs]

        rows.append(compare("night_exposure", {"agents": len(ids), "pairs": len(pairs), "visits_per_agent": visits_per_agent},
                            lambda: [reference.exposure(a, b, data) for a, b in pairs], indexed, args.repeats))

    for n_days in (7, 14, 28):
        if "timeline" not in args.operations:
            continue
        days = [(date(2021, 9, 1) + timedelta(days=i)).isoformat() for i in range(n_days)]
        agents = {f"a{i}": [{"day": day, "ord": order, "time": f"{hour:02}:00", "lon": 126.9,
                             "lat": 37.5, "cat": "식사", "intent": str(order), "sat": 0.5,
                             "spent": 1000, "anchor": "residence"}
                            for day in days for order, hour in enumerate((6, 9, 12, 15, 18, 22))]
                  for i in range(100)}
        rows.append(compare("timeline_frames", {"agents": 100, "days": n_days, "events_per_day": 6},
                            lambda: reference.timeline_frames(agents, days),
                            lambda: export.build_timeline_frames(agents, days), args.repeats))

    for size in (1000, 4000, 8000):
        if "rank" not in args.operations:
            continue
        records = [{"uuid": f"u{i}", "sex": "여자", "age": 35, "district": f"gu{i % 4}",
                    "education_level": ("고등학교", "대학원")[i % 2]} for i in range(size)]
        pool_index = rank.index_nvidia_pool(records)
        all_sorted = sorted(records, key=rank.ses_proxy)
        cell = ("absent", "F", "30대")  # common fallback across districts

        def original_rank():
            used, result = set(), []
            for i in range(100):
                record, level = rank.pick_nvidia_by_rank(pool_index, all_sorted, cell, i / 99, used)
                used.add(record["uuid"])
                result.append((record, level))
            return result

        def indexed_rank():
            matcher = rank.RankMatcher(pool_index, all_sorted, rank._AGE_NEIGHBORS, rank.ses_proxy)
            result = []
            for i in range(100):
                record, level = matcher.pick(cell, i / 99)
                matcher.mark_used(record["uuid"])
                result.append((record, level))
            return result

        rows.append(compare("persona_rank", {"records": size, "assignments": 100, "fallback": "sex_age"},
                            original_rank, indexed_rank, args.repeats))

    result = {"measured_at": datetime.now(timezone.utc).isoformat(), "python": sys.version,
              "platform": platform.platform(), "baseline_commit": "c684f83098f7585fc8f202805725202b081fcd79",
              "worktree_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "repeats": args.repeats, "statistic": "median", "includes_index_construction": True,
              "scope": "local synthetic CPU benchmark; not full simulation or GPU speedup", "results": rows}
    result["optimized_source_sha256"] = {
        str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [ROOT / "scripts/neo4j_load/06_social.py", ROOT / "scripts/sim/night_interaction.py",
                     ROOT / "scripts/sim/export_visualization.py", ROOT / "scripts/persona/build_rank_coupling.py",
                     ROOT / "scripts/persona/rank_index.py"]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
