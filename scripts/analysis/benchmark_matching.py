"""Measure seeded-equivalent Night matching, including all index construction."""
import argparse
import hashlib
import importlib
import json
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
with patch.dict(sys.modules, {"_common": importlib.import_module("scripts.neo4j_load._common")}):
    from scripts.sim.night_interaction import _softmax_select_scan
from scripts.sim.weighted_matching import indexed_softmax_select


def graph(nodes, degree=8):
    rng = random.Random(615)
    seen, rows = set(), []
    while len(rows) < nodes * degree:
        a, b = sorted(rng.sample(range(nodes), 2))
        if (a, b) not in seen:
            seen.add((a, b))
            rows.append({"aid_a": str(a), "aid_b": str(b), "score": rng.randrange(38, 101) / 100})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    results = []
    for nodes in [256, 512, 1024]:
        rows = graph(nodes)
        timings = {"scan": [], "indexed": []}
        for _ in range(args.repeats):
            old_rng, new_rng = random.Random(11), random.Random(11)
            started = time.perf_counter()
            expected = _softmax_select_scan(rows, 2, 0.5, old_rng)
            timings["scan"].append(time.perf_counter() - started)
            stats = {}
            started = time.perf_counter()
            actual = indexed_softmax_select(rows, 2, 0.5, new_rng, _softmax_select_scan, stats=stats)
            timings["indexed"].append(time.perf_counter() - started)
            assert actual == expected
            assert old_rng.getstate() == new_rng.getstate()
        before, after = statistics.median(timings["scan"]), statistics.median(timings["indexed"])
        result = {"agents": nodes, "candidate_pairs": len(rows), "selected": len(actual),
                  "original_seconds": before, "indexed_seconds": after, "speedup": before / after,
                  "exact_output_and_rng_state": True, "index_stats": stats}
        results.append(result)
        print(json.dumps(result), flush=True)
    report = {"measured_at": datetime.now(timezone.utc).isoformat(), "python": sys.version,
              "platform": platform.platform(), "repeats": args.repeats, "statistic": "median",
              "scope": "local CPU synthetic matching; not GPU/whole simulation speedup",
              "sources_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [ROOT / "scripts/sim/night_interaction.py", ROOT / "scripts/sim/weighted_matching.py"]},
              "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
