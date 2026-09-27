"""Post-score, read-only audit of P016 C2/C3's structural zero.

The frozen score is unchanged.  This display-only audit shows whether the
target POI subclasses exhaust the recorded mart L1 bucket in both arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


TARGET_P016 = ("청과", "정육", "슈퍼마켓", "식료품")
MART_L1 = "마트"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> dict[tuple[str, str], tuple[int, int]]:
    indexed = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["aid"], row["day"])
            if key in indexed:
                raise ValueError(f"duplicate citizen-day: {key}")
            sub, l1 = row["by_sub"], row["by_l1"]
            target = sum(sub.get(name, 0) for name in TARGET_P016)
            mart = l1.get(MART_L1, 0)
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                   for value in (target, mart)):
                raise ValueError(f"invalid spending at {key}")
            indexed[key] = target, mart
    return indexed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--numeric", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    score = json.loads(args.numeric.read_text(encoding="utf-8"))
    run = score["runs"][0]
    if run["policy"] != "P016" or run["citizens"] != 40:
        raise ValueError("not the paired P016 score")
    evidence = {item["path"].replace("\\", "/"): item["sha256"]
                for item in run["evidence"]}
    for path in (args.on, args.off):
        if evidence.get(path.as_posix()) != sha256(path):
            raise ValueError(f"sector ledger not bound to frozen score: {path}")

    on = read_rows(args.on)
    off = read_rows(args.off)
    if on.keys() != off.keys() or len(on) != 200:
        raise ValueError("expected 40 paired citizens across five days")
    start, end = run["on"].split(":")
    post = {key for key in on if start <= key[1] <= end}
    if len(post) != 40 * 3:
        raise ValueError("effect period is not three complete paired days")
    indicators = {indicator["id"]: indicator for indicator in run["indicators"]}
    if any(indicators[name]["simulation"] != 0 for name in ("C2", "C3")):
        raise ValueError("frozen C2/C3 scores are not both zero")
    if indicators["C2"]["simulation_components"]["target_poi_pct"] != indicators["C2"]["simulation_components"]["all_mart_poi_pct"]:
        raise ValueError("frozen C2 components are not identical")

    arms = {}
    for label, rows, path in (("on", on, args.on), ("off", off, args.off)):
        mismatches = [key for key, (target, mart) in rows.items() if target != mart]
        if mismatches:
            raise ValueError(f"{label} has {len(mismatches)} non-identical rows")
        arms[label] = {
            "sector_ledger_path": path.as_posix(),
            "sector_ledger_sha256": sha256(path),
            "citizen_days_all": len(rows),
            "identity_count_all": len(rows),
            "citizen_days_effect": len(post),
            "identity_count_effect": len(post),
            "effect_target_poi_won": sum(rows[key][0] for key in post),
            "effect_mart_l1_won": sum(rows[key][1] for key in post),
        }
    payload = {
        "schema": "p016_taxonomy_identity_display_audit_v1",
        "purpose": "Post-score display-only taxonomy audit; frozen score and preregistration unchanged",
        "policy": "P016",
        "indicators": ["C2", "C3"],
        "effect_window": run["on"],
        "numeric_path": args.numeric.as_posix(),
        "numeric_sha256": sha256(args.numeric),
        "target_poi_subclasses": list(TARGET_P016),
        "mart_l1_category": MART_L1,
        "on": arms["on"],
        "off": arms["off"],
        "structural_identity_all_rows": True,
        "score_c2_percentage_points": indicators["C2"]["simulation"],
        "score_c3_percentage_points": indicators["C3"]["simulation"],
        "interpretation": "Recorded target POI spending equals the mart L1 bucket for every citizen-day. C2/C3 zero is a taxonomy identity, not evidence of no policy response or agreement with within-mart product benchmarks.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        temporary.replace(args.out)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{args.out}: structural identity in {len(on) + len(off)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
