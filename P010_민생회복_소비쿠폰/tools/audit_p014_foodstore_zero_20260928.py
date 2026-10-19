"""Post-score display audit for P014's undefined food-store growth proxy.

This never changes the frozen scorer. It binds the reported 0/0 denominator
to the exact SHA-verified paired sector ledgers and numeric result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


SUB = "식료품"
COMPARATOR = "슈퍼마켓"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def totals(path: Path, days: set[str]) -> tuple[dict[str, int], set[tuple[str, str]]]:
    result = defaultdict(int)
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["day"] not in days:
            continue
        key = (row["aid"], row["day"])
        if key in keys:
            raise ValueError(f"duplicate citizen-day: {key}")
        keys.add(key)
        bucket = row["by_sub"]
        for name in (SUB, COMPARATOR):
            amount = bucket.get(name, 0)
            if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
                raise ValueError(f"invalid {name} amount: {key}")
            result[name] += amount
            result[f"{name}:positive_citizen_days"] += bool(amount)
    return dict(result), keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--numeric", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    score = json.loads(args.numeric.read_text(encoding="utf-8"))
    run = score["runs"][0]
    if run.get("policy_id") != "P014" or run.get("citizens") != 40:
        raise ValueError("not the paired P014 score")
    evidence = {item["path"].replace("\\", "/"): item["sha256"]
                for item in run["evidence"]}
    for path in (args.on, args.off):
        if evidence.get(path.as_posix()) != sha256(path):
            raise ValueError(f"ledger not bound to frozen numeric evidence: {path}")
    indicator = next(item for item in run["indicators"]
                     if item["id"] == "P014-KIPF-47129")
    if indicator["simulation"] is not None or "OFF denominator is zero" not in indicator["reason"]:
        raise ValueError("frozen 47129 score is not denominator-null")
    effect_days = set()
    first, last = run["on"].split(":")
    from datetime import date, timedelta
    start, end = date.fromisoformat(first), date.fromisoformat(last)
    effect_days = {(start + timedelta(days=i)).isoformat()
                   for i in range((end-start).days+1)}
    if len(effect_days) != 3:
        raise ValueError("P014 effect window must be three days")
    on, on_keys = totals(args.on, effect_days)
    off, off_keys = totals(args.off, effect_days)
    if on_keys != off_keys or len(on_keys) != 40 * 3:
        raise ValueError("incomplete paired effect-period citizen-days")
    if on.get(SUB, 0) != 0 or off.get(SUB, 0) != 0:
        raise ValueError("food-store 0/0 evidence changed")
    payload = {
        "schema": "p014_food_zero_display_audit_v1",
        "purpose": "Post-score display-only disclosure; frozen score and indicator selection unchanged",
        "policy": "P014", "indicator": "P014-KIPF-47129",
        "effect_days": sorted(effect_days), "paired_citizen_days": len(on_keys),
        "numeric_path": args.numeric.as_posix(),
        "numeric_sha256": sha256(args.numeric),
        "subclass": SUB,
        "on": {"sector_ledger_path": args.on.as_posix(),
               "sector_ledger_sha256": sha256(args.on),
               "won": on.get(SUB, 0), "citizen_days": len(on_keys),
               "positive_citizen_days": on.get(f"{SUB}:positive_citizen_days", 0)},
        "off": {"sector_ledger_path": args.off.as_posix(),
                "sector_ledger_sha256": sha256(args.off),
                "won": off.get(SUB, 0), "citizen_days": len(off_keys),
                "positive_citizen_days": off.get(f"{SUB}:positive_citizen_days", 0)},
        "simulation": None,
        "reason": "Neither arm records any spending in this POI subclass over the three-day effect window; 0/0 growth is undefined, not zero response. Catalog and candidate availability are separate questions.",
        "comparator_supermarket_on_won": on.get(COMPARATOR, 0),
        "comparator_supermarket_off_won": off.get(COMPARATOR, 0),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        temporary.replace(args.out)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{args.out}: {SUB} ON/OFF 0/0 verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
