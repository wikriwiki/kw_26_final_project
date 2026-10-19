"""Read-only, post-score display audit for P012-5's small sector denominator.

This file is not part of the frozen scoring rule and never changes numeric.json.
It binds the visible won totals to the exact SHA-verified sector ledgers and score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


APPLIANCE_SUBCLASSES = ("가구", "가전·통신", "통신")
BEAUTY_L1 = "미용"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ledger(path: Path) -> tuple[list[dict], set[tuple[str, str]]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    keys = {(row["aid"], row["day"]) for row in rows}
    if len(keys) != len(rows) or any(not isinstance(row.get("by_sub"), dict)
                                     or not isinstance(row.get("by_l1"), dict)
                                     for row in rows):
        raise ValueError(f"bad or duplicate citizen-day rows: {path}")
    return rows, keys


def total(rows: list[dict], field: str, names: tuple[str, ...]) -> int:
    values = [sum(row[field].get(name, 0) for name in names) for row in rows]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in values):
        raise ValueError(f"invalid {field} won")
    return sum(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--numeric", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    score = json.loads(args.numeric.read_text(encoding="utf-8"))
    run = score["runs"][0]
    if run["policy"] != "P012" or run["citizens"] != 12 or run["on"] != "2021-10-01:2021-10-31":
        raise ValueError("not the frozen P012 October score")
    evidence = {entry["path"].replace("\\", "/"): entry["sha256"]
                for entry in run["evidence"]}
    for path in (args.on, args.off):
        expected = evidence.get(path.as_posix())
        if expected is None or sha256(path) != expected:
            raise ValueError(f"ledger is not the score's SHA-frozen evidence: {path}")
    on_rows, on_keys = ledger(args.on)
    off_rows, off_keys = ledger(args.off)
    if on_keys != off_keys or len(on_keys) != 12 * 31:
        raise ValueError("ON/OFF citizen-day matrix is not complete and paired")

    appliance_on = total(on_rows, "by_sub", APPLIANCE_SUBCLASSES)
    appliance_off = total(off_rows, "by_sub", APPLIANCE_SUBCLASSES)
    beauty_on = total(on_rows, "by_l1", (BEAUTY_L1,))
    beauty_off = total(off_rows, "by_l1", (BEAUTY_L1,))
    if appliance_off <= 0 or beauty_off <= 0:
        raise ValueError("zero sector denominator")
    appliance_pct = 100 * (appliance_on - appliance_off) / appliance_off
    beauty_pct = 100 * (beauty_on - beauty_off) / beauty_off
    indicator = next(item for item in run["indicators"] if item["id"] == "P012-5")
    components = indicator["simulation_components"]
    if (not math.isclose(appliance_pct, components["appliance_furniture_pct"], abs_tol=1e-9)
            or not math.isclose(beauty_pct, components["hair_beauty_pct"], abs_tol=1e-9)
            or not math.isclose(appliance_pct - beauty_pct, indicator["simulation"], abs_tol=1e-9)):
        raise ValueError("sector totals do not reproduce frozen P012-5 score")

    payload = {
        "schema": "p012_sector_denominator_display_audit_v1",
        "purpose": "Post-score display-only denominator disclosure; no score or preregistration changes",
        "policy": "P012",
        "indicator": "P012-5",
        "effect_window": run["on"],
        "citizens": 12,
        "paired_citizen_days": len(on_keys),
        "numeric_path": args.numeric.as_posix(),
        "numeric_sha256": sha256(args.numeric),
        "on_sector_ledger_path": args.on.as_posix(),
        "on_sector_ledger_sha256": sha256(args.on),
        "off_sector_ledger_path": args.off.as_posix(),
        "off_sector_ledger_sha256": sha256(args.off),
        "appliance_furniture": {
            "source_field": "by_sub",
            "subclasses": list(APPLIANCE_SUBCLASSES),
            "on_won": appliance_on,
            "off_won": appliance_off,
            "on_off_percent": appliance_pct,
        },
        "hair_beauty": {
            "source_field": "by_l1",
            "category": BEAUTY_L1,
            "on_won": beauty_on,
            "off_won": beauty_off,
            "on_off_percent": beauty_pct,
        },
        "gap_percentage_points": appliance_pct - beauty_pct,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.out)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{args.out}: {appliance_off} KRW appliance/furniture OFF denominator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
