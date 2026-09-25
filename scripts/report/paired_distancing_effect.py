"""Score matched-date distancing arms without claiming a year-over-year match."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

from export_distancing_daily_ledger import ENVIRONMENTS, MONEY_FIELDS
from paired_grant_effect import dates, read_jsonl, roster_file

SECTORS = ("restaurant_won", "korean_restaurant_won", "retail_won", "cafe_won")


def _index(rows: list[dict], *, roster: list[str], days: list[str], arm: str) -> dict:
    expected = {(aid, day) for aid in roster for day in days}
    found = {}
    for row in rows:
        if row.get("arm") != arm:
            raise ValueError(f"wrong arm in {arm} ledger")
        key = (row.get("aid"), row.get("day"))
        if key in found:
            raise ValueError(f"duplicate citizen-day in {arm}: {key}")
        found[key] = row
    if set(found) != expected:
        raise ValueError(f"incomplete citizen-day matrix in {arm}")
    for row in found.values():
        for key in MONEY_FIELDS:
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid {key}: {row.get('aid')} {row.get('day')}")
        if (row["classified_by_code_won"] + row["classified_by_category_won"]
                + row["unclassified_won"] != row["offline_spent"]):
            raise ValueError("classification coverage does not reconcile")
        if (row["korean_restaurant_won"] > row["restaurant_won"]
                or row["restaurant_won"] + row["retail_won"] + row["cafe_won"]
                > row["offline_spent"] - row["unclassified_won"]):
            raise ValueError("sector spending exceeds classified offline spending")
    return found


def _interval(values: list[float]) -> list[float] | None:
    if not values:
        return None
    values.sort()
    return [values[int(.025 * (len(values) - 1))],
            values[int(.975 * (len(values) - 1))]]


def score(restricted_rows: list[dict], control_rows: list[dict], *, roster: list[str],
          days: list[str], draws: int = 2000, seed: int = 20260926) -> dict:
    if not roster or len(set(roster)) != len(roster) or not days or draws < 0:
        raise ValueError("nonempty unique roster, dates and nonnegative draws required")
    on = _index(restricted_rows, roster=roster, days=days, arm="restricted")
    off = _index(control_rows, roster=roster, days=days, arm="control")
    citizens = []
    for aid in roster:
        sums = {arm: {key: 0 for key in (*SECTORS, "offline_spent", "online_spent",
                                         "unclassified_won")}
                for arm in ("restricted", "control")}
        previous = {arm: ("", 0) for arm in sums}
        for day in days:
            for arm, rows in (("restricted", on), ("control", off)):
                row = rows[(aid, day)]
                old_month, old_value = previous[arm]
                prior = old_value if old_month == day[:7] else 0
                if row["self_month_cumulative"] - prior != (row["offline_spent"]
                                                              + row["online_spent"]):
                    raise ValueError(f"monthly State ledger mismatch: {aid} {day} {arm}")
                previous[arm] = (day[:7], row["self_month_cumulative"])
                for key in sums[arm]:
                    sums[arm][key] += row[key]
        citizens.append(sums)

    def measure(sample: list[dict], key: str) -> dict:
        treated = sum(row["restricted"][key] for row in sample)
        control = sum(row["control"][key] for row in sample)
        unknown_on = sum(row["restricted"]["unclassified_won"] for row in sample)
        unknown_off = sum(row["control"]["unclassified_won"] for row in sample)
        difference = treated - control
        lower = difference - unknown_off
        upper = difference + unknown_on
        robust = "positive" if lower > 0 else "negative" if upper < 0 else "uncertain"
        return {
            "restricted_won": treated, "control_won": control,
            "difference_per_citizen_won": difference / len(sample),
            "relative_change": difference / control if control else None,
            "unknown_allocation_difference_bounds_per_citizen_won":
                [lower / len(sample), upper / len(sample)],
            "direction_robust_to_unclassified": robust,
        }

    results = {key: measure(citizens, key) for key in SECTORS}
    rng = random.Random(seed)
    boot = {key: [] for key in SECTORS}
    for _ in range(draws):
        sample = rng.choices(citizens, k=len(citizens))
        for key in SECTORS:
            value = measure(sample, key)["relative_change"]
            if value is not None:
                boot[key].append(value)
    for key in SECTORS:
        results[key]["citizen_bootstrap_95_relative_interval"] = _interval(boot[key])
        results[key]["valid_draws"] = len(boot[key])
    offline_on = sum(c["restricted"]["offline_spent"] for c in citizens)
    offline_off = sum(c["control"]["offline_spent"] for c in citizens)
    unknown_on = sum(c["restricted"]["unclassified_won"] for c in citizens)
    unknown_off = sum(c["control"]["unclassified_won"] for c in citizens)
    return {
        "citizens": len(roster), "days": len(days), "complete_paired_matrix": True,
        "sectors": results,
        "classification_coverage": {
            "restricted_unclassified_won": unknown_on,
            "control_unclassified_won": unknown_off,
            "restricted_unclassified_share": unknown_on / offline_on if offline_on else None,
            "control_unclassified_share": unknown_off / offline_off if offline_off else None,
        },
        "comparison": "within-model restriction counterfactual; not historical card-sales YoY",
        "scope": "Same citizens and dates, disease facts held fixed. External restaurant and "
                 "retail figures use other sectors, population, outcome and time contrast; "
                 "do not subtract as direct accuracy errors.",
    }


def verify_manifests(restricted_path: Path, control_path: Path, *, roster: list[str],
                     days: list[str]) -> None:
    expected_roster_sha = hashlib.sha256(json.dumps(sorted(roster), ensure_ascii=False).encode(
        "utf-8")).hexdigest()
    manifests = []
    for arm, path in (("restricted", restricted_path), ("control", control_path)):
        manifest = json.loads(path.with_name(path.name + ".manifest.json").read_text(
            encoding="utf-8"))
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if (manifest.get("arm") != arm or manifest.get("environment_id") != ENVIRONMENTS[arm]
                or manifest.get("start") != days[0] or manifest.get("end") != days[-1]
                or manifest.get("days") != len(days) or manifest.get("citizens") != len(roster)
                or manifest.get("rows") != len(roster) * len(days)
                or manifest.get("roster_sha256") != expected_roster_sha
                or manifest.get("output_sha256") != digest
                or manifest.get("quality_gate_pass") is not True):
            raise ValueError(f"invalid {arm} ledger manifest")
        manifests.append(manifest)
    for key in ("mapping_sha256", "paired_environment_fingerprint",
                "baseline_income_map_sha256"):
        if not manifests[0].get(key) or manifests[0][key] != manifests[1].get(key):
            raise ValueError(f"paired arms differ in {key}")
    full_fingerprints = [item.get("execution_fingerprint") for item in manifests]
    if (any(not isinstance(v, str) or not v for v in full_fingerprints)
            or full_fingerprints[0] == full_fingerprints[1]):
        raise ValueError("environment arms need distinct full execution fingerprints")
    run_ids = [item.get("run_id") for item in manifests]
    if (any(not isinstance(v, str) or not v for v in run_ids)
            or run_ids[0] == run_ids[1]):
        raise ValueError("paired arms need distinct run IDs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restricted", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    roster = roster_file(args.roster)
    days = dates(args.start, args.end)
    verify_manifests(args.restricted, args.control, roster=roster, days=days)
    result = score(read_jsonl(args.restricted), read_jsonl(args.control),
                   roster=roster, days=days, draws=args.draws)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.json_out.with_name(args.json_out.name + f".tmp.{os.getpid()}")
    try:
        partial.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
        partial.replace(args.json_out)
    finally:
        partial.unlink(missing_ok=True)
    print(args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
