"""Score paired cashback ON/OFF ledgers for a complete calendar month.

The citizen-level payout and cap share are approximate population references.
The ON/OFF spending contrast is an internal model effect, not the external
household triple-difference estimator. No answer-key value enters this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

from export_cashback_month import month_days
from paired_grant_effect import (CLEAN_CHOICES, VALID_CHOICES, pair_provenance,
                                 read_jsonl, roster_file)


def _index(rows: list[dict], *, roster: list[str], days: list[str], arm: str,
           policy_id: str, month: str) -> dict[tuple[str, str], dict]:
    expected = {(aid, day) for aid in roster for day in days}
    found = {}
    for row in rows:
        if row.get("arm") != arm or row.get("policy_id") != policy_id or row.get("month") != month:
            raise ValueError(f"wrong arm, policy or month in {arm} ledger")
        key = (row.get("aid"), row.get("day"))
        if key in found:
            raise ValueError(f"duplicate citizen-day in {arm}: {key}")
        found[key] = row
    if set(found) != expected:
        raise ValueError(f"incomplete citizen-day matrix in {arm}: "
                         f"missing={len(expected - set(found))}, extra={len(set(found) - expected)}")
    for row in found.values():
        choice = row.get("s2_choice_status")
        if not isinstance(choice, str) or choice not in VALID_CHOICES:
            raise ValueError("missing or invalid Stage2 choice provenance")
        for field in ("offline_spent", "online_spent", "eligible_spent",
                      "eligible_cumulative", "self_month_cumulative",
                      "anchor_won", "threshold_won"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid {field}: {row.get('aid')} {row.get('day')}")
        payout = row.get("cashback_accrued_won")
        if not isinstance(payout, (int, float)) or isinstance(payout, bool) or not math.isfinite(payout) or payout < 0:
            raise ValueError(f"invalid cashback accrual: {row.get('aid')} {row.get('day')}")
        if not isinstance(row.get("cashback_cap_reached"), bool):
            raise ValueError(f"invalid cashback cap marker: {row.get('aid')} {row.get('day')}")
        if row["eligible_spent"] > row["offline_spent"]:
            raise ValueError("eligible spending exceeds offline gross")
    return found


def _quantile_interval(values: list[float]) -> list[float] | None:
    if not values:
        return None
    values.sort()
    return [values[int(0.025 * (len(values) - 1))],
            values[int(0.975 * (len(values) - 1))]]


def score(on_rows: list[dict], off_rows: list[dict], *, roster: list[str],
          month: str, policy_id: str, draws: int = 2000,
          seed: int = 20260926) -> dict:
    if not roster or len(roster) != len(set(roster)) or draws < 0:
        raise ValueError("nonempty unique roster and nonnegative draws required")
    days = month_days(month)
    on = _index(on_rows, roster=roster, days=days, arm="on",
                policy_id=policy_id, month=month)
    off = _index(off_rows, roster=roster, days=days, arm="off",
                 policy_id=policy_id, month=month)
    citizens = []
    for aid in roster:
        on_eligible = off_eligible = on_total = off_total = 0
        choice_repaired = False
        previous = {"on": (0, 0), "off": (0, 0)}
        anchor = threshold = None
        for day in days:
            p, c = on[(aid, day)], off[(aid, day)]
            choice_repaired |= (p["s2_choice_status"] not in CLEAN_CHOICES or
                                c["s2_choice_status"] not in CLEAN_CHOICES)
            if (p["anchor_won"], p["threshold_won"]) != (c["anchor_won"], c["threshold_won"]):
                raise ValueError(f"baseline or threshold changed between arms: {aid}")
            if anchor is None:
                anchor, threshold = p["anchor_won"], p["threshold_won"]
            elif (anchor, threshold) != (p["anchor_won"], p["threshold_won"]):
                raise ValueError(f"baseline or threshold changed within month: {aid}")
            for arm, row in (("on", p), ("off", c)):
                prev_eligible, prev_self = previous[arm]
                if row["eligible_cumulative"] - prev_eligible != row["eligible_spent"]:
                    raise ValueError(f"eligible cumulative ledger mismatch: {aid} {day} {arm}")
                if row["self_month_cumulative"] - prev_self != row["offline_spent"] + row["online_spent"]:
                    raise ValueError(f"self monthly ledger mismatch: {aid} {day} {arm}")
                previous[arm] = (row["eligible_cumulative"], row["self_month_cumulative"])
            if day != days[-1] and (p["cashback_accrued_won"] or p["cashback_cap_reached"]):
                raise ValueError("cashback was recorded before month end")
            if c["cashback_accrued_won"] or c["cashback_cap_reached"]:
                raise ValueError("cashback leaked into control")
            on_eligible += p["eligible_spent"]
            off_eligible += c["eligible_spent"]
            on_total += p["offline_spent"] + p["online_spent"]
            off_total += c["offline_spent"] + c["online_spent"]
        final = on[(aid, days[-1])]
        if final["cashback_cap_reached"] and final["cashback_accrued_won"] <= 0:
            raise ValueError(f"cap marker without payout: {aid}")
        citizens.append({"aid": aid, "on_eligible": on_eligible,
                         "off_eligible": off_eligible, "on_total": on_total,
                         "off_total": off_total,
                         "cashback": final["cashback_accrued_won"],
                         "capped": final["cashback_cap_reached"],
                         "threshold_reached": on_eligible >= threshold,
                         "choice_repaired": choice_repaired})

    def measures(sample: list[dict]) -> dict[str, float | None]:
        recipients = [r for r in sample if r["cashback"] > 0]
        eligible_off = sum(r["off_eligible"] for r in sample)
        return {
            "eligible_difference_per_citizen_won": sum(
                r["on_eligible"] - r["off_eligible"] for r in sample) / len(sample),
            "eligible_relative_change": (sum(r["on_eligible"] - r["off_eligible"]
                                             for r in sample) / eligible_off
                                         if eligible_off else None),
            "recorded_total_difference_per_citizen_won": sum(
                r["on_total"] - r["off_total"] for r in sample) / len(sample),
            "cashback_per_recipient_won": (sum(r["cashback"] for r in recipients) / len(recipients)
                                            if recipients else None),
            "cap_share_recipients": (sum(r["capped"] for r in recipients) / len(recipients)
                                     if recipients else None),
            "threshold_reach_share": sum(r["threshold_reached"] for r in sample) / len(sample),
        }

    values = measures(citizens)
    clean_citizens = [row for row in citizens if not row["choice_repaired"]]
    boot = {key: [] for key in values}
    rng = random.Random(seed)
    for _ in range(draws):
        sample = rng.choices(citizens, k=len(citizens))
        for key, value in measures(sample).items():
            if value is not None:
                boot[key].append(value)
    return {
        "policy_id": policy_id, "month": month, "citizens": len(roster),
        "days": len(days), "complete_paired_matrix": True,
        "recipients": sum(r["cashback"] > 0 for r in citizens),
        "total_cashback_accrued_won": sum(r["cashback"] for r in citizens),
        "capped_recipients": sum(r["cashback"] > 0 and r["capped"] for r in citizens),
        "metrics": {key: {"value": value,
                          "citizen_bootstrap_95_interval": _quantile_interval(boot[key]),
                          "valid_draws": len(boot[key])}
                    for key, value in values.items()},
        "choice_repair_sensitivity": {
            "unrepaired_citizens": len(clean_citizens),
            "excluded_citizens": len(roster) - len(clean_citizens),
            "metrics": measures(clean_citizens) if clean_citizens else None,
            "scope": "Diagnostic complete-citizen restriction: excludes any citizen "
                     "with a repaired Stage2 place choice in either arm. Not a "
                     "population effect or prompt accuracy score.",
        },
        "comparison": "approximate_population_reference_for_payout; indirect_on_off_for_spending",
        "scope": "Matched synthetic citizens and complete month. External household triple-difference "
                 "and nationwide recipient population are not reproduced; do not subtract those "
                 "estimands as a direct accuracy error.",
    }


def compare_reference(result: dict, reference: dict) -> dict:
    """Describe scale against a recipient-level external table, without an effect score."""
    if (reference.get("policy_id") != result["policy_id"]
            or reference.get("month") != result["month"]):
        raise ValueError("external reference policy or month differs from simulated month")
    recipients = reference.get("recipient_count")
    cashback_total = reference.get("cashback_total_won")
    capped = reference.get("capped_recipient_count")
    if (any(isinstance(v, bool) or not isinstance(v, int) for v in
            (recipients, cashback_total, capped))
            or recipients <= 0 or cashback_total <= 0 or not 0 <= capped <= recipients):
        raise ValueError("invalid external recipient counts or cashback total")
    external_mean = cashback_total / recipients
    external_cap_share = capped / recipients
    simulated_mean = result["metrics"]["cashback_per_recipient_won"]["value"]
    simulated_cap_share = result["metrics"]["cap_share_recipients"]["value"]
    return {
        "source": reference.get("source"),
        "external_population": reference.get("population"),
        "comparison_status": "approximate_population_reference; no direct effect accuracy score",
        "recipient_average": {
            "external_won": external_mean,
            "simulated_won": simulated_mean,
            "simulated_over_external": (simulated_mean / external_mean
                                        if simulated_mean is not None else None),
        },
        "cap_share": {
            "external": external_cap_share,
            "simulated": simulated_cap_share,
            "percentage_point_difference": (100 * (simulated_cap_share - external_cap_share)
                                            if simulated_cap_share is not None else None),
        },
    }


def verify_manifests(on_path: Path, off_path: Path, roster: list[str], month: str) -> dict:
    expected_roster_sha = hashlib.sha256(
        json.dumps(sorted(roster), ensure_ascii=False).encode("utf-8")).hexdigest()
    manifests = []
    for arm, path in (("on", on_path), ("off", off_path)):
        manifest_path = path.with_name(path.name + ".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if (manifest.get("arm") != arm or manifest.get("month") != month
                or manifest.get("roster_sha256") != expected_roster_sha
                or manifest.get("output_sha256") != digest
                or manifest.get("quality_gate_pass") is not True):
            raise ValueError(f"invalid {arm} ledger manifest")
        manifests.append(manifest)
    for key in ("policy_id", "policy_file_sha256", "base_ratio",
                "execution_fingerprint", "baseline_income_map_sha256", "citizens", "days",
                "prompt_variant", "system_prompt_sha256"):
        if manifests[0].get(key) is None or manifests[0].get(key) != manifests[1].get(key):
            raise ValueError(f"paired arms differ in {key}")
    run_ids = [manifest.get("run_id") for manifest in manifests]
    if any(not isinstance(run_id, str) or not run_id for run_id in run_ids):
        raise ValueError("paired arms have missing run ID")
    if run_ids[0] == run_ids[1]:
        raise ValueError("paired arms must have distinct run IDs")
    return pair_provenance(manifests, ("on", "off"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--reference", type=Path,
                        help="Separate, read-only external recipient table for descriptive scale")
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    roster = roster_file(args.roster)
    provenance = verify_manifests(args.on, args.off, roster, args.month)
    result = score(read_jsonl(args.on), read_jsonl(args.off), roster=roster,
                   month=args.month, policy_id=args.policy_id, draws=args.draws)
    result["provenance"] = provenance
    if args.reference:
        source_bytes = args.reference.read_bytes()
        result["empirical_scale_reference"] = compare_reference(
            result, json.loads(source_bytes))
        result["empirical_scale_reference"]["source_sha256"] = hashlib.sha256(
            source_bytes).hexdigest()
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
