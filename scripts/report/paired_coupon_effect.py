"""Score P010's registered spending indicators from complete matched ledgers.

The survey MPC (P010-1) is a different outcome and is deliberately excluded.
The two spending indicators are internal ON/OFF effects, not direct external
accuracy scores. Both arms must contain the same complete pre/post calendar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from paired_grant_effect import dates, read_jsonl, roster_file, score, verify_manifests


ROOT = Path(__file__).resolve().parents[2]
POLICY_FILE = ROOT / "data/neo4j_load/policies/P010.json"
ALL_DAYS = dates("2025-07-12", "2025-07-23")
PRE_DAYS = dates("2025-07-15", "2025-07-16")
POST_DAYS = dates("2025-07-22", "2025-07-23")


def score_coupon(on_rows: list[dict], off_rows: list[dict], *, roster: list[str],
                 draws: int = 2000, expected_amount_by_aid: dict[str, int]) -> dict:
    if (not isinstance(expected_amount_by_aid, dict)
            or set(expected_amount_by_aid) != set(roster)
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                   for value in expected_amount_by_aid.values())):
        raise ValueError("complete positive per-citizen entitlement map required")
    expected_issued_won = sum(expected_amount_by_aid.values())
    if isinstance(expected_issued_won, bool) or not isinstance(expected_issued_won, int) \
            or expected_issued_won <= 0:
        raise ValueError("independently expected issued grant must be positive")
    common = dict(roster=roster, days=ALL_DAYS, policy_id="P010", draws=draws,
                  expected_recipients=len(roster),
                  expected_issued_won=expected_issued_won)
    post = score(on_rows, off_rows, effect_days=POST_DAYS, **common)
    final_receipts = {row["aid"]: row["grant_received_cumulative"]
                      for row in on_rows if row["day"] == ALL_DAYS[-1]}
    if final_receipts != expected_amount_by_aid:
        raise ValueError("citizen grant receipts differ from frozen entitlements")
    pre = score(on_rows, off_rows, effect_days=PRE_DAYS, **common)
    return {
        "policy_id": "P010", "ledger_start": ALL_DAYS[0], "ledger_end": ALL_DAYS[-1],
        "prepolicy_start": PRE_DAYS[0], "prepolicy_end": PRE_DAYS[-1],
        "effect_start": POST_DAYS[0], "effect_end": POST_DAYS[-1],
        "comparison": "within_model_same_calendar_on_off; no_direct_empirical_magnitude_score",
        "P010-2": {
            "outcome": "eligible_offline_realized_spending",
            "on_won": post["eligible_offline_spend_on_won"],
            "off_won": post["eligible_offline_spend_off_won"],
            "difference_won": post["eligible_offline_difference_won"],
            "relative_change": post["eligible_offline_relative_change"],
            "citizen_bootstrap_95_interval": post[
                "eligible_offline_relative_citizen_bootstrap_95_interval"],
        },
        "P010-3": {
            "outcome": "total_recorded_realized_spending",
            "on_won": post["recorded_total_spend_on_won"],
            "off_won": post["recorded_total_spend_off_won"],
            "difference_won": post["recorded_total_spend_difference_won"],
            "relative_change": post["recorded_total_relative_change"],
            "citizen_bootstrap_95_interval": post[
                "recorded_total_relative_citizen_bootstrap_95_interval"],
        },
        "prepolicy_balance_diagnostic": {
            "eligible_difference_won": pre["eligible_offline_difference_won"],
            "total_difference_won": pre["recorded_total_spend_difference_won"],
            "eligible_relative_change": pre["eligible_offline_relative_change"],
            "total_relative_change": pre["recorded_total_relative_change"],
            "scope": "Same-date ON/OFF difference before policy activation; inspect before "
                     "attributing the post-window difference to the policy.",
        },
        "funding": {"issued_won": post["grant_issued_won"],
                    "spent_won": post["grant_spent_won"],
                    "remaining_won": post["grant_remaining_won"],
                    "recipients": post["grant_recipients"]},
        "choice_repair_sensitivity": post["choice_repair_sensitivity"],
        "scope": "P010-2/3 are matched synthetic-citizen spending outcomes. The BOK "
                 "survey's nine-item self-report MPC is not reproduced by these effects.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--entitlements", type=Path, required=True,
                        help="Frozen per-citizen P010 grant entitlements")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads(POLICY_FILE.read_text(encoding="utf-8"))
    if (policy.get("id") != "P010" or policy.get("type") != "grant"
            or policy.get("effective_from") != "2025-07-21"):
        raise ValueError("frozen P010 policy schedule differs from registered experiment")
    roster = roster_file(args.roster)
    provenance = verify_manifests(args.on, args.off, roster=roster,
                                  days=ALL_DAYS, policy_id="P010")
    policy_sha = hashlib.sha256(POLICY_FILE.read_bytes()).hexdigest()
    if provenance["policy_file_sha256"] != policy_sha:
        raise ValueError("paired ledgers were not exported with frozen P010 policy")
    entitlement_bytes = args.entitlements.read_bytes()
    entitlements = json.loads(entitlement_bytes)
    amounts = entitlements.get("amount_by_aid")
    roster_sha = hashlib.sha256(json.dumps(sorted(roster), ensure_ascii=False).encode(
        "utf-8")).hexdigest()
    if (entitlements.get("schema") != "grant_entitlements_v1"
            or entitlements.get("policy_id") != "P010"
            or entitlements.get("effective_day") != "2025-07-21"
            or entitlements.get("policy_file_sha256") != policy_sha
            or not isinstance(entitlements.get("source_archive_sha256"), str)
            or len(entitlements["source_archive_sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in
                   entitlements["source_archive_sha256"])
            or not str(entitlements.get("source_agent_member") or "").endswith(
                "_agent.jsonl.gz")
            or entitlements.get("roster_sha256") != roster_sha
            or entitlements.get("citizen_count") != len(roster)
            or entitlements.get("recipient_count") != len(roster)
            or not isinstance(amounts, dict)
            or entitlements.get("expected_issued_won") != sum(
                amount for amount in amounts.values() if isinstance(amount, int)
                and not isinstance(amount, bool))):
        raise ValueError("frozen P010 entitlement manifest is inconsistent")
    result = score_coupon(read_jsonl(args.on), read_jsonl(args.off), roster=roster,
                          draws=args.draws,
                          expected_amount_by_aid=amounts)
    result["provenance"] = provenance
    result["provenance"]["entitlements_sha256"] = hashlib.sha256(
        entitlement_bytes).hexdigest()
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
