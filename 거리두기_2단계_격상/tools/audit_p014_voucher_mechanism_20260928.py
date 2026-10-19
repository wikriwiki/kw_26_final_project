"""Read-only, post-score P014 mechanism audit from preserved daily metrics.

P014 is a price-discounted *purchase* of local currency, not a grant wallet.
The current simulator has no voucher-purchase/remaining-face-value ledger.  This
audit reports what the preserved execution actually contains; it deliberately
does not turn generic `policy_hits` into voucher usage or eligibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from datetime import date, timedelta
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dates(first: str, last: str) -> list[str]:
    start, end = date.fromisoformat(first), date.fromisoformat(last)
    return [(start + timedelta(days=i)).isoformat() for i in range((end-start).days+1)]


def nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be nonnegative integer")
    return value


def nonnegative_number(value: object, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{label} must be a finite nonnegative number")
    return float(value)


def arm_audit(ledger: Path, *, arm: str, run: dict, evidence: dict[str, str]) -> dict:
    manifest_path = ledger.with_name(ledger.name + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path in (ledger, manifest_path):
        if evidence.get(path.as_posix()) != sha256(path):
            raise ValueError(f"{arm} sector evidence SHA mismatch: {path}")
    if manifest.get("arm") != arm or manifest.get("citizens") != run["citizens"]:
        raise ValueError(f"{arm} manifest cohort mismatch")
    first, last = run["on"].split(":")
    effect_days = dates(first, last)
    if not set(effect_days).issubset(dates(manifest["start"], manifest["end"])):
        raise ValueError("effect window outside preserved metrics")
    if arm == "on" and manifest.get("policy_id") != "P014":
        raise ValueError("ON policy ID mismatch")
    if arm == "off" and manifest.get("policy_id") is not None:
        raise ValueError("OFF unexpectedly exposed to policy")
    stage2_path = ledger.parent / "stage2.json"
    if evidence.get(stage2_path.as_posix()) != sha256(stage2_path):
        raise ValueError(f"{arm} Stage2 evidence SHA mismatch")
    stage2 = json.loads(stage2_path.read_text(encoding="utf-8"))
    stage2_totals = stage2.get("totals") or {}
    if (stage2.get("quality_gate_pass") is not True
            or stage2_totals.get("metrics_rows") != run["citizens"] * manifest["days"]):
        raise ValueError(f"{arm} Stage2 audit does not cover the full run")

    totals = Counter()
    roster = set()
    seen_receipts = set()
    files = []
    for day in effect_days:
        path = ledger.parent / "metrics" / f"day_{day}.jsonl"
        digest = sha256(path)
        if evidence.get(path.as_posix()) != digest or manifest["metrics_sha256"].get(day) != digest:
            raise ValueError(f"{arm} daily metrics SHA mismatch: {day}")
        files.append({"path": path.as_posix(), "sha256": digest})
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if len(rows) != run["citizens"] or len({row.get("aid") for row in rows}) != len(rows):
            raise ValueError(f"{arm} daily metrics cohort incomplete: {day}")
        daily_roster = {row["aid"] for row in rows}
        if roster and daily_roster != roster:
            raise ValueError(f"{arm} roster drift: {day}")
        roster = daily_roster
        for row in rows:
            if row.get("status") != "ok" or set(row.get("experience_policy_ids") or []) != ({"P014"} if arm == "on" else set()):
                raise ValueError(f"{arm} canonical success/policy exposure mismatch")
            totals["citizen_days"] += 1
            for field in ("policy_spend_today", "instant_discount_today",
                          "grant_applied_today", "policy_hits"):
                value = nonnegative_int(row.get(field), field)
                totals[field + "_total"] += value
                if value:
                    totals[field + "_positive_citizen_days"] += 1
            for receipt in row.get("execution_receipts") or []:
                if receipt.get("kind") != "purchase_receipt":
                    continue
                amount = nonnegative_int(receipt.get("amount"), "receipt amount")
                if receipt.get("agent_id") != row["aid"] or receipt.get("occurred_at") != day:
                    raise ValueError("receipt citizen/day differs from canonical metric")
                if amount <= 0:
                    continue
                event_id = receipt.get("event_id")
                if not event_id or event_id in seen_receipts:
                    raise ValueError("missing or duplicated purchase receipt event ID")
                seen_receipts.add(event_id)
                totals["positive_purchase_receipts"] += 1
                totals["positive_purchase_won"] += amount
                if "P014" in (receipt.get("policy_facts") or {}):
                    totals["receipts_with_p014_eligibility_fact"] += 1
                requested = (receipt.get("decision") or {}).get("requested_payments") or {}
                if nonnegative_number(requested.get("P014", 0), "requested P014 wallet amount"):
                    totals["receipts_requesting_p014_wallet"] += 1
    if totals["citizen_days"] != run["citizens"] * len(effect_days):
        raise ValueError(f"{arm} incomplete effect-period citizen-days")
    sector_rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    effect_sector = [row for row in sector_rows if row["day"] in effect_days]
    if len(effect_sector) != totals["citizen_days"] or {
        (row["aid"], row["day"]) for row in effect_sector
    } != {(aid, day) for aid in roster for day in effect_days}:
        raise ValueError(f"{arm} sector/metric citizen-day matrix differs")
    sector_offline = sum(nonnegative_int(row["offline_spent"], "offline_spent")
                         for row in effect_sector)
    sector_funded = sum(nonnegative_int(row["policy_funded_won"], "policy_funded_won")
                        for row in effect_sector)
    return {"arm": arm, "sector_ledger_path": ledger.as_posix(),
            "sector_ledger_sha256": sha256(ledger),
            "stage2_audit_path": stage2_path.as_posix(),
            "stage2_audit_sha256": sha256(stage2_path),
            "stage2_quality_gate_pass": stage2["quality_gate_pass"],
            "stage2_unrepaired_choice_trace_pass": stage2["unrepaired_choice_trace_pass"],
            "stage2_choice_repair_agents": stage2_totals["stage2_choice_repair_agents"],
            "stage2_hallucinations_corrected": stage2_totals["stage2_hallucinations_corrected"],
            "stage2_spend_amount_fallbacks": stage2_totals["stage2_spend_amount_fallbacks"],
            "effect_days": effect_days, "citizens": len(roster),
            "sector_offline_spent_won": sector_offline,
            "sector_policy_funded_won": sector_funded,
            "receipt_minus_sector_offline_won": totals["positive_purchase_won"] - sector_offline,
            **{name: totals[name] for name in (
                "citizen_days", "positive_purchase_receipts", "positive_purchase_won",
                "policy_spend_today_total", "policy_spend_today_positive_citizen_days",
                "instant_discount_today_total", "instant_discount_today_positive_citizen_days",
                "grant_applied_today_total", "grant_applied_today_positive_citizen_days",
                "policy_hits_total", "policy_hits_positive_citizen_days",
                "receipts_with_p014_eligibility_fact", "receipts_requesting_p014_wallet")},
            "metrics_evidence": files}


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
        raise ValueError("not the frozen P014 40-citizen score")
    evidence = {item["path"].replace("\\", "/"): item["sha256"]
                for item in run["evidence"]}
    policy_path = Path(run["run_provenance"]["policy_file_path"])
    if sha256(policy_path) != run["run_provenance"]["policy_file_sha256"]:
        raise ValueError("frozen P014 policy file SHA mismatch")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("id") != "P014" or policy.get("type") != "price_discount":
        raise ValueError("P014 is not the registered discounted-purchase mechanism")
    on = arm_audit(args.on, arm="on", run=run, evidence=evidence)
    off = arm_audit(args.off, arm="off", run=run, evidence=evidence)
    payload = {
        "schema": "p014_voucher_mechanism_postscore_audit_v1",
        "purpose": "Post-score mechanism/coverage diagnosis; no score or indicator selection changes",
        "policy": "P014", "policy_type": "price_discount",
        "policy_file_path": policy_path.as_posix(),
        "policy_file_sha256": sha256(policy_path),
        "numeric_path": args.numeric.as_posix(),
        "numeric_sha256": sha256(args.numeric),
        "on": on, "off": off,
        "voucher_purchase_count": None,
        "voucher_redemption_count": None,
        "voucher_usage_rate_among_eligible_purchases": None,
        "limitations": [
            "P014 does not create a grant wallet. policy_spend_today is a grant-wallet field, not a local-currency purchase/redemption counter.",
            "The current instant-discount settlement implements sector_voucher, not price_discount; instant_discount_today is not a P014 voucher discount outcome.",
            "Execution receipts expose policy_facts only for grant policies; P014 per-receipt eligibility and voucher redemption are unrecorded.",
            "policy_hits is a coarse category/region event count, not the same-district and merchant-exclusion eligible-purchase denominator. It must not be called voucher uptake.",
            "Exact modeled eligibility would require read-only merchant/home-district joins on a preserved graph; actual voucher purchase/redemption requires new simulator accounting and a new run.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.out)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{args.out}: P014 mechanism audit, uptake unmeasured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
