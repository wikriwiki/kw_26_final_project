"""Read-only posthoc decomposition of the preserved P010 paired spending gap.

This is a simulator channel diagnostic, not a causal policy-effect estimate or
an empirical MPC score. It does not change the frozen scorer or arm outputs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/recovery_20260928/multipolicy_v53/p010"
DAYS = ("2025-07-21", "2025-07-22", "2025-07-23")
FIELDS = (
    "cm_today_total_incl_online", "cm_online_total", "cm_planned_total",
    "cm_anchor_total", "cm_policy_allocated_total", "cm_policy_liquidity_relief",
    "cm_mechanical_policy_uplift", "cm_selected_policy_liquidity",
    "n_events", "n_includes",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_arm(arm: str):
    metrics = BASE / arm / arm / "metrics"
    result = {}
    source_hashes = {}
    for day in DAYS:
        path = metrics / f"day_{day}.jsonl"
        source_hashes[path.relative_to(ROOT).as_posix()] = file_sha256(path)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if len(rows) != 80 or any(row.get("status") != "ok" for row in rows):
            raise ValueError(f"Incomplete {arm} {day}")
        for row in rows:
            key = (row["aid"], day)
            if key in result:
                raise ValueError(f"Duplicate citizen-day {key}")
            receipts = [r for r in row.get("execution_receipts") or []
                        if r.get("kind") == "purchase_receipt" and int(r.get("amount") or 0) > 0]
            offline = sum(int(r["amount"]) for r in receipts)
            online = int(row["cm_online_total"])
            total = int(row["cm_today_total_incl_online"])
            if total != offline + online:
                raise ValueError(f"Daily channel sum mismatch {arm} {key}")
            result[key] = {**{name: row.get(name) for name in FIELDS},
                           "offline_positive_receipt_won": offline,
                           "positive_receipt_events": len(receipts)}
    if len(result) != 240:
        raise ValueError(f"Unexpected {arm} citizen-day count {len(result)}")
    return result, source_hashes


def aggregate(rows):
    fields = (*FIELDS, "offline_positive_receipt_won", "positive_receipt_events")
    return {name: sum(float(row.get(name) or 0) for row in rows.values()) for name in fields}


def main() -> None:
    on, on_sources = read_arm("on")
    off, off_sources = read_arm("off")
    if set(on) != set(off):
        raise ValueError("ON/OFF citizen-day keys differ")
    on_sum = aggregate(on)
    off_sum = aggregate(off)
    delta = {name: on_sum[name] - off_sum[name] for name in on_sum}
    total_delta = delta["cm_today_total_incl_online"]
    online_delta = delta["cm_online_total"]
    offline_delta = delta["offline_positive_receipt_won"]
    if total_delta != online_delta + offline_delta:
        raise ValueError("Paired gap does not decompose")
    if on_sum["cm_policy_allocated_total"] != 45672:
        raise ValueError("Unexpected P010 funded ledger sum")
    result = {
        "status": "posthoc_simulator_channel_diagnostic_not_empirical_policy_effect",
        "source_sha256": {**on_sources, **off_sources},
        "policy_effect_days": list(DAYS),
        "paired_citizen_days": len(on),
        "on_sum": on_sum,
        "off_sum": off_sum,
        "on_minus_off": delta,
        "online_fraction_of_total_gap": online_delta / total_delta if total_delta else None,
        "offline_fraction_of_total_gap": offline_delta / total_delta if total_delta else None,
        "interpretation_limit": "Online is a modelled non-coupon-eligible channel. Channel arithmetic cannot identify why the plan changed or establish real-world MPC agreement.",
    }
    out = BASE / "p010_channel_gap_audit.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    by_citizen = {}
    for (aid, _day), row in on.items():
        by_citizen.setdefault(aid, {"on": 0, "off": 0})["on"] += int(
            row["cm_today_total_incl_online"])
    for (aid, _day), row in off.items():
        by_citizen.setdefault(aid, {"on": 0, "off": 0})["off"] += int(
            row["cm_today_total_incl_online"])
    deltas = sorted((value["on"] - value["off"] for value in by_citizen.values()),
                    reverse=True)
    if len(deltas) != 80 or sum(deltas) != total_delta:
        raise ValueError("Citizen-level paired deltas do not reconcile")
    concentration = {
        "status": "posthoc_citizen_gap_concentration_diagnostic_not_policy_effect",
        "source_sha256": {**on_sources, **off_sources},
        "paired_citizens": len(deltas),
        "positive_delta_citizens": sum(value > 0 for value in deltas),
        "negative_delta_citizens": sum(value < 0 for value in deltas),
        "zero_delta_citizens": sum(value == 0 for value in deltas),
        "top_five_positive_deltas_won": deltas[:5],
        "top_five_sum_won": sum(deltas[:5]),
        "top_five_share_of_net_gap": sum(deltas[:5]) / total_delta if total_delta else None,
        "net_gap_won": total_delta,
        "interpretation_limit": "A few simulated citizens can dominate a short-window net gap; this does not measure real-world population effects.",
    }
    concentration_out = BASE / "p010_citizen_gap_concentration_audit.json"
    concentration_out.write_text(
        json.dumps(concentration, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({"total_gap_won": total_delta, "online_gap_won": online_delta,
                      "offline_gap_won": offline_delta,
                      "online_fraction": result["online_fraction_of_total_gap"],
                      "output_sha256": file_sha256(out),
                      "concentration_sha256": file_sha256(concentration_out)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
