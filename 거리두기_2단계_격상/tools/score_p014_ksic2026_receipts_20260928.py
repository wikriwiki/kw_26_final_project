"""Post-result exploratory P014 receipt grouping with an intact 2026 catalog.

This is a separate CPU-only sidecar, not a replacement for the frozen score.
It does not prove that the recovered CSV is byte-identical to graph ingestion,
and it cannot estimate the published municipality-year KIPF effect.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


GROUPS = ("47121", "47129")


def file_stats(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    size = nul = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
            nul += chunk.count(b"\x00")
    return digest.hexdigest(), size, nul


def sha256(path: Path) -> str:
    return file_stats(path)[0]


def integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label}: expected nonnegative integer")
    return value


def normalize_ksic(code: str) -> str | None:
    if re.fullmatch(r"G[0-9]{5}", code):
        return code[1:]
    if re.fullmatch(r"[0-9]{5}", code):
        return code
    return None


def has_resolved_industry(code: str) -> bool:
    # A normal non-G prefix (for example I56111) is a definite non-target
    # industry. It remains outside the two groups; its prefix is not relabeled.
    return bool(re.fullmatch(r"(?:[A-Z])?[0-9]{5}", code))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap(on: dict[str, int], off: dict[str, int], roster: list[str],
              scoring: dict) -> tuple[list[float] | None, int]:
    rng = random.Random(scoring["paired_citizen_bootstrap_seed"])
    values = []
    for _ in range(scoring["paired_citizen_bootstrap_draws"]):
        sampled = rng.choices(roster, k=len(roster))
        denominator = sum(off.get(aid, 0) for aid in sampled)
        if denominator > 0:
            numerator = sum(on.get(aid, 0) for aid in sampled)
            values.append(100 * (numerator - denominator) / denominator)
    if len(values) < scoring["minimum_valid_bootstrap_draws"]:
        return None, len(values)
    return [percentile(values, q) for q in scoring["ci_percentiles"]], len(values)


def read_arm(ledger: Path, arm: str, run: dict, evidence: dict,
             plan: dict) -> tuple[dict, list[dict], set[str]]:
    manifest_path = ledger.with_name(ledger.name + ".manifest.json")
    for path in (ledger, manifest_path):
        if evidence.get(path.as_posix()) != sha256(path):
            raise ValueError(f"{arm} evidence mismatch: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate = plan["input_gate"]
    if manifest.get("arm") != arm or manifest.get("citizens") != gate["citizens"]:
        raise ValueError(f"{arm} manifest cohort mismatch")
    if manifest.get("policy_id") != ("P014" if arm == "on" else None):
        raise ValueError(f"{arm} policy exposure mismatch")
    run_id = (manifest.get("prompt_provenance") or {}).get("run_id")
    if (not isinstance(run_id, str) or not run_id
            or (manifest.get("provenance") or {}).get("experience_run_id") != [run_id]):
        raise ValueError(f"{arm} manifest run provenance mismatch")
    sector = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["day"] not in gate["effect_days"]:
            continue
        key = (row["aid"], row["day"])
        if key in sector:
            raise ValueError("duplicate sector citizen-day")
        sector[key] = integer(row["offline_spent"], "offline_spent")

    roster: set[str] = set()
    receipts = []
    seen_events = set()
    keys = set()
    metric_evidence = []
    for day in gate["effect_days"]:
        path = ledger.parent / "metrics" / f"day_{day}.jsonl"
        digest = sha256(path)
        if (evidence.get(path.as_posix()) != digest
                or manifest["metrics_sha256"].get(day) != digest):
            raise ValueError(f"{arm} metrics SHA mismatch: {day}")
        metric_evidence.append({"path": path.as_posix(), "sha256": digest})
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        daily_roster = {row.get("aid") for row in rows}
        if (len(rows) != gate["citizens"] or len(daily_roster) != len(rows)
                or (roster and roster != daily_roster)):
            raise ValueError(f"{arm} canonical roster differs: {day}")
        roster = daily_roster
        for row in rows:
            aid = row["aid"]
            key = (aid, day)
            if key in keys:
                raise ValueError("duplicate canonical citizen-day")
            keys.add(key)
            if (row.get("status") != "ok" or row.get("experience_day") != day
                    or row.get("experience_run_id") != run_id
                    or set(row.get("experience_policy_ids") or []) !=
                    ({"P014"} if arm == "on" else set())):
                raise ValueError(f"{arm} canonical identity/exposure mismatch")
            daily_amount = 0
            for receipt in row.get("execution_receipts") or []:
                if receipt.get("kind") != "purchase_receipt":
                    continue
                amount = integer(receipt.get("amount"), "receipt amount")
                if (receipt.get("agent_id") != aid or receipt.get("occurred_at") != day
                        or receipt.get("run_id") != run_id):
                    raise ValueError("receipt citizen/day/run mismatch")
                if amount == 0:
                    continue
                event_id, poi_id = receipt.get("event_id"), receipt.get("poi_id")
                if (not isinstance(event_id, str) or not event_id
                        or event_id in seen_events or not isinstance(poi_id, str)
                        or not poi_id or receipt.get("purchase_status") != "purchased"):
                    raise ValueError("invalid positive purchase receipt identity/status")
                seen_events.add(event_id)
                receipts.append({"aid": aid, "day": day, "poi_id": poi_id,
                                 "event_id": event_id, "amount": amount})
                daily_amount += amount
            if sector.get(key) != daily_amount:
                raise ValueError("per-citizen-day receipt/sector offline amount mismatch")
    if keys != set(sector) or len(keys) != gate["citizen_days_each_arm"]:
        raise ValueError(f"{arm} effect-period citizen-day matrix mismatch")
    return {"arm": arm, "run_id": run_id,
            "sector_ledger_path": ledger.as_posix(), "sector_ledger_sha256": sha256(ledger),
            "metrics_evidence": metric_evidence, "citizen_days": len(keys),
            "positive_receipts": len(receipts),
            "positive_receipt_won": sum(row["amount"] for row in receipts)}, receipts, roster


def read_catalog(source: Path, needed_pois: set[str], plan: dict) -> tuple[dict, dict]:
    expected = plan["source"]
    digest, size, nul = file_stats(source)
    if (digest != expected["sha256"] or size != expected["bytes"]
            or nul != expected["nul_bytes"]):
        raise ValueError("recovered catalog byte/SHA/NUL gate failed")
    duplicate_ids = 0
    seen_ids = set()
    rows = 0
    mapping = {}
    group_counts = Counter()
    group_names = defaultdict(set)
    missing_codes = malformed_codes = 0
    id_field, code_field, name_field = (expected[field] for field in
                                      ("merchant_id_field", "industry_code_field",
                                       "industry_name_field"))
    with source.open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader)
        if len(set(header)) != len(header):
            raise ValueError("duplicate source header")
        indices = [header.index(field) for field in (id_field, code_field, name_field)]
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"catalog column count differs at row {rows + 1}")
            merchant_id, raw_code, name = (row[index] for index in indices)
            if not merchant_id:
                raise ValueError("catalog merchant ID missing")
            missing_codes += not bool(raw_code)
            malformed_codes += bool(raw_code) and not has_resolved_industry(raw_code)
            duplicate_ids += merchant_id in seen_ids
            seen_ids.add(merchant_id)
            rows += 1
            code = normalize_ksic(raw_code)
            if code in GROUPS:
                group_counts[code] += 1
                group_names[code].add(name)
            poi_id = "C_" + merchant_id
            if poi_id in needed_pois:
                if poi_id in mapping:
                    raise ValueError("ambiguous receipt/catalog merchant join")
                mapping[poi_id] = {"ksic": code, "raw_code": raw_code, "name": name,
                                   "classification_resolved": has_resolved_industry(raw_code)}
    if rows != expected["strict_utf8_rows"]:
        raise ValueError("catalog row-count gate failed")
    if duplicate_ids > plan["input_gate"]["catalog_duplicate_merchant_ids_allowed"]:
        raise ValueError("catalog duplicate-merchant-ID gate failed")
    return {"path": source.as_posix(), "sha256": digest, "bytes": size,
            "nul_bytes": nul, "strict_utf8_rows": rows,
            "duplicate_merchant_ids": duplicate_ids,
            "missing_industry_codes": missing_codes,
            "malformed_industry_codes": malformed_codes,
            "group_catalog_rows": {code: group_counts[code] for code in GROUPS},
            "group_names": {code: sorted(group_names[code]) for code in GROUPS},
            "vintage": expected["vintage"],
            "original_graph_ingest_byte_identity_confirmed": False,
            "poi_id_rule": expected["poi_id_rule"],
            "code_normalization": expected["code_normalization"]}, mapping


def group_arm(summary: dict, receipts: list[dict], mapping: dict, plan: dict) -> tuple[dict, dict]:
    id_joined = [row for row in receipts if row["poi_id"] in mapping]
    matched = [row for row in id_joined if mapping[row["poi_id"]]["classification_resolved"]]
    matched_won = sum(row["amount"] for row in matched)
    total_won = summary["positive_receipt_won"]
    count_fraction = len(matched) / len(receipts) if receipts else 0
    won_fraction = matched_won / total_won if total_won else 0
    gate = plan["input_gate"]
    join_pass = (count_fraction >= gate["minimum_join_count_fraction_each_arm"]
                 and won_fraction >= gate["minimum_join_won_fraction_each_arm"])
    by_citizen = {code: defaultdict(int) for code in GROUPS}
    stats = {}
    for code in GROUPS:
        rows = [row for row in matched if mapping[row["poi_id"]]["ksic"] == code]
        for row in rows:
            by_citizen[code][row["aid"]] += row["amount"]
        stats[code] = {"won": sum(row["amount"] for row in rows),
                       "positive_receipts": len(rows),
                       "unique_citizens": len({row["aid"] for row in rows}),
                       "unique_pois": len({row["poi_id"] for row in rows})}
    summary.update({"matched_receipts": len(matched), "matched_receipt_won": matched_won,
                    "id_joined_receipts": len(id_joined),
                    "unresolved_industry_receipts": len(id_joined) - len(matched),
                    "unresolved_industry_won": sum(row["amount"] for row in id_joined) - matched_won,
                    "join_count_fraction": count_fraction, "join_won_fraction": won_fraction,
                    "join_gate_pass": join_pass, "ambiguous_join_receipts": 0,
                    "unmatched_receipts": len(receipts) - len(matched),
                    "unmatched_receipt_won": total_won - matched_won,
                    "unmatched_pois": sorted({row["poi_id"] for row in receipts
                                              if row["poi_id"] not in mapping
                                              or not mapping[row["poi_id"]]["classification_resolved"]}),
                    "groups": stats})
    return summary, by_citizen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--numeric", type=Path, required=True)
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema") != "p014_ksic2026_posthoc_receipt_proxy_plan_v1":
        raise ValueError("unexpected posthoc plan schema")
    if sha256(args.numeric) != plan["input_gate"]["numeric_sha256"]:
        raise ValueError("frozen numeric SHA differs")
    score = json.loads(args.numeric.read_text(encoding="utf-8"))
    if len(score["runs"]) != 1:
        raise ValueError("expected single frozen P014 run")
    run = score["runs"][0]
    effect_range = f'{plan["input_gate"]["effect_days"][0]}:{plan["input_gate"]["effect_days"][-1]}'
    if (run.get("policy_id") != "P014" or run.get("citizens") != plan["input_gate"]["citizens"]
            or run.get("on") != effect_range or run.get("off") != effect_range):
        raise ValueError("not the planned P014 effect-period pair")
    evidence = {item["path"].replace("\\", "/"): item["sha256"] for item in run["evidence"]}
    on, on_receipts, on_roster = read_arm(args.on, "on", run, evidence, plan)
    off, off_receipts, off_roster = read_arm(args.off, "off", run, evidence, plan)
    if on_roster != off_roster or on["run_id"] == off["run_id"]:
        raise ValueError("ON/OFF roster or distinct-run gate failed")
    source, mapping = read_catalog(args.source, {row["poi_id"] for row in
                                               on_receipts + off_receipts}, plan)
    on, on_by = group_arm(on, on_receipts, mapping, plan)
    off, off_by = group_arm(off, off_receipts, mapping, plan)
    technical_gate_pass = on["join_gate_pass"] and off["join_gate_pass"]
    indicators = []
    sparse_gate = plan["scoring"]["sparse_interpretation_gate"]
    for code in GROUPS:
        a, b = on["groups"][code], off["groups"][code]
        sparse = any(stats["positive_receipts"] < sparse_gate["minimum_positive_receipts_each_arm_group"]
                     or stats["unique_citizens"] < sparse_gate["minimum_unique_citizens_each_arm_group"]
                     for stats in (a, b))
        defined = technical_gate_pass and b["won"] > 0
        point = 100 * (a["won"] - b["won"]) / b["won"] if defined else None
        ci, valid_draws = bootstrap(on_by[code], off_by[code], sorted(on_roster),
                                    plan["scoring"]) if defined else (None, 0)
        reason = ("Technical source/join gate failed" if not technical_gate_pass else
                  "OFF receipt-won denominator is zero" if b["won"] == 0 else
                  "Technical receipt-group change only; sparse support blocks direction/size interpretation"
                  if sparse else "Descriptive 2026-KSIC receipt-group ON/OFF change; no empirical-effect accuracy claim")
        indicators.append({"id": f"P014-KSIC2026-{code}", "ksic": code,
                           "exploratory_not_registered": True, "posthoc_exploratory": True,
                           "simulation": point, "simulation_unit": "%", "ci": ci,
                           "n": len(on_roster), "bootstrap_valid_draws": valid_draws,
                           "bootstrap_requested_draws": plan["scoring"]["paired_citizen_bootstrap_draws"],
                           "on": a, "off": b,
                           "sparse_interpretation_blocked": sparse,
                           "sparse_interpretation_gate": sparse_gate,
                           "direction_comparable": False, "direct_gap_allowed": False,
                           "approximate_empirical_size_comparable": False,
                           "empirical_reference_id": f"P014-KIPF-{code}",
                           "estimand_alignment": "different", "reason": reason,
                           "method": plan["scoring"]["point_formula"]})
    payload = {"schema": "p014_ksic2026_posthoc_receipt_proxy_v1",
               "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "policy": "P014", "posthoc_exploratory": True,
               "purpose": "Additional receipt grouping; original frozen score and selection unchanged",
               "plan_path": args.plan.as_posix(), "plan_sha256": sha256(args.plan),
               "tool_path": Path(__file__).as_posix(), "tool_sha256": sha256(Path(__file__)),
               "numeric_path": args.numeric.as_posix(), "numeric_sha256": sha256(args.numeric),
               "effect_days": plan["input_gate"]["effect_days"], "citizens": len(on_roster),
               "source": source, "on": on, "off": off,
               "technical_gate_pass": technical_gate_pass,
               "join_gate": {"minimum_count_fraction_each_arm": plan["input_gate"]["minimum_join_count_fraction_each_arm"],
                             "minimum_won_fraction_each_arm": plan["input_gate"]["minimum_join_won_fraction_each_arm"],
                             "catalog_duplicate_merchant_ids_allowed": 0,
                             "ambiguous_join_receipts_allowed": 0},
               "ci_method": "Paired citizen bootstrap, Python random.choices, linear-interpolated percentiles",
               "bootstrap_seed": plan["scoring"]["paired_citizen_bootstrap_seed"],
               "indicators": indicators, "interpretation": plan["interpretation"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
        temporary.replace(args.out)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"path": args.out.as_posix(), "technical_gate_pass": technical_gate_pass,
                      "on_join_fraction": on["join_count_fraction"],
                      "off_join_fraction": off["join_count_fraction"],
                      "indicators": indicators}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
