"""Score a completed, matched policy/no-policy grant experiment from daily ledgers.

The outcome is an *indirect spending proxy*: this simulator does not record the
payment instrument needed to reproduce an external card-sales estimand. Neither
the empirical target nor a policy-specific result is used in this calculation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from datetime import date, timedelta
from pathlib import Path


MONEY = ("offline_spent", "online_spent", "self_month_cumulative",
         "grant_received_cumulative",
         "grant_remaining", "grant_spent_today")
CLEAN_CHOICES = frozenset(("unrepaired", "not_applicable"))
VALID_CHOICES = CLEAN_CHOICES | {"partial_repair"}


def dates(start: str, end: str) -> list[str]:
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last:
        raise ValueError("start exceeds end")
    return [(first + timedelta(days=i)).isoformat()
            for i in range((last - first).days + 1)]


def roster_file(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    ids = list(raw) if isinstance(raw, (list, dict)) else []
    if not ids or any(not isinstance(aid, str) or not aid for aid in ids):
        raise ValueError("roster must be a nonempty JSON list or object keyed by aid")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate citizen in roster")
    return ids


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _money(row: dict, field: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer: {row.get('aid')} {row.get('day')}")
    return value


def _index(rows: list[dict], roster: list[str], days: list[str], arm: str,
           policy_id: str) -> dict[tuple[str, str], dict]:
    expected = {(aid, day) for aid in roster for day in days}
    got: dict[tuple[str, str], dict] = {}
    for row in rows:
        if row.get("arm") != arm or row.get("policy_id") != policy_id:
            raise ValueError(f"wrong arm or policy in {arm} ledger")
        key = (row.get("aid"), row.get("day"))
        if key in got:
            raise ValueError(f"duplicate citizen-day in {arm}: {key}")
        got[key] = row
    if set(got) != expected:
        raise ValueError(f"incomplete or extra citizen-day matrix in {arm}: "
                         f"missing={len(expected - set(got))}, extra={len(set(got) - expected)}")
    for row in got.values():
        choice = row.get("s2_choice_status")
        if not isinstance(choice, str) or choice not in VALID_CHOICES:
            raise ValueError("missing or invalid Stage2 choice provenance")
        for field in MONEY:
            _money(row, field)
        eligible = row.get("eligible_offline_spent")
        if (isinstance(eligible, bool) or not isinstance(eligible, int)
                or eligible < 0 or eligible > row["offline_spent"]):
            raise ValueError("invalid eligible offline spend")
        if row["grant_spent_today"] > row["offline_spent"]:
            raise ValueError("grant payment exceeds offline gross")
        if row["grant_spent_today"] > eligible:
            raise ValueError("grant payment exceeds eligible gross")
    return got


def score(on_rows: list[dict], off_rows: list[dict], *, roster: list[str],
          days: list[str], policy_id: str, draws: int = 2000,
          seed: int = 20260926,
          effect_days: list[str] | None = None,
          expected_recipients: int | None = None,
          expected_issued_won: int | None = None) -> dict:
    if not roster or not days or len(set(roster)) != len(roster) or len(set(days)) != len(days):
        raise ValueError("nonempty unique roster and days required")
    if draws < 0:
        raise ValueError("draws must be nonnegative")
    if dates(days[0], days[-1]) != days:
        raise ValueError("ledger days must be contiguous and chronological")
    effect_days = days if effect_days is None else effect_days
    if (not effect_days or len(set(effect_days)) != len(effect_days)
            or dates(effect_days[0], effect_days[-1]) != effect_days
            or not set(effect_days).issubset(days)):
        raise ValueError("effect days must be a nonempty contiguous subset of ledger days")
    effect_day_set = set(effect_days)
    on = _index(on_rows, roster, days, "on", policy_id)
    off = _index(off_rows, roster, days, "off", policy_id)
    per_citizen = []
    for aid in roster:
        received, spent = 0, 0
        choice_repaired = False
        previous_self = {"on": ("", 0), "off": ("", 0)}
        values = {key: 0 for key in ("on_total", "off_total", "on_offline",
                                     "off_offline", "on_eligible", "off_eligible")}
        for day in days:
            p, c = on[(aid, day)], off[(aid, day)]
            choice_repaired |= (p["s2_choice_status"] not in CLEAN_CHOICES or
                                c["s2_choice_status"] not in CLEAN_CHOICES)
            if any(c[field] != 0 for field in
                   ("grant_received_cumulative", "grant_remaining", "grant_spent_today")):
                raise ValueError(f"policy funding leaked into control: {aid} {day}")
            month = day[:7]
            for arm, row in (("on", p), ("off", c)):
                old_month, old_amount = previous_self[arm]
                prior = old_amount if old_month == month else 0
                own_outflow = (row["offline_spent"] - row["grant_spent_today"]
                               + row["online_spent"])
                if row["self_month_cumulative"] - prior != own_outflow:
                    raise ValueError(f"own-spend monthly ledger mismatch: {aid} {day} {arm}")
                previous_self[arm] = (month, row["self_month_cumulative"])
            new_received = p["grant_received_cumulative"]
            if new_received < received:
                raise ValueError(f"grant receipt decreased: {aid} {day}")
            received = new_received
            spent += p["grant_spent_today"]
            if received - spent != p["grant_remaining"]:
                raise ValueError(f"grant wallet does not reconcile: {aid} {day}")
            if day in effect_day_set:
                values["on_offline"] += p["offline_spent"]
                values["off_offline"] += c["offline_spent"]
                values["on_total"] += p["offline_spent"] + p["online_spent"]
                values["off_total"] += c["offline_spent"] + c["online_spent"]
                values["on_eligible"] += p["eligible_offline_spent"]
                values["off_eligible"] += c["eligible_offline_spent"]
        values.update(aid=aid, received=received, spent=spent,
                      remaining=on[(aid, days[-1])]["grant_remaining"],
                      choice_repaired=choice_repaired)
        per_citizen.append(values)
    issued = sum(row["received"] for row in per_citizen)
    if issued <= 0:
        raise ValueError("no grant was delivered in treatment")
    recipients = sum(row["received"] > 0 for row in per_citizen)
    if expected_recipients is not None and recipients != expected_recipients:
        raise ValueError(f"grant recipient count mismatch: {recipients} != {expected_recipients}")
    if expected_issued_won is not None and issued != expected_issued_won:
        raise ValueError(f"grant issued amount mismatch: {issued} != {expected_issued_won}")
    summed = {key: sum(row[key] for row in per_citizen)
              for key in ("on_total", "off_total", "on_offline", "off_offline",
                          "on_eligible", "off_eligible", "received", "spent", "remaining")}
    def contrast(rows: list[dict], field: str) -> float:
        numerator = sum(r["on_" + field] - r["off_" + field] for r in rows)
        denominator = sum(r["received"] for r in rows)
        return numerator / denominator if denominator else float("nan")
    def relative_change(rows: list[dict], field: str) -> float:
        off_total = sum(r["off_" + field] for r in rows)
        return (sum(r["on_" + field] - r["off_" + field] for r in rows) / off_total
                if off_total else float("nan"))
    ratio = contrast(per_citizen, "total")
    clean_citizens = [row for row in per_citizen if not row["choice_repaired"]]

    def clean_contrast(field: str) -> float | None:
        value = contrast(clean_citizens, field) if clean_citizens else float("nan")
        return value if math.isfinite(value) else None
    rng = random.Random(seed)
    boot = {field: [] for field in ("total", "offline", "eligible")}
    boot_relative = {field: [] for field in ("total", "eligible")}
    for _ in range(draws):
        sample = rng.choices(per_citizen, k=len(per_citizen))
        for field in boot:
            value = contrast(sample, field)
            if math.isfinite(value):  # a resample can contain no recipients
                boot[field].append(value)
        for field in boot_relative:
            value = relative_change(sample, field)
            if math.isfinite(value):
                boot_relative[field].append(value)
    for values in boot.values():
        values.sort()
    for values in boot_relative.values():
        values.sort()

    def interval(values: list[float]) -> list[float] | None:
        return ([values[int(0.025 * (len(values) - 1))],
                 values[int(0.975 * (len(values) - 1))]] if values else None)
    return {
        "policy_id": policy_id, "start": days[0], "end": days[-1],
        "citizens": len(roster), "days": len(days),
        "effect_start": effect_days[0], "effect_end": effect_days[-1],
        "effect_days": len(effect_days),
        "grant_recipients": recipients,
        "complete_matrix": True, "funding_reconciled": True,
        "recorded_total_spend_on_won": summed["on_total"],
        "recorded_total_spend_off_won": summed["off_total"],
        "recorded_total_spend_difference_won": summed["on_total"] - summed["off_total"],
        "recorded_total_relative_change": (relative_change(per_citizen, "total")
                                           if summed["off_total"] else None),
        "recorded_total_relative_citizen_bootstrap_95_interval": interval(
            boot_relative["total"]),
        "grant_issued_won": issued, "grant_spent_won": summed["spent"],
        "grant_remaining_won": summed["remaining"],
        "incremental_recorded_spend_per_grant_won": ratio,
        "citizen_bootstrap_95_interval": interval(boot["total"]),
        "bootstrap_valid_draws": len(boot["total"]),
        "offline_difference_won": summed["on_offline"] - summed["off_offline"],
        "offline_effect_per_grant_won": contrast(per_citizen, "offline"),
        "offline_citizen_bootstrap_95_interval": interval(boot["offline"]),
        "eligible_offline_spend_on_won": summed["on_eligible"],
        "eligible_offline_spend_off_won": summed["off_eligible"],
        "eligible_offline_difference_won": summed["on_eligible"] - summed["off_eligible"],
        "eligible_offline_relative_change": (relative_change(per_citizen, "eligible")
                                             if summed["off_eligible"] else None),
        "eligible_offline_relative_citizen_bootstrap_95_interval": interval(
            boot_relative["eligible"]),
        "eligible_offline_effect_per_grant_won": contrast(per_citizen, "eligible"),
        "eligible_offline_citizen_bootstrap_95_interval": interval(boot["eligible"]),
        "choice_repair_sensitivity": {
            "unrepaired_citizens": len(clean_citizens),
            "excluded_citizens": len(roster) - len(clean_citizens),
            "incremental_recorded_spend_per_grant_won": clean_contrast("total"),
            "eligible_offline_effect_per_grant_won": clean_contrast("eligible"),
            "scope": "Diagnostic complete-citizen restriction: excludes any citizen "
                     "with a repaired Stage2 place choice in either arm. Not a "
                     "population effect or prompt accuracy score.",
        },
        "comparison": "indirect_proxy",
        "scope": "Within-run, same-calendar matched-citizen spending effect for the "
                 "reported effect window. External outcome, population and "
                 "counterfactual definitions require separate alignment audit; do not "
                 "subtract this estimate as a direct empirical accuracy error.",
    }


def compare_reference(result: dict, reference: dict) -> dict:
    """Give a rough sector-matched scale view without declaring estimand equality."""
    if (result.get("effect_start", result.get("start")) != result.get("start")
            or result.get("effect_end", result.get("end")) != result.get("end")):
        raise ValueError("external grant reference requires the complete effect window")
    if (reference.get("policy_id") != result.get("policy_id")
            or reference.get("simulation_start") != result.get("start")
            or reference.get("simulation_end") != result.get("end")
            or reference.get("simulation_proxy") != "eligible_offline_effect_per_grant_won"):
        raise ValueError("grant reference does not match the scored policy, window or proxy")
    bounds = reference.get("external_ratio_interval")
    if (not isinstance(bounds, list) or len(bounds) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in bounds)
            or not 0 < bounds[0] <= bounds[1]):
        raise ValueError("invalid external grant-effect range")
    simulated = result["eligible_offline_effect_per_grant_won"]
    if not isinstance(simulated, (int, float)) or not math.isfinite(simulated):
        raise ValueError("missing eligible-offline grant-effect proxy")
    midpoint = (bounds[0] + bounds[1]) / 2
    return {
        "source": reference.get("source"),
        "source_url": reference.get("source_url"),
        "external_estimand": reference.get("external_estimand"),
        "external_population": reference.get("population"),
        "external_window_note": reference.get("external_window_note"),
        "comparison_status": "proxy_scale_reference; no direct accuracy score",
        "external_ratio_interval": bounds,
        "simulation_proxy": "eligible_offline_effect_per_grant_won",
        "simulated_ratio": simulated,
        "simulated_citizen_bootstrap_95_interval": result.get(
            "eligible_offline_citizen_bootstrap_95_interval"),
        "same_positive_direction": simulated > 0,
        "descriptive_overlap_only": bounds[0] <= simulated <= bounds[1],
        "simulated_over_external_midpoint": simulated / midpoint,
    }


def pair_provenance(manifests: list[dict], arms: tuple[str, str]) -> dict:
    """Carry verified run and prompt identities into the score artifact."""
    prompt_variant = manifests[0].get("prompt_variant")
    system_prompt_sha = manifests[0].get("system_prompt_sha256")
    if (not isinstance(prompt_variant, str) or not prompt_variant
            or not isinstance(system_prompt_sha, str) or len(system_prompt_sha) != 64
            or any(char not in "0123456789abcdef" for char in system_prompt_sha)):
        raise ValueError("paired ledger manifests need a prompt variant and system prompt hash")
    stage2_sha = manifests[0].get("stage2_system_prompt_sha256")
    if prompt_variant == "v53" and (
        not isinstance(stage2_sha, str) or len(stage2_sha) != 64
        or any(char not in "0123456789abcdef" for char in stage2_sha)
        or any(m.get("stage2_system_prompt_sha256") != stage2_sha for m in manifests[1:])
    ):
        raise ValueError("paired v53 ledgers need the same Stage2 system prompt hash")
    provenance = {
        "prompt_variant": prompt_variant,
        "system_prompt_sha256": system_prompt_sha,
        "baseline_income_map_sha256": manifests[0]["baseline_income_map_sha256"],
        "policy_file_sha256": manifests[0].get("policy_file_sha256"),
        "mapping_sha256": manifests[0].get("mapping_sha256"),
        "paired_environment_fingerprint": manifests[0].get(
            "paired_environment_fingerprint"),
        "arms": {arm: {"run_id": manifest["run_id"],
                       "execution_fingerprint": manifest["execution_fingerprint"],
                       "ledger_sha256": manifest["output_sha256"],
                       "unrepaired_choice_trace_pass": manifest.get(
                           "unrepaired_choice_trace_pass"),
                       "choice_repaired_citizen_days": (manifest.get(
                           "generation_totals") or {}).get("stage2_choice_repair_agents")}
                 for arm, manifest in zip(arms, manifests)},
    }
    if stage2_sha is not None:
        provenance["stage2_system_prompt_sha256"] = stage2_sha
    return provenance


def verify_manifests(on_path: Path, off_path: Path, *, roster: list[str],
                     days: list[str], policy_id: str) -> dict:
    expected_roster_sha = hashlib.sha256(
        json.dumps(sorted(roster), ensure_ascii=False).encode("utf-8")).hexdigest()
    manifests = []
    for arm, path in (("on", on_path), ("off", off_path)):
        manifest = json.loads(path.with_name(path.name + ".manifest.json").read_text(
            encoding="utf-8"))
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if (manifest.get("arm") != arm or manifest.get("policy_id") != policy_id
                or manifest.get("start") != days[0] or manifest.get("end") != days[-1]
                or manifest.get("days") != len(days) or manifest.get("citizens") != len(roster)
                or manifest.get("rows") != len(roster) * len(days)
                or manifest.get("roster_sha256") != expected_roster_sha
                or manifest.get("output_sha256") != digest
                or manifest.get("quality_gate_pass") is not True):
            raise ValueError(f"invalid {arm} ledger manifest")
        manifests.append(manifest)
    for key in ("policy_file_sha256", "execution_fingerprint",
                "baseline_income_map_sha256", "prompt_variant",
                "system_prompt_sha256"):
        if not manifests[0].get(key) or manifests[0][key] != manifests[1].get(key):
            raise ValueError(f"paired arms differ in {key}")
    run_ids = [manifest.get("run_id") for manifest in manifests]
    if (any(not isinstance(run_id, str) or not run_id for run_id in run_ids)
            or run_ids[0] == run_ids[1]):
        raise ValueError("paired arms need distinct run IDs")
    return pair_provenance(manifests, ("on", "off"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--effect-start", help="First scored day; full ledger still reconciled")
    parser.add_argument("--effect-end", help="Last scored day; must accompany effect-start")
    parser.add_argument("--expected-recipients", type=int)
    parser.add_argument("--expected-issued-won", type=int)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--reference", type=Path,
                        help="Separate empirical range for a descriptive eligible-sector proxy")
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    roster = roster_file(args.roster)
    days = dates(args.start, args.end)
    if bool(args.effect_start) != bool(args.effect_end):
        parser.error("effect-start and effect-end must be supplied together")
    effect_days = (dates(args.effect_start, args.effect_end)
                   if args.effect_start else None)
    provenance = verify_manifests(args.on, args.off, roster=roster, days=days,
                                  policy_id=args.policy_id)
    result = score(read_jsonl(args.on), read_jsonl(args.off),
                   roster=roster, days=days,
                   policy_id=args.policy_id, draws=args.draws,
                   effect_days=effect_days,
                   expected_recipients=args.expected_recipients,
                   expected_issued_won=args.expected_issued_won)
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
