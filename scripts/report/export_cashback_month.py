"""Export a complete cashback month for paired policy ON/OFF scoring.

Reads the actual State and INCLUDES ledger before an arm's graph is reset.
No empirical answer key is used here; policy facts come from its frozen JSON.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from neo4j_load._common import driver_session  # noqa: E402
from report.audit_stage2_generation import choice_status, inspect, load_sources  # noqa: E402
from report.paired_grant_effect import roster_file  # noqa: E402


AGENT_QUERY = """
MATCH (a:Agent) WHERE a.id IN $aids
RETURN a.id AS aid, a.s_daily_wd AS daily_wd,
       a.s_daily_we AS daily_we,
       a.sangsaeng_base_daily AS sangsaeng_base_daily
"""
STATE_QUERY = """
MATCH (a:Agent)-[:HAS_STATE {day: date($day)}]->(st:State)
WHERE a.id IN $aids
RETURN a.id AS aid, st.sangsaeng_month_spent AS eligible_cumulative,
       st.month_spent AS self_month_cumulative,
       st.online_spent AS online_spent
"""
SPEND_QUERY = """
MATCH (a:Agent)-[:HAS_PLAN {day: date($day)}]->(:Plan)-[i:INCLUDES]->(p:POI)
WHERE a.id IN $aids
RETURN a.id AS aid, i.actual_spent AS spent,
       p.sangsaeng_eligible AS eligible
"""
POLICY_QUERY = "MATCH (p:Policy) RETURN properties(p) AS policy"


def month_days(month: str) -> list[str]:
    first = date.fromisoformat(month + "-01")
    return [first.replace(day=d).isoformat()
            for d in range(1, calendar.monthrange(first.year, first.month)[1] + 1)]


def _money(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def load_policy(path: Path, month: str, policy_id: str) -> dict:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("id") != policy_id or policy.get("type") != "cashback":
        raise ValueError("selected policy file is not the cashback policy")
    days = month_days(month)
    effective_from = policy.get("effective_from")
    effective_until = policy.get("effective_until")
    if (not isinstance(effective_from, str) or not isinstance(effective_until, str)
            or effective_from > days[0] or effective_until < days[-1]):
        raise ValueError("cashback policy does not cover the complete calendar month")
    rate = policy.get("benefit_rate")
    threshold_ratio = policy.get("threshold_ratio")
    cap = policy.get("cap_per_agent")
    if (not isinstance(rate, (int, float)) or isinstance(rate, bool) or rate <= 0 or rate > 1
            or not isinstance(threshold_ratio, (int, float)) or isinstance(threshold_ratio, bool)
            or threshold_ratio <= 0 or not isinstance(cap, int) or isinstance(cap, bool)
            or cap <= 0):
        raise ValueError("invalid cashback rate, threshold or cap in policy file")
    return policy


def monthly_anchor(agent: dict, base_ratio: float) -> tuple[int, str]:
    measured = agent.get("sangsaeng_base_daily")
    if measured is not None:
        try:
            value = float(measured)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return round(value * 30), "measured_eligible_daily"
    wd = float(agent.get("daily_wd") or 0)
    we = float(agent.get("daily_we") or wd)
    if wd <= 0:
        wd = we
    if wd <= 0 or we <= 0:
        raise ValueError(f"citizen has no positive daily spending anchor: {agent.get('aid')}")
    return round((wd * 5 + we * 2) / 7 * 30 * base_ratio), "total_daily_scaled"


def check_graph_policy(rows: list[dict], arm: str, policy: dict) -> None:
    if arm == "off":
        if rows:
            raise ValueError("control graph contains a Policy node")
        return
    if len(rows) != 1:
        raise ValueError("policy graph must contain exactly one selected Policy")
    actual = rows[0]["policy"]
    for key in ("id", "type", "effective_from", "effective_until",
                "benefit_rate", "threshold_ratio", "cap_per_agent"):
        if str(actual.get(key)) != str(policy.get(key)):
            raise ValueError(f"graph Policy disagrees with frozen file: {key}")


def aggregate_day(states: list[dict], spends: list[dict], roster: list[str],
                  day: str, previous: dict[str, tuple[int, int]]) -> list[dict]:
    state_by_aid = {}
    for row in states:
        aid = row.get("aid")
        if aid in state_by_aid:
            raise ValueError(f"duplicate State: {aid} {day}")
        state_by_aid[aid] = row
    if set(state_by_aid) != set(roster):
        raise ValueError(f"incomplete State roster: {day}")
    totals = defaultdict(lambda: [0, 0])
    for row in spends:
        aid = row.get("aid")
        if aid not in state_by_aid:
            raise ValueError(f"transaction outside roster: {aid} {day}")
        amount = _money(row.get("spent"), f"transaction amount {aid} {day}")
        if amount and row.get("eligible") is None:
            raise ValueError(f"positive transaction missing eligibility: {aid} {day}")
        totals[aid][0] += amount
        if row.get("eligible") is True:
            totals[aid][1] += amount
    result = []
    for aid in roster:
        state = state_by_aid[aid]
        eligible = _money(state.get("eligible_cumulative"), f"eligible cumulative {aid} {day}")
        self_spent = _money(state.get("self_month_cumulative"),
                            f"self month cumulative {aid} {day}")
        online = _money(state.get("online_spent"), f"online amount {aid} {day}")
        prev_eligible, prev_self = previous.get(aid, (0, 0))
        if eligible - prev_eligible != totals[aid][1]:
            raise ValueError(f"eligible ledger disagrees with State: {aid} {day}")
        if self_spent - prev_self != totals[aid][0] + online:
            raise ValueError(f"total month ledger disagrees with State: {aid} {day}")
        previous[aid] = (eligible, self_spent)
        result.append({"aid": aid, "day": day, "offline_spent": totals[aid][0],
                       "online_spent": online, "eligible_spent": totals[aid][1],
                       "eligible_cumulative": eligible,
                       "self_month_cumulative": self_spent})
    return result


def verify_cohorts(metrics_dir: Path, days: list[str], roster: list[str]) -> dict:
    output_dir = metrics_dir.parent
    fingerprints = set()
    income_maps = set()
    run_ids = set()
    prompt_variants = set()
    system_prompt_hashes = set()
    for day in days:
        path = output_dir / f"cohort_{day}.json"
        if not path.is_file():
            raise ValueError(f"missing execution cohort: {path}")
        cohort = json.loads(path.read_text(encoding="utf-8"))
        agent_ids = cohort.get("agent_ids")
        if (not isinstance(agent_ids, list) or len(agent_ids) != len(roster)
                or set(agent_ids) != set(roster)):
            raise ValueError(f"execution cohort differs from frozen roster: {day}")
        fingerprint = cohort.get("execution_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError(f"execution fingerprint missing: {day}")
        fingerprints.add(fingerprint)
        income_map = cohort.get("baseline_income_map_sha256")
        if not isinstance(income_map, str) or not income_map:
            raise ValueError(f"frozen baseline income map missing: {day}")
        income_maps.add(income_map)
        run_id = cohort.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"run ID missing: {day}")
        run_ids.add(run_id)
        prompt_variant = cohort.get("prompt_variant")
        system_prompt_sha = cohort.get("system_prompt_sha256")
        if (not isinstance(prompt_variant, str) or not prompt_variant
                or not isinstance(system_prompt_sha, str) or len(system_prompt_sha) != 64
                or any(char not in "0123456789abcdef" for char in system_prompt_sha)):
            raise ValueError(f"prompt variant or system prompt hash missing: {day}")
        prompt_variants.add(prompt_variant)
        system_prompt_hashes.add(system_prompt_sha)
    if (len(fingerprints) != 1 or len(income_maps) != 1 or len(run_ids) != 1
            or len(prompt_variants) != 1 or len(system_prompt_hashes) != 1):
        raise ValueError("run ID, execution fingerprint, income map or prompt changed within run")
    return {"execution_fingerprint": next(iter(fingerprints)),
            "baseline_income_map_sha256": next(iter(income_maps)),
            "run_id": next(iter(run_ids)),
            "prompt_variant": next(iter(prompt_variants)),
            "system_prompt_sha256": next(iter(system_prompt_hashes))}


def verify_metric_provenance(daily_metrics: dict[str, list[dict]], cohort: dict) -> None:
    for day, rows in daily_metrics.items():
        for row in rows:
            if (row.get("execution_fingerprint") != cohort["execution_fingerprint"]
                    or row.get("experience_run_id") != cohort["run_id"]):
                raise ValueError(f"metrics execution provenance differs from cohort: "
                                 f"{row.get('aid')} {day}")


def export(*, month: str, arm: str, policy_id: str, policy_file: Path,
           base_ratio: float, roster: list[str], metrics_dir: Path, out: Path) -> int:
    if arm not in ("on", "off") or not 0 < base_ratio <= 1:
        raise ValueError("invalid arm or eligible base ratio")
    days = month_days(month)
    policy = load_policy(policy_file, month, policy_id)
    source = load_sources(metrics_dir, [])
    if not set(days).issubset(source):
        raise ValueError("missing calendar-month metrics days")
    daily_metrics = {day: source[day] for day in days}
    audit = inspect(daily_metrics, expected_per_day=len(roster))
    if not audit["quality_gate_pass"]:
        raise ValueError(f"Stage2/month metrics quality gate failed: {audit['totals']}")
    for day, rows in daily_metrics.items():
        if {row["aid"] for row in rows} != set(roster):
            raise ValueError(f"metrics roster differs from frozen roster: {day}")
        for row in rows:
            if arm == "off" and any(row.get(k) for k in
                                    ("policy_hits", "grant_applied_today", "policy_spend_today")):
                raise ValueError(f"policy activity in control metrics: {row['aid']} {day}")
    cohort = verify_cohorts(metrics_dir, days, roster)
    verify_metric_provenance(daily_metrics, cohort)
    choice_by_day = {day: {metric["aid"]: choice_status(metric) for metric in rows}
                     for day, rows in daily_metrics.items()}

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp.{os.getpid()}")
    try:
        with driver_session() as session, tmp.open("w", encoding="utf-8") as stream:
            policies = [dict(row) for row in session.run(POLICY_QUERY)]
            check_graph_policy(policies, arm, policy)
            agents = {}
            for row in session.run(AGENT_QUERY, aids=roster):
                aid = row["aid"]
                if aid in agents:
                    raise ValueError(f"duplicate Agent ID in graph: {aid}")
                agents[aid] = dict(row)
            if set(agents) != set(roster):
                raise ValueError("Agent graph roster differs from frozen roster")
            anchors = {aid: monthly_anchor(agents[aid], base_ratio) for aid in roster}
            previous: dict[str, tuple[int, int]] = {}
            for day in days:
                states = [dict(row) for row in session.run(STATE_QUERY, aids=roster, day=day)]
                spends = [dict(row) for row in session.run(SPEND_QUERY, aids=roster, day=day)]
                records = aggregate_day(states, spends, roster, day, previous)
                for row in records:
                    row["s2_choice_status"] = choice_by_day[day][row["aid"]]
                    anchor, source_name = anchors[row["aid"]]
                    threshold = round(anchor * float(policy["threshold_ratio"]))
                    payout = (min(int(policy["cap_per_agent"]),
                                  max(0.0, (row["eligible_cumulative"] - threshold)
                                      * float(policy["benefit_rate"])))
                              if arm == "on" and day == days[-1] else 0.0)
                    row.update({"arm": arm, "policy_id": policy_id, "month": month,
                                "anchor_won": anchor, "anchor_source": source_name,
                                "threshold_won": threshold,
                                "cashback_accrued_won": payout,
                                "cashback_cap_reached": bool(payout >= policy["cap_per_agent"])
                                if day == days[-1] and arm == "on" else False})
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    with out.open("rb") as stream:
        output_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {
        "month": month, "arm": arm, "policy_id": policy_id,
        "policy_file": str(policy_file),
        "policy_file_sha256": hashlib.sha256(policy_file.read_bytes()).hexdigest(),
        "base_ratio": base_ratio,
        "roster_sha256": hashlib.sha256(
            json.dumps(sorted(roster), ensure_ascii=False).encode("utf-8")).hexdigest(),
        "quality_gate_pass": audit["quality_gate_pass"],
        "unrepaired_choice_trace_pass": audit["unrepaired_choice_trace_pass"],
        "generation_totals": audit["totals"],
        **cohort,
        "citizens": len(roster), "days": len(days), "rows": len(roster) * len(days),
        "output_sha256": output_sha,
    }
    manifest_path = out.with_name(out.name + ".manifest.json")
    manifest_tmp = manifest_path.with_name(manifest_path.name + f".tmp.{os.getpid()}")
    try:
        manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
        manifest_tmp.replace(manifest_path)
    finally:
        manifest_tmp.unlink(missing_ok=True)
    return len(roster) * len(days)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", required=True, help="Calendar month YYYY-MM")
    parser.add_argument("--arm", choices=("on", "off"), required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--policy-file", type=Path, required=True)
    parser.add_argument("--base-ratio", type=float, required=True,
                        help="Frozen EXP_SANGSAENG_BASE_RATIO from the run manifest")
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    count = export(month=args.month, arm=args.arm, policy_id=args.policy_id,
                   policy_file=args.policy_file, base_ratio=args.base_ratio,
                   roster=roster_file(args.roster), metrics_dir=args.metrics_dir,
                   out=args.out)
    print(f"{args.out}: {count} complete citizen-days")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
