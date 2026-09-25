"""Read-only Neo4j export of completed citizen-day spending and grant accounting.

Run once per arm *before* resetting its graph. Requires canonical all-ok daily
metrics and an explicit frozen citizen roster. Output is written atomically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
from neo4j_load._common import driver_session  # noqa: E402
from score_policy import apply_policy_eligibility  # noqa: E402
from paired_grant_effect import dates, read_jsonl, roster_file  # noqa: E402
from report.audit_stage2_generation import choice_status, inspect  # noqa: E402
from report.export_cashback_month import (verify_cohorts,
                                           verify_metric_provenance)  # noqa: E402


STATE_QUERY = """
MATCH (a:Agent)-[:HAS_STATE {day: date($day)}]->(st:State)
WHERE a.id IN $aids
RETURN a.id AS aid, st.online_spent AS online_spent,
       st.month_spent AS self_month_cumulative,
       st.grant_received AS grant_received,
       st.grant_remaining AS grant_remaining
"""

SPEND_QUERY = """
MATCH (a:Agent)-[:HAS_PLAN]->(pl:Plan)-[i:INCLUDES]->(p:POI)
WHERE a.id IN $aids AND toString(pl.day) = $day
OPTIONAL MATCH (p)-[:IN_CATEGORY]->(c:Category)
WITH a, i, p, head(collect(c)) AS c
OPTIONAL MATCH (a)-[:LIVES_AT]->(h:POI)
WITH a, i, p, c, head(collect(h)) AS h
RETURN a.id AS aid, i.actual_spent AS amt,
       i.spent_from_policy AS spent_from_policy,
       p.name AS pname, c.name AS sub, c.parent AS l1,
       p.upjong_l3 AS upjong_l3, p.dong_code AS pdong,
       h.dong_code AS hdong
"""

POLICY_QUERY = "MATCH (p:Policy) RETURN properties(p) AS policy"


def verify_graph_policy(rows: list[dict], arm: str, policy: dict) -> None:
    if arm == "off":
        if rows:
            raise ValueError("control graph contains a Policy node")
        return
    if len(rows) != 1:
        raise ValueError("treatment graph must contain exactly one Policy")
    actual = rows[0]["policy"]
    for key in ("id", "type", "effective_from", "effective_until", "grant_key"):
        if str(actual.get(key)) != str(policy.get(key)):
            raise ValueError(f"graph Policy disagrees with frozen file: {key}")
    if bool(actual.get("poi_restricted")) != bool(policy.get("poi_restricted")):
        raise ValueError("graph Policy disagrees with frozen file: poi_restricted")
    grants = actual.get("decile_grants")
    if isinstance(grants, str):
        grants = json.loads(grants)
    if grants != (policy.get("decile_grants") or {}):
        raise ValueError("graph Policy disagrees with frozen file: decile_grants")


def _policy_amount(value: object, policy_id: str) -> int:
    if value is None or value == "":
        value = {}
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("policy amount must be a JSON object")
    got = value.get(policy_id, 0)
    if isinstance(got, bool) or not isinstance(got, int) or got < 0:
        raise ValueError("policy amount must be a nonnegative integer")
    return got


def verify_metrics(path: Path, roster: list[str], arm: str,
                   policy_id: str | None = None) -> dict[str, dict]:
    rows = read_jsonl(path)
    seen = {}
    for row in rows:
        aid = row.get("aid")
        if aid in seen:
            raise ValueError(f"duplicate metrics row: {path} {aid}")
        seen[aid] = row
    if set(seen) != set(roster):
        raise ValueError(f"metrics citizen roster incomplete: {path}")
    for aid, row in seen.items():
        if row.get("status") != "ok":
            raise ValueError(f"non-ok metrics row: {path} {aid}")
        exposed = row.get("experience_policy_ids")
        if policy_id is not None:
            if not isinstance(exposed, list):
                raise ValueError(f"missing policy exposure evidence: {path} {aid}")
            expected = {policy_id} if arm == "on" else set()
            if set(exposed) != expected:
                raise ValueError(f"wrong policy exposure: {path} {aid}")
        if arm == "on" and row.get("grant_due_but_zero"):
            raise ValueError(f"grant due but not delivered: {path} {aid}")
        if arm == "off" and any(row.get(key) for key in
                                ("policy_hits", "grant_applied_today", "policy_spend_today")):
            raise ValueError(f"policy activity in control metrics: {path} {aid}")
    return seen


def verify_receipt_deltas(rows: list[dict], metrics: dict[str, dict],
                          previous: dict[str, int]) -> None:
    """The first export day starts with zero selected-policy grant in the new graph."""
    for row in rows:
        aid = row["aid"]
        observed = metrics[aid].get("grant_applied_today")
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
            raise ValueError(f"invalid daily grant receipt metric: {aid} {row['day']}")
        receipt = row["grant_received_cumulative"]
        if receipt - previous.get(aid, 0) != observed:
            raise ValueError(f"daily grant receipt disagrees with State: {aid} {row['day']}")
        previous[aid] = receipt


def verify_self_spend_deltas(rows: list[dict],
                             previous: dict[str, tuple[str, int]]) -> None:
    """State.month_spent must equal own offline outflow plus online spending."""
    for row in rows:
        aid, month = row["aid"], row["day"][:7]
        old_month, old_cumulative = previous.get(aid, (month, 0))
        prior = old_cumulative if old_month == month else 0
        expected = row["offline_spent"] - row["grant_spent_today"] + row["online_spent"]
        if row["self_month_cumulative"] - prior != expected:
            raise ValueError(f"State own-spend ledger disagrees with transactions: {aid} {row['day']}")
        previous[aid] = (month, row["self_month_cumulative"])


def aggregate_day(states: list[dict], spends: list[dict], *, roster: list[str],
                  day: str, arm: str, policy_id: str,
                  policy_file: str, restricted: bool = True) -> list[dict]:
    by_aid = {}
    for row in states:
        aid = row.get("aid")
        if aid in by_aid:
            raise ValueError(f"duplicate State for {aid} {day}")
        by_aid[aid] = row
    if set(by_aid) != set(roster):
        raise ValueError(f"incomplete State roster on {day}")
    eligibility_rows = [dict(row) for row in spends]
    if restricted:
        rule = apply_policy_eligibility(eligibility_rows, policy_file)
        if rule.startswith("정책 파일 없음"):
            raise ValueError(rule)
    else:
        for row in eligibility_rows:
            row["elig"] = True
    totals = defaultdict(lambda: [0, 0, 0])
    for row in eligibility_rows:
        aid = row.get("aid")
        if aid not in by_aid:
            raise ValueError(f"transaction for unregistered citizen: {aid} {day}")
        amt = row.get("amt")
        if isinstance(amt, bool) or not isinstance(amt, int) or amt < 0:
            raise ValueError(f"invalid transaction amount: {aid} {day}")
        funded = _policy_amount(row.get("spent_from_policy"), policy_id)
        if funded > amt:
            raise ValueError(f"grant funds exceed transaction gross: {aid} {day}")
        if funded and not row.get("elig"):
            raise ValueError(f"grant payment at ineligible POI: {aid} {day}")
        totals[aid][0] += amt
        totals[aid][1] += amt if row.get("elig") else 0
        totals[aid][2] += funded
    out = []
    for aid in roster:
        state = by_aid[aid]
        online = state.get("online_spent")
        if isinstance(online, bool) or not isinstance(online, int) or online < 0:
            raise ValueError(f"missing or invalid State.online_spent: {aid} {day}")
        self_cumulative = state.get("self_month_cumulative")
        if (isinstance(self_cumulative, bool) or not isinstance(self_cumulative, int)
                or self_cumulative < 0):
            raise ValueError(f"missing or invalid State.month_spent: {aid} {day}")
        out.append({
            "aid": aid, "day": day, "arm": arm, "policy_id": policy_id,
            "offline_spent": totals[aid][0], "online_spent": online,
            "self_month_cumulative": self_cumulative,
            "eligible_offline_spent": totals[aid][1],
            "grant_spent_today": totals[aid][2],
            "grant_received_cumulative": _policy_amount(state.get("grant_received"), policy_id),
            "grant_remaining": _policy_amount(state.get("grant_remaining"), policy_id),
        })
    return out


def export(*, roster: list[str], days: list[str], arm: str, policy_id: str,
           policy_file: str, metrics_dir: Path, out: Path) -> int:
    if (not roster or len(roster) != len(set(roster)) or not days
            or dates(days[0], days[-1]) != days):
        raise ValueError("nonempty unique roster and contiguous days required")
    policy_path = ROOT / policy_file
    if not policy_path.is_file():
        raise ValueError(f"missing policy file: {policy_file}")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("id") != policy_id or policy.get("type") != "grant":
        raise ValueError("policy file is not the selected grant")
    restricted = bool(policy.get("poi_restricted") or policy.get("eligibility"))
    if arm not in ("on", "off"):
        raise ValueError("arm must be on or off")
    # Complete the inexpensive disk gate before querying the graph.
    metrics_by_day = {day: verify_metrics(metrics_dir / f"day_{day}.jsonl",
                                          roster, arm, policy_id)
                      for day in days}
    audit = inspect({day: list(metrics_by_day[day].values()) for day in days},
                    expected_per_day=len(roster))
    if not audit["quality_gate_pass"]:
        raise ValueError(f"Stage2 generation quality gate failed: {audit['totals']}")
    cohort = verify_cohorts(metrics_dir, days, roster)
    verify_metric_provenance({day: list(rows.values()) for day, rows in metrics_by_day.items()},
                             cohort)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp.{os.getpid()}")
    previous_receipts: dict[str, int] = {}
    previous_self_spend: dict[str, tuple[str, int]] = {}
    try:
        with driver_session() as session, tmp.open("w", encoding="utf-8") as stream:
            verify_graph_policy([dict(r) for r in session.run(POLICY_QUERY)], arm, policy)
            for day in days:
                states = [dict(r) for r in session.run(STATE_QUERY, day=day, aids=roster)]
                spends = [dict(r) for r in session.run(SPEND_QUERY, day=day, aids=roster)]
                daily = aggregate_day(states, spends, roster=roster, day=day,
                                      arm=arm, policy_id=policy_id,
                                      policy_file=policy_file,
                                      restricted=restricted)
                verify_receipt_deltas(daily, metrics_by_day[day], previous_receipts)
                verify_self_spend_deltas(daily, previous_self_spend)
                for row in daily:
                    row["s2_choice_status"] = choice_status(metrics_by_day[day][row["aid"]])
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    with out.open("rb") as stream:
        output_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {
        "arm": arm, "policy_id": policy_id, "start": days[0], "end": days[-1],
        "policy_file_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "roster_sha256": hashlib.sha256(
            json.dumps(sorted(roster), ensure_ascii=False).encode("utf-8")).hexdigest(),
        "quality_gate_pass": audit["quality_gate_pass"],
        "unrepaired_choice_trace_pass": audit["unrepaired_choice_trace_pass"],
        "generation_totals": audit["totals"], **cohort,
        "citizens": len(roster), "days": len(days),
        "rows": len(roster) * len(days), "output_sha256": output_sha,
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
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--arm", choices=("on", "off"), required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--policy-file", required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    count = export(roster=roster_file(args.roster), days=dates(args.start, args.end),
                   arm=args.arm, policy_id=args.policy_id,
                   policy_file=args.policy_file, metrics_dir=args.metrics_dir,
                   out=args.out)
    print(f"{args.out}: {count} complete citizen-days")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
