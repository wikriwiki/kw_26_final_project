"""Read-only Neo4j export of completed citizen-day spending and grant accounting.

Run once per arm *before* resetting its graph. Requires canonical all-ok daily
metrics and an explicit frozen citizen roster. Output is written atomically.
"""
from __future__ import annotations

import argparse
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


STATE_QUERY = """
MATCH (a:Agent)-[:HAS_STATE {day: date($day)}]->(st:State)
WHERE a.id IN $aids
RETURN a.id AS aid, st.online_spent AS online_spent,
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
RETURN a.id AS aid, coalesce(i.actual_spent, 0) AS amt,
       i.spent_from_policy AS spent_from_policy,
       p.name AS pname, c.name AS sub, c.parent AS l1,
       p.upjong_l3 AS upjong_l3, p.dong_code AS pdong,
       h.dong_code AS hdong
"""


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
        if exposed is not None and policy_id is not None:
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
        out.append({
            "aid": aid, "day": day, "arm": arm, "policy_id": policy_id,
            "offline_spent": totals[aid][0], "online_spent": online,
            "eligible_offline_spent": totals[aid][1],
            "grant_spent_today": totals[aid][2],
            "grant_received_cumulative": _policy_amount(state.get("grant_received"), policy_id),
            "grant_remaining": _policy_amount(state.get("grant_remaining"), policy_id),
        })
    return out


def export(*, roster: list[str], days: list[str], arm: str, policy_id: str,
           policy_file: str, metrics_dir: Path, out: Path) -> int:
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
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp.{os.getpid()}")
    previous_receipts: dict[str, int] = {}
    try:
        with driver_session() as session, tmp.open("w", encoding="utf-8") as stream:
            for day in days:
                states = [dict(r) for r in session.run(STATE_QUERY, day=day, aids=roster)]
                spends = [dict(r) for r in session.run(SPEND_QUERY, day=day, aids=roster)]
                daily = aggregate_day(states, spends, roster=roster, day=day,
                                      arm=arm, policy_id=policy_id,
                                      policy_file=policy_file,
                                      restricted=restricted)
                verify_receipt_deltas(daily, metrics_by_day[day], previous_receipts)
                for row in daily:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
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
