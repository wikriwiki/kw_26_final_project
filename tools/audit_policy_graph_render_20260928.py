"""Read-only pre-call audit of the policy facts actually rendered from Neo4j.

This checks the live graph path, rather than policy_preflight's synthetic row.
It never generates behavior, edits the graph, or reads empirical outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date
from pathlib import Path

from scripts.neo4j_load._common import driver_session
from scripts.sim.dawn_context import (
    POLICY_CYPHER,
    _format_policy_facts,
    _format_policy_status,
    _with_params,
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-file", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--today", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads(args.policy_file.read_text(encoding="utf-8"))
    roster = json.loads(args.roster.read_text(encoding="utf-8"))
    if not roster or len(set(roster)) != len(roster):
        raise ValueError("roster must be nonempty and unique")
    if not (policy["effective_from"] <= args.today <= policy["effective_until"]):
        raise ValueError("audit date is outside policy effective window")
    today = date.fromisoformat(args.today)

    example_facts = example_status = None
    with driver_session() as session:
        for aid in roster:
            rows = [dict(record) for record in session.run(
                POLICY_CYPHER, aid=aid, today=today
            )]
            if len(rows) != 1 or rows[0].get("id") != policy["id"]:
                raise ValueError(f"live policy exposure is not exactly one: {aid}")
            row = rows[0]
            enriched = _with_params(row)
            for key in ("id", "type", "description"):
                if enriched.get(key) != policy.get(key):
                    raise ValueError(f"live graph policy differs at {key}: {aid}")
            for key in ("eligibility", "sectors", "discount_rate", "use_scope"):
                if key in policy and enriched.get(key) != policy[key]:
                    raise ValueError(f"live mechanism differs at {key}: {aid}")
            facts = _format_policy_facts(rows)
            status = _format_policy_status(rows)
            if not facts or not status or policy["description"] not in facts:
                raise ValueError(f"policy description absent from dawn facts: {aid}")
            if policy["id"] not in facts or policy["id"] not in status:
                raise ValueError(f"policy ID absent from rendered context: {aid}")
            if "적용 업종 — 없음" in status:
                raise ValueError(f"mechanism sectors disappeared in dawn status: {aid}")
            if policy["type"] == "sector_voucher":
                sectors = policy.get("sectors") or {}
                if not sectors:
                    raise ValueError("sector voucher has no sectors in frozen file")
                for name, spec in sectors.items():
                    if name not in facts or name not in status:
                        raise ValueError(f"sector name absent from dawn: {name} {aid}")
                    if spec.get("mode") == "rate":
                        rate = f"{float(spec['rate'])*100:.0f}%"
                        cap = f"{int(spec['cap']):,}원"
                        if rate not in facts or cap not in facts or cap not in status:
                            raise ValueError(f"discount terms absent from dawn: {aid}")
                    if "결제할 때 바로" not in facts:
                        raise ValueError(f"payment timing absent from dawn: {aid}")
            if policy["type"] == "price_discount":
                rate = f"{float(policy['discount_rate'])*100:.0f}%"
                cap = f"{int(policy['purchase_cap_monthly']):,}원"
                if (rate not in status or cap not in status
                        or "사는 자치구 안" not in status
                        or "본인 돈" not in status):
                    raise ValueError(f"discount purchase facts absent from dawn: {aid}")
            if example_facts is None:
                example_facts, example_status = facts, status

    result = {
        "schema": "live_policy_dawn_render_audit_v1",
        "policy_id": policy["id"],
        "audit_date": args.today,
        "citizens_checked": len(roster),
        "policy_file_sha256": digest(args.policy_file),
        "roster_sha256": digest(args.roster),
        "facts": example_facts,
        "status": example_status,
        "model_calls": 0,
        "audit_graph_mutations": 0,
        "pass": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        temporary.replace(args.out)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"PASS {policy['id']}: live dawn facts/status for {len(roster)} citizens")


if __name__ == "__main__":
    main()
