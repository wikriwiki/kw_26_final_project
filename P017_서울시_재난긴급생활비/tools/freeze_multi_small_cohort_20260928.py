"""Freeze a small stratified cohort and policy-independent daily budget.

Call only after restoring the verified pre-pilot graph and before Day0.
The map uses static persona spending anchors and no simulated outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sim.run_simulation import fetch_agents
from scripts.neo4j_load._common import driver_session


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bsha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--citizens", type=int, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    assert 1 <= args.citizens <= 80
    assert (args.backup / "neo4j.dump").is_file()
    ids = fetch_agents(limit=args.citizens)
    assert len(ids) == args.citizens and len(set(ids)) == args.citizens
    with driver_session() as session:
        query = ("MATCH (a:Agent) WHERE a.id IN $ids "
                 "RETURN a.id AS id, a.s_daily_wd AS wd, a.s_daily_we AS we ORDER BY a.id")
        first = [dict(r) for r in session.run(query, ids=ids)]
        second = [dict(r) for r in session.run(query, ids=ids)]
    assert first == second and len(first) == args.citizens
    budgets = {}
    both_missing = 0
    for row in first:
        wd = float(row["wd"]) if row["wd"] is not None and float(row["wd"]) > 0 else None
        we = float(row["we"]) if row["we"] is not None and float(row["we"]) > 0 else None
        if wd is None and we is None:
            value = round(1500000 / 39)
            both_missing += 1
        else:
            wd = wd if wd is not None else we
            we = we if we is not None else wd
            value = round((5 * wd + 2 * we) / 7)
        assert value > 0
        budgets[row["id"]] = value
    data = {
        "schema": "fixed_persona_budget_v1",
        "source_kind": "stable_persona_spending_anchors",
        "source_field_stability_verified": True,
        "policy_outcome_used": False,
        "source_fields": ["s_daily_wd", "s_daily_we"],
        "formula": "round((5*s_daily_wd + 2*s_daily_we)/7)",
        "source_archive_sha256": digest(args.backup / "neo4j.dump"),
        "confirmation_archive_sha256": digest(args.backup / "SHA256SUMS"),
        "source_roster_sha256": bsha(ids),
        "source_agent_projection_sha256": bsha(first),
        "citizen_count": len(ids),
        "total_daily_budget_won": sum(budgets.values()),
        "daily_income_by_aid": budgets,
        "normalization_note": "One missing anchor uses the other; both missing use policy-independent 1,500,000/39 won per day.",
        "both_missing_anchor_count": both_missing,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    files = {"roster.json": json.dumps(ids, ensure_ascii=False, indent=2) + "\n",
             "frozen_income.json": json.dumps(data, ensure_ascii=False, indent=2) + "\n"}
    for name, body in files.items():
        dest = args.out / name
        if dest.exists():
            assert dest.read_text(encoding="utf-8") == body, f"frozen {name} changed"
        else:
            dest.write_text(body, encoding="utf-8")
    print(f"frozen {len(ids)} citizens; map_sha256={digest(args.out / 'frozen_income.json')}")


if __name__ == "__main__":
    main()
