"""Export matched-date distancing arms from realized Neo4j transactions.

The government restriction is supplied through SIM_ENVIRONMENT; both arms
must have no Policy node. Sector names come from the actual POI, never from
the citizen's intended activity label. The file is an analysis ledger, not
an external year-over-year card-sales estimate.
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
from neo4j_load._common import driver_session  # noqa: E402
from report.audit_stage2_generation import inspect  # noqa: E402
from report.export_cashback_month import (verify_cohorts,
                                           verify_metric_provenance)  # noqa: E402
from report.paired_grant_effect import dates, read_jsonl, roster_file  # noqa: E402


ENVIRONMENTS = {"restricted": "covid_2021", "control": "covid_no_distancing"}
DEFAULT_MAPPING = ROOT / "data/neo4j_load/mapping/mapping_upjong_to_sub.json"

STATE_QUERY = """
MATCH (a:Agent)-[:HAS_STATE {day: date($day)}]->(st:State)
WHERE a.id IN $aids
RETURN a.id AS aid, st.online_spent AS online_spent,
       st.month_spent AS self_month_cumulative
"""
SPEND_QUERY = """
MATCH (a:Agent)-[:HAS_PLAN {day: date($day)}]->(:Plan)-[i:INCLUDES]->(p:POI)
WHERE a.id IN $aids
OPTIONAL MATCH (p)-[:IN_CATEGORY]->(c:Category)
WITH a, i, p, collect(DISTINCT c.name) AS category_names,
     collect(DISTINCT c.parent) AS category_parents
RETURN a.id AS aid, i.actual_spent AS amount,
       i.category AS intended_category, i.sub_category AS intended_subcategory,
       p.upjong_l3 AS upjong_l3,
       category_names, category_parents
"""
POLICY_QUERY = "MATCH (p:Policy) RETURN count(p) AS n"

MONEY_FIELDS = ("offline_spent", "online_spent", "self_month_cumulative",
                "restaurant_won", "korean_restaurant_won", "retail_won",
                "cafe_won", "unclassified_won", "classified_by_code_won",
                "classified_by_category_won")


def _money(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def classify_poi(row: dict, mapping: dict) -> tuple[str | None, str | None, str, bool]:
    """Return actual POI category, subcategory, source and retail marker."""
    code = row.get("upjong_l3")
    mapped = mapping.get(code) if code else None
    names = [v for v in (row.get("category_names") or []) if v is not None]
    parents = [v for v in (row.get("category_parents") or []) if v is not None]
    if len(set(names)) > 1 or len(set(parents)) > 1:
        raise ValueError(f"ambiguous POI Category for code {code!r}")
    graph_cat = parents[0] if parents else None
    graph_sub = names[0] if names else None
    if mapped:
        cat, sub = mapped.get("cat"), mapped.get("sub")
        if graph_cat and (graph_cat, graph_sub) != (cat, sub):
            raise ValueError(f"POI code/Category conflict: {code}")
        return cat, sub, "code", (mapped.get("_industry_l1") == "소매" or
                                   cat in ("쇼핑", "마트", "편의점") or sub == "약국")
    if graph_cat and graph_sub:
        return graph_cat, graph_sub, "category", (graph_cat in
            ("쇼핑", "마트", "편의점") or graph_sub == "약국")
    return None, None, "unclassified", False


def aggregate_day(states: list[dict], spends: list[dict], *, roster: list[str],
                  day: str, arm: str, mapping: dict,
                  previous: dict[str, tuple[str, int]]) -> list[dict]:
    state_by_aid = {}
    for row in states:
        aid = row.get("aid")
        if aid in state_by_aid:
            raise ValueError(f"duplicate State: {aid} {day}")
        state_by_aid[aid] = row
    if set(state_by_aid) != set(roster):
        raise ValueError(f"incomplete State roster: {day}")
    totals = defaultdict(lambda: defaultdict(int))
    for row in spends:
        aid = row.get("aid")
        if aid not in state_by_aid:
            raise ValueError(f"transaction outside roster: {aid} {day}")
        amount = _money(row.get("amount"), f"transaction {aid} {day}")
        cat, sub, source, retail = classify_poi(row, mapping)
        values = totals[aid]
        values["offline_spent"] += amount
        values[f"classified_by_{source}_won" if source != "unclassified"
               else "unclassified_won"] += amount
        if cat == "식사":
            values["restaurant_won"] += amount
            if sub == "한식":
                values["korean_restaurant_won"] += amount
        if cat == "카페":
            values["cafe_won"] += amount
        if retail:
            values["retail_won"] += amount
    output = []
    month = day[:7]
    for aid in roster:
        state = state_by_aid[aid]
        online = _money(state.get("online_spent"), f"online_spent {aid} {day}")
        cumulative = _money(state.get("self_month_cumulative"),
                            f"month_spent {aid} {day}")
        old_month, old_cumulative = previous.get(aid, (month, 0))
        prior = old_cumulative if old_month == month else 0
        if cumulative - prior != totals[aid]["offline_spent"] + online:
            raise ValueError(f"State spending disagrees with realized ledger: {aid} {day}")
        previous[aid] = (month, cumulative)
        record = {"aid": aid, "day": day, "arm": arm}
        record.update({key: (online if key == "online_spent" else cumulative
                             if key == "self_month_cumulative" else totals[aid][key])
                       for key in MONEY_FIELDS})
        if (record["classified_by_code_won"] + record["classified_by_category_won"]
                + record["unclassified_won"] != record["offline_spent"]):
            raise ValueError(f"classification coverage does not reconcile: {aid} {day}")
        output.append(record)
    return output


def verify_metrics(metrics_dir: Path, days: list[str], roster: list[str],
                   environment_id: str) -> tuple[dict, dict[str, list[dict]]]:
    daily = {}
    for day in days:
        rows = read_jsonl(metrics_dir / f"day_{day}.jsonl")
        aids = [row.get("aid") for row in rows]
        if len(aids) != len(roster) or len(set(aids)) != len(aids) or set(aids) != set(roster):
            raise ValueError(f"incomplete or duplicate metrics roster: {day}")
        for row in rows:
            if row.get("experience_environment_id") != environment_id:
                raise ValueError(f"wrong environment exposure: {row.get('aid')} {day}")
            if row.get("experience_policy_ids") != [] or any(row.get(key) for key in
                ("policy_hits", "grant_applied_today", "policy_spend_today",
                 "instant_discount_today")):
                raise ValueError(f"policy funding in distancing arm: {row.get('aid')} {day}")
        daily[day] = rows
    audit = inspect(daily, expected_per_day=len(roster))
    if not audit["quality_gate_pass"]:
        raise ValueError(f"Stage2/metrics quality gate failed: {audit['totals']}")
    return audit, daily


def verify_environment_cohorts(metrics_dir: Path, days: list[str], roster: list[str],
                               environment_id: str) -> dict:
    cohort = verify_cohorts(metrics_dir, days, roster)
    paired = set()
    for day in days:
        item = json.loads((metrics_dir.parent / f"cohort_{day}.json").read_text(
            encoding="utf-8"))
        if item.get("environment_id") != environment_id:
            raise ValueError(f"cohort environment differs from requested arm: {day}")
        value = item.get("paired_environment_fingerprint")
        if not isinstance(value, str) or not value:
            raise ValueError(f"paired environment fingerprint missing: {day}")
        paired.add(value)
    if len(paired) != 1:
        raise ValueError("paired environment fingerprint changed within arm")
    return {**cohort, "environment_id": environment_id,
            "paired_environment_fingerprint": next(iter(paired))}


def export(*, roster: list[str], days: list[str], arm: str, metrics_dir: Path,
           mapping_path: Path, out: Path) -> int:
    if (arm not in ENVIRONMENTS or not roster or len(roster) != len(set(roster))
            or not days or dates(days[0], days[-1]) != days):
        raise ValueError("invalid arm, roster or date range")
    environment_id = ENVIRONMENTS[arm]
    mapping_bytes = mapping_path.read_bytes()
    mapping = json.loads(mapping_bytes)
    audit, daily_metrics = verify_metrics(metrics_dir, days, roster, environment_id)
    cohort = verify_environment_cohorts(metrics_dir, days, roster, environment_id)
    verify_metric_provenance(daily_metrics, cohort)
    for day, rows in daily_metrics.items():
        if any(row.get("paired_environment_fingerprint") !=
               cohort["paired_environment_fingerprint"] for row in rows):
            raise ValueError(f"metrics paired environment provenance differs from cohort: {day}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp.{os.getpid()}")
    previous: dict[str, tuple[str, int]] = {}
    try:
        with driver_session() as session, tmp.open("w", encoding="utf-8") as stream:
            if session.run(POLICY_QUERY).single()["n"] != 0:
                raise ValueError("distancing graph contains a Policy node")
            for day in days:
                states = [dict(row) for row in session.run(STATE_QUERY, day=day, aids=roster)]
                spends = [dict(row) for row in session.run(SPEND_QUERY, day=day, aids=roster)]
                for row in aggregate_day(states, spends, roster=roster, day=day,
                                         arm=arm, mapping=mapping, previous=previous):
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    with out.open("rb") as stream:
        output_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {
        "arm": arm, "start": days[0], "end": days[-1], "days": len(days),
        "citizens": len(roster), "rows": len(roster) * len(days),
        "roster_sha256": hashlib.sha256(json.dumps(sorted(roster), ensure_ascii=False).encode(
            "utf-8")).hexdigest(),
        "mapping_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
        "output_sha256": output_sha, "quality_gate_pass": audit["quality_gate_pass"],
        "generation_totals": audit["totals"], **cohort,
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
    parser.add_argument("--arm", choices=tuple(ENVIRONMENTS), required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    count = export(roster=roster_file(args.roster), days=dates(args.start, args.end),
                   arm=args.arm, metrics_dir=args.metrics_dir,
                   mapping_path=args.mapping, out=args.out)
    print(f"{args.out}: {count} complete citizen-days")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
