"""Audit or rebuild an explicitly isolated restored graph for the smoking experiment.

The default is read-only. The source dump is never opened or overwritten here.
Only the reviewed staging copy may be changed, and unknown graph schemas fail
closed. Export the completed staging graph once, then restore that same export
for both experiment arms; this script does not manufacture a snapshot hash.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.experiments.no_smoking_zone import DEFAULT_DAY_ZERO

STATIC_LABELS = {"Agent", "POI", "Category", "District", "Dong"}
DYNAMIC_LABELS = {"Plan", "Memory", "Conversation", "State", "Policy"}
MARKER_LABELS = {"ExperimentSnapshot", "NoSmokingRebuildSource", "NoSmokingBaseline"}
STATIC_RELATIONS = {"HAS_DONG", "ADJACENT_TO", "IN_DONG", "IN_CATEGORY", "LIVES_AT", "WORKS_AT", "KNOWS"}
RELATION_ENDPOINTS = {
    "HAS_DONG": {("District", "Dong")}, "ADJACENT_TO": {("Dong", "Dong")},
    "IN_DONG": {("POI", "Dong")}, "IN_CATEGORY": {("POI", "Category")},
    "LIVES_AT": {("Agent", "POI")}, "WORKS_AT": {("Agent", "POI")},
    "KNOWS": {("Agent", "Agent")}, "KNOWS_POI": {("Agent", "POI")},
    "HAS_PLAN": {("Agent", "Plan")}, "INCLUDES": {("Plan", "POI")},
    "REMEMBERS": {("Agent", "Memory")}, "ABOUT_POI": {("Memory", "POI")},
    "FROM_CONVERSATION": {("Memory", "Conversation")},
    "PARTICIPATES_IN": {("Agent", "Conversation")},
    "MENTIONS_POI": {("Conversation", "POI")},
    "ABOUT_POLICY": {("Conversation", "Policy")},
    "HAS_STATE": {("Agent", "State")},
    "applied_to": {("Policy", "District"), ("Policy", "Dong")},
    # BASE7500H has both casing variants; live audit confirmed Policy -> District.
    "APPLIED_TO": {("Policy", "District")},
    "targets": {("Policy", "Category")},
}
SOURCE_ID = "no_smoking_zone_staging"
BALANCE_DAYS = 39
AWARENESS_PROPERTIES = {"source", "since", "affinity", "visit_count", "avg_satisfaction", "last_visit", "recent_visit_dates"}
DEFAULT_BATCH_SIZE = 500


def batch_size_value(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("batch size must be an integer from 1 to 1000") from None
    if isinstance(value, bool) or str(parsed) != str(value) or not 1 <= parsed <= 1000:
        raise argparse.ArgumentTypeError("batch size must be an integer from 1 to 1000")
    return parsed


def print_progress(event):
    print(json.dumps({"rebuild_progress": event}, ensure_ascii=False), flush=True)


def progress_event(callback, phase, status, **details):
    event = {"at": datetime.now(timezone.utc).isoformat(), "phase": phase, "status": status, **details}
    (callback or print_progress)(event)


def json_default(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Unsupported graph property type: {type(value).__name__}")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False, default=json_default)


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False,
                                    default=json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalized_sex(value):
    return {"M": "male", "남자": "male", "남성": "male", "F": "female",
            "여자": "female", "여성": "female", "male": "male", "female": "female"}.get(value, "all")


def initial_balance(weekday, weekend):
    """The current 08_initial_state.py anchor × 39 days rule, without env overrides."""
    values = []
    for raw in (weekday, weekend):
        if raw is None:
            values.append(0.0)
        elif type(raw) not in (int, float) or not math.isfinite(raw) or raw < 0:
            raise ValueError("Invalid nonnegative finite daily spending anchor")
        else:
            values.append(float(raw))
    weekday, weekend = values
    if weekday <= 0 and weekend <= 0:
        return 1500000
    weekday = weekday or weekend
    weekend = weekend or weekday
    return int(round((weekday * 5 + weekend * 2) / 7 * BALANCE_DAYS))


def state_rows(agents, day_zero):
    date.fromisoformat(day_zero)
    return [{"id": f"{a['id']}_{day_zero}", "agent_id": a["id"], "day": day_zero,
             "balance": initial_balance(a.get("weekday"), a.get("weekend")),
             "energy": 0.8, "yesterday_satisfaction": 0.5, "mood": 0.5,
             "fatigue": 0.3, "month_spent": 0, "sangsaeng_month_spent": 0,
             "policy_lifecycle": "{}"} for a in agents]


def schema_blockers(nodes, relationships):
    blockers = []
    for row in nodes:
        labels = row["labels"]
        if len(labels) != 1 or labels[0] not in STATIC_LABELS | DYNAMIC_LABELS | MARKER_LABELS:
            blockers.append(f"Unreviewed node labels: {labels}")
    for row in relationships:
        endpoints = (tuple(row["source_labels"]), tuple(row["target_labels"]))
        valid = RELATION_ENDPOINTS.get(row["type"], set())
        if len(endpoints[0]) != 1 or len(endpoints[1]) != 1 or (endpoints[0][0], endpoints[1][0]) not in valid:
            blockers.append(f"Unreviewed relationship: {endpoints[0]} -{row['type']}-> {endpoints[1]}")
    return blockers


def cohort_blockers(agents, runtime):
    """Never join persona generations by ID alone, and never overwrite persona data."""
    if runtime is None:
        return [], {"checked": False}
    cohort = runtime.get("cohort")
    if runtime.get("experiment_id") != "no_smoking_zone" or not isinstance(cohort, list) or not cohort:
        return ["Invalid no-smoking runtime cohort"], {"checked": False}
    graph = {a["id"]: a for a in agents}
    ids = [a.get("id") for a in cohort]
    if any(not isinstance(aid, str) or not aid for aid in ids) or len(ids) != len(set(ids)):
        return ["Invalid or duplicate cohort IDs"], {"checked": False}
    missing, mismatch = [], []
    for row in cohort:
        actual = graph.get(row["id"])
        if actual is None:
            missing.append(row["id"])
        elif row.get("age") != actual.get("age") or row.get("sex") != normalized_sex(actual.get("sex")):
            mismatch.append(row["id"])
    issues = []
    if missing:
        issues.append(f"Cohort agents missing from graph: {len(missing)}")
    if mismatch:
        issues.append(f"Cohort age/sex differs from graph: {len(mismatch)}; source generation must be reconciled")
    return issues, {"checked": True, "cohort_size": len(cohort), "missing": len(missing),
                    "age_sex_mismatch": len(mismatch), "mismatch_sample": sorted(mismatch)[:10]}


def audit_graph(session, runtime=None):
    nodes = [dict(r) for r in session.run("MATCH (n) RETURN labels(n) AS labels, count(n) AS count ORDER BY labels")]
    relationships = [dict(r) for r in session.run(
        "MATCH (a)-[r]->(b) RETURN labels(a) AS source_labels, type(r) AS type, "
        "labels(b) AS target_labels, count(r) AS count ORDER BY type, source_labels, target_labels")]
    agents = [dict(r) for r in session.run(
        "MATCH (a:Agent) RETURN a.id AS id, a.personal_age AS age, "
        "coalesce(a.personal_gender,a.p_gender) AS sex, a.s_daily_wd AS weekday, "
        "a.s_daily_we AS weekend ORDER BY a.id")]
    blockers = schema_blockers(nodes, relationships)
    agent_ids = [a["id"] for a in agents]
    if not agents or any(not isinstance(aid, str) or not aid for aid in agent_ids) or len(set(agent_ids)) != len(agent_ids):
        blockers.append("Missing or duplicate graph Agent IDs")
    invalid_anchors = 0
    for agent in agents:
        try:
            initial_balance(agent["weekday"], agent["weekend"])
        except ValueError:
            invalid_anchors += 1
    if invalid_anchors:
        blockers.append(f"Invalid spending anchors: {invalid_anchors}")
    awareness = [dict(r) for r in session.run(
        "MATCH ()-[kp:KNOWS_POI]->() RETURN kp.source AS source, count(kp) AS count ORDER BY source")]
    if any(row["source"] not in {"initial", "visited", "rumor"} for row in awareness):
        blockers.append("Unreviewed KNOWS_POI source; do not guess initial versus learned provenance")
    if not any(r["source"] == "initial" and r["count"] for r in awareness):
        blockers.append("Initial KNOWS_POI pool absent; explicit reviewed re-seeding is required")
    awareness_keys = {r["key"] for r in session.run(
        "MATCH ()-[kp:KNOWS_POI]->() UNWIND keys(kp) AS key RETURN DISTINCT key")}
    if awareness_keys - AWARENESS_PROPERTIES:
        blockers.append(f"Unreviewed KNOWS_POI properties: {sorted(awareness_keys - AWARENESS_PROPERTIES)}")
    cohorts, alignment = cohort_blockers(agents, runtime)
    blockers.extend(cohorts)
    markers = [dict(r["properties"]) for r in session.run(
        "MATCH (s:NoSmokingRebuildSource) RETURN properties(s) AS properties")]
    report = {"nodes": nodes, "relationships": relationships, "awareness_sources": awareness,
              "awareness_property_keys": sorted(awareness_keys),
              "agents": len(agents), "cohort_alignment": alignment, "source_markers": markers,
              "blockers": blockers}
    return report, agents


def validate_apply(args, report, environ):
    if urlsplit(args.uri).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Apply is restricted to a loopback staging endpoint; use an isolated local instance or tunnel")
    if environ.get("NO_SMOKING_REBUILD_ISOLATED") != "1":
        raise ValueError("Apply requires NO_SMOKING_REBUILD_ISOLATED=1 for the isolated staging copy")
    if args.confirm_target != f"{args.uri}|{args.database}":
        raise ValueError("--confirm-target must exactly match URI|database")
    if report["blockers"]:
        raise ValueError("Graph audit blocked: " + "; ".join(report["blockers"]))
    if not args.source_cohort_label or not args.source_cohort_label.strip():
        raise ValueError("Apply requires an explicit --source-cohort-label")
    markers = report["source_markers"]
    if len(markers) != 1:
        raise ValueError("Exactly one NoSmokingRebuildSource staging marker is required")
    marker = markers[0]
    expected = {"id": SOURCE_ID, "source_sha256": args.source_sha256, "isolated": True,
                "intended_day_zero": args.day_zero, "target_uri": args.uri, "target_database": args.database}
    if marker.get("isolated") is not True or any(marker.get(key) != value for key, value in expected.items()):
        raise ValueError("Staging source marker does not match source hash, date or explicit target")
    if marker.get("status") not in (None, "restored"):
        raise ValueError("Staging copy already started/rebuilt; restore the original into a new isolated copy")


def static_fingerprint(session, *, progress=None, phase="static_fingerprint"):
    """Stream an order-independent multiset checksum; no graph-sized client/server sort."""
    result = {}
    queries = {
        "nodes": ("MATCH (n) WHERE any(label IN labels(n) WHERE label IN $names) "
                  "RETURN elementId(n) AS element_id, labels(n) AS labels, properties(n) AS properties",
                  sorted(STATIC_LABELS)),
        "relationships": ("MATCH (a)-[r]->(b) WHERE type(r) IN $names RETURN elementId(r) AS element_id, "
                          "elementId(a) AS source, elementId(b) AS target, type(r) AS type, "
                          "properties(r) AS properties", sorted(STATIC_RELATIONS)),
    }
    for name, (query, names) in queries.items():
        progress_event(progress, phase + "." + name, "started")
        checksum = 0
        square_checksum = 0
        count = 0
        for record in session.run(query, names=names):
            row = dict(record)
            if name == "nodes" and row["labels"] == ["Agent"]:
                row["properties"] = {k: v for k, v in row["properties"].items() if k != "execution_lock"}
            hashed = int.from_bytes(hashlib.sha256(canonical(row).encode("utf-8")).digest(), "big")
            checksum = (checksum + hashed) % (1 << 256)
            square_checksum = (square_checksum + hashed * hashed) % (1 << 256)
            count += 1
            if count % 50000 == 0:
                progress_event(progress, phase + "." + name, "running", rows=count)
        result[name] = {"count": count, "algorithm": "sha256-multiset-sum-and-square-v1",
                        "sum": f"{checksum:064x}", "sum_square": f"{square_checksum:064x}"}
        progress_event(progress, phase + "." + name, "complete", rows=count)
    return result


def run_write(session, query, **params):
    session.run(query, **params).consume()


def run_batched(session, query, phase, batch_size, progress=None, **params):
    """Consume one small result per affected entity so progress remains observable."""
    progress_event(progress, phase, "started", batch_size=batch_size)
    rows = 0
    result = session.run(query, **params)
    for record in result:
        rows += record["touched"]
        if rows % 10000 == 0:
            progress_event(progress, phase, "running", rows=rows, batch_size=batch_size)
    result.consume()
    progress_event(progress, phase, "complete", rows=rows, batch_size=batch_size)
    return rows


def rebuild(session, args, agents, progress=None):
    batch_size = batch_size_value(getattr(args, "batch_size", DEFAULT_BATCH_SIZE))
    before = static_fingerprint(session, progress=progress, phase="static_before")
    progress_event(progress, "mark_in_progress", "started")
    run_write(session, "MATCH (s:NoSmokingRebuildSource {id:$id}) SET s.status='in_progress'", id=SOURCE_ID)
    # A failure after this point leaves in_progress and requires a fresh restore.
    # All dynamic node labels were validated as single labels before any write.
    for label in sorted(DYNAMIC_LABELS | {"ExperimentSnapshot", "NoSmokingBaseline"}):
        run_batched(session, f"MATCH (n:{label}) CALL (n) {{ DETACH DELETE n RETURN 1 AS touched }} "
                    f"IN TRANSACTIONS OF {batch_size} ROWS RETURN touched",
                    "delete_" + label, batch_size, progress)
    run_batched(session, "MATCH (a:Agent) WHERE a.execution_lock IS NOT NULL "
                "CALL (a) { REMOVE a.execution_lock RETURN 1 AS touched } "
                f"IN TRANSACTIONS OF {batch_size} ROWS RETURN touched",
                "reset_agent_execution_lock", batch_size, progress)
    # Keep only original awareness identities; discard all later visits and rumors.
    run_batched(session, "MATCH ()-[kp:KNOWS_POI]->() WHERE kp.source <> 'initial' "
                "CALL (kp) { DELETE kp RETURN 1 AS touched } "
                f"IN TRANSACTIONS OF {batch_size} ROWS RETURN touched",
                "delete_learned_awareness", batch_size, progress)
    run_batched(session, "MATCH ()-[kp:KNOWS_POI]->() CALL (kp) { "
                "SET kp = {source:'initial', since:date($day), affinity:0.5, visit_count:0} "
                "RETURN 1 AS touched } "
                f"IN TRANSACTIONS OF {batch_size} ROWS RETURN touched",
                "reset_initial_awareness", batch_size, progress, day=args.day_zero)
    rows = state_rows(agents, args.day_zero)
    progress_event(progress, "seed_initial_states", "started", batch_size=batch_size)
    for offset in range(0, len(rows), batch_size):
        run_write(session, "UNWIND $rows AS x MATCH (a:Agent {id:x.agent_id}) "
                  "CREATE (s:State) SET s=x, s.day=date(x.day) "
                  "CREATE (a)-[:HAS_STATE {day:date(x.day)}]->(s)", rows=rows[offset:offset + batch_size])
        progress_event(progress, "seed_initial_states", "running", rows=min(offset + batch_size, len(rows)))
    progress_event(progress, "seed_initial_states", "complete", rows=len(rows))
    after = static_fingerprint(session, progress=progress, phase="static_after")
    if before != after:
        raise RuntimeError("Static graph changed during rebuild; reject staging copy and restore source")
    progress_event(progress, "validate_initial_states", "started")
    bad_states = session.run(
        "MATCH (a:Agent) OPTIONAL MATCH (a)-[h:HAS_STATE]->(s:State) "
        "WITH a, count(s) AS n, collect({id:s.agent_id, day:s.day, edge_day:h.day, balance:s.balance}) AS states "
        "WHERE n<>1 OR any(st IN states WHERE st.id IS NULL OR st.id<>a.id "
        "OR st.day IS NULL OR st.day<>date($day) OR st.edge_day IS NULL OR st.edge_day<>date($day) "
        "OR st.balance IS NULL OR st.balance<0) RETURN count(a) AS count",
        day=args.day_zero).single()["count"]
    if bad_states:
        raise RuntimeError("Fresh initial State coverage failed; reject staging copy")
    progress_event(progress, "finalize_baseline", "started")
    created_at = datetime.now(timezone.utc).isoformat()
    run_write(session, "CREATE (b:NoSmokingBaseline {id:'no_smoking_zone'}) "
              "SET b.source_sha256=$source, b.day_zero=date($day), b.source_cohort_label=$cohort, "
              "b.initial_balance_days=$balance_days, b.sangsaeng_seed_enabled=false, "
              "b.created_at=$created, b.static_fingerprint_json=$fingerprint, "
              "b.interpretation='Synthetic historical-policy scenario; not 2017 observed population'",
              source=args.source_sha256, day=args.day_zero, cohort=args.source_cohort_label,
              balance_days=BALANCE_DAYS, created=created_at, fingerprint=canonical(after))
    run_write(session, "MATCH (s:NoSmokingRebuildSource {id:$id}) SET s.status='complete'", id=SOURCE_ID)
    progress_event(progress, "finalize_baseline", "complete")
    return {"static_before": before, "static_after": after, "initial_states": len(rows),
            "day_zero": args.day_zero, "balance_days": BALANCE_DAYS, "batch_size": batch_size, "created_at": created_at}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", required=True, help="Explicit isolated staging Bolt endpoint")
    parser.add_argument("--database", required=True)
    parser.add_argument("--source-sha256", required=True, help="SHA256 of untouched original dump")
    parser.add_argument("--day-zero", default=DEFAULT_DAY_ZERO)
    parser.add_argument("--runtime-manifest", type=Path, help="Optional frozen cohort for age/sex cross-check")
    parser.add_argument("--source-cohort-label", help="Reviewed source population identity; never inferred from IDs")
    parser.add_argument("--no-auth", action="store_true", help="Explicitly use a local isolated Neo4j with authentication disabled")
    parser.add_argument("--batch-size", type=batch_size_value, default=DEFAULT_BATCH_SIZE,
                        help="Entities per write transaction (1..1000, default 500 for small staging heaps)")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Rebuild the already audited isolated staging copy")
    parser.add_argument("--confirm-target", help="Exact URI|database, required for --apply")
    args = parser.parse_args(argv)
    target = urlsplit(args.uri)
    if target.scheme not in {"bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s", "neo4j+ssc"} or not target.hostname or target.username or target.password:
        parser.error("Use an explicit Bolt URI without credentials")
    if not args.database or args.database == "system":
        parser.error("Select the explicit staging data database, never system")
    if not re.fullmatch(r"[0-9a-f]{64}", args.source_sha256):
        parser.error("source-sha256 must be a lowercase 64-character SHA256")
    date.fromisoformat(args.day_zero)
    password = os.environ.get("NEO4J_PASSWORD")
    if args.no_auth and target.hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("--no-auth is allowed only on loopback staging endpoints")
    if not password and not args.no_auth:
        parser.error("Set NEO4J_PASSWORD in the environment")
    runtime = json.loads(args.runtime_manifest.read_text(encoding="utf-8-sig")) if args.runtime_manifest else None
    from neo4j import GraphDatabase
    auth = None if args.no_auth else (os.environ.get("NEO4J_USER", "neo4j"), password)
    with GraphDatabase.driver(args.uri, auth=auth) as driver:
        with driver.session(database=args.database, fetch_size=500) as session:
            report, agents = audit_graph(session, runtime)
            report.update(mode="audit", target_uri=args.uri, target_database=args.database,
                          source_sha256=args.source_sha256, day_zero=args.day_zero)
            write_report(args.report, report)
            if args.apply:
                validate_apply(args, report, os.environ)
                report["mode"] = "apply"
                report["progress"] = []

                def save_progress(event):
                    report["progress"].append(event)
                    report["current_phase"] = event["phase"]
                    write_report(args.report, report)
                    print_progress(event)

                try:
                    report["rebuild"] = rebuild(session, args, agents, progress=save_progress)
                    report["status"] = "complete"
                except Exception as exc:
                    report.update(status="failed", error=str(exc), failed_phase=report.get("current_phase"))
                    write_report(args.report, report)
                    raise
                write_report(args.report, report)
    print(json.dumps({"mode": report["mode"], "status": report.get("status", "audited"),
                      "blockers": report["blockers"], "report": str(args.report)}, ensure_ascii=False))
    return 2 if report["blockers"] else 0


if __name__ == "__main__":
    sys.exit(main())
