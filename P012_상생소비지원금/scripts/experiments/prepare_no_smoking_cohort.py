"""Preserve full-population smoking assignment while selecting a usable graph roster.

Input graph JSON is a read-only export of properties(a) AS a and home_ok.
Graph demographics and calibrated consumption anchors take precedence over the
older persona archive. No missing values are imputed into the graph.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile


def read(path):
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            return json.loads(archive.read("agents_final.json"))
    return json.loads(path.read_text(encoding="utf-8-sig"))


def prepare(graph_rows, raw_agents, source_sha256):
    originals = {a["agent_id"]: a for a in raw_agents}
    if len(originals) != len(raw_agents):
        raise ValueError("Duplicate original persona ID")
    if len({r["a"]["id"] for r in graph_rows}) != len(graph_rows):
        raise ValueError("Duplicate graph Agent ID")
    personas, eligible, excluded = [], [], []
    changes = Counter()
    for row in sorted(graph_rows, key=lambda r: r["a"]["id"]):
        p = row["a"]
        # Keep every original field for provenance, then explicitly use the graph's inputs.
        a = json.loads(json.dumps(originals[p["id"]]))
        personal = a.setdefault("personal", {})
        spending = a.setdefault("spending", {})
        for field, key in (("age", "personal_age"), ("gender", "personal_gender"), ("income_level", "p_income_level")):
            if personal.get(field, a.get(field)) != p.get(key):
                changes[field] += 1
            personal[field] = p.get(key)
        for field, key in (("daily_spending_weekday", "s_daily_wd"), ("daily_spending_weekend", "s_daily_we"),
                           ("weekday_spending_level", "spending_level_wd"), ("weekend_spending_level", "spending_level_we")):
            if spending.get(field) != p.get(key):
                changes[field] += 1
            spending[field] = p.get(key)
        a["source_graph_properties"] = p
        a["graph_source_sha256"] = source_sha256
        personas.append(a)
        reasons = []
        if row.get("home_ok") is not True:
            reasons.append("missing_home_to_dong_anchor")
        anchors = [p.get("s_daily_wd"), p.get("s_daily_we")]
        if not any(isinstance(v, (float, int)) and not isinstance(v, bool) and v > 0 for v in anchors):
            reasons.append("missing_positive_calibrated_consumption_anchor")
        if reasons:
            excluded.append({"id": p["id"], "reasons": reasons})
        else:
            eligible.append(p["id"])
    return personas, eligible, {"source_population_size": len(personas), "eligible_count": len(eligible),
                               "excluded": excluded, "fields_reconciled_from_graph": dict(changes),
                               "smoking_assignment_before_eligibility_filter": True,
                               "graph_was_modified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-json", type=Path, required=True)
    parser.add_argument("--raw-agents", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists; choose a new directory")
    if len(args.source_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.source_sha256):
        parser.error("Expected the verified source dump SHA-256")
    personas, eligible, audit = prepare(read(args.graph_json), read(args.raw_agents), args.source_sha256)
    audit.update(source_dump_sha256=args.source_sha256,
                 graph_export_sha256=hashlib.sha256(args.graph_json.read_bytes()).hexdigest(),
                 raw_agents_sha256=hashlib.sha256(args.raw_agents.read_bytes()).hexdigest())
    args.out.mkdir(parents=True)
    for name, value in (("personas.json", personas), ("eligible_ids.json", eligible), ("audit.json", audit)):
        (args.out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"source_population_size": len(personas), "eligible_count": len(eligible),
                      "excluded_count": len(audit["excluded"]), "reconciled_fields": audit["fields_reconciled_from_graph"]}))


if __name__ == "__main__":
    main()
