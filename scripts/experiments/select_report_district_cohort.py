"""Freeze the report-district subset of an existing no-smoking cohort.

This preserves the prior experiment roster and its smoking assignments. It does
not claim that synthetic personas represent the districts' census populations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

DISTRICTS = {"11350": "노원구", "11650": "서초구", "11710": "송파구"}


def read_list(path: Path) -> list:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return value


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(agents: list, eligible_ids: list[str], source_ids: list[str]) -> tuple[list[str], dict]:
    for label, ids in (("eligible", eligible_ids), ("source", source_ids)):
        if not ids or any(not isinstance(aid, str) or not aid for aid in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{label} IDs must be nonempty, unique strings")
    eligible = set(eligible_ids)
    source = set(source_ids)
    if not source <= eligible:
        raise ValueError("Source cohort contains ineligible IDs")

    by_id: dict[str, str] = {}
    for agent in agents:
        if not isinstance(agent, dict):
            raise ValueError("Every persona must be an object")
        aid = agent.get("id") or agent.get("agent_id") or agent.get("uuid")
        if not isinstance(aid, str) or not aid or aid in by_id:
            raise ValueError("Persona IDs must be nonempty and unique")
        residence = agent.get("residence") or {}
        code = str(residence.get("dong_code") or "")[:5]
        gu = residence.get("gu")
        if code in DISTRICTS or gu in DISTRICTS.values():
            if DISTRICTS.get(code) != gu:
                raise ValueError(f"District code/name mismatch for {aid}")
        by_id[aid] = code
    if not eligible <= by_id.keys():
        raise ValueError("Eligible roster contains IDs missing from personas")

    selected = sorted(aid for aid in source if by_id[aid] in DISTRICTS)
    counts = Counter(by_id[aid] for aid in selected)
    if any(counts[code] == 0 for code in DISTRICTS):
        raise ValueError("Each report district needs a nonempty cohort")
    eligible_counts = Counter(by_id[aid] for aid in eligible if by_id[aid] in DISTRICTS)
    audit = {
        "method": "subset_existing_frozen_cohort_by_residence_district",
        "districts": DISTRICTS,
        "source_cohort_size": len(source),
        "eligible_population_size": len(eligible),
        "selected_size": len(selected),
        "selected_by_district": {code: counts[code] for code in DISTRICTS},
        "eligible_by_district": {code: eligible_counts[code] for code in DISTRICTS},
        "prior_smoking_labels_preserved_by_prepare": True,
        "census_representativeness_claimed": False,
    }
    return selected, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=Path, required=True)
    parser.add_argument("--eligible-ids", type=Path, required=True)
    parser.add_argument("--source-cohort-ids", type=Path, required=True)
    parser.add_argument("--out-ids", type=Path, required=True)
    parser.add_argument("--out-audit", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    if args.out_ids.exists() or args.out_audit.exists():
        parser.error("Output already exists; preserve the frozen roster")
    selected, audit = select(read_list(args.agents), read_list(args.eligible_ids),
                             read_list(args.source_cohort_ids))
    if len(selected) != args.expected:
        parser.error(f"Expected {args.expected} selected agents, found {len(selected)}")
    audit["sources_sha256"] = {
        "agents": file_hash(args.agents), "eligible_ids": file_hash(args.eligible_ids),
        "source_cohort_ids": file_hash(args.source_cohort_ids),
    }
    args.out_ids.parent.mkdir(parents=True, exist_ok=True)
    args.out_audit.parent.mkdir(parents=True, exist_ok=True)
    args.out_ids.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit["selected_ids_sha256"] = file_hash(args.out_ids)
    args.out_audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
