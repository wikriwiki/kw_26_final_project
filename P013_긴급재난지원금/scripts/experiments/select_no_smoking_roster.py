"""Repair a fixed-size source roster using eligible IDs and a frozen hash rank.

Retain every eligible member of the original roster, then replace only missing
members. Selection never reads demographics, smoking labels, or outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RANK_PREFIX = "no-smoking-roster"


def valid_ids(value, name):
    if (not isinstance(value, list) or not value
            or any(not isinstance(aid, str) or not aid.strip() or aid != aid.strip() for aid in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{name} must be a nonempty list of unique nonblank string IDs")
    return set(value)


def canonical_hash(ids):
    value = json.dumps(sorted(ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_roster(source_ids, eligible_ids, size, seed):
    if type(size) is not int or size <= 0:
        raise ValueError("size must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    source = valid_ids(source_ids, "source IDs")
    eligible = valid_ids(eligible_ids, "eligible IDs")
    if len(source) != size:
        raise ValueError("source roster size must equal the requested fixed experiment size")
    retained = source & eligible
    if len(retained) > size:
        raise ValueError("retained source intersection exceeds requested size")
    pool = eligible - source
    required = size - len(retained)
    if len(pool) < required:
        raise ValueError("eligible replacement pool is insufficient for the requested size")

    def rank(aid):
        return hashlib.sha256(f"{RANK_PREFIX}:{seed}:{aid}".encode("utf-8")).hexdigest(), aid

    added = sorted(sorted(pool, key=rank)[:required])
    selected = sorted(retained | set(added))
    removed = sorted(source - eligible)
    audit = {
        "method": "preserve_eligible_source_ids_then_sha256_rank_replacements",
        "rank_input": "UTF-8 no-smoking-roster:<integer seed>:<exact agent ID>",
        "rank_order": "ascending SHA256 hex digest, then exact ID as collision tie-break",
        "seed": seed, "requested_size": size,
        "source_count": len(source), "eligible_count": len(eligible),
        "retained_count": len(retained), "removed_count": len(removed),
        "replacement_pool_count": len(pool), "added_count": len(added),
        "selected_count": len(selected), "removed_source_ids": removed,
        "added_replacement_ids": added,
        "source_ids_sha256": canonical_hash(source),
        "eligible_ids_sha256": canonical_hash(eligible),
        "retained_ids_sha256": canonical_hash(retained),
        "selected_ids_sha256": canonical_hash(selected),
        "selection_inputs": ["source IDs", "eligible IDs", "size", "seed"],
        "demographics_smoking_and_outcomes_used": False,
    }
    return selected, audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ids", type=Path, required=True)
    parser.add_argument("--eligible-ids", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20171203)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Output exists; choose a new directory to preserve the frozen roster")
    source_bytes, eligible_bytes = args.source_ids.read_bytes(), args.eligible_ids.read_bytes()
    selected, audit = select_roster(json.loads(source_bytes.decode("utf-8-sig")),
                                    json.loads(eligible_bytes.decode("utf-8-sig")), args.size, args.seed)
    audit.update(source_file_sha256=hashlib.sha256(source_bytes).hexdigest(),
                 eligible_file_sha256=hashlib.sha256(eligible_bytes).hexdigest(),
                 source_file=str(args.source_ids), eligible_file=str(args.eligible_ids))
    args.out.mkdir(parents=True, exist_ok=False)
    for name, value in (("cohort_ids.json", selected), ("audit.json", audit)):
        (args.out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"selected_count": len(selected), "retained_count": audit["retained_count"],
                      "removed_count": audit["removed_count"], "added_count": audit["added_count"],
                      "selected_ids_sha256": audit["selected_ids_sha256"], "output": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
