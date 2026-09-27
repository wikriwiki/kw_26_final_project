"""Bind P016 score to both patched inputs and preserved failed/preflight evidence.

This is a post-score, read-only provenance sidecar. It never alters the frozen
numeric indicators or treats the failed prepatch arm as an effect observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


PATCHED_INSTANT_DISCOUNT_SHA256 = (
    "bc3819adaee5c8c922309aa0dc6b62c22a89f8630023939edb874b2b8fec50c4"
)
COMMON_FROZEN_SUFFIXES = (
    "scripts/sim/prompts/v53.py",
    "scripts/sim/stage1_intent.py",
    "scripts/sim/stage2_poi.py",
    "scripts/sim/run_simulation.py",
    "scripts/sim/instant_discount.py",
    "scripts/sim/environments/registry.py",
    "/roster.json",
    "/frozen_income.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha_list(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"malformed SHA256 line in {path}")
        digest, remote_name = parts
        if (len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower())
                or remote_name in entries):
            raise ValueError(f"invalid or duplicate SHA256 entry in {path}")
        entries[remote_name.lstrip("*")] = digest.lower()
    return entries


def unique_suffix(entries: dict[str, str], suffix: str) -> str:
    matches = [digest for path, digest in entries.items() if path.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one frozen SHA entry ending {suffix!r}")
    return matches[0]


def audited_arm(arm_dir: Path, arm: str, score_run: dict) -> dict:
    ledger = arm_dir / "sector.ledger.jsonl"
    ledger_manifest = arm_dir / "sector.ledger.jsonl.manifest.json"
    frozen = arm_dir / "frozen_inputs.sha256"
    run_manifest = arm_dir / "run_manifest.json"
    model_evidence = arm_dir / "served_model_evidence.json"
    if any(not path.is_file() for path in
           (ledger, ledger_manifest, frozen, run_manifest, model_evidence)):
        raise ValueError(f"{arm} patched arm is missing preserved files")

    evidence = {entry["path"].replace("\\", "/"): entry["sha256"]
                for entry in score_run["evidence"]}
    if evidence.get(ledger.as_posix()) != sha256(ledger):
        raise ValueError(f"{arm} sector ledger does not belong to the numeric score")
    if evidence.get(ledger_manifest.as_posix()) != sha256(ledger_manifest):
        raise ValueError(f"{arm} sector manifest does not belong to the numeric score")

    frozen_hashes = sha_list(frozen)
    manifest = read_json(run_manifest)
    sector = read_json(ledger_manifest)
    if manifest.get("case") != "p016" or manifest.get("arm") != arm:
        raise ValueError(f"{arm} wrong run manifest")
    expected_policy = "P016" if arm == "on" else None
    if sector.get("arm") != arm or sector.get("policy_id") != expected_policy:
        raise ValueError(f"{arm} wrong sector manifest")
    score_provenance = score_run["run_provenance"]
    expected_run_id = score_provenance[f"{arm}_run_id"]
    actual_run_ids = (sector.get("provenance") or {}).get("experience_run_id")
    if actual_run_ids != [expected_run_id]:
        raise ValueError(f"{arm} run ID differs from score provenance")
    if manifest.get("start") != sector.get("start") or manifest.get("days") != sector.get("days"):
        raise ValueError(f"{arm} run date window differs from sector evidence")
    if (unique_suffix(frozen_hashes, "/run_manifest.json") != sha256(run_manifest)
            or unique_suffix(frozen_hashes, "/served_model_evidence.json") != sha256(model_evidence)):
        raise ValueError(f"{arm} preserved run/model evidence differs from frozen hashes")
    model_provenance = score_provenance["served_model_provenance"]
    if model_provenance[f"{arm}_evidence_sha256"] != sha256(model_evidence):
        raise ValueError(f"{arm} served-model evidence differs from score")

    roster = arm_dir.parent / "roster.json"
    income = arm_dir.parent / "frozen_income.json"
    if (unique_suffix(frozen_hashes, "/roster.json") != sha256(roster)
            or unique_suffix(frozen_hashes, "/frozen_income.json") != sha256(income)):
        raise ValueError(f"{arm} cohort/budget file differs from frozen hashes")
    if sector.get("roster_sha256") != sha256(roster):
        raise ValueError(f"{arm} sector roster hash differs from frozen cohort")
    patched = unique_suffix(frozen_hashes, "scripts/sim/instant_discount.py")
    if patched != PATCHED_INSTANT_DISCOUNT_SHA256:
        raise ValueError(f"{arm} instant-discount engine is not the audited patch")
    return {
        "arm": arm,
        "run_id": expected_run_id,
        "run_revision": manifest.get("run_revision"),
        "run_manifest_source_commit_unreliable_for_live_code": manifest.get("source_commit"),
        "run_manifest_path": run_manifest.as_posix(),
        "run_manifest_sha256": sha256(run_manifest),
        "frozen_inputs_path": frozen.as_posix(),
        "frozen_inputs_sha256": sha256(frozen),
        "frozen_hashes": {suffix: unique_suffix(frozen_hashes, suffix)
                          for suffix in COMMON_FROZEN_SUFFIXES},
        "roster_sha256": sha256(roster),
        "income_map_sha256": sha256(income),
        "sector_manifest_sha256": sha256(ledger_manifest),
        "sector_ledger_sha256": sha256(ledger),
        "served_model_evidence_sha256": sha256(model_evidence),
        "source_fingerprint": (sector.get("provenance") or {}).get("source_fingerprint"),
        "paired_environment_fingerprint": (sector.get("provenance") or {}).get(
            "paired_environment_fingerprint"),
    }


def preflight_audit(case_dir: Path) -> dict:
    failure = case_dir / "preflight_first_failure.manifest.json"
    success_hashes = case_dir / "preflight_success.sha256"
    if not failure.is_file() or not success_hashes.is_file():
        raise ValueError("missing preserved P016 preflight evidence")
    first = read_json(failure)
    if (first.get("model_calls") != 0 or first.get("exitcode") != 1
            or first.get("graph_policy_ids") != ["P012", "P016"]):
        raise ValueError("unexpected P016 first preflight failure evidence")
    hashes = sha_list(success_hashes)
    for relative, digest in hashes.items():
        candidate = case_dir / relative
        if not candidate.is_file() or sha256(candidate) != digest:
            raise ValueError(f"P016 passed preflight checksum mismatch: {relative}")
    retry = case_dir / "preflight_retry.log"
    if "PREFLIGHT_PASS p016" not in retry.read_text(encoding="utf-8"):
        raise ValueError("P016 retry does not record a passed preflight")
    return {
        "first_failure_manifest_path": failure.as_posix(),
        "first_failure_manifest_sha256": sha256(failure),
        "first_failure_model_calls": 0,
        "first_failure_reason": first.get("finding"),
        "passed_preflight_checksums_path": success_hashes.as_posix(),
        "passed_preflight_checksums_sha256": sha256(success_hashes),
        "passed_preflight_files": hashes,
        "passed_preflight_model_calls": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--on-dir", type=Path, required=True)
    parser.add_argument("--off-dir", type=Path, required=True)
    parser.add_argument("--numeric", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    score = read_json(args.numeric)
    run = score["runs"][0]
    if run.get("policy") != "P016":
        raise ValueError("not a P016 numeric score")
    if sha256(Path("scripts/sim/instant_discount.py")) != PATCHED_INSTANT_DISCOUNT_SHA256:
        raise ValueError("local audited discount patch SHA differs")
    on = audited_arm(args.on_dir, "on", run)
    off = audited_arm(args.off_dir, "off", run)
    if on["run_id"] == off["run_id"]:
        raise ValueError("P016 paired runs are not distinct")
    if on["run_revision"] != off["run_revision"]:
        raise ValueError("P016 paired arms have different patch revisions")
    for suffix in COMMON_FROZEN_SUFFIXES:
        if on["frozen_hashes"][suffix] != off["frozen_hashes"][suffix]:
            raise ValueError(f"P016 paired frozen input differs: {suffix}")
    if (on["source_fingerprint"] != off["source_fingerprint"]
            or on["paired_environment_fingerprint"] != off["paired_environment_fingerprint"]):
        raise ValueError("P016 paired execution/factual fingerprints differ")

    failed = args.case_dir / "failed_prepatch" / "failed_prepatch_snapshot.json"
    snapshot = read_json(failed)
    if (snapshot.get("run_status") != "incomplete_failed"
            or not snapshot.get("raw_keyerror_rows")):
        raise ValueError("P016 invalidated prepatch arm is not documented")
    payload = {
        "schema": "p016_postfix_provenance_display_audit_v1",
        "purpose": "Evidence-only; no indicator score or preregistration change",
        "policy": "P016",
        "numeric_path": args.numeric.as_posix(),
        "numeric_sha256": sha256(args.numeric),
        "effect_window": run["on"],
        "patched_instant_discount_sha256": PATCHED_INSTANT_DISCOUNT_SHA256,
        "run_manifest_source_commit_is_not_live_code_evidence": True,
        "on": on,
        "off": off,
        "preflight": preflight_audit(args.case_dir),
        "invalidated_prepatch_arm": {
            "excluded_from_score": True,
            "snapshot_path": failed.as_posix(),
            "snapshot_sha256": sha256(failed),
            "raw_keyerror_rows": snapshot["raw_keyerror_rows"],
            "failure_date": snapshot["date_of_failure"],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.out)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{args.out}: paired patched P016 provenance verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
