"""Copy one finished simulation day off a live Neo4j database without writes.

The archive is evidence for analysis, not a restorable Neo4j backup. Require
the next day's metrics file so the target day's graph cannot still be written.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tarfile
import tempfile
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


def runner_environment(pid: int) -> dict[str, str]:
    result = {}
    for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            result[key.decode()] = value.decode()
    return result


def verify_finished_metrics(output_dir: Path, day: date,
                            expected: int) -> tuple[Path, list[dict], dict]:
    target = output_dir / "metrics" / f"day_{day.isoformat()}.jsonl"
    next_day = output_dir / "metrics" / f"day_{(day + timedelta(days=1)).isoformat()}.jsonl"
    if not next_day.is_file():
        raise RuntimeError(f"Next day has not started; refusing a racing snapshot: {next_day}")
    if not target.is_file():
        raise RuntimeError(f"Target metrics file missing: {target}")
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    ids = [row.get("aid") for row in rows]
    if len(rows) != expected or any(not isinstance(aid, str) or not aid for aid in ids):
        raise RuntimeError(f"Incomplete or invalid metrics: {len(rows)}/{expected}")
    if len(set(ids)) != expected:
        raise RuntimeError("Duplicate citizen IDs in target day's metrics")
    checkpoint = output_dir / "checkpoints" / f"done_{day.isoformat()}.json"
    if not checkpoint.is_file():
        raise RuntimeError(f"Completion checkpoint missing: {checkpoint}")
    done_ids = json.loads(checkpoint.read_text(encoding="utf-8"))
    if (not isinstance(done_ids, list) or len(done_ids) != len(set(done_ids))
            or not set(done_ids).issubset(set(ids))):
        raise RuntimeError("Completion checkpoint does not match the target roster")
    return target, rows, {"metrics": len(rows),
                          "metrics_status": dict(Counter(r.get("status") for r in rows)),
                          "checkpoint_done": len(done_ids)}


QUERIES = {
    "agent": """
        MATCH (a:Agent)-[:HAS_STATE {day: date($day)}]->(:State)
        RETURN a.id AS aid, properties(a) AS agent
    """,
    "state": """
        MATCH (a:Agent)-[:HAS_STATE {day: date($day)}]->(s:State)
        RETURN a.id AS aid, $day AS day, properties(s) AS state
    """,
    "spend": """
        MATCH (a:Agent)-[:HAS_PLAN {day: date($day)}]->(p:Plan)
              -[i:INCLUDES]->(poi:POI)
        RETURN a.id AS aid, $day AS day, properties(i) AS spend, poi.id AS poi_id
    """,
    "policy": """
        MATCH (p:Policy {id:$policy_id})
        RETURN p.id AS id, properties(p) AS policy
    """,
}


def export_query(session, query: str, path: Path, **params) -> tuple[int, set[str]]:
    count = 0
    aids: set[str] = set()
    with gzip.open(path, "wt", encoding="utf-8") as out:
        for record in session.run(query, **params):
            row = dict(record)
            out.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            count += 1
            if row.get("aid"):
                aids.add(row["aid"])
    return count, aids


def snapshot(day: date, output_dir: Path, evidence_dir: Path, stem: str,
             policy_id: str, expected: int, runner_pid: int) -> dict:
    metrics, rows, counts = verify_finished_metrics(output_dir, day, expected)
    env = runner_environment(runner_pid)
    for key in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
        if not env.get(key):
            raise RuntimeError(f"Runner has no {key}; cannot read graph")

    from neo4j import GraphDatabase, READ_ACCESS

    evidence_dir.mkdir(parents=True, exist_ok=True)
    archive = evidence_dir / f"{stem}.tar.gz"
    if archive.exists():
        raise FileExistsError(f"Evidence archive already exists: {archive}")
    with tempfile.TemporaryDirectory(dir=evidence_dir, prefix=f".{stem}.") as tmp:
        tmpdir = Path(tmp)
        exported = {}
        with GraphDatabase.driver(env["NEO4J_URI"],
                                  auth=(env["NEO4J_USER"], env["NEO4J_PASSWORD"])) as driver:
            with driver.session(default_access_mode=READ_ACCESS) as session:
                for name, query in QUERIES.items():
                    path = tmpdir / f"{stem}_{name}.jsonl.gz"
                    count, aids = export_query(session, query, path,
                                               day=day.isoformat(), policy_id=policy_id)
                    exported[name] = {"path": path, "count": count, "aids": aids}

        metrics_ids = {row["aid"] for row in rows}
        state_ids = exported["state"]["aids"]
        counts.update({
            "agent": exported["agent"]["count"],
            "state": exported["state"]["count"],
            "state_unique_aids": len(state_ids),
            "state_missing_for_ok": len({r["aid"] for r in rows if r.get("status") == "ok"}
                                        - state_ids),
            "state_outside_metrics_roster": len(state_ids - metrics_ids),
            "spend": exported["spend"]["count"],
            "policy": exported["policy"]["count"],
        })
        partial = tmpdir / f"{stem}.tar.gz.partial"
        with tarfile.open(partial, "w:gz") as tar:
            tar.add(metrics, arcname=f"metrics/day_{day.isoformat()}.jsonl")
            for folder, filename in (
                ("checkpoints", f"done_{day.isoformat()}.json"),
                ("checkpoints", f"failed_{day.isoformat()}.json"),
                ("timing", f"day_{day.isoformat()}.json"),
                ("timing", f"slow_{day.isoformat()}.json"),
            ):
                source = output_dir / folder / filename
                if source.is_file():
                    tar.add(source, arcname=f"{folder}/{filename}")
            for name, item in exported.items():
                tar.add(item["path"], arcname=f"graph/{stem}_{name}.jsonl.gz")
        os.replace(partial, archive)

    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "scope": f"completed {day.isoformat()} read-only evidence; not a Neo4j restore dump",
        "counts": counts,
        "archive_sha256": digest,
        "bytes": archive.stat().st_size,
    }
    manifest_path = evidence_dir / f"{stem}.manifest.json"
    manifest_partial = evidence_dir / f".{stem}.manifest.json.partial"
    manifest_partial.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    os.replace(manifest_partial, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=date.fromisoformat, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--expected-per-day", type=int, required=True)
    parser.add_argument("--runner-pid", type=int, required=True)
    args = parser.parse_args()
    result = snapshot(args.day, args.output_dir, args.evidence_dir,
                      args.stem, args.policy_id, args.expected_per_day, args.runner_pid)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
