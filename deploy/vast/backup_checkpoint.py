"""Create a recoverable, immutable post-day checkpoint outside a Vast instance.

Called only after a complete simulation day. Neo4j Community requires an offline
dump, so this script stops exactly the selected database and always restarts it
before attempting the Google Drive upload. A remote commit marker is written last.
Secrets are read from environment/config files and are never serialized.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/sim'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hash_file(path, name="sha256"):
    digest = hashlib.new(name)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_day_files(run_dir, day):
    """Select sealed files for this day plus the latest phase state metadata."""
    result = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError("Backup source contains a symlink")
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir)
        if (day in relative.as_posix()
                or relative.as_posix() in {"experiment_run.json", "summary.json", "stage1_failures.jsonl", "stage2_failures.jsonl"}):
            result.append(path)
    require(any(p.relative_to(run_dir).as_posix() == f"metrics/day_{day}.jsonl" for p in result),
            "Completed day metrics are missing")
    require(any(p.relative_to(run_dir).as_posix() == f"night2_completed_{day}.json" for p in result),
            "Night completion marker is missing")
    return result


def validate_day(run_dir, day, manifest):
    from scripts.sim.evidence_integrity import verify

    ids = set(manifest["cohort_ids"])
    seen = set()
    for line in (run_dir / "metrics" / f"day_{day}.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        verify(row)
        if (row.get("status") not in {"ok", "skipped"} or row.get("aid") not in ids
                or row["aid"] in seen or row.get("experience_day") != day
                or row.get("experience_run_id") != manifest["run_id"]
                or row.get("no_smoking", {}).get("arm") != manifest["arm"]):
            raise ValueError("Incomplete or foreign daily metrics")
        if row["status"] == "skipped" and (row.get("attempts") != 6
                or row.get("skip_kind") != "failed_after_retries"
                or row.get("observed_behavior") is not False
                or row.get("no_smoking", {}).get("observed_behavior") is not False):
            raise ValueError("Skipped day has no valid terminal failure receipt")
        seen.add(row["aid"])
    require(seen == ids, f"Daily metrics have {len(seen)}/{len(ids)} valid agents")
    marker = json.loads((run_dir / f"night2_completed_{day}.json").read_text(encoding="utf-8"))
    verify(marker)
    require(marker.get("run_id") == manifest["run_id"] and marker.get("arm") == manifest["arm"]
            and marker.get("day") == day and marker.get("status") == "complete",
            "Night completion marker has wrong run identity")


def make_archive(run_dir, day, target, *, partial=False):
    paths = select_day_files(run_dir, day) if not partial else sorted(run_dir.rglob('*'))
    paths = [p for p in paths if p.is_file()]
    require(all(not p.is_symlink() for p in paths), 'Backup source contains a symlink')
    with tarfile.open(target, mode="w:gz") as archive:
        for path in paths:
            archive.add(path, arcname="run/" + path.relative_to(run_dir).as_posix(), recursive=False)
    return len(paths)


def check_path(path, *, directory=False):
    resolved = path.resolve(strict=True)
    require(not path.is_symlink() and (resolved.is_dir() if directory else resolved.is_file()),
            f"Invalid backup input: {path.name}")
    return resolved


def run_quiet(argv, *, env=None, timeout=3600):
    result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"Backup command {Path(argv[0]).name} failed with exit {result.returncode}")
    return result.stdout


def offline_dump(home, dump_dir, bolt_port):
    neo4j = home / "bin/neo4j"
    admin = home / "bin/neo4j-admin"
    require(neo4j.is_file() and admin.is_file(), "Neo4j home does not contain the expected binaries")
    env = dict(os.environ, NEO4J_HOME=str(home), NEO4J_CONF=str(home / "conf"))
    # Neo4j itself does not need the rclone token or application passwords.
    for name in list(env):
        if any(term in name.upper() for term in ("PASSWORD", "TOKEN", "SECRET", "API_KEY")):
            env.pop(name, None)
    stopped = False
    try:
        run_quiet([str(neo4j), "stop"], env=env, timeout=180)
        stopped = True
        run_quiet([str(admin), "database", "dump", "neo4j", "--to-path=" + str(dump_dir)],
                  env=env, timeout=3600)
    finally:
        if stopped:
            run_quiet([str(neo4j), "start"], env=env, timeout=180)
            deadline = time.monotonic() + 180
            while True:
                try:
                    with socket.create_connection(("127.0.0.1", bolt_port), timeout=2):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Neo4j did not become available after checkpoint")
                    time.sleep(2)
    dump = dump_dir / "neo4j.dump"
    require(dump.is_file() and dump.stat().st_size > 0, "Offline graph dump missing")
    return dump


def remote_md5(rclone, config, remote_file):
    output = run_quiet([str(rclone), "--config", str(config), "md5sum", remote_file], timeout=300)
    match = re.fullmatch(r"([0-9a-f]{32})  .+\s*", output)
    require(match is not None, "Drive did not return a usable MD5 checksum")
    return match.group(1)


def upload_verified(rclone, config, local, remote):
    run_quiet([str(rclone), "--config", str(config), "copyto", str(local), remote,
               "--immutable", "--retries", "5", "--low-level-retries", "10", "--stats", "0"],
              timeout=7200)
    require(remote_md5(rclone, config, remote) == hash_file(local, "md5"),
            "Drive checksum does not match the local checkpoint")


def checkpoint(day, run_dir, *, partial=False):
    require(bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", day)), "Invalid checkpoint date")
    run_dir = check_path(run_dir, directory=True)
    require(run_dir.is_relative_to(Path("/workspace/no-smoking-results")),
            "Checkpoint run must live under /workspace/no-smoking-results")
    manifest = json.loads((run_dir / "experiment_run.json").read_text(encoding="utf-8"))
    require(manifest.get("phase") in {"shared_pre", "post_branch"}, "Not a shared experiment phase")
    first = datetime.fromisoformat(manifest["start"]).date()
    current = datetime.fromisoformat(day).date()
    require(first <= current < first + timedelta(days=manifest["days"]),
            "Checkpoint day is outside the phase window")
    if not partial:
        validate_day(run_dir, day, manifest)
    home = check_path(Path(os.environ["BACKUP_NEO4J_HOME"]), directory=True)
    require(home.is_relative_to(Path("/workspace"))
            and any(part.startswith("no-smoking-neo4j") for part in home.parts),
            "Neo4j backup home must be a dedicated no-smoking instance")
    config = check_path(Path(os.environ["BACKUP_RCLONE_CONFIG"]))
    rclone = check_path(Path(os.environ["BACKUP_RCLONE_BINARY"]))
    remote = os.environ["BACKUP_DRIVE_REMOTE"].rstrip("/")
    require(bool(re.fullmatch(r"[A-Za-z0-9_]+:[A-Za-z0-9_./-]+", remote)) and ".." not in remote,
            "Backup Drive destination must be a dedicated path")
    root = Path(os.environ.get("BACKUP_CHECKPOINT_ROOT", "/workspace/no-smoking-checkpoints")).resolve()
    require(root.is_relative_to(Path("/workspace")) and not root.is_relative_to(home),
            "Checkpoint staging must be outside the Neo4j home")
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    work = root / run_dir.name / (('progress-' + day) if partial else day) / attempt
    work.mkdir(parents=True, mode=0o700)
    archive = work / "run-day.tar.gz"
    count = make_archive(run_dir, day, archive, partial=partial)
    port = int(os.environ["BACKUP_NEO4J_BOLT_PORT"])
    require(1024 <= port <= 65535, "Invalid Neo4j port")
    dump = offline_dump(home, work, port)
    record = {"schema_version": 1, "phase": manifest["phase"], "arm": manifest["arm"],
              "kind": 'quiescent_partial_graph_and_full_run' if partial else 'complete_day',
              "run_id": manifest["run_id"], "day": day, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "files": {p.name: {"bytes": p.stat().st_size, "sha256": hash_file(p)}
                        for p in (archive, dump)}, "archive_file_count": count,
              "source_manifest_sha256": hash_file(run_dir / "experiment_run.json")}
    record['parent_checkpoints'] = ([] if partial else [
        json.loads(p.read_text(encoding='utf-8'))
        for p in sorted(run_dir.glob('backup_completed_*.json'))
        if p.name < f'backup_completed_{day}.json'])
    record_path = work / "checkpoint.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    category = 'recoverable-progress' if partial else 'checkpoints'
    prefix = f"{remote}/runs/{run_dir.name}/{category}/{day}/{attempt}"
    for path in (archive, dump, record_path):
        upload_verified(rclone, config, path, f"{prefix}/{path.name}")
    committed = work / "committed.json"
    committed.write_text(json.dumps({"checkpoint_sha256": hash_file(record_path),
                                     "day": day, "complete": True}, indent=2) + "\n", encoding="utf-8")
    upload_verified(rclone, config, committed, f"{prefix}/{committed.name}")
    from scripts.sim.experience_provenance import atomic_json
    receipt = {'remote': prefix, 'checkpoint_sha256': hash_file(record_path),
               'graph_sha256': record['files']['neo4j.dump']['sha256'],
               'verified_at_utc': datetime.now(timezone.utc).isoformat(),
               'day': day, 'run_id': manifest['run_id'], 'arm': manifest['arm'],
               'kind': record['kind']}
    atomic_json(run_dir / 'recoverable_backup.json', receipt)
    if not partial:
        atomic_json(run_dir / f'backup_completed_{day}.json', receipt)
    archive.unlink()
    retain_branch_dump = not partial and manifest["phase"] == "shared_pre" and day == "2017-12-02"
    if not retain_branch_dump:
        dump.unlink()
    print(json.dumps({"status": "drive_checkpoint_verified", "day": day,
                      "run": run_dir.name, "archive_file_count": count,
                      "graph_dump_sha256": record["files"]["neo4j.dump"]["sha256"],
                      "retained_branch_dump": str(dump) if retain_branch_dump else None}))


def finalize(run_dir):
    """Publish the final manifest and summary after the last daily checkpoint."""
    run_dir = check_path(run_dir, directory=True)
    require(run_dir.is_relative_to(Path("/workspace/no-smoking-results")),
            "Finalized run must live under /workspace/no-smoking-results")
    manifest_path = check_path(run_dir / "experiment_run.json")
    summary_path = check_path(run_dir / "summary.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(manifest.get("status") == "complete" and manifest.get("exit_code") == 0,
            "Only a successful run can be finalized")
    require(manifest.get("phase") in {"shared_pre", "post_branch"},
            "Only shared experiment phases need finalization")
    require(len(summary.get("summary", [])) == manifest["days"],
            "Final summary does not cover every day")
    first = datetime.fromisoformat(manifest["start"]).date()
    last = (first + timedelta(days=manifest["days"] - 1)).isoformat()
    validate_day(run_dir, last, manifest)
    config = check_path(Path(os.environ["BACKUP_RCLONE_CONFIG"]))
    rclone = check_path(Path(os.environ["BACKUP_RCLONE_BINARY"]))
    remote = os.environ["BACKUP_DRIVE_REMOTE"].rstrip("/")
    require(bool(re.fullmatch(r"[A-Za-z0-9_]+:[A-Za-z0-9_./-]+", remote)) and ".." not in remote,
            "Backup Drive destination must be a dedicated path")
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    prefix = f"{remote}/runs/{run_dir.name}/final/{attempt}"
    for path in (manifest_path, summary_path):
        upload_verified(rclone, config, path, f"{prefix}/{path.name}")
    record = {"status": "complete", "last_day": last, "run_id": manifest["run_id"],
              "manifest_sha256": hash_file(manifest_path), "summary_sha256": hash_file(summary_path)}
    root = Path(os.environ.get("BACKUP_CHECKPOINT_ROOT", "/workspace/no-smoking-checkpoints")).resolve()
    require(root.is_relative_to(Path("/workspace")), "Invalid checkpoint staging root")
    marker = root / run_dir.name / "final" / attempt / "committed.json"
    marker.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    marker.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    upload_verified(rclone, config, marker, f"{prefix}/{marker.name}")
    print(json.dumps({"status": "drive_final_manifest_verified", "run": run_dir.name,
                      "last_day": last, "manifest_sha256": record["manifest_sha256"]}))


def probe():
    config = check_path(Path(os.environ["BACKUP_RCLONE_CONFIG"]))
    rclone = check_path(Path(os.environ["BACKUP_RCLONE_BINARY"]))
    remote = os.environ["BACKUP_DRIVE_REMOTE"].rstrip("/")
    require(bool(re.fullmatch(r"[A-Za-z0-9_]+:[A-Za-z0-9_./-]+", remote)) and ".." not in remote,
            "Backup Drive destination must be a dedicated path")
    root = Path(os.environ.get("BACKUP_CHECKPOINT_ROOT", "/workspace/no-smoking-checkpoints"))
    require(root.resolve().is_relative_to(Path("/workspace")), "Invalid checkpoint staging root")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    local = root / ("probe-" + uuid4().hex + ".txt")
    local.write_text("NoSmokingZone restore probe\n", encoding="utf-8")
    try:
        upload_verified(rclone, config, local, f"{remote}/diagnostic/{local.name}")
    finally:
        local.unlink(missing_ok=True)
    print(json.dumps({"status": "drive_upload_checksum_verified"}))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == '--progress':
        checkpoint(sys.argv[2], Path(sys.argv[3]), partial=True)
    elif len(sys.argv) == 2 and sys.argv[1] == "--probe":
        probe()
    elif len(sys.argv) == 3 and sys.argv[1] == "--finalize":
        finalize(Path(sys.argv[2]))
    elif len(sys.argv) == 3:
        checkpoint(sys.argv[1], Path(sys.argv[2]))
    else:
        raise SystemExit("Usage: backup_checkpoint.py --probe | --finalize RUN_DIR | YYYY-MM-DD RUN_DIR")
