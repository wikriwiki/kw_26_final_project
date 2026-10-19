"""Fail-closed, localhost-only restoration of a verified baseline into a new pair.

The source dump is opened read-only. No existing directory or database is reused.
Use bootstrap_neo4j.sh --dry-run to hash/check inputs without changing anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request

VERSION = "5.26.0"
ARTIFACT_URL = "https://dist.neo4j.org/neo4j-community-5.26.0-unix.tar.gz"
ARTIFACT_SHA256 = "ad8ac3398606145502b8f489530bbd39333707ae4668156af16b6086b5b037d7"
WORKSPACE = Path("/workspace")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def integer(env, key, default, minimum, maximum):
    raw = str(env.get(key, default))
    require(bool(re.fullmatch(r"[0-9]+", raw)), f"{key} must be an integer")
    value = int(raw)
    require(minimum <= value <= maximum, f"{key} must be {minimum}..{maximum}")
    return value


def settings(env, workspace=None):
    """Validate paths/secrets/options without creating files or printing secrets."""
    workspace = workspace or WORKSPACE
    raw_dump = env.get("CLEAN_BASELINE_DUMP", "")
    require(bool(raw_dump), "Set CLEAN_BASELINE_DUMP to the verified clean Day 0 .dump")
    dump = Path(raw_dump).absolute()
    require(not dump.is_symlink() and dump.is_file(), "CLEAN_BASELINE_DUMP must be a regular, non-symlink file")
    dump = dump.resolve()
    digest = env.get("NO_SMOKING_SNAPSHOT_SHA256", "").lower()
    require(bool(re.fullmatch(r"[0-9a-f]{64}", digest)), "Set NO_SMOKING_SNAPSHOT_SHA256 to the verified dump SHA256")
    root = Path(env.get("NO_SMOKING_NEO4J_ROOT", str(workspace / "no-smoking-neo4j"))).absolute()
    require(workspace.is_dir(), "The /workspace directory must already exist")
    require(root != workspace and root.is_relative_to(workspace), "Neo4j root must be a NEW child directory inside /workspace")
    require(root.resolve() == root, "Neo4j root and its parents must not contain symlinks or '..'")
    require(not root.exists() and not root.is_symlink(), "Neo4j root already exists; refusing to reuse or overwrite it")
    require(root.parent.is_dir(), "Neo4j root parent must already exist")
    require(not dump.is_relative_to(root), "The source dump must be outside the new Neo4j root")
    require(env.get("NEO4J_USER", "neo4j") == "neo4j", "Fresh Community instances require NEO4J_USER=neo4j")
    passwords = {}
    for arm in ("off", "on"):
        value = env.get(f"NO_SMOKING_{arm.upper()}_NEO4J_PASSWORD") or env.get("NEO4J_PASSWORD", "")
        require(len(value) >= 8 and value != "neo4j", f"Set a password of at least 8 characters for {arm.upper()} (arm-specific or NEO4J_PASSWORD)")
        passwords[arm] = value
    ports = {arm: integer(env, f"NO_SMOKING_{arm.upper()}_BOLT_PORT", default, 1024, 65535)
             for arm, default in (("off", 17687), ("on", 17688))}
    require(len(set(ports.values())) == 2, "OFF and ON Bolt ports must be different")
    heap_mb = integer(env, "NO_SMOKING_NEO4J_HEAP_MB", 2048, 256, 65536)
    cache_mb = integer(env, "NO_SMOKING_NEO4J_PAGECACHE_MB", 512, 128, 65536)
    return {"dump": dump, "sha256": digest, "root": root, "passwords": passwords,
            "ports": ports, "heap_mb": heap_mb, "cache_mb": cache_mb,
            "startup_timeout": integer(env, "NO_SMOKING_NEO4J_STARTUP_SECONDS", 180, 30, 1800),
            "load_timeout": integer(env, "NO_SMOKING_NEO4J_LOAD_SECONDS", 900, 30, 7200),
            "min_free_gb": integer(env, "NO_SMOKING_NEO4J_MIN_FREE_GB", 20, 4, 10000)}


def configuration(cfg, arm):
    return "\n".join([
        "# Dedicated experiment instance; generated from a verified clean dump.",
        "server.default_listen_address=127.0.0.1",
        "server.default_advertised_address=127.0.0.1",
        "server.bolt.enabled=true",
        f"server.bolt.listen_address=127.0.0.1:{cfg['ports'][arm]}",
        f"server.bolt.advertised_address=127.0.0.1:{cfg['ports'][arm]}",
        "server.http.enabled=false", "server.https.enabled=false",
        "dbms.security.auth_enabled=true", "initial.dbms.default_database=neo4j",
        f"server.memory.heap.initial_size={cfg['heap_mb']}m",
        f"server.memory.heap.max_size={cfg['heap_mb']}m",
        f"server.memory.pagecache.size={cfg['cache_mb']}m",
        "server.directories.data=data", "server.directories.logs=logs",
        "server.directories.run=run", "server.directories.transaction.logs.root=data/transactions",
        "dbms.usage_report.enabled=false", "",
    ])


def write_configuration(home, cfg, arm):
    path = home / "conf" / "neo4j.conf"
    original = path.read_text(encoding="utf-8")
    overrides = configuration(cfg, arm)
    keys = {line.split("=", 1)[0] for line in overrides.splitlines() if "=" in line and not line.startswith("#")}
    # Retain the pinned distribution's JVM options and other supported defaults.
    lines = [line for line in original.splitlines()
             if line.lstrip().startswith("#") or line.split("=", 1)[0].strip() not in keys]
    path.write_text("\n".join(lines) + "\n" + overrides, encoding="utf-8")


def summary(cfg):
    return {"neo4j_version": VERSION, "artifact_url": ARTIFACT_URL,
            "artifact_sha256": ARTIFACT_SHA256, "source_dump": str(cfg["dump"]),
            "source_dump_sha256": cfg["sha256"], "root": str(cfg["root"]),
            "database": "neo4j", "user": "neo4j", "passwords": "environment only; never serialized",
            "heap_mb_per_instance": cfg["heap_mb"], "pagecache_mb_per_instance": cfg["cache_mb"],
            "arms": {arm: {"home": str(cfg["root"] / arm),
                           "uri": f"bolt://127.0.0.1:{cfg['ports'][arm]}",
                           "snapshot_marker": cfg["sha256"]} for arm in ("off", "on")},
            "existing_directory_policy": "refuse", "source_dump_mode": "read_only",
            "load_overwrite_destination": False, "http_enabled": False}


def check_input(cfg):
    require(sha256(cfg["dump"]) == cfg["sha256"], "Source dump SHA256 mismatch; no files changed")
    require(shutil.disk_usage(cfg["root"].parent).free >= cfg["min_free_gb"] * 1024**3,
            f"At least {cfg['min_free_gb']} GiB free disk space is required for two restored instances")
    for port in cfg["ports"].values():
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                raise ValueError(f"Local Bolt port {port} is occupied") from None


def java_version():
    java = str(Path(os.environ["JAVA_HOME"]) / "bin/java") if os.environ.get("JAVA_HOME") else shutil.which("java")
    require(bool(java), "Java 17 or 21 is required. On the Ubuntu PyTorch host: apt-get update && apt-get install -y openjdk-17-jre-headless")
    result = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=20, check=True)
    match = re.search(r'version "(\d+)', result.stderr + result.stdout)
    require(match is not None and int(match.group(1)) in (17, 21), "Neo4j 5.26.0 requires Java 17 or 21; select JAVA_HOME accordingly")
    return int(match.group(1))


def neo_env(home):
    env = dict(os.environ)
    # Server subprocesses do not need to inherit application credentials.
    for key in list(env):
        if any(word in key.upper() for word in ("PASSWORD", "TOKEN", "SECRET", "API_KEY")):
            env.pop(key)
    for key in ("HEAP_SIZE", "JAVA_OPTS", "NEO4J_AUTH"):
        env.pop(key, None)
    env["NEO4J_HOME"] = str(home)
    env["NEO4J_CONF"] = str(home / "conf")
    return env


def command(home, args, timeout, stdin=None):
    # Fixed arguments only: passwords never enter subprocess argv or logs.
    with (home / "bootstrap-command.log").open("ab") as log:
        subprocess.run([str(home / "bin" / args[0]), *args[1:]], cwd=home,
                       env=neo_env(home), stdin=stdin, stdout=log, stderr=log,
                       timeout=timeout, check=True)


def extract(archive, home):
    # A pinned official archive is still checked for path traversal and links.
    prefix = f"neo4j-community-{VERSION}"
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            parts = Path(member.name).parts
            require(parts and parts[0] == prefix and ".." not in parts, "Unexpected path in Neo4j archive")
            require(member.isdir() or member.isfile(), "Links/devices in Neo4j archive are not allowed")
            if len(parts) == 1:
                continue
            target = home.joinpath(*parts[1:])
            require(target.is_relative_to(home), "Archive path escapes its dedicated home")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open("xb") as dest:
                    shutil.copyfileobj(source, dest)
                target.chmod(member.mode & 0o755)


def initialize_graph(cfg, arm):
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, SessionExpired

    uri = f"bolt://127.0.0.1:{cfg['ports'][arm]}"
    deadline = time.monotonic() + cfg["startup_timeout"]
    # Only the 'neo4j' database was restored; each new system database starts with
    # the standard password. Change it over localhost immediately, with parameters.
    with GraphDatabase.driver(uri, auth=("neo4j", "neo4j"), connection_timeout=3) as driver:
        while True:
            try:
                with driver.session(database="system") as session:
                    session.run("ALTER CURRENT USER SET PASSWORD FROM $old TO $new",
                                old="neo4j", new=cfg["passwords"][arm]).consume()
                break
            except (ServiceUnavailable, SessionExpired):
                require(time.monotonic() < deadline, f"{arm.upper()} database startup timed out")
                time.sleep(1)
    with GraphDatabase.driver(uri, auth=("neo4j", cfg["passwords"][arm]), connection_timeout=5) as driver:
        driver.verify_connectivity()
        with driver.session(database="neo4j") as session:
            def mark(tx):
                count = tx.run("MATCH (s:ExperimentSnapshot) RETURN count(s) AS n").single()["n"]
                require(count <= 1, "Unexpected multiple ExperimentSnapshot markers; refusing to relabel")
                tx.run("MATCH (s:ExperimentSnapshot) DETACH DELETE s").consume()
                tx.run("CREATE (:ExperimentSnapshot {id:$sha, sha256:$sha, source_dump_sha256:$sha, source_dump_name:$name})",
                       sha=cfg["sha256"], name=cfg["dump"].name).consume()
            session.execute_write(mark)
            markers = list(session.run("MATCH (s:ExperimentSnapshot) RETURN s.id AS id, s.sha256 AS sha256"))
            require(len(markers) == 1 and markers[0]["id"] == cfg["sha256"] and markers[0]["sha256"] == cfg["sha256"],
                    f"{arm.upper()} snapshot marker verification failed")


def execute(cfg):
    require(sys.platform.startswith("linux"), "Execution requires Linux; --dry-run supports local validation")
    java = java_version()
    try:
        import neo4j  # noqa: F401
    except ImportError:
        raise ValueError("Install deploy/vast/requirements-runtime.txt in the selected PYTHON environment first") from None
    root = cfg["root"]
    root.mkdir(mode=0o700)  # Atomic refusal if another process created it since validation.
    started = []
    def terminate(signum, _frame):
        raise SystemExit(128 + signum)
    previous_term = signal.signal(signal.SIGTERM, terminate)
    try:
        archive = root / f"neo4j-community-{VERSION}-unix.tar.gz"
        with urllib.request.urlopen(ARTIFACT_URL, timeout=60) as response, archive.open("xb") as handle:
            shutil.copyfileobj(response, handle)
        require(sha256(archive) == ARTIFACT_SHA256, "Downloaded Neo4j artifact checksum mismatch")
        for arm in ("off", "on"):
            home = root / arm
            home.mkdir(mode=0o700)
            extract(archive, home)
            write_configuration(home, cfg, arm)
            command(home, ["neo4j-admin", "server", "validate-config"], 30)
            # Read-only input stream avoids renaming, copying or altering the original.
            with cfg["dump"].open("rb") as handle:
                command(home, ["neo4j-admin", "database", "load", "--from-stdin", "--overwrite-destination=false", "neo4j"],
                        cfg["load_timeout"], stdin=handle)
            started.append(home)
            command(home, ["neo4j", "start"], cfg["startup_timeout"])
            initialize_graph(cfg, arm)
        require(sha256(cfg["dump"]) == cfg["sha256"], "Source dump changed during restore; pair is not valid")
        report = summary(cfg)
        report.update(status="both_databases_restored_and_markers_verified", java_major=java)
        (root / "pair-manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        lines = ["# No passwords are stored here. Keep the original password environment.",
                 "export NEO4J_USER=neo4j", f"export NO_SMOKING_SNAPSHOT_SHA256={cfg['sha256']}"]
        for arm in ("off", "on"):
            lines += [f"export NO_SMOKING_{arm.upper()}_NEO4J_URI=bolt://127.0.0.1:{cfg['ports'][arm]}",
                      f"export NO_SMOKING_{arm.upper()}_NEO4J_DATABASE=neo4j"]
        (root / "pair.env").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    except BaseException:
        for home in reversed(started):
            try:
                command(home, ["neo4j", "stop"], 90)
            except Exception:
                print(f"Check the failed instance manually: {home}", file=sys.stderr)
        # Keep artifacts/logs for diagnosis. Never delete/reuse a partial root.
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate settings, hash, disk and ports; create nothing")
    args = parser.parse_args()
    try:
        os.umask(0o077)
        cfg = settings(os.environ)
        check_input(cfg)
        if args.dry_run:
            report = summary(cfg)
            report["status"] = "dry_run_validated_no_changes"
            print(json.dumps(report, indent=2))
        else:
            execute(cfg)
        return 0
    except ValueError as error:
        print(f"Neo4j bootstrap stopped: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        # Driver and subprocess exceptions may include credentials/query parameters.
        print(f"Neo4j bootstrap stopped ({type(error).__name__}); inspect private bootstrap-command.log files", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
