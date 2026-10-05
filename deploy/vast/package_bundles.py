"""Package a validated full/pilot no-smoking pair without overwriting it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.experiments.no_smoking_zone import inspect_bundle

FILES = ("assignment_audit.json", "bundle.json", "experiment_cohort_ids.json",
         "personas.json", "preflight.json", "runtime.json", "smoking_rates.json")


def sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def package(full: Path, pilot: Path, target: Path, manifest_path: Path) -> dict:
    if target.exists() or manifest_path.exists():
        raise ValueError("Output exists; choose a new release name")
    if full.resolve() == pilot.resolve() or full.name == pilot.name:
        raise ValueError("Full and pilot bundle paths/names must differ")
    full_check, pilot_check = inspect_bundle(full), inspect_bundle(pilot)
    if not full_check["preparation_ready"] or not pilot_check["preparation_ready"]:
        raise ValueError("Both bundle preflights must pass")
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    full_runtime, pilot_runtime = read(full / "runtime.json"), read(pilot / "runtime.json")
    full_ids = read(full / "experiment_cohort_ids.json")
    pilot_ids = read(pilot / "experiment_cohort_ids.json")
    if full_ids != pilot_ids or {r["id"] for r in full_runtime["cohort"]} != set(full_ids):
        raise ValueError("Full runtime and frozen experiment roster disagree")
    if not {r["id"] for r in pilot_runtime["cohort"]} < set(full_ids):
        raise ValueError("Pilot must be a proper subset of the same roster")
    for field in ("assignment_seed", "simulation_seed", "pois"):
        if full_runtime[field] != pilot_runtime[field]:
            raise ValueError(f"Full and pilot {field} disagree")

    members = {}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, "x:gz") as archive:
        for folder in (full, pilot):
            for name in FILES:
                path = folder / name
                if not path.is_file() or path.is_symlink():
                    raise ValueError(f"Missing or linked bundle file: {path}")
                arcname = f"{folder.name}/{name}"
                members[arcname] = sha(path.read_bytes())
                archive.add(path, arcname=arcname, recursive=False)
    with tarfile.open(target, "r:gz") as archive:
        if sorted(archive.getnames()) != sorted(members):
            raise ValueError("Packaged member list disagrees with input")
        for member in archive:
            if sha(archive.extractfile(member).read()) != members[member.name]:
                raise ValueError(f"Packaged content differs: {member.name}")
    manifest = {"archive": target.name, "sha256": sha(target.read_bytes()),
                "bytes": target.stat().st_size, "experiment_agents": len(full_ids),
                "pilot_agents": len(pilot_runtime["cohort"]), "members": members,
                "database_dump_included": False}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = package(args.full, args.pilot, args.out,
                         args.out.with_suffix(".manifest.json"))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Bundle packaging stopped: {exc}\n")
    print(json.dumps({key: result[key] for key in ("archive", "sha256", "bytes", "experiment_agents", "pilot_agents")}, indent=2))


if __name__ == "__main__":
    main()
