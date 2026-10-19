#!/usr/bin/env python3
"""Archive the current code/config snapshot, including new experiment files.

Git tracked files and unignored new files under runtime allowlists are eligible.
The simulation's output/stats JSON references are explicitly included and hashed.
Private env/key files, databases, original documents and .git are never sent.
Experiment bundles and database dumps are transferred separately.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
SCOPES = ("scripts", "data/experiments", "data/neo4j_load/categories",
          "data/neo4j_load/mapping", "data/neo4j_load/policies", "deploy/vast", "requirements.txt")
SUFFIXES = {".py", ".sh", ".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".example"}
REQUIRED_REFERENCES = ("unit_price.json", "dong_context.json", "hub_catalog.json", "dong_centroids.json", "hub_signature.json")
OPTIONAL_REFERENCES = ("poi_menu_price.json",)


def permitted(path: Path) -> bool:
    return (not path.is_symlink() and path.is_file()
            and (path.suffix in SUFFIXES or path.name == ".gitattributes")
            and not any(part in {"local", "__pycache__", "node_modules", ".git"} for part in path.parts)
            and (not path.name.startswith(".env") or path.name == ".env.example")
            and not path.name.startswith(("id_rsa", "id_ed25519")))


def collect_references(root: Path) -> dict[str, str | None]:
    """Fail before packaging if a statistical input would silently fall back."""
    root = root.resolve()
    stats = root / "output/stats"
    missing = [name for name in REQUIRED_REFERENCES if not (stats / name).is_file()]
    if missing:
        raise ValueError("missing required output/stats references: " + ", ".join(missing))
    references = {}
    for path in sorted(stats.glob("*.json")):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"reference must be a regular file inside the repository: {path.name}")
        content = path.read_bytes()
        if content.lstrip().startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"Git LFS pointer is not usable input: {path.name}; fetch the actual file")
        try:
            json.loads(content.decode("utf-8-sig"))
        except (UnicodeError, ValueError) as exc:
            raise ValueError(f"invalid statistics JSON: {path.name}") from exc
        references[path.relative_to(root).as_posix()] = hashlib.sha256(content).hexdigest()
    for name in OPTIONAL_REFERENCES:
        references.setdefault(f"output/stats/{name}", None)
    return references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    target = args.out.resolve()
    if target.exists():
        parser.error("output exists; choose a new archive filename")
    result = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *SCOPES],
                            cwd=ROOT, capture_output=True, check=True)
    files = sorted(set(name.decode("utf-8") for name in result.stdout.split(b"\0") if name))
    files = [name for name in files if permitted(ROOT / name)]
    try:
        references = collect_references(ROOT)
    except ValueError as exc:
        parser.error(str(exc))
    files = sorted(set(files) | {name for name, sha in references.items() if sha is not None})
    for required in ("scripts/experiments/no_smoking_zone.py", "deploy/vast/run_experiment.sh"):
        if required not in files:
            parser.error(f"missing required source file: {required}")
    target.parent.mkdir(parents=True, exist_ok=True)
    file_sha256 = {}
    with tarfile.open(target, "x:gz") as archive:
        for name in files:
            if name.endswith(".sh"):
                # Windows working trees may still contain CRLF in legacy launchers.
                content = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
                file_sha256[name] = hashlib.sha256(content).hexdigest()
                item = archive.gettarinfo(ROOT / name, arcname=name)
                item.size = len(content)
                archive.addfile(item, io.BytesIO(content))
            else:
                file_sha256[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                archive.add(ROOT / name, arcname=name, recursive=False)
        manifest = json.dumps({"kind": "source_and_statistics", "reference_sha256": references,
                               "shell_newline_normalization": "CRLF to LF in all .sh archive members",
                               "source_file_count": len(files), "file_sha256": file_sha256,
                               "experiment_bundle_included": False,
                               "database_dump_included": False}, indent=2).encode("utf-8") + b"\n"
        item = tarfile.TarInfo("deployment-manifest.json")
        item.size = len(manifest)
        item.mode = 0o600
        archive.addfile(item, io.BytesIO(manifest))
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(target.suffix + ".sha256").write_text(f"{digest}  {target.name}\n", encoding="ascii")
    print(json.dumps({"archive": str(target), "sha256": digest, "file_count": len(files),
                      "includes_statistics": True, "statistics_file_count": sum(v is not None for v in references.values()),
                      "includes_experiment_bundle": False, "includes_database_dump": False,
                      "excludes_private_env_and_keys": True}, indent=2))


if __name__ == "__main__":
    main()
