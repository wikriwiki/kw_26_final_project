"""Verify the completed local screen and preserve identical C:/G: evidence."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("C:/Users/Administrator/Documents/kw26_a100_recovery_20260926/") / "multipolicy_v53_20260928/v54_stage1_screen"
DESTINATION = ROOT / "output/validation_v54_example_frozen_20260928"
NAMES = (
    "prereg.json", "candidate_manifest.json", "frozen_inputs.json",
    "system_v53.txt", "system_v54.txt", "freeze_manifest.json",
    "served_model_before_run.json", "served_model_evidence.json",
    "execution_provenance.json", "responses.jsonl", "summary.json", "audit.json",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_copy(source: Path, destination: Path) -> str:
    expected = sha(source)
    for attempt in range(5):
        try:
            shutil.copyfile(source, destination)
            if sha(destination) != expected:
                raise ValueError(f"SHA256 mismatch: {destination}")
            return expected
        except (OSError, ValueError):
            if attempt == 4:
                raise
            time.sleep(1)
    raise AssertionError("unreachable")


def main() -> None:
    summary = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads((SOURCE / "audit.json").read_text(encoding="utf-8"))
    if not summary["complete_unique_matrix"] or summary["expected_responses"] != 192:
        raise ValueError("screen is not a complete first-response matrix")
    if not audit["integrity"].startswith("PASS:"):
        raise ValueError("independent raw-response audit did not pass")
    if summary["co_primary"] != audit["co_primary"]:
        raise ValueError("summary and independent audit differ")
    files = []
    for name in NAMES:
        source, destination = SOURCE / name, DESTINATION / name
        if name in audit["files_sha256"] and sha(source) != audit["files_sha256"][name]:
            raise ValueError(f"audit hash changed: {name}")
        if destination.exists() and name in NAMES[:6] and sha(destination) != sha(source):
            raise ValueError(f"original frozen input differs: {name}")
        digest = verified_copy(source, destination)
        files.append({"file": name, "sha256": digest,
                      "c_path": str(source), "g_path": str(destination)})
    record = {"schema": "v54_stage1_screen_preservation_v1",
              "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "execution_primary": "C:", "copies_sha256_match": True,
              "policy_effect_claim": False, "files": files}
    manifest = SOURCE / "preservation_manifest.json"
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    verified_copy(manifest, DESTINATION / manifest.name)
    print(json.dumps({"verified_files": len(files), "manifest_sha256": sha(manifest),
                      "co_primary": summary["co_primary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
