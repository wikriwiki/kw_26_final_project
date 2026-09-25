"""Freeze a new prompt pilot using previously audited, identical contexts.

Only candidates and decoding/decision settings may change. This performs no
Neo4j or model calls; validate_prompt_v3.py later runs the frozen job matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/sim"))
from validate_prompt_v3 import atomic, digest, source_hashes  # noqa: E402

SAME_CONTEXT = ("model", "sample_n", "replicate_seeds", "cases", "arms",
                "max_tokens", "initial_state", "cashback_seed", "design")
ALLOWED_CHANGES = {"id", "registered_on", "candidates", "temperature", "phase",
                   "decision", "inference", "holdout", "stop_rule", "power",
                   "predicted", "not_changed"}


def prepare(source: Path, config_path: Path, destination: Path,
            *, allow_new_candidates: bool = False) -> dict:
    old_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    old_inputs = json.loads((source / "frozen_inputs.json").read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if digest(old_manifest["config"]) != old_manifest["config_sha256"]:
        raise ValueError("source registration hash mismatch")
    if digest(old_inputs) != old_manifest["inputs_sha256"]:
        raise ValueError("source frozen inputs hash mismatch")
    if any(config.get(key) != old_manifest["config"].get(key) for key in SAME_CONTEXT):
        raise ValueError("new registration changes a frozen context field")
    if any(config.get(key) != old_manifest["config"].get(key)
           for key in set(config) | set(old_manifest["config"])
           if key not in ALLOWED_CHANGES):
        raise ValueError("new registration changes an unapproved execution setting")
    variants = config["candidates"]
    if not variants or len(variants) != len(set(variants)):
        raise ValueError("candidates must be nonempty and unique")
    if not allow_new_candidates and any(v not in old_inputs["systems"] for v in variants):
        raise ValueError("new candidates require explicit --allow-new-candidates")

    from prompts import get  # noqa: E402
    systems = {v: get(v).SYSTEM_PROMPT for v in variants}
    if any(v in old_inputs["systems"] and old_inputs["systems"][v] != systems[v]
           for v in variants):
        raise ValueError("current candidate text differs from frozen prompt")
    inputs = {**old_inputs, "systems": systems}
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    manifest = {
        "config": config,
        "config_sha256": digest(config),
        "inputs_sha256": digest(inputs),
        "script_sha256": hashlib.sha256((ROOT / "scripts/sim/validate_prompt_v3.py").read_bytes()).hexdigest(),
        "system_hashes": {v: digest(s) for v, s in systems.items()},
        "source_hashes": source_hashes(),
        "frozen_from": {
            "manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
            "inputs_sha256": hashlib.sha256((source / "frozen_inputs.json").read_bytes()).hexdigest(),
        },
    }
    atomic(destination / "frozen_inputs.json", inputs)
    atomic(destination / "manifest.json", manifest)
    return {"contexts": len(inputs["cells"]), "candidates": variants,
            "config_sha256": manifest["config_sha256"],
            "inputs_sha256": manifest["inputs_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-new-candidates", action="store_true",
                        help="freeze newly registered prompt text with the old contexts")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.config, args.out,
                             allow_new_candidates=args.allow_new_candidates), ensure_ascii=False))


if __name__ == "__main__":
    main()
