"""Read-only independent audit of all 192 saved first raw v53/v54 responses."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from run_v54_stage1_screen import (VARIANTS, context_key, inspect_raw, job_key,
                                    load_json, sha_file, summarize, verify_freeze, digest)


def audit(folder: Path) -> dict:
    manifest, inputs, _ = verify_freeze(folder)
    summary_path = folder / "summary.json"
    responses_path = folder / "responses.jsonl"
    if not summary_path.exists() or not responses_path.exists():
        raise ValueError("screen incomplete: no completed summary and response file")
    summary = load_json(summary_path)
    if summary.get("responses_sha256") != sha_file(responses_path):
        raise ValueError("raw response file SHA mismatch")
    if summary.get("freeze_manifest_sha256") != sha_file(folder / "freeze_manifest.json"):
        raise ValueError("freeze manifest SHA mismatch")
    provenance = load_json(folder / "execution_provenance.json")
    if summary.get("execution_provenance_sha256") != sha_file(folder / "execution_provenance.json"):
        raise ValueError("execution provenance SHA mismatch")
    if provenance["served_model_evidence_sha256"] != sha_file(folder / "served_model_evidence.json"):
        raise ValueError("served model process snapshot SHA mismatch")
    evidence = load_json(folder / "served_model_evidence.json")
    if evidence.get("served_model_ids") != [manifest["model"]] or \
            f"--model-path {manifest['model']}" not in evidence.get("server_command", ""):
        raise ValueError("served model process evidence mismatch")
    if [model.get("id") for model in provenance["live_models"].get("data", [])] != [manifest["model"]]:
        raise ValueError("live /models evidence mismatch")
    if digest(provenance["live_models"]) != provenance["live_models_sha256"]:
        raise ValueError("live /models digest mismatch")
    rows = [json.loads(line) for line in responses_path.read_text(encoding="utf-8").splitlines()]
    expected = {(variant, *context_key(cell)) for variant in VARIANTS for cell in inputs["cells"]}
    counts = Counter(job_key(row) for row in rows)
    if len(rows) != 192 or set(counts) != expected or any(count != 1 for count in counts.values()):
        raise ValueError("incomplete, duplicate or unexpected first-response matrix")
    if {row["ordinal"] for row in rows} != set(range(192)):
        raise ValueError("job ordinal matrix mismatch")
    cells = {context_key(cell): cell for cell in inputs["cells"]}
    personas = {persona["id"]: persona for persona in inputs["personas"]}
    invalid_keys, request_error_keys, fiscal_off_keys = [], [], []
    for row in rows:
        cell = cells[(row["aid"], row["case"], row["arm"])]
        variant = row["variant"]
        expected_prompt_sha = manifest["base_system_sha256"] if variant == "v53" else manifest["candidate_system_sha256"]
        if row["system_prompt_sha256"] != expected_prompt_sha or \
                row["context_sha256"] != cell["context_sha256"] or \
                row["date"] != cell["date"]:
            raise ValueError("row prompt, context or calendar provenance mismatch")
        seed = int(digest([manifest["replicate_seed"], cell["aid"], cell["case"]])[:8], 16) % 2147483647
        if row["seed"] != seed:
            raise ValueError("paired seed mismatch")
        ident = {key: row[key] for key in ("variant", "aid", "case", "arm")}
        if "request_error" in row:
            request_error_keys.append(ident)
            if "raw" in row or row["v3_valid"] or row["full_structural_valid"] or \
                    row["v3_errors"] != ["request_or_response"] or \
                    row["full_structural_errors"] != ["request_or_response"]:
                raise ValueError("request error counted as raw success")
        else:
            from run_v54_stage1_screen import sha_bytes
            if sha_bytes(row["raw"].encode("utf-8")) != row["raw_sha256"]:
                raise ValueError("row raw response digest mismatch")
            recomputed = inspect_raw(row["raw"], cell, personas[row["aid"]])
            if any(row.get(key) != value for key, value in recomputed.items()):
                raise ValueError("stored validity differs from unmodified raw response")
        if not row["v3_valid"] or not row["full_structural_valid"]:
            invalid_keys.append({**ident, "v3_errors": row["v3_errors"],
                                 "full_errors": row["full_structural_errors"]})
        if row.get("fiscal_off_attribution_screen"):
            fiscal_off_keys.append(ident)
    calculated = summarize(rows, manifest, inputs["cells"])
    for key, value in calculated.items():
        if summary.get(key) != value:
            raise ValueError(f"saved summary differs from original raw responses: {key}")
    if not calculated["complete_unique_matrix"]:
        raise ValueError("saved summary was not complete")
    return {
        "schema": "v54_stage1_screen_audit_v1",
        "integrity": "PASS: 192 unique first responses, prompt/input/model evidence hashes and both contracts reproduced",
        "policy_effect_claim": False,
        "co_primary": calculated["co_primary"],
        "variants": calculated["variants"],
        "paired_discordance": calculated["paired_discordance"],
        "request_error_keys": request_error_keys,
        "fiscal_off_screen_keys": fiscal_off_keys,
        "raw_invalid_keys": invalid_keys,
        "files_sha256": {name: sha_file(folder / name) for name in (
            "prereg.json", "candidate_manifest.json", "frozen_inputs.json",
            "system_v53.txt", "system_v54.txt", "freeze_manifest.json",
            "served_model_evidence.json", "execution_provenance.json",
            "responses.jsonl", "summary.json")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"integrity": result["integrity"],
                      "co_primary": result["co_primary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
