"""Frozen matched v53/v54 Stage1 screen. ``prepare`` makes no model calls.

This is a technical screen of first raw responses, not a policy-effect run.
Never retry a response or silently resume a partially written output: after an
interruption, preserve the JSONL as evidence and report an incomplete screen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts/sim"))
from planning_contract import inspect_schedule  # noqa: E402
from validate_prompt_v3 import contract, digest  # noqa: E402

PREREG = HERE / "candidate_v54_prereg_draft.json"
CANDIDATE_MANIFEST = HERE / "candidate_v54_example_manifest.json"
V3_VALIDATOR = ROOT / "scripts/sim/validate_prompt_v3.py"
FULL_VALIDATOR = ROOT / "scripts/sim/planning_contract.py"
VARIANTS = ("v53", "v54-example-contract")
FISCAL_OFF_PATTERN = re.compile(r"P01[234]|캐시백|쿠폰|바우처|지원금|상품권")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def dump_x(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def context_key(cell: dict) -> tuple[str, str, str]:
    return cell["aid"], cell["case"], cell["arm"]


def job_key(row: dict) -> tuple[str, str, str, str]:
    return row["variant"], row["aid"], row["case"], row["arm"]


def validate_source(prereg: dict, candidate_manifest: dict, source: dict,
                    candidate_bytes: bytes) -> None:
    assert prereg["contexts"] == 96 and prereg["run_size_if_started"] == 192
    assert prereg["replicate_seed"] == 1701 and prereg["temperature"] == 0.7
    assert prereg["top_p"] == 1.0 and prereg["max_tokens"] == 2200
    assert prereg["thinking"] is False and prereg["arms"] == ["off", "on"]
    if prereg["co_primary_thresholds"] != {
            "old_v3_min_pass": 92, "full_structural_min_pass": 92,
            "max_request_failures": 0, "max_fiscal_off_screen_flags": 0,
            "max_excess_paired_failures_vs_v53": 0}:
        raise ValueError("unrecognized preregistered co-primary thresholds")
    assert prereg["candidate_system_sha256"] == candidate_manifest["candidate_system_sha256"]
    assert prereg["base_system_sha256"] == candidate_manifest["base_system_sha256"]
    assert sha_bytes(candidate_bytes) == prereg["candidate_system_sha256"]
    assert candidate_manifest["model_calls"] == 0
    assert candidate_manifest["superseded_before_calls"] is True
    assert len(source["cells"]) == prereg["contexts"]
    keys = [context_key(cell) for cell in source["cells"]]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate frozen context")
    if len({key[0] for key in keys}) != 12:
        raise ValueError("expected exactly 12 frozen citizens")
    if {key[1] for key in keys} != {"cashback", "grant", "local_voucher", "distancing"}:
        raise ValueError("unexpected frozen policy cases")
    if {key[2] for key in keys} != {"off", "on"}:
        raise ValueError("unexpected arms")
    for cell in source["cells"]:
        if cell["context_sha256"] != digest(cell["user"]):
            raise ValueError("frozen user context digest mismatch")
        date.fromisoformat(cell["date"])
        if not isinstance(cell["zones"], list) or not cell["zones"]:
            raise ValueError("missing allowed zones")
    base = source["systems"]["v53"]
    if sha_bytes(base.encode("utf-8")) != prereg["base_system_sha256"]:
        raise ValueError("frozen v53 prompt hash mismatch")


def prepare(out: Path) -> dict:
    """Copy all frozen inputs and prompt bytes into a new immutable run folder."""
    prereg_bytes = PREREG.read_bytes()
    prereg = json.loads(prereg_bytes)
    candidate_manifest_bytes = CANDIDATE_MANIFEST.read_bytes()
    candidate_manifest = json.loads(candidate_manifest_bytes)
    source_path = ROOT / prereg["frozen_inputs_path"]
    source_bytes = source_path.read_bytes()
    if sha_bytes(source_bytes) != prereg["frozen_inputs_sha256"]:
        raise ValueError("source inputs changed since preregistration")
    source = json.loads(source_bytes)
    candidate_bytes = (ROOT / prereg["candidate_system_path"]).read_bytes()
    validate_source(prereg, candidate_manifest, source, candidate_bytes)
    base_bytes = source["systems"]["v53"].encode("utf-8")
    out.mkdir(parents=True, exist_ok=False)
    assets = {
        "prereg.json": prereg_bytes,
        "candidate_manifest.json": candidate_manifest_bytes,
        "frozen_inputs.json": source_bytes,
        "system_v53.txt": base_bytes,
        "system_v54.txt": candidate_bytes,
    }
    for name, data in assets.items():
        with (out / name).open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    manifest = {
        "schema": "v54_stage1_screen_frozen_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "192 first raw Stage1 responses; technical screen only",
        "assets_sha256": {name: sha_bytes(data) for name, data in assets.items()},
        "runner_sha256": sha_file(Path(__file__)),
        "v3_validator_sha256": sha_file(V3_VALIDATOR),
        "full_validator_sha256": sha_file(FULL_VALIDATOR),
        "base_system_sha256": prereg["base_system_sha256"],
        "candidate_system_sha256": prereg["candidate_system_sha256"],
        "expected_contexts": 96,
        "expected_responses": 192,
        "replicate_seed": 1701,
        "model": prereg["model"],
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 2200,
        "thinking": False,
        "co_primary_thresholds": prereg["co_primary_thresholds"],
    }
    dump_x(out / "freeze_manifest.json", manifest)
    return manifest


def verify_freeze(out: Path) -> tuple[dict, dict, dict[str, str]]:
    manifest = load_json(out / "freeze_manifest.json")
    if manifest["schema"] != "v54_stage1_screen_frozen_v1":
        raise ValueError("unknown freeze schema")
    for name, expected in manifest["assets_sha256"].items():
        if sha_file(out / name) != expected:
            raise ValueError(f"frozen asset changed: {name}")
    for path, key in ((Path(__file__), "runner_sha256"),
                      (V3_VALIDATOR, "v3_validator_sha256"),
                      (FULL_VALIDATOR, "full_validator_sha256")):
        if sha_file(path) != manifest[key]:
            raise ValueError(f"code changed after freeze: {path}")
    prereg = load_json(out / "prereg.json")
    candidate_manifest = load_json(out / "candidate_manifest.json")
    inputs = load_json(out / "frozen_inputs.json")
    systems = {"v53": (out / "system_v53.txt").read_text(encoding="utf-8"),
               "v54-example-contract": (out / "system_v54.txt").read_text(encoding="utf-8")}
    validate_source(prereg, candidate_manifest, inputs,
                    (out / "system_v54.txt").read_bytes())
    if systems["v53"] != inputs["systems"]["v53"]:
        raise ValueError("frozen v53 body changed")
    return manifest, inputs, systems


def verify_served_model(evidence_path: Path, expected_model: str,
                        base_url: str) -> tuple[bytes, dict]:
    evidence_bytes = evidence_path.read_bytes()
    evidence = json.loads(evidence_bytes)
    if evidence.get("served_model_ids") != [expected_model] or \
            f"--model-path {expected_model}" not in evidence.get("server_command", "") or \
            not isinstance(evidence.get("server_pid"), int):
        raise ValueError("server process evidence does not identify expected weights")
    captured = datetime.fromisoformat(evidence["captured_at_utc"])
    if captured.tzinfo is None:
        raise ValueError("server-process evidence lacks a timezone")
    age_seconds = (datetime.now(timezone.utc) - captured).total_seconds()
    if not -60 <= age_seconds <= 600:
        raise ValueError("server-process evidence must be freshly captured before the screen")
    with urlopen(base_url + "/models", timeout=10) as response:
        live_models = json.load(response)
    if [item.get("id") for item in live_models.get("data", [])] != [expected_model]:
        raise ValueError("live /models differs from server process evidence")
    return evidence_bytes, live_models


def workplace_minutes(events: list, weekend: bool, has_work: bool) -> int | None:
    if weekend or not has_work or not all(isinstance(e, dict) for e in events):
        return None
    try:
        minutes = [int(e["time"][:2]) * 60 + int(e["time"][3:]) for e in events]
    except (ValueError, TypeError, KeyError):
        return None
    if minutes != sorted(minutes):
        return None
    return sum(max(0, min(b, 18 * 60) - max(a, 9 * 60))
               for event, a, b in zip(events, minutes, minutes[1:])
               if event.get("anchor") == "workplace")


def inspect_raw(raw: str, cell: dict, persona: dict) -> dict:
    weekend = date.fromisoformat(cell["date"]).weekday() >= 5
    _, v3_errors = contract(raw, cell["zones"], weekend)
    full_cell = {**cell, "has_work": bool(persona.get("work_poi_id"))}
    obj, full_errors, flags = inspect_schedule(raw, full_cell)
    events = obj.get("events", []) if isinstance(obj, dict) else []
    if not isinstance(events, list):
        events = []
    off_flag = (cell["arm"] == "off" and cell["case"] != "distancing"
                and any(isinstance(event, dict) and event.get("trigger") == "policy"
                        and FISCAL_OFF_PATTERN.search(str(event.get("reasoning", "")))
                        for event in events))
    return {
        "v3_errors": sorted(set(v3_errors)),
        "full_structural_errors": sorted(set(full_errors)),
        "v3_valid": not v3_errors,
        "full_structural_valid": not full_errors,
        "event_count": len(events),
        "event_count_max_exceeded": len(events) > (8 if weekend else 10),
        "workplace_minutes_09_to_18": workplace_minutes(events, weekend, full_cell["has_work"]),
        "fiscal_off_attribution_screen": bool(off_flag),
        "other_screen_flags": [flag["kind"] for flag in flags
                               if flag["kind"] != "unsupported_fiscal_benefit_screen"],
    }


def invoke(job: tuple[int, str, dict], manifest: dict, systems: dict[str, str],
           personas: dict[str, dict], base_url: str) -> dict:
    ordinal, variant, cell = job
    started = time.monotonic()
    seed = int(digest([manifest["replicate_seed"], cell["aid"], cell["case"]])[:8], 16) % 2147483647
    row = {key: cell[key] for key in ("aid", "case", "arm", "date", "context_sha256")}
    row.update(ordinal=ordinal, variant=variant, seed=seed,
               system_prompt_sha256=manifest["base_system_sha256"] if variant == "v53"
               else manifest["candidate_system_sha256"])
    payload = {
        "model": manifest["model"],
        "messages": [{"role": "system", "content": systems[variant]},
                     {"role": "user", "content": cell["user"]}],
        "temperature": manifest["temperature"],
        "top_p": manifest["top_p"],
        "max_tokens": manifest["max_tokens"],
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": manifest["thinking"]},
    }
    try:
        request = Request(base_url + "/chat/completions", data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=300) as response:
            body = json.load(response)
        choice = body["choices"][0]
        raw = choice["message"]["content"]
        if not isinstance(raw, str):
            raise ValueError("response content is not a string")
        row.update(raw=raw, raw_sha256=sha_bytes(raw.encode("utf-8")),
                   finish_reason=choice.get("finish_reason"), usage=body.get("usage"),
                   response_model_field=body.get("model"),
                   **inspect_raw(raw, cell, personas[cell["aid"]]))
    except Exception as exc:
        row.update(request_error=f"{type(exc).__name__}: {exc}",
                   v3_valid=False, full_structural_valid=False,
                   v3_errors=["request_or_response"],
                   full_structural_errors=["request_or_response"],
                   fiscal_off_attribution_screen=False)
    row["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return row


def scheduled_jobs(cells: list[dict], seed: int) -> list[tuple[int, str, dict]]:
    rng = random.Random(seed)
    ordered = list(cells)
    rng.shuffle(ordered)
    result = []
    for cell in ordered:
        variants = list(VARIANTS)
        rng.shuffle(variants)
        for variant in variants:
            result.append((len(result), variant, cell))
    return result


def summarize(rows: list[dict], manifest: dict, cells: list[dict]) -> dict:
    expected = {(variant, *context_key(cell)) for variant in VARIANTS for cell in cells}
    counts = Counter(job_key(row) for row in rows)
    complete = (len(rows) == manifest["expected_responses"] and set(counts) == expected
                and all(count == 1 for count in counts.values()))
    by_variant = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        by_variant[variant] = {
            "responses": len(selected),
            "v3_pass": sum(bool(row["v3_valid"]) for row in selected),
            "full_structural_pass": sum(bool(row["full_structural_valid"]) for row in selected),
            "request_failures": sum("request_error" in row for row in selected),
            "fiscal_off_attribution_screen_count": sum(bool(row.get("fiscal_off_attribution_screen"))
                                                         for row in selected),
            "event_count_max_exceeded": sum(bool(row.get("event_count_max_exceeded"))
                                             for row in selected),
            "v3_error_counts": dict(Counter(error for row in selected
                                            for error in row["v3_errors"])),
            "full_error_counts": dict(Counter(error for row in selected
                                              for error in row["full_structural_errors"])),
        }
    paired = {job_key(row): row for row in rows}
    discordance = {}
    for metric in ("v3_valid", "full_structural_valid"):
        new_only, old_only = 0, 0
        for cell in cells:
            old = paired.get(("v53", *context_key(cell)))
            new = paired.get(("v54-example-contract", *context_key(cell)))
            if old is None or new is None:
                continue
            new_only += bool(new[metric]) and not bool(old[metric])
            old_only += bool(old[metric]) and not bool(new[metric])
        discordance[metric] = {"v54_only_pass": new_only, "v53_only_pass": old_only}
    new = by_variant["v54-example-contract"]
    thresholds = manifest["co_primary_thresholds"]
    max_excess = thresholds["max_excess_paired_failures_vs_v53"]
    v3_gate = (complete and new["v3_pass"] >= thresholds["old_v3_min_pass"]
               and new["request_failures"] <= thresholds["max_request_failures"]
               and new["fiscal_off_attribution_screen_count"] <=
               thresholds["max_fiscal_off_screen_flags"]
               and discordance["v3_valid"]["v53_only_pass"] -
               discordance["v3_valid"]["v54_only_pass"] <= max_excess)
    full_gate = (complete and new["full_structural_pass"] >=
                 thresholds["full_structural_min_pass"]
                 and discordance["full_structural_valid"]["v53_only_pass"] -
                 discordance["full_structural_valid"]["v54_only_pass"] <= max_excess)
    return {
        "schema": "v54_stage1_screen_summary_v1",
        "scope": "first raw Stage1 format, not Stage2 or policy effect",
        "complete_unique_matrix": complete,
        "expected_responses": manifest["expected_responses"],
        "variants": by_variant,
        "paired_discordance": discordance,
        "co_primary": {"old_v3_contract_pass": bool(v3_gate),
                       "full_structural_contract_pass": bool(full_gate),
                       "overall_pass": bool(v3_gate and full_gate)},
    }


def execute(out: Path, evidence_path: Path, base_url: str, workers: int) -> None:
    if workers < 1 or workers > 24:
        raise ValueError("workers must be 1..24")
    manifest, inputs, systems = verify_freeze(out)
    if (out / "responses.jsonl").exists():
        raise ValueError("responses exist; refusing overwrite, retry or implicit resume")
    if any((out / name).exists() for name in ("served_model_evidence.json", "execution_provenance.json",
                                             "summary.json")):
        raise ValueError("run output already exists")
    base_url = base_url.rstrip("/")
    evidence_bytes, live_models = verify_served_model(evidence_path, manifest["model"], base_url)
    with (out / "served_model_evidence.json").open("xb") as stream:
        stream.write(evidence_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    dump_x(out / "execution_provenance.json", {
        "served_model_evidence_sha256": sha_bytes(evidence_bytes),
        "live_models": live_models,
        "live_models_sha256": digest(live_models),
        "base_url": base_url,
        "workers": workers,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Response model field is not proof of loaded weights; process and live /models are.",
    })
    jobs = scheduled_jobs(inputs["cells"], manifest["replicate_seed"])
    personas = {p["id"]: p for p in inputs["personas"]}
    rows = []
    with (out / "responses.jsonl").open("x", encoding="utf-8", newline="\n") as stream, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(invoke, job, manifest, systems, personas, base_url)
                   for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            if len(rows) % 24 == 0:
                print(f"recorded {len(rows)}/{len(jobs)} first responses", flush=True)
    summary = summarize(rows, manifest, inputs["cells"])
    summary["responses_sha256"] = sha_file(out / "responses.jsonl")
    summary["execution_provenance_sha256"] = sha_file(out / "execution_provenance.json")
    summary["freeze_manifest_sha256"] = sha_file(out / "freeze_manifest.json")
    summary["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    dump_x(out / "summary.json", summary)
    print(json.dumps({"co_primary": summary["co_primary"],
                      "variants": summary["variants"]}, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--served-model-evidence", type=Path)
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.mode == "prepare":
        manifest = prepare(args.out)
        print(f"Frozen {manifest['expected_contexts']} contexts and two prompt bodies; model calls: 0")
    else:
        if args.served_model_evidence is None:
            parser.error("run requires --served-model-evidence")
        execute(args.out, args.served_model_evidence, args.base_url, args.workers)


if __name__ == "__main__":
    main()
