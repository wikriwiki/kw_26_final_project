"""Frozen, read-only Stage1 decoding screen for an unchanged v53 prompt."""
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
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/sim"))

from planning_contract import schedule_schema  # noqa: E402
from prompts import get  # noqa: E402
from validate_prompt_v3 import atomic, contract, digest, source_hashes  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_or_verify(out: Path, config: dict) -> dict:
    source = ROOT / config["source_inputs"]
    if sha256(source) != config["source_inputs_sha256"]:
        raise ValueError("Source frozen-input bytes changed")
    source_inputs = json.loads(source.read_text(encoding="utf-8"))
    if len(source_inputs["cells"]) != config["expected_contexts"]:
        raise ValueError("Wrong number of frozen contexts")
    system = get(config["prompt"]).SYSTEM_PROMPT
    if source_inputs["systems"][config["prompt"]] != system:
        raise ValueError("Candidate body differs from source frozen inputs")
    inputs = {
        "personas": source_inputs["personas"],
        "cells": source_inputs["cells"],
        "systems": {config["prompt"]: system},
    }
    expected = {
        "config_sha256": digest(config),
        "source_inputs_sha256": config["source_inputs_sha256"],
        "inputs_sha256": digest(inputs),
        "system_sha256": digest(system),
        "runner_sha256": sha256(Path(__file__)),
        "source_hashes": source_hashes(ROOT),
    }
    out.mkdir(parents=True, exist_ok=True)
    frozen_path, manifest_path = out / "frozen_inputs.json", out / "manifest.json"
    if frozen_path.exists() or manifest_path.exists():
        if not (frozen_path.exists() and manifest_path.exists()):
            raise ValueError("Incomplete freeze; refusing to overwrite")
        if json.loads(frozen_path.read_text(encoding="utf-8")) != inputs:
            raise ValueError("Frozen inputs changed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest != expected:
            raise ValueError("Manifest or source changed after freezing")
    else:
        atomic(frozen_path, inputs)
        atomic(manifest_path, expected)
    return inputs


def invoke(job: tuple[dict, dict], config: dict, system: str,
           personas: dict, base: str) -> dict:
    candidate, cell = job
    started = time.monotonic()
    seed = int(digest([config["replicate_seed"], cell["aid"], cell["case"]])[:8], 16) % 2147483647
    result = {key: cell[key] for key in ("aid", "case", "arm", "date", "context_sha256")}
    result.update(candidate=candidate["id"], seed=seed,
                  temperature=candidate["temperature"],
                  structured=candidate["structured"])
    payload = {
        "model": config["model"],
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": cell["user"]}],
        "temperature": candidate["temperature"], "top_p": config["top_p"],
        "max_tokens": config["max_tokens"], "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if candidate["structured"]:
        schema = schedule_schema(
            cell["zones"], date.fromisoformat(cell["date"]).weekday() >= 5,
            bool(personas[cell["aid"]].get("work_poi_id")),
        )
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "citizen_day", "strict": True, "schema": schema},
        }
        result["schema_sha256"] = digest(schema)
    try:
        req = Request(base + "/chat/completions", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=300) as response:
            body = json.load(response)
        choice = body["choices"][0]
        raw = choice["message"]["content"]
        obj, errors = contract(raw, cell["zones"],
                               date.fromisoformat(cell["date"]).weekday() >= 5)
        events = (obj or {}).get("events") or []
        result.update(
            raw=raw, errors=errors, valid=not errors,
            finish_reason=choice.get("finish_reason"), usage=body.get("usage"),
            propensity=(obj or {}).get("daily_propensity"),
            fiscal_off_attributions=sum(
                e.get("trigger") == "policy"
                and bool(re.search(r"P01[234]|캐시백|쿠폰|바우처|지원금|상품권",
                                   str(e.get("reasoning", ""))))
                for e in events if isinstance(e, dict)
            ) if cell["arm"] == "off" and cell["case"] != "distancing" else 0,
            excess_events=len(events) > (8 if date.fromisoformat(cell["date"]).weekday() >= 5 else 10),
        )
    except Exception as exc:
        result.update(valid=False, errors=["request_or_response"], error=str(exc))
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return result


def summarize(rows: list[dict], config: dict, cells: list[dict]) -> dict:
    expected = {(candidate["id"], cell["aid"], cell["case"], cell["arm"])
                for candidate in config["candidates"] for cell in cells}
    counts = Counter((row["candidate"], row["aid"], row["case"], row["arm"])
                     for row in rows)
    result = {"scope": config["phase"], "complete_unique_matrix":
              set(counts) == expected and all(n == 1 for n in counts.values()),
              "policy_effect_claim": False, "candidates": {}}
    for candidate in config["candidates"]:
        rr = [row for row in rows if row["candidate"] == candidate["id"]]
        valid = sum(bool(row["valid"]) for row in rr)
        result["candidates"][candidate["id"]] = {
            "responses": len(rr), "strict_valid": valid,
            "strict_valid_rate": valid / len(rr) if rr else 0,
            "request_failures": sum("error" in row for row in rr),
            "fiscal_off_attribution_responses": sum(bool(row.get("fiscal_off_attributions")) for row in rr),
            "error_counts": dict(Counter(error for row in rr for error in row["errors"])),
            "finish_reasons": dict(Counter(row.get("finish_reason", "missing") for row in rr)),
            "excess_events": sum(bool(row.get("excess_events")) for row in rr),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("PYTHONHASHSEED=0 required before process starts")
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    out = Path(args.out)
    inputs = freeze_or_verify(out, config)
    if args.prepare_only:
        print(f"Frozen {len(inputs['cells'])} contexts. No LLM calls.")
        return
    if (out / "responses.jsonl").exists():
        raise SystemExit("Existing run; refusing overwrite or implicit retry")
    base = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    with urlopen(base + "/models", timeout=10) as response:
        models = json.load(response)
    if config["model"] not in [model["id"] for model in models["data"]]:
        raise SystemExit("Wrong served model")
    jobs = [(candidate, cell) for candidate in config["candidates"]
            for cell in inputs["cells"]]
    random.Random(20260926).shuffle(jobs)
    personas = {persona["id"]: persona for persona in inputs["personas"]}
    rows = []
    with (out / "responses.jsonl").open("x", encoding="utf-8") as stream, \
            ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        futures = [pool.submit(invoke, job, config, inputs["systems"][config["prompt"]],
                               personas, base) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            if len(rows) % 24 == 0:
                print(f"completed {len(rows)}/{len(jobs)}", flush=True)
    summary = summarize(rows, config, inputs["cells"])
    atomic(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
