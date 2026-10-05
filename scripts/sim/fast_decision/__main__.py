"""Offline tools. No model is loaded unless replay is explicitly selected."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from .contracts import DEFAULT_MODEL, fingerprint, canonical
from .planner import propose, teacher_examples
from .runtime import append_record


def smoke_record():
    """Handwritten engineering fixture, never a real-world or teacher label."""
    snapshot = {
        "aid": "synthetic-001", "today": "2026-01-05",
        "stage1": {"events": [{"time": "12:00", "anchor": "zone:11680580",
                               "category": "식사", "intent": "점심", "trigger": "lifestyle"}]},
        "persona": {"income": "중", "daily_wd": 30000, "home_dong_code": "11680580"},
        "state": {"mood": .6, "fatigue": .3, "balance": 100000},
        "candidates": {"0": [
            {"poi_id": "fixture-a", "name": "예시 한식", "known": True, "avg_satisfaction": .7,
             "unit_anchor": 10000, "price_band": 2, "km": .2},
            {"poi_id": "fixture-b", "name": "예시 국수", "known": False, "avg_satisfaction": None,
             "unit_anchor": 10000, "price_band": 2, "km": .3},
        ]},
        "active_policies": [], "grant_remaining": {}, "recent_poi_ids": [],
        "context": {"memory": [], "appointment": [], "social": []},
        "system_prompt": "synthetic fixture", "user_prompt": "synthetic fixture",
    }
    snapshot["snapshot_id"] = fingerprint(snapshot)
    output = {"picks": [{"order": 0, "poi_id": "fixture-a", "actual_spent": 10000,
                         "actual_satisfaction": .7, "pick_factor": "known", "policy_spend": {}}]}
    return {"schema_version": 1, "snapshot": snapshot,
            "teacher": {"model_id": DEFAULT_MODEL, "output": output, "meta": {"validated": True}},
            "provenance": {"synthetic": True, "source": "handwritten_engineering_fixture_not_model_output"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="No weights/GPU/network; validate synthetic pipeline")
    smoke.add_argument("--output", type=Path, required=True)
    replay = commands.add_parser("replay", help="Explicit offline inference against captured snapshots")
    replay.add_argument("--input", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--model", default=DEFAULT_MODEL)
    replay.add_argument("--revision", required=True, help="Pinned HF commit SHA")
    replay.add_argument("--adapter")
    replay.add_argument("--device", default="cpu")
    replay.add_argument("--max-tokens", type=int, default=4096)
    replay.add_argument("--allow-download", action="store_true")
    replay.add_argument("--selection", choices=("argmax", "sample"), default="argmax")
    replay.add_argument("--seed", type=int, default=0)
    replay.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists; choose a new experiment path")
    if args.command == "smoke":
        row = smoke_record()
        examples = teacher_examples(row["snapshot"], row["teacher"]["output"])
        append_record(args.output, row)
        print(canonical({"status": "ok", "synthetic": True, "questions": len(examples),
                         "weights_loaded": False, "gpu_used": False, "eligible_for_live": False}))
        return 0
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must differ")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    from .backend import ExaoneChoiceBackend
    backend = ExaoneChoiceBackend(model_id=args.model, revision=args.revision,
                                  adapter_path=args.adapter, device=args.device,
                                  max_tokens=args.max_tokens, allow_download=args.allow_download)
    count, errors = 0, 0
    with args.input.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            started = time.perf_counter()
            try:
                student = propose(row["snapshot"], backend, selection=args.selection, seed=args.seed)
            except Exception as exc:
                errors += 1
                student = {"status": "error", "error_type": type(exc).__name__, "error": str(exc),
                           "eligible_for_live": False}
            student.update(latency_seconds=time.perf_counter() - started,
                           model_id=args.model, revision=args.revision)
            row["student"] = student
            append_record(args.output, row)
            count += 1
            if args.limit and count >= args.limit:
                break
    print(canonical({"records": count, "errors": errors, "eligible_for_live": False}))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
