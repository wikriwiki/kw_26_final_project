"""Frozen, read-only Stage2 technical screen for the completed v53 Stage1 cells.

prepare: query the existing graph only for POI candidates and freeze them.
run: replay frozen candidates against the existing model, preserving every raw reply.
No policy arm, graph write, scoring target, or prompt search occurs here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
sys.path.insert(0, str(ROOT / "scripts"))
from stage1_intent import Stage1Output  # noqa: E402
from scripts.sim import stage2_poi as s2  # noqa: E402

CONFIG = ROOT / "data/experiments/validation_v53_stage2_smoke_20260926.json"
SOURCES = [
    CONFIG,
    Path(__file__),
    ROOT / "scripts/sim/stage2_poi.py",
    ROOT / "scripts/sim/eligibility.py",
    ROOT / "scripts/sim/prompts/v53.py",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _source_hashes(source_dir: Path) -> dict[str, str]:
    files = SOURCES + [source_dir / "frozen_inputs.json", source_dir / "responses.jsonl"]
    return {str(p.relative_to(ROOT)): _sha(p) for p in files}


def _load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _external_count(stage1: Stage1Output) -> int:
    return sum(ev.category not in s2.INTERNAL_CATS and not ev.pinned_poi
               for ev in stage1.events)


def prepare() -> None:
    cfg = _load_config()
    os.environ["SIM_PROMPT_VARIANT"] = "v53"
    source = ROOT / cfg["stage1_source"]
    output = ROOT / cfg["output"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "inputs.json").exists() or (output / "manifest.json").exists():
        raise RuntimeError("Frozen input already exists; do not overwrite it")

    frozen = json.loads((source / "frozen_inputs.json").read_text(encoding="utf-8"))
    personas = {p["id"]: p for p in frozen["personas"]}
    cells = {(c["case"], c["arm"], c["aid"]): c for c in frozen["cells"]}
    responses = [json.loads(line) for line in (source / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
    by_key = {(r["case"], r["arm"], r["aid"], r["candidate"]): r for r in responses}
    rows = []
    selection_log = []

    for case in cfg["cases"]:
        aids = sorted(aid for c, arm, aid in cells if c == case and arm == cfg["arm"])
        aids.sort(key=lambda aid: hashlib.sha256(
            f'{cfg["selection_seed"]}:{case}:{aid}'.encode("utf-8")
        ).hexdigest())
        selected = 0
        for aid in aids:
            cell = cells[(case, cfg["arm"], aid)]
            pair = []
            reason = None
            for variant in cfg["stage1_candidates"]:
                source_row = by_key.get((case, cfg["arm"], aid, variant))
                if not source_row or not source_row.get("valid"):
                    reason = f"{variant}: invalid Stage1"
                    break
                try:
                    stage1 = Stage1Output.model_validate(json.loads(source_row["raw"]))
                except Exception as exc:
                    reason = f"{variant}: Stage1 parse {type(exc).__name__}"
                    break
                if _external_count(stage1) < cfg["minimum_external_events_per_schedule"]:
                    reason = f"{variant}: no external event"
                    break
                pair.append((variant, stage1))
            if reason:
                selection_log.append({"case": case, "aid": aid, "reason": reason})
                continue

            persona = copy.deepcopy(personas[aid])
            persona["coupon_poi_restricted"] = False
            persona["sangsaeng_active"] = False
            balance = re.search(r"잔액\(내 돈\):\s*([\d,]+)원", cell["user"])
            if not balance:
                raise ValueError(f"balance missing from frozen cell {case}/{aid}")
            state = {"balance": int(balance.group(1).replace(",", ""))}
            day = date.fromisoformat(cell["date"])
            prepared = []
            for variant, stage1 in pair:
                stats = {}
                candidates = s2.fetch_candidates_for_events(aid, stage1.events, persona, day, stats=stats)
                nonempty = sum(bool(v) for v in candidates.values())
                if nonempty < cfg["minimum_nonempty_candidate_pools_per_schedule"]:
                    reason = f"{variant}: no nonempty candidate pool"
                    break
                prepared.append({
                    "key": f"{case}:{aid}:{variant}", "case": case, "aid": aid,
                    "arm": cfg["arm"], "stage1_candidate": variant,
                    "date": cell["date"], "context_sha256": cell["context_sha256"],
                    "stage1": stage1.model_dump(), "persona": persona,
                    "state": state, "candidates": candidates,
                    "n_events": len(stage1.events), "n_external": _external_count(stage1),
                    "n_candidate_pools": nonempty, "candidate_stats": stats,
                })
            if reason:
                selection_log.append({"case": case, "aid": aid, "reason": reason})
                continue
            rows.extend(prepared)
            selected += 1
            if selected == cfg["cells_per_case"]:
                break
        if selected != cfg["cells_per_case"]:
            raise RuntimeError(f"{case}: only {selected} eligible cells")

    payload = {"rows": rows, "selection_log": selection_log}
    _write_json(output / "inputs.json", payload)
    manifest = {
        "config": cfg,
        "source_sha256": _source_hashes(source),
        "inputs_sha256": _sha(output / "inputs.json"),
        "stage2_system_sha256": hashlib.sha256(s2.active_stage2_system().encode("utf-8")).hexdigest(),
        "n_rows": len(rows), "n_cells": len(rows) // len(cfg["stage1_candidates"]),
    }
    _write_json(output / "manifest.json", manifest)
    print(f"FROZEN {len(rows)} rows; inputs_sha256={manifest['inputs_sha256']}")


@contextmanager
def _empty_memory_session():
    class Session:
        def run(self, *_args, **_kwargs):
            return []
    yield Session()


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def run() -> None:
    cfg = _load_config()
    os.environ["SIM_PROMPT_VARIANT"] = "v53"
    if os.getenv("LLM_MODE") != "exaone_4_5":
        raise RuntimeError("Set LLM_MODE=exaone_4_5")
    if not os.getenv("LLM_BASE_URL") and not os.getenv("SGLANG_BASE_URL"):
        raise RuntimeError("Explicitly set the frozen A100 tunnel base URL")
    if os.getenv("SIM_ALLOW_STAGE2_FALLBACK") == "1" or os.getenv("POLICY_BACKTEST_DETERMINISTIC") == "1":
        raise RuntimeError("Fallback and forced temperature are forbidden in this screen")
    source = ROOT / cfg["stage1_source"]
    output = ROOT / cfg["output"]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["config"] != cfg or manifest["source_sha256"] != _source_hashes(source):
        raise RuntimeError("Config/source fingerprint changed after freeze")
    if manifest["inputs_sha256"] != _sha(output / "inputs.json"):
        raise RuntimeError("Frozen inputs changed")
    if manifest["stage2_system_sha256"] != hashlib.sha256(s2.active_stage2_system().encode("utf-8")).hexdigest():
        raise RuntimeError("Stage2 system changed")
    rows = json.loads((output / "inputs.json").read_text(encoding="utf-8"))["rows"]
    if len(rows) != manifest["n_rows"] or len({r["key"] for r in rows}) != len(rows):
        raise RuntimeError("Frozen input keys are incomplete or duplicated")
    raw_path, result_path = output / "responses.jsonl", output / "results.jsonl"
    done = _read_jsonl(result_path)
    done_keys = [r["key"] for r in done]
    if len(done_keys) != len(set(done_keys)):
        raise RuntimeError("Duplicate completed result keys")
    orphan = {r["key"] for r in _read_jsonl(raw_path)} - set(done_keys)
    if orphan:
        raise RuntimeError(f"Incomplete prior requests require manual audit: {sorted(orphan)}")

    from neo4j_load import _common
    original_session = _common.driver_session
    original_fetch = s2.fetch_candidates_for_events
    original_llm = s2._llm_call
    original_review = s2.lookup_reviews_batch
    _common.driver_session = _empty_memory_session
    s2.lookup_reviews_batch = lambda *_a, **_k: {}
    try:
        for row in rows:
            if row["key"] in done_keys:
                continue
            cands = {int(k): copy.deepcopy(v) for k, v in row["candidates"].items()}
            s2.fetch_candidates_for_events = lambda *_a, **_k: copy.deepcopy(cands)
            call_number = 0

            def recorded_call(*args, **kwargs):
                nonlocal call_number
                call_number += 1
                started = time.monotonic()
                record = {
                    "key": row["key"], "attempt": call_number,
                    "system_sha256": hashlib.sha256(args[1].encode("utf-8")).hexdigest(),
                    "user_sha256": hashlib.sha256(args[2].encode("utf-8")).hexdigest(),
                    "temperature": kwargs.get("temperature"), "max_tokens": kwargs.get("max_tokens"),
                    "schema_sha256": hashlib.sha256(json.dumps(kwargs.get("response_format"), sort_keys=True).encode()).hexdigest(),
                }
                try:
                    reply = original_llm(*args, **kwargs)
                    record.update({
                        "raw": reply.choices[0].message.content,
                        "finish_reason": reply.choices[0].finish_reason,
                        "prompt_tokens": getattr(reply.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(reply.usage, "completion_tokens", None),
                    })
                    return reply
                except Exception as exc:
                    record.update({"error_type": type(exc).__name__, "error": str(exc)[:500]})
                    raise
                finally:
                    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
                    _append(raw_path, record)

            s2._llm_call = recorded_call
            result = {"key": row["key"], "case": row["case"],
                      "stage1_candidate": row["stage1_candidate"],
                      "n_events": row["n_events"], "n_external": row["n_external"],
                      "n_candidate_pools": row["n_candidate_pools"]}
            try:
                stage1 = Stage1Output.model_validate(row["stage1"])
                picks, _, meta = s2.call_stage2(
                    row["aid"], stage1, row["persona"], date.fromisoformat(row["date"]),
                    max_retry=cfg["stage2_max_retry"], state=row["state"], active_policies=[],
                )
                result.update({"status": "ok", "picks": [p.model_dump() for p in picks.picks],
                               "attempt": meta.get("attempt"),
                               "hallucinations_corrected": meta.get("hallucinations_corrected", 0),
                               "order_mismatch": meta.get("order_mismatch", 0),
                               "missing_picks_filled": meta.get("missing_picks_filled", 0),
                               "spend_amount_fallbacks": meta.get("spend_amount_fallbacks", 0),
                               "s2_timing": meta.get("s2_timing")})
            except Exception as exc:
                result.update({"status": "failed", "error_type": type(exc).__name__,
                               "error": str(exc)[:500]})
            result["model_calls"] = call_number
            _append(result_path, result)
            done_keys.append(row["key"])
            print(f"{len(done_keys)}/{len(rows)} {row['key']} {result['status']} calls={call_number}", flush=True)
    finally:
        _common.driver_session = original_session
        s2.fetch_candidates_for_events = original_fetch
        s2._llm_call = original_llm
        s2.lookup_reviews_batch = original_review
    score()


def score() -> None:
    cfg = _load_config()
    output = ROOT / cfg["output"]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    inputs = json.loads((output / "inputs.json").read_text(encoding="utf-8"))["rows"]
    results = _read_jsonl(output / "results.jsonl")
    raw = _read_jsonl(output / "responses.jsonl")
    expected = {r["key"] for r in inputs}
    actual = [r["key"] for r in results]
    if len(results) != manifest["n_rows"] or len(set(actual)) != len(actual) or set(actual) != expected:
        raise RuntimeError("Results are incomplete or duplicated")
    if sum(r["model_calls"] for r in results) != len(raw):
        raise RuntimeError("Raw response count does not match model calls")
    by_variant = defaultdict(list)
    for r in results:
        by_variant[r["stage1_candidate"]].append(r)
    def tally(rows):
        return {
            "n": len(rows), "full_failures": sum(r["status"] != "ok" for r in rows),
            "any_partial_correction": sum(r["status"] == "ok" and any(
                r.get(k, 0) for k in ("hallucinations_corrected", "order_mismatch",
                                      "missing_picks_filled", "spend_amount_fallbacks")) for r in rows),
            "n_events": dict(sorted(Counter(r["n_events"] for r in rows).items())),
        }
    summary = {
        "interpretation": "Stage2 technical screen only; not policy effect or prompt ranking",
        "manifest_sha256": _sha(output / "manifest.json"),
        "inputs_sha256": _sha(output / "inputs.json"),
        "responses_sha256": _sha(output / "responses.jsonl"),
        "results_sha256": _sha(output / "results.jsonl"),
        "model_calls": len(raw),
        "overall": tally(results),
        "by_stage1_candidate": {k: tally(v) for k, v in sorted(by_variant.items())},
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run", "score"))
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "score": score}[args.mode]()


if __name__ == "__main__":
    main()
