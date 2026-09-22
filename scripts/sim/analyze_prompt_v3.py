"""Analyze the completed, immutable pilot; no LLM calls and no DB access.

Secondary integrity/diagnostic analysis. Does not alter the registered gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics

from validate_prompt_v3 import contract, digest, summarize


def key(row):
    return tuple(row[k] for k in ("variant", "replicate", "aid", "case", "arm"))


def describe(values):
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {"n": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values), "max": max(values),
            "p95_nearest_rank": ordered[max(0, (95*len(values)+99)//100-1)]}


def analyze(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    inputs = json.loads((folder / "frozen_inputs.json").read_text(encoding="utf-8"))
    registered_summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    if digest(inputs) != manifest["inputs_sha256"] or digest(config) != manifest["config_sha256"]:
        raise ValueError("Frozen input or config hash mismatch")
    rows = [json.loads(s) for s in (folder / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = {(v, rep, c["aid"], c["case"], c["arm"])
                for v in config["candidates"] for rep in config["replicate_seeds"]
                for c in inputs["cells"]}
    counts = Counter(map(key, rows))
    if set(counts) != expected or any(n != 1 for n in counts.values()):
        raise ValueError("Incomplete, duplicate or unexpected cells")
    cells = {(c["aid"], c["case"], c["arm"]): c for c in inputs["cells"]}
    for r in rows:
        c = cells[(r["aid"], r["case"], r["arm"])]
        if c["context_sha256"] != digest(c["user"]) or r["context_sha256"] != c["context_sha256"]:
            raise ValueError("Response context hash mismatch")
        if "raw" in r:
            _, errors = contract(r["raw"], c["zones"])
            if errors != r["errors"] or (not errors) != r["valid"]:
                raise ValueError("Stored validity differs from original raw contract")
    recomputed = summarize(rows, config)
    if recomputed != registered_summary:
        raise ValueError("Summary does not reproduce from raw responses")
    out = {
        "integrity": "PASS: all registered cells unique and complete; hashes and summary reproduced",
        "analysis_scope": "registered diagnostics plus secondary integrity and error review; no spending estimate",
        "files_sha256": {name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
                         for name in ["manifest.json", "frozen_inputs.json", "responses.jsonl", "summary.json"]},
        "variants": {}, "fiscal_off_flag_review": [], "invalid_response_review": []}
    paired = {key(r): r for r in rows}
    for variant in config["candidates"]:
        vv = [r for r in rows if r["variant"] == variant]
        syntax = Counter()
        for r in vv:
            try:
                events = json.loads(r.get("raw", ""))["events"]
            except (ValueError, KeyError, TypeError):
                continue
            if any(re.fullmatch(r"\d{8}", str(e.get("anchor", ""))) for e in events):
                syntax["responses_with_bare_8_digit_anchor"] += 1
            if any(not e.get("reasoning") for e in events):
                syntax["responses_with_empty_or_missing_reasoning"] += 1
            if any(not e.get("intent") for e in events):
                syntax["responses_with_empty_or_missing_intent"] += 1
        result = {"registered_gate": recomputed["variants"][variant]["rollout_gate_pass"],
                  "posthoc_syntax_diagnostics": dict(syntax),
                  "finish_reasons": dict(Counter(r.get("finish_reason", "missing") for r in vv)),
                  "latency_seconds": describe([r["elapsed_seconds"] for r in vv]),
                  "completion_tokens": describe([r["usage"]["completion_tokens"] for r in vv if r.get("usage")]),
                  "cases": {}}
        for case in config["cases"]:
            cc = [r for r in vv if r["case"] == case["id"]]
            reps = []
            for rep in config["replicate_seeds"]:
                pp, commerce = [], []
                for aid in sorted({r["aid"] for r in cc}):
                    off = paired[(variant, rep, aid, case["id"], "off")]
                    on = paired[(variant, rep, aid, case["id"], "on")]
                    if off.get("valid") and on.get("valid"):
                        pp.append(on["propensity"] - off["propensity"])
                        commerce.append(on["commerce_events"] - off["commerce_events"])
                reps.append({"replicate": rep, "paired_n": len(pp),
                             "propensity_difference": describe(pp),
                             "commerce_event_difference": describe(commerce)})
            means = [r["propensity_difference"].get("mean") for r in reps]
            sign = [0 if m == 0 else (1 if m > 0 else -1) for m in means if m is not None]
            result["cases"][case["id"]] = {
                "strict_valid": sum(bool(r["valid"]) for r in cc), "responses": len(cc),
                "errors": dict(Counter(e for r in cc for e in r.get("errors", []))),
                "policy_trigger_events_by_arm": {arm: sum(r.get("policy_triggers", 0) for r in cc if r["arm"] == arm)
                                                 for arm in ["off", "on"]},
                "paired_diagnostics": reps,
                "propensity_replicate_sign_agreement": len(sign) == len(means) and len(set(sign)) == 1}
        out["variants"][variant] = result
    for r in rows:
        ident = {k: r[k] for k in ("variant", "replicate", "aid", "case", "arm")}
        if r["arm"] == "off" and r["case"] != "distancing" and r.get("fiscal_policy_attributions"):
            out["fiscal_off_flag_review"].append({**ident, "raw": r.get("raw")})
        if not r["valid"]:
            out["invalid_response_review"].append({**ident, "errors": r["errors"], "raw": r.get("raw"), "error": r.get("error")})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = analyze(args.run)
    # Never overwrite the immutable registered inputs or run outputs.
    with Path(args.output).open("x", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
    print(json.dumps({"integrity": result["integrity"],
                      "gates": {v: r["registered_gate"] for v, r in result["variants"].items()},
                      "flagged": len(result["fiscal_off_flag_review"]),
                      "invalid": len(result["invalid_response_review"])}, ensure_ascii=False))
