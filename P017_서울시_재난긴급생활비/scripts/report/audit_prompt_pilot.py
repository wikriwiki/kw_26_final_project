"""Independently audit a frozen prompt pilot from its raw response artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/sim"))
from validate_prompt_v3 import contract, digest, summarize  # noqa: E402


def audit(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    inputs = json.loads((run_dir / "frozen_inputs.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    if digest(config) != manifest["config_sha256"] or digest(inputs) != manifest["inputs_sha256"]:
        raise ValueError("frozen config/input digest mismatch")
    if hashlib.sha256((ROOT / "scripts/sim/validate_prompt_v3.py").read_bytes()).hexdigest() != manifest["script_sha256"]:
        raise ValueError("runner source differs from frozen manifest")
    if {v: digest(s) for v, s in inputs["systems"].items()} != manifest["system_hashes"]:
        raise ValueError("system prompt digest mismatch")

    cells = {}
    for cell in inputs["cells"]:
        key = (cell["aid"], cell["case"], cell["arm"])
        if key in cells or digest(cell["user"]) != cell["context_sha256"]:
            raise ValueError(f"duplicate or corrupt frozen context: {key}")
        cells[key] = cell
    aids = [persona["id"] for persona in inputs["personas"]]
    if len(aids) != config["sample_n"] or len(set(aids)) != len(aids):
        raise ValueError("frozen persona roster differs from registration")
    case_dates = {case["id"]: case["date"] for case in config["cases"]}
    registered_cells = {(aid, case, arm) for aid in aids
                        for case in case_dates for arm in config["arms"]}
    if set(cells) != registered_cells or any(
        cell["date"] != case_dates[key[1]] for key, cell in cells.items()
    ):
        raise ValueError("frozen contexts differ from registered roster/cases/arms/dates")
    expected = {(v, rep, *key) for v in config["candidates"]
                for rep in config["replicate_seeds"] for key in cells}

    rows = []
    seen = set()
    for line_no, line in enumerate((run_dir / "responses.jsonl").open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["variant"], row["replicate"], row["aid"], row["case"], row["arm"])
        if key not in expected or key in seen:
            raise ValueError(f"unexpected or duplicate response at line {line_no}: {key}")
        seen.add(key)
        cell = cells[key[2:]]
        if row["date"] != cell["date"] or row["context_sha256"] != cell["context_sha256"]:
            raise ValueError(f"response context differs at line {line_no}")
        seed = int(digest([row["replicate"], row["aid"], row["case"]])[:8], 16) % 2147483647
        if row["seed"] != seed:
            raise ValueError(f"response seed differs at line {line_no}")
        if "raw" in row:
            obj, errors = contract(row["raw"], cell["zones"],
                                   date.fromisoformat(cell["date"]).weekday() >= 5)
            if row["errors"] != errors or row["valid"] != (not errors):
                raise ValueError(f"raw response/contract verdict differs at line {line_no}")
            events = (obj or {}).get("events") or []
            calculated = {
                "propensity": (obj or {}).get("daily_propensity"),
                "commerce_events": sum(e.get("category") not in {"집", "직장"} for e in events),
                "fiscal_policy_attributions": sum(
                    e.get("trigger") == "policy" and bool(re.search(
                        r"P01[234]|캐시백|쿠폰|바우처|지원금|상품권", str(e.get("reasoning", ""))))
                    for e in events),
                "policy_triggers": sum(e.get("trigger") == "policy" for e in events),
            }
            if any(row.get(field) != value for field, value in calculated.items()):
                raise ValueError(f"raw response/diagnostics differ at line {line_no}")
        elif row.get("errors") != ["request_or_response"] or row.get("valid") is not False:
            raise ValueError(f"response lacks raw text without a request failure at line {line_no}")
        rows.append(row)
    if seen != expected:
        raise ValueError(f"missing {len(expected - seen)} registered response cells")
    if summarize(rows, config) != summary:
        raise ValueError("saved summary differs from raw responses")

    by_variant = {}
    for row in rows:
        key = (row["aid"], row["case"], row["arm"], row["replicate"])
        by_variant.setdefault(row["variant"], {})[key] = bool(row["valid"])
    from paired_contract_gate import mcnemar  # noqa: E402
    baseline = config["candidates"][0]
    paired = {}
    for variant in config["candidates"][1:]:
        improved, worsened, p = mcnemar(by_variant[baseline], by_variant[variant])
        paired[f"{baseline}:{variant}"] = {"improved": improved, "worsened": worsened,
                                           "p_two_sided": p}
    return {"responses": len(rows), "cells_per_variant": len(expected) // len(config["candidates"]),
            "strict_valid": {v: sum(values.values()) for v, values in by_variant.items()},
            "errors": {v: dict(Counter(e for row in rows if row["variant"] == v
                                       for e in row.get("errors", []))) for v in by_variant},
            "paired": paired, "raw_sha256": hashlib.sha256((run_dir / "responses.jsonl").read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
