"""Audit whether Stage2 choices came from valid model output, not silent fallback.

Accept completed metrics directories or read-only checkpoint archives. This is
an execution-quality gate shared by every policy and prompt candidate.
"""
from __future__ import annotations

import argparse
import json
import re
import tarfile
from collections import Counter
from pathlib import Path


DAY = re.compile(r"day_(\d{4}-\d{2}-\d{2})\.jsonl$")


def load_sources(metrics_dir: Path | None, archives: list[Path]) -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = {}

    def add(name: str, lines) -> None:
        match = DAY.search(name)
        if not match:
            return
        day = match.group(1)
        if day in found:
            raise ValueError(f"duplicate metrics day across sources: {day}")
        found[day] = [json.loads(line) for line in lines if line.strip()]

    if metrics_dir is not None:
        for path in sorted(metrics_dir.glob("day_*.jsonl")):
            with path.open("rb") as stream:
                add(path.name, stream)
    for archive in archives:
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.startswith("metrics/"):
                    with tar.extractfile(member) as stream:
                        add(member.name, stream)
    if not found:
        raise ValueError("no daily metrics found")
    return found


def inspect(day_rows: dict[str, list[dict]], *, expected_per_day: int | None = None,
            legacy_token_cap: int = 1400) -> dict:
    per_day = []
    reference_ids: set[str] | None = None
    for day, rows in sorted(day_rows.items()):
        ids = [r.get("aid") for r in rows]
        if any(not isinstance(aid, str) or not aid for aid in ids):
            raise ValueError(f"missing citizen ID on {day}")
        duplicate = len(ids) - len(set(ids))
        day_ids = set(ids)
        roster_matches = reference_ids is None or day_ids == reference_ids
        if reference_ids is None:
            reference_ids = day_ids
        ok = [r for r in rows if r.get("status") == "ok"]
        attempts = [a for r in ok for a in (r.get("s2_timing") or {}).get("attempts", [])]
        errors = Counter(a.get("error_stage") for a in attempts if a.get("status") == "error")
        limited = sum(bool(a.get("output_limited")) or a.get("finish_reason") == "length"
                      or (a.get("error_stage") in ("json_parse", "json_extract")
                          and a.get("tokens_out") == legacy_token_cap)
                      for a in attempts)
        fallback_rows = [r for r in ok if bool(r.get("s2_fallback_only")) or
                         (bool((r.get("s2_timing") or {}).get("attempts"))
                          and not any(a.get("status") == "ok"
                                      for a in r["s2_timing"]["attempts"]))]
        fallback = len(fallback_rows)
        partial_repair_rows = [r for r in ok if
                               int(r.get("fb_missing_picks_filled") or 0) > 0 or
                               int(r.get("fb_hallucinations_corrected") or 0) > 0]
        repaired_ids = {id(r) for r in fallback_rows + partial_repair_rows}

        def had_limited_output(row: dict) -> bool:
            return any(bool(a.get("output_limited")) or a.get("finish_reason") == "length"
                       or (a.get("error_stage") in ("json_parse", "json_extract")
                           and a.get("tokens_out") == legacy_token_cap)
                       for a in (row.get("s2_timing") or {}).get("attempts", []))

        def had_review_error(row: dict) -> bool:
            return any(a.get("error_stage") == "review_lookup"
                       and a.get("status") == "error"
                       for a in (row.get("s2_timing") or {}).get("attempts", []))
        missing_decision_evidence = sum(
            not r.get("s2_skipped") and
            not (r.get("s2_timing") or {}).get("attempts")
            for r in ok)
        calls = sum(int((r.get("s2_timing") or {}).get("n_llm_calls") or 0) for r in ok)
        generated = sum(int(a.get("tokens_out") or 0) for a in attempts)
        per_day.append({
            "day": day, "metrics_rows": len(rows), "agents_ok": len(ok),
            "agents_error": len(rows) - len(ok), "duplicate_aids": duplicate,
            "stage2_llm_calls": calls, "stage2_extra_calls":
                sum(max(0, int((r.get("s2_timing") or {}).get("n_llm_calls") or 0) - 1)
                    for r in ok),
            "stage2_review_lookup_errors": errors["review_lookup"],
            "stage2_json_parse_errors": errors["json_parse"],
            "stage2_output_limited_attempts": limited,
            "stage2_fallback_only_agents": fallback,
            "stage2_partial_repair_agents": len(partial_repair_rows),
            "stage2_choice_repair_agents": len(repaired_ids),
            "stage2_fallback_with_output_limit_agents": sum(
                had_limited_output(r) for r in fallback_rows),
            "stage2_fallback_with_review_error_agents": sum(
                had_review_error(r) for r in fallback_rows),
            "stage2_missing_decision_evidence_agents": missing_decision_evidence,
            "stage2_missing_picks_filled": sum(int(r.get("fb_missing_picks_filled") or 0)
                                               for r in ok),
            "stage2_hallucinations_corrected": sum(
                int(r.get("fb_hallucinations_corrected") or 0) for r in ok),
            "stage2_spend_amount_fallbacks": sum(
                int(r.get("fb_spend_amount_fallbacks") or 0)
                for r in ok if "fb_spend_amount_fallbacks" in r),
            "stage2_spend_amount_observed_agents": sum(
                "fb_spend_amount_fallbacks" in r for r in ok),
            "stage2_generated_tokens_all_attempts": generated,
            "expected_count_met": expected_per_day is None or len(rows) == expected_per_day,
            "roster_matches_first_day": roster_matches,
        })
    totals = {key: sum(d[key] for d in per_day) for key in (
        "metrics_rows", "agents_ok", "agents_error", "duplicate_aids",
        "stage2_llm_calls", "stage2_extra_calls", "stage2_review_lookup_errors",
        "stage2_json_parse_errors", "stage2_output_limited_attempts",
        "stage2_fallback_only_agents", "stage2_partial_repair_agents",
        "stage2_choice_repair_agents", "stage2_fallback_with_output_limit_agents",
        "stage2_fallback_with_review_error_agents",
        "stage2_missing_decision_evidence_agents",
        "stage2_missing_picks_filled", "stage2_hallucinations_corrected",
        "stage2_spend_amount_fallbacks", "stage2_spend_amount_observed_agents",
        "stage2_generated_tokens_all_attempts")}
    if totals["stage2_spend_amount_observed_agents"] == 0:
        totals["stage2_spend_amount_fallbacks"] = None
    for daily in per_day:
        if daily["stage2_spend_amount_observed_agents"] == 0:
            daily["stage2_spend_amount_fallbacks"] = None
    quality_pass = (all(d["expected_count_met"] and d["roster_matches_first_day"]
                        for d in per_day)
                    and totals["agents_error"] == 0 and totals["duplicate_aids"] == 0
                    and totals["stage2_fallback_only_agents"] == 0
                    and totals["stage2_missing_decision_evidence_agents"] == 0)
    return {"days": len(per_day), "expected_per_day": expected_per_day,
            "quality_gate_pass": quality_pass,
            "unrepaired_choice_trace_pass": quality_pass and
                totals["stage2_choice_repair_agents"] == 0,
            "totals": totals, "per_day": per_day,
            "interpretation": "Generation completeness only. A clean run does not prove "
                              "empirical effect magnitude or prompt generalization."}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path)
    parser.add_argument("--archive", action="append", type=Path, default=[])
    parser.add_argument("--expected-per-day", type=int)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--require-unrepaired-choices", action="store_true",
                        help="Fail if any Stage2 POI choice was wholly or partly repaired")
    args = parser.parse_args()
    result = inspect(load_sources(args.metrics_dir, args.archive),
                     expected_per_day=args.expected_per_day)
    print(json.dumps({"quality_gate_pass": result["quality_gate_pass"],
                      "unrepaired_choice_trace_pass": result["unrepaired_choice_trace_pass"],
                      "days": result["days"], "totals": result["totals"]},
                     ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    return 1 if ((args.strict and not result["quality_gate_pass"])
                 or (args.require_unrepaired_choices and
                     not result["unrepaired_choice_trace_pass"])) else 0


if __name__ == "__main__":
    raise SystemExit(main())
