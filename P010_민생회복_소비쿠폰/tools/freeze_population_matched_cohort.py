"""Fail-closed, CPU-only four-margin cohort calibration.

This creates a proposed roster, never changes Neo4j or starts a simulation.
Income must come from a documented empirical assignment, not spending anchors
or an LLM tier. Marginal raking does not identify the joint distribution.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

FIELDS = ("sex", "age_band", "admin_dong", "income_band")
INCOME_ORIGIN = "verified_empirical_income_assignment"
INCOME_ORIGINS = {INCOME_ORIGIN, "calibrated_synthetic_income_assignment"}


class CalibrationError(ValueError):
    """A population/provenance/support gate failed."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(source: dict, root: Path) -> dict:
    if not isinstance(source, dict) or not source.get("path"):
        raise CalibrationError("source path and SHA256 are mandatory")
    path = (root / source["path"]).resolve()
    expected = source.get("sha256", "")
    if len(expected) != 64 or not path.is_file() or sha256(path) != expected.lower():
        raise CalibrationError(f"source SHA256 mismatch: {source.get('path')}")
    return {"path": source["path"], "sha256": expected.lower()}


def validate_inputs(frame: dict, candidates: dict, root: Path) -> tuple[list[dict], dict, list[dict]]:
    if frame.get("schema") != "population_calibration_frame_v1":
        raise CalibrationError("four-field empirical frame is missing or unsupported")
    if candidates.get("schema") != "population_candidates_v1":
        raise CalibrationError("candidate schema is unsupported")
    unit, year = frame.get("population_unit"), frame.get("reference_year")
    if unit != "resident_person" or not isinstance(year, int) or not 1900 <= year <= 2100:
        raise CalibrationError("frame requires resident_person and an explicit reference year")
    if candidates.get("population_unit") != unit or candidates.get("reference_year") != year:
        raise CalibrationError("candidate and frame year/population unit do not match")
    dong_system = frame.get("targets", {}).get("admin_dong", {}).get("code_system")
    if not dong_system or candidates.get("admin_dong_code_system") != dong_system:
        raise CalibrationError("administrative/legal dong code system and vintage must be explicitly matched")
    dong_audit = candidates.get("admin_dong_assignment_audit", {})
    if (dong_audit.get("uses_actual_residence_anchor") is not True or
            dong_audit.get("official_crosswalk_verified") is not True or
            not dong_audit.get("source_evidence")):
        raise CalibrationError("actual residence-anchor to official administrative dong crosswalk is not verified")
    if candidates.get("income_candidate_origin") not in INCOME_ORIGINS:
        raise CalibrationError("spending/LLM/unknown income is not verified empirical income")
    if not candidates.get("income_assignment_method") or not candidates.get("income_assignment_evidence"):
        raise CalibrationError("empirical income assignment method and source evidence are mandatory")
    evidence = [verify_source(x, root) for x in candidates["income_assignment_evidence"]]
    evidence.extend(verify_source(x, root) for x in dong_audit["source_evidence"])
    if not candidates.get("source_evidence"):
        raise CalibrationError("candidate projection source evidence is mandatory")
    evidence.extend(verify_source(x, root) for x in candidates["source_evidence"])
    targets = frame.get("targets", {})
    if set(targets) != set(FIELDS):
        raise CalibrationError("targets must contain exactly sex, age_band, admin_dong, income_band")
    proportions = {}
    for field in FIELDS:
        target = targets[field]
        if target.get("population_unit") != unit or target.get("reference_year") != year:
            raise CalibrationError(f"incompatible target unit/year: {field}")
        if not target.get("field_definition"):
            raise CalibrationError(f"field definition missing: {field}")
        evidence.append(verify_source(target.get("source"), root))
        source_unit = target.get("source_population_unit")
        source_year = target.get("source_reference_year")
        if not source_unit or not isinstance(source_year, int):
            raise CalibrationError(f"original source population unit/year missing: {field}")
        if source_unit != unit or source_year != year:
            bridge = target.get("population_alignment", {})
            if not bridge.get("method") or not bridge.get("source_evidence"):
                raise CalibrationError(f"source year/unit conversion audit missing: {field}")
            evidence.extend(verify_source(x, root) for x in bridge["source_evidence"])
        values = target.get("proportions")
        if not isinstance(values, dict) or not values:
            raise CalibrationError(f"target proportions missing: {field}")
        if any(not isinstance(k, str) or not k or isinstance(v, bool) or
               not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
               for k, v in values.items()):
            raise CalibrationError(f"invalid proportions: {field}")
        if abs(sum(values.values()) - 1.0) > 1e-8:
            raise CalibrationError(f"proportions do not sum to one: {field}")
        proportions[field] = values
    income_audit = candidates.get("income_assignment_audit", {})
    if (income_audit.get("official_target_sha256") != targets["income_band"]["source"]["sha256"] or
            income_audit.get("policy_outcome_used") is not False or
            income_audit.get("uses_spending_decile_as_income") is not False or
            income_audit.get("distribution_validated") is not True or
            not income_audit.get("income_definition")):
        raise CalibrationError("income donor/imputation/distribution audit is incomplete or uses policy/spending outcomes")
    rows = candidates.get("rows", [])
    if not rows or len({r.get("aid") for r in rows}) != len(rows):
        raise CalibrationError("candidate IDs must be nonempty and unique")
    for row in rows:
        if not isinstance(row.get("aid"), str) or not row["aid"]:
            raise CalibrationError("candidate ID missing")
        for field in FIELDS:
            if row.get(field) not in proportions[field]:
                raise CalibrationError(f"missing or unknown candidate {field}: {row['aid']}")
    for field in FIELDS:
        support = Counter(r[field] for r in rows)
        missing = [k for k, p in proportions[field].items() if p > 0 and not support[k]]
        if missing:
            raise CalibrationError(f"missing positive target cells for {field}: {missing}")
    return sorted(rows, key=lambda r: r["aid"]), proportions, evidence


def distribution(rows: list[dict], weights: list[float], targets: dict) -> dict:
    denominator = sum(weights)
    if denominator <= 0:
        raise CalibrationError("nonpositive weighted denominator")
    result = {}
    for field, categories in targets.items():
        sums = {k: 0.0 for k in categories}
        for row, weight in zip(rows, weights):
            sums[row[field]] += weight
        result[field] = {k: sums[k] / denominator for k in categories}
    return result


def max_error(observed: dict, targets: dict) -> float:
    return max(abs(observed[f][k] - p) for f, cats in targets.items() for k, p in cats.items())


def verify_runtime_input_binding(candidates: dict, root: Path, selected: list[dict]) -> tuple[bool, list, list]:
    binding = candidates.get("runtime_input_binding")
    if not binding:
        return False, ["frozen income profile has not been proved to feed both persona and policy eligibility"], []
    try:
        roster = [r["aid"] for r in selected]
        profile_source = verify_source(binding.get("profile"), root)
        audit_source = verify_source(binding.get("audit"), root)
        proof = json.loads((root / audit_source["path"]).read_text(encoding="utf-8-sig"))
        profile = json.loads((root / profile_source["path"]).read_text(encoding="utf-8-sig"))
        if (proof.get("schema") != "population_runtime_binding_audit_v1" or
                proof.get("profile_sha256") != profile_source["sha256"] or
                proof.get("persona_and_policy_income_match") is not True or
                proof.get("identity_anchor_checks_pass") is not True or
                proof.get("no_aggregate_targets_in_persona") is not True or
                sorted(r.get("aid") for r in profile.get("rows", [])) != roster):
            raise CalibrationError("runtime income/persona/eligibility/roster proof is incomplete")
        code_sources = binding.get("code_source_evidence", [])
        if {Path(x.get("path", "")).name for x in code_sources} != {"population_profile.py", "dawn_context.py", "run_simulation.py"}:
            raise CalibrationError("runtime loader/Dawn/simulation code SHA evidence is incomplete")
        verified = [profile_source, audit_source] + [verify_source(x, root) for x in code_sources]
        production_root = Path(__file__).resolve().parents[1]
        for source in code_sources:
            if (root / source["path"]).resolve() != (production_root / "scripts/sim" / Path(source["path"]).name).resolve():
                raise CalibrationError("binding evidence must identify the actual runtime source files")
        import importlib.util
        spec = importlib.util.spec_from_file_location("_frozen_profile_validator", production_root / "scripts/sim/population_profile.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        valid = module.validate_profile(profile, root / profile_source["path"])
        if valid["reference_year"] != candidates["reference_year"]:
            raise CalibrationError("runtime profile reference year differs from calibrated population")
        for candidate in selected:
            row = valid["rows_by_aid"][candidate["aid"]]
            if (row["income_band"] != candidate["income_band"] or row["sex"] != candidate["sex"] or
                    row["age"] != candidate.get("age") or row["home_dong_code"] != candidate.get("home_dong_code")):
                raise CalibrationError("runtime profile is not the selected calibrated income/identity projection")
        return True, [], verified
    except (CalibrationError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return False, [str(exc)], []


def rake(rows: list[dict], targets: dict, *, tolerance: float = 1e-8,
         iterations: int = 2000) -> tuple[list[float], int]:
    weights = [1.0] * len(rows)
    for iteration in range(1, iterations + 1):
        for field, categories in targets.items():
            total = sum(weights)
            current = {k: 0.0 for k in categories}
            for row, weight in zip(rows, weights):
                current[row[field]] += weight
            for i, row in enumerate(rows):
                target = categories[row[field]]
                denominator = current[row[field]]
                if target > 0 and denominator <= 0:
                    raise CalibrationError(f"unsupported selected target cell: {field}/{row[field]}")
                weights[i] *= total * target / denominator if denominator else 0.0
        total = sum(weights)
        if not math.isfinite(total) or total <= 0:
            raise CalibrationError("nonfinite/zero calibration weights")
        weights = [w * len(rows) / total for w in weights]
        if max_error(distribution(rows, weights, targets), targets) <= tolerance:
            return weights, iteration
    raise CalibrationError("four-margin calibration did not converge; joint support may conflict")


def calibrate(frame: dict, candidates: dict, root: Path, *, sample_size: int,
              seed: int = 20260928, max_unweighted_error: float = 0.02,
              min_ess_fraction: float = 0.5, max_normalized_weight: float = 10.0) -> dict:
    rows, targets, sources = validate_inputs(frame, candidates, root)
    if not isinstance(sample_size, int) or sample_size < 1 or sample_size > len(rows):
        raise CalibrationError("sample_size must be within candidate count")
    if not 0 <= max_unweighted_error <= 1 or not 0 < min_ess_fraction <= 1 or max_normalized_weight < 1:
        raise CalibrationError("invalid predeclared calibration thresholds")
    if any(sum(p > 0 for p in cats.values()) > sample_size for cats in targets.values()):
        raise CalibrationError("sample cannot support all positive categories; increase sample before calls")
    full_weights, full_iterations = rake(rows, targets)
    if sample_size == len(rows):
        selected = rows
    else:
        # Deterministic weighted sampling without replacement. Selected support
        # is checked again; failures never trigger hidden seed retries.
        rng = random.Random(seed)
        keys = [(math.log(max(rng.random(), 1e-300)) / w if w > 0 else -math.inf, r)
                for r, w in zip(rows, full_weights)]
        selected = sorted((r for _, r in sorted(keys, key=lambda x: x[0], reverse=True)[:sample_size]),
                          key=lambda r: r["aid"])
    selected_counts = {f: Counter(r[f] for r in selected) for f in FIELDS}
    for field, cats in targets.items():
        missing = [k for k, p in cats.items() if p > 0 and not selected_counts[field][k]]
        if missing:
            raise CalibrationError(f"selected sample has missing target cells: {field}/{missing}")
    weights, selected_iterations = rake(selected, targets)
    raw = distribution(selected, [1.0] * sample_size, targets)
    weighted = distribution(selected, weights, targets)
    raw_error = max_error(raw, targets)
    ess = sum(weights) ** 2 / sum(w * w for w in weights)
    max_weight = max(weights)
    failures = []
    if raw_error > max_unweighted_error + 1e-12:
        failures.append("unweighted roster exceeds fixed marginal tolerance; weights do not change model population")
    if ess / sample_size < min_ess_fraction:
        failures.append("calibration effective sample size is too small")
    if max_weight > max_normalized_weight:
        failures.append("maximum normalized calibration weight exceeds limit")
    warnings = []
    for field, cats in targets.items():
        sparse = [k for k, p in cats.items() if p > 0 and selected_counts[field][k] < 5]
        if sparse:
            warnings.append({"field": field, "categories_below_five_candidates": sparse})
    roster = [r["aid"] for r in selected]
    runtime_verified, runtime_failures, runtime_sources = verify_runtime_input_binding(candidates, root, selected)
    return {"schema": "population_matched_cohort_v1", "population_gate_pass": not failures,
            "runtime_input_binding_verified": runtime_verified,
            "runtime_input_binding_failures": runtime_failures,
            "model_calls_allowed": not failures and runtime_verified,
            "population_unit": frame["population_unit"], "reference_year": frame["reference_year"],
            "income_candidate_origin": candidates["income_candidate_origin"], "seed": seed, "citizens": sample_size,
            "candidate_count": len(rows), "roster": roster,
            "admin_dong_code_system": candidates["admin_dong_code_system"],
            "weights_by_aid": {r["aid"]: w for r, w in zip(selected, weights)},
            "targets": targets, "unweighted_proportions": raw, "weighted_proportions": weighted,
            "unweighted_max_absolute_error": raw_error, "weighted_max_absolute_error": max_error(weighted, targets),
            "effective_sample_size": ess, "maximum_normalized_weight": max_weight,
            "gates": {"max_unweighted_error": max_unweighted_error,
                      "min_ess_fraction": min_ess_fraction, "max_normalized_weight": max_normalized_weight},
            "raking_iterations": {"candidate_pool": full_iterations, "selected": selected_iterations},
            "failures": failures, "support_warnings": warnings, "source_evidence": sources + runtime_sources,
            "joint_distribution_verified": False,
            "scope_note": "Four empirical marginal distributions only; joint income/demographic distribution is not identified. Weights are evaluator data and cannot be claimed to alter an unweighted simulation roster.",
            "policy_result_used_for_selection": False, "graph_modified": False, "model_calls": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--citizens", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--max-unweighted-error", type=float, default=0.02)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise SystemExit("output exists; proposed frozen roster is immutable")
    try:
        frame = json.loads(args.frame.read_text(encoding="utf-8-sig"))
        candidates = json.loads(args.candidates.read_text(encoding="utf-8-sig"))
        result = calibrate(frame, candidates, args.root, sample_size=args.citizens,
                           seed=args.seed, max_unweighted_error=args.max_unweighted_error)
        result["input_files"] = [{"path": str(p), "sha256": sha256(p)} for p in (args.frame, args.candidates)]
    except (CalibrationError, OSError, ValueError, KeyError, TypeError) as exc:
        result = {"schema": "population_matched_cohort_v1", "model_calls_allowed": False,
                  "population_gate_pass": False, "runtime_input_binding_verified": False,
                  "status": "blocked", "reason": str(exc), "model_calls": 0, "graph_modified": False}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "model_calls_allowed": result["model_calls_allowed"],
                      "reason": result.get("reason", result.get("failures", []))}, ensure_ascii=False))
    if not result["model_calls_allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
