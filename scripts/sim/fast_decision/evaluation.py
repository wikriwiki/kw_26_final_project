"""Shadow comparison and held-out temperature calibration; never certifies live use.

Teacher agreement is not real-world predictive accuracy. Per-request timings are
reported as measurements and are not extrapolated into a simulation speedup.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .dataset import capture_rejection, fingerprint, read_jsonl


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


@dataclass
class _Comparison:
    """Sufficient statistics; never retain prompts, candidates or individual picks."""

    compared: int = 0
    poi_matches: int = 0
    spend_pairs: int = 0
    spend_error: float = 0.0
    satisfaction_pairs: int = 0
    satisfaction_error: float = 0.0
    satisfaction_drift: float = 0.0
    negative_teacher: int = 0
    negative_student: int = 0
    missed_negative: int = 0
    negative_disagreement: int = 0

    def add(self, pairs: Iterable[tuple[dict, dict]], negative_threshold: float) -> None:
        for teacher, student in pairs:
            self.compared += 1
            self.poi_matches += teacher.get("poi_id") == student.get("poi_id")
            expected, predicted = teacher.get("actual_spent"), student.get("actual_spent")
            if _number(expected) and _number(predicted):
                self.spend_pairs += 1
                self.spend_error += abs(predicted - expected)
            expected, predicted = teacher.get("actual_satisfaction"), student.get("actual_satisfaction")
            if _number(expected) and _number(predicted):
                self.satisfaction_pairs += 1
                self.satisfaction_error += abs(predicted - expected)
                self.satisfaction_drift += predicted - expected
                self.negative_teacher += expected < negative_threshold
                self.negative_student += predicted < negative_threshold
                self.missed_negative += expected < negative_threshold <= predicted
                self.negative_disagreement += (expected < negative_threshold) != (predicted < negative_threshold)

    def summary(self, negative_threshold: float) -> dict:
        n = self.satisfaction_pairs
        return {
            "compared_picks": self.compared,
            "poi_agreement": self.poi_matches / self.compared if self.compared else None,
            "spend_pairs": self.spend_pairs,
            "spend_mae_won": self.spend_error / self.spend_pairs if self.spend_pairs else None,
            "satisfaction_pairs": n, "satisfaction_mae": self.satisfaction_error / n if n else None,
            "satisfaction_mean_student_minus_teacher": self.satisfaction_drift / n if n else None,
            "negative_satisfaction_threshold": negative_threshold,
            "teacher_negative_rate": self.negative_teacher / n if n else None,
            "student_negative_rate": self.negative_student / n if n else None,
            "negative_tail_disagreement_rate": self.negative_disagreement / n if n else None,
            "missed_teacher_negative_rate": self.missed_negative / self.negative_teacher if self.negative_teacher else None,
            "teacher_negative_count": self.negative_teacher,
        }


def _student_rejection(output: Any, snapshot: dict) -> str | None:
    if not isinstance(output, dict) or not isinstance(output.get("picks"), list):
        return "missing_output"
    from .contracts import validate_output
    errors = validate_output(snapshot, output)
    if errors:
        return errors[0]
    for pick in output["picks"]:
        event = snapshot["stage1"]["events"][pick["order"]]
        if event.get("category") not in {"집", "직장"} and pick["actual_spent"] <= 0:
            return "invalid_spend"
    return None


def evaluate_shadow(records: Iterable[dict], *, negative_threshold: float = 0.3) -> dict:
    if not 0 <= negative_threshold <= 1:
        raise ValueError("negative threshold must lie in [0, 1]")
    comparison, synthetic_comparison = _Comparison(), _Comparison()
    groups: dict[str, dict[str, _Comparison]] = {dimension: defaultdict(_Comparison) for dimension in ("income", "region", "policy", "mood")}
    rejected_teachers: Counter = Counter()
    invalid_students: Counter = Counter()
    route_reasons: Counter = Counter()
    timings: dict[str, list[float]] = defaultdict(list)
    input_records = real_records = synthetic_records = compared_records = proposed_records = 0
    seen: set[str] = set()
    for record in records:
        input_records += 1
        reason = capture_rejection(record)
        if reason:
            rejected_teachers[reason] += 1
            continue
        snapshot = record["snapshot"]
        identity = str(snapshot["snapshot_id"])
        if identity in seen:
            rejected_teachers["duplicate_snapshot"] += 1
            continue
        seen.add(identity)
        synthetic = record["provenance"]["synthetic"]
        synthetic_records += int(synthetic)
        real_records += int(not synthetic)
        student = record.get("student") or {}
        if not isinstance(student, dict):
            invalid_students["invalid_student_record"] += 1
            continue
        for item in student.get("reasons", []) or []:
            route_reasons[str(item)] += 1
        # Explicitly separate synthetic timing from measured real requests.
        if not synthetic:
            for source in ("teacher", "student"):
                source_data = record.get(source) or {}
                value = source_data.get("elapsed_ms", source_data.get("latency_ms"))
                if value is None and _number(source_data.get("latency_seconds")):
                    value = source_data["latency_seconds"] * 1000
                if _number(value) and value >= 0:
                    timings[source].append(float(value))
        output = student.get("output")
        if output is None:
            continue  # Defer is coverage loss, not an agreement observation.
        proposed_records += int(not synthetic)
        teacher_output = record["teacher"]["output"]
        reason = _student_rejection(output, snapshot)
        if reason:
            invalid_students[("synthetic:" if synthetic else "") + reason] += 1
            continue
        student_by_order = {pick["order"]: pick for pick in output["picks"]}
        events = dict(enumerate(snapshot["stage1"]["events"]))
        # Internal home/work anchors are trivial copies; excluding them prevents
        # high agreement scores from hiding errors in actual consumer decisions.
        day_pairs = [(pick, student_by_order[pick["order"]]) for pick in teacher_output["picks"]
                     if events[pick["order"]].get("category") not in {"집", "직장"}]
        if synthetic:
            synthetic_comparison.add(day_pairs, negative_threshold)
            continue
        compared_records += 1
        comparison.add(day_pairs, negative_threshold)
        persona, state = snapshot.get("persona", {}), snapshot.get("state", {})
        mood = state.get("mood")
        dimensions = {
            "income": str(persona.get("income_tier", persona.get("income", "unknown"))),
            "region": str(persona.get("home_dong_code", "unknown")),
            "policy": "active" if snapshot.get("active_policies") or snapshot.get("grant_remaining") else "none",
            "mood": "unknown" if not _number(mood) else "low" if mood < 0.35 else "high" if mood > 0.65 else "middle",
        }
        for dimension, value in dimensions.items():
            groups[dimension][value].add(day_pairs, negative_threshold)
    return {
        "schema_version": 1, "status": "shadow_only_not_certified", "eligible_for_live": False,
        "input_records": input_records, "real_records": real_records,
        "synthetic_records": synthetic_records, "compared_real_records": compared_records,
        "student_proposed_real_records": proposed_records,
        "valid_proposal_coverage": compared_records / real_records if real_records else None,
        "rejected_teachers": dict(rejected_teachers), "invalid_students": dict(invalid_students),
        "route_reasons": dict(route_reasons), "teacher_comparison": comparison.summary(negative_threshold),
        "subgroups": {dimension: {name: items.summary(negative_threshold) for name, items in sorted(values.items())}
                      for dimension, values in groups.items()},
        "synthetic_smoke_comparison": synthetic_comparison.summary(negative_threshold),
        "request_timings_ms": {name: {"n": len(values), "p50": _percentile(values, .5), "p95": _percentile(values, .95)}
                               for name, values in timings.items()},
        "system_speedup": None,
        "unvalidated_requirements": [
            "Held-out real-world visits and spending predictive accuracy.",
            "Independently annotated sentiment accuracy, when observations exist.",
            "Multi-day emotion, memory, social and policy-effect trajectories.",
            "Paired end-to-end simulation wall time including routing and fallbacks.",
            "Predeclared non-inferiority limits and uncertainty intervals by subgroup.",
        ],
        "interpretation": "Disagreement with EXAONE is not an error against reality; timings are individual observations, not a speedup claim.",
    }


def _prepare_nll(rows: list[dict]) -> list[tuple[tuple[float, ...], float]]:
    prepared = []
    for row in rows:
        peak = max(row["logits"])
        centered = tuple(value - peak for value in row["logits"])
        prepared.append((centered, centered[row["labels"].index(row["target"])]))
    return prepared


def _prepared_nll(rows: list[tuple[tuple[float, ...], float]], temperature: float) -> float:
    # Center once before scaling: this also avoids cancelling a large shared
    # logit offset when evaluating calibration loss at small temperatures.
    inverse = 1.0 / temperature
    return sum(math.log(sum(math.exp(value * inverse) for value in centered)) - target * inverse
               for centered, target in rows) / len(rows)


def _nll(rows: list[dict], temperature: float) -> float:
    return _prepared_nll(_prepare_nll(rows), temperature)


def score_calibration_examples(rows: list[dict], training_manifest: dict, backend: Any,
                               model_fingerprint: str) -> list[dict]:
    """Score teacher-forced held-out questions, with no answer generation."""
    from .contracts import ChoiceQuestion
    if not rows:
        raise ValueError("calibration dataset is empty")
    if model_fingerprint != training_manifest.get("model_fingerprint"):
        raise ValueError("loaded adapter does not match the training manifest")
    trained_groups = set(training_manifest.get("groups", []))
    if not trained_groups or training_manifest.get("synthetic_examples", 0):
        raise ValueError("a nonsynthetic training manifest with actor groups is required")
    # Validate all rows before any model computation.
    questions = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        provenance = row.get("provenance", {})
        if not isinstance(provenance, dict) or row.get("split") != "calibration" or provenance.get("synthetic") is not False or provenance.get("calibration_eligible") is not True:
            raise ValueError("only real calibration examples may be scored for calibration")
        if not row.get("group") or row["group"] in trained_groups:
            raise ValueError("calibration actor overlaps training or is missing")
        if row.get("dataset_fingerprint") != training_manifest.get("dataset_fingerprint"):
            raise ValueError("calibration dataset differs from the training split manifest")
        question = ChoiceQuestion(**row["question"])
        if row.get("target") not in question.options:
            raise ValueError("calibration target is not an allowed choice")
        identity = (row.get("snapshot_id"), question.key, row.get("kind"))
        if not all(isinstance(value, str) and value for value in identity) or identity in seen:
            raise ValueError("missing or duplicate calibration question identity")
        seen.add(identity)
        questions.append(question)
    batch_size = getattr(backend, "batch_size", 1)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("calibration batch size must be a positive integer")
    output = []
    for begin in range(0, len(rows), batch_size):
        batch_questions = questions[begin:begin + batch_size]
        batch_scores = backend.score(batch_questions)
        if len(batch_scores) != len(batch_questions):
            raise ValueError("backend returned mismatched calibration batch length")
        for row, question, scores in zip(rows[begin:begin + batch_size], batch_questions, batch_scores):
            if scores.key != question.key or scores.labels != list(question.options):
                raise ValueError("backend returned mismatched calibration choices")
            output.append({
                "schema_version": 1, "snapshot_id": row["snapshot_id"], "key": question.key,
                "group": row["group"], "kind": row["kind"], "split": "calibration",
                "provenance": row["provenance"], "dataset_fingerprint": row["dataset_fingerprint"],
                "labels": scores.labels, "logits": scores.logits, "target": row["target"],
                "model_id": training_manifest["model_id"], "revision": training_manifest["revision"],
                "model_fingerprint": model_fingerprint,
                "conditioning": "teacher_forced_previous_choices_not_rollout",
            })
    return output


def calibrate_temperature(rows: list[dict], training_manifest: dict) -> dict:
    """Fit per-question-kind temperature on independent, real calibration groups.

The resulting distribution describes agreement with the teacher labels only.
It must never be interpreted as citizens' real visit probabilities or a live gate.
"""
    if not rows:
        raise ValueError("calibration scores are empty")
    trained_groups = set(training_manifest.get("groups", []))
    dataset_id = training_manifest.get("dataset_fingerprint")
    model_id, revision = training_manifest.get("model_id"), training_manifest.get("revision")
    if not trained_groups or not dataset_id or not model_id or not revision or not training_manifest.get("model_fingerprint"):
        raise ValueError("a complete training manifest with training groups, model revision and dataset fingerprint is required")
    if training_manifest.get("synthetic_examples", 0):
        raise ValueError("synthetic-trained smoke adapters are ineligible for calibration")
    by_kind: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple] = set()
    model_fingerprints: set[str] = set()
    for row in rows:
        provenance = row.get("provenance", {})
        if not isinstance(provenance, dict) or row.get("split") != "calibration" or provenance.get("synthetic") is not False or provenance.get("calibration_eligible") is not True:
            raise ValueError("only real, explicitly eligible calibration examples are accepted")
        if not row.get("group") or row["group"] in trained_groups:
            raise ValueError("calibration actor overlaps training or is missing")
        if row.get("dataset_fingerprint") != dataset_id:
            raise ValueError("calibration and training must share one actor-split dataset manifest")
        if row.get("model_id") != model_id or row.get("revision") != revision or not row.get("model_fingerprint"):
            raise ValueError("calibration requires exact student model, revision and adapter fingerprint")
        if row["model_fingerprint"] != training_manifest["model_fingerprint"]:
            raise ValueError("calibration adapter fingerprint differs from the training manifest")
        labels, logits = row.get("labels"), row.get("logits")
        if not isinstance(labels, list) or not isinstance(logits, list) or len(labels) != len(logits) or len(labels) < 2:
            raise ValueError("calibration needs at least two label logits")
        if any(not isinstance(label, str) for label in labels) or len(set(labels)) != len(labels) or row.get("target") not in labels or any(not _number(value) for value in logits):
            raise ValueError("invalid calibration labels, target or logits")
        identity = (row.get("snapshot_id"), row.get("key"), row.get("kind"))
        if not all(identity) or identity in seen:
            raise ValueError("missing or duplicate calibration question identity")
        seen.add(identity)
        model_fingerprints.add(row["model_fingerprint"])
        by_kind[row["kind"]].append(row)
    if len(model_fingerprints) != 1:
        raise ValueError("calibration scores contain different model/adapter fingerprints")
    fits = {}
    for kind, examples in sorted(by_kind.items()):
        prepared = _prepare_nll(examples)
        # Golden-section search in bounded log-temperature space, no scipy needed.
        lower, upper = math.log(0.05), math.log(20.0)
        ratio = (math.sqrt(5) - 1) / 2
        a, b = upper - ratio * (upper - lower), lower + ratio * (upper - lower)
        fa, fb = _prepared_nll(prepared, math.exp(a)), _prepared_nll(prepared, math.exp(b))
        for _ in range(70):
            if fa < fb:
                upper, b, fb = b, a, fa
                a = upper - ratio * (upper - lower)
                fa = _prepared_nll(prepared, math.exp(a))
            else:
                lower, a, fa = a, b, fb
                b = lower + ratio * (upper - lower)
                fb = _prepared_nll(prepared, math.exp(b))
        temperature = math.exp((lower + upper) / 2)
        before, after = _prepared_nll(prepared, 1.0), _prepared_nll(prepared, temperature)
        if after > before:
            temperature, after = 1.0, before
        fits[kind] = {"temperature": temperature, "examples": len(examples),
                      "nll_before": before, "nll_after": after,
                      "at_search_boundary": temperature < .0501 or temperature > 19.99}
    return {
        "schema_version": 1, "status": "teacher_agreement_temperature_only",
        "eligible_for_live": False, "certified": False, "model_id": model_id,
        "revision": revision, "model_fingerprint": next(iter(model_fingerprints)),
        "dataset_fingerprint": dataset_id, "scores_fingerprint": fingerprint(rows),
        "calibration_groups": sorted({row["group"] for row in rows}), "by_kind": fits,
        "limitations": ["Calibration fit is not a held-out test result.",
                       "No coverage/risk threshold or real-world visit distribution is certified.",
                       "Independent rollout and real-world validation remain required."],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Shadow capture JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-threshold", type=float, default=.3)
    parser.add_argument("--calibration-scores", type=Path)
    parser.add_argument("--score-examples", type=Path, help="Explicit model inference on held-out calibration.jsonl")
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1, help="Questions per calibration inference batch")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a new experiment path")
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    if args.score_examples:
        if not args.training_manifest or not args.adapter or args.input or args.calibration_scores:
            parser.error("--score-examples requires --training-manifest and --adapter and is a separate operation")
        from .backend import ExaoneChoiceBackend
        from .training import adapter_fingerprint
        manifest = json.loads(args.training_manifest.read_text(encoding="utf-8"))
        adapter_id = adapter_fingerprint(args.adapter)
        if adapter_id != manifest.get("model_fingerprint"):
            parser.error("adapter fingerprint does not match training manifest")
        backend = ExaoneChoiceBackend(model_id=manifest["model_id"], revision=manifest["revision"],
                                     device=args.device, adapter_path=str(args.adapter),
                                     max_tokens=args.max_tokens, batch_size=args.batch_size,
                                     allow_download=args.allow_download)
        scores = score_calibration_examples(list(read_jsonl(args.score_examples)), manifest, backend, adapter_id)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as stream:
            for row in scores:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps({"scored_questions": len(scores), "eligible_for_live": False}))
        return 0
    if args.calibration_scores:
        if not args.training_manifest or args.input:
            parser.error("--calibration-scores requires --training-manifest and cannot use --input")
        manifest = json.loads(args.training_manifest.read_text(encoding="utf-8"))
        result = calibrate_temperature(list(read_jsonl(args.calibration_scores)), manifest)
    else:
        if not args.input:
            parser.error("--input is required for shadow evaluation")
        result = evaluate_shadow(read_jsonl(args.input), negative_threshold=args.negative_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
