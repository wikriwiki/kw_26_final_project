"""Create auditable, actor-disjoint choice datasets from pre-decision captures.

No model weights or optional ML packages are needed. Synthetic records are saved
separately and cannot enter calibration or final evaluation splits.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
DEFAULT_SPLITS = (0.70, 0.15, 0.15)
# These exact Stage2 counters describe constructing the captured candidate set
# before the teacher is called. They are not repairs to its chosen outputs.
_CANDIDATE_RETRIEVAL_COUNTERS = frozenset({
    "cand_fallback_l1_dong", "cand_fallback_l1_district", "resolve_dong_placeholder_fallback",
})
_TEACHER_REPAIR_KEYS = frozenset({
    "hallucinations_dropped", "order_mismatch", "missing_picks_filled", "spend_imputed",
})


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def split_for_group(group: str, seed: str = "fast-decision-v1", ratios=DEFAULT_SPLITS) -> str:
    if len(ratios) != 3 or any(x < 0 for x in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("train/calibration/test ratios must be nonnegative and sum to one")
    if not group:
        raise ValueError("a stable actor group is required")
    number = int(hashlib.sha256(f"{seed}\0{group}".encode()).hexdigest()[:16], 16) / 2**64
    return "train" if number < ratios[0] else "calibration" if number < sum(ratios[:2]) else "test"


def _finite_number(value: Any, low: float, high: float = math.inf) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and low <= value <= high


def _has_repairs(meta: Any, *, _top_level: bool = True) -> bool:
    if isinstance(meta, list):
        return any(_has_repairs(item, _top_level=False) for item in meta)
    if not isinstance(meta, dict):
        return False
    for key, value in meta.items():
        name = str(key).lower()
        if (_top_level and name in _CANDIDATE_RETRIEVAL_COUNTERS
                and isinstance(value, int) and not isinstance(value, bool) and value >= 0):
            continue
        if value and (name in _TEACHER_REPAIR_KEYS or
                      any(marker in name for marker in ("fallback", "correct", "repair", "imput"))):
            return True
        if isinstance(value, (dict, list)) and _has_repairs(value, _top_level=False):
            return True
    return False


def capture_rejection(record: dict, *, allow_other_teachers: bool = False) -> str | None:
    """Reject uncertain labels instead of silently treating fallback values as truth."""
    if record.get("schema_version") != SCHEMA_VERSION:
        return "unsupported_schema"
    snapshot = record.get("snapshot")
    teacher = record.get("teacher")
    provenance = record.get("provenance")
    if not isinstance(snapshot, dict) or not isinstance(teacher, dict) or not isinstance(provenance, dict):
        return "missing_capture_fields"
    if not isinstance(provenance.get("synthetic"), bool):
        return "missing_synthetic_provenance"
    if not snapshot.get("snapshot_id") or not snapshot.get("aid") or not snapshot.get("today"):
        return "missing_snapshot_identity"
    if any(key in snapshot for key in ("teacher", "student", "teacher_output", "post_decision_state", "future_state", "outcome")):
        return "post_decision_input"
    model_id = teacher.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        return "missing_teacher_model"
    if not allow_other_teachers and "exaone" not in model_id.lower():
        return "non_exaone_teacher"
    meta = teacher.get("meta")
    if not isinstance(meta, dict) or meta.get("validated") is not True:
        return "unvalidated_teacher"
    if _has_repairs(meta):
        return "repaired_teacher"
    output = teacher.get("output")
    if not isinstance(output, dict) or not isinstance(output.get("picks"), list):
        return "missing_teacher_output"
    if output.get("review_lookup_requests"):
        return "unresolved_review_request"
    stage1 = snapshot.get("stage1")
    events = stage1.get("events") if isinstance(stage1, dict) else None
    candidates = snapshot.get("candidates")
    if not isinstance(events, list) or not events or not isinstance(candidates, dict):
        return "missing_predecision_events"
    if any(not isinstance(event, dict) for event in events):
        return "invalid_event"
    # Stage1 events use their list position as order. Internal/pinned events do
    # not have candidates and therefore do not require a Stage2 teacher pick.
    for key, options in candidates.items():
        try:
            order = int(key)
        except (TypeError, ValueError):
            return "invalid_event_order"
        if str(order) != str(key) or not 0 <= order < len(events):
            return "invalid_event_order"
        if not isinstance(options, list) or any(not isinstance(option, dict) for option in options):
            return "invalid_candidates"
        ids = [option.get("poi_id") for option in options]
        if any(not isinstance(poi_id, str) or not poi_id for poi_id in ids) or len(ids) != len(set(ids)):
            return "invalid_candidates"
    from .contracts import validate_output
    errors = validate_output(snapshot, output)
    if errors:
        return errors[0]
    for pick in output["picks"]:
        event = events[pick["order"]]
        # Residence/workplace can denote a meal's departure neighborhood; they
        # are not sufficient to identify an internal non-consumer activity.
        if event.get("category") not in {"집", "직장"} and not _finite_number(pick.get("actual_spent"), 0.000001):
            return "invalid_spend"
    return None


def build_dataset(records: Iterable[dict], output_dir: str | Path, *, seed: str = "fast-decision-v1", allow_other_teachers: bool = False) -> dict:
    from .planner import teacher_examples

    destination = Path(output_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError("dataset output must be a new or empty directory")
    destination.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[dict]] = {name: [] for name in ("train", "calibration", "test", "synthetic")}
    rejected: Counter = Counter()
    accepted = 0
    seen: set[str] = set()
    source_hashes: list[str] = []
    teacher_models: set[str] = set()
    for record in records:
        reason = capture_rejection(record, allow_other_teachers=allow_other_teachers)
        if reason:
            rejected[reason] += 1
            continue
        snapshot, teacher = record["snapshot"], record["teacher"]
        snapshot_id = str(snapshot["snapshot_id"])
        if snapshot_id in seen:
            rejected["duplicate_snapshot"] += 1
            continue
        seen.add(snapshot_id)
        try:
            examples = teacher_examples(snapshot, teacher["output"])
        except (KeyError, ValueError, TypeError) as exc:
            rejected[f"unusable_training_example:{type(exc).__name__}"] += 1
            continue
        if not examples:
            rejected["no_training_examples"] += 1
            continue
        synthetic = record["provenance"]["synthetic"]
        group = str(snapshot["aid"])
        split = "synthetic" if synthetic else split_for_group(group, seed)
        source_fingerprint = fingerprint(record)
        pending = []
        for example in examples:
            item = asdict(example) if is_dataclass(example) else dict(example)
            question = item.get("question")
            if not isinstance(question, dict) or item.get("target") not in question.get("options", {}):
                raise ValueError(f"planner emitted an invalid choice target for {snapshot_id}")
            # Use the capture's stable actor, never a date/event-specific planner ID.
            item.update(schema_version=SCHEMA_VERSION, group=group, split=split, snapshot_id=snapshot_id)
            item["provenance"] = {
                **record["provenance"], "teacher_model_id": teacher["model_id"],
                "source_fingerprint": source_fingerprint,
                "calibration_eligible": not synthetic and split == "calibration",
                "decision_date": snapshot["today"],
            }
            pending.append(item)
        rows[split].extend(pending)
        accepted += 1
        source_hashes.append(source_fingerprint)
        teacher_models.add(teacher["model_id"])
    dataset_fingerprint = fingerprint({"schema_version": SCHEMA_VERSION, "seed": seed, "sources": sorted(source_hashes)})
    for split, items in rows.items():
        items.sort(key=lambda item: (item["group"], item["snapshot_id"], item["question"]["key"]))
        with (destination / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for item in items:
                item["dataset_fingerprint"] = dataset_fingerprint
                stream.write(canonical_json(item) + "\n")
    manifest = {
        "schema_version": SCHEMA_VERSION, "dataset_fingerprint": dataset_fingerprint,
        "split_seed": seed, "split_unit": "aid", "accepted_captures": accepted,
        "rejected_captures": dict(sorted(rejected.items())),
        "examples": {key: len(value) for key, value in rows.items()},
        "groups": {key: sorted({item["group"] for item in value}) for key, value in rows.items()},
        "teacher_models": sorted(teacher_models), "other_teachers_allowed": allow_other_teachers,
        "synthetic_calibration_allowed": False, "eligible_for_live": False,
        "limitations": ["Teacher decisions are supervision, not real-world ground truth.",
                       "Actor splits do not replace held-out time/region and multi-day rollout evaluation."],
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default="fast-decision-v1")
    parser.add_argument("--allow-other-teachers", action="store_true", help="Explicit experiment only; preserve model provenance.")
    args = parser.parse_args(argv)
    records = (row for path in args.input for row in read_jsonl(path))
    print(json.dumps(build_dataset(records, args.output, seed=args.seed, allow_other_teachers=args.allow_other_teachers), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
