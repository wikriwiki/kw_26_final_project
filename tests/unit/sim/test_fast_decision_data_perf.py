"""Streaming/batch regressions that need neither model weights nor a GPU."""
from __future__ import annotations

import copy
import math
import weakref

import pytest

from scripts.sim.fast_decision.__main__ import smoke_record
from scripts.sim.fast_decision.contracts import ChoiceScores
from scripts.sim.fast_decision.dataset import build_dataset, capture_rejection, read_jsonl
from scripts.sim.fast_decision.evaluation import (
    _nll, _prepare_nll, _prepared_nll, evaluate_shadow, score_calibration_examples,
)
from scripts.sim.fast_decision.training import validate_examples


def example(index=0, *, calibration=False):
    return {
        "schema_version": 1, "group": f"person-{index}",
        "split": "calibration" if calibration else "train",
        "snapshot_id": f"snapshot-{index}", "kind": "route", "target": "PROCEED",
        "dataset_fingerprint": "dataset-v1",
        "question": {"key": f"route-{index}", "state": "기분 0.6", "instructions": "진행할까요?",
                     "options": {"PROCEED": "진행", "DEFER": "보류"}},
        "provenance": {"synthetic": False, "teacher_model_id": "LGAI-EXAONE/test",
                       "source_fingerprint": f"source-{index}", "calibration_eligible": calibration},
    }


def manifest():
    return {"groups": ["training-person"], "dataset_fingerprint": "dataset-v1",
            "model_id": "LGAI-EXAONE/test", "revision": "fixture-revision",
            "model_fingerprint": "fixture-adapter", "synthetic_examples": 0}


class RecordingBackend:
    batch_size = 3

    def __init__(self):
        self.calls = []

    def score(self, questions):
        self.calls.append([question.key for question in questions])
        return [ChoiceScores(question.key, list(question.options), [1., 0.], [.75, .25])
                for question in questions]


def test_calibration_uses_bounded_batches_and_preserves_question_order():
    backend = RecordingBackend()
    rows = [example(i, calibration=True) for i in range(8)]
    scores = score_calibration_examples(rows, manifest(), backend, "fixture-adapter")
    assert list(map(len, backend.calls)) == [3, 3, 2]
    assert [row["snapshot_id"] for row in scores] == [row["snapshot_id"] for row in rows]
    assert all(row["conditioning"] == "teacher_forced_previous_choices_not_rollout" for row in scores)


def test_duplicate_calibration_question_rejected_before_any_inference():
    row = example(calibration=True)
    backend = RecordingBackend()
    with pytest.raises(ValueError, match="duplicate calibration"):
        score_calibration_examples([row, copy.deepcopy(row)], manifest(), backend, "fixture-adapter")
    assert backend.calls == []


def test_later_invalid_calibration_row_rejected_before_first_batch():
    backend = RecordingBackend()
    rows = [example(i, calibration=True) for i in range(8)]
    rows[-1]["split"] = "test"
    with pytest.raises(ValueError, match="real calibration"):
        score_calibration_examples(rows, manifest(), backend, "fixture-adapter")
    assert backend.calls == []


def test_incomplete_calibration_batch_cannot_silently_drop_rows():
    class IncompleteBackend(RecordingBackend):
        def score(self, questions):
            return super().score(questions)[:-1]

    with pytest.raises(ValueError, match="batch length"):
        score_calibration_examples([example(calibration=True)], manifest(), IncompleteBackend(), "fixture-adapter")


def test_training_validation_streams_and_rejects_repeated_supervision():
    summary = validate_examples(example(i) for i in range(7))
    assert summary["examples"] == 7
    row = example()
    with pytest.raises(ValueError, match="duplicate training"):
        validate_examples(iter([row, copy.deepcopy(row)]))
    with pytest.raises(ValueError, match="empty"):
        validate_examples(iter([]))


def test_synthetic_override_does_not_allow_mislabeled_real_split():
    row = example()
    row["split"] = "synthetic"
    with pytest.raises(ValueError, match="real data"):
        validate_examples([row], allow_synthetic=True)


def test_nested_repaired_teacher_is_excluded_from_training_and_evaluation():
    row = smoke_record()
    row["teacher"]["meta"]["events"] = [{"attempts": [{"fallback_used": True}]}]
    assert capture_rejection(row) == "repaired_teacher"
    report = evaluate_shadow(iter([row]))
    assert report["rejected_teachers"] == {"repaired_teacher": 1}
    assert report["teacher_comparison"]["compared_picks"] == 0


def test_candidate_retrieval_expansion_keeps_valid_teacher_labels(tmp_path):
    row = smoke_record()
    row["teacher"]["meta"].update(cand_fallback_l1_dong=1, cand_fallback_l1_district=2,
                                  resolve_dong_placeholder_fallback=1)
    # The teacher's exact choice is still required to belong to the captured
    # candidate set after expansion; this does not waive output validation.
    assert capture_rejection(row) is None
    row["student"] = {"output": copy.deepcopy(row["teacher"]["output"])}
    report = evaluate_shadow([row])
    assert report["synthetic_smoke_comparison"]["poi_agreement"] == 1
    result = build_dataset([row], tmp_path)
    assert result["accepted_captures"] == 1
    assert len(list(read_jsonl(tmp_path / "synthetic.jsonl"))) > 0
    row["teacher"]["output"]["picks"][0]["poi_id"] = "not-in-current-candidates"
    assert capture_rejection(row) == "poi_not_in_event_candidates"


@pytest.mark.parametrize("metadata", [
    {"fallback_only": True}, {"hallucinations_corrected": 1},
    {"hallucinations_dropped": 1}, {"order_mismatch": 1},
    {"missing_picks_filled": 1}, {"spend_imputed": 1},
    {"future_unknown_repair_flag": True}, {"new_fallback_strategy": "applied"},
    {"cand_fallback_l1_dong": {"repair_count": 1}},
    {"events": [{"cand_fallback_l1_dong": 1}]},
])
def test_candidate_expansion_does_not_exempt_output_repairs(metadata):
    row = smoke_record()
    row["teacher"]["meta"].update(cand_fallback_l1_dong=1)
    row["teacher"]["meta"].update(metadata)
    assert capture_rejection(row) == "repaired_teacher"


@pytest.mark.parametrize("temperature", [.05, 1., 20.])
def test_prepared_nll_preserves_loss_and_large_offset_invariance(temperature):
    row = {"labels": ["A", "B"], "logits": [1., -1.], "target": "B"}
    expected = math.log1p(math.exp(-2 / temperature)) + 2 / temperature
    assert _prepared_nll(_prepare_nll([row]), temperature) == pytest.approx(expected)
    offset = {**row, "logits": [1e12 + 1., 1e12 - 1.]}
    assert _nll([offset], temperature) == pytest.approx(expected, abs=1e-12)


def test_shadow_stream_releases_old_picks_and_preserves_emotion_metrics():
    class TrackedPick(dict):
        pass

    retained = []

    def records():
        for index in range(100):
            # At most the previous capture can still be held by evaluation
            # locals. Older captures must not accumulate in subgroup lists.
            assert sum(reference() is not None for reference in retained) <= 2
            row = smoke_record()
            row["provenance"]["synthetic"] = False
            row["snapshot"]["snapshot_id"] = f"snapshot-{index}"
            teacher = TrackedPick(row["teacher"]["output"]["picks"][0])
            teacher["actual_satisfaction"] = .2
            student = TrackedPick({**teacher, "actual_satisfaction": .4, "actual_spent": 12000,
                                   "poi_id": "fixture-b"})
            row["teacher"]["output"]["picks"] = [teacher]
            row["student"] = {"output": {"picks": [student]}}
            retained.extend([weakref.ref(teacher), weakref.ref(student)])
            yield row

    report = evaluate_shadow(records())
    metrics = report["teacher_comparison"]
    assert report["input_records"] == 100
    assert report["valid_proposal_coverage"] == 1
    assert metrics["compared_picks"] == 100
    assert metrics["poi_agreement"] == 0
    assert metrics["spend_mae_won"] == 2000
    assert metrics["satisfaction_mae"] == pytest.approx(.2)
    assert metrics["missed_teacher_negative_rate"] == 1
    assert metrics["student_negative_rate"] == 0
    assert report["subgroups"]["income"]["중"] == metrics
    assert report["eligible_for_live"] is False
    assert all(reference() is None for reference in retained)
