"""Whole-bundle quality guards and pre-decision-only model inputs."""
import copy
import dataclasses
import json
import math
import random

import pytest

from scripts.sim.fast_decision.__main__ import smoke_record
from scripts.sim.fast_decision.contracts import ChoiceScores, canonical, validate_output
from scripts.sim.fast_decision.planner import preflight, propose, teacher_examples
from scripts.sim.fast_decision.runtime import Capture


@pytest.fixture
def row():
    return smoke_record()


class ScriptedBackend:
    def __init__(self, labels):
        self.labels = iter(labels)
        self.questions = []

    def score(self, questions):
        assert len(questions) == 1
        q = questions[0]
        self.questions.append(q)
        label = next(self.labels)
        assert label in q.options
        labels = list(q.options)
        return [ChoiceScores(q.key, labels, [1.0 if k == label else 0.0 for k in labels],
                             [1.0 if k == label else 0.0 for k in labels])]


@pytest.mark.parametrize("value", [None, False, -1, float("nan"), float("inf")])
def test_missing_or_nonfinite_budget_defers(row, value):
    row["snapshot"]["state"]["balance"] = value
    assert "invalid_balance" in preflight(row["snapshot"])


@pytest.mark.parametrize("field", ["memory", "appointment", "social"])
def test_malformed_context_defers_without_model(row, field):
    row["snapshot"]["context"][field] = None
    assert propose(row["snapshot"], None)["status"] == "deferred"


@pytest.mark.parametrize("field,value", [("actual_spent", 0), ("actual_spent", -1),
    ("actual_spent", float("nan")), ("actual_satisfaction", 1.1), ("actual_satisfaction", None),
    ("actual_satisfaction", True), ("policy_spend", []), ("policy_spend", False)])
def test_invalid_teacher_numbers_are_not_training_truth(row, field, value):
    row["teacher"]["output"]["picks"][0][field] = value
    assert validate_output(row["snapshot"], row["teacher"]["output"])


def test_strict_event_membership_and_whole_day_coverage(row):
    output = row["teacher"]["output"]
    assert validate_output(row["snapshot"], output) == []
    assert "incomplete_day_bundle" in validate_output(row["snapshot"], {"picks": []})
    output["picks"][0]["poi_id"] = "invented"
    assert "poi_not_in_event_candidates" in validate_output(row["snapshot"], output)


def test_policy_wallet_sum_and_eligibility_are_validated(row):
    s = row["snapshot"]
    s["active_policies"] = [{"id": "p1", "type": "grant", "poi_restricted": True}]
    s["grant_remaining"] = {"p1": 2000}
    row["teacher"]["output"]["picks"][0]["policy_spend"] = {"p1": 4000}
    errors = validate_output(s, row["teacher"]["output"])
    assert "ineligible_policy_poi" in errors and "policy_balance_exceeded" in errors
    assert "policy_context" in preflight(s)


def test_proposal_conditions_spend_and_emotion_on_previous_decisions(row):
    backend = ScriptedBackend(["PROCEED", "fixture-b", "12000", "0.40", "distance"])
    original = copy.deepcopy(row["snapshot"])
    result = propose(row["snapshot"], backend)
    assert result["status"] == "proposed" and result["eligible_for_live"] is False
    assert row["snapshot"] == original
    spend_state = json.loads(backend.questions[2].state)
    emotion_state = json.loads(backend.questions[3].state)
    assert spend_state["provisional_picks"][0]["poi_id"] == "fixture-b"
    assert emotion_state["provisional_picks"][0]["actual_spent"] == 12000
    assert result["output"]["picks"][0]["actual_satisfaction"] == .4


def test_student_deferral_discards_entire_bundle(row):
    backend = ScriptedBackend(["PROCEED", "fixture-a", "DEFER"])
    result = propose(row["snapshot"], backend)
    assert result["output"] is None and result["reasons"] == ["model_deferred:0.spend"]


def test_duplicate_visit_defers_before_unneeded_spend_and_emotion_calls(row):
    snapshot = row["snapshot"]
    snapshot["stage1"]["events"].append(copy.deepcopy(snapshot["stage1"]["events"][0]))
    snapshot["candidates"]["1"] = copy.deepcopy(snapshot["candidates"]["0"])
    backend = ScriptedBackend(["PROCEED", "fixture-a", "10000", "0.70", "known", "fixture-a"])
    result = propose(snapshot, backend)
    assert result["status"] == "deferred" and result["output"] is None
    assert result["reasons"] == ["duplicate_poi_in_day"]
    assert len(backend.questions) == 6


def test_training_uses_predecision_fields_and_never_future_labels(row):
    row["snapshot"]["teacher_output"] = "SECRET_FUTURE_LABEL"
    examples = teacher_examples(row["snapshot"], row["teacher"]["output"])
    assert len(examples) == 5
    assert all("SECRET_FUTURE_LABEL" not in e.question.state for e in examples)
    assert "actual_satisfaction" not in json.loads(examples[1].question.state)["provisional_picks"]
    assert "actual_spent" not in json.loads(examples[2].question.state)["provisional_picks"][0]
    assert "actual_satisfaction" not in json.loads(examples[3].question.state)["provisional_picks"][0]


def test_satisfaction_quantization_never_teaches_crossing_low_tail(row):
    row["teacher"]["output"]["picks"][0]["actual_satisfaction"] = .28
    examples = teacher_examples(row["snapshot"], row["teacher"]["output"])
    assert examples[-1].kind == "satisfaction" and examples[-1].target == "DEFER"


def test_sampling_does_not_perturb_global_simulation_rng(row):
    before = random.getstate()
    labels = ["PROCEED", "fixture-a", "10000", "0.70", "known"]
    a = propose(row["snapshot"], ScriptedBackend(labels), selection="sample", seed=42)
    b = propose(row["snapshot"], ScriptedBackend(labels), selection="sample", seed=42)
    assert a == b
    assert before == random.getstate()


def test_capture_records_review_rejection_and_keeps_teacher_unchanged(row, tmp_path):
    output = row["teacher"]["output"]
    previous = copy.deepcopy(output)
    capture = Capture(row["snapshot"], tmp_path / "record.jsonl", "record", "LGAI-EXAONE/example", "exaone")
    meta = capture.finish(output, {"review_lookup_count": 1, "s2_timing": {"t_llm": 1.25}})
    assert meta["applied"] is False and meta["teacher_validated"] is False
    data = json.loads((tmp_path / "record.jsonl").read_text(encoding="utf-8"))
    assert data["teacher"]["latency_seconds"] == 1.25
    assert output == previous
    assert capture.finish(output, {})["status"] == "already_finished"


def test_capture_failure_never_changes_teacher(row, tmp_path):
    capture = Capture(row["snapshot"], tmp_path, "record", "LGAI-EXAONE/example", "exaone")
    assert capture.finish(row["teacher"]["output"], {})["status"] == "capture_error"


def test_nonfinite_snapshot_cannot_be_serialized_as_valid_json(row):
    row["snapshot"]["state"]["mood"] = float("nan")
    with pytest.raises(ValueError):
        canonical(row)
