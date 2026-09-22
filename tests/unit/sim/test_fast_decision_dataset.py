"""Offline provenance, split, calibration and CPU training integrity tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.sim.fast_decision.__main__ import smoke_record
from scripts.sim.fast_decision.dataset import build_dataset, capture_rejection, fingerprint, read_jsonl, split_for_group
from scripts.sim.fast_decision.evaluation import calibrate_temperature, evaluate_shadow, score_calibration_examples
from scripts.sim.fast_decision.training import validate_examples


def capture(aid="person-1", synthetic=False):
    row = smoke_record()
    row["provenance"]["synthetic"] = synthetic
    row["provenance"]["source"] = "unit_test_fixture_not_observational_data"
    row["snapshot"]["aid"] = aid
    row["snapshot"]["snapshot_id"] = fingerprint([aid, synthetic])
    return row


def training_row():
    return {
        "schema_version": 1, "group": "train-person", "split": "train", "snapshot_id": "training-snapshot",
        "kind": "route", "target": "PROCEED", "dataset_fingerprint": "dataset-v1",
        "question": {"key": "route", "state": "기분 0.6, 피로 0.3", "instructions": "진행할까요?",
                     "options": {"PROCEED": "진행", "DEFER": "보류"}},
        "provenance": {"synthetic": False, "teacher_model_id": "LGAI-EXAONE/EXAONE-4.0-32B",
                       "source_fingerprint": "fixture-source", "calibration_eligible": False},
    }


def calibration_inputs():
    manifest = {"groups": ["train-person"], "dataset_fingerprint": "dataset-v1",
                "model_id": "LGAI-EXAONE/EXAONE-4.0-1.2B", "revision": "pinned-test-revision",
                "model_fingerprint": "test-adapter", "synthetic_examples": 0}
    rows = [{"snapshot_id": f"cal-{index}", "key": "route", "kind": "route", "split": "calibration",
             "group": f"cal-person-{index}", "dataset_fingerprint": "dataset-v1",
             "model_id": manifest["model_id"], "revision": manifest["revision"], "model_fingerprint": "test-adapter",
             "labels": ["PROCEED", "DEFER"], "logits": [5.0, -5.0],
             "target": "PROCEED" if index % 2 else "DEFER",
             "provenance": {"synthetic": False, "calibration_eligible": True}}
            for index in range(12)]
    return manifest, rows


def test_capture_accepts_actual_stage1_index_schema_and_omitted_internal_picks():
    row = capture()
    row["snapshot"]["stage1"]["events"].insert(0, {"category": "집", "anchor": "residence"})
    row["snapshot"]["stage1"]["events"].append({"category": "직장", "anchor": "workplace"})
    row["snapshot"]["stage1"]["events"][1]["anchor"] = "workplace"  # Meal in workplace neighborhood.
    row["snapshot"]["candidates"]["1"] = row["snapshot"]["candidates"].pop("0")
    row["teacher"]["output"]["picks"][0]["order"] = 1
    assert capture_rejection(row) is None
    row["teacher"]["output"]["picks"][0]["actual_spent"] = 0
    assert capture_rejection(row) == "invalid_spend"


@pytest.mark.parametrize("mutation, expected", [
    (lambda r: r["teacher"].update(model_id="Qwen/test"), "non_exaone_teacher"),
    (lambda r: r["teacher"]["meta"].update(validated=False), "unvalidated_teacher"),
    (lambda r: r["teacher"]["meta"].update(hallucinations_corrected=1), "repaired_teacher"),
    (lambda r: r["teacher"]["meta"].update(fallback_only=True), "repaired_teacher"),
    (lambda r: r["teacher"]["output"]["picks"][0].update(actual_satisfaction=float("nan")), "invalid_satisfaction"),
    (lambda r: r["teacher"]["output"]["picks"].clear(), "incomplete_day_bundle"),
    (lambda r: r["snapshot"].update(teacher_output={"picks": []}), "post_decision_input"),
    (lambda r: r["provenance"].pop("synthetic"), "missing_synthetic_provenance"),
])
def test_rejects_unusable_supervision(mutation, expected):
    row = capture()
    mutation(row)
    assert capture_rejection(row) == expected


def test_dataset_splits_actors_and_keeps_synthetic_out_of_real_splits(tmp_path):
    rows = [capture(f"person-{index}") for index in range(60)] + [smoke_record()]
    tomorrow = copy.deepcopy(rows[0])
    tomorrow["snapshot"]["today"] = "2026-01-06"
    tomorrow["snapshot"]["snapshot_id"] = "tomorrow"
    rows.append(tomorrow)
    manifest = build_dataset(rows, tmp_path)
    groups = {split: set(manifest["groups"][split]) for split in ("train", "calibration", "test")}
    assert all(groups.values())
    assert not groups["train"] & groups["calibration"]
    assert not groups["train"] & groups["test"]
    assert not groups["calibration"] & groups["test"]
    assert manifest["examples"]["synthetic"] == 5
    assert manifest["eligible_for_live"] is False
    for split in groups:
        examples = list(read_jsonl(tmp_path / f"{split}.jsonl"))
        assert all(row["provenance"]["synthetic"] is False for row in examples)
        assert all(split_for_group(row["group"]) == split for row in examples)
        assert all(row["provenance"]["calibration_eligible"] == (split == "calibration") for row in examples)
    person_split = split_for_group("person-0")
    assert len([row for row in read_jsonl(tmp_path / f"{person_split}.jsonl") if row["group"] == "person-0"]) == 10


def test_teacher_target_does_not_leak_into_same_question(tmp_path):
    build_dataset([capture()], tmp_path)
    rows = list(read_jsonl(tmp_path / f"{split_for_group('person-1')}.jsonl"))
    by_kind = {row["kind"]: row for row in rows}
    assert json.loads(by_kind["route"]["question"]["state"])["provisional_picks"] == []
    assert json.loads(by_kind["poi"]["question"]["state"])["provisional_picks"] == []
    spend_prefix = json.loads(by_kind["spend"]["question"]["state"])["provisional_picks"][-1]
    assert "actual_spent" not in spend_prefix
    sat_prefix = json.loads(by_kind["satisfaction"]["question"]["state"])["provisional_picks"][-1]
    assert "actual_satisfaction" not in sat_prefix


def test_dry_run_imports_no_ml_packages_and_uses_no_weights(tmp_path):
    data = tmp_path / "dataset"
    build_dataset([smoke_record()], data)
    destination = tmp_path / "dry-run"
    command = (
        "import sys; from scripts.sim.fast_decision.training import main; "
        "main(['--input',sys.argv[1],'--output',sys.argv[2],'--dry-run','--allow-synthetic']); "
        "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; assert 'peft' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", command, str(data / "synthetic.jsonl"), str(destination)],
                            capture_output=True, text=True, cwd=Path(__file__).resolve().parents[3])
    assert result.returncode == 0, result.stderr
    report = json.loads((destination / "dry_run.json").read_text(encoding="utf-8"))
    assert report["weights_loaded"] is False and report["trained"] is False
    assert report["tokenization_validated"] is False


@pytest.mark.parametrize("split", ["calibration", "test"])
def test_training_refuses_held_out_data(split):
    row = training_row()
    row["split"] = split
    with pytest.raises(ValueError, match="calibration/test"):
        validate_examples([row])


def test_shadow_metrics_count_negative_tail_and_report_coverage_without_speedup():
    row = capture()
    row["teacher"]["output"]["picks"][0]["actual_satisfaction"] = .2
    student_output = copy.deepcopy(row["teacher"]["output"])
    student_output["picks"][0].update(poi_id="fixture-b", actual_spent=12000, actual_satisfaction=.3)
    row["student"] = {"output": student_output, "latency_seconds": .01}
    row["teacher"]["latency_seconds"] = 1.0
    deferred = capture("person-2")
    deferred["student"] = {"output": None, "reasons": ["low_mood"]}
    report = evaluate_shadow([row, deferred, smoke_record()])
    assert report["teacher_comparison"]["poi_agreement"] == 0
    assert report["teacher_comparison"]["spend_mae_won"] == 2000
    assert report["teacher_comparison"]["missed_teacher_negative_rate"] == 1
    assert report["teacher_comparison"]["student_negative_rate"] == 0  # Strict < .3.
    assert report["valid_proposal_coverage"] == .5
    assert report["request_timings_ms"]["student"]["p50"] == 10
    assert report["system_speedup"] is None and report["eligible_for_live"] is False


def test_bad_student_is_counted_not_silently_scored_as_good():
    row = capture()
    row["student"] = {"output": copy.deepcopy(row["teacher"]["output"])}
    row["student"]["output"]["picks"][0]["actual_satisfaction"] = 2
    report = evaluate_shadow([row])
    assert report["invalid_students"] == {"invalid_satisfaction": 1}
    assert report["valid_proposal_coverage"] == 0
    assert report["teacher_comparison"]["poi_agreement"] is None


def test_calibration_improves_nll_but_never_certifies_quality():
    manifest, rows = calibration_inputs()
    result = calibrate_temperature(rows, manifest)
    assert result["by_kind"]["route"]["nll_after"] < result["by_kind"]["route"]["nll_before"]
    assert result["eligible_for_live"] is False and result["certified"] is False


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(group="train-person"),
    lambda r: r.update(split="test"),
    lambda r: r["provenance"].update(synthetic=True),
    lambda r: r.update(dataset_fingerprint="different-dataset"),
    lambda r: r.update(model_fingerprint="different-adapter"),
    lambda r: r.update(logits=[float("nan"), 1.0]),
])
def test_calibration_rejects_leakage_and_wrong_provenance(mutation):
    manifest, rows = calibration_inputs()
    mutation(rows[0])
    with pytest.raises(ValueError):
        calibrate_temperature(rows, manifest)


def test_score_export_checks_provenance_before_using_backend():
    manifest, _ = calibration_inputs()
    row = training_row()
    class ForbiddenBackend:
        def score(self, questions):
            pytest.fail("invalid calibration rows must fail before model scoring")
    with pytest.raises(ValueError, match="calibration examples"):
        score_calibration_examples([row], manifest, ForbiddenBackend(), "test-adapter")


def test_tiny_cpu_lora_training_saves_and_reloads_same_scores(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    from transformers import Exaone4Config, Exaone4ForCausalLM
    from scripts.sim.fast_decision.backend import ExaoneChoiceBackend
    from scripts.sim.fast_decision.contracts import ChoiceQuestion
    from scripts.sim.fast_decision.training import train, adapter_fingerprint

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1
        all_special_ids = [0, 1]
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"] + "\n"
        def encode(self, text, **kwargs):
            return [ord(char) + 3 if ord(char) < 128 else 160 + ord(char) % 96 for char in text]
        def save_pretrained(self, destination):
            Path(destination, "unit_test_tokenizer.json").write_text('{"test_only":true}', encoding="utf-8")

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            config = Exaone4Config(vocab_size=320, hidden_size=16, intermediate_size=32,
                                  num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                  max_position_embeddings=512, layer_types=["full_attention"],
                                  attention_dropout=0.0, pad_token_id=0, eos_token_id=1)
            config._attn_implementation = "eager"
            base = Exaone4ForCausalLM(config).float().eval()
        original = copy.deepcopy(base)
        scorer = ExaoneChoiceBackend(model=base, tokenizer=Tokenizer(), device="cpu")
        row = training_row()
        question = ChoiceQuestion(**row["question"])
        before = scorer.score([question])[0].logits
        result = train([row], tmp_path, model_id="LGAI-EXAONE/EXAONE-4.0-1.2B",
                       revision="pinned-test-revision", device="cpu", epochs=1,
                       accumulation_steps=1, rank=2, backend=scorer)
        assert result["eligible_for_live"] is False
        assert result["model_fingerprint"] == adapter_fingerprint(tmp_path)
        assert not hasattr(base.get_output_embeddings(), "lora_A")
        base.eval()
        after = scorer.score([question])[0].logits
        assert before != after
        loaded = peft.PeftModel.from_pretrained(original, tmp_path).eval()
        restored = ExaoneChoiceBackend(model=loaded, tokenizer=Tokenizer(), device="cpu").score([question])[0]
        torch.testing.assert_close(torch.tensor(after), torch.tensor(restored.logits), rtol=1e-5, atol=1e-6)
    finally:
        torch.set_num_threads(previous_threads)
