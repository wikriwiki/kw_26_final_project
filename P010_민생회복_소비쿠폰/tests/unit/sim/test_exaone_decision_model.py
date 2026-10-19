"""Actual decision architecture tests, all CPU-only with random tiny EXAONE."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.sim.fast_decision.__main__ import smoke_record
from scripts.sim.fast_decision.dataset import build_dataset, read_jsonl
from scripts.sim.fast_decision.decision_data import IGNORE, KINDS, LABELS, MAX_CANDIDATES, model_examples
from scripts.sim.fast_decision.decision_training import train_decision


def rows_for(tmp_path, row=None, name="dataset"):
    build_dataset([row or smoke_record()], tmp_path / name)
    return list(read_jsonl(tmp_path / name / "synthetic.jsonl"))


class TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"] + "\n"

    def encode(self, text, **kwargs):
        return [2 + ord(char) % 120 for char in text]

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        Path(path, "test_tokenizer.json").write_text('{"test_only":true}', encoding="utf-8")


@pytest.fixture
def tiny():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from transformers import Exaone4Config, Exaone4Model
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        config = Exaone4Config(vocab_size=128, hidden_size=16, intermediate_size=32,
                              num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                              max_position_embeddings=8192, layer_types=["full_attention"],
                              attention_dropout=0.0, pad_token_id=0, eos_token_id=1)
        config._attn_implementation = "eager"
        yield Exaone4Model(config).float().eval()
    torch.set_num_threads(prior_threads)


def decision_model(encoder):
    from scripts.sim.fast_decision.decision_model import DecisionConfig, ExaoneDecisionModel
    return ExaoneDecisionModel(encoder, DecisionConfig(hidden_size=16, decision_width=12,
                                                       revision="test-pinned-revision")).eval()


def test_targets_change_without_current_answers_entering_input(tmp_path):
    row = smoke_record()
    first = model_examples(rows_for(tmp_path, row), allow_synthetic=True)[0]
    row["teacher"]["output"]["picks"][0].update(poi_id="fixture-b", actual_spent=12000, actual_satisfaction=.2)
    second = model_examples(rows_for(tmp_path, row, "changed"), allow_synthetic=True)[0]
    assert first.text == second.text
    assert first.targets != second.targets
    assert first.candidate_ids == second.candidate_ids
    assert '"provisional_picks":[]' in first.text


def test_later_event_input_keeps_previous_transaction(tmp_path):
    row = smoke_record()
    snapshot = row["snapshot"]
    snapshot["stage1"]["events"].append(copy.deepcopy(snapshot["stage1"]["events"][0]))
    snapshot["candidates"]["1"] = [{**candidate, "poi_id": candidate["poi_id"] + "-later"}
                                      for candidate in snapshot["candidates"]["0"]]
    row["teacher"]["output"]["picks"].append({**row["teacher"]["output"]["picks"][0],
                                              "order": 1, "poi_id": "fixture-b-later"})
    examples = model_examples(rows_for(tmp_path, row), allow_synthetic=True)
    assert len(examples) == 2
    assert '"actual_spent":10000' not in examples[0].text
    assert '"actual_spent":10000' in examples[1].text
    assert examples[1].targets["poi"] == 1


def test_scope_defer_and_partial_defer_mask_unobserved_labels(tmp_path):
    row = smoke_record()
    row["snapshot"]["state"]["mood"] = .1
    blocked = model_examples(rows_for(tmp_path, row), allow_synthetic=True)[0]
    assert blocked.targets["route"] == 1
    assert all(blocked.targets[kind] == IGNORE for kind in KINDS[1:])
    row = smoke_record()
    row["teacher"]["output"]["picks"][0]["actual_satisfaction"] = .28
    partial = model_examples(rows_for(tmp_path, row, "partial"), allow_synthetic=True)[0]
    assert partial.targets["satisfaction"] == LABELS["satisfaction"].index("DEFER")
    assert partial.targets["factor"] == IGNORE


def test_corrupted_conditioning_is_rejected(tmp_path):
    rows = rows_for(tmp_path)
    next(row for row in rows if row["kind"] == "spend")["question"]["state"] += "leaked-answer"
    with pytest.raises(ValueError, match="conditional question"):
        model_examples(rows, allow_synthetic=True)


def test_dry_run_imports_no_ml_and_creates_no_weights(tmp_path):
    rows_for(tmp_path)
    command = (
        "import sys; from scripts.sim.fast_decision.decision_training import main; "
        "main(['--input',sys.argv[1],'--output',sys.argv[2],'--allow-synthetic','--dry-run']); "
        "assert all(name not in sys.modules for name in ['torch','transformers','peft'])"
    )
    result = subprocess.run([sys.executable, "-c", command, str(tmp_path / "dataset/synthetic.jsonl"),
                             str(tmp_path / "dry")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "dry/dry_run.json").read_text(encoding="utf-8"))
    assert report["decision_examples"] == 1 and not report["trained"] and not report["weights_loaded"]


def test_real_cpu_training_is_not_silently_started(tmp_path):
    with pytest.raises(ValueError, match="requires CUDA"):
        train_decision(rows_for(tmp_path), tmp_path / "model", revision="fixed-revision", device="cpu", allow_synthetic=True)


def test_one_backbone_call_and_available_candidate_mask(tiny):
    import torch
    model = decision_model(tiny)
    calls = []
    hook = tiny.register_forward_hook(lambda *args: calls.append(1))
    try:
        with torch.no_grad():
            model.heads.outputs["poi"].bias[MAX_CANDIDATES - 1] = 10000
        output = model(torch.tensor([[2, 3, 4]]), torch.ones(1, 3, dtype=torch.long), torch.tensor([2]))
    finally:
        hook.remove()
    assert len(calls) == 1 and set(output.logits) == set(KINDS)
    assert not hasattr(model, "generate") and tiny.get_output_embeddings() is None
    assert torch.isneginf(output.logits["poi"][0, 2:MAX_CANDIDATES]).all()
    assert output.selected["poi"].item() in (0, 1, MAX_CANDIDATES)


def test_condition_heads_preserve_selected_place_and_spend(tiny):
    import torch
    model = decision_model(tiny)
    ids = torch.tensor([[2, 3, 4]])
    target = {kind: torch.tensor([0]) for kind in KINDS}
    first = model(ids, torch.ones_like(ids), torch.tensor([2]), target)
    target["poi"] = torch.tensor([1])
    second = model(ids, torch.ones_like(ids), torch.tensor([2]), target)
    torch.testing.assert_close(first.logits["poi"], second.logits["poi"])
    assert not torch.equal(first.logits["spend"], second.logits["spend"])
    target["spend"] = torch.tensor([3])
    third = model(ids, torch.ones_like(ids), torch.tensor([2]), target)
    torch.testing.assert_close(second.logits["spend"], third.logits["spend"])
    assert not torch.equal(second.logits["satisfaction"], third.logits["satisfaction"])


def test_masked_actions_do_not_backpropagate_on_route_defer(tiny):
    import torch
    model = decision_model(tiny).train()
    target = {kind: torch.tensor([1 if kind == "route" else IGNORE]) for kind in KINDS}
    ids = torch.tensor([[2, 3, 4]])
    result = model(ids, torch.ones_like(ids), torch.tensor([0]), target)
    result.loss.backward()
    assert model.heads.outputs["route"].weight.grad.abs().sum() > 0
    assert model.heads.outputs["poi"].weight.grad is None
    target["poi"] = torch.tensor([0])
    with pytest.raises(ValueError, match="after DEFER"):
        model(ids, torch.ones_like(ids), torch.tensor([0]), target)


def test_left_and_right_padding_match_single_input(tiny):
    import torch
    model = decision_model(tiny)
    single = model(torch.tensor([[2, 3]]), torch.tensor([[1, 1]]), torch.tensor([2]))
    padded = model(torch.tensor([[0, 2, 3], [2, 3, 0]]), torch.tensor([[0, 1, 1], [1, 1, 0]]), torch.tensor([2, 2]))
    for kind in KINDS:
        torch.testing.assert_close(single.logits[kind][0], padded.logits[kind][0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(single.logits[kind][0], padded.logits[kind][1], rtol=1e-5, atol=1e-6)


def test_cpu_lora_and_heads_train_save_reload_and_detect_tampering(tmp_path, tiny):
    import torch
    from scripts.sim.fast_decision.decision_model import load_checkpoint
    original = copy.deepcopy(tiny)
    original_weights = {key: value.clone() for key, value in tiny.state_dict().items()}
    manifest = train_decision(rows_for(tmp_path), tmp_path / "model", revision="test-pinned-revision",
                              device="cpu", epochs=2, learning_rate=.01, accumulation_steps=1,
                              rank=2, decision_width=12, max_tokens=8192, allow_synthetic=True,
                              encoder=tiny, tokenizer=TinyTokenizer())
    assert manifest["trained"] and manifest["injected_test_encoder"] and not manifest["eligible_for_live"]
    assert manifest["encoder_forwards_per_event"] == 1
    assert manifest["history"][-1]["mean_conditional_head_loss"] < manifest["history"][0]["mean_conditional_head_loss"]
    assert any("lora_B" in key and value.abs().sum() > 0 for key, value in tiny.state_dict().items())
    # The original base tensors are frozen; only adapters and new heads learn.
    for key, value in tiny.state_dict().items():
        if "lora_" not in key:
            torch.testing.assert_close(value, original_weights[key.replace(".base_layer.", ".")])
    first, _ = load_checkpoint(tmp_path / "model", encoder=copy.deepcopy(original))
    second, _ = load_checkpoint(tmp_path / "model", encoder=copy.deepcopy(original))
    ids = torch.tensor([[2, 3, 4]])
    for kind in KINDS:
        torch.testing.assert_close(first(ids, torch.ones_like(ids), torch.tensor([2])).logits[kind],
                                   second(ids, torch.ones_like(ids), torch.tensor([2])).logits[kind])
    with pytest.raises(ValueError, match="test/synthetic"):
        load_checkpoint(tmp_path / "model")
    (tmp_path / "model/tokenizer/test_tokenizer.json").write_text('{}', encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_checkpoint(tmp_path / "model", encoder=copy.deepcopy(original))


def test_predictor_uses_one_forward_and_defers_without_filling(tiny):
    import torch
    from scripts.sim.fast_decision.decision_inference import DecisionPredictor
    model = decision_model(tiny)
    with torch.no_grad():
        for kind, head in model.heads.outputs.items():
            head.weight.zero_()
            head.bias.fill_(-10)
            chosen = {"route": 0, "poi": 0, "spend": LABELS["spend"].index("10000"),
                      "satisfaction": LABELS["satisfaction"].index("0.70"), "factor": 0}[kind]
            head.bias[chosen] = 10
    predictor = DecisionPredictor(model, TinyTokenizer(), max_tokens=8192)
    snapshot = smoke_record()["snapshot"]
    result = predictor.propose(snapshot)
    assert result["status"] == "proposed" and result["encoder_forwards"] == 1
    assert result["output"]["picks"][0]["actual_satisfaction"] == .7
    with torch.no_grad():
        model.heads.outputs["satisfaction"].bias[-1] = 100
    result = predictor.propose(snapshot)
    assert result["output"] is None and result["reasons"] == ["decision_model_deferred:satisfaction"]
    snapshot["state"]["mood"] = .1
    assert predictor.propose(snapshot)["encoder_forwards"] == 0
