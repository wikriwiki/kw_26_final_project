"""CPU numerical/contract checks; random weights make no model-quality claim.

The tiny EXAONE is constructed from configuration and never downloads weights.
The deliberately simple tokenizer tests the backend's boundaries, not LG's real
tokenizer or a pretrained EXAONE's understanding of a simulation state.
"""
from __future__ import annotations

import string
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import AutoModelForCausalLM, AutoTokenizer, Exaone4Config, Exaone4ForCausalLM

SIM_DIR = Path(__file__).resolve().parents[3] / "scripts" / "sim"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from fast_decision.backend import ExaoneChoiceBackend
from fast_decision.contracts import ChoiceQuestion


class CharacterTokenizer:
    """Keep ASCII choice codes unique while representing Korean input cheaply."""

    pad_token_id = 0
    eos_token_id = 1
    all_special_ids = [0, 1]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return f"<user>\n{messages[0]['content']}\n<assistant>\n"

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) + 3 if ord(c) < 128 else 160 + ord(c) % 96 for c in text]


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    # Very small matrix operations otherwise spend most time scheduling threads.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_model():
    config = Exaone4Config(
        vocab_size=320,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        sliding_window=24,
        layer_types=["sliding_attention", "full_attention"],
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        model = Exaone4ForCausalLM(config).to(device="cpu", dtype=torch.float32).eval()
    assert next(model.parameters()).device.type == "cpu"
    return model


def question(key="meal", state="기분 0.4, 직장 근처의 점심 후보"):
    return ChoiceQuestion(
        key=key,
        state=state,
        instructions="상태에 맞는 방문 후보를 고르세요.",
        options={"known": "단골 한식집", "new": "새로운 한식집", "defer": "정보 부족: 보류"},
    )


def backend(model, tokenizer=None, **kwargs):
    return ExaoneChoiceBackend(
        model=model, tokenizer=tokenizer or CharacterTokenizer(), device="cpu", **kwargs,
    )


def test_selected_head_logits_match_official_exaone_forward(tiny_model, monkeypatch):
    def forbidden_generate(*args, **kwargs):
        pytest.fail("Typed scoring must not generate an autoregressive answer")

    monkeypatch.setattr(tiny_model, "generate", forbidden_generate)
    scorer = backend(tiny_model)
    q = question()
    encoded = scorer.encode_question(q)
    ids = torch.tensor([encoded["input_ids"]], dtype=torch.long)
    with torch.inference_mode():
        official = tiny_model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            position_ids=torch.arange(ids.shape[1]).unsqueeze(0),
            use_cache=False,
            logits_to_keep=1,
        ).logits[0, -1, encoded["token_ids"]].float()
    actual = scorer.score([q])[0]

    torch.testing.assert_close(torch.tensor(actual.logits), official, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        torch.tensor(actual.probabilities), official.softmax(-1), rtol=1e-5, atol=1e-6,
    )
    assert actual.labels == list(q.options)
    assert actual.key == q.key
    assert actual.input_tokens == ids.shape[1]
    assert actual.latency_seconds >= 0


def test_left_padded_batch_matches_individual_unpadded_scoring(tiny_model, monkeypatch):
    qs = [question("short", "짧음"), question("long", "기억과 상태를 포함합니다. " * 8),
          question("middle", "직장 근처 한식 선호, 피로 0.6")]
    individual = backend(tiny_model, batch_size=1).score(qs)
    observed = []
    original = tiny_model.base_model.forward

    def capture_forward(*args, **kwargs):
        observed.append({k: v.clone() if torch.is_tensor(v) else v for k, v in kwargs.items()})
        return original(*args, **kwargs)

    monkeypatch.setattr(tiny_model.base_model, "forward", capture_forward)
    batched = backend(tiny_model, batch_size=3).score(qs)

    assert len(observed) == 1
    call = observed[0]
    assert call["use_cache"] is False
    assert (call["attention_mask"][:, -1] == 1).all()
    assert (call["attention_mask"][0] == 0).any()
    assert (call["attention_mask"][1] == 1).all()
    for i, (single, batch) in enumerate(zip(individual, batched)):
        n = single.input_tokens
        assert call["attention_mask"][i, -n:].sum().item() == n
        torch.testing.assert_close(call["position_ids"][i, -n:], torch.arange(n))
        torch.testing.assert_close(
            torch.tensor(batch.logits), torch.tensor(single.logits), rtol=1e-5, atol=1e-6,
        )
        torch.testing.assert_close(
            torch.tensor(batch.probabilities), torch.tensor(single.probabilities),
            rtol=1e-5, atol=1e-6,
        )
        assert batch.key == single.key
        assert batch.labels == single.labels
        assert batch.input_tokens == n


class ContextMergingTokenizer(CharacterTokenizer):
    def encode(self, text, **kwargs):
        ids = super().encode(text, **kwargs)
        if text.endswith("\nA"):
            ids[-2] = 299  # Appending a code retokenizes the preceding prompt.
        return ids


class MultiTokenCodeTokenizer(CharacterTokenizer):
    def encode(self, text, **kwargs):
        ids = super().encode(text, **kwargs)
        return ids + [298] if text == "A" else ids


class ContextDifferentCodeTokenizer(CharacterTokenizer):
    def encode(self, text, **kwargs):
        ids = super().encode(text, **kwargs)
        if text.endswith("\nA"):
            ids[-1] = 299  # One token, but not the token encoded in isolation.
        return ids


class CollidingCodeTokenizer(CharacterTokenizer):
    def encode(self, text, **kwargs):
        return super().encode(text.replace("B", "A"), **kwargs)


class SpecialCodeTokenizer(CharacterTokenizer):
    all_special_ids = [0, 1, ord("A") + 3]


@pytest.mark.parametrize("tokenizer_type, message", [
    (ContextMergingTokenizer, "exact context"),
    (MultiTokenCodeTokenizer, "exact context"),
    (ContextDifferentCodeTokenizer, "exact context"),
    (SpecialCodeTokenizer, "exact context"),
    (CollidingCodeTokenizer, "collide"),
])
def test_invalid_label_encoding_fails_before_inference(tiny_model, monkeypatch, tokenizer_type, message):
    def forbidden_forward(*args, **kwargs):
        pytest.fail("Invalid label encoding must be rejected before model work")

    monkeypatch.setattr(tiny_model.base_model, "forward", forbidden_forward)
    with pytest.raises(ValueError, match=message):
        backend(tiny_model, tokenizer_type()).score([question()])


def test_all_supported_labels_are_context_verified_without_pruning(tiny_model):
    options = {f"candidate-{i}": f"후보 {i}" for i in range(52)}
    q = ChoiceQuestion("many", "후보 전체", "선택", options)
    encoded = backend(tiny_model).encode_question(q)
    assert encoded["labels"] == list(options)
    assert encoded["token_ids"] == [ord(c) + 3 for c in string.ascii_uppercase + string.ascii_lowercase]


def test_context_limit_rejects_instead_of_truncating(tiny_model, monkeypatch):
    q = question()
    n = len(backend(tiny_model).encode_question(q)["input_ids"])
    assert backend(tiny_model, max_tokens=n).score([q])[0].input_tokens == n

    def forbidden_forward(*args, **kwargs):
        pytest.fail("Oversized context must be rejected, not truncated or evaluated")

    monkeypatch.setattr(tiny_model.base_model, "forward", forbidden_forward)
    with pytest.raises(ValueError, match="no truncation"):
        backend(tiny_model, max_tokens=n - 1).score([q])


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_choice_logits_are_rejected(tiny_model, bad_value):
    with torch.no_grad():
        tiny_model.get_output_embeddings().weight[ord("A") + 3].fill_(bad_value)
    with pytest.raises(ValueError, match="Nonfinite"):
        backend(tiny_model).score([question()])


def test_eos_is_used_for_padding_when_pad_token_is_missing(tiny_model):
    tokenizer = CharacterTokenizer()
    tokenizer.pad_token_id = None
    qs = [question("short", "짧음"), question("long", "긴 상태 " * 5)]
    actual = backend(tiny_model, tokenizer, batch_size=2).score(qs)
    reference = backend(tiny_model, batch_size=1).score(qs)
    for left, right in zip(actual, reference):
        torch.testing.assert_close(
            torch.tensor(left.logits), torch.tensor(right.logits), rtol=1e-5, atol=1e-6,
        )


def test_missing_padding_and_eos_tokens_is_an_explicit_error(tiny_model):
    tokenizer = CharacterTokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id = None
    with pytest.raises(ValueError, match="padding/EOS"):
        backend(tiny_model, tokenizer).score([question()])


def test_loader_is_local_only_by_default(tiny_model, monkeypatch):
    calls = []

    def load_tokenizer(*args, **kwargs):
        calls.append(("tokenizer", kwargs))
        return CharacterTokenizer()

    def load_model(*args, **kwargs):
        calls.append(("model", kwargs))
        return tiny_model

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load_model)
    scorer = ExaoneChoiceBackend(revision="test-pinned-revision", device="cpu")
    assert [kind for kind, _ in calls] == ["tokenizer", "model"]
    for _, kwargs in calls:
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
        assert kwargs["revision"] == "test-pinned-revision"
    assert scorer.model.training is False
    assert scorer.device.type == "cpu"


def test_empty_batch_does_not_run_model(tiny_model, monkeypatch):
    def forbidden_forward(*args, **kwargs):
        pytest.fail("An empty batch must not run inference")

    monkeypatch.setattr(tiny_model.base_model, "forward", forbidden_forward)
    assert backend(tiny_model).score([]) == []
