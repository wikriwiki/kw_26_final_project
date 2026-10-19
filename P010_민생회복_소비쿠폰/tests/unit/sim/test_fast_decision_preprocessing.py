"""Exact-input regressions for bounded preprocessing optimizations."""
import copy
import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from scripts.sim.fast_decision.__main__ import smoke_record
from scripts.sim.fast_decision.contracts import canonical
from scripts.sim.fast_decision.planner import PreparedState, teacher_examples
from tests.unit.sim.test_fast_decision_backend import (
    CharacterTokenizer, ContextMergingTokenizer, backend, question, tiny_model, cpu_threads,
)


class BatchedTokenizer(CharacterTokenizer):
    is_fast = True

    def __init__(self):
        self.scalar_calls = 0
        self.batch_calls = 0

    def encode(self, text, **kwargs):
        self.scalar_calls += 1
        return super().encode(text, **kwargs)

    def __call__(self, texts, **kwargs):
        self.batch_calls += 1
        assert kwargs["padding"] is False and kwargs["truncation"] is False
        assert kwargs["return_attention_mask"] is False
        return {"input_ids": [CharacterTokenizer.encode(self, text, add_special_tokens=False)
                              for text in texts]}


def test_batch_tokenization_matches_scalar_and_repeated_prompts_hit_cache(tiny_model):
    tokenizer = BatchedTokenizer()
    scorer = backend(tiny_model, tokenizer)
    q = question()
    expected = backend(tiny_model).encode_question(q)
    assert scorer.encode_question(q) == expected
    assert tokenizer.batch_calls == 1
    calls = tokenizer.scalar_calls
    # Question keys are log identities, not semantic inputs.
    repeated = dataclasses.replace(q, key="different-log-key")
    assert scorer.encode_question(repeated) == expected
    assert tokenizer.scalar_calls == calls and tokenizer.batch_calls == 1


def test_cache_return_copies_and_changed_state_revalidated(tiny_model):
    tokenizer = BatchedTokenizer()
    scorer = backend(tiny_model, tokenizer)
    q = question()
    expected = copy.deepcopy(scorer.encode_question(q))
    result = scorer.encode_question(q)
    result["input_ids"].clear()
    result["token_ids"].clear()
    result["labels"].reverse()
    assert scorer.encode_question(q) == expected
    scorer.encode_question(dataclasses.replace(q, state="다른 날의 다른 상태"))
    assert tokenizer.batch_calls == 2


def test_cache_is_bounded_by_entries_and_total_tokens(tiny_model):
    tokenizer = BatchedTokenizer()
    scorer = backend(tiny_model, tokenizer, encoding_cache_size=1)
    a, b = question("a", "첫날"), question("b", "둘째 날")
    scorer.encode_question(a)
    scorer.encode_question(b)
    scorer.encode_question(a)
    assert tokenizer.batch_calls == 3 and len(scorer._encoding_cache) == 1
    token_bounded = backend(tiny_model, encoding_cache_tokens=1)
    token_bounded.encode_question(a)
    assert not token_bounded._encoding_cache


def test_cached_prompt_respects_lowered_context_limit(tiny_model):
    scorer = backend(tiny_model)
    q = question()
    scorer.encode_question(q)
    scorer.max_tokens = 1
    with pytest.raises(ValueError, match="no truncation"):
        scorer.encode_question(q)


def test_explicit_tokenizer_cache_reset_revalidates_context(tiny_model):
    scorer = backend(tiny_model, BatchedTokenizer())
    scorer.encode_question(question())
    scorer.tokenizer = ContextMergingTokenizer()
    scorer.clear_encoding_cache()
    with pytest.raises(ValueError, match="exact context"):
        scorer.encode_question(question())


def test_unsupported_output_adapter_rejected_before_prefill(tiny_model, monkeypatch):
    tiny_model.get_output_embeddings().lora_A = object()
    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported output adapter must not spend a forward pass")
    monkeypatch.setattr(tiny_model.base_model, "forward", forbidden)
    with pytest.raises(ValueError, match="Output-head"):
        backend(tiny_model).score([question()])


def test_parallel_encoders_share_one_verified_cache_entry(tiny_model):
    tokenizer = BatchedTokenizer()
    scorer = backend(tiny_model, tokenizer)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: scorer.encode_question(question()), range(24)))
    assert all(item == results[0] for item in results)
    assert tokenizer.batch_calls == 1


def test_batched_context_merge_is_still_rejected(tiny_model):
    class MergingBatchTokenizer(BatchedTokenizer):
        def __call__(self, texts, **kwargs):
            return {"input_ids": [ContextMergingTokenizer().encode(text) for text in texts]}
    with pytest.raises(ValueError, match="exact context"):
        backend(tiny_model, MergingBatchTokenizer()).encode_question(question())


def test_prepared_state_is_byte_identical_to_original_canonical_state():
    snapshot = smoke_record()["snapshot"]
    snapshot["context"]["memory"] = [{"summary": '문자열: "provisional_picks": [], \n한글'}]
    original = copy.deepcopy(snapshot)
    prepared = PreparedState(snapshot)
    fields = ("today", "stage1", "persona", "state", "candidates", "recent_poi_ids",
              "active_policies", "grant_remaining", "context")
    for picks in ([], [{"order": 0, "poi_id": "fixture-a"}],
                  [{"order": 0, "poi_id": "fixture-b", "actual_spent": 12000}]):
        expected = {k: original.get(k) for k in fields}
        expected["provisional_picks"] = picks
        assert prepared.render(picks) == canonical(expected)
    # Preparation freezes only the input copy; later caller mutation cannot leak.
    snapshot["state"]["mood"] = .1
    assert json.loads(prepared.render([]))["state"]["mood"] == .6


def test_teacher_questions_share_exact_evidence_and_keep_conditional_choices():
    row = smoke_record()
    examples = teacher_examples(row["snapshot"], row["teacher"]["output"])
    for example in examples:
        state = json.loads(example.question.state)
        assert state["state"] == row["snapshot"]["state"]
        assert state["candidates"] == row["snapshot"]["candidates"]
    spend = next(e for e in examples if e.kind == "spend")
    assert json.loads(spend.question.state)["provisional_picks"][0]["poi_id"] == "fixture-a"
