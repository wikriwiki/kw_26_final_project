"""CPU-only runtime regressions: failed loads and immutable capture context."""
from __future__ import annotations

import copy
import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from scripts.sim.fast_decision import backend, runtime
from scripts.sim.fast_decision.__main__ import smoke_record
from scripts.sim.fast_decision.contracts import plain


@pytest.fixture
def backend_cache(monkeypatch):
    monkeypatch.setattr(runtime, "_BACKENDS", {})
    monkeypatch.setattr(runtime, "_BACKEND_FAILURES", {})
    for key in ("SIM_FAST_MODEL", "SIM_FAST_REVISION", "SIM_FAST_DEVICE",
                "SIM_FAST_ADAPTER", "SIM_FAST_MAX_TOKENS"):
        monkeypatch.delenv(key, raising=False)
    clock = [100.0]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock[0])
    return clock


def test_concurrent_missing_weights_load_once_then_recover(monkeypatch, backend_cache):
    loads = []
    recovered = object()

    def load(**kwargs):
        loads.append(kwargs)
        if len(loads) == 1:
            raise OSError("local weights unavailable")
        return recovered

    monkeypatch.setattr(backend, "ExaoneChoiceBackend", load)

    def attempt(_):
        try:
            runtime._backend()
        except Exception as exc:
            return type(exc)
        raise AssertionError("Missing local weights must not succeed")

    with ThreadPoolExecutor(max_workers=8) as pool:
        errors = list(pool.map(attempt, range(64)))
    assert len(loads) == 1
    assert errors.count(OSError) == 1
    assert errors.count(runtime.BackendLoadBackoff) == 63
    assert loads[0]["allow_download"] is False
    assert loads[0]["device"] == "cpu"
    assert len(runtime._BACKEND_FAILURES) == 1
    assert all(isinstance(x, (str, float)) for x in next(iter(runtime._BACKEND_FAILURES.values())))

    backend_cache[0] += runtime._BACKEND_FAILURE_RETRY_SECONDS
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: runtime._backend(), range(64)))
    assert all(value is recovered for value in results)
    assert len(loads) == 2
    assert runtime._BACKEND_FAILURES == {}


def test_changed_model_settings_bypass_failure_backoff(monkeypatch, backend_cache):
    calls = []
    recovered = object()

    def load(**kwargs):
        calls.append(kwargs)
        if kwargs["revision"] is None:
            raise OSError("missing revision")
        return recovered

    monkeypatch.setattr(backend, "ExaoneChoiceBackend", load)
    with pytest.raises(OSError):
        runtime._backend()
    monkeypatch.setenv("SIM_FAST_REVISION", "newly-installed-revision")
    assert runtime._backend() is recovered
    assert len(calls) == 2


def test_failure_cache_is_bounded(monkeypatch, backend_cache):
    def fail(**kwargs):
        raise OSError("x" * 1000)

    monkeypatch.setattr(backend, "ExaoneChoiceBackend", fail)
    for index in range(runtime._BACKEND_FAILURE_LIMIT + 5):
        monkeypatch.setenv("SIM_FAST_REVISION", str(index))
        with pytest.raises(OSError):
            runtime._backend()
    assert len(runtime._BACKEND_FAILURES) == runtime._BACKEND_FAILURE_LIMIT
    assert all(len(failure[2]) == 500 for failure in runtime._BACKEND_FAILURES.values())


@dataclasses.dataclass
class Context:
    persona: dict
    state: dict
    memory: list
    appointment: list
    social: list
    knows_poi_summary: list
    zone_candidates: list
    dawn_timing: dict


@pytest.mark.parametrize("context_type", ["dataclass", "dict"])
def test_context_pruning_keeps_identical_detached_evidence(context_type):
    context = Context(
        persona={"profile": ["duplicate"]}, state={"mood": 0.5},
        memory=[{"satisfaction": 0.2}], appointment=[{"peer": "friend"}],
        social=[{"mood": 0.4}], knows_poi_summary=[{"poi": "restaurant"}],
        zone_candidates=[{"zone": "home"}], dawn_timing={"elapsed": 123.0},
    )
    if context_type == "dict":
        context = dataclasses.asdict(context)
    old = plain(context)
    expected = {key: old[key] for key in runtime._CONTEXT_FIELDS}
    snapshot_context = plain(runtime._capture_context(context))
    assert snapshot_context == expected

    for key in runtime._CONTEXT_FIELDS:
        original = context[key] if isinstance(context, dict) else getattr(context, key)
        original[0].clear()
        original.clear()
    assert snapshot_context == expected


def test_capture_preserves_pre_teacher_snapshot_and_teacher_on_load_failure(
    monkeypatch, tmp_path, backend_cache,
):
    from scripts.sim import llm_client

    monkeypatch.setenv("SIM_FAST_MODE", "shadow")
    target = tmp_path / "capture.jsonl"
    monkeypatch.setenv("SIM_FAST_CAPTURE_PATH", str(target))
    monkeypatch.setattr(llm_client, "get_spec", lambda: SimpleNamespace(
        hf_id="LGAI-EXAONE/test", key="exaone"))
    attempts = []

    def missing_weights(**kwargs):
        attempts.append(kwargs)
        raise OSError("weights unavailable")

    monkeypatch.setattr(backend, "ExaoneChoiceBackend", missing_weights)
    row = smoke_record()
    kwargs = copy.deepcopy(row["snapshot"])
    kwargs.pop("snapshot_id")
    kwargs["context"]["persona"] = {"large_discarded_duplicate": [1, 2, 3]}
    teacher = row["teacher"]["output"]
    expected_teacher = copy.deepcopy(teacher)
    for index in range(2):
        capture = runtime.start_capture(**kwargs)
        assert capture is not None
        before = copy.deepcopy(capture.snapshot)
        kwargs["state"]["mood"] = 0.55 + index * 0.1
        kwargs["context"]["memory"].append({"message": "after snapshot"})
        assert capture.snapshot == before
        # Changes made after capture must not leak into the pre-decision evidence.
        result = capture.finish(teacher, {"s2_timing": {"t_llm": 1.0}})
        assert result["status"] == "recorded"
        assert result["applied"] is False
        assert teacher == expected_teacher
        # Keep the next test capture eligible for a load attempt.
        kwargs["context"]["memory"].clear()
    records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    pending = [json.loads(line) for line in target.with_suffix(".pending.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert len(attempts) == 1
    assert [r["student"]["error_type"] for r in records] == ["OSError", "BackendLoadBackoff"]
    assert [r["snapshot"] for r in records] == [r["snapshot"] for r in pending]
    assert all(r["teacher"]["output"] == expected_teacher for r in records)


def test_disabled_capture_skips_context_serialization(monkeypatch):
    monkeypatch.delenv("SIM_FAST_MODE", raising=False)
    kwargs = smoke_record()["snapshot"]
    kwargs.pop("snapshot_id")
    kwargs["context"] = object()  # Would be unsupported if traversed.
    assert runtime.start_capture(**kwargs) is None
