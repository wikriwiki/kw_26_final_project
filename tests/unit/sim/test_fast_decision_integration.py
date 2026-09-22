"""Fast decision capture must never alter teacher decisions or live state."""
from __future__ import annotations

import copy
import json
import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SIM_DIR = Path(__file__).resolve().parents[3] / "scripts" / "sim"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

import stage2_poi as s2
import neo4j_load._common as db_common
from dawn_context import DawnContext
from stage1_intent import Stage1Output


def _stage1():
    return Stage1Output.model_validate({"daily_propensity": 0.62, "events": [
        {"time": "06:30", "anchor": "residence", "category": "집", "intent": "기상"},
        {"time": "12:00", "anchor": "workplace", "category": "식사",
         "sub_category": "한식", "intent": "점심", "trigger": "lifestyle"},
        {"time": "17:00", "anchor": "residence", "category": "카페",
         "sub_category": "커피", "intent": "휴식", "trigger": "none"},
        {"time": "18:00", "anchor": "residence", "category": "집", "intent": "귀가"},
    ]})


def _teacher():
    return {"picks": [
        {"order": 1, "poi_id": "C_MEAL", "actual_spent": 11000,
         "actual_satisfaction": 0.28, "policy_spend": {},
         "pick_reason": "최근 방문 경험과 점심 일정", "pick_factor": "known"},
        {"order": 2, "poi_id": "C_CAFE", "actual_spent": 5000,
         "actual_satisfaction": 0.82, "policy_spend": {},
         "pick_reason": "휴식 장소까지 가까운 거리", "pick_factor": "distance"},
    ]}


def _response(payload):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))],
        usage=SimpleNamespace(prompt_tokens=150, completion_tokens=70),
    )


@pytest.fixture
def isolated(monkeypatch):
    """All model/DB work is replaced; unexpected network accesses are not needed."""
    monkeypatch.delenv("SIM_FAST_MODE", raising=False)
    candidates = {
        order: [{"poi_id": pid, "known": True, "name": pid, "km": 0.1,
                 "avg_satisfaction": sat, "visit_count": 3,
                 "price_band": 2, "price_factor": 1.0,
                 "coupon_eligible": False, "unit_anchor": price}]
        for order, pid, sat, price in [(1, "C_MEAL", 0.35, 11000), (2, "C_CAFE", 0.8, 5000)]
    }
    calls = []
    queries = []
    monkeypatch.setattr(s2, "fetch_candidates_for_events", lambda *a, **k: copy.deepcopy(candidates))

    def llm(*args, **kwargs):
        calls.append((args, kwargs))
        return _response(_teacher())

    monkeypatch.setattr(s2, "_llm_call", llm)

    class ReadSession:
        def run(self, query, **params):
            queries.append((query, params))
            return [{"pid": "C_OLD"}]

    @contextmanager
    def driver_session():
        yield ReadSession()

    monkeypatch.setattr(db_common, "driver_session", driver_session)
    return SimpleNamespace(candidates=candidates, calls=calls, queries=queries)


def _install_capture(monkeypatch, start):
    runtime = ModuleType("fast_decision.runtime")
    runtime.start_capture = start
    monkeypatch.setitem(sys.modules, "fast_decision.runtime", runtime)


def _call(**kwargs):
    return s2.call_stage2(
        "A_TEST", _stage1(), {"daily_wd": 30000, "daily_we": 35000},
        date(2026, 5, 5), state={"balance": 100000, "mood": 0.4, "fatigue": 0.6},
        **kwargs,
    )


def test_default_off_keeps_teacher_contract_without_starting_capture(monkeypatch, isolated):
    started = []
    _install_capture(monkeypatch, lambda **kwargs: started.append(kwargs))
    output, candidates, meta = _call()

    assert started == []
    assert output.model_dump()["picks"] == _teacher()["picks"]
    assert candidates == isolated.candidates
    assert len(isolated.calls) == 1
    assert "acceleration" not in meta
    assert "t_fast_capture" not in meta["s2_timing"]


def test_recent_memory_query_excludes_current_and_future_dates(isolated):
    _call()
    query, params = isolated.queries[0]
    assert "m.day >= date($since)" in query
    assert "m.day < date($today)" in query
    assert params == {"aid": "A_TEST", "since": "2026-05-02", "today": "2026-05-05"}


@pytest.mark.parametrize("mode", ["record", "shadow"])
def test_capture_sees_full_context_but_cannot_mutate_teacher(monkeypatch, isolated, mode):
    monkeypatch.setenv("SIM_FAST_MODE", mode)
    observed = {}
    context = DawnContext(
        persona={"income": "중"}, state={"mood": 0.4},
        memory=[{"id": "M1", "satisfaction": 0.25}],
        social=[{"id": "A_PEER"}],
    )

    class Capture:
        def finish(self, output, meta):
            observed["output"] = output.model_dump()
            observed["meta"] = copy.deepcopy(meta)
            # Even a faulty extension must be isolated from returned teacher data.
            output.picks[0].actual_satisfaction = 1.0
            meta["price_by_poi"].clear()
            return {"mode": mode, "teacher_preserved": True}

    def start(**kwargs):
        assert isolated.calls == []  # Decision snapshot is made before teacher inference.
        observed["input"] = copy.deepcopy(kwargs)
        kwargs["persona"].clear()
        kwargs["state"].clear()
        kwargs["stage1"].events.clear()
        kwargs["candidates"].clear()
        kwargs["context"].memory.clear()
        return Capture()

    _install_capture(monkeypatch, start)
    output, candidates, meta = _call(decision_context=context)

    assert set(observed["input"]["candidates"]) == {1, 2}
    assert observed["input"]["state"]["mood"] == 0.4
    assert observed["input"]["recent_poi_ids"] == {"C_OLD"}
    assert observed["input"]["context"].memory == context.memory
    assert "C_MEAL" in observed["input"]["user_prompt"]
    assert observed["output"]["picks"] == _teacher()["picks"]
    assert output.model_dump()["picks"] == _teacher()["picks"]
    assert candidates == isolated.candidates
    assert meta["price_by_poi"]["C_MEAL"] == (2, 1.0)
    assert meta["acceleration"] == {"mode": mode, "teacher_preserved": True}
    assert meta["s2_timing"]["t_fast_finish"] >= 0
    assert len(context.memory) == 1


@pytest.mark.parametrize("failure", ["start", "finish"])
def test_capture_errors_leave_teacher_result_unchanged(monkeypatch, isolated, failure):
    monkeypatch.setenv("SIM_FAST_MODE", "shadow")

    class Capture:
        def finish(self, output, meta):
            raise OSError("test log failure")

    def start(**kwargs):
        if failure == "start":
            raise ValueError("test setup failure")
        return Capture()

    _install_capture(monkeypatch, start)
    output, _, meta = _call()
    assert output.model_dump()["picks"] == _teacher()["picks"]
    assert len(isolated.calls) == 1
    assert meta["acceleration"]["status"] == "capture_error"
    assert meta["acceleration"]["teacher_preserved"] is True


def test_enabled_capture_returning_none_is_reported(monkeypatch, isolated):
    monkeypatch.setenv("SIM_FAST_MODE", "record")
    _install_capture(monkeypatch, lambda **kwargs: None)
    output, _, meta = _call()

    assert output.model_dump()["picks"] == _teacher()["picks"]
    assert meta["acceleration"] == {
        "mode": "record", "status": "capture_unavailable",
        "applied": False, "teacher_preserved": True,
    }


def test_review_second_pass_finishes_capture_once_with_review_evidence(monkeypatch, isolated):
    monkeypatch.setenv("SIM_FAST_MODE", "record")
    finished = []
    teacher_calls = []

    class Capture:
        def finish(self, output, meta):
            finished.append((output.model_dump(), meta))
            return {"mode": "record"}

    def teacher(*args, **kwargs):
        teacher_calls.append(args)
        payload = _teacher()
        if len(teacher_calls) == 1:
            payload["review_lookup_requests"] = ["C_CAFE"]
        return _response(payload)

    _install_capture(monkeypatch, lambda **kwargs: Capture())
    monkeypatch.setattr(s2, "_llm_call", teacher)
    monkeypatch.setattr(s2, "lookup_reviews_batch", lambda *a, **k: {
        "C_CAFE": {"rating": 4.2, "rating_count": 3, "reviews": []},
    })
    monkeypatch.setattr(s2, "format_review_block", lambda pid, info: f"{pid}: rating {info['rating']}")
    output, _, meta = _call()

    assert len(teacher_calls) == 2
    assert len(finished) == 1
    assert finished[0][1]["review_lookup_count"] == 1
    assert "C_CAFE" in finished[0][1]["review_lookup_used"]
    assert output.model_dump()["picks"] == _teacher()["picks"]
    assert meta["s2_timing"]["n_llm_calls"] == 2


def test_teacher_fallback_is_captured_and_marked_without_student_replacement(monkeypatch, isolated):
    monkeypatch.setenv("SIM_FAST_MODE", "record")
    finished = []

    class Capture:
        def finish(self, output, meta):
            finished.append((output.model_dump(), meta))
            return {"mode": "record"}

    def failed_teacher(*args, **kwargs):
        raise TimeoutError("test teacher timeout")

    _install_capture(monkeypatch, lambda **kwargs: Capture())
    monkeypatch.setattr(s2, "_llm_call", failed_teacher)
    output, _, meta = _call(max_retry=0)

    assert len(finished) == 1
    assert finished[0][1]["fallback_only"] is True
    assert finished[0][0] == output.model_dump()
    assert meta["fallback_only"] is True
    assert {p.poi_id for p in output.picks} == {"C_MEAL", "C_CAFE"}


def test_real_record_runtime_exports_real_stage1_shape_to_training_dataset(
    monkeypatch, tmp_path, isolated,
):
    """Exercise the real capture/export boundary, including home events without picks."""
    from fast_decision.dataset import build_dataset, read_jsonl

    monkeypatch.setenv("SIM_FAST_MODE", "record")
    monkeypatch.setenv("LLM_MODE", "exaone")
    path = tmp_path / "captures.jsonl"
    monkeypatch.setenv("SIM_FAST_CAPTURE_PATH", str(path))
    context = DawnContext(
        persona={"income": "중"}, state={"mood": 0.4, "fatigue": 0.6, "balance": 100000},
    )

    output, _, meta = _call(decision_context=context)
    assert meta["acceleration"]["status"] == "recorded"
    assert meta["acceleration"]["teacher_validated"] is True
    records = list(read_jsonl(path))
    pending = list(read_jsonl(path.with_suffix(".pending.jsonl")))
    assert len(records) == len(pending) == 1
    assert records[0]["teacher"]["output"] == output.model_dump()
    assert "teacher" not in pending[0]
    assert records[0]["snapshot"] == pending[0]["snapshot"]
    assert len(records[0]["snapshot"]["stage1"]["events"]) == 4
    assert len(records[0]["teacher"]["output"]["picks"]) == 2

    # Unit fixtures are explicitly segregated before exercising dataset export.
    records[0]["provenance"]["synthetic"] = True
    manifest = build_dataset(records, tmp_path / "dataset")
    assert manifest["accepted_captures"] == 1
    assert manifest["rejected_captures"] == {}
    assert manifest["examples"]["synthetic"] > 0
    assert manifest["examples"]["test"] == manifest["examples"]["calibration"] == 0


def test_real_shadow_policy_defer_never_loads_small_model(monkeypatch, tmp_path, isolated):
    from fast_decision import runtime
    from fast_decision.dataset import read_jsonl

    monkeypatch.setenv("SIM_FAST_MODE", "shadow")
    monkeypatch.setenv("LLM_MODE", "exaone")
    path = tmp_path / "shadow.jsonl"
    monkeypatch.setenv("SIM_FAST_CAPTURE_PATH", str(path))
    loads = []

    def unexpected_model_load():
        loads.append(True)
        raise AssertionError("Policy decisions must defer before model load")

    monkeypatch.setattr(runtime, "_backend", unexpected_model_load)
    context = DawnContext(persona={}, state={"balance": 100000, "mood": 0.4, "fatigue": 0.6})
    output, _, meta = _call(
        decision_context=context,
        active_policies=[{"id": "P_TEST", "type": "grant", "poi_restricted": True}],
        grant_remaining={"P_TEST": 20000},
    )

    assert loads == []
    assert output.model_dump()["picks"] == _teacher()["picks"]
    assert meta["acceleration"]["student_status"] == "deferred"
    student = list(read_jsonl(path))[0]["student"]
    assert "policy_context" in student["reasons"]
    assert student["output"] is None
    assert student["eligible_for_live"] is False


@pytest.mark.parametrize("invalid_spend", [0, None])
def test_imputed_teacher_spend_is_excluded_from_real_capture_dataset(
    monkeypatch, tmp_path, isolated, invalid_spend,
):
    from fast_decision.dataset import build_dataset, read_jsonl

    monkeypatch.setenv("SIM_FAST_MODE", "record")
    monkeypatch.setenv("LLM_MODE", "exaone")
    path = tmp_path / "imputed.jsonl"
    monkeypatch.setenv("SIM_FAST_CAPTURE_PATH", str(path))
    teacher = _teacher()
    teacher["picks"][0]["actual_spent"] = invalid_spend
    monkeypatch.setattr(s2, "_llm_call", lambda *a, **k: _response(teacher))
    output, _, meta = _call()

    assert output.picks[0].actual_spent > 0
    assert meta["missing_picks_filled"] == 0
    assert meta["spend_imputed"] == 1
    assert meta["acceleration"]["teacher_validated"] is False
    records = list(read_jsonl(path))
    assert "spend_imputed" in records[0]["teacher"]["meta"]["validation_errors"]
    records[0]["provenance"]["synthetic"] = True
    manifest = build_dataset(records, tmp_path / "dataset")
    assert manifest["accepted_captures"] == 0
    assert manifest["rejected_captures"] == {"unvalidated_teacher": 1}


@pytest.mark.parametrize("mode", ["off", "shadow"])
def test_process_one_passes_context_only_when_enabled_and_preserves_accounting(
    monkeypatch, tmp_path, mode,
):
    # run_simulation creates output folders at import time; direct them to this test.
    monkeypatch.setenv("SIM_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("SIM_FAST_MODE", mode)
    monkeypatch.setenv("CONSUMPTION_MODEL", "legacy")
    import run_simulation as sim

    ctx = DawnContext(
        persona={"income": "중", "daily_wd": 30000, "daily_we": 35000},
        state={"balance": 100000, "mood": 0.4, "fatigue": 0.6},
        memory=[{"id": "M1", "satisfaction": 0.25}],
    )
    call_kwargs = []
    written = []
    monkeypatch.setattr(sim, "build_dawn_context", lambda *args: ctx)
    monkeypatch.setattr(sim, "call_stage1", lambda *a, **k: (
        _stage1(), {"tokens_in": 100, "tokens_out": 40, "attempt": 0},
    ))

    def teacher_stage2(*args, **kwargs):
        call_kwargs.append(kwargs)
        meta = {"tokens_in": 150, "tokens_out": 70}
        if mode != "off":
            meta["acceleration"] = {"mode": mode, "teacher_preserved": True}
        return s2.Stage2Output.model_validate(_teacher()), {}, meta

    def write_plan(*args, **kwargs):
        written.append(copy.deepcopy(args[2]))
        return "plan", len(args[2])

    monkeypatch.setattr(sim, "call_stage2", teacher_stage2)
    monkeypatch.setattr(sim, "write_plan", write_plan)
    monkeypatch.setattr(sim, "night_create_state", lambda *a, **k: {
        "balance": 84000, "mood": 0.445, "fatigue": 0.45,
    })
    result = sim.process_one("A_TEST", date(2026, 5, 5), 0)

    assert result["status"] == "ok"
    assert sum(ev["actual_spent"] or 0 for ev in written[0]) == 16000
    assert [ev["actual_satisfaction"] for ev in written[0]] == [None, 0.28, 0.82, None]
    assert result["tokens_in"] == 250
    if mode == "off":
        assert "decision_context" not in call_kwargs[0]
        assert "acceleration" not in result
    else:
        assert call_kwargs[0]["decision_context"] is ctx
        assert result["acceleration"] == {"mode": mode, "teacher_preserved": True}
