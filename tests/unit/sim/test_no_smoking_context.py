"""The policy reaches decisions without encoding the empirical effect."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/sim"))
from no_smoking_context import (NoSmokingContext, begin_llm_scope, clear_llm_scope,
                                configured_context, next_llm_seed, _load)


def inputs():
    return {"cohort": [{"id": "A", "smoking_status": "smoker"},
                       {"id": "B", "smoking_status": "non_smoker"},
                       {"id": "C", "smoking_status": "unknown"}],
            "pois": [{"poi_id": "billiard", "district_code": "11650", "facility_type": "billiard"},
                     {"poi_id": "golf", "district_code": "11350", "facility_type": "indoor_golf"},
                     {"poi_id": "screen", "district_code": "11710", "facility_type": "screen_golf"},
                     {"poi_id": "gangnam", "district_code": "11680", "facility_type": "billiard"},
                     {"poi_id": "outdoor", "district_code": "11650", "facility_type": "other"}],
            "assignment_seed": 45}


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.delenv("SIM_NO_SMOKING_MANIFEST", raising=False)
    monkeypatch.delenv("SIM_NO_SMOKING_ARM", raising=False)
    _load.cache_clear()
    clear_llm_scope()
    yield
    _load.cache_clear()
    clear_llm_scope()


def configured(tmp_path, monkeypatch, arm="on", **extra):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(dict(experiment_id="no_smoking_zone", **inputs(), **extra)), encoding="utf-8")
    monkeypatch.setenv("SIM_NO_SMOKING_MANIFEST", str(path))
    monkeypatch.setenv("SIM_NO_SMOKING_ARM", arm)
    return configured_context()


def test_pre_intervention_arms_have_identical_context_and_no_outcome_direction():
    off = NoSmokingContext(arm="off", **inputs())
    on = NoSmokingContext(arm="on", **inputs())
    for aid in off.agent_ids:
        assert off.context_for(aid, "2017-12-02") == on.context_for(aid, "2017-12-02")
    assert not off.is_active("2017-12-03")
    assert on.is_active("2017-12-03")
    assert "비흡연자" in on.context_for("B", "2017-12-03")["prompt"]
    assert "정보 없음" in on.context_for("C", "2017-12-03")["prompt"]
    assert "13.54" not in on.context_for("A", "2017-12-03")["prompt"]
    assert "증가" not in on.context_for("A", "2017-12-03")["prompt"]
    assert "감소" not in on.context_for("A", "2017-12-03")["prompt"]


def test_only_classified_indoor_facilities_in_three_actual_study_districts():
    runtime = NoSmokingContext(arm="on", **inputs())
    assert all(runtime.is_evaluation_poi(pid) for pid in ["billiard", "golf", "screen"])
    assert not any(runtime.is_evaluation_poi(pid) for pid in ["gangnam", "outdoor", "unmapped"])
    assert "실내 골프" in runtime.candidate_facts(["golf"])
    assert "gangnam" not in runtime.candidate_facts(["gangnam"])


def test_frozen_roster_fails_instead_of_silent_graph_subsample():
    runtime = NoSmokingContext(arm="on", **inputs())
    assert runtime.require_graph_roster(["D", "C", "B", "A"]) == ["A", "B", "C"]
    with pytest.raises(ValueError, match="missing from initialized graph"):
        runtime.require_graph_roster(["A", "B"])
    with pytest.raises(KeyError):
        runtime.context_for("D", "2017-12-03")


@pytest.mark.parametrize("kind", ["duplicate_agent", "missing_status", "duplicate_poi", "outdoor_as_golf"])
def test_ambiguous_or_invalid_registry_rejected(kind):
    doc = inputs()
    if kind == "duplicate_agent":
        doc["cohort"].append(doc["cohort"][0])
    elif kind == "missing_status":
        del doc["cohort"][0]["smoking_status"]
    elif kind == "duplicate_poi":
        doc["pois"].append(doc["pois"][0])
    else:
        doc["pois"][1]["facility_type"] = "golf"
    with pytest.raises(ValueError):
        NoSmokingContext(arm="off", **doc)


def receipt(pid, index, amount=10000, **extra):
    return dict(poi_id=pid, event_id=f"EX_{index}", agent_id="A", occurred_at="2018-01-03",
                amount=amount, purchase_status="purchased", **extra)


def test_accounting_counts_payments_not_people_visits_or_rejected_purchases():
    runtime = NoSmokingContext(arm="on", **inputs())
    rows = [receipt("billiard", 1), receipt("billiard", 2, 5000), receipt("golf", 3, 0),
            receipt("gangnam", 4), receipt("outdoor", 5), receipt("unmapped", 6)]
    rows.append(dict(receipt("screen", 7), purchase_status="not_purchased"))
    result = runtime.summarize_receipts(rows, "A", "2018-01-03")
    assert result["payment_count"] == 2
    assert result["revenue_krw"] == 15000
    assert result["by_poi"] == [{"poi_id": "billiard", "district_code": "11650",
                                 "facility_type": "billiard", "payment_count": 2, "revenue_krw": 15000}]


def test_budget_reduced_purchase_still_counts_its_actual_payment():
    row = dict(receipt("billiard", 1, 3000), purchase_status="reduced")
    result = NoSmokingContext(arm="on", **inputs()).summarize_receipts([row], "A", "2018-01-03")
    assert result["payment_count"] == 1
    assert result["revenue_krw"] == 3000


@pytest.mark.parametrize("kind", ["duplicate", "wrong_agent", "wrong_date", "nan", "negative", "fraction"])
def test_bad_executed_receipts_fail_closed(kind):
    rows = [receipt("billiard", 1)]
    if kind == "duplicate": rows.append(rows[0])
    elif kind == "wrong_agent": rows[0]["agent_id"] = "B"
    elif kind == "wrong_date": rows[0]["occurred_at"] = "2018-01-04"
    elif kind == "nan": rows[0]["amount"] = float("nan")
    elif kind == "negative": rows[0]["amount"] = -1
    else: rows[0]["amount"] = 1.5
    with pytest.raises(ValueError):
        NoSmokingContext(arm="on", **inputs()).summarize_receipts(rows, "A", "2018-01-03")


def test_manifest_rejects_evaluation_fields_and_requires_both_envs(tmp_path, monkeypatch):
    assert configured_context() is None
    monkeypatch.setenv("SIM_NO_SMOKING_ARM", "on")
    with pytest.raises(ValueError, match="both required"):
        configured_context()
    with pytest.raises(ValueError, match="evaluation data"):
        configured(tmp_path, monkeypatch, ground_truth={"effect": 13.54})


def test_both_decision_stages_receive_same_rule_without_mutating_persona_cache(tmp_path, monkeypatch):
    from dawn_context import _format_persona
    from stage2_poi import build_stage2_prompt
    runtime = configured(tmp_path, monkeypatch)
    original = {"id": "A", "lifestyle": "평소 운동을 즐긴다."}
    ctx = SimpleNamespace(persona=original)
    runtime.apply(ctx, "A", date(2017, 12, 3))
    assert "smoking_status" not in original
    stage1 = _format_persona(ctx.persona)
    event = SimpleNamespace(time="18:00", anchor="zone:11650101", category="여가", sub_category="당구장", intent="여가")
    candidates = {0: [{"poi_id": "billiard", "name": "실내운동", "known": False}]}
    stage2 = build_stage2_prompt([event], candidates, persona=ctx.persona)
    for text in [stage1, stage2]:
        assert "흡연자 (smoker)" in text
        assert "담배를 피울 수 없다" in text
        assert "13.54" not in text
    assert "billiard: 당구장" in stage2


def test_arm_independent_seed_scopes_and_thread_isolation(tmp_path, monkeypatch):
    on = configured(tmp_path, monkeypatch)
    off = NoSmokingContext(arm="off", **inputs())
    assert on.stable_seed("A", "2018-01-03", "stage1") == off.stable_seed("A", "2018-01-03", "stage1")
    def calls(identity):
        begin_llm_scope(identity, "2018-01-03", "stage1")
        return [next_llm_seed(), next_llm_seed()]
    with ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(calls, ["A", "B", "A"]))
    assert result[0] == result[2]
    assert result[0] != result[1]
    assert result[0][1] == result[0][0] + 1
    assert next_llm_seed() is None


def test_llm_request_receives_paired_seed_only_when_opted_in(tmp_path, monkeypatch):
    from llm_client import call_chat
    requests = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: requests.append(kwargs))))
    call_chat(None, "system", "user", client=client)
    assert "seed" not in requests[-1]
    configured(tmp_path, monkeypatch)
    begin_llm_scope("A", "2018-01-03", "stage1")
    call_chat(None, "system", "user", client=client)
    seed = requests[-1]["seed"]
    monkeypatch.setenv("SIM_NO_SMOKING_ARM", "off")
    begin_llm_scope("A", "2018-01-03", "stage1")
    call_chat(None, "system", "user", client=client)
    assert requests[-1]["seed"] == seed


def test_simulation_seed_can_change_without_reassigning_smoking(tmp_path, monkeypatch):
    first = configured(tmp_path, monkeypatch, simulation_seed=17001)
    second = NoSmokingContext(arm="on", **inputs(), simulation_seed=17002)
    assert first.assignment_seed == second.assignment_seed == 45
    assert first.people == second.people
    assert first.stable_seed("A", "2018-01-03", "stage1") != second.stable_seed("A", "2018-01-03", "stage1")
