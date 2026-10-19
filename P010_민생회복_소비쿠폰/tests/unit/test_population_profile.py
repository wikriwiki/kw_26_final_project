"""Synthetic income profiles test opt-in runtime wiring without graph/models."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("population_profile_tests", ROOT / "scripts/sim/population_profile.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    MODULE._CACHE.clear()
    MODULE._CONFIG_FROZEN = False
    MODULE._CONFIG_SELECTION = None
    monkeypatch.delenv("POPULATION_PROFILE_FILE", raising=False)
    monkeypatch.delenv("POPULATION_PROFILE_EXPECTED_SHA", raising=False)
    evidence = tmp_path / "synthetic_income_mapping.txt"
    evidence.write_text("synthetic only, observed bands A<B", encoding="utf8")
    source = {"path": str(evidence), "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}
    row = {"aid": "synthetic_person", "sex": "F", "age": 30, "home_dong_code": "synthetic_home",
           "income_band": "A", "income_tier": "하", "household_income_definition": "월 가구소득, 세전, 합성 할당"}
    data = {"schema": "frozen_population_profile_v1", "population_unit": "resident_person", "reference_year": 2020,
            "assignment_kind": "calibrated_synthetic_income_assignment", "policy_outcome_used_for_assignment": False,
            "source_evidence": [source], "household_income_definition": row["household_income_definition"],
            "admin_dong_code_system": "synthetic_admin_2020", "official_admin_crosswalk_verified": True,
            "mapping_definition": {"method": "observed_income_band_mapping", "income_band_order": ["A", "B"],
                                   "band_to_tier": {"A": "하", "B": "상"}, "policy_outcome_used": False,
                                   "observed_band_direct_mapping": True, "source_evidence": [source]}, "rows": [row]}
    path = tmp_path / "profile.json"
    def freeze():
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf8")
        monkeypatch.setenv("POPULATION_PROFILE_FILE", str(path))
        monkeypatch.setenv("POPULATION_PROFILE_EXPECTED_SHA", hashlib.sha256(path.read_bytes()).hexdigest())
        MODULE._CACHE.clear()
        MODULE._CONFIG_FROZEN = False
        MODULE._CONFIG_SELECTION = None
    return data, path, freeze


def persona():
    return {"id": "synthetic_person", "gender": "F", "home_dong_code": "synthetic_home", "income": "상",
            "age_group": "30대", "job": "직장인", "daily_wd": 100, "daily_we": 100}


def test_default_preserves_legacy_income_and_no_file_access(fixture):
    p = persona()
    assert MODULE.bind_persona("synthetic_person", p, 30) is p
    assert MODULE.income_for_eligibility(p) == "상"
    assert MODULE.profile_roster() is None and MODULE.preflight_profile_roster(["a"]) is None


def test_persona_policy_income_same_frozen_profile(fixture):
    _, _, freeze = fixture
    freeze()
    original = persona()
    assigned = MODULE.bind_persona("synthetic_person", original, 30)
    assert original["income"] == "상" and assigned["income"] == "하"
    assert MODULE.income_for_eligibility(assigned) == "하"
    assert assigned["assigned_income_band"] == "A"
    assert MODULE.preflight_profile_roster(["synthetic_person"])["citizens"] == 1


@pytest.mark.parametrize("field,value", [("gender", "M"), ("home_dong_code", "id_only_wrong_dong")])
def test_graph_identity_or_anchor_mismatch_blocks(fixture, field, value):
    _, _, freeze = fixture; freeze()
    p = persona(); p[field] = value
    with pytest.raises(MODULE.PopulationProfileError, match="disagrees"):
        MODULE.bind_persona("synthetic_person", p, 30)


def test_exact_age_required(fixture):
    _, _, freeze = fixture; freeze()
    with pytest.raises(MODULE.PopulationProfileError, match="disagrees"):
        MODULE.bind_persona("synthetic_person", persona(), None)


def test_whole_roster_identity_gate_before_workers(fixture):
    _, _, freeze = fixture; freeze()
    rows = [{"aid": "synthetic_person", "sex": "F", "age": 30, "home_dong_code": "synthetic_home"}]
    assert len(MODULE.verify_graph_projection(rows)) == 64
    rows[0]["age"] = 31
    with pytest.raises(MODULE.PopulationProfileError, match="whole-roster"):
        MODULE.verify_graph_projection(rows)


def test_graph_duplicate_or_missing_home_is_not_silent(fixture):
    _, _, freeze = fixture; freeze()
    with pytest.raises(MODULE.PopulationProfileError, match="missing/duplicating"):
        MODULE.verify_graph_projection([])


def test_unbound_old_llm_income_cannot_pass_policy_gate(fixture):
    _, _, freeze = fixture; freeze()
    with pytest.raises(MODULE.PopulationProfileError, match="frozen persona"):
        MODULE.income_for_eligibility(persona())


def test_roster_difference_blocks(fixture):
    _, _, freeze = fixture; freeze()
    with pytest.raises(MODULE.PopulationProfileError, match="roster"):
        MODULE.preflight_profile_roster(["other_person"])


def test_declared_profile_sha_mismatch_blocks(fixture, monkeypatch):
    _, _, freeze = fixture; freeze()
    monkeypatch.setenv("POPULATION_PROFILE_EXPECTED_SHA", "0" * 64)
    with pytest.raises(MODULE.PopulationProfileError, match="SHA256"):
        MODULE.active_profile()


def test_partial_profile_env_fails_closed(fixture, monkeypatch):
    monkeypatch.setenv("POPULATION_PROFILE_FILE", "not_verified.json")
    with pytest.raises(MODULE.PopulationProfileError, match="both"):
        MODULE.profile_enabled()


def test_profile_cannot_be_disabled_mid_execution(fixture, monkeypatch):
    _, _, freeze = fixture; freeze(); MODULE.active_profile()
    monkeypatch.delenv("POPULATION_PROFILE_FILE")
    monkeypatch.delenv("POPULATION_PROFILE_EXPECTED_SHA")
    with pytest.raises(MODULE.PopulationProfileError, match="configuration changed"):
        MODULE.active_profile()


def test_frozen_file_changed_after_cache_blocks(fixture):
    _, path, freeze = fixture; freeze(); MODULE.active_profile()
    path.write_text("{}", encoding="utf8")
    with pytest.raises(MODULE.PopulationProfileError, match="changed"):
        MODULE.active_profile()


def test_source_sha_mismatch_blocks(fixture):
    data, _, freeze = fixture
    data["source_evidence"][0]["sha256"] = "0" * 64
    freeze()
    with pytest.raises(MODULE.PopulationProfileError, match="source SHA"):
        MODULE.active_profile()


def test_unknown_income_refused_is_not_eligibility(fixture):
    data, _, freeze = fixture
    data["mapping_definition"]["income_band_order"][0] = "무응답"
    data["mapping_definition"]["band_to_tier"] = {"무응답": "하", "B": "상"}
    data["rows"][0]["income_band"] = "무응답"; freeze()
    with pytest.raises(MODULE.PopulationProfileError, match="unknown/refused"):
        MODULE.active_profile()


def test_income_rank_reversal_blocks(fixture):
    data, _, freeze = fixture
    data["mapping_definition"]["band_to_tier"] = {"A": "상", "B": "하"}; freeze()
    with pytest.raises(MODULE.PopulationProfileError, match="reverse"):
        MODULE.active_profile()


def test_income_tier_cannot_be_implicit_quintile(fixture):
    data, _, freeze = fixture
    data["mapping_definition"]["observed_band_direct_mapping"] = False; freeze()
    with pytest.raises(MODULE.PopulationProfileError, match="quintiles"):
        MODULE.active_profile()


def test_household_definition_mismatch_blocks(fixture):
    data, _, freeze = fixture
    data["rows"][0]["household_income_definition"] = "개인소득 다른 정의"; freeze()
    with pytest.raises(MODULE.PopulationProfileError, match="definitions differ"):
        MODULE.active_profile()


def test_only_individual_fields_enter_persona(fixture):
    data, _, freeze = fixture
    data["targets"] = {"policy_effect": 9999};data["empirical_policy_effect"] = "do not inject"
    freeze()
    assigned = MODULE.bind_persona("synthetic_person", persona(), 30)
    assert "targets" not in assigned and "empirical_policy_effect" not in assigned
    assert "source_evidence" not in assigned and "mapping_definition" not in assigned


def test_dawn_income_render_and_policy_grant_match(fixture, monkeypatch):
    _, _, freeze = fixture; freeze()
    assigned = MODULE.bind_persona("synthetic_person", persona(), 30)
    # Existing helpers run on the bound persona without any graph/model call.
    from scripts.sim.dawn_context import _format_persona
    from scripts.sim.plan_writer import _grant_for_single_policy
    monkeypatch.setenv("EXP_DURABLES", "0")
    text = _format_persona(assigned)
    assert "소득: 하" in text and "개인에게 배정된 소득구간: A" in text
    assert "합성 할당" in text and "9999" not in text
    grant = _grant_for_single_policy(MODULE.income_for_eligibility(assigned), {"type": "grant", "income_grants": {"하": 123, "상": 456}})
    assert grant == 123


def test_dawn_builder_binds_profile_before_prompt(fixture, monkeypatch):
    from contextlib import contextmanager
    from datetime import date
    import population_profile as runtime_profile
    from scripts.sim import dawn_context
    _, _, freeze = fixture; freeze()
    monkeypatch.setattr(runtime_profile, "_CONFIG_FROZEN", False)
    monkeypatch.setattr(runtime_profile, "_CONFIG_SELECTION", None)
    monkeypatch.setattr(runtime_profile, "_CACHE", {})
    class Rows:
        def __init__(self, values): self.values = values
        def single(self): return self.values[0] if self.values else None
        def __iter__(self): return iter(self.values)
    class Session:
        def run(self, query, **params):
            if query == dawn_context.PERSONA_CYPHER: return Rows([persona()])
            if "personal_age AS age" in query: return Rows([{"age": 30}])
            return Rows([])
    @contextmanager
    def driver(): yield Session()
    monkeypatch.setattr(dawn_context, "driver_session", driver)
    monkeypatch.setattr(dawn_context, "_persona_from_cache", lambda aid: None)
    monkeypatch.setattr(dawn_context, "_cache_persona", lambda aid, p: None)
    monkeypatch.setattr(dawn_context, "_policy_from_cache", lambda today, p: [])
    monkeypatch.setattr(dawn_context, "_build_zone_candidates", lambda p, today: [])
    monkeypatch.setenv("EXP_DURABLES", "0")
    ctx = dawn_context.build_dawn_context("synthetic_person", date(2020, 6, 2))
    assert ctx.persona["income"] == "하"
    assert runtime_profile.income_for_eligibility(ctx.persona) == "하"
    assert "개인에게 배정된 소득구간: A" in dawn_context._format_persona(ctx.persona)


def test_run_day_identity_fails_before_any_model_worker(fixture, monkeypatch):
    from contextlib import contextmanager
    from datetime import date
    import population_profile as runtime_profile
    from scripts.sim import run_simulation
    _, _, freeze = fixture; freeze()
    monkeypatch.setattr(runtime_profile, "_CONFIG_FROZEN", False)
    monkeypatch.setattr(runtime_profile, "_CONFIG_SELECTION", None)
    monkeypatch.setattr(runtime_profile, "_CACHE", {})
    class Session:
        def run(self, query, **params):
            return [{"aid": "synthetic_person", "sex": "F", "age": 31, "home_dong_code": "synthetic_home"}]
    @contextmanager
    def driver(): yield Session()
    monkeypatch.setattr(run_simulation, "driver_session", driver)
    monkeypatch.setattr(run_simulation, "active_prompt_name", lambda: run_simulation._ACTIVE_PROMPT_VARIANT)
    monkeypatch.setattr(run_simulation, "call_stage1", lambda *a, **k: pytest.fail("model called before identity preflight"))
    with pytest.raises(runtime_profile.PopulationProfileError, match="whole-roster"):
        run_simulation.run_day(["synthetic_person"], date(2020, 6, 2), 0, workers=1)
