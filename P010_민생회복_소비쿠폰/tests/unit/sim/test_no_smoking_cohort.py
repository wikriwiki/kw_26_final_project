import argparse
import copy

import pytest

from scripts.experiments import no_smoking_zone as experiment
from scripts.experiments import prepare_no_smoking_cohort as graph_cohort


def original(aid="a"):
    return {"agent_id": aid, "age": 52, "personal": {"age": 52, "gender": "F", "income_level": "중하"},
            "spending": {"daily_spending_weekday": 9000, "daily_spending_weekend": 14000,
                         "weekday_spending_level": 1, "weekend_spending_level": 2},
            "personality": {"lifestyle": "original provenance"}}


def graph_row(aid="a", **properties):
    return {"a": {"id": aid, "personal_age": 52, "personal_gender": "F",
                  "personal_income_level": "중하", "s_daily_wd": 31085, "s_daily_we": 45646,
                  "spending_level_wd": 3, "spending_level_we": 4, **properties}, "home_ok": True}


def test_missing_graph_age_stays_unknown_despite_older_top_level_and_nested_age():
    source, graph = [original()], [graph_row(personal_age=None)]
    source_before, graph_before = copy.deepcopy(source), copy.deepcopy(graph)
    personas, eligible, audit = graph_cohort.prepare(graph, source, "a" * 64)
    assert personas[0]["age"] == 52  # Kept as source provenance, never used for assignment.
    assert personas[0]["personal"]["age"] is None
    cohort, _ = experiment.assign_smoking(personas, experiment.read_json(experiment.DATA / "smoking_rates.json"), 20171203)
    assert cohort[0]["age"] is None and cohort[0]["smoking_status"] == "unknown"
    assert eligible == ["a"] and audit["fields_reconciled_from_graph"]["age"] == 1
    assert source == source_before and graph == graph_before


def test_calibrated_graph_anchors_override_archive_and_raw_graph_anchors():
    graph = [graph_row(s_daily_wd_raw=9000, s_daily_we_raw=14000)]
    personas, _, _ = graph_cohort.prepare(graph, [original()], "a" * 64)
    spending = personas[0]["spending"]
    assert spending["daily_spending_weekday"] == 31085
    assert spending["daily_spending_weekend"] == 45646
    assert spending["weekday_spending_level"] == 3
    assert spending["weekend_spending_level"] == 4
    assert personas[0]["source_graph_properties"]["s_daily_wd_raw"] == 9000
    assert personas[0]["graph_source_sha256"] == "a" * 64


def test_runtime_income_uses_reconciled_graph_flat_field_and_preserves_original():
    graph = [graph_row(personal_income_level="중하", p_income_level="하")]
    personas, _, _ = graph_cohort.prepare(graph, [original()], "a" * 64)
    assert personas[0]["personal"]["income_level"] == "하"
    assert personas[0]["source_graph_properties"]["personal_income_level"] == "중하"


@pytest.mark.parametrize("target", ["original", "graph"])
def test_duplicate_graph_or_original_id_is_rejected(target):
    raw, graph = [original()], [graph_row()]
    if target == "original":
        raw += copy.deepcopy(raw)
    else:
        graph += copy.deepcopy(graph)
    with pytest.raises(ValueError, match="Duplicate"):
        graph_cohort.prepare(graph, raw, "a" * 64)


def test_home_and_spending_eligibility_preserves_full_assignment_population():
    graph = [graph_row("valid"), graph_row("no_home"), graph_row("no_anchor", s_daily_wd=None, s_daily_we=0)]
    graph[1]["home_ok"] = False
    personas, eligible, audit = graph_cohort.prepare(graph, [original(r["a"]["id"]) for r in graph], "a" * 64)
    assert len(personas) == 3 and eligible == ["valid"]
    assert audit["source_population_size"] == 3 and audit["eligible_count"] == 1
    assert {r["id"] for r in audit["excluded"]} == {"no_home", "no_anchor"}
    assert audit["smoking_assignment_before_eligibility_filter"] is True


def prepare_args(tmp_path, eligible, *, limit=None, out="bundle"):
    agents = [{"agent_id": f"p{i:03}", "personal": {"age": 35, "gender": "M"}} for i in range(100)]
    source_path, eligible_path = tmp_path / "source.json", tmp_path / "eligible.json"
    experiment.write_json(source_path, agents)
    experiment.write_json(eligible_path, eligible)
    return argparse.Namespace(agents=source_path, eligible_ids=eligible_path, limit=limit,
                              pois=None, out=tmp_path / out, seed=20171203, simulation_seed=17001), agents


@pytest.mark.parametrize("eligible", [["absent"], ["p000", "p000"], [], [None], {"id": "p000"}])
def test_invalid_or_unknown_eligible_ids_fail_before_creating_bundle(tmp_path, eligible):
    args, _ = prepare_args(tmp_path, eligible)
    with pytest.raises(ValueError, match="eligible IDs"):
        experiment.prepare(args)
    assert not args.out.exists()


def test_eligible_filter_keeps_labels_from_full_population_without_reallocation(tmp_path):
    eligible = [f"p{i:03}" for i in range(0, 100, 3)]
    args, agents = prepare_args(tmp_path, list(reversed(eligible)))
    full, allocation = experiment.assign_smoking(agents, experiment.read_json(experiment.DATA / "smoking_rates.json"), args.seed)
    experiment.prepare(args)
    result = experiment.read_json(args.out / "runtime.json")["cohort"]
    assert result == [r for r in full if r["id"] in set(eligible)]
    audit = experiment.read_json(args.out / "assignment_audit.json")
    assert audit["source_population_size"] == 100
    assert audit["eligible_population_size"] == len(eligible)
    assert audit["strata_full_population"] == allocation
    assert audit["excluded_ineligible_count"] == 100 - len(eligible)
    bundle = experiment.read_json(args.out / "bundle.json")
    assert bundle["eligibility_source_sha256"] == experiment.file_hash(args.eligible_ids)


def test_pilot_samples_only_eligible_ids_and_preserves_full_labels(tmp_path):
    eligible = [f"p{i:03}" for i in range(40, 50)]
    args, agents = prepare_args(tmp_path, eligible, limit=5)
    full, _ = experiment.assign_smoking(agents, experiment.read_json(experiment.DATA / "smoking_rates.json"), args.seed)
    experiment.prepare(args)
    result = experiment.read_json(args.out / "runtime.json")["cohort"]
    assert len(result) == 5
    assert {r["id"] for r in result} <= set(eligible)
    assert all(r in full for r in result)
    args2, _ = prepare_args(tmp_path, list(reversed(eligible)), limit=5, out="repeat")
    experiment.prepare(args2)
    assert result == experiment.read_json(args2.out / "runtime.json")["cohort"]


def test_pilot_limit_cannot_exceed_eligible_population(tmp_path):
    args, _ = prepare_args(tmp_path, ["p000", "p001"], limit=3)
    with pytest.raises(ValueError, match="eligible population"):
        experiment.prepare(args)
    assert not args.out.exists()


def with_experiment_ids(args, ids):
    args.cohort_ids = args.agents.parent / "selected-experiment.json"
    experiment.write_json(args.cohort_ids, ids)
    return args


@pytest.mark.parametrize("ids", [[], ["p000", "p000"], ["unknown"], ["p050"], [None], {"id": "p000"}])
def test_explicit_experiment_ids_must_be_unique_and_eligible(tmp_path, ids):
    args, _ = prepare_args(tmp_path, [f"p{i:03}" for i in range(40)])
    with_experiment_ids(args, ids)
    with pytest.raises(ValueError, match="cohort IDs"):
        experiment.prepare(args)
    assert not args.out.exists()


def test_explicit_experiment_roster_preserves_eligibility_and_full_population_labels(tmp_path):
    eligible = [f"p{i:03}" for i in range(80)]
    selected = [f"p{i:03}" for i in range(0, 80, 2)]
    args, agents = prepare_args(tmp_path, eligible)
    with_experiment_ids(args, list(reversed(selected)))
    full, _ = experiment.assign_smoking(agents, experiment.read_json(experiment.DATA / "smoking_rates.json"), args.seed)
    experiment.prepare(args)
    actual = experiment.read_json(args.out / "runtime.json")["cohort"]
    assert actual == [r for r in full if r["id"] in selected]
    assert experiment.read_json(args.out / "experiment_cohort_ids.json") == selected
    metadata = experiment.read_json(args.out / "bundle.json")
    audit = experiment.read_json(args.out / "assignment_audit.json")
    for record in (metadata, audit):
        assert record["source_population_size"] == 100
        assert record["eligible_population_size"] == 80
        assert record["experiment_cohort_size"] == 40
        assert record["cohort_source_sha256"] == experiment.file_hash(args.cohort_ids)
        assert record["experiment_cohort_ids_sha256"] == experiment.digest(selected)
        assert record["cohort_selection"] == "explicit_ids"
    assert audit["excluded_ineligible_count"] == 20
    assert audit["selected_cohort_size"] == 40
    assert not any("selection" in b for b in experiment.inspect_bundle(args.out)["blockers"])


def test_pilot_is_within_selected_experiment_and_cannot_exceed_it(tmp_path):
    eligible = [f"p{i:03}" for i in range(80)]
    selected = [f"p{i:03}" for i in range(30, 40)]
    args, _ = prepare_args(tmp_path, eligible, limit=5)
    with_experiment_ids(args, selected)
    experiment.prepare(args)
    runtime_ids = {p["id"] for p in experiment.read_json(args.out / "runtime.json")["cohort"]}
    assert len(runtime_ids) == 5 and runtime_ids <= set(selected)
    metadata = experiment.read_json(args.out / "bundle.json")
    assert metadata["experiment_cohort_size"] == 10 and metadata["pilot_limit"] == 5
    assert not any("selection" in b for b in experiment.inspect_bundle(args.out)["blockers"])
    args.out = tmp_path / "invalid-pilot"
    args.limit = 11
    with pytest.raises(ValueError, match="selected experiment cohort"):
        experiment.prepare(args)


@pytest.mark.parametrize("tamper", ["missing_ids", "changed_ids", "wrong_size", "audit_disagrees", "missing_provenance", "wrong_pilot"])
def test_preflight_detects_broken_frozen_experiment_selection(tmp_path, tamper):
    args, _ = prepare_args(tmp_path, [f"p{i:03}" for i in range(80)], limit=5)
    with_experiment_ids(args, [f"p{i:03}" for i in range(30)])
    experiment.prepare(args)
    if tamper == "missing_ids":
        (args.out / "experiment_cohort_ids.json").unlink()
    elif tamper == "changed_ids":
        experiment.write_json(args.out / "experiment_cohort_ids.json", ["p099"])
    elif tamper in {"wrong_size", "missing_provenance"}:
        path = args.out / "bundle.json"
        metadata = experiment.read_json(path)
        metadata["experiment_cohort_size" if tamper == "wrong_size" else "cohort_source_sha256"] = 99 if tamper == "wrong_size" else None
        experiment.write_json(path, metadata)
    elif tamper == "audit_disagrees":
        path = args.out / "assignment_audit.json"
        audit = experiment.read_json(path)
        audit["experiment_cohort_size"] = 29
        experiment.write_json(path, audit)
    else:
        path = args.out / "runtime.json"
        runtime = experiment.read_json(path)
        old_ids = {r["id"] for r in runtime["cohort"]}
        runtime["cohort"][0]["id"] = next(f"p{i:03}" for i in range(30) if f"p{i:03}" not in old_ids)
        experiment.write_json(path, runtime)
        metadata = experiment.read_json(args.out / "bundle.json")
        metadata["runtime_sha256"] = experiment.digest(runtime)
        experiment.write_json(args.out / "bundle.json", metadata)
    assert any("Invalid experiment cohort selection" in b for b in experiment.inspect_bundle(args.out)["blockers"])


def test_legacy_bundle_without_separate_experiment_roster_remains_readable(tmp_path):
    args, _ = prepare_args(tmp_path, ["p000", "p001"])
    experiment.prepare(args)
    path = args.out / "bundle.json"
    metadata = experiment.read_json(path)
    for key in ("experiment_cohort_ids_sha256", "experiment_cohort_size", "cohort_source_sha256", "cohort_selection", "pilot_limit"):
        metadata.pop(key)
    experiment.write_json(path, metadata)
    (args.out / "experiment_cohort_ids.json").unlink()
    assert not any("selection" in b for b in experiment.inspect_bundle(args.out)["blockers"])
