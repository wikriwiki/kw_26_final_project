import argparse
import copy
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from scripts.experiments import no_smoking_zone as exp
from scripts.sim.evidence_integrity import seal


def rates():
    return exp.read_json(exp.DATA / "smoking_rates.json")


@pytest.mark.parametrize("engine", ["sglang", "vllm"])
def test_server_record_preserves_explicit_engine_and_version(engine):
    config = {"engine": engine, engine + "_version": "0.5.test", "argv": ["python", "serve"]}
    actual = exp.validate_server_config(config)
    assert actual["engine"] == engine and actual["engine_version"] == "0.5.test"


def test_server_record_rejects_missing_sglang_version_and_ambiguous_engine():
    with pytest.raises(ValueError, match="Incomplete"):
        exp.validate_server_config({"engine": "sglang", "vllm_version": "0.5", "argv": ["serve"]})
    with pytest.raises(ValueError, match="explicit"):
        exp.validate_server_config({"sglang_version": "0.5", "vllm_version": "0.5", "argv": ["serve"]})


def people(n=100):
    return [{"agent_id": f"p{i:03}", "personal": {"age": 35, "gender": "M"}} for i in range(n)]


def test_real_seoul_quota_is_reproducible_and_order_independent():
    a, audit = exp.assign_smoking(people(), rates(), 20171203)
    b, _ = exp.assign_smoking(list(reversed(people())), rates(), 20171203)
    assert a == b
    assert sum(p["smoking_status"] == "smoker" for p in a) == 42
    assert audit[0]["rate"] == 0.418
    assert rates()["source"]["primary_source_verified"] is True


def test_minors_missing_age_and_unknown_sex_have_explicit_handling():
    rows = [
        {"id": "minor", "age": 16, "sex": "male"},
        {"id": "missing", "sex": "female"},
        *[{"id": f"unknown{i}", "age": 35} for i in range(100)],
    ]
    result, audit = exp.assign_smoking(rows, rates(), 3)
    assert {r["id"] for r in result if r["smoking_status"] == "unknown"} == {"minor", "missing"}
    assert sum(r["smoking_status"] == "smoker" for r in result) == 23
    assert audit[0]["rate"] == 0.228


def test_duplicate_agent_is_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        exp.assign_smoking([people()[0], people()[0]], rates(), 1)


@pytest.mark.parametrize("changes", [{"facility_type": "leisure"}, {"district_code": "11680"}, {"classification_source": ""}])
def test_unverified_or_wrong_scope_pois_fail(changes):
    row = {"poi_id": "p", "district_code": "11650", "facility_type": "billiard", "classification_source": "fixture reviewed"}
    with pytest.raises(ValueError):
        exp.normalize_pois([row | changes])


def test_prepare_freezes_labels_and_stops_for_missing_data(tmp_path):
    agents = tmp_path / "agents.json"
    exp.write_json(agents, people())
    args = argparse.Namespace(agents=agents, pois=None, seed=20171203, simulation_seed=17001, limit=10, out=tmp_path / "bundle")
    result = exp.prepare(args)
    assert result["agents"] == 10 and len(result["blockers"]) == 6
    runtime = exp.read_json(args.out / "runtime.json")
    assert set(runtime) == {"schema_version", "experiment_id", "assignment_seed", "simulation_seed", "cohort", "pois"}
    full, _ = exp.assign_smoking(people(), rates(), 20171203)
    assert all(p in full for p in runtime["cohort"])
    runtime["cohort"][0]["smoking_status"] = "unknown"
    exp.write_json(args.out / "runtime.json", runtime)
    assert "Frozen runtime.json hash mismatch" in exp.inspect_bundle(args.out)["blockers"]
    with pytest.raises(ValueError, match="overwrite"):
        exp.prepare(args)


def create_run(folder, arm, revenue=10000, count=1):
    manifest = {"arm": arm, "status": "complete", "runtime_sha256": "same", "runtime_file_sha256": "bytes",
                "snapshot_sha256": "baseline", "start": "2017-12-03", "days": 1, "cohort_ids": ["a"],
                "assignment_seed": 20171203, "simulation_seed": 17001, "model": "fixed", "code_sha256": {},
                "workers": 4, "engine_settings": {}, "reference_sha256": {}, "server_config": {}}
    exp.write_json(folder / "experiment_run.json", manifest)
    row = {"aid": "a", "status": "ok", "experience_day": "2017-12-03", "fb_cand_l1_dong": 1,
           "no_smoking": {"arm": arm, "policy_active": arm == "on", "manifest_sha256": "bytes", "smoking_status": "smoker",
           "by_poi": [{"poi_id": "poi", "district_code": "11650", "facility_type": "billiard", "payment_count": count, "revenue_krw": revenue}]}}
    path = folder / "metrics/day_2017-12-03.jsonl"
    path.parent.mkdir()
    path.write_text(json.dumps(seal(row)) + "\n", encoding="utf-8")
    return row


def test_score_reports_payments_and_never_claims_empirical_validation(tmp_path):
    create_run(tmp_path / "off", "off")
    create_run(tmp_path / "on", "on", revenue=12000)
    report = exp.score_runs(tmp_path / "off", tmp_path / "on", {"metrics": []})
    revenue = next(x for x in report["comparisons"] if x["group"] == "billiard" and x["metric"] == "revenue_krw")
    assert revenue["percent_change"] == 20
    assert report["empirical_validation_pass"] is None
    golf = next(x for x in report["comparisons"] if x["group"] == "indoor_golf")
    assert golf["percent_change"] is None and golf["status"] == "zero_baseline_not_estimable"


@pytest.mark.parametrize("failure", ["missing", "duplicate", "foreign_manifest", "wrong_day", "changed_seed", "failed", "corrupt"])
def test_score_rejects_broken_pairing_or_incomplete_runs(tmp_path, failure):
    create_run(tmp_path / "off", "off")
    row = create_run(tmp_path / "on", "on")
    path = tmp_path / "on/metrics/day_2017-12-03.jsonl"
    if failure == "missing":
        path.write_text("")
    elif failure == "duplicate":
        path.write_text(path.read_text() * 2)
    elif failure in {"foreign_manifest", "wrong_day"}:
        if failure == "foreign_manifest": row["no_smoking"]["manifest_sha256"] = "different"
        else: row["experience_day"] = "2017-12-04"
        path.write_text(json.dumps(seal(row)) + "\n")
    elif failure == "corrupt":
        corrupted = json.loads(path.read_text())
        corrupted["no_smoking"]["by_poi"][0]["revenue_krw"] += 1
        path.write_text(json.dumps(corrupted) + "\n")
    else:
        manifest_path = tmp_path / "on/experiment_run.json"
        manifest = exp.read_json(manifest_path)
        manifest["simulation_seed" if failure == "changed_seed" else "status"] = 99 if failure == "changed_seed" else "failed"
        exp.write_json(manifest_path, manifest)
    with pytest.raises(ValueError):
        exp.score_runs(tmp_path / "off", tmp_path / "on", {"metrics": []})


def test_default_window_is_fourteen_days_each_side_of_legal_onset():
    from scripts.experiments.rebuild_no_smoking_baseline import DEFAULT_DAY_ZERO
    assert exp.DEFAULT_START == "2017-11-19"
    assert exp.DEFAULT_END == "2017-12-16"
    assert exp.DEFAULT_DAYS == 28
    assert DEFAULT_DAY_ZERO == exp.DEFAULT_DAY_ZERO == "2017-11-18"
    window = exp.simulation_window(exp.DEFAULT_START, exp.DEFAULT_DAYS)
    assert window["pre"] == {"start": "2017-11-19", "end": "2017-12-02", "days": 14}
    assert window["post"] == {"start": "2017-12-03", "end": "2017-12-16", "days": 14}
    assert window["matches_default_14_pre_14_post"] is True
    policy = exp.read_json(exp.DATA / "policy.json")
    assert window["policy_effective_date"] == policy["policy"]["effective_date"] == "2017-12-03"
    assert policy["experiment_scope"]["report_policy_indicator"]["post_start"] == "2018-01-01"


@pytest.mark.parametrize("start,days,pre,post", [
    ("2017-12-02", 2, 1, 1), ("2017-12-03", 1, 0, 1),
    ("2017-11-19", 2, 2, 0), ("2018-01-01", 2, 0, 2),
])
def test_pilot_window_counts_inclusive_legal_boundary(start, days, pre, post):
    window = exp.simulation_window(start, days)
    assert (window["pre"]["days"], window["post"]["days"]) == (pre, post)
    assert pre + post == window["days"]


@pytest.mark.parametrize("days", [0, -1, True, 1.5])
def test_window_rejects_invalid_duration(days):
    with pytest.raises(ValueError, match="positive integer"):
        exp.simulation_window(exp.DEFAULT_START, days)


def test_run_cli_uses_new_window_defaults(monkeypatch, tmp_path):
    called = {}
    monkeypatch.setattr(exp, "run", lambda args: called.update(start=args.start, days=args.days) or {"status": "fixture"})
    assert exp.main(["run", "--bundle", str(tmp_path), "--arm", "off", "--out", str(tmp_path / "out")]) == 0
    assert called == {"start": "2017-11-19", "days": 28}


def create_window_run(folder, arm):
    template = create_run(folder, arm)
    (folder / "metrics/day_2017-12-03.jsonl").unlink()
    manifest = exp.read_json(folder / "experiment_run.json")
    manifest.update(start=exp.DEFAULT_START, days=exp.DEFAULT_DAYS,
                    simulation_window=exp.simulation_window(exp.DEFAULT_START, exp.DEFAULT_DAYS))
    exp.write_json(folder / "experiment_run.json", manifest)
    for index in range(28):
        day = (date(2017, 11, 19) + timedelta(days=index)).isoformat()
        row = copy.deepcopy(template)
        row["experience_day"] = day
        row["no_smoking"]["policy_active"] = arm == "on" and day >= "2017-12-03"
        # Large, unequal pre totals must never affect the post-period contrast.
        revenue = 1000000 if arm == "off" else 9000000
        if day >= "2017-12-03":
            revenue = {"2017-12-03": 101, "2017-12-16": 103}.get(day, 1) * (1 if arm == "off" else 2)
        row["no_smoking"]["by_poi"][0]["revenue_krw"] = revenue
        path = folder / f"metrics/day_{day}.jsonl"
        path.write_text(json.dumps(seal(row)) + "\n", encoding="utf-8")


def test_score_reports_both_fourteen_day_periods_but_compares_only_dec3_to_dec16(tmp_path):
    for arm in ("off", "on"):
        create_window_run(tmp_path / arm, arm)
    report = exp.score_runs(tmp_path / "off", tmp_path / "on", {"metrics": []})
    assert report["comparison_period"] == "post" and report["paired_agent_days"] == 28
    assert report["simulation_window"]["pre"]["days"] == report["simulation_window"]["post"]["days"] == 14
    assert report["period_totals"]["pre"]["off"]["billiard"] == {"revenue_krw": 14000000, "payment_count": 14}
    assert report["period_totals"]["pre"]["on"]["billiard"]["revenue_krw"] == 126000000
    assert report["period_totals"]["post"]["off"]["billiard"] == {"revenue_krw": 216, "payment_count": 14}
    revenue = next(c for c in report["comparisons"] if c["group"] == "billiard" and c["metric"] == "revenue_krw")
    assert revenue["off"] == 216 and revenue["on"] == 432 and revenue["percent_change"] == 100
    assert report["empirical_validation_pass"] is None


def test_score_excludes_both_arms_for_a_skipped_post_agent_day(tmp_path):
    for arm in ("off", "on"):
        create_window_run(tmp_path / arm, arm)
    path = tmp_path / "on/metrics/day_2017-12-03.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    row.update(status="skipped", skip_kind="failed_after_retries", attempts=6,
               observed_behavior=False, last_error="model validation")
    row["no_smoking"] = {k: row["no_smoking"][k] for k in
                         ("arm", "policy_active", "manifest_sha256")}
    row["no_smoking"]["observed_behavior"] = False
    path.write_text(json.dumps(seal(row)) + "\n", encoding="utf-8")
    report = exp.score_runs(tmp_path / "off", tmp_path / "on", {"metrics": []})
    assert report["observed_paired_post_agent_days"] == 13
    assert report["excluded_post_agent_days"] == 1
    assert report["agent_day_status_counts"]["on"]["skipped"] == 1
    assert report["period_totals"]["post"]["off"]["billiard"]["revenue_krw"] == 115
    assert report["period_totals"]["post"]["on"]["billiard"]["revenue_krw"] == 230


@pytest.mark.parametrize("day,active", [("2017-12-02", True), ("2017-12-03", False)])
def test_score_rejects_activation_shifted_across_december3(tmp_path, day, active):
    create_window_run(tmp_path / "on", "on")
    path = tmp_path / "on/metrics" / f"day_{day}.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["no_smoking"]["policy_active"] = active
    path.write_text(json.dumps(seal(row)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="activation"):
        exp.load_run(tmp_path / "on", "on")


def test_load_rejects_manifest_window_that_disagrees_with_actual_dates(tmp_path):
    create_window_run(tmp_path / "on", "on")
    path = tmp_path / "on/experiment_run.json"
    manifest = exp.read_json(path)
    manifest["simulation_window"]["post"]["days"] = 13
    exp.write_json(path, manifest)
    with pytest.raises(ValueError, match="window boundaries"):
        exp.load_run(tmp_path / "on", "on")


def shared_phase_fixture(tmp_path):
    create_window_run(tmp_path / "full_off", "off")
    create_window_run(tmp_path / "full_on", "on")
    run_id = str((tmp_path / "pre").resolve())
    dump = tmp_path / "dec2.dump"
    dump.write_bytes(b"fixture offline graph dump")
    for phase, source in (("pre", "full_off"), ("off", "full_off"), ("on", "full_on")):
        folder = tmp_path / phase
        (folder / "metrics").mkdir(parents=True)
        manifest = exp.read_json(tmp_path / source / "experiment_run.json")
        manifest.update(phase="shared_pre" if phase == "pre" else "post_branch",
                        start=exp.DEFAULT_START if phase == "pre" else "2017-12-03",
                        days=14, run_id=run_id,
                        snapshot_sha256="day_zero" if phase == "pre" else exp.file_hash(dump))
        manifest["prompt_contract"] = {}
        manifest["simulation_window"] = exp.simulation_window(manifest["start"], manifest["days"])
        exp.write_json(folder / "experiment_run.json", manifest)
        for i in range(14):
            day = (date(2017, 11, 19) + timedelta(days=i if phase == "pre" else i + 14)).isoformat()
            source_path = tmp_path / source / "metrics" / f"day_{day}.jsonl"
            (folder / "metrics" / source_path.name).write_bytes(source_path.read_bytes())
    branch_path = tmp_path / "branch.json"
    branch = exp.record_branch(tmp_path / "pre", dump, branch_path)
    for phase in ("off", "on"):
        path = tmp_path / phase / "experiment_run.json"
        manifest = exp.read_json(path)
        manifest["branch"] = {"record_sha256": exp.file_hash(branch_path),
                              "snapshot_sha256": branch["snapshot_sha256"],
                              "source_manifest_sha256": branch["source_manifest_sha256"],
                              "source_run_id": branch["source_run_id"]}
        exp.write_json(path, manifest)
    return branch_path


def test_shared_pre_score_reuses_exact_pre_evidence_without_relabeling(tmp_path):
    branch = shared_phase_fixture(tmp_path)
    report = exp.score_shared_runs(tmp_path / "pre", tmp_path / "off", tmp_path / "on",
                                   branch, {"metrics": []})
    assert report["paired_agent_days"] == 28
    assert report["shared_pre_provenance"]["executed_agent_days"] == 42
    assert report["period_totals"]["pre"]["off"] == report["period_totals"]["pre"]["on"]
    assert report["period_totals"]["post"]["off"]["billiard"]["revenue_krw"] == 216
    assert next(c for c in report["comparisons"] if c["group"] == "billiard" and c["metric"] == "revenue_krw")["percent_change"] == 100


@pytest.mark.parametrize("tamper", ["pre_manifest", "branch_hash", "post_snapshot", "post_day"])
def test_shared_score_rejects_broken_lineage_or_partial_phase(tmp_path, tamper):
    branch = shared_phase_fixture(tmp_path)
    if tamper == "pre_manifest":
        path = tmp_path / "pre/experiment_run.json"
        manifest = exp.read_json(path)
        manifest["workers"] = 7
        exp.write_json(path, manifest)
    elif tamper == "branch_hash":
        record = exp.read_json(branch)
        record["source_manifest_sha256"] = "0" * 64
        exp.write_json(branch, record)
    elif tamper == "post_snapshot":
        path = tmp_path / "on/experiment_run.json"
        manifest = exp.read_json(path)
        manifest["snapshot_sha256"] = "foreign"
        exp.write_json(path, manifest)
    else:
        (tmp_path / "on/metrics/day_2017-12-16.jsonl").unlink()
    with pytest.raises(ValueError):
        exp.score_shared_runs(tmp_path / "pre", tmp_path / "off", tmp_path / "on", branch, {"metrics": []})
