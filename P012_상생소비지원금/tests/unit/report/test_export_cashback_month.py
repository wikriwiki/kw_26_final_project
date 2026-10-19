"""Monthly cashback must use the policy file and complete settled daily ledger."""
from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "report"))
import export_cashback_month as cashback  # noqa: E402


def policy():
    return {"id": "P012", "type": "cashback", "effective_from": "2021-10-01",
            "effective_until": "2021-11-30", "benefit_rate": 0.1,
            "threshold_ratio": 1.03, "cap_per_agent": 100000}


def test_policy_window_and_amounts_are_from_frozen_file(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy()), encoding="utf-8")
    assert cashback.load_policy(path, "2021-10", "P012")["benefit_rate"] == 0.1
    changed = policy()
    changed["effective_from"] = "2021-10-15"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="complete calendar month"):
        cashback.load_policy(path, "2021-10", "P012")


def test_agent_anchor_matches_runtime_fallback_and_measured_priority():
    agent = {"aid": "a", "daily_wd": 70000, "daily_we": 35000,
             "sangsaeng_base_daily": None}
    expected = round((70000 * 5 + 35000 * 2) / 7 * 30 * 0.268)
    assert cashback.monthly_anchor(agent, 0.268) == (expected, "total_daily_scaled")
    agent["sangsaeng_base_daily"] = 12345
    assert cashback.monthly_anchor(agent, 0.268) == (370350, "measured_eligible_daily")


def test_daily_state_increment_must_equal_actual_eligible_and_total_spend():
    previous = {}
    first = cashback.aggregate_day(
        [{"aid": "a", "eligible_cumulative": 100,
          "self_month_cumulative": 120, "online_spent": 20},
         {"aid": "b", "eligible_cumulative": 0,
          "self_month_cumulative": 0, "online_spent": 0}],
        [{"aid": "a", "spent": 100, "eligible": True}],
        ["a", "b"], "2021-10-01", previous)
    assert first[0]["eligible_spent"] == 100
    assert first[1]["offline_spent"] == 0
    assert previous["a"] == (100, 120)
    next_state = [{"aid": "a", "eligible_cumulative": 150,
                   "self_month_cumulative": 170, "online_spent": 0},
                  {"aid": "b", "eligible_cumulative": 0,
                   "self_month_cumulative": 0, "online_spent": 0}]
    cashback.aggregate_day(next_state,
                           [{"aid": "a", "spent": 50, "eligible": True}],
                           ["a", "b"], "2021-10-02", previous)
    with pytest.raises(ValueError, match="eligible ledger disagrees"):
        cashback.aggregate_day(next_state,
                               [{"aid": "a", "spent": 49, "eligible": True}],
                               ["a", "b"], "2021-10-03", previous)


def test_positive_transaction_without_eligibility_is_not_silently_excluded():
    with pytest.raises(ValueError, match="missing eligibility"):
        cashback.aggregate_day(
            [{"aid": "a", "eligible_cumulative": 0,
              "self_month_cumulative": 100, "online_spent": 0}],
            [{"aid": "a", "spent": 100, "eligible": None}],
            ["a"], "2021-10-01", {})


def test_control_cannot_contain_policy_and_on_must_match_file():
    with pytest.raises(ValueError, match="control graph"):
        cashback.check_graph_policy([{"policy": policy()}], "off", policy())
    cashback.check_graph_policy([], "off", policy())
    cashback.check_graph_policy([{"policy": policy()}], "on", policy())
    changed = policy()
    changed["cap_per_agent"] = 50000
    with pytest.raises(ValueError, match="cap_per_agent"):
        cashback.check_graph_policy([{"policy": changed}], "on", policy())


def test_complete_month_exports_one_audited_row_per_citizen_day(tmp_path, monkeypatch):
    days = cashback.month_days("2021-10")
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    for day in days:
        rows = [{"aid": aid, "status": "ok",
                 "execution_fingerprint": "same-code-and-settings",
                 "experience_run_id": "on-run", "s2_timing": {
            "n_llm_calls": 1, "attempts": [{"status": "ok"}]}} for aid in ("a", "b")]
        (metrics_dir / f"day_{day}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        (tmp_path / f"cohort_{day}.json").write_text(json.dumps({
            "agent_ids": ["a", "b"], "execution_fingerprint": "same-code-and-settings",
            "baseline_income_map_sha256": "same-income-map", "run_id": "on-run",
            "prompt_variant": "v51", "system_prompt_sha256": "a" * 64}),
            encoding="utf-8")
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy()), encoding="utf-8")

    class Session:
        def run(self, query, **params):
            if query == cashback.POLICY_QUERY:
                return [{"policy": policy()}]
            if query == cashback.AGENT_QUERY:
                return [{"aid": aid, "daily_wd": 100, "daily_we": 100,
                         "sangsaeng_base_daily": None} for aid in ("a", "b")]
            idx = days.index(params["day"]) + 1
            if query == cashback.STATE_QUERY:
                return [{"aid": "a", "eligible_cumulative": 100 * idx,
                         "self_month_cumulative": 100 * idx, "online_spent": 0},
                        {"aid": "b", "eligible_cumulative": 0,
                         "self_month_cumulative": 0, "online_spent": 0}]
            if query == cashback.SPEND_QUERY:
                return [{"aid": "a", "spent": 100, "eligible": True}]
            raise AssertionError("unexpected query")

    monkeypatch.setattr(cashback, "driver_session", lambda: nullcontext(Session()))
    out = tmp_path / "on.jsonl"
    count = cashback.export(month="2021-10", arm="on", policy_id="P012",
                            policy_file=path, base_ratio=0.268,
                            roster=["a", "b"], metrics_dir=metrics_dir, out=out)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert count == len(rows) == 62
    assert rows[-2]["eligible_cumulative"] == 3100
    assert rows[-2]["cashback_accrued_won"] > 0
    assert rows[-1]["cashback_accrued_won"] == 0
    assert all(row["s2_choice_status"] == "unrepaired" for row in rows)
    manifest = json.loads((tmp_path / "on.jsonl.manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == 62
    assert manifest["quality_gate_pass"] is True
    assert manifest["execution_fingerprint"] == "same-code-and-settings"
    assert manifest["baseline_income_map_sha256"] == "same-income-map"
    assert manifest["prompt_variant"] == "v51"
    assert manifest["system_prompt_sha256"] == "a" * 64
    assert manifest["unrepaired_choice_trace_pass"] is True
    assert not list(tmp_path.glob("*.tmp.*"))


def test_cohort_fingerprint_change_blocks_month_export(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    for day, fingerprint in (("2021-10-01", "a"), ("2021-10-02", "b")):
        (tmp_path / f"cohort_{day}.json").write_text(json.dumps({
            "agent_ids": ["citizen"], "execution_fingerprint": fingerprint,
            "baseline_income_map_sha256": "same-income-map", "run_id": "same-run",
            "prompt_variant": "v51", "system_prompt_sha256": "a" * 64}),
            encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        cashback.verify_cohorts(metrics_dir, ["2021-10-01", "2021-10-02"], ["citizen"])


def test_cohort_prompt_change_blocks_month_export(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    days = ["2021-10-01", "2021-10-02"]
    for day, prompt_sha in zip(days, ("a" * 64, "b" * 64)):
        (tmp_path / f"cohort_{day}.json").write_text(json.dumps({
            "agent_ids": ["citizen"], "execution_fingerprint": "same-code",
            "baseline_income_map_sha256": "same-income-map", "run_id": "same-run",
            "prompt_variant": "v51", "system_prompt_sha256": prompt_sha}),
            encoding="utf-8")
    with pytest.raises(ValueError, match="prompt changed"):
        cashback.verify_cohorts(metrics_dir, days, ["citizen"])


def test_v53_requires_consistent_stage2_prompt_hash(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    days = ["2021-10-01", "2021-10-02"]
    for day, stage2_sha in zip(days, ("a" * 64, "b" * 64)):
        (tmp_path / f"cohort_{day}.json").write_text(json.dumps({
            "agent_ids": ["citizen"], "execution_fingerprint": "same-code",
            "baseline_income_map_sha256": "same-income-map", "run_id": "same-run",
            "prompt_variant": "v53", "system_prompt_sha256": "c" * 64,
            "stage2_system_prompt_sha256": stage2_sha,
        }), encoding="utf-8")
    with pytest.raises(ValueError, match="prompt changed"):
        cashback.verify_cohorts(metrics_dir, days, ["citizen"])
    (tmp_path / f"cohort_{days[1]}.json").write_text(json.dumps({
        "agent_ids": ["citizen"], "execution_fingerprint": "same-code",
        "baseline_income_map_sha256": "same-income-map", "run_id": "same-run",
        "prompt_variant": "v53", "system_prompt_sha256": "c" * 64,
        "stage2_system_prompt_sha256": "a" * 64,
    }), encoding="utf-8")
    assert cashback.verify_cohorts(metrics_dir, days, ["citizen"])[
        "stage2_system_prompt_sha256"] == "a" * 64


def test_missing_run_id_blocks_month_export(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (tmp_path / "cohort_2021-10-01.json").write_text(json.dumps({
        "agent_ids": ["citizen"], "execution_fingerprint": "same-code",
        "baseline_income_map_sha256": "same-income-map"}), encoding="utf-8")
    with pytest.raises(ValueError, match="run ID missing"):
        cashback.verify_cohorts(metrics_dir, ["2021-10-01"], ["citizen"])


def test_metrics_from_another_run_cannot_join_same_day_cohort():
    cohort = {"execution_fingerprint": "code-A", "run_id": "run-A"}
    with pytest.raises(ValueError, match="provenance differs"):
        cashback.verify_metric_provenance({"2021-10-01": [{
            "aid": "a", "execution_fingerprint": "code-A",
            "experience_run_id": "run-B"}]}, cohort)
