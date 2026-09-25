"""The funding denominator and complete paired ledger are validation gates."""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "report"))
import paired_grant_effect as effect  # noqa: E402
import export_policy_daily_ledger as exporter  # noqa: E402


def row(aid, day, arm, offline, online, eligible, received=0, spent=0,
        remaining=0, self_cumulative=0):
    return {"aid": aid, "day": day, "arm": arm, "policy_id": "P013",
            "s2_choice_status": "unrepaired",
            "offline_spent": offline, "online_spent": online,
            "self_month_cumulative": self_cumulative,
            "eligible_offline_spent": eligible,
            "grant_received_cumulative": received,
            "grant_spent_today": spent, "grant_remaining": remaining}


def complete_pair():
    days = ["2020-05-11", "2020-05-12"]
    on = [row("a", days[0], "on", 130, 20, 110, 100, 40, 60, 110),
          row("a", days[1], "on", 80, 0, 70, 100, 20, 40, 170),
          row("b", days[0], "on", 50, 0, 50, 100, 0, 100, 50),
          row("b", days[1], "on", 40, 0, 40, 100, 0, 100, 90)]
    off = [row("a", days[0], "off", 100, 10, 90, self_cumulative=110),
           row("a", days[1], "off", 80, 0, 70, self_cumulative=190),
           row("b", days[0], "off", 50, 0, 50, self_cumulative=50),
           row("b", days[1], "off", 40, 0, 40, self_cumulative=90)]
    return on, off, days


def test_paired_incremental_spending_divides_by_once_issued_grant():
    on, off, days = complete_pair()
    result = effect.score(on, off, roster=["a", "b"], days=days,
                          policy_id="P013", draws=100)
    assert result["grant_issued_won"] == 200  # cumulative receipt, not 400 citizen-days
    assert result["grant_recipients"] == 2
    assert result["grant_spent_won"] == 60
    assert result["grant_remaining_won"] == 140
    assert result["recorded_total_spend_difference_won"] == 40
    assert result["incremental_recorded_spend_per_grant_won"] == pytest.approx(0.2)
    assert result["offline_effect_per_grant_won"] == pytest.approx(0.15)
    assert result["eligible_offline_effect_per_grant_won"] == pytest.approx(0.1)
    assert result["bootstrap_valid_draws"] == 100
    assert result["eligible_offline_citizen_bootstrap_95_interval"][0] <= 0.1
    assert result["eligible_offline_citizen_bootstrap_95_interval"][1] >= 0.1
    assert result["comparison"] == "indirect_proxy"


def test_grant_effect_reports_unrepaired_citizen_sensitivity():
    on, off, days = complete_pair()
    on[0]["s2_choice_status"] = "partial_repair"
    result = effect.score(on, off, roster=["a", "b"], days=days,
                          policy_id="P013", draws=0)
    sensitivity = result["choice_repair_sensitivity"]
    assert sensitivity["unrepaired_citizens"] == 1
    assert sensitivity["excluded_citizens"] == 1
    assert sensitivity["eligible_offline_effect_per_grant_won"] == 0
    assert result["eligible_offline_effect_per_grant_won"] == pytest.approx(0.1)
    on[2]["s2_choice_status"] = "partial_repair"
    empty = effect.score(on, off, roster=["a", "b"], days=days,
                         policy_id="P013", draws=0)
    assert empty["choice_repair_sensitivity"]["unrepaired_citizens"] == 0
    assert empty["choice_repair_sensitivity"]["eligible_offline_effect_per_grant_won"] is None
    off[0].pop("s2_choice_status")
    with pytest.raises(ValueError, match="choice provenance"):
        effect.score(on, off, roster=["a", "b"], days=days,
                     policy_id="P013", draws=0)
    off[0]["s2_choice_status"] = []
    with pytest.raises(ValueError, match="choice provenance"):
        effect.score(on, off, roster=["a", "b"], days=days,
                     policy_id="P013", draws=0)


def test_grant_reference_uses_eligible_sector_proxy_without_accuracy_claim():
    on, off, days = complete_pair()
    result = effect.score(on, off, roster=["a", "b"], days=days,
                          policy_id="P013", draws=100)
    reference = {"policy_id": "P013", "simulation_start": days[0],
                 "simulation_end": days[-1],
                 "simulation_proxy": "eligible_offline_effect_per_grant_won",
                 "external_ratio_interval": [0.262, 0.361]}
    comparison = effect.compare_reference(result, reference)
    assert comparison["simulated_ratio"] == pytest.approx(0.1)
    assert comparison["same_positive_direction"] is True
    assert comparison["descriptive_overlap_only"] is False
    assert "no direct accuracy score" in comparison["comparison_status"]
    with pytest.raises(ValueError, match="does not match"):
        effect.compare_reference(result, {**reference, "simulation_end": "2020-06-21"})


def test_registered_kdi_reference_matches_proxy_contract():
    reference = json.loads((ROOT / "data/experiments/p013_grant_reference_20260926.json")
                           .read_text(encoding="utf-8"))
    result = {"policy_id": "P013", "start": "2020-05-11", "end": "2020-06-21",
              "eligible_offline_effect_per_grant_won": 0.30,
              "eligible_offline_citizen_bootstrap_95_interval": [0.25, 0.35]}
    comparison = effect.compare_reference(result, reference)
    assert comparison["descriptive_overlap_only"] is True
    assert comparison["external_ratio_interval"] == [0.262, 0.361]
    assert comparison["comparison_status"] == "proxy_scale_reference; no direct accuracy score"


def test_incomplete_pair_and_control_leak_cannot_be_scored():
    on, off, days = complete_pair()
    with pytest.raises(ValueError, match="incomplete"):
        effect.score(on, off[:-1], roster=["a", "b"], days=days,
                     policy_id="P013", draws=0)
    off[0]["grant_spent_today"] = 1
    with pytest.raises(ValueError, match="policy funding leaked"):
        effect.score(on, off, roster=["a", "b"], days=days,
                     policy_id="P013", draws=0)


def test_universal_grant_requires_registered_recipient_count():
    on, off, days = complete_pair()
    with pytest.raises(ValueError, match="recipient count mismatch"):
        effect.score(on, off, roster=["a", "b"], days=days,
                     policy_id="P013", draws=0, expected_recipients=3)
    with pytest.raises(ValueError, match="issued amount mismatch"):
        effect.score(on, off, roster=["a", "b"], days=days,
                     policy_id="P013", draws=0, expected_issued_won=201)


def test_wallet_mismatch_and_eligible_payment_violation_are_rejected():
    on, off, days = complete_pair()
    on[1]["grant_remaining"] = 41
    with pytest.raises(ValueError, match="does not reconcile"):
        effect.score(on, off, roster=["a", "b"], days=days,
                     policy_id="P013", draws=0)
    on[1]["grant_remaining"] = 40
    on[1]["eligible_offline_spent"] = 19
    with pytest.raises(ValueError, match="exceeds eligible"):
        effect.score(on, off, roster=["a", "b"], days=days,
                     policy_id="P013", draws=0)
    on[1].pop("eligible_offline_spent")
    with pytest.raises(ValueError, match="invalid eligible"):
        effect.score(on, off, roster=["a", "b"], days=days,
                     policy_id="P013", draws=0)


def test_monthly_own_spend_must_match_realized_transactions():
    on, off, days = complete_pair()
    on[1]["self_month_cumulative"] += 1
    with pytest.raises(ValueError, match="own-spend monthly ledger mismatch"):
        effect.score(on, off, roster=["a", "b"], days=days,
                     policy_id="P013", draws=0)


def test_exporter_keeps_zero_spend_citizen_and_fails_on_missing_state(monkeypatch):
    def eligibility(rows, _file):
        for item in rows:
            item["elig"] = item.get("sub") == "식사"
        return "test ruler"
    monkeypatch.setattr(exporter, "apply_policy_eligibility", eligibility)
    states = [{"aid": "a", "online_spent": 10, "self_month_cumulative": 20,
               "grant_received": '{"P013": 100}', "grant_remaining": '{"P013": 60}'},
              {"aid": "b", "online_spent": 0, "self_month_cumulative": 0,
               "grant_received": '{"P013": 100}', "grant_remaining": '{"P013": 100}'}]
    spends = [{"aid": "a", "amt": 50, "sub": "식사",
               "spent_from_policy": '{"P013": 40}'}]
    rows = exporter.aggregate_day(states, spends, roster=["a", "b"],
                                  day="2020-05-11", arm="on", policy_id="P013",
                                  policy_file="data/neo4j_load/policies/P013.json")
    assert rows[0]["offline_spent"] == 50
    assert rows[0]["eligible_offline_spent"] == 50
    assert rows[1]["offline_spent"] == 0
    with pytest.raises(ValueError, match="incomplete State"):
        exporter.aggregate_day(states[:1], spends, roster=["a", "b"],
                               day="2020-05-11", arm="on", policy_id="P013",
                               policy_file="data/neo4j_load/policies/P013.json")


def test_exporter_requires_canonical_all_ok_metrics(tmp_path):
    metrics = tmp_path / "day_2020-05-11.jsonl"
    metrics.write_text(json.dumps({"aid": "a", "status": "ok"}) + "\n"
                       + json.dumps({"aid": "b", "status": "error"}) + "\n",
                       encoding="utf-8")
    with pytest.raises(ValueError, match="non-ok"):
        exporter.verify_metrics(metrics, ["a", "b"], "on")


def test_exporter_checks_day_of_receipt_against_final_state():
    previous = {}
    day1 = [{"aid": "a", "day": "2020-05-11", "grant_received_cumulative": 100}]
    exporter.verify_receipt_deltas(day1, {"a": {"grant_applied_today": 100}}, previous)
    day2 = [{"aid": "a", "day": "2020-05-12", "grant_received_cumulative": 100}]
    exporter.verify_receipt_deltas(day2, {"a": {"grant_applied_today": 0}}, previous)
    with pytest.raises(ValueError, match="disagrees with State"):
        exporter.verify_receipt_deltas(day1, {"a": {"grant_applied_today": 0}}, {})


def test_exporter_self_spend_resets_at_month_boundary():
    previous = {}
    may = [{"aid": "a", "day": "2020-05-31", "offline_spent": 100,
            "grant_spent_today": 40, "online_spent": 10,
            "self_month_cumulative": 70}]
    june = [{"aid": "a", "day": "2020-06-01", "offline_spent": 80,
             "grant_spent_today": 0, "online_spent": 20,
             "self_month_cumulative": 100}]
    exporter.verify_self_spend_deltas(may, previous)
    exporter.verify_self_spend_deltas(june, previous)
    june[0]["self_month_cumulative"] = 101
    with pytest.raises(ValueError, match="own-spend ledger disagrees"):
        exporter.verify_self_spend_deltas(june, {"a": ("2020-05", 70)})


def test_exporter_requires_policy_exposure_evidence(tmp_path):
    path = tmp_path / "day_2020-05-11.jsonl"
    path.write_text(json.dumps({"aid": "a", "status": "ok"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing policy exposure"):
        exporter.verify_metrics(path, ["a"], "on", "P013")


def test_exporter_rejects_wrong_policy_graph():
    policy = {"id": "P013", "type": "grant", "effective_from": "2020-05-11",
              "effective_until": "2020-08-31", "grant_key": "spend_decile",
              "decile_grants": {"1": 280000}, "poi_restricted": True}
    graph = {**policy, "decile_grants": '{"1":280000}'}
    exporter.verify_graph_policy([{"policy": graph}], "on", policy)
    with pytest.raises(ValueError, match="control graph contains"):
        exporter.verify_graph_policy([{"policy": graph}], "off", policy)
    with pytest.raises(ValueError, match="decile_grants"):
        exporter.verify_graph_policy([{"policy": {**graph,
                                                   "decile_grants": '{"1":100000}'}}],
                                     "on", policy)


def test_exporter_writes_audited_ledger_and_manifest(tmp_path, monkeypatch):
    policy = {"id": "P013", "type": "grant", "effective_from": "2020-05-11",
              "effective_until": "2020-08-31", "grant_key": "income",
              "decile_grants": {}, "poi_restricted": False}
    policy_file = tmp_path / "policy.json"
    policy_file.write_text(json.dumps(policy), encoding="utf-8")
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    days = ["2020-05-11", "2020-05-12"]
    for index, day in enumerate(days):
        rows = [{"aid": aid, "status": "ok", "experience_policy_ids": ["P013"],
                 "execution_fingerprint": "same-code", "experience_run_id": "on-run",
                 "grant_applied_today": 100 if index == 0 else 0,
                 "s2_timing": {"n_llm_calls": 1, "attempts": [{"status": "ok"}]}}
                for aid in ("a", "b")]
        (metrics_dir / f"day_{day}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        (tmp_path / f"cohort_{day}.json").write_text(json.dumps({
            "agent_ids": ["a", "b"], "execution_fingerprint": "same-code",
            "baseline_income_map_sha256": "same-income", "run_id": "on-run",
            "prompt_variant": "v51", "system_prompt_sha256": "a" * 64}),
            encoding="utf-8")

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def run(self, query, **_kwargs):
            if query == exporter.POLICY_QUERY:
                return [{"policy": {**policy, "decile_grants": "{}"}}]
            if query == exporter.STATE_QUERY:
                return [{"aid": aid, "online_spent": 0,
                         "self_month_cumulative": 0,
                         "grant_received": '{"P013":100}',
                         "grant_remaining": '{"P013":100}'} for aid in ("a", "b")]
            if query == exporter.SPEND_QUERY:
                return []
            raise AssertionError("unexpected query")

    monkeypatch.setattr(exporter, "driver_session", lambda: Session())
    out = tmp_path / "on.jsonl"
    count = exporter.export(roster=["a", "b"], days=days, arm="on",
                            policy_id="P013", policy_file=str(policy_file),
                            metrics_dir=metrics_dir, out=out)
    assert count == len(out.read_text(encoding="utf-8").splitlines()) == 4
    manifest = json.loads((tmp_path / "on.jsonl.manifest.json").read_text(encoding="utf-8"))
    assert manifest["quality_gate_pass"] is True
    assert manifest["run_id"] == "on-run"
    assert manifest["output_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert all(json.loads(line)["s2_choice_status"] == "unrepaired"
               for line in out.read_text(encoding="utf-8").splitlines())


def test_paired_grant_manifest_rejects_reused_run(tmp_path):
    roster = ["a"]
    paths = {}
    for arm in ("on", "off"):
        path = tmp_path / f"{arm}.jsonl"
        path.write_text('{}\n', encoding="utf-8")
        (tmp_path / f"{arm}.jsonl.manifest.json").write_text(json.dumps({
            "arm": arm, "policy_id": "P013", "start": "2020-05-11",
            "end": "2020-05-11", "days": 1, "citizens": 1, "rows": 1,
            "roster_sha256": hashlib.sha256(json.dumps(roster).encode()).hexdigest(),
            "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "quality_gate_pass": True, "policy_file_sha256": "same-policy",
            "execution_fingerprint": "same-code",
            "baseline_income_map_sha256": "same-income", "run_id": "reused-run",
            "prompt_variant": "v51", "system_prompt_sha256": "a" * 64,
        }), encoding="utf-8")
        paths[arm] = path
    with pytest.raises(ValueError, match="distinct run IDs"):
        effect.verify_manifests(paths["on"], paths["off"], roster=roster,
                                days=["2020-05-11"], policy_id="P013")
    off_manifest = tmp_path / "off.jsonl.manifest.json"
    altered = json.loads(off_manifest.read_text(encoding="utf-8"))
    altered["run_id"] = "off-run"
    off_manifest.write_text(json.dumps(altered), encoding="utf-8")
    provenance = effect.verify_manifests(paths["on"], paths["off"], roster=roster,
                                         days=["2020-05-11"], policy_id="P013")
    assert provenance["prompt_variant"] == "v51"
    assert provenance["arms"]["off"]["run_id"] == "off-run"
