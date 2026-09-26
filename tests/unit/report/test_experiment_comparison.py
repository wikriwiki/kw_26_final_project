"""Per-experiment reports cover the catalog without fabricating empirical gaps."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.report import build_experiment_comparison as report


ROOT = Path(__file__).resolve().parents[3]


def test_historical_score_covers_every_registered_indicator_without_false_gap():
    result = report.build([ROOT / "output/answerkey_round2/score_r2_v5.json"])
    assert result["indicator_count"] == 38
    assert result["simulated_count"] == 5
    assert result["direct_gap_count"] == 0
    rows = {r["id"]: r for r in result["rows"]}
    assert rows["P012-1"]["truth_unit"] == "log-point"
    assert rows["P012-1"]["gap"] is None
    assert rows["P012-2"]["status"] == "정의 변경"
    assert rows["P010-1"]["status"] == "미실행"
    assert {p["id"] for p in result["unregistered_policies"]} == {"P008", "P011"}


def _synthetic(tmp_path, *, score_on="2020-01-04:2020-01-05", audit_on="2020-01-04:2020-01-05"):
    audit = {
        "comparison": "matched_estimand", "source": "source/table 1",
        "reported_estimand": "paired spending change", "simulation_estimand": "paired spending change",
        "reported_window": "2020-01-01 to 2020-01-05",
        "simulation_window": "2020-01-01 to 2020-01-05",
        "reported_population": "same sample", "simulation_population": "same sample",
        "reported_denominator": "baseline spending", "simulation_denominator": "baseline spending",
        "reported_unit": "%", "simulation_unit": "%", "reported_value": 5.0,
        "simulation_off": "2020-01-01:2020-01-02", "simulation_on": audit_on,
    }
    scoring = tmp_path / "scoring.json"
    scoring.write_text(json.dumps({"P010": {"indicators": [
        {"id": "T-1", "metric": "total_spend_paired", "expect": "+",
         "desc": "test", "empirical_audit": audit}]}}), encoding="utf-8")
    score = tmp_path / "score.json"
    score.write_text(json.dumps({
        "policy": "P010", "label": "candidate", "off": "2020-01-01:2020-01-02",
        "on": score_on, "scoring_table_sha256": hashlib.sha256(scoring.read_bytes()).hexdigest(),
        "results": [{"id": "T-1", "metric": "total_spend_paired", "expect": "+",
                     "mean": 10, "base": 100, "ci": [5, 15], "n": 100, "hit": True}],
    }), encoding="utf-8")
    return scoring, score


def test_run_specific_matching_audit_allows_signed_gap_and_truth_marker(tmp_path):
    scoring, score = _synthetic(tmp_path)
    result = report.build([score], scoring)
    row = result["rows"][0]
    assert row["simulation"] == 10
    assert row["truth"] == 5
    assert row["gap"] == 5
    assert row["gap_unit"] == "%"
    assert row["external_direction_match"] is True
    markup = report.render(result)
    assert "시뮬−실측" in markup
    assert 'class="mk tru"' in markup


def test_changed_run_window_blocks_gap(tmp_path):
    scoring, score = _synthetic(tmp_path, score_on="2020-02-04:2020-02-05")
    row = report.build([score], scoring)["rows"][0]
    assert row["status"] == "실험 창 미감사"
    assert row["gap"] is None
    assert row["external_direction_match"] is None


def test_share_change_is_in_percentage_points_without_relative_denominator():
    ind = {"metric": "home_dong_spend_share", "expect": "+"}
    value, unit, ci = report._simulation(
        ind, {"mean": .05, "base": .50, "ci": [.02, .08]})
    assert (value, unit, ci) == (5.0, "%p", [2.0, 8.0])


def test_duplicate_policy_scores_refused(tmp_path):
    scoring, score = _synthetic(tmp_path)
    with pytest.raises(ValueError, match="duplicate policy"):
        report.build([score, score], scoring, experiment="bad_merge")


def test_generated_report_keeps_sources_and_machine_readable_coverage(tmp_path):
    scoring, score = _synthetic(tmp_path)
    out, json_out, result = report.generate([score], out=tmp_path / "comparison.html",
                                            scoring_path=scoring)
    saved = json.loads(json_out.read_text(encoding="utf-8"))
    assert saved["indicator_count"] == result["indicator_count"]
    assert saved["score_files"][0]["sha256"] == hashlib.sha256(score.read_bytes()).hexdigest()
    assert "T-1" in out.read_text(encoding="utf-8")
