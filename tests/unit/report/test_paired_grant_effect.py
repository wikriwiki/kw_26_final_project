"""The funding denominator and complete paired ledger are validation gates."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "report"))
import paired_grant_effect as effect  # noqa: E402
import export_policy_daily_ledger as exporter  # noqa: E402


def row(aid, day, arm, offline, online, eligible, received=0, spent=0, remaining=0):
    return {"aid": aid, "day": day, "arm": arm, "policy_id": "P013",
            "offline_spent": offline, "online_spent": online,
            "eligible_offline_spent": eligible,
            "grant_received_cumulative": received,
            "grant_spent_today": spent, "grant_remaining": remaining}


def complete_pair():
    days = ["2020-05-11", "2020-05-12"]
    on = [row("a", days[0], "on", 130, 20, 110, 100, 40, 60),
          row("a", days[1], "on", 80, 0, 70, 100, 20, 40),
          row("b", days[0], "on", 50, 0, 50, 100, 0, 100),
          row("b", days[1], "on", 40, 0, 40, 100, 0, 100)]
    off = [row("a", days[0], "off", 100, 10, 90),
           row("a", days[1], "off", 80, 0, 70),
           row("b", days[0], "off", 50, 0, 50),
           row("b", days[1], "off", 40, 0, 40)]
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
    assert result["comparison"] == "indirect_proxy"


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


def test_exporter_keeps_zero_spend_citizen_and_fails_on_missing_state(monkeypatch):
    def eligibility(rows, _file):
        for item in rows:
            item["elig"] = item.get("sub") == "식사"
        return "test ruler"
    monkeypatch.setattr(exporter, "apply_policy_eligibility", eligibility)
    states = [{"aid": "a", "online_spent": 10,
               "grant_received": '{"P013": 100}', "grant_remaining": '{"P013": 60}'},
              {"aid": "b", "online_spent": 0,
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
