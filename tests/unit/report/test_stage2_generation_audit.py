"""A model-free Stage2 fallback invalidates policy magnitude scoring."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "report"))
import audit_stage2_generation as audit  # noqa: E402


def test_all_failed_stage2_attempts_fail_quality_gate_even_with_ok_metrics():
    rows = {"2020-05-11": [
        {"aid": "a", "status": "ok", "s2_timing": {
            "n_llm_calls": 1, "attempts": [{"status": "ok", "tokens_out": 300}]}},
        {"aid": "b", "status": "ok", "s2_timing": {
            "n_llm_calls": 3, "attempts": [
                {"status": "error", "error_stage": "json_parse", "tokens_out": 1400},
                {"status": "error", "error_stage": "review_lookup", "tokens_out": 800},
                {"status": "error", "error_stage": "json_parse", "tokens_out": 1400}]}}
    ]}
    result = audit.inspect(rows, expected_per_day=2)
    assert result["quality_gate_pass"] is False
    assert result["totals"]["agents_error"] == 0
    assert result["totals"]["stage2_fallback_only_agents"] == 1
    assert result["totals"]["stage2_choice_repair_agents"] == 1
    assert result["unrepaired_choice_trace_pass"] is False
    assert result["totals"]["stage2_fallback_with_output_limit_agents"] == 1
    assert result["totals"]["stage2_fallback_with_review_error_agents"] == 1
    assert result["totals"]["stage2_output_limited_attempts"] == 2
    assert result["totals"]["stage2_extra_calls"] == 2


def test_clean_day_and_missing_citizen_are_distinguished():
    ok = {"aid": "a", "status": "ok", "s2_timing": {
        "n_llm_calls": 1, "attempts": [{"status": "ok", "tokens_out": 300}]}}
    clean = audit.inspect({"2020-05-11": [ok]}, expected_per_day=1)
    assert clean["quality_gate_pass"]
    assert clean["unrepaired_choice_trace_pass"] is True
    result = audit.inspect({"2020-05-11": [ok]}, expected_per_day=2)
    assert result["quality_gate_pass"] is False
    assert result["totals"]["stage2_fallback_only_agents"] == 0
    assert result["totals"]["stage2_fallback_with_output_limit_agents"] == 0
    assert result["unrepaired_choice_trace_pass"] is False


def test_same_daily_count_with_changed_citizen_roster_fails():
    def clean(aid):
        return {"aid": aid, "status": "ok", "s2_timing": {
            "n_llm_calls": 1, "attempts": [{"status": "ok", "tokens_out": 10}]}}
    result = audit.inspect({"2020-05-11": [clean("a"), clean("b")],
                            "2020-05-12": [clean("a"), clean("c")]},
                           expected_per_day=2)
    assert result["quality_gate_pass"] is False
    assert result["per_day"][1]["roster_matches_first_day"] is False


def test_review_retry_without_final_valid_pick_is_not_a_model_decision():
    result = audit.inspect({"2020-05-11": [{
        "aid": "a", "status": "ok", "s2_timing": {"n_llm_calls": 2,
            "attempts": [{"status": "review_retry"}, {"status": "error"}]},
        "fb_missing_picks_filled": 2, "fb_hallucinations_corrected": 1,
        "fb_spend_amount_fallbacks": 3,
    }]}, expected_per_day=1)
    assert result["quality_gate_pass"] is False
    assert result["totals"]["stage2_fallback_only_agents"] == 1
    assert result["totals"]["stage2_missing_picks_filled"] == 2
    assert result["totals"]["stage2_hallucinations_corrected"] == 1
    assert result["totals"]["stage2_partial_repair_agents"] == 1
    assert result["totals"]["stage2_choice_repair_agents"] == 1
    assert result["totals"]["stage2_spend_amount_fallbacks"] == 3


def test_old_metrics_do_not_claim_zero_spend_amount_fallbacks():
    result = audit.inspect({"2020-05-11": [{
        "aid": "a", "status": "ok", "s2_timing": {"n_llm_calls": 1,
            "attempts": [{"status": "ok"}]},
    }]}, expected_per_day=1)
    assert result["totals"]["stage2_spend_amount_observed_agents"] == 0
    assert result["totals"]["stage2_spend_amount_fallbacks"] is None


def test_missing_stage2_trace_needs_explicit_skip_marker():
    row = {"aid": "a", "status": "ok", "s2_timing": {"attempts": []}}
    result = audit.inspect({"2020-05-11": [row]}, expected_per_day=1)
    assert result["quality_gate_pass"] is False
    assert result["totals"]["stage2_missing_decision_evidence_agents"] == 1
    row["s2_skipped"] = True
    assert audit.inspect({"2020-05-11": [row]}, expected_per_day=1)["quality_gate_pass"]
