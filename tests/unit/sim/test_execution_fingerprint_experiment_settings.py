"""Resuming with changed policy experiment settings must be rejected."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
import experience_provenance as provenance  # noqa: E402


def test_experiment_setting_changes_execution_fingerprint(monkeypatch):
    monkeypatch.setattr(provenance, "source_fingerprint", lambda: "same-source")
    monkeypatch.setenv("EXP_DAILY_INCOME", "anchor")
    first = provenance.execution_fingerprint()
    monkeypatch.setenv("EXP_DAILY_INCOME", "baseline")
    assert provenance.execution_fingerprint() != first
    monkeypatch.setenv("EXP_DAILY_INCOME", "anchor")
    monkeypatch.setenv("EXP_SANGSAENG_BASE_RATIO", "0.3")
    assert provenance.execution_fingerprint() != first


def test_environment_pair_fingerprint_excludes_only_environment_id(monkeypatch):
    monkeypatch.setattr(provenance, "source_fingerprint", lambda: "same-source")
    monkeypatch.setenv("EXP_DAILY_INCOME", "baseline")
    monkeypatch.setenv("SIM_ENVIRONMENT", "covid_2021")
    on_full = provenance.execution_fingerprint()
    paired = provenance.paired_environment_fingerprint()
    monkeypatch.setenv("SIM_ENVIRONMENT", "covid_no_distancing")
    assert provenance.execution_fingerprint() != on_full
    assert provenance.paired_environment_fingerprint() == paired
    monkeypatch.setenv("EXP_DAILY_INCOME", "anchor")
    assert provenance.paired_environment_fingerprint() != paired
