"""Cashback size references use recipients and a complete paired month."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "report"))
import paired_cashback_month as paired  # noqa: E402


def fixture_rows():
    on, off = [], []
    for index, day in enumerate(paired.month_days("2021-10"), 1):
        for aid in ("a", "b"):
            for arm, target in (("on", on), ("off", off)):
                eligible = (100 if arm == "on" else 80) if aid == "a" else 0
                target.append({
                    "aid": aid, "day": day, "arm": arm, "policy_id": "P012",
                    "month": "2021-10", "offline_spent": eligible,
                    "online_spent": 0, "eligible_spent": eligible,
                    "eligible_cumulative": eligible * index,
                    "self_month_cumulative": eligible * index,
                    "anchor_won": 1000, "threshold_won": 1030,
                    "cashback_accrued_won": 207.0 if aid == "a" and arm == "on" and index == 31 else 0.0,
                    "cashback_cap_reached": False,
                })
    return on, off


def test_recipient_denominator_and_paired_spending_are_distinct():
    on, off = fixture_rows()
    result = paired.score(on, off, roster=["a", "b"], month="2021-10",
                          policy_id="P012", draws=100)
    assert result["recipients"] == 1
    assert result["total_cashback_accrued_won"] == 207
    assert result["metrics"]["cashback_per_recipient_won"]["value"] == 207
    assert result["metrics"]["cap_share_recipients"]["value"] == 0
    assert result["metrics"]["eligible_difference_per_citizen_won"]["value"] == 310
    assert result["metrics"]["eligible_relative_change"]["value"] == 0.25
    assert result["metrics"]["threshold_reach_share"]["value"] == 0.5


def test_missing_day_or_control_cashback_cannot_score():
    on, off = fixture_rows()
    with pytest.raises(ValueError, match="incomplete citizen-day"):
        paired.score(on[:-1], off, roster=["a", "b"], month="2021-10",
                     policy_id="P012", draws=0)
    off[-2]["cashback_accrued_won"] = 1
    with pytest.raises(ValueError, match="leaked into control"):
        paired.score(on, off, roster=["a", "b"], month="2021-10",
                     policy_id="P012", draws=0)


def test_baseline_must_match_between_arms():
    on, off = fixture_rows()
    off[0]["threshold_won"] = 999
    with pytest.raises(ValueError, match="between arms"):
        paired.score(on, off, roster=["a", "b"], month="2021-10",
                     policy_id="P012", draws=0)


def test_manifest_rejects_different_execution_fingerprint(tmp_path):
    roster = ["a", "b"]
    roster_sha = hashlib.sha256(json.dumps(roster, ensure_ascii=False).encode()).hexdigest()
    paths = {}
    for arm, fingerprint in (("on", "source-a"), ("off", "source-b")):
        path = tmp_path / f"{arm}.jsonl"
        path.write_text('{}\n', encoding="utf-8")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        (tmp_path / f"{arm}.jsonl.manifest.json").write_text(json.dumps({
            "arm": arm, "month": "2021-10", "policy_id": "P012",
            "roster_sha256": roster_sha, "output_sha256": sha,
            "quality_gate_pass": True, "policy_file_sha256": "same-policy",
            "base_ratio": 0.268, "execution_fingerprint": fingerprint,
            "baseline_income_map_sha256": "same-income", "citizens": 2, "days": 31,
        }), encoding="utf-8")
        paths[arm] = path
    with pytest.raises(ValueError, match="execution_fingerprint"):
        paired.verify_manifests(paths["on"], paths["off"], roster, "2021-10")
