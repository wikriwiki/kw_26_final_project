"""P010 uses a complete pre/post ledger but scores same-date post-policy arms."""
from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "report"))
import paired_coupon_effect as coupon  # noqa: E402
from paired_coupon_effect import ALL_DAYS, score_coupon  # noqa: E402


def _pair():
    on, off = [], []
    own_on = own_off = 0
    for day in ALL_DAYS:
        received = 200 if day >= "2025-07-21" else 0
        if day == "2025-07-21":
            on_spend, eligible, funded, remaining = 100, 80, 80, 120
        elif day == "2025-07-22":
            on_spend, eligible, funded, remaining = 120, 100, 100, 20
        else:
            on_spend, eligible, funded = 100, 80, 0
            remaining = 20 if day > "2025-07-22" else 0
        own_on += on_spend - funded
        own_off += 100
        common = {"aid": "a", "day": day, "policy_id": "P010",
                  "s2_choice_status": "unrepaired", "online_spent": 0}
        on.append({**common, "arm": "on", "offline_spent": on_spend,
                   "eligible_offline_spent": eligible,
                   "grant_received_cumulative": received,
                   "grant_spent_today": funded, "grant_remaining": remaining,
                   "self_month_cumulative": own_on})
        off.append({**common, "arm": "off", "offline_spent": 100,
                    "eligible_offline_spent": 80,
                    "grant_received_cumulative": 0, "grant_spent_today": 0,
                    "grant_remaining": 0, "self_month_cumulative": own_off})
    return on, off


def test_p010_registered_post_effect_and_prepolicy_diagnostic_are_separate():
    on, off = _pair()
    result = score_coupon(on, off, roster=["a"], draws=50,
                          expected_issued_won=200)
    assert result["P010-2"]["difference_won"] == 20
    assert result["P010-2"]["relative_change"] == pytest.approx(0.125)
    assert result["P010-3"]["difference_won"] == 20
    assert result["P010-3"]["relative_change"] == pytest.approx(0.1)
    assert result["prepolicy_balance_diagnostic"]["total_difference_won"] == 0
    assert result["funding"] == {"issued_won": 200, "spent_won": 180,
                                  "remaining_won": 20, "recipients": 1}
    assert "P010-1" not in result


def test_p010_requires_independent_entitlement_and_complete_prepolicy_days():
    on, off = _pair()
    with pytest.raises(ValueError, match="issued amount mismatch"):
        score_coupon(on, off, roster=["a"], draws=0,
                     expected_issued_won=201)
    with pytest.raises(ValueError, match="incomplete"):
        score_coupon(on[1:], off, roster=["a"], draws=0,
                     expected_issued_won=200)


def test_p010_cli_rejects_manifest_from_modified_policy(tmp_path, monkeypatch):
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps(["a"]), encoding="utf-8")
    monkeypatch.setattr(coupon, "verify_manifests", lambda *_args, **_kwargs: {
        "policy_file_sha256": "0" * 64,
    })
    output = tmp_path / "score.json"
    monkeypatch.setattr(sys, "argv", ["paired_coupon_effect.py", "--on", str(tmp_path / "on"),
                                     "--off", str(tmp_path / "off"), "--roster", str(roster),
                                     "--expected-issued-won", "200", "--json-out", str(output)])
    with pytest.raises(ValueError, match="frozen P010 policy"):
        coupon.main()
    assert not output.exists()
