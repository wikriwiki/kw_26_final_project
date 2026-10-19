"""scripts/persona/reconcile.py — 모순 검출 + 봉합 (방식 C)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "persona"))
import reconcile as R  # noqa: E402


def _persona(level: int, hobbies=None, job="회사원") -> dict:
    return {
        "personal": {"job": job, "income_level": "중"},
        "spending": {"weekday_spending_level": level, "weekend_spending_level": level,
                     "weekday_top_categories": {}, "weekend_top_categories": {}},
        "behavior": {},
        "personality": {"spending_tendency": "보통"},
        "nvidia_persona": {"hobbies": hobbies or []},
        "_match": {},
    }


# ---------------------------------------------------------------------------
# _level_to_unit
# ---------------------------------------------------------------------------
def test_level_to_unit():
    assert R._level_to_unit(1) == 0.0
    assert R._level_to_unit(10) == 1.0
    assert abs(R._level_to_unit(5) - 4 / 9) < 1e-6


# ---------------------------------------------------------------------------
# check_consistency
# ---------------------------------------------------------------------------
def test_no_warning_when_aligned():
    # 분위 8 (unit 0.78) ↔ SES 0.75 → gap 작음
    w = R.check_consistency(_persona(8), ses=0.75)
    assert w == []


def test_ses_consume_gap_high_spend_low_ses():
    # 분위 10 (unit 1.0) ↔ SES 0.2 → gap 0.8
    w = R.check_consistency(_persona(10), ses=0.2)
    assert any("ses_consume_gap:고소비_저SES" in x for x in w)


def test_ses_consume_gap_low_spend_high_ses():
    # 분위 1 (unit 0.0) ↔ SES 0.9 → gap -0.9
    w = R.check_consistency(_persona(1, job="의사"), ses=0.9)
    assert any("ses_consume_gap:저소비_고SES" in x for x in w)


def test_luxury_hobby_low_spend():
    w = R.check_consistency(_persona(2, hobbies=["주말 골프", "와인 시음"]), ses=0.5)
    assert "luxury_hobby_low_spend" in w


def test_frugal_hobby_high_spend():
    w = R.check_consistency(_persona(10, hobbies=["동네 산책", "공원 나들이", "도서관"]), ses=0.5)
    assert "frugal_hobby_high_spend" in w


def test_high_ses_job_low_spend_excludes_jobseeker():
    # 구직중이면 고SES-저소비 모순으로 안 잡음
    w = R.check_consistency(_persona(1, job="전직 의사, 현재 구직중"), ses=0.9)
    assert "high_ses_job_low_spend" not in w


def test_high_ses_job_low_spend_flags_employed():
    w = R.check_consistency(_persona(1, job="현직 변호사"), ses=0.9)
    assert "high_ses_job_low_spend" in w


# ---------------------------------------------------------------------------
# reconcile_spending (build_quant 의존 — mock profile/deciles)
# ---------------------------------------------------------------------------
_MOCK_PROFILE = {
    "consumption": {"industry_ratio": {"한식": 0.5, "커피전문점": 0.5},
                    "weekday_spending_level": 5, "weekend_spending_level": 5},
    "mobility": {"mobility_level": 5},
    "telecom": {},
}
_MOCK_DECILES = {
    "weekday_spending_level": {"boundaries": [{"decile": d, "min": d * 10000, "max": (d + 1) * 10000} for d in range(1, 11)]},
    "weekend_spending_level": {"boundaries": [{"decile": d, "min": d * 10000, "max": (d + 1) * 10000} for d in range(1, 11)]},
}


def test_reconcile_pulls_low_spend_toward_high_ses():
    p = _persona(2, job="의사")   # 분위 2, 고SES
    rng = random.Random(0)
    out = R.reconcile_spending(p, ses=0.9, profile=_MOCK_PROFILE,
                               deciles=_MOCK_DECILES, rng=rng)
    # 분위가 SES(목표≈9) 방향으로 당겨짐 (최대 +2 → 4)
    assert out["spending"]["weekday_spending_level"] >= 4
    assert out["_match"]["reconciled"] is True


def test_reconcile_no_pull_when_aligned():
    p = _persona(7, job="회사원")   # 분위 7 (unit 0.67) ↔ SES 0.7 → gap 작음
    rng = random.Random(0)
    out = R.reconcile_spending(p, ses=0.7, profile=_MOCK_PROFILE,
                               deciles=_MOCK_DECILES, rng=rng)
    assert out["_match"]["reconciled"] is False
    assert out["spending"]["weekday_spending_level"] == 7


def test_reconcile_records_warnings_after_pull():
    # gap 이 max_pull(2) 로 다 못 메우는 극단 케이스 → 잔여 경고 기록
    p = _persona(1, job="변호사")   # 분위1, SES 0.95 → 목표 ~10, +2 해도 3
    rng = random.Random(0)
    out = R.reconcile_spending(p, ses=0.95, profile=_MOCK_PROFILE,
                               deciles=_MOCK_DECILES, rng=rng)
    assert out["_match"]["reconciled"] is True
    # 분위 3 (unit 0.22) ↔ SES 0.95 → gap 0.73 여전 > 0.4 → 경고 남음
    assert len(out["_match"]["warnings"]) >= 1
