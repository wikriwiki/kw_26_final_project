"""scripts/sim/desire.py — pure desire 함수 검증.

핵심 보장:
  1. monotonicity (Δ↑/v30↑/affinity↑)
  2. 경계 (Δ=None, sat=None, affinity=0)
  3. 카테고리 파라미터 효과 (식사 tau=3 vs 미용 tau=30)
  4. 안 가본 곳이 단골(어제 방문)보다 desire 가 높을 수 있음 (다양성 유도)
"""

from __future__ import annotations

import math

import pytest

from scripts.sim.desire import (
    DesireInputs,
    compute_desire,
    inputs_from_candidate_row,
    FALLBACK_CAT_TAU,
    FALLBACK_CAT_DROP,
    FALLBACK_CAT_SATURATION_N,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _inputs(**overrides) -> DesireInputs:
    """기본 식사 카테고리 + 미인지 단골 — 변수 일부 override."""
    base = dict(
        affinity=0.0, avg_satisfaction=None,
        days_since_visit=None, visits_in_last_30d=0,
        cat_tau=3.0, cat_drop=0.85, cat_saturation_n=12,
    )
    base.update(overrides)
    return DesireInputs(**base)


# ---------------------------------------------------------------------------
# 1. 미방문(novelty) baseline
# ---------------------------------------------------------------------------
def test_never_visited_gets_novelty_bonus():
    d = compute_desire(_inputs())
    # baseline(affinity=0, sat=None=0.5) = 0.3 + 0.7*(0.6*0 + 0.4*0.5) = 0.44
    # recency = 1.0, saturation ≈ 1.0 (v30=0), novelty = 0.15
    assert d == pytest.approx(0.44 + 0.15, abs=0.02)


def test_never_visited_with_high_affinity_higher_baseline():
    """미인지 가게의 affinity 는 0 이라 보통 baseline=0.44.
    affinity 가 양수면(예: rumor 로 인지) baseline 도 올라감."""
    low = compute_desire(_inputs(affinity=0.0))
    high = compute_desire(_inputs(affinity=0.8))
    assert high > low


# ---------------------------------------------------------------------------
# 2. recency 단조성 — Δ ↑ → desire ↑
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("days,expected_recency", [
    (0, 1.0 - 0.85),                          # 직후: 0.15
    (3, 1.0 - 0.85 * math.exp(-1)),           # tau=3: ~0.69
    (7, 1.0 - 0.85 * math.exp(-7/3)),         # ~0.92
    (30, 1.0 - 0.85 * math.exp(-10)),         # 거의 완전 회복
])
def test_recency_monotone_in_days(days, expected_recency):
    # 식사: tau=3, drop=0.85
    d = _inputs(
        affinity=0.8, avg_satisfaction=0.7,
        days_since_visit=days, visits_in_last_30d=1,
    )
    # baseline = 0.3 + 0.7*(0.6*0.8 + 0.4*0.571) = 0.3 + 0.7*(0.48+0.229) = 0.3+0.496 = 0.796
    # saturation(v30=1, sat_n=12): 시그모이드 (1-12)/2 = -5.5 → ≈ 0.996
    expected = 0.796 * expected_recency * 0.996
    assert compute_desire(d) == pytest.approx(expected, abs=0.05)


def test_recency_strictly_increasing():
    """Δ 가 증가하면 desire 단조 증가 (다른 변수 고정)."""
    prev = -1
    for d_since in [1, 2, 3, 5, 7, 10, 14, 30]:
        v = compute_desire(_inputs(
            affinity=0.7, avg_satisfaction=0.7,
            days_since_visit=d_since, visits_in_last_30d=2,
        ))
        assert v > prev, f"Δ={d_since}: desire {v} should be > prev {prev}"
        prev = v


# ---------------------------------------------------------------------------
# 3. saturation 단조 감소 — v30 ↑ → desire ↓
# ---------------------------------------------------------------------------
def test_saturation_decreases_with_v30():
    desires = []
    for v30 in [0, 5, 10, 15, 20]:
        d = compute_desire(_inputs(
            affinity=0.7, avg_satisfaction=0.7,
            days_since_visit=14, visits_in_last_30d=v30,
            cat_saturation_n=10,
        ))
        desires.append(d)
    assert desires == sorted(desires, reverse=True)


def test_saturation_half_at_threshold():
    """v30 == sat_n 이면 saturation = 0.5."""
    d_above = compute_desire(_inputs(
        affinity=0.5, avg_satisfaction=0.7,
        days_since_visit=30, visits_in_last_30d=0,
        cat_saturation_n=10,
    ))
    d_at = compute_desire(_inputs(
        affinity=0.5, avg_satisfaction=0.7,
        days_since_visit=30, visits_in_last_30d=10,
        cat_saturation_n=10,
    ))
    # at threshold 는 baseline*recency*0.5, above 는 baseline*recency*~1.0
    # 비율 ≈ 0.5
    assert d_at / d_above == pytest.approx(0.5, abs=0.05)


# ---------------------------------------------------------------------------
# 4. 다양성 유도 — 어제 방문 단골 < 미방문 (sane case)
# ---------------------------------------------------------------------------
def test_novelty_beats_recently_visited_regular():
    """단골이라도 어제 갔으면 새 가게보다 desire 가 낮을 수 있다."""
    visited_yesterday = compute_desire(_inputs(
        affinity=0.7, avg_satisfaction=0.7,
        days_since_visit=1, visits_in_last_30d=4,
        cat_tau=3.0, cat_drop=0.85,
    ))
    new_place = compute_desire(_inputs(
        affinity=0.0, avg_satisfaction=None,
        days_since_visit=None, visits_in_last_30d=0,
        cat_tau=3.0, cat_drop=0.85,
    ))
    assert new_place > visited_yesterday


# ---------------------------------------------------------------------------
# 5. 카테고리별 효과 — 식사(tau=3) vs 미용(tau=30) 같은 Δ=7일 후
# ---------------------------------------------------------------------------
def test_long_tau_means_slower_recovery():
    food = compute_desire(_inputs(
        affinity=0.7, avg_satisfaction=0.7, days_since_visit=7,
        visits_in_last_30d=2, cat_tau=3.0, cat_drop=0.85, cat_saturation_n=12,
    ))
    beauty = compute_desire(_inputs(
        affinity=0.7, avg_satisfaction=0.7, days_since_visit=7,
        visits_in_last_30d=2, cat_tau=30.0, cat_drop=0.95, cat_saturation_n=2,
    ))
    # 같은 7일 전 방문이지만 미용은 tau=30 이라 recency 가 훨씬 낮음
    assert food > beauty


# ---------------------------------------------------------------------------
# 6. 경계값 — None / 0 / 비정상
# ---------------------------------------------------------------------------
def test_sat_none_uses_neutral_05():
    """avg_satisfaction=None 이면 sat_norm=0.5."""
    d_none = compute_desire(_inputs(
        affinity=0.5, avg_satisfaction=None, days_since_visit=10,
    ))
    d_neutral = compute_desire(_inputs(
        affinity=0.5, avg_satisfaction=0.65, days_since_visit=10,   # (0.65-0.3)/0.7 = 0.5
    ))
    assert d_none == pytest.approx(d_neutral, abs=0.005)


def test_zero_affinity_zero_sat():
    """모든 입력 최저값 — 음수 안 나오고 finite."""
    d = compute_desire(_inputs(
        affinity=0.0, avg_satisfaction=0.0, days_since_visit=0,
        visits_in_last_30d=100, cat_saturation_n=2,
    ))
    assert 0.0 <= d < 1.0


def test_tau_zero_safe():
    """비정상 tau=0 도 NaN 안 남기고 safe (recency=1)."""
    d = compute_desire(_inputs(
        affinity=0.5, avg_satisfaction=0.5,
        days_since_visit=5, cat_tau=0.0,
    ))
    assert math.isfinite(d) and d > 0


def test_drop_clamped_to_unit_interval():
    """drop>1 같은 잘못된 입력 clamp."""
    d = compute_desire(_inputs(
        affinity=0.5, avg_satisfaction=0.5,
        days_since_visit=0, cat_drop=2.0,
    ))
    # drop clamp → 1.0, recency = 1 - 1*1 = 0
    assert d == pytest.approx(0.0, abs=0.01)


def test_saturation_no_overflow_on_extreme_v30():
    """비정상적으로 큰 v30 (예: 같은 가게 1만 번 방문) 가 들어와도 OverflowError 안 남."""
    d = compute_desire(_inputs(
        affinity=0.8, avg_satisfaction=0.8,
        days_since_visit=14, visits_in_last_30d=10000, cat_saturation_n=2,
    ))
    assert math.isfinite(d)
    # saturation ≈ 0 이라 desire 는 novelty(0)+baseline*recency*0 ≈ 0
    assert d == pytest.approx(0.0, abs=0.01)


def test_recency_no_overflow_on_extreme_days():
    """매우 큰 days_since 도 OverflowError 안 남 (exp(-inf) = 0)."""
    d = compute_desire(_inputs(
        affinity=0.5, avg_satisfaction=0.5,
        days_since_visit=10000,
    ))
    assert math.isfinite(d)
    # recency ≈ 1 (완전 회복)
    assert d > 0.3


# ---------------------------------------------------------------------------
# 7. inputs_from_candidate_row — Cypher RETURN dict 변환
# ---------------------------------------------------------------------------
def test_inputs_from_row_full():
    row = {
        "affinity": 0.7,
        "avg_satisfaction": 0.6,
        "v30": 3,
        "cat_tau": 5.0,
        "cat_drop": 0.8,
        "cat_sat_n": 8,
    }
    d = inputs_from_candidate_row(row, days_since_visit=4.0)
    assert d.affinity == 0.7
    assert d.avg_satisfaction == 0.6
    assert d.visits_in_last_30d == 3
    assert d.cat_tau == 5.0
    assert d.days_since_visit == 4.0


def test_inputs_from_row_missing_uses_fallbacks():
    """그래프 backfill 전이라 cat_tau 등이 None 인 케이스."""
    row = {"affinity": None, "cat_tau": None, "cat_drop": None, "cat_sat_n": None}
    d = inputs_from_candidate_row(row, days_since_visit=None)
    assert d.affinity == 0.0
    assert d.cat_tau == FALLBACK_CAT_TAU
    assert d.cat_drop == FALLBACK_CAT_DROP
    assert d.cat_saturation_n == FALLBACK_CAT_SATURATION_N
    assert d.days_since_visit is None
