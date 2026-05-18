"""KNOWS_POI.recent_visit_dates 30일 슬라이딩 윈도우 — Python 헬퍼 검증.

plan_writer.NIGHT_VISITED_CYPHER 의 동작과 등가성 보장을 단위 테스트로 명세.
"""

from datetime import date, timedelta

from scripts.sim.visit_window import (
    DEFAULT_WINDOW_DAYS,
    count_recent_visits,
    days_since_last,
    trim_and_push_visit,
)


# ---------------------------------------------------------------------------
# trim_and_push_visit
# ---------------------------------------------------------------------------
def test_push_to_empty_list():
    today = date(2026, 5, 15)
    result = trim_and_push_visit([], today)
    assert result == [today]


def test_push_keeps_recent_dates():
    today = date(2026, 5, 15)
    existing = [date(2026, 5, 10), date(2026, 5, 1)]    # 5일 / 14일 전
    result = trim_and_push_visit(existing, today)
    assert result == [date(2026, 5, 10), date(2026, 5, 1), today]


def test_push_drops_old_dates():
    today = date(2026, 5, 15)
    existing = [
        date(2026, 5, 14),    # 1일 전 — 유지
        date(2026, 4, 16),    # 29일 전 — 유지 (< 30일)
        date(2026, 4, 15),    # 30일 전 — 정확히 경계 → drop (d > cutoff 엄격)
        date(2026, 4, 1),     # 44일 전 — drop
    ]
    result = trim_and_push_visit(existing, today)
    assert date(2026, 5, 14) in result
    assert date(2026, 4, 16) in result
    assert date(2026, 4, 15) not in result   # 30일째는 drop
    assert date(2026, 4, 1) not in result
    assert result[-1] == today               # 항상 마지막에 append


def test_push_appends_duplicate_same_day():
    """같은 날 두 번 visited 가 들어오면 중복 허용 (Cypher 와 등가)."""
    today = date(2026, 5, 15)
    existing = [today]
    result = trim_and_push_visit(existing, today)
    assert result == [today, today]


def test_custom_window():
    today = date(2026, 5, 15)
    existing = [date(2026, 5, 8)]    # 7일 전
    # window=5 → drop, window=10 → keep
    assert trim_and_push_visit(existing, today, window_days=5) == [today]
    assert trim_and_push_visit(existing, today, window_days=10) == [date(2026, 5, 8), today]


# ---------------------------------------------------------------------------
# count_recent_visits
# ---------------------------------------------------------------------------
def test_count_basic():
    today = date(2026, 5, 15)
    dates = [date(2026, 5, 14), date(2026, 5, 10), date(2026, 4, 1)]
    # 1일 / 5일 / 44일 전 → 30일 안엔 2개
    assert count_recent_visits(dates, today) == 2


def test_count_empty():
    assert count_recent_visits([], date(2026, 5, 15)) == 0


def test_count_boundary_excludes_30days_old():
    today = date(2026, 5, 15)
    dates = [
        date(2026, 4, 16),    # 29일 전 — 카운트
        date(2026, 4, 15),    # 30일 전 — 제외
    ]
    assert count_recent_visits(dates, today) == 1


def test_count_includes_today():
    today = date(2026, 5, 15)
    dates = [today, today]    # 같은 날 중복
    assert count_recent_visits(dates, today) == 2


def test_count_default_window_matches_constant():
    assert DEFAULT_WINDOW_DAYS == 30


# ---------------------------------------------------------------------------
# days_since_last
# ---------------------------------------------------------------------------
def test_days_since_last_basic():
    today = date(2026, 5, 15)
    dates = [date(2026, 5, 10), date(2026, 5, 14), date(2026, 4, 1)]
    assert days_since_last(dates, today) == 1   # max = 5/14


def test_days_since_last_empty():
    assert days_since_last([], date(2026, 5, 15)) is None


def test_days_since_last_zero_today():
    today = date(2026, 5, 15)
    assert days_since_last([today], today) == 0


# ---------------------------------------------------------------------------
# 통합 — Cypher equivalence sanity (sequence of nightly updates)
# ---------------------------------------------------------------------------
def test_sequential_nightly_updates_simulating_60_days():
    """매일 같은 POI 방문이라 가정. 60일 후 윈도우엔 정확히 30개 (30일 전 ~ 어제)."""
    visits: list[date] = []
    start = date(2026, 5, 1)
    for d_off in range(60):
        day = start + timedelta(days=d_off)
        visits = trim_and_push_visit(visits, day)

    # 60일째 nightly update 직후. 윈도우엔 (5/1 + 30 ~ 5/1 + 59) = 30개
    # cutoff = day(60) - 30 = day(30). day > cutoff 인 d 들 + 오늘.
    today_at_60 = start + timedelta(days=59)
    assert len(visits) == 30
    assert min(visits) > today_at_60 - timedelta(days=30)
    assert max(visits) == today_at_60
