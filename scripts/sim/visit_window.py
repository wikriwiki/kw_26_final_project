"""
visit_window.py
================
KNOWS_POI.recent_visit_dates 30일 슬라이딩 윈도우 — Cypher 와 등가인 Python 헬퍼.

`plan_writer.NIGHT_VISITED_CYPHER` 의 ON MATCH 절:

    kp.recent_visit_dates =
      [d IN coalesce(kp.recent_visit_dates, [])
       WHERE duration.inDays(d, date($yesterday)).days < 30]
      + [date($yesterday)]

이 함수는 동일한 의미를 Python datetime.date 로 표현한다. 단위 테스트가
"이 동작이 Cypher 에서 일어난다" 를 명세하고, backfill 이나 desire 계산에서
in-memory 로도 재사용 가능.

윈도우 30일을 카테고리별 가변으로 바꾸고 싶으면 window_days 인자를 카테고리
saturation_n 과 별도로 두면 됨 — 현재는 전역 상수.
"""

from __future__ import annotations

from datetime import date, timedelta


DEFAULT_WINDOW_DAYS = 30


def trim_and_push_visit(
    dates: list[date],
    visit_date: date,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> list[date]:
    """과거 window_days 일 안 방문 날짜만 남기고 visit_date 추가.

    경계 정의 (Cypher 와 등가):
      - d > visit_date - window_days  → 유지  (즉 정확히 window_days 일 전 = drop)
      - visit_date 는 항상 append (같은 날 두 번 호출 시 중복 들어감)
    """
    cutoff = visit_date - timedelta(days=window_days)
    kept = [d for d in dates if d > cutoff]
    return kept + [visit_date]


def count_recent_visits(
    dates: list[date],
    today: date,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> int:
    """today 기준 window_days 일 안 방문 횟수.

    nightly trim 후 며칠 지난 시점에 조회될 수 있어 cutoff 재계산이 안전.
    동시에 같은 날 여러 visit 가 들어있으면 모두 카운트 (중복 허용).
    """
    cutoff = today - timedelta(days=window_days)
    return sum(1 for d in dates if d > cutoff)


def days_since_last(
    dates: list[date],
    today: date,
) -> int | None:
    """가장 최근 방문일부터 today 까지 일수. dates 가 비면 None."""
    if not dates:
        return None
    return (today - max(dates)).days
