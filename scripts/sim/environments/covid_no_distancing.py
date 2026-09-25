"""Counterfactual environment for a paired distancing experiment.

The disease observations are identical to the restricted arm. Only the
government's additional venue and meeting restrictions are absent. This is
an experimental condition, not a description of the historical day.
"""
from __future__ import annotations

from datetime import date

from .covid_2021 import _regime_for, disease_facts


def build(day: date) -> dict:
    if _regime_for(day) is None:
        return {}
    return {
        "headline": "서울 코로나19 유행 — 추가 거리두기 제한 없는 비교 조건",
        "facts": disease_facts(day) + [
            "식당·카페와 다른 매장의 추가 방역 영업시간·매장 이용 제한 없음. "
            "각 매장의 고유 운영시간은 유지",
            "추가 방역 사적모임 인원 제한 없음",
        ],
        "note": None,
    }
