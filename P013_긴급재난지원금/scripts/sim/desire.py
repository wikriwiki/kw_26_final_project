"""
desire.py — DEPRECATED
=======================
이전: affinity·recency·saturation·novelty 4요인 곱셈으로 desire 점수 계산.
현재: 폐기. Stage 2 LLM이 페르소나 + 후보 정렬(avg_satisfaction → km) 기반으로 직접 선택.

stage2_poi.py의 _score_and_sort_by_desire가 인라인으로 처리하므로 이 모듈은 사용되지 않음.
하위호환을 위해 시그니처만 남겨두며, 새 코드에서 사용 금지.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DesireInputs:
    """DEPRECATED. avg_satisfaction과 km만 필드로 유지."""
    avg_satisfaction: float | None
    km: float | None


def compute_desire(d: DesireInputs) -> float:
    """DEPRECATED. avg_satisfaction 있으면 그 값, 없으면 0.0.

    실제 정렬은 stage2_poi._score_and_sort_by_desire가 인라인으로 처리.
    """
    if d.avg_satisfaction is not None:
        return float(d.avg_satisfaction)
    return 0.0


def inputs_from_candidate_row(row: dict, days_since_visit: float | None = None) -> DesireInputs:
    """DEPRECATED. days_since_visit은 무시."""
    return DesireInputs(
        avg_satisfaction=row.get("avg_satisfaction"),
        km=row.get("km"),
    )
