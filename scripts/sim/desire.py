"""
desire.py
==========
POI 방문 후보 정렬 점수 — avg_satisfaction 기반, 없으면 거리 기반.

이전의 affinity·recency·saturation·novelty 복잡도를 제거하고
Stage 2 LLM 프롬프트의 페르소나 기반 선택으로 다양성을 유도한다.

정렬 점수:
  - KNOWS_POI(avg_satisfaction)가 있으면 그 값 사용 (0~1)
  - 처음 방문 or 기록 없으면 0.0 → 거리가 짧은 순으로 최종 정렬
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DesireInputs:
    """한 (agent, POI) 쌍의 정렬 점수 계산에 필요한 입력."""
    avg_satisfaction: float | None   # KNOWS_POI.avg_satisfaction, 없으면 None
    km: float | None                 # 거리(km), 없으면 None


def compute_desire(d: DesireInputs) -> float:
    """정렬 점수.
    avg_satisfaction이 있으면 그 값, 없으면 0.0.
    최종 정렬은 desire 내림차순 → km 오름차순으로 호출자가 처리.
    """
    if d.avg_satisfaction is not None:
        return float(d.avg_satisfaction)
    return 0.0


def inputs_from_candidate_row(row: dict, days_since_visit: float | None = None) -> DesireInputs:
    """Cypher RETURN 행 → DesireInputs. days_since_visit 파라미터는 하위호환용(무시)."""
    return DesireInputs(
        avg_satisfaction=row.get("avg_satisfaction"),
        km=row.get("km"),
    )
