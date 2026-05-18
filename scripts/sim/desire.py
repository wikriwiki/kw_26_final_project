"""
desire.py
==========
POI 방문 욕구(desire) 점수 함수 — Stage 2 후보 정렬·LLM 노출에 사용.

핵심 아이디어 (사용자 직관):
  방문 직후 욕구는 훅 떨어졌다가 시간이 지나며 서서히 회복.
  단골이라도 어제 갔으면 오늘 가고 싶지 않은 게 자연스럽다.

수식 (high-order — 4개 요인 곱셈):
  desire = baseline(affinity, sat) * recency(Δ) * saturation(v30) + novelty

  baseline = 0.3 + 0.7 * (0.6·affinity + 0.4·sat_norm)
    affinity ∈ [0,1]  KNOWS_POI.affinity (미인지 → 0)
    sat_norm = clip((avg_sat - 0.3) / 0.7, 0, 1) | 0.5 (없으면)

  recency(Δ) = max(0, 1 - drop · exp(-Δ/tau))     ← 카테고리별 drop, tau
    Δ = days_since_visit (None → 1.0 즉 미방문은 페널티 없음)

  saturation(v30) = 1 / (1 + exp((v30 - sat_n) / 2))   ← 시그모이드
    30일 안에 sat_n 회 방문 시 0.5, 그 이상이면 빠르게 ↓

  novelty = 0.15 if 미방문 else 0     ← 다양성 보너스

설계 결정:
  - pure 함수, Neo4j 의존성 없음. 단위 테스트만으로 모든 경로 검증.
  - 카테고리 파라미터는 호출자가 주입 (Cypher RETURN 으로 같이 가져옴).
  - 절대값보다 *상대 순서* 가 LLM 결정에 영향 — 정확한 캘리브레이션은 시뮬 결과 보고.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 안전 fallback — Category 노드에 desire 파라미터 누락 시 (그래프 backfill 전)
# ---------------------------------------------------------------------------
FALLBACK_CAT_TAU = 7.0
FALLBACK_CAT_DROP = 0.75
FALLBACK_CAT_SATURATION_N = 6.0


@dataclass(frozen=True)
class DesireInputs:
    """한 (agent, POI) 쌍의 desire 계산에 필요한 모든 입력."""
    affinity: float                       # KNOWS_POI.affinity, 미인지면 0.0
    avg_satisfaction: float | None        # KNOWS_POI.avg_satisfaction, 미인지면 None
    days_since_visit: float | None        # 미방문 = None
    visits_in_last_30d: int               # 0 if 미인지
    cat_tau: float                        # Category.recovery_tau_days
    cat_drop: float                       # Category.desire_drop, [0,1]
    cat_saturation_n: float               # Category.saturation_n


def _baseline(affinity: float, avg_sat: float | None) -> float:
    """친밀도·만족도 가중평균. 미인지면 affinity=0, avg_sat=None 으로 baseline=0.44."""
    if avg_sat is not None:
        sat_norm = max(0.0, min(1.0, (avg_sat - 0.3) / 0.7))
    else:
        sat_norm = 0.5
    return 0.3 + 0.7 * (0.6 * max(0.0, min(1.0, affinity)) + 0.4 * sat_norm)


def _recency(days_since: float | None, tau: float, drop: float) -> float:
    """시간 회복 (지수). days_since=None → 1.0 (페널티 없음)."""
    if days_since is None:
        return 1.0
    if tau <= 0:
        return 1.0
    drop = max(0.0, min(1.0, drop))
    return max(0.0, 1.0 - drop * math.exp(-max(0.0, days_since) / tau))


def _saturation(v30: int, sat_n: float) -> float:
    """시그모이드 — v30=sat_n 에서 0.5, v30 ≫ sat_n 이면 빠르게 0."""
    if sat_n <= 0:
        # 비정상 입력 — 페널티 없음
        return 1.0
    return 1.0 / (1.0 + math.exp((v30 - sat_n) / 2.0))


def _novelty(days_since: float | None) -> float:
    return 0.15 if days_since is None else 0.0


def compute_desire(d: DesireInputs) -> float:
    """방문 욕구 점수. 단조성:
      - days_since_visit ↑    → desire ↑  (다른 변수 고정)
      - visits_in_last_30d ↑  → desire ↓
      - affinity ↑            → desire ↑
      - 미방문 (None) 은 recency=1 + novelty 보너스 → 새 가게 다양성 유도
    """
    baseline = _baseline(d.affinity, d.avg_satisfaction)
    recency = _recency(d.days_since_visit, d.cat_tau, d.cat_drop)
    saturation = _saturation(d.visits_in_last_30d, d.cat_saturation_n)
    novelty = _novelty(d.days_since_visit)
    return baseline * recency * saturation + novelty


# ---------------------------------------------------------------------------
# Cypher RETURN row → DesireInputs 변환 헬퍼
# ---------------------------------------------------------------------------
def inputs_from_candidate_row(
    row: dict,
    days_since_visit: float | None,
) -> DesireInputs:
    """fetch_candidates_for_events 의 한 cand dict 를 DesireInputs 로.

    cand 는 dawn_context.STAGE2_CANDIDATE_CYPHER 의 RETURN 행 구조를 따름.
    days_since_visit 은 Python date 차이로 계산해서 주입 (visit_window.days_since_last).
    """
    return DesireInputs(
        affinity=float(row.get("affinity") or 0.0),
        avg_satisfaction=row.get("avg_satisfaction"),
        days_since_visit=days_since_visit,
        visits_in_last_30d=int(row.get("v30") or 0),
        cat_tau=float(row.get("cat_tau") or FALLBACK_CAT_TAU),
        cat_drop=float(row.get("cat_drop") or FALLBACK_CAT_DROP),
        cat_saturation_n=float(row.get("cat_sat_n") or FALLBACK_CAT_SATURATION_N),
    )
