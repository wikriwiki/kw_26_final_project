"""광역 목적지 선택 — Huff 소매 중력모형 prior (Problem A).

순수 함수 모듈 (Neo4j·LLM 비의존, 단위 테스트 가능). Stage1 프롬프트에 주입할
"오늘 끌리는 광역 상권 후보"를 통계적으로 선별한다. LLM은 이 후보 중에서 reasoning으로
채택/기각 → 통계 prior(쏠림·환각 억제) + LLM 판단의 하이브리드(AgentMove 철학).

Huff(1964): P(목적지 j) = (A_j / d_ij^β) / Σ_k (A_k / d_ik^β)
  A_j = 매력도(b069_sales, hub_catalog),  d_ij = 거주지→j 거리(km),  β = 거리 민감도.

거리 민감도 β는 요일·이동분위로 동적 조정:
  - 평일 ↑ (생활권 집중) / 주말 ↓ (광역 이동 허용)
  - mobility_level 높음 ↓ (잘 돌아다니는 사람)
"""
from __future__ import annotations

import math
import random

BASE_BETA = 1.8          # 소매 중력모형 통상 β(1.5~2.5) 중앙
DIST_FLOOR_KM = 0.3      # 거주 동 자기 자신(거리≈0) 무한가중 방지
_DAY_FACTOR = {"weekday": 1.35, "weekend": 0.85}


def effective_beta(day_type: str, mobility_level: int | None, base_beta: float = BASE_BETA) -> float:
    """요일·이동분위 반영 거리 민감도.

    mobility_level 1(저)~10(고): 1→1.25배, 10→0.75배 (잘 돌아다닐수록 거리 둔감).
    """
    day_f = _DAY_FACTOR.get(day_type, 1.0)
    lv = mobility_level if isinstance(mobility_level, (int, float)) and mobility_level else 5
    lv = min(10, max(1, int(lv)))
    mob_f = 1.25 - (lv - 1) / 9.0 * 0.5      # 1.25 .. 0.75
    return max(0.6, min(3.5, base_beta * day_f * mob_f))


def huff_probabilities(
    attractions: list[float],
    distances_km: list[float],
    beta: float,
    dist_floor_km: float = DIST_FLOOR_KM,
) -> list[float]:
    """Huff 확률 벡터. 길이·합(=1) 보장. 입력 비면 빈 리스트."""
    n = len(attractions)
    if n == 0 or len(distances_km) != n:
        return []
    weights = []
    for a, d in zip(attractions, distances_km):
        a = max(0.0, float(a or 0.0))
        d = max(dist_floor_km, float(d if d is not None else dist_floor_km))
        weights.append(a / (d ** beta))
    total = sum(weights)
    if total <= 0:
        return [1.0 / n] * n
    return [w / total for w in weights]


def rank_destinations(
    hubs: list[dict],
    day_type: str,
    mobility_level: int | None,
    base_beta: float = BASE_BETA,
) -> list[dict]:
    """후보 hub(각 dict에 'attraction','distance_km')에 Huff 확률을 부여해 내림차순 반환.

    원본을 변형하지 않고 'prob' 키를 추가한 얕은 복사본 리스트를 돌려준다.
    """
    if not hubs:
        return []
    beta = effective_beta(day_type, mobility_level, base_beta)
    probs = huff_probabilities(
        [h.get("attraction", 0.0) for h in hubs],
        [h.get("distance_km") for h in hubs],
        beta,
    )
    out = [{**h, "prob": round(p, 5)} for h, p in zip(hubs, probs)]
    out.sort(key=lambda h: h["prob"], reverse=True)
    return out


def sample_destinations(
    hubs: list[dict],
    day_type: str,
    mobility_level: int | None,
    k: int,
    rng: random.Random | None = None,
    base_beta: float = BASE_BETA,
) -> list[dict]:
    """Huff 확률 기반 비복원 가중 추출 k개. 결과는 prob 내림차순.

    상위 확률(생활권)이 거의 항상 뽑히되, 매력적인 원거리 허브도 확률만큼 간헐적으로
    표면화 → 주말 광역 이동의 변동성을 통계적으로 보장.
    """
    ranked = rank_destinations(hubs, day_type, mobility_level, base_beta)
    if k >= len(ranked):
        return ranked
    rng = rng or random.Random()
    pool = list(ranked)
    chosen: list[dict] = []
    for _ in range(k):
        total = sum(h["prob"] for h in pool)
        if total <= 0:
            chosen.extend(pool[: k - len(chosen)])
            break
        r = rng.random() * total
        acc = 0.0
        for idx, h in enumerate(pool):
            acc += h["prob"]
            if r <= acc:
                chosen.append(pool.pop(idx))
                break
    chosen.sort(key=lambda h: h["prob"], reverse=True)
    return chosen


# =========================================================
# 자체 테스트
# =========================================================
if __name__ == "__main__":
    # 시나리오: 생활권(가까움·중간매력) + 강남(멀고 고매력) + 동네상권(가깝고 저매력)
    hubs = [
        {"code": "11110515", "name": "거주동",   "attraction": 20, "distance_km": 0.2},
        {"code": "11680640", "name": "강남역",   "attraction": 55, "distance_km": 12.0},
        {"code": "11440700", "name": "홍대",     "attraction": 45, "distance_km": 8.0},
        {"code": "11110530", "name": "옆동네",   "attraction": 10, "distance_km": 1.5},
    ]
    print("=== 평일 (생활권 집중 기대) ===")
    for h in rank_destinations(hubs, "weekday", mobility_level=5):
        print(f"  {h['name']:6} prob={h['prob']:.3f}  (β={effective_beta('weekday',5):.2f})")
    print("=== 주말 (광역 이동 허용 기대) ===")
    for h in rank_destinations(hubs, "weekend", mobility_level=8):
        print(f"  {h['name']:6} prob={h['prob']:.3f}  (β={effective_beta('weekend',8):.2f})")

    # 표본 추출 분포 확인 (주말, k=2, 2000회)
    from collections import Counter
    rng = random.Random(0)
    cnt = Counter()
    for _ in range(2000):
        for h in sample_destinations(hubs, "weekend", 8, k=2, rng=rng):
            cnt[h["name"]] += 1
    print("=== 주말 k=2 표본 2000회 선택 빈도 ===")
    for name, c in cnt.most_common():
        print(f"  {name:6} {c}")
