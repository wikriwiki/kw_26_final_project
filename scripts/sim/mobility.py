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

import io
import json
import math
import random
from pathlib import Path

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
# 거리 + 카탈로그 로딩 + 허브 추천 (런타임 Stage1 후보 zone)
# =========================================================
_STATS = Path(__file__).resolve().parents[2] / "output" / "stats"
_CACHE: dict = {}


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """두 좌표(경도, 위도) 사이 대권거리(km)."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.asin(math.sqrt(a))


def _load(stats_dir: Path | None = None) -> dict:
    """hub_catalog + dong_centroids 로드(캐시). 파일 없으면 빈 구조."""
    sd = stats_dir or _STATS
    key = str(sd)
    if key in _CACHE:
        return _CACHE[key]
    data = {"top_hubs": [], "centroids": {}, "gu_fallback": {}}
    try:
        cat = json.load(io.open(sd / "hub_catalog.json", encoding="utf-8"))
        data["top_hubs"] = [h for h in cat.get("hubs", []) if h.get("is_top_hub")]
    except Exception:
        pass
    try:
        cen = json.load(io.open(sd / "dong_centroids.json", encoding="utf-8"))
        data["centroids"] = cen.get("centroids", {})
        data["gu_fallback"] = cen.get("gu_fallback", {})
    except Exception:
        pass
    _CACHE[key] = data
    return data


def centroid_of(code: str, data: dict) -> list[float] | None:
    """동 중심좌표 → 없으면 자치구(5자리) fallback → 없으면 None."""
    if not code:
        return None
    c = data["centroids"].get(code)
    if c:
        return c
    return data["gu_fallback"].get(code[:5])


def suggest_hubs(
    home_dong_code: str,
    exclude_codes: set[str],
    day_type: str,
    mobility_level: int | None,
    k: int = 3,
    rng: random.Random | None = None,
    stats_dir: Path | None = None,
) -> list[dict]:
    """거주지 기준 Huff prior로 광역 상권 허브 k개 추천.

    반환: [{code, name, gu, attraction, distance_km, prob}, ...] (prob 내림차순).
    카탈로그/좌표 없거나 거주지 좌표 없으면 빈 리스트(→ 생활권만으로 자연 degrade).
    """
    data = _load(stats_dir)
    home_c = centroid_of(home_dong_code, data)
    if not home_c or not data["top_hubs"]:
        return []
    cand: list[dict] = []
    for h in data["top_hubs"]:
        code = h["code"]
        if code in exclude_codes:
            continue
        hc = centroid_of(code, data)
        if not hc:
            continue
        cand.append({
            "code": code, "name": h.get("name", ""), "gu": h.get("gu", ""),
            "attraction": h.get("attraction", 0.0),
            "distance_km": round(haversine_km(home_c[0], home_c[1], hc[0], hc[1]), 2),
        })
    if not cand:
        return []
    return sample_destinations(cand, day_type, mobility_level, k=k, rng=rng)


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

    # 실제 카탈로그/좌표 기반 suggest_hubs (강북 주거동 예시)
    print("\n=== suggest_hubs(실데이터) — 거주지 11305xxx 가정 ===")
    data = _load()
    print(f"  로드: top_hubs={len(data['top_hubs'])} centroids={len(data['centroids'])} "
          f"gu_fallback={len(data['gu_fallback'])}")
    # 좌표가 있는 임의 거주동 하나 선택
    home = next(iter(data["centroids"]), None)
    if home:
        for dt, lv in [("weekday", 5), ("weekend", 8)]:
            hubs = suggest_hubs(home, {home}, dt, lv, k=3, rng=random.Random(1))
            tag = ", ".join(f"{h['name']}({h['distance_km']}km,p={h['prob']:.3f})" for h in hubs)
            print(f"  {dt:8} home={home}: {tag or '(없음)'}")
