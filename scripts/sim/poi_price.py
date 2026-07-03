"""POI 가격대 — 결정론적 가격 밴드·배율 (데이터 기반 동 prior × seeded hash).

배경: 기존 파이프라인은 어떤 POI를 골라도 소비 금액이 달라지지 않았다
(Stage2 actual_spent는 상대 가중치, 하루 총액은 propensity×가용자산으로 재정규화).
판매자 에이전트의 가격 채널이 의미를 가지려면 ① POI마다 가격대가 있고
② 소비자가 그것을 보고 선택하며 ③ 선택이 금액에 반영되어야 한다. 이 모듈은 ①을 담당.

구성:
  price_factor(poi) = dong_factor(행정동) × band_factor(POI 밴드)
  - dong_factor: BDC 상권발달지수 매출지수(b069_sales) 분위 → [0.88, 1.12], 평균 1.0
    (프리미엄 상권일수록 객단가↑ — 임대료·단가 상관 proxy.
     추후 서울 상권분석서비스 추정매출 '건당 결제금액'으로 교체 가능한 단일 지점)
  - band: sha1(poi_id) 결정론 → ₩(저가 30%) / ₩₩(중간 50%) / ₩₩₩(고가 20%)
    경험재(식사·카페 등)는 폭 넓게(0.75/1.00/1.35), 편의점·마트는 거의 균일(0.95/1.00/1.05)
  - 도시 전체 기대값 ≈ 1.0 (0.3×0.75+0.5×1.0+0.2×1.35=0.995) →
    apply_consumption_model 의 총량 캘리브레이션(평상일 spend≈daily_wd) 보존.

순수 함수 + 모듈 캐시. Neo4j/SQLite 접근 없음, 호출 O(1).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

# 가격 분산이 큰 경험재 vs 거의 균일가인 commodity
_WIDE_BANDS = (0.75, 1.00, 1.35)     # ₩ / ₩₩ / ₩₩₩
_NARROW_BANDS = (0.95, 1.00, 1.05)
_NARROW_L1 = {"편의점", "마트"}
# 밴드 분포 (%): ₩ 30 / ₩₩ 50 / ₩₩₩ 20  → E[band_factor]=0.995
_BAND_CUT = (30, 80)

_FACTOR_CLAMP = (0.60, 1.60)
_DONG_RANGE = (0.88, 1.12)   # 평균 1.0 되도록 rank 중심 대칭

_dong_factor_cache: dict[str, float] | None = None


def _load_dong_factors() -> dict[str, float]:
    """dong_context.json 의 b069_sales 분위 rank → 동 가격 계수. 실패 시 빈 dict(=전부 1.0)."""
    global _dong_factor_cache
    if _dong_factor_cache is not None:
        return _dong_factor_cache
    out: dict[str, float] = {}
    try:
        path = Path(__file__).resolve().parents[2] / "output" / "stats" / "dong_context.json"
        dc = json.loads(path.read_text(encoding="utf-8"))
        sales = {k: float(v.get("b069_sales") or 0) for k, v in dc.items() if isinstance(v, dict)}
        ranked = sorted(sales, key=lambda k: sales[k])
        n = max(1, len(ranked) - 1)
        lo, hi = _DONG_RANGE
        for i, k in enumerate(ranked):
            out[k] = round(lo + (hi - lo) * (i / n), 4)
    except Exception:
        out = {}
    _dong_factor_cache = out
    return out


def price_band(poi_id: str) -> int:
    """POI 고유 가격 밴드 1(₩)/2(₩₩)/3(₩₩₩) — sha1 결정론(실행·플랫폼 불변)."""
    h = int(hashlib.sha1((poi_id or "").encode("utf-8")).hexdigest()[:8], 16) % 100
    if h < _BAND_CUT[0]:
        return 1
    if h < _BAND_CUT[1]:
        return 2
    return 3


def poi_price(poi_id: str, dong_code: str | None, l1: str | None) -> tuple[int, float]:
    """(price_band, price_factor). dong_code 미상이면 동 계수 1.0."""
    band = price_band(poi_id)
    bands = _NARROW_BANDS if (l1 or "") in _NARROW_L1 else _WIDE_BANDS
    f = bands[band - 1] * _load_dong_factors().get(dong_code or "", 1.0)
    lo, hi = _FACTOR_CLAMP
    return band, round(max(lo, min(hi, f)), 3)


def price_icon(band: int | None) -> str:
    """후보 라인 표기: ₩ / ₩₩ / ₩₩₩ (미상 '')."""
    return "₩" * band if band in (1, 2, 3) else ""


# =========================================================
# 자체 테스트
# =========================================================
if __name__ == "__main__":
    import statistics
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=== ① 결정론: 같은 poi_id는 항상 같은 밴드 ===")
    assert price_band("C_12345") == price_band("C_12345")
    print("  ✔", price_band("C_12345"), price_icon(price_band("C_12345")))

    print("\n=== ② 밴드 분포·기대값 (합성 10,000 POI, 동 미상) ===")
    ids = [f"C_{i:06d}" for i in range(10000)]
    bands = [price_band(p) for p in ids]
    dist = {b: bands.count(b) / len(bands) for b in (1, 2, 3)}
    fs_wide = [poi_price(p, None, "식사")[1] for p in ids]
    fs_narrow = [poi_price(p, None, "편의점")[1] for p in ids]
    print(f"  분포: {dist} (목표 0.30/0.50/0.20)")
    print(f"  E[factor] 식사={statistics.mean(fs_wide):.4f} (목표≈0.995) "
          f"편의점={statistics.mean(fs_narrow):.4f}")
    assert abs(dist[1] - 0.30) < 0.02 and abs(dist[3] - 0.20) < 0.02
    assert abs(statistics.mean(fs_wide) - 0.995) < 0.01
    assert max(fs_narrow) - min(fs_narrow) < 0.11, "편의점은 거의 균일가"

    print("\n=== ③ 동 가격 prior (dong_context.json 로드) ===")
    df = _load_dong_factors()
    if df:
        vals = list(df.values())
        print(f"  동 {len(df)}개, 계수 범위 [{min(vals):.3f}, {max(vals):.3f}], 평균 {statistics.mean(vals):.4f}")
        assert abs(statistics.mean(vals) - 1.0) < 0.01
    else:
        print("  (dong_context.json 없음 → 전부 1.0 fallback)")
    print("\nALL OK")
