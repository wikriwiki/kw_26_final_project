"""POI 가격대 — 실측 우선 계층 구조 (v2).

소비 알고리즘의 가격 축. 세 레이어를 실측 우선으로 합성한다:

  [Layer B] POI 개별 메뉴가  (output/stats/poi_menu_price.json, 있으면)
      카카오 플레이스 메뉴가격 크롤 → 업종 내 백분위 밴드 + 절대 상대배율(rel).
      생성: scripts/prep/extract_kakao_menu_price.py (크롤 머신에서 실행)
  [Layer A] 동 가격계수 + 업종 기준단가  (output/stats/unit_price.json)
      BDC 신용카드 동별 건단가 복원(금액계/건수계) — 실측. golmok CSV(추정매출-행정동)
      가 있으면 업종(L1)×동 건단가로 자동 승격. 생성: scripts/prep/build_unit_price.py
      ⇒ 기존 b069_sales(발달지수) prior의 순환성(Huff prior·공간 검증 정답과 동일 변수) 제거
  [fallback] 결정론 해시 밴드 + b069 rank + 하드코딩 업종 단가
      위 파일이 없어도 기존 v1과 동일하게 동작 (하위호환).

합성 규칙 (이중계상 방지):
  menu rel 있음  → factor = rel                (메뉴가가 이미 동네 물가 포함 — 동계수 곱 금지)
  menu 없음      → factor = band_factor × dong_factor
전 레이어 E[factor] ≈ 1.0 캘리브레이션 → apply_consumption_model 총량 보존.

순수 함수 + 모듈 캐시. 호출 O(1), 소비자 경로 신규 I/O 없음(기동 시 1회 로드).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_STATS = Path(__file__).resolve().parents[2] / "output" / "stats"

# 가격 분산이 큰 경험재 vs 거의 균일가인 commodity
_WIDE_BANDS = (0.75, 1.00, 1.35)     # ₩ / ₩₩ / ₩₩₩
_NARROW_BANDS = (0.95, 1.00, 1.05)
_NARROW_L1 = {"편의점", "마트"}
# 해시 밴드 분포 (%): ₩ 30 / ₩₩ 50 / ₩₩₩ 20  → E[band_factor]=0.995
_BAND_CUT = (30, 80)

_FACTOR_CLAMP = (0.60, 1.60)
_B069_DONG_RANGE = (0.88, 1.12)   # fallback 전용 (unit_price.json 없을 때)

# 업종(L1) 기준단가 fallback (원) — golmok l1_unit_price 없을 때.
# 실측 아님(경험 상수): unit_price.json 의 l1_unit_price 가 생기면 자동 대체됨.
_L1_FALLBACK_WON = {
    "편의점": 5000, "마트": 25000,
    "식사": 12000, "카페": 6000, "디저트": 8000, "주점": 30000,
    "미용": 30000, "쇼핑": 50000,
    "여가": 20000, "건강": 15000, "교육": 50000, "기타": 10000,
}

_unit_cache: dict | None = None          # unit_price.json
_menu_cache: dict | None = None          # poi_menu_price.json
_b069_cache: dict[str, float] | None = None


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _unit() -> dict:
    """unit_price.json — {source, dong_factor, l1_unit_price}. 없으면 {}."""
    global _unit_cache
    if _unit_cache is None:
        _unit_cache = _load_json(_STATS / "unit_price.json")
    return _unit_cache


def _menu() -> dict:
    """poi_menu_price.json — {"poi": {"C_x": {band, rel, ...}}}. 없으면 {}."""
    global _menu_cache
    if _menu_cache is None:
        d = _load_json(_STATS / "poi_menu_price.json")
        _menu_cache = d.get("poi", d) if isinstance(d, dict) else {}
    return _menu_cache


def _b069_dong_factors() -> dict[str, float]:
    """[fallback] b069_sales 분위 rank → 동 계수. unit_price.json 없을 때만 사용."""
    global _b069_cache
    if _b069_cache is not None:
        return _b069_cache
    out: dict[str, float] = {}
    dc = _load_json(_STATS / "dong_context.json")
    sales = {k: float(v.get("b069_sales") or 0) for k, v in dc.items() if isinstance(v, dict)}
    if sales:
        ranked = sorted(sales, key=lambda k: sales[k])
        n = max(1, len(ranked) - 1)
        lo, hi = _B069_DONG_RANGE
        out = {k: round(lo + (hi - lo) * (i / n), 4) for i, k in enumerate(ranked)}
    _b069_cache = out
    return out


def dong_factor(dong_code: str | None) -> float:
    """동 가격계수 — 실측 건단가 기반(unit_price.json) 우선, 없으면 b069 rank, 최후 1.0."""
    if not dong_code:
        return 1.0
    df = _unit().get("dong_factor") or {}
    if df:
        return float(df.get(dong_code, 1.0))
    return _b069_dong_factors().get(dong_code, 1.0)


def band_factor(l1: str | None, band: int) -> float:
    """밴드(1~3) → 업종군별 배율. 편의점·마트는 거의 균일가."""
    bands = _NARROW_BANDS if (l1 or "") in _NARROW_L1 else _WIDE_BANDS
    return bands[max(1, min(3, band)) - 1]


def price_band(poi_id: str) -> int:
    """[fallback] POI 해시 밴드 1(₩)/2(₩₩)/3(₩₩₩) — sha1 결정론(실행·플랫폼 불변)."""
    h = int(hashlib.sha1((poi_id or "").encode("utf-8")).hexdigest()[:8], 16) % 100
    if h < _BAND_CUT[0]:
        return 1
    if h < _BAND_CUT[1]:
        return 2
    return 3


def poi_price(poi_id: str, dong_code: str | None, l1: str | None) -> tuple[int, float]:
    """(price_band, price_factor).

    Layer B(메뉴가 실측) 우선 — rel은 서울 동일업종 대비 절대배율이라 동계수 미적용.
    없으면 해시 밴드 × 동계수(실측 건단가 기반).
    """
    lo, hi = _FACTOR_CLAMP
    m = _menu().get(poi_id)
    if m and m.get("rel"):
        band = int(m.get("band") or 2)
        f = float(m["rel"])
        return band, round(max(lo, min(hi, f)), 3)
    band = price_band(poi_id)
    f = band_factor(l1, band) * dong_factor(dong_code)
    return band, round(max(lo, min(hi, f)), 3)


def unit_price_anchor(dong_code: str | None, l1: str | None) -> int | None:
    """이 동네×업종의 평균 결제단가 앵커(원) — Stage2 프롬프트·fallback 지출의 실측 스케일.

    기준선: golmok l1_unit_price(실측, 있으면) > 하드코딩 표. × 동 가격계수. 500원 반올림.
    """
    base = (_unit().get("l1_unit_price") or {}).get(l1 or "") or _L1_FALLBACK_WON.get(l1 or "")
    if not base:
        return None
    won = float(base) * dong_factor(dong_code)
    return int(round(won / 500.0) * 500)


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

    print("\n=== ② 해시 밴드 분포·기대값 (합성 10,000 POI, 동 미상) ===")
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

    print("\n=== ③ 동 가격계수 — 실측 건단가(unit_price.json) 로드 확인 ===")
    u = _unit()
    df = u.get("dong_factor") or {}
    if df:
        vals = list(df.values())
        print(f"  source={u.get('source')} 동 {len(df)}개 "
              f"[{min(vals):.3f}, {max(vals):.3f}] 평균 {statistics.mean(vals):.4f}")
        assert abs(statistics.mean(vals) - 1.0) < 0.01
        assert u.get("source", "").startswith(("bdc", "golmok")), "실측 소스 아님"
    else:
        print("  (unit_price.json 없음 → b069 rank fallback 사용)")
        assert _b069_dong_factors(), "fallback도 없음"

    print("\n=== ④ 단가 앵커 — 동네×업종 평균 결제단가 ===")
    some_dong = next(iter(df), None)
    for l1 in ("식사", "카페", "마트"):
        a0 = unit_price_anchor(None, l1)
        a1 = unit_price_anchor(some_dong, l1)
        src = "golmok 실측" if (_unit().get("l1_unit_price") or {}).get(l1) else "fallback 표"
        print(f"  {l1}: 기본 {a0:,}원 / {some_dong} {a1:,}원  [{src}]")
        assert a0 and a0 % 500 == 0

    print("\n=== ⑤ Layer B 훅 — 메뉴가 실측이 있으면 rel 우선·동계수 미적용 ===")
    _menu_cache = {"C_MENU01": {"band": 3, "rel": 1.42, "median_won": 38000}}
    b, f = poi_price("C_MENU01", some_dong, "식사")
    print(f"  C_MENU01 → band={b} factor={f} (rel 그대로, dong_factor 곱 없음)")
    assert (b, f) == (3, 1.42)
    _menu_cache = None
    print("\nALL OK")
