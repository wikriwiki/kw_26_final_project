"""즉시 할인·사용처 판정은 고른 가게 자신의 업종으로 한다 (2026-10-05).

상위 업종으로 넓혀 가져온 후보는 Stage1 이 계획한 업종과 다를 수 있다. 계획 업종으로 판정하면
원장(그래프의 가게 업종)과 어긋난다 — 검수에서 P010·P016 둘 다 지적했다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/sim"))
from eligibility import Rules  # noqa: E402
from instant_discount import settle_instant_discounts  # noqa: E402

SPEC = [{"id": "P016", "rate": 0.2, "cap": 10000,
         "rules": Rules({"mode": "include", "include": {"subs": ["청과", "정육", "슈퍼마켓", "식료품"]}})}]


def test_fallback_shop_of_other_category_gets_no_discount():
    # 슈퍼마켓을 계획했지만 같은 동에 없어 상위 업종으로 가져온 수산 가게를 골랐다.
    ev = {"poi_id": "X", "category": "쇼핑", "sub_category": "슈퍼마켓",
          "poi_sub_category": "수산", "poi_category": "쇼핑"}
    out = settle_instant_discounts([ev], [20000], SPEC)
    assert out["total"] == 0 and out["eligible_gross"] == 0


def test_fallback_shop_of_eligible_category_gets_discount():
    ev = {"poi_id": "Y", "category": "쇼핑", "sub_category": "종합소매",
          "poi_sub_category": "청과", "poi_category": "쇼핑"}
    out = settle_instant_discounts([ev], [20000], SPEC)
    assert out["total"] == 4000


def test_without_own_category_falls_back_to_planned():
    # 후보풀 밖 POI(거주지·직장 등)는 자기 업종이 없다 — 계획 업종으로 둔다(지금과 같다).
    ev = {"poi_id": "Z", "category": "쇼핑", "sub_category": "정육"}
    assert settle_instant_discounts([ev], [10000], SPEC)["total"] == 2000
