# -*- coding: utf-8 -*-
"""2026-10-07 — 가격 그대로 소비 모형 · 실측 결제 1건당 · 피로/외출 의무 설정."""
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
sys.path.insert(0, str(ROOT))


def _cons(monkeypatch):
    monkeypatch.setenv("EXP_PRICE_FIRST", "1")
    import consumption
    return importlib.reload(consumption)


def test_prices_kept_and_late_purchases_skipped(monkeypatch):
    c = _cons(monkeypatch)
    ev = [{"time": "08:00", "poi_id": "A", "sub_category": "카페"},
          {"time": "12:00", "poi_id": "B", "sub_category": "한식"},
          {"time": "19:00", "poi_id": "C", "sub_category": "한식"}]
    spends, meta = c._price_first_spends(ev, [4500.0, 9000.0, 30000.0], 13500, "X")
    assert spends[:2] == [4500, 9000]          # 가격은 깎지 않는다
    assert spends[2] == 0 and ev[2]["budget_skipped"] and ev[2]["actual_spent_planned"] == 30000
    assert meta["budget_skipped"] == 1


def test_irregular_paid_on_top(monkeypatch):
    c = _cons(monkeypatch)
    ev = [{"time": "10:00", "poi_id": "H", "sub_category": "의원"},
          {"time": "12:00", "poi_id": "B", "sub_category": "한식"}]
    spends, meta = c._price_first_spends(ev, [15000.0, 9000.0], 9000, "X")
    assert spends == [15000, 9000] and meta["irregular_total"] == 15000


def test_crossing_purchase_is_deterministic(monkeypatch):
    c = _cons(monkeypatch)
    ev1 = [{"time": "12:00", "poi_id": "B", "sub_category": "한식"}]
    ev2 = [{"time": "12:00", "poi_id": "B", "sub_category": "한식"}]
    a, _ = c._price_first_spends(ev1, [10000.0], 4000, "AGT")
    b, _ = c._price_first_spends(ev2, [10000.0], 4000, "AGT")
    assert a == b and a[0] in (0, 10000)


def test_ticket_label_and_unmapped(monkeypatch):
    monkeypatch.setenv("EXP_PRICE_MODE", "ticket")
    import price_ticket
    pt = importlib.reload(price_ticket)
    i = pt.info("11110515", "한식")
    assert i and i["svc"] == "한식음식점" and i["p10"] < i["p50"] < i["p90"]
    lab = pt.label(i)
    assert "카드 결제 1건당" in lab and "여럿이 함께 낸 결제 포함" in lab and "2025" in lab
    assert pt.info("11110515", "여행사") is None and pt.label(None) == ""


def test_old_mode_unchanged(monkeypatch):
    monkeypatch.delenv("EXP_PRICE_MODE", raising=False)
    import price_ticket
    assert importlib.reload(price_ticket).ON is False


def test_essentials_never_skipped_even_after_clinic(monkeypatch):
    c = _cons(monkeypatch)
    ev = [{"time": "09:00", "poi_id": "H", "sub_category": "의원", "category": "건강"},
          {"time": "12:00", "poi_id": "L", "sub_category": "한식", "category": "식사"},
          {"time": "15:00", "poi_id": "C", "sub_category": "카페", "category": "카페"},
          {"time": "19:00", "poi_id": "D", "sub_category": "한식", "category": "식사"}]
    spends, meta = c._price_first_spends(ev, [50000.0, 9000.0, 5000.0, 12000.0], 10000, "X")
    assert spends[0] == 50000                      # 병원은 예산 밖
    assert spends[1] == 9000 and spends[3] == 12000  # 끼니는 예산을 넘어도 거르지 않는다
    assert spends[2] == 0 and ev[2]["budget_skipped"]  # 재량 지출만 거른다


def test_monthly_paid_once(monkeypatch):
    c = _cons(monkeypatch)
    ev = [{"time": "16:00", "poi_id": "A", "sub_category": "학원", "category": "교육"}]
    s1, m1 = c._price_first_spends(ev, [250000.0], 20000, "X", monthly_paid=set())
    ev2 = [{"time": "16:00", "poi_id": "A", "sub_category": "학원", "category": "교육"}]
    s2, m2 = c._price_first_spends(ev2, [250000.0], 20000, "X", monthly_paid={"학원"})
    assert s1 == [250000] and m1["monthly_total"] == 250000
    assert s2 == [0] and ev2[0]["monthly_already_paid"]
