"""받은 날부터 센 날수 — 프롬프트 '지급 후 N일' (2026-10-12).

지키려는 사고: 지급 일정이 있는 정책에서 시행일부터 세어, 7/26 에 받는 사람이 받는 날 '지급 후 5일'로 보였다.
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "sim"))

from plan_writer import grant_days_since, grant_receipt_date  # noqa: E402

SCHED = [{"from": "2025-07-22", "to": "2025-07-26", "share": 1.0}]
POL = {"id": "P010", "type": "grant", "effective_from": "2025-07-21", "receipt_schedule": SCHED}


def test_받는_날은_0일():
    for i in range(50):
        aid = f"AGT_{i}"
        due = grant_receipt_date(POL, aid)
        assert grant_days_since([POL], due, aid) == {"P010": 0}
        assert grant_days_since([POL], due + timedelta(days=2), aid) == {"P010": 2}


def test_받기_전에는_없다():
    aid = next(f"AGT_{i}" for i in range(100) if grant_receipt_date(POL, f"AGT_{i}") > date(2025, 7, 22))
    due = grant_receipt_date(POL, aid)
    assert grant_days_since([POL], due - timedelta(days=1), aid) == {}


def test_그래프_행_from_만_있어도_된다():
    row = {"id": "P010", "type": "grant", "from_": "2025-07-21",
           "mech_params": json.dumps({"receipt_schedule": SCHED})}
    aid = "AGT_7"
    due = grant_receipt_date(POL, aid)
    assert grant_days_since([row], due, aid) == {"P010": 0}


def test_일정_없는_정책은_시행일부터():
    pol = {"id": "P013", "type": "grant", "effective_from": "2020-05-13"}
    assert grant_days_since([pol], date(2020, 5, 15), "AGT_1") == {"P013": 2}


def test_grant_가_아니면_넣지_않는다():
    assert grant_days_since([{"id": "P012", "type": "cashback", "effective_from": "2021-10-01"}],
                            date(2021, 10, 3), "AGT_1") == {}
