"""지원금 카드 몫 보정 도구 — 다시 셈이 정의대로이고, 이분법이 목표를 맞히는지."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import calibrate_p013_grant_use as K  # noqa: E402

DAYS = ["2020-05-%02d" % d for d in range(11, 15)]


def _rows():
    # 한 사람: 5/12 에 28만원 받음, 매일 사용처 지출 10만원
    by = {}
    for d in DAYS:
        by[d] = {"aid": "a", "day": d, "eligible_offline_spent": 100000,
                 "grant_received_cumulative": 280000 if d >= "2020-05-12" else 0}
    return {"a": by}


def test_replay_starts_on_receipt_and_stops_at_zero():
    cum = K.replay(_rows(), 1.0, 280000, DAYS[-1])
    assert cum == {"2020-05-11": 0.0, "2020-05-12": 100000.0, "2020-05-13": 200000.0, "2020-05-14": 280000.0}
    half = K.replay(_rows(), 0.5, 280000, DAYS[-1])
    assert half["2020-05-14"] == 150000.0


def test_solve_hits_target_and_refuses_to_raise_above_one():
    f, got, note = K.solve(_rows(), 280000, "2020-05-13", 50.0)    # 목표 14만원 = 2일 x f x 10만원 → f = 0.7
    assert abs(f - 0.7) < 1e-3 and abs(got - 50.0) < 0.05 and note == "ok"
    f, got, note = K.solve(_rows(), 280000, "2020-05-12", 90.0)    # 하루치로는 1.0 이어도 35.7%
    assert f == 1.0 and note != "ok"
