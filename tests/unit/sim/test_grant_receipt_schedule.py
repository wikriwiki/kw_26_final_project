"""지원금 지급 일정 — 사람마다 공개된 신청·충전 일정에 맞춰 지급일을 정한다(P013)."""
from __future__ import annotations

import collections
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
sys.path.insert(0, str(ROOT / "scripts"))
import plan_writer as PW  # noqa: E402

POL = json.loads((ROOT / "data/experiments/P013_v53_policy_20261003.json").read_text(encoding="utf-8"))


def test_without_schedule_pays_on_effective_from():
    old = {k: v for k, v in POL.items() if k != "receipt_schedule"}
    assert PW.grant_receipt_date(old, "A") == date(2020, 5, 11)
    assert PW.grant_receipt_date(POL, None) == date(2020, 5, 11)      # 예비점검 경로는 그대로


def test_schedule_shares_are_reproduced_and_stable():
    aids = ["AGT_%05d" % i for i in range(40000)]
    got = collections.Counter()
    for a in aids:
        d = PW.grant_receipt_date(POL, a)
        if d is None:
            got["none"] += 1
        elif d == date(2020, 5, 12):
            got["d1"] += 1
        elif d <= date(2020, 5, 18):
            got["w1"] += 1
        elif d <= date(2020, 5, 25):
            got["w2"] += 1
        else:
            assert d <= date(2020, 6, 3)
            got["w3"] += 1
    n = len(aids)
    for key, share in (("d1", 0.0956), ("w1", 0.5094), ("w2", 0.31), ("w3", 0.069), ("none", 0.016)):
        assert abs(got[key] / n - share) < 0.01, (key, got[key] / n)
    assert PW.grant_receipt_date(POL, "AGT_00007") == PW.grant_receipt_date(POL, "AGT_00007")


def test_schedule_read_from_graph_mech_params():
    core = {k: v for k, v in POL.items() if k not in ("receipt_schedule",)}
    row = dict(core, mech_params=json.dumps({"receipt_schedule": POL["receipt_schedule"]}, ensure_ascii=False))
    for a in ("AGT_1", "AGT_2", "AGT_3"):
        assert PW.grant_receipt_date(row, a) == PW.grant_receipt_date(POL, a)


def test_grant_is_paid_once_on_the_persons_day():
    a = "AGT_00011"
    d = PW.grant_receipt_date(POL, a)
    assert d is not None
    paid = PW.grants_to_apply([POL], d, {}, "", "5", aid=a)
    assert paid == {"P013": 280000}
    other = date(2020, 5, 11) if d != date(2020, 5, 11) else date(2020, 5, 30)
    assert PW.grants_to_apply([POL], other, {}, "", "5", aid=a) == {}
    assert PW.grants_to_apply([POL], d, {"P013": 280000}, "", "5", aid=a) == {}   # 이미 받았으면 다시 안 준다
