"""지급 일정이 있는 정책은 그 사람이 받는 날부터 보인다 — 받기 전에는 금액·사용처 표시가 프롬프트에 없다.

P013 파일럿 300c: 시행일(5/11)부터 모든 사람에게 정책이 보였고, 아직 받지 않은 사람의 사용처 지출이
+11.4% 였다. 일시금 정책(일정 없음)은 받는 날 = 시행일이라 바뀌지 않아야 한다.
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
sys.path.insert(0, str(ROOT / "scripts"))
import dawn_context as DC  # noqa: E402
import plan_writer as PW  # noqa: E402

POL = json.loads((ROOT / "data/experiments/P013_v53_policy_20261003.json").read_text(encoding="utf-8"))
LUMP = json.loads((ROOT / "data/experiments/P013_v53_policy_20260926.json").read_text(encoding="utf-8"))


def _graph_row(pol):
    """그래프에서 읽힌 모양 — 지급 일정은 mech_params JSON 안에 있다."""
    row = {k: v for k, v in pol.items() if k != "receipt_schedule"}
    row["mech_params"] = json.dumps({"receipt_schedule": pol["receipt_schedule"]}, ensure_ascii=False)
    return row


def _someone_due_after_first_day():
    for i in range(1000):
        aid = "AGT_11110515_F_70대이상_%03d" % i
        due = PW.grant_receipt_date(POL, aid)
        if due is not None and due > date(2020, 5, 13):
            return aid, due
    raise AssertionError("일정 뒤쪽에 받는 사람을 찾지 못했다")


def test_hidden_before_receipt_visible_from_receipt():
    aid, due = _someone_due_after_first_day()
    row = _graph_row(POL)
    assert DC.visible_from_receipt([row], aid, due - timedelta(days=1)) == []
    assert DC.visible_from_receipt([row], aid, due) == [row]
    assert DC.visible_from_receipt([row], aid, due + timedelta(days=5)) == [row]


def test_never_received_in_window_never_sees_it():
    for i in range(5000):
        aid = "AGT_11680640_M_30대_%04d" % i
        if PW.grant_receipt_date(POL, aid) is None:
            assert DC.visible_from_receipt([_graph_row(POL)], aid, date(2020, 6, 30)) == []
            return
    raise AssertionError("관측창 밖 사람을 찾지 못했다")


def test_lump_sum_policy_is_unchanged():
    row = dict(LUMP)
    assert "receipt_schedule" not in row
    for d in (date(2020, 5, 11), date(2020, 5, 20)):
        assert DC.visible_from_receipt([row], "AGT_11110515_F_70대이상_002", d) == [row]


def test_cached_list_is_not_mutated():
    aid, due = _someone_due_after_first_day()
    rows = [_graph_row(POL)]
    DC.visible_from_receipt(rows, aid, due - timedelta(days=1))
    assert len(rows) == 1                       # 동네 캐시를 사람별 거르기가 지우면 안 된다
