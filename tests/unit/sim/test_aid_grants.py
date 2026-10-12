"""사람별 지급액(aid_grants) — P010 1차 수급 자격별 금액 (2026-10-12).

지키려는 것: 지급(grants_to_apply)·프롬프트 표시(dawn_context._format_policy_status)·그래프 적재 경로
(mech_params JSON)가 같은 사람에게 같은 금액을 준다. 셋 중 하나만 표를 읽으면 프롬프트에는 40만원,
지갑에는 15만원이 들어가는 식으로 조용히 어긋난다.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "sim"))

from plan_writer import grant_receipt_date, grants_to_apply, _grant_for_single_policy  # noqa: E402

BASE = {"id": "P010", "type": "grant", "effective_from": "2025-07-21", "grant_key": "spend_decile",
        "decile_grants": {str(i): 150000 for i in range(1, 11)},
        "receipt_schedule": [{"from": "2025-07-22", "to": "2025-07-26", "share": 1.0}],
        "aid_grants": {"AGT_A": 400000, "AGT_B": 300000}}


def _due(pol, aid):
    return grant_receipt_date(pol, aid)


def test_표에_있는_사람은_표의_금액을_받는다():
    for aid, amt in (("AGT_A", 400000), ("AGT_B", 300000)):
        d = _due(BASE, aid)
        assert grants_to_apply([BASE], d, {}, "하", 5, aid=aid) == {"P010": amt}


def test_표에_없는_사람은_기존_규칙():
    d = _due(BASE, "AGT_C")
    assert grants_to_apply([BASE], d, {}, "중", 5, aid="AGT_C") == {"P010": 150000}


def test_그래프_경로_mech_params_에서도_읽는다():
    core = {k: v for k, v in BASE.items() if k not in ("aid_grants", "receipt_schedule")}
    g = {**core, "mech_params": json.dumps({"aid_grants": BASE["aid_grants"],
                                             "receipt_schedule": BASE["receipt_schedule"]})}
    d = _due(g, "AGT_A")
    assert d == _due(BASE, "AGT_A")
    assert grants_to_apply([g], d, {}, "하", 5, aid="AGT_A") == {"P010": 400000}


def test_aid_가_없으면_표를_쓰지_않는다():
    assert _grant_for_single_policy("하", BASE, spend_decile=5) == 150000


def test_받는_날은_7월22일부터_26일_사이():
    days = {_due(BASE, f"AGT_{i}") for i in range(500)}
    assert min(days) == date(2025, 7, 22) and max(days) == date(2025, 7, 26)
    assert len(days) == 5


def test_프롬프트_표시도_같은_금액():
    import dawn_context as dc
    rows = [{**{k: v for k, v in BASE.items() if k != "aid_grants"},
             "mech_params": json.dumps({"aid_grants": BASE["aid_grants"],
                                        "receipt_schedule": BASE["receipt_schedule"]})}]
    for aid, amt in (("AGT_A", 400000), ("AGT_C", 150000)):
        txt = dc._format_policy_status(rows, persona={"id": aid, "income": "하", "spend_decile": 5,
                                                      "daily_wd": 30000}, state={})
        assert f"지급액 {amt:,}원" in txt, txt
