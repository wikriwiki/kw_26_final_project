"""게이트가 막은 칸은 계약이 유효하다 — 그래서 수정 경로가 못 보던 실패다.

되먹임에 자원 부족을 실어 보내되, 어느 해법을 쓰라고는 말하지 않는다.
확보하거나 자원이 필요 없는 다른 방법으로 바꾸거나, 둘 다 열려 있어야 한다.
"""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
from action_repair_feedback import append_feedback, feedback

SHORTFALL = {'event_id': 'event:3', 'time': '16:00', 'resource': 'meal_stock',
             'maximum_available': 0, 'required': 1}


def valid_cell():
    return {'aid': 'A', 'case': 'grant', 'arm': 'off', 'user': '입력',
            'evaluation_requirements': []}


def test_a_valid_plan_with_no_shortfall_is_not_regenerated(monkeypatch):
    import action_repair_feedback as m
    monkeypatch.setattr(m, 'inspect', lambda raw, cell, max_shift=10: {
        'valid': True, 'errors': [], 'execution_plan': {'events': []}})
    assert feedback('{}', valid_cell()) is None


def test_a_valid_plan_with_a_shortfall_is_sent_back(monkeypatch):
    """계약은 통과했는데 물리적으로 불가능한 계획 — 이것이 잡히지 않았다."""
    import action_repair_feedback as m
    monkeypatch.setattr(m, 'inspect', lambda raw, cell, max_shift=10: {
        'valid': True, 'errors': [], 'execution_plan': {'events': []}})
    got = feedback('{}', valid_cell(), resource={'shortfalls': [SHORTFALL]})
    assert got is not None
    assert got['resource_shortfalls'] == [SHORTFALL]
    assert got['violations'] == []


def test_the_shortfall_carries_the_numbers_the_gate_measured(monkeypatch):
    import action_repair_feedback as m
    monkeypatch.setattr(m, 'inspect', lambda raw, cell, max_shift=10: {
        'valid': True, 'errors': [], 'execution_plan': {'events': []}})
    got = feedback('{}', valid_cell(), resource={'shortfalls': [SHORTFALL]})
    s = got['resource_shortfalls'][0]
    assert s['resource'] == 'meal_stock' and s['maximum_available'] == 0 and s['required'] == 1


def test_the_request_leaves_both_resolutions_open():
    text = append_feedback('입력', {'resource_shortfalls': [SHORTFALL]})
    assert '자원을 확보하는 일을 함께 넣거나' in text
    assert '자원이 필요 없는 다른 방법으로 바꾼다' in text


def test_the_request_still_forbids_gaming_and_targets():
    text = append_feedback('입력', {'resource_shortfalls': [SHORTFALL]})
    assert '구매·외출을 일괄 삭제하지 않는다' in text
    assert '정책의 성공·소비 변화 방향·금액을 목표로 수정하지 않는다' in text
    for word in ('장보', '마트', '음식점', '카페', '배달을 시'):
        assert word not in text.split('{')[0], word


def test_a_valid_decision_is_never_regenerated():
    with pytest.raises(ValueError):
        append_feedback('입력', None)


def test_verdicts_without_their_replicate_are_refused(tmp_path):
    """게이트 판정은 한 seed 의 계획에 속한다. 다른 seed 행에 붙이면 없던 부족을 알려 주게 된다."""
    import prepare_action_repair as m
    with pytest.raises(ValueError, match='replicate'):
        m.prepare(tmp_path, resource_verdicts={('A', 'grant', 'off'): {'shortfalls': [SHORTFALL]}})


def test_the_case_name_is_preserved_so_quotes_stay_comparable():
    """case 는 어느 지갑이 받는지를 정한다. 이름을 바꾸면 견적이 미리 준 것과 달라진다."""
    import inspect as _inspect
    import prepare_action_repair as m
    src = _inspect.getsource(m.prepare)
    assert "__repair_" not in src, 'case 를 다시 renaming 하면 안 된다'
    assert "cell['repair_of']" in src, '수정 표시는 별도 필드에 남긴다'
