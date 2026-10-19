"""모델이 JSON 뒤에 말을 덧붙여도 하루가 날아가지 않아야 한다.

라이브에서 같은 시민이 사흘 연속 "Extra data: line N" 으로 죽어 하루치를 세 번씩
다시 돌렸다(벽시계 2배). 첫 객체만 읽으면 되고, 그것은 **모델의 판단을 바꾸지 않는다.**
"""
from __future__ import annotations

import json

import pytest


def first_object(text: str):
    """stage1_intent 의 보정 순서를 그대로 재현한다."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.JSONDecoder().raw_decode(text.lstrip())[0]


def test_trailing_prose_after_a_complete_object_is_discarded():
    got = first_object('{"plan": [{"time": "12:00"}]}\n이상입니다. 추가 설명: ...')
    assert got == {"plan": [{"time": "12:00"}]}


def test_a_second_object_after_the_first_is_discarded():
    got = first_object('{"a": 1}\n{"a": 2}')
    assert got == {"a": 1}          # 첫 것만 — 뒤엣것이 이기지 않는다


def test_clean_json_is_unchanged():
    assert first_object('{"a": 1, "b": [2, 3]}') == {"a": 1, "b": [2, 3]}


def test_genuinely_broken_json_still_raises():
    with pytest.raises(json.JSONDecodeError):
        first_object('{"a": ')
