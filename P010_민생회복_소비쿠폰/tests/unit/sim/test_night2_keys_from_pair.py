"""대화의 기본키는 쌍에서 온다 — 모델이 베낀 id 가 아니다.

모델이 id 를 한 글자라도 틀리게 돌려주면 적재 MATCH 에서 그 행이 조용히 사라지고,
하루치가 버려졌다(3,000명 런에서 세 번 연속). 그래서 모델의 출력이 틀려도 적재
키는 입력 쌍 그대로여야 하고, 틀린 사실은 기록돼야 한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "sim"))

import night_intent_llm as nil


def fake_llm(initiator, recipient):
    body = {
        "intent": "추천", "initiator_id": initiator, "recipient_id": recipient,
        "topic_type": "none", "topic_value": None, "reasoning": "테스트",
        "plan_signal": {"should_inject": False, "target_day_offset": None,
                        "target_time": None, "meeting_location_hint": None},
    }
    msg = SimpleNamespace(content=json.dumps(body, ensure_ascii=False))
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    return lambda *a, **k: SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


def run(monkeypatch, echo_a, echo_b):
    monkeypatch.setattr(nil, "_llm_call", fake_llm(echo_a, echo_b))
    monkeypatch.setattr(nil, "build_user_block", lambda pk, d: "user")
    return nil.classify_intent(("AGT_11530560_F_50대_001", "AGT_11140550_M_70대이상_001"), {})


def test_a_mistyped_echo_does_not_change_the_stored_keys(monkeypatch):
    r = run(monkeypatch, "AGT_11530560_F_50대_01", "AGT_11140550_M_70대이상_001")
    assert r["initiator_id"] == "AGT_11530560_F_50대_001"      # 입력 쌍 그대로
    assert r["recipient_id"] == "AGT_11140550_M_70대이상_001"
    assert r["id_echo_ok"] is False                            # 틀린 사실은 남는다
    assert r["id_echo_raw"][0] == "AGT_11530560_F_50대_01"


def test_a_correct_echo_is_recorded_as_correct(monkeypatch):
    r = run(monkeypatch, "AGT_11530560_F_50대_001", "AGT_11140550_M_70대이상_001")
    assert r["id_echo_ok"] is True and r["id_echo_raw"] is None


def test_the_models_judgement_still_comes_from_the_model(monkeypatch):
    r = run(monkeypatch, "x", "y")
    assert r["intent"] == "추천" and r["reasoning"] == "테스트"
