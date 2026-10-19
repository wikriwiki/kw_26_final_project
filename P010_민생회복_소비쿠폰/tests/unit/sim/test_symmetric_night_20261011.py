"""[2026-10-11] EXP_SYMMETRIC_NIGHT — 정책을 아는지 여부와 policy 기억이 밤 대화 짝 고르기에 닿지 않는다."""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
sys.path.insert(0, str(ROOT / "scripts"))


def _data(knows_a, info_a):
    return {"state": {"a": {"mood": 0.5, "fatigue": 0.3, "knows_policy": knows_a},
                      "b": {"mood": 0.5, "fatigue": 0.3, "knows_policy": False}},
            "info_count": {"a": info_a, "b": 0}}


def test_default_keeps_old_asymmetry(monkeypatch):
    monkeypatch.delenv("EXP_SYMMETRIC_NIGHT", raising=False)
    import night_interaction as ni
    importlib.reload(ni)
    assert ni._INFO_TYPES == ["policy", "sns"]
    assert ni.calc_urgency("a", "b", _data(True, 0)) > ni.calc_urgency("a", "b", _data(False, 0))


def test_symmetric_ignores_policy_knowledge(monkeypatch):
    monkeypatch.setenv("EXP_SYMMETRIC_NIGHT", "1")
    import night_interaction as ni
    importlib.reload(ni)
    assert ni._INFO_TYPES == ["sns"]
    assert ni.calc_urgency("a", "b", _data(True, 0)) == ni.calc_urgency("a", "b", _data(False, 0))
    assert "$types" in ni.FETCH_INFO_MEMORY_CYPHER
    monkeypatch.delenv("EXP_SYMMETRIC_NIGHT")
    importlib.reload(ni)
