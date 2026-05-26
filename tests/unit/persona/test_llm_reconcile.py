"""scripts/persona/llm_reconcile.py — 방식 A+LLM 모순 보정 레이어."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "persona"))
import llm_reconcile as L  # noqa: E402


def _persona(level: int, ses: float, summary="", hobbies=None,
             job="회사원", gu="송파구") -> dict:
    return {
        "residence": {"gu": gu, "dong": "테스트동"},
        "personal": {"job": job, "income_level": "중"},
        "spending": {"weekday_spending_level": level, "weekend_spending_level": level,
                     "daily_spending_weekday": level * 20000,
                     "weekday_top_categories": {"식사": 0.5, "카페": 0.5}},
        "behavior": {"delivery_days": 10, "weekday_move_km": 5, "home_hours_weekday": 12},
        "personality": {"spending_tendency": "보통", "lifestyle": summary[:60]},
        "nvidia_persona": {"summary": summary, "hobbies": hobbies or [],
                           "cultural_background": "", "marital_status": "미혼",
                           "family_type": "1인가구"},
        "_match": {"method": "rank-coupling", "nvidia_ses": ses},
    }


# ---------------------------------------------------------------------------
# detect_contradictions
# ---------------------------------------------------------------------------
def test_detect_none_when_aligned():
    # 소비분위 8(unit .78) ↔ SES .75 → gap 작음, 위치 충돌 없음
    p = _persona(8, 0.75, summary="송파구에 사는 김씨", gu="송파구")
    assert L.detect_contradictions(p) == []


def test_detect_ses_gap():
    p = _persona(10, 0.2, summary="평범한 일상")     # 고소비 ↔ 저SES
    w = L.detect_contradictions(p)
    assert any("ses_consume_gap" in x for x in w)


def test_detect_location_conflict():
    # 거주는 송파구인데 서사가 강남구 언급 → 위치 충돌
    p = _persona(5, 0.5, summary="강남구 역삼동에서 자란 박씨", gu="송파구")
    w = L.detect_contradictions(p)
    assert any(x.startswith("location_conflict") for x in w)


def test_no_location_conflict_when_same_gu():
    p = _persona(5, 0.5, summary="송파구 토박이 박씨", gu="송파구")
    assert not any(x.startswith("location_conflict") for x in L.detect_contradictions(p))


# ---------------------------------------------------------------------------
# llm_reconcile_persona — mock fixer (서버 불필요)
# ---------------------------------------------------------------------------
def test_reconcile_skips_when_no_contradiction():
    p = _persona(8, 0.75, summary="송파구 김씨", gu="송파구")
    called = []

    def fixer(persona, warnings):
        called.append(1)
        return {"fused_lifestyle": "X"}

    out = L.llm_reconcile_persona(p, fixer=fixer)
    assert out["_match"]["llm_reconciled"] is False
    assert called == []                       # 모순 없으면 LLM 호출 안 함


def test_reconcile_calls_fixer_and_records():
    p = _persona(10, 0.2, summary="명품을 즐기는 삶")   # 모순 有
    orig_summary = p["nvidia_persona"]["summary"]
    orig_spend = dict(p["spending"])

    def fixer(persona, warnings):
        return {"fused_lifestyle": "검소하게 사는 사람", "resolution": "저축지향으로 정합화"}

    out = L.llm_reconcile_persona(p, fixer=fixer)
    assert out["_match"]["llm_reconciled"] is True
    assert out["personality"]["lifestyle"] == "검소하게 사는 사람"
    assert out["nvidia_persona"]["fused_lifestyle"] == "검소하게 사는 사람"
    # 숫자는 절대 불변, 원본 서사 보존
    assert out["spending"] == orig_spend
    assert out["nvidia_persona"]["summary"] == orig_summary
    assert out["_match"]["llm_contradictions"]


def test_reconcile_empty_fix_marks_unreconciled():
    p = _persona(10, 0.2, summary="모순 인물")

    def fixer(persona, warnings):
        return {}                              # LLM 파싱 실패 등

    out = L.llm_reconcile_persona(p, fixer=fixer)
    assert out["_match"]["llm_reconciled"] is False


# ---------------------------------------------------------------------------
# parse_fix
# ---------------------------------------------------------------------------
def test_parse_fix_extracts_json():
    txt = '쓸데없는 말 {"fused_lifestyle": "abc", "consistent": true} 뒤에도 잡설'
    d = L.parse_fix(txt)
    assert d["fused_lifestyle"] == "abc"


def test_parse_fix_bad_returns_empty():
    assert L.parse_fix("JSON 없음") == {}
    assert L.parse_fix("") == {}


# ---------------------------------------------------------------------------
# stub_fixer — 결정적
# ---------------------------------------------------------------------------
def test_stub_fixer_deterministic_and_shaped():
    p = _persona(2, 0.9, summary="이순신 씨는 전문직이다", job="변호사", gu="서초구")
    f1 = L.stub_fixer(p, ["ses_consume_gap:저소비_고SES"])
    f2 = L.stub_fixer(p, ["ses_consume_gap:저소비_고SES"])
    assert f1 == f2                            # 결정적
    assert "검소" in f1["fused_lifestyle"]      # 저소비(분위2) → 검소 톤
    assert "서초구" in f1["fused_lifestyle"]
