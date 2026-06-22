"""scripts/persona/llm_reconcile.py — 방식 A+LLM 전수 LLM 일관성 감수."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "persona"))
import llm_reconcile as L  # noqa: E402


def _persona(level: int, ses: float, summary="", hobbies=None,
             job="회사원", gu="송파구") -> dict:
    return {
        "residence": {"gu": gu, "dong": "테스트동"},
        "personal": {"job": job, "income_level": "중", "life_stage": "독립"},
        "spending": {"weekday_spending_level": level, "weekend_spending_level": level,
                     "daily_spending_weekday": level * 20000,
                     "daily_spending_weekend": level * 18000,
                     "weekday_top_categories": {"식사": 0.5, "카페": 0.5}},
        "behavior": {"delivery_days": 10, "weekday_move_km": 5, "home_hours_weekday": 12},
        "personality": {"spending_tendency": "보통", "lifestyle": summary[:60]},
        "nvidia_persona": {"summary": summary, "hobbies": hobbies or [],
                           "cultural_background": "", "education_level": "4년제 대학교",
                           "marital_status": "미혼", "family_type": "1인가구"},
        "_match": {"method": "rank-coupling", "nvidia_ses": ses},
    }


# ---------------------------------------------------------------------------
# llm_audit_persona — 전수 검증 (mock judge, 서버 불필요)
# ---------------------------------------------------------------------------
def test_audit_called_for_every_persona():
    p = _persona(8, 0.75, summary="송파구 김씨")
    calls = []

    def judge(persona):
        calls.append(1)
        return {"consistent": True}

    out = L.llm_audit_persona(p, judge=judge)
    assert calls == [1]                          # 모순 여부와 무관하게 항상 호출
    assert out["_match"]["llm_audited"] is True
    assert out["_match"]["llm_consistent"] is True
    assert out["_match"]["llm_reconciled"] is False


def test_audit_consistent_keeps_narrative_and_numbers():
    p = _persona(8, 0.75, summary="원본 서사")
    orig_life = p["personality"]["lifestyle"]
    orig_spend = dict(p["spending"])

    out = L.llm_audit_persona(p, judge=lambda _p: {"consistent": True,
                                                   "fused_lifestyle": "무시되어야"})
    # consistent면 서사·숫자 그대로 (fused 적용 안 함)
    assert out["personality"]["lifestyle"] == orig_life
    assert "fused_lifestyle" not in out["nvidia_persona"]
    assert out["spending"] == orig_spend


def test_audit_inconsistent_reconciles_narrative_only():
    p = _persona(10, 0.2, summary="명품을 즐기는 삶")
    orig_summary = p["nvidia_persona"]["summary"]
    orig_spend = dict(p["spending"])

    def judge(persona):
        return {"consistent": False, "issues": "고소비↔저SES",
                "fused_lifestyle": "검소하게 사는 사람", "resolution": "저축지향 정합화"}

    out = L.llm_audit_persona(p, judge=judge)
    assert out["_match"]["llm_consistent"] is False
    assert out["_match"]["llm_reconciled"] is True
    assert out["personality"]["lifestyle"] == "검소하게 사는 사람"
    assert out["nvidia_persona"]["fused_lifestyle"] == "검소하게 사는 사람"
    assert out["_match"]["llm_issues"] == "고소비↔저SES"
    # 숫자 불변, 원본 서사 보존
    assert out["spending"] == orig_spend
    assert out["nvidia_persona"]["summary"] == orig_summary


def test_audit_inconsistent_but_no_fused_marks_not_reconciled():
    p = _persona(10, 0.2, summary="모순 인물")

    def judge(persona):
        return {"consistent": False, "fused_lifestyle": ""}   # 봉합문 비어 옴(파싱실패 등)

    out = L.llm_audit_persona(p, judge=judge)
    assert out["_match"]["llm_consistent"] is False
    assert out["_match"]["llm_reconciled"] is False           # 적용할 게 없으면 봉합 안 함


def test_audit_empty_verdict_defaults_consistent():
    p = _persona(5, 0.5)
    out = L.llm_audit_persona(p, judge=lambda _p: {})         # 빈 dict
    assert out["_match"]["llm_consistent"] is True
    assert out["_match"]["llm_reconciled"] is False


# ---------------------------------------------------------------------------
# build_audit_prompt — '무엇을 볼지'만, 기준(임계값) 미명시
# ---------------------------------------------------------------------------
def test_prompt_contains_info_dimensions_not_rules():
    p = _persona(3, 0.9, summary="변호사 서사", job="변호사", gu="서초구")
    system, user = L.build_audit_prompt(p)
    # 정보 차원 안내가 있어야
    assert "소비 수준" in system and "거주지" in system and "행동 지표" in system
    # 기계적 기준(임계값)을 적지 말라는 원칙 명시
    assert "임계값" in system and "규칙" in system
    # 페르소나 수치/서사가 user 프롬프트에 포함
    assert "서초구" in user and "변호사" in user
    # 하드코딩된 판정 임계값(0.4, '분위 2' 등)이 프롬프트에 없어야 (룰 베이스 방지)
    assert "0.4" not in system and "0.4" not in user


# ---------------------------------------------------------------------------
# parse_verdict
# ---------------------------------------------------------------------------
def test_parse_verdict_extracts_json():
    txt = '잡설 {"consistent": false, "fused_lifestyle": "abc"} 뒤에도 잡설'
    d = L.parse_verdict(txt)
    assert d["consistent"] is False and d["fused_lifestyle"] == "abc"


def test_parse_verdict_bad_returns_empty():
    assert L.parse_verdict("JSON 없음") == {}
    assert L.parse_verdict("") == {}


# ---------------------------------------------------------------------------
# stub_judge — 결정적 (테스트/오프라인 placeholder)
# ---------------------------------------------------------------------------
def test_stub_judge_consistent_when_gap_small():
    p = _persona(8, 0.75)                                     # gap ~0.03
    v = L.stub_judge(p)
    assert v["consistent"] is True and v["fused_lifestyle"] == ""


def test_stub_judge_flags_and_shapes_when_gap_large():
    p = _persona(2, 0.9, summary="이순신 씨는 전문직이다", job="변호사", gu="서초구")
    v1 = L.stub_judge(p)
    v2 = L.stub_judge(p)
    assert v1 == v2                                           # 결정적
    assert v1["consistent"] is False
    assert "검소" in v1["fused_lifestyle"] and "서초구" in v1["fused_lifestyle"]
    assert v1["issues"].startswith("[STUB]")
