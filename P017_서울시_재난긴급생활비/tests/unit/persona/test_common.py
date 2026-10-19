"""scripts/persona/_common.py — pure 함수 검증.

두 방식(rank-coupling / conditional-graft)이 공유하는 정규화·SES·통계샘플링 로직.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "persona"))
import _common as C  # noqa: E402


# ---------------------------------------------------------------------------
# 정규화
# ---------------------------------------------------------------------------
def test_nvidia_sex():
    assert C.nvidia_sex({"sex": "여자"}) == "F"
    assert C.nvidia_sex({"sex": "남자"}) == "M"
    assert C.nvidia_sex({"sex": ""}) == ""


@pytest.mark.parametrize("age,grp", [
    (19, "20대"), (29, "20대"), (30, "30대"), (45, "40대"),
    (59, "50대"), (60, "60대"), (69, "60대"), (70, "70대이상"), (99, "70대이상"),
])
def test_age_to_group(age, grp):
    assert C.age_to_group(age) == grp


def test_nvidia_gu_strips_prefix():
    assert C.nvidia_gu({"district": "서울-서초구"}) == "서초구"
    assert C.nvidia_gu({"district": "노원구"}) == "노원구"


def test_nvidia_cell():
    rec = {"sex": "여자", "age": 34, "district": "서울-마포구"}
    assert C.nvidia_cell(rec) == ("마포구", "F", "30대")


# ---------------------------------------------------------------------------
# SES_proxy
# ---------------------------------------------------------------------------
def test_ses_proxy_monotone_in_education():
    base = {"occupation": "사무원", "housing_type": "아파트"}
    low = C.ses_proxy({**base, "education_level": "초등학교"})
    high = C.ses_proxy({**base, "education_level": "대학원"})
    assert high > low


def test_ses_proxy_occupation_tiers():
    base = {"education_level": "4년제 대학교", "housing_type": "아파트"}
    doctor = C.ses_proxy({**base, "occupation": "의사"})
    clerk = C.ses_proxy({**base, "occupation": "사무원"})
    laborer = C.ses_proxy({**base, "occupation": "단순 노무 종사원"})
    assert doctor > clerk > laborer


def test_ses_proxy_range():
    # 최고/최저 입력 → [0,1]
    hi = C.ses_proxy({"education_level": "대학원", "occupation": "변호사", "housing_type": "아파트"})
    lo = C.ses_proxy({"education_level": "초등학교", "occupation": "배달원", "housing_type": "주택이외의거처"})
    assert 0.0 <= lo < hi <= 1.0


def test_ses_proxy_unknown_uses_middle():
    # 미상 직업/주거 → 중간값, 음수·overflow 없음
    s = C.ses_proxy({"education_level": "고등학교", "occupation": "정체불명직업", "housing_type": "기타"})
    assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# 통계 샘플링
# ---------------------------------------------------------------------------
def test_parse_cell_key():
    assert C.parse_cell_key("11110515_F_20대") == ("11110515", "F", "20대")
    assert C.parse_cell_key("11110515_M_70대이상") == ("11110515", "M", "70대이상")


def test_decile_amount_within_bounds():
    deciles = {"weekday_spending_level": {"boundaries": [
        {"decile": 5, "min": 10000, "max": 20000},
    ]}}
    rng = random.Random(0)
    for _ in range(50):
        amt = C.decile_amount(deciles, "weekday", 5, rng)
        assert 10000 <= amt <= 20000


def test_decile_amount_tail_rules():
    """1분위는 상단(하한 outlier 회피), 10분위는 하단."""
    deciles = {"weekday_spending_level": {"boundaries": [
        {"decile": 1, "min": 0, "max": 10000},
        {"decile": 10, "min": 100000, "max": 1000000},
    ]}}
    rng = random.Random(1)
    d1 = [C.decile_amount(deciles, "weekday", 1, rng) for _ in range(30)]
    d10 = [C.decile_amount(deciles, "weekday", 10, rng) for _ in range(30)]
    # 1분위는 상단 60~95% → 6000 이상
    assert min(d1) >= 6000
    # 10분위는 하단 5~40% → max 460000 이하
    assert max(d10) <= 460000


def test_industry_to_l1_ratio_normalizes():
    ind = {"한식": 0.3, "커피전문점": 0.2, "ZZ_나머지": 0.5}
    l1 = C.industry_to_l1_ratio(ind)
    # ZZ 제외, 식사+카페만 → 재정규화 합 ≈ 1
    assert abs(sum(l1.values()) - 1.0) < 0.01
    assert "식사" in l1 and "카페" in l1
    assert l1["식사"] > l1["카페"]   # 0.3 > 0.2


def test_industry_to_l1_empty():
    assert C.industry_to_l1_ratio({}) == {}
    assert C.industry_to_l1_ratio({"ZZ_나머지": 1.0}) == {}


@pytest.mark.parametrize("wd,we,expect", [
    (1, 2, "절약형"), (5, 6, "보통"), (9, 10, "소비형"),
])
def test_spending_tendency(wd, we, expect):
    assert C.spending_tendency_from(wd, we) == expect


# ---------------------------------------------------------------------------
# NVIDIA 필드 분리 (LLM 입력 vs 저장-only)
# ---------------------------------------------------------------------------
def test_split_nvidia_fields():
    rec = {
        "persona": "요약문",
        "hobbies_and_interests_list": "['등산', '독서']",
        "cultural_background": "배경",
        "marital_status": "미혼", "housing_type": "아파트",
        "family_type": "1인가구", "education_level": "대학원",
        "professional_persona": "직업관 서사",   # 저장만
        "career_goals_and_ambitions": "목표",     # 저장만
        "occupation": "의사",
    }
    llm, reserved = C.split_nvidia_fields(rec)
    # LLM 입력
    assert llm["summary"] == "요약문"
    assert llm["hobbies"] == ["등산", "독서"]   # 문자열 list 파싱
    assert "cultural_background" in llm
    # 직업관·목표는 LLM 입력에 없어야
    assert "professional_persona" not in llm
    assert "career_goals_and_ambitions" not in llm
    # 저장-only 에 있어야
    assert reserved["professional_persona"] == "직업관 서사"
    assert reserved["career_goals_and_ambitions"] == "목표"


def test_parse_list_field_handles_formats():
    assert C._parse_list_field(["a", "b"]) == ["a", "b"]
    assert C._parse_list_field("['x', 'y']") == ["x", "y"]
    assert C._parse_list_field("") == []
    assert C._parse_list_field("plain") == ["plain"]
