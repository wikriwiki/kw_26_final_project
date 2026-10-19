"""
vocabulary.py
=============
정책 추출·검증에 공통으로 쓰이는 어휘 사전. **단일 진실의 원천.**

agent_persona 브랜치의 Neo4j 스키마에 맞춰 적응:
- 자치구: 서울 25개 District.name (그대로)
- 카테고리: data/neo4j_load/categories/categories.yaml 의 L1 12종 (식사/카페/...)
- 지역 단위는 District 또는 Dong (서울 전역 또는 자치구 또는 동)
"""

from __future__ import annotations

from collections.abc import Iterable


# ---------------------------------------------------------------------------
# 서울 25개 자치구 (Neo4j (:District {name}) 와 1:1)
# ---------------------------------------------------------------------------
SEOUL_DISTRICTS: frozenset[str] = frozenset({
    "강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구",
    "금천구", "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구",
    "서초구", "성동구", "성북구", "송파구", "양천구", "영등포구", "용산구",
    "은평구", "종로구", "중구", "중랑구",
})


# ---------------------------------------------------------------------------
# 카테고리 — L1 12종 (categories.yaml 의 name 그대로)
# Neo4j 의 (:Category {parent}) 속성과 정확히 일치해야 한다.
# ---------------------------------------------------------------------------
VALID_L1_CATEGORIES: frozenset[str] = frozenset({
    "식사", "카페", "디저트", "주점",
    "편의점", "마트",
    "미용", "쇼핑", "여가", "건강", "교육", "기타",
})


# ---------------------------------------------------------------------------
# 지역 범위 어휘
# ---------------------------------------------------------------------------
SEOUL_WIDE_TERMS: frozenset[str] = frozenset({
    "서울", "서울시", "서울특별시", "서울 전체", "서울시 전체", "서울 전역",
})

NATIONAL_SCOPE_TERMS: frozenset[str] = frozenset({
    "전국", "전 지역", "대한민국", "국내 전체",
})

AMBIGUOUS_SCOPE_TERMS: frozenset[str] = frozenset({
    "일부", "일부 지역", "일부 업종", "관련 지역", "관련 업종",
    "해당 지역", "해당 업종", "지역 상권", "인근 지역", "주요 상권", "취약 지역",
})


# ---------------------------------------------------------------------------
# 대상 그룹 어휘 (Agent 노드의 p_age_group / p_income_level / p_life_stage 와 매칭)
# Neo4j Agent 스키마: p_age_group(20/30/40/...), p_income_level(상/중/하), p_life_stage 등
# ---------------------------------------------------------------------------
VALID_TARGET_GROUP_TERMS: frozenset[str] = frozenset({
    "전체", "전체 시민", "서울시민", "주민",
    "소상공인", "자영업자",
    "청년", "노인", "어르신", "저소득층", "취약계층",
    "관광객", "학생", "가구",
})


# ---------------------------------------------------------------------------
# 검토 트리거 필드 — 추출 결과에 이 필드가 missing/ambiguous 면 사람 검토 필요
# ---------------------------------------------------------------------------
REVIEW_TRIGGER_FIELDS: frozenset[str] = frozenset({
    "target_districts", "benefit_categories",
    "effective_from", "effective_until",
    "benefit_rate", "cap_per_agent",
})


# ---------------------------------------------------------------------------
# 정규화 / 매칭 헬퍼
# ---------------------------------------------------------------------------
def normalize_term(value: str) -> str:
    return value.replace(" ", "").lower()


def contains_any_term(values: Iterable[str], terms: Iterable[str]) -> bool:
    norm_values = [normalize_term(v) for v in values]
    norm_terms = [normalize_term(t) for t in terms]
    return any(term in value for value in norm_values for term in norm_terms)


def exact_match_any(values: Iterable[str], terms: Iterable[str]) -> bool:
    norm_values = {normalize_term(v) for v in values}
    norm_terms = {normalize_term(t) for t in terms}
    return bool(norm_values & norm_terms)


def filter_invalid_districts(districts: Iterable[str]) -> list[str]:
    return [d for d in districts if d not in SEOUL_DISTRICTS]


def filter_invalid_categories(cats: Iterable[str]) -> list[str]:
    """L1 화이트리스트에 없는 카테고리 이름을 반환."""
    return [c for c in cats if c not in VALID_L1_CATEGORIES]


def is_seoul_wide_scope(regions: Iterable[str]) -> bool:
    return contains_any_term(regions, SEOUL_WIDE_TERMS)


def is_national_scope(regions: Iterable[str]) -> bool:
    if not contains_any_term(regions, NATIONAL_SCOPE_TERMS):
        return False
    return not is_seoul_wide_scope(regions)


def has_ambiguous_scope(values: Iterable[str]) -> bool:
    return contains_any_term(values, AMBIGUOUS_SCOPE_TERMS)
