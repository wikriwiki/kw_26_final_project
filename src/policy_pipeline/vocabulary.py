"""
vocabulary.py
=============
정책 추출·검증에 공통으로 쓰이는 어휘 사전과 정규화 헬퍼.

기존 `extractor.py` 와 `validator.py` 양쪽에 중복 정의돼 있던 상수/함수를
한 곳으로 합쳐 단일 진실의 원천(SoT)을 만든다.

규칙:
- 두 모듈 모두 여기서 import 한다.
- 어휘 변경 PR은 이 파일 하나만 건드린다.
- 정규화는 `normalize_term()` 한 함수만 사용 (공백 제거 + lowercase).
"""

from __future__ import annotations

from collections.abc import Iterable


# ---------------------------------------------------------------------------
# 서울 25개 자치구
# ---------------------------------------------------------------------------
SEOUL_DISTRICTS: frozenset[str] = frozenset({
    "강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구",
    "금천구", "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구",
    "서초구", "성동구", "성북구", "송파구", "양천구", "영등포구", "용산구",
    "은평구", "종로구", "중구", "중랑구",
})


# ---------------------------------------------------------------------------
# 지역 범위 어휘
# ---------------------------------------------------------------------------
# 서울 전역으로 명확하게 매핑되는 표현
SEOUL_WIDE_TERMS: frozenset[str] = frozenset({
    "서울", "서울시", "서울특별시", "서울 전체", "서울시 전체", "서울 전역",
})

# 전국 범위 — 서울 단위로 직접 매핑 불가, 사람 검토 필요
NATIONAL_SCOPE_TERMS: frozenset[str] = frozenset({
    "전국", "전 지역", "대한민국", "국내 전체",
})

# 범위가 모호한 표현 — 정량화 불가, 사람 검토 필요
AMBIGUOUS_SCOPE_TERMS: frozenset[str] = frozenset({
    "일부", "일부 지역", "일부 업종", "관련 지역", "관련 업종",
    "해당 지역", "해당 업종", "지역 상권", "인근 지역", "주요 상권", "취약 지역",
})


# ---------------------------------------------------------------------------
# 업종 / 대상 어휘 (시뮬레이션이 인식 가능한 카테고리)
# ---------------------------------------------------------------------------
VALID_INDUSTRY_TERMS: frozenset[str] = frozenset({
    "전체", "전체 업종",
    "소상공인", "전통시장",
    "음식점", "외식업", "카페", "숙박업", "도소매", "도소매업",
    "서비스업", "관광업", "문화", "공연", "편의점",
})

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
    "target_regions", "target_districts", "target_industries",
    "effective_start_date", "effective_end_date",
    "benefit_amount", "benefit_rate", "conditions",
})


# ---------------------------------------------------------------------------
# 정규화 + 매칭 헬퍼
# ---------------------------------------------------------------------------
def normalize_term(value: str) -> str:
    """공백 제거 + 소문자화. 한국어에는 영향 없으나 영문 혼용 시 안전."""
    return value.replace(" ", "").lower()


def contains_any_term(values: Iterable[str], terms: Iterable[str]) -> bool:
    """`values` 중 하나라도 `terms` 의 한 항목을 부분문자열로 포함하면 True.

    경고: 부분문자열 매칭이므로 "관광객" 이 "관광업"을 포함하는 케이스 등이 있을 수 있다.
    그래서 호출자는 가능한 한 `exact_match_any()` 를 먼저 시도하길 권장.
    """
    norm_values = [normalize_term(v) for v in values]
    norm_terms = [normalize_term(t) for t in terms]
    return any(term in value for value in norm_values for term in norm_terms)


def exact_match_any(values: Iterable[str], terms: Iterable[str]) -> bool:
    """정규화 후 완전 일치 검사. 화이트리스트 검증에 사용."""
    norm_values = {normalize_term(v) for v in values}
    norm_terms = {normalize_term(t) for t in terms}
    return bool(norm_values & norm_terms)


def filter_invalid_districts(districts: Iterable[str]) -> list[str]:
    """서울 자치구 화이트리스트에 없는 항목을 반환 (원본 표기 유지)."""
    return [d for d in districts if d not in SEOUL_DISTRICTS]


def is_seoul_wide_scope(regions: Iterable[str]) -> bool:
    """`regions` 가 서울 전역 표현을 포함하는지 (서울로 직접 매핑 가능)."""
    return contains_any_term(regions, SEOUL_WIDE_TERMS)


def is_national_scope(regions: Iterable[str]) -> bool:
    """`regions` 가 전국 범위인지 (단, 서울 전역 표현이 함께 있으면 False)."""
    if not contains_any_term(regions, NATIONAL_SCOPE_TERMS):
        return False
    return not is_seoul_wide_scope(regions)


def has_ambiguous_scope(values: Iterable[str]) -> bool:
    """모호어가 하나라도 섞여 있으면 True."""
    return contains_any_term(values, AMBIGUOUS_SCOPE_TERMS)
