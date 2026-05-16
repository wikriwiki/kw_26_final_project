"""vocabulary 모듈: 단일 진실의 원천이 잘 동작하는가."""

from src.policy_pipeline.vocabulary import (
    SEOUL_DISTRICTS,
    contains_any_term,
    exact_match_any,
    filter_invalid_districts,
    has_ambiguous_scope,
    is_national_scope,
    is_seoul_wide_scope,
    normalize_term,
)


def test_seoul_districts_has_25():
    assert len(SEOUL_DISTRICTS) == 25
    assert "강남구" in SEOUL_DISTRICTS


def test_normalize_term_strips_spaces_and_lowercases():
    assert normalize_term("Seoul ") == "seoul"
    assert normalize_term("서울 시 ") == "서울시"


def test_contains_any_term_partial_match():
    assert contains_any_term(["서울시 전체"], ["서울시"])
    assert not contains_any_term(["부산"], ["서울"])


def test_exact_match_any_requires_full_match():
    assert exact_match_any(["전체"], ["전체"])
    assert not exact_match_any(["전체업종"], ["전체"])  # 부분 매칭 X


def test_filter_invalid_districts_returns_unknown_only():
    assert filter_invalid_districts(["강남구", "분당구", "중구"]) == ["분당구"]
    assert filter_invalid_districts(["강남구", "중구"]) == []


def test_is_seoul_wide_scope():
    assert is_seoul_wide_scope(["서울시 전체"])
    assert is_seoul_wide_scope(["서울"])
    assert not is_seoul_wide_scope(["분당구"])


def test_is_national_scope_only_when_no_seoul_marker():
    assert is_national_scope(["전국"])
    assert not is_national_scope(["전국", "서울시"])  # 서울 표현이 있으면 매핑 가능


def test_has_ambiguous_scope():
    assert has_ambiguous_scope(["일부 지역"])
    assert has_ambiguous_scope(["주요 상권"])
    assert not has_ambiguous_scope(["강남구"])
