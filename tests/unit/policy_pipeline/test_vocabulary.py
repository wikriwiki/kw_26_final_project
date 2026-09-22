"""vocabulary: 단일 진실의 원천 동작 확인."""

from scripts.policy_pipeline.vocabulary import (
    SEOUL_DISTRICTS,
    VALID_L1_CATEGORIES,
    contains_any_term,
    exact_match_any,
    filter_invalid_categories,
    filter_invalid_districts,
    has_ambiguous_scope,
    is_national_scope,
    is_seoul_wide_scope,
    normalize_term,
)


def test_seoul_districts_has_25():
    assert len(SEOUL_DISTRICTS) == 25
    for d in ("강남구", "마포구", "중구"):
        assert d in SEOUL_DISTRICTS


def test_categories_match_yaml_l1_12():
    expected = {
        "식사", "카페", "디저트", "주점", "편의점", "마트",
        "미용", "쇼핑", "여가", "건강", "교육", "기타",
    }
    assert set(VALID_L1_CATEGORIES) == expected


def test_normalize_term():
    assert normalize_term("서울 시 ") == "서울시"


def test_contains_any_term_partial():
    assert contains_any_term(["서울시 전체"], ["서울시"])
    assert not contains_any_term(["부산"], ["서울"])


def test_exact_match_any_no_partial():
    assert exact_match_any(["전체"], ["전체"])
    assert not exact_match_any(["전체업종"], ["전체"])


def test_filter_invalid_districts():
    assert filter_invalid_districts(["강남구", "분당구"]) == ["분당구"]


def test_filter_invalid_categories():
    assert filter_invalid_categories(["식사", "우주항공"]) == ["우주항공"]
    assert filter_invalid_categories([]) == []


def test_is_seoul_wide_scope():
    assert is_seoul_wide_scope(["서울시 전체"])
    assert not is_seoul_wide_scope(["분당구"])


def test_is_national_only_without_seoul():
    assert is_national_scope(["전국"])
    assert not is_national_scope(["전국", "서울시"])


def test_has_ambiguous_scope():
    assert has_ambiguous_scope(["일부 지역"])
    assert not has_ambiguous_scope(["강남구"])
