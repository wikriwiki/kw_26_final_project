"""validator: 도메인 룰 (자치구 화이트리스트, L1 카테고리, 모호어)."""

from scripts.policy_pipeline.models import ExtractedPolicy, PolicyType
from scripts.policy_pipeline.state import PolicyStatus
from scripts.policy_pipeline.validator import validate_policy


def _make(**overrides) -> ExtractedPolicy:
    base = {
        "policy_id": "policy_x",
        "title": "t",
        "summary": "s",
        "source_file": "/tmp/x.txt",
        "target_districts": ["강남구"],
        "benefit_categories": ["식사"],
        "policy_type": PolicyType.COUPON,
        "confidence": 0.95,
    }
    base.update(overrides)
    return ExtractedPolicy(**base)


def test_valid_seoul_district_passes():
    out = validate_policy(_make())
    assert out.status == PolicyStatus.VALIDATED
    assert out.validated_policy is not None


def test_invalid_district_marks_review():
    out = validate_policy(_make(target_districts=["강남구", "분당구"]))
    assert out.status == PolicyStatus.NEEDS_REVIEW
    assert any("분당구" in i.message for i in out.issues)


def test_unknown_l1_category_marks_review():
    out = validate_policy(_make(benefit_categories=["우주항공"]))
    assert out.status == PolicyStatus.NEEDS_REVIEW
    assert any("우주항공" in i.message for i in out.issues)


def test_empty_categories_means_all_commerce_passes():
    # P007 케이스: benefit_categories=[] = 전체 commerce
    out = validate_policy(_make(benefit_categories=[]))
    assert out.status == PolicyStatus.VALIDATED


def test_ambiguous_scope_marks_review():
    out = validate_policy(_make(benefit_categories=["일부 업종"]))
    assert out.status == PolicyStatus.NEEDS_REVIEW


def test_low_confidence_propagates():
    out = validate_policy(_make(confidence=0.5))
    assert out.status == PolicyStatus.NEEDS_REVIEW


def test_seoul_25_districts_passes():
    from scripts.policy_pipeline.vocabulary import SEOUL_DISTRICTS
    out = validate_policy(_make(target_districts=list(SEOUL_DISTRICTS)))
    assert out.status == PolicyStatus.VALIDATED
