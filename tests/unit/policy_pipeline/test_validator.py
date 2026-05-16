"""validator: 도메인 룰 단위 테스트."""

from src.policy_pipeline.models import BenefitType, ExtractedPolicy
from src.policy_pipeline.state import PolicyStatus
from src.policy_pipeline.validator import validate_policy


def _make(**overrides) -> ExtractedPolicy:
    base = {
        "policy_id": "policy_x",
        "title": "t",
        "summary": "s",
        "source_file": "/tmp/x.txt",
        "target_regions": ["서울시"],
        "target_districts": [],
        "target_industries": ["전체 업종"],
        "target_groups": [],
        "benefit_type": BenefitType.COUPON,
        "confidence": 0.95,
    }
    base.update(overrides)
    return ExtractedPolicy(**base)


def test_seoul_wide_valid_industry_passes():
    out = validate_policy(_make())
    assert out.status == PolicyStatus.VALIDATED
    assert out.validated_policy is not None
    assert out.validated_policy.title == "t"


def test_invalid_district_marks_review():
    out = validate_policy(_make(
        target_regions=[],
        target_districts=["강남구", "분당구"],  # 분당구 = 서울 X
    ))
    assert out.status == PolicyStatus.NEEDS_REVIEW
    assert any("분당구" in i.message for i in out.issues)


def test_region_not_mapped_marks_review():
    # 자치구도 없고 서울 표현도 없음
    out = validate_policy(_make(target_regions=["부산"]))
    assert out.status == PolicyStatus.NEEDS_REVIEW


def test_national_scope_marks_review():
    out = validate_policy(_make(target_regions=["전국"]))
    assert out.status == PolicyStatus.NEEDS_REVIEW
    assert any("national" in i.message.lower() for i in out.issues)


def test_ambiguous_scope_marks_review():
    out = validate_policy(_make(target_industries=["일부 업종"]))
    assert out.status == PolicyStatus.NEEDS_REVIEW


def test_unknown_industry_marks_review():
    out = validate_policy(_make(target_industries=["우주항공", "특수업종"]))
    assert out.status == PolicyStatus.NEEDS_REVIEW


def test_low_confidence_already_flagged_in_model_propagates():
    out = validate_policy(_make(confidence=0.5))  # 모델이 review 플래그 자동 부여
    assert out.status == PolicyStatus.NEEDS_REVIEW
