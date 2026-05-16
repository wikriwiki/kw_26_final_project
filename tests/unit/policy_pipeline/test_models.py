"""models: Pydantic 자기일관성 + 자동 review 플래그."""

from datetime import date

import pytest
from pydantic import ValidationError

from src.policy_pipeline.models import (
    BenefitType,
    ExtractedPolicy,
    PolicyModel,
    to_validated_policy,
)


def _base_kwargs(**overrides):
    base = {
        "policy_id": "policy_x",
        "title": "t",
        "summary": "s",
        "source_file": "/tmp/x.txt",
        "target_regions": ["서울시"],
        "target_industries": ["전체"],
        "benefit_type": BenefitType.COUPON,
    }
    base.update(overrides)
    return base


def test_policy_model_rejects_inverted_date_range():
    with pytest.raises(ValidationError):
        PolicyModel(
            **_base_kwargs(
                effective_start_date=date(2026, 6, 1),
                effective_end_date=date(2026, 5, 1),
            )
        )


def test_policy_model_rejects_negative_benefit_amount():
    with pytest.raises(ValidationError):
        PolicyModel(**_base_kwargs(benefit_amount=-1))


def test_policy_model_rejects_invalid_benefit_rate():
    with pytest.raises(ValidationError):
        PolicyModel(**_base_kwargs(benefit_rate=1.5))


def test_extracted_policy_flags_low_confidence():
    p = ExtractedPolicy(
        **_base_kwargs(),
        confidence=0.5,
    )
    assert p.requires_human_review is True
    assert any("confidence" in r for r in p.review_reasons)


def test_extracted_policy_flags_ambiguous_field():
    p = ExtractedPolicy(
        **_base_kwargs(),
        confidence=0.9,
        ambiguous_fields=["target_industries"],
    )
    assert p.requires_human_review is True
    assert any("ambiguous" in r for r in p.review_reasons)


def test_to_validated_policy_refuses_review_required():
    p = ExtractedPolicy(
        **_base_kwargs(),
        confidence=0.5,  # 자동으로 requires_human_review=True
    )
    with pytest.raises(ValueError):
        to_validated_policy(p)


def test_to_validated_policy_strips_extraction_meta():
    p = ExtractedPolicy(**_base_kwargs(), confidence=0.95)
    v = to_validated_policy(p, validation_notes=["domain ok"])
    assert v.title == "t"
    assert v.validation_notes == ["domain ok"]
    assert not hasattr(v, "confidence")


def test_dump_for_audit_excludes_raw_text():
    p = ExtractedPolicy(
        **_base_kwargs(raw_text="원문 텍스트"),
        source_file_hash="abc123",
        confidence=0.95,
    )
    dumped = p.dump_for_audit()
    assert "raw_text" not in dumped
    assert dumped["raw_text_ref"] == "abc123"
