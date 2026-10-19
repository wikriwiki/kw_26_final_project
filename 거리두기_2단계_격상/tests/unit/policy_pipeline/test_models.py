"""models: Pydantic 자기일관성 + 자동 review 플래그 + curated JSON 변환."""

from datetime import date

import pytest
from pydantic import ValidationError

from scripts.policy_pipeline.models import (
    ExtractedPolicy,
    PolicyModel,
    PolicyType,
    extracted_from_curated_json,
    to_validated_policy,
)


def _base(**overrides):
    base = {
        "policy_id": "policy_x",
        "title": "t",
        "summary": "s",
        "source_file": "/tmp/x.txt",
        "target_districts": ["강남구"],
        "benefit_categories": ["식사"],
        "policy_type": PolicyType.COUPON,
    }
    base.update(overrides)
    return base


def test_inverted_date_range_rejected():
    with pytest.raises(ValidationError):
        PolicyModel(
            **_base(
                effective_from=date(2026, 6, 1),
                effective_until=date(2026, 5, 1),
            )
        )


def test_negative_cap_rejected():
    with pytest.raises(ValidationError):
        PolicyModel(**_base(cap_per_agent=-1))


def test_benefit_rate_outside_unit_interval_rejected():
    with pytest.raises(ValidationError):
        PolicyModel(**_base(benefit_rate=1.5))


def test_low_confidence_flags_review():
    p = ExtractedPolicy(**_base(), confidence=0.5)
    assert p.requires_human_review is True
    assert any("confidence" in r for r in p.review_reasons)


def test_ambiguous_field_flags_review():
    p = ExtractedPolicy(**_base(), confidence=0.95, ambiguous_fields=["benefit_categories"])
    assert p.requires_human_review is True


def test_to_validated_refuses_review():
    p = ExtractedPolicy(**_base(), confidence=0.5)
    with pytest.raises(ValueError):
        to_validated_policy(p)


def test_to_validated_strips_extraction_meta():
    p = ExtractedPolicy(**_base(), confidence=0.95)
    v = to_validated_policy(p, validation_notes=["ok"])
    assert v.title == "t"
    assert v.validation_notes == ["ok"]
    assert not hasattr(v, "confidence")


def test_dump_for_audit_excludes_raw_text():
    p = ExtractedPolicy(**_base(raw_text="원문"), source_file_hash="abc123", confidence=0.95)
    d = p.dump_for_audit()
    assert "raw_text" not in d
    assert d["raw_text_ref"] == "abc123"


def test_curated_json_to_extracted_p007():
    """data/neo4j_load/policies/P007.json 포맷 호환성 회귀 테스트."""
    payload = {
        "id": "P007",
        "name": "서울시민 소상공인 응원 쿠폰",
        "type": "subsidy",
        "description": "서울 전체",
        "benefit_rate": 1.0,
        "cap_per_agent": 100000,
        "announce_date": "2026-04-25",
        "effective_from": "2026-05-01",
        "effective_until": "2026-06-30",
        "target_districts": ["강남구", "마포구"],
        "benefit_categories": [],
    }
    e = extracted_from_curated_json(payload, source_file="P007.json", file_hash="dead" * 16)
    assert e.policy_id == "P007"
    assert e.title.startswith("서울시민")
    assert e.policy_type == PolicyType.SUBSIDY
    assert e.benefit_rate == 1.0
    assert e.cap_per_agent == 100000
    assert e.effective_from == date(2026, 5, 1)
    assert e.target_districts == ["강남구", "마포구"]
    assert e.benefit_categories == []
    assert e.confidence == 1.0
    assert e.requires_human_review is False
