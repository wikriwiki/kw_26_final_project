from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class PolicyDocument(BaseModel):
    source_file: str
    file_hash: str
    raw_text: str
    document_type: str
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BenefitType(str, Enum):
    COUPON = "coupon"
    CASH = "cash"
    DISCOUNT = "discount"
    TAX_RELIEF = "tax_relief"
    LOAN = "loan"
    VOUCHER = "voucher"
    OTHER = "other"
    UNKNOWN = "unknown"


class PolicyModel(BaseModel):
    policy_id: str
    title: str
    summary: str
    source_file: str
    source_file_hash: str | None = None
    raw_text: str | None = None
    effective_start_date: date | None = None
    effective_end_date: date | None = None
    target_regions: list[str] = Field(default_factory=list)
    target_districts: list[str] = Field(default_factory=list)
    target_industries: list[str] = Field(default_factory=list)
    target_groups: list[str] = Field(default_factory=list)
    benefit_type: BenefitType = BenefitType.UNKNOWN
    benefit_amount: int | None = None
    benefit_rate: float | None = None
    conditions: list[str] = Field(default_factory=list)
    restrictions: list[str] = Field(default_factory=list)
    expected_behavior_effects: list[str] = Field(default_factory=list)

    @field_validator("policy_id", "title", "summary", "source_file")
    @classmethod
    def require_non_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator(
        "target_regions",
        "target_districts",
        "target_industries",
        "target_groups",
        "conditions",
        "restrictions",
        "expected_behavior_effects",
    )
    @classmethod
    def remove_blank_items(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]

    @model_validator(mode="after")
    def validate_date_range(self) -> PolicyModel:
        if (
            self.effective_start_date is not None
            and self.effective_end_date is not None
            and self.effective_start_date > self.effective_end_date
        ):
            raise ValueError("effective_start_date must be before effective_end_date")
        return self

    @model_validator(mode="after")
    def validate_benefit_values(self) -> PolicyModel:
        if self.benefit_amount is not None and self.benefit_amount < 0:
            raise ValueError("benefit_amount must be greater than or equal to 0")
        if self.benefit_rate is not None and not 0 <= self.benefit_rate <= 1:
            raise ValueError("benefit_rate must be between 0 and 1")
        return self


class ExtractedPolicy(PolicyModel):
    confidence: float = Field(ge=0, le=1)
    ambiguous_fields: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    requires_human_review: bool = False

    @field_validator("ambiguous_fields", "missing_fields")
    @classmethod
    def remove_blank_field_names(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]

    @model_validator(mode="after")
    def flag_low_confidence_for_review(self) -> ExtractedPolicy:
        if self.confidence < 0.7 or self.ambiguous_fields or self.missing_fields:
            self.requires_human_review = True
        return self


class ValidatedPolicy(PolicyModel):
    validated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    validation_notes: list[str] = Field(default_factory=list)

    @field_validator("validation_notes")
    @classmethod
    def remove_blank_validation_notes(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]


def validate_extracted_policy(
    extracted_policy: ExtractedPolicy,
    validation_notes: list[str] | None = None,
) -> ValidatedPolicy:
    if extracted_policy.requires_human_review:
        raise ValueError("cannot validate a policy that requires human review")

    return ValidatedPolicy(
        **extracted_policy.model_dump(
            exclude={
                "confidence",
                "ambiguous_fields",
                "missing_fields",
                "requires_human_review",
            }
        ),
        validation_notes=validation_notes or [],
    )
