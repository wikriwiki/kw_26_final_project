from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from src.policy_pipeline.models import (
    ExtractedPolicy,
    ValidatedPolicy,
    validate_extracted_policy,
)
from src.policy_pipeline.state import PolicyStatus, append_policy_status


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VALIDATION_LOG_PATH = PROJECT_ROOT / "output" / "logs" / "policy_validation.jsonl"
DEFAULT_EXTRACTED_POLICY_LOG_PATH = (
    PROJECT_ROOT / "output" / "logs" / "extracted_policies.jsonl"
)
MIN_CONFIDENCE = 0.7

SEOUL_DISTRICTS = {
    "강남구",
    "강동구",
    "강북구",
    "강서구",
    "관악구",
    "광진구",
    "구로구",
    "금천구",
    "노원구",
    "도봉구",
    "동대문구",
    "동작구",
    "마포구",
    "서대문구",
    "서초구",
    "성동구",
    "성북구",
    "송파구",
    "양천구",
    "영등포구",
    "용산구",
    "은평구",
    "종로구",
    "중구",
    "중랑구",
}
SEOUL_REGION_TERMS = {"서울", "서울시", "서울특별시", "서울 전체", "서울시 전체", "서울 전역"}
BROAD_REGION_TERMS = {"전국", "대한민국", "국내 전체", "전 지역"}
AMBIGUOUS_SCOPE_TERMS = {
    "일부",
    "일부 지역",
    "일부 업종",
    "관련 지역",
    "관련 업종",
    "해당 지역",
    "해당 업종",
    "지역 상권",
    "인근 지역",
    "주요 상권",
    "취약 지역",
}
VALID_INDUSTRY_TERMS = {
    "전체",
    "전체 업종",
    "소상공인",
    "전통시장",
    "음식점",
    "외식업",
    "카페",
    "숙박업",
    "도소매",
    "도소매업",
    "서비스업",
    "관광업",
    "문화",
    "공연",
    "편의점",
}
VALID_TARGET_GROUP_TERMS = {
    "전체",
    "전체 시민",
    "서울시민",
    "주민",
    "소상공인",
    "자영업자",
    "청년",
    "노인",
    "어르신",
    "저소득층",
    "취약계층",
    "관광객",
    "학생",
    "가구",
}
REQUIRED_POLICY_FIELDS = {
    "title",
    "summary",
    "source_file",
    "target_regions",
    "target_districts",
    "target_industries",
}


class ValidationSeverity(str, Enum):
    ERROR = "error"
    NEEDS_REVIEW = "needs_review"


class PolicyValidationIssue(BaseModel):
    field: str
    message: str
    severity: ValidationSeverity


class PolicyValidationOutcome(BaseModel):
    status: PolicyStatus
    issues: list[PolicyValidationIssue] = Field(default_factory=list)
    validated_policy: ValidatedPolicy | None = None
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def validate_policy(extracted_policy: ExtractedPolicy) -> PolicyValidationOutcome:
    issues = [
        *_validate_structure(extracted_policy),
        *_validate_domain(extracted_policy),
    ]

    if any(issue.severity == ValidationSeverity.ERROR for issue in issues):
        return PolicyValidationOutcome(status=PolicyStatus.FAILED, issues=issues)

    if issues or extracted_policy.requires_human_review:
        return PolicyValidationOutcome(status=PolicyStatus.NEEDS_REVIEW, issues=issues)

    validated_policy = validate_extracted_policy(
        extracted_policy,
        validation_notes=["Pydantic structure checks and local domain checks passed."],
    )
    return PolicyValidationOutcome(
        status=PolicyStatus.VALIDATED,
        issues=issues,
        validated_policy=validated_policy,
    )


def validate_and_record_policy(
    extracted_policy: ExtractedPolicy,
    validation_log_path: Path = DEFAULT_VALIDATION_LOG_PATH,
    extracted_policy_log_path: Path = DEFAULT_EXTRACTED_POLICY_LOG_PATH,
) -> PolicyValidationOutcome:
    append_extracted_policy_log(extracted_policy, extracted_policy_log_path)
    outcome = validate_policy(extracted_policy)
    append_validation_outcome_log(extracted_policy, outcome, validation_log_path)
    append_policy_status(
        policy_id=extracted_policy.policy_id,
        file_hash=extracted_policy.source_file_hash or "",
        source_path=extracted_policy.source_file,
        status=outcome.status,
        error_message=_issue_summary(outcome.issues),
    )
    return outcome


def append_extracted_policy_log(
    extracted_policy: ExtractedPolicy,
    log_path: Path = DEFAULT_EXTRACTED_POLICY_LOG_PATH,
) -> None:
    _append_jsonl(
        log_path,
        {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "extracted_policy": extracted_policy.model_dump(mode="json"),
        },
    )


def append_validation_outcome_log(
    extracted_policy: ExtractedPolicy,
    outcome: PolicyValidationOutcome,
    log_path: Path = DEFAULT_VALIDATION_LOG_PATH,
) -> None:
    _append_jsonl(
        log_path,
        {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "policy_id": extracted_policy.policy_id,
            "source_file": extracted_policy.source_file,
            "status": outcome.status.value,
            "issues": [issue.model_dump(mode="json") for issue in outcome.issues],
        },
    )


def _validate_structure(extracted_policy: ExtractedPolicy) -> list[PolicyValidationIssue]:
    issues: list[PolicyValidationIssue] = []

    if extracted_policy.effective_start_date and extracted_policy.effective_end_date:
        if extracted_policy.effective_start_date > extracted_policy.effective_end_date:
            issues.append(
                _error(
                    "effective_start_date",
                    "effective_start_date must be before effective_end_date.",
                )
            )

    if extracted_policy.benefit_amount is not None and extracted_policy.benefit_amount < 0:
        issues.append(_error("benefit_amount", "benefit_amount must not be negative."))

    if extracted_policy.benefit_rate is not None and not 0 <= extracted_policy.benefit_rate <= 1:
        issues.append(_error("benefit_rate", "benefit_rate must be between 0 and 1."))

    missing_required = REQUIRED_POLICY_FIELDS.intersection(extracted_policy.missing_fields)
    for field in sorted(missing_required):
        issues.append(_review(field, "Required field is missing from the source document."))

    if extracted_policy.confidence < MIN_CONFIDENCE:
        issues.append(_review("confidence", "LLM confidence is below the review threshold."))

    for field in sorted(set(extracted_policy.ambiguous_fields)):
        issues.append(_review(field, "LLM marked this field as ambiguous."))

    return issues


def _validate_domain(extracted_policy: ExtractedPolicy) -> list[PolicyValidationIssue]:
    issues: list[PolicyValidationIssue] = []

    invalid_districts = [
        district
        for district in extracted_policy.target_districts
        if district not in SEOUL_DISTRICTS
    ]
    if invalid_districts:
        issues.append(
            _review(
                "target_districts",
                f"Unknown Seoul district names: {', '.join(invalid_districts)}.",
            )
        )

    if not extracted_policy.target_districts and not _contains_any_term(
        extracted_policy.target_regions,
        SEOUL_REGION_TERMS,
    ):
        issues.append(
            _review(
                "target_regions",
                "Policy region is not mapped to Seoul districts or a clear Seoul-wide scope.",
            )
        )

    if _contains_any_term(extracted_policy.target_regions, BROAD_REGION_TERMS):
        issues.append(
            _review(
                "target_regions",
                "Broad national scope needs review before mapping to Seoul simulation units.",
            )
        )

    if _contains_any_term(_all_scope_values(extracted_policy), AMBIGUOUS_SCOPE_TERMS):
        issues.append(
            _review(
                "scope",
                "Policy scope is broad or vague and needs human interpretation.",
            )
        )

    if extracted_policy.target_industries and not _contains_any_term(
        extracted_policy.target_industries,
        VALID_INDUSTRY_TERMS,
    ):
        issues.append(
            _review(
                "target_industries",
                "Industry names do not match the current local policy vocabulary.",
            )
        )

    if extracted_policy.target_groups and not _contains_any_term(
        extracted_policy.target_groups,
        VALID_TARGET_GROUP_TERMS,
    ):
        issues.append(
            _review(
                "target_groups",
                "Target groups do not match the current local policy vocabulary.",
            )
        )

    return issues


def _all_scope_values(extracted_policy: ExtractedPolicy) -> list[str]:
    return [
        *extracted_policy.target_regions,
        *extracted_policy.target_districts,
        *extracted_policy.target_industries,
        *extracted_policy.target_groups,
    ]


def _contains_any_term(values: list[str], terms: set[str]) -> bool:
    normalized_values = [value.replace(" ", "").lower() for value in values]
    normalized_terms = [term.replace(" ", "").lower() for term in terms]
    return any(
        term in value
        for value in normalized_values
        for term in normalized_terms
    )


def _error(field: str, message: str) -> PolicyValidationIssue:
    return PolicyValidationIssue(
        field=field,
        message=message,
        severity=ValidationSeverity.ERROR,
    )


def _review(field: str, message: str) -> PolicyValidationIssue:
    return PolicyValidationIssue(
        field=field,
        message=message,
        severity=ValidationSeverity.NEEDS_REVIEW,
    )


def _issue_summary(issues: list[PolicyValidationIssue]) -> str | None:
    if not issues:
        return None
    return "; ".join(f"{issue.field}: {issue.message}" for issue in issues)


def _append_jsonl(log_path: Path, payload: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")
