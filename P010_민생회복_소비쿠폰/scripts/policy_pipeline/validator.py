"""
validator.py
============
ExtractedPolicy 에 도메인 룰 적용 → VALIDATED / NEEDS_REVIEW / FAILED.

데이터 자기일관성(date, benefit) 은 models.py 의 Pydantic 이 이미 책임진다.
여기는 어휘/매핑 검증만:
  - target_districts 가 서울 25 자치구에 있는가
  - benefit_categories 가 L1 12종에 있는가
  - 모호어가 섞이지 않았는가
  - 전국 범위가 서울로 매핑 가능한가
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from scripts.policy_pipeline.models import (
    ExtractedPolicy,
    ValidatedPolicy,
    to_validated_policy,
)
from scripts.policy_pipeline.state import PolicyStatus, append_policy_status
from scripts.policy_pipeline.vocabulary import (
    filter_invalid_categories,
    filter_invalid_districts,
    has_ambiguous_scope,
    is_national_scope,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VALIDATION_LOG_PATH = PROJECT_ROOT / "output" / "policy_pipeline" / "policy_validation.jsonl"
DEFAULT_EXTRACTED_POLICY_LOG_PATH = (
    PROJECT_ROOT / "output" / "policy_pipeline" / "extracted_policies.jsonl"
)


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


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------
def validate_policy(extracted: ExtractedPolicy) -> PolicyValidationOutcome:
    issues = _domain_checks(extracted)

    if any(i.severity == ValidationSeverity.ERROR for i in issues):
        return PolicyValidationOutcome(status=PolicyStatus.FAILED, issues=issues)

    if issues or extracted.requires_human_review:
        return PolicyValidationOutcome(status=PolicyStatus.NEEDS_REVIEW, issues=issues)

    validated = to_validated_policy(
        extracted,
        validation_notes=["domain checks passed"],
    )
    return PolicyValidationOutcome(
        status=PolicyStatus.VALIDATED,
        issues=issues,
        validated_policy=validated,
    )


def validate_and_record_policy(
    extracted: ExtractedPolicy,
    validation_log_path: Path | None = None,
    extracted_policy_log_path: Path | None = None,
) -> PolicyValidationOutcome:
    validation_log_path = validation_log_path or DEFAULT_VALIDATION_LOG_PATH
    extracted_policy_log_path = extracted_policy_log_path or DEFAULT_EXTRACTED_POLICY_LOG_PATH

    _append_extracted_policy_log(extracted, extracted_policy_log_path)
    outcome = validate_policy(extracted)
    _append_validation_outcome_log(extracted, outcome, validation_log_path)
    append_policy_status(
        policy_id=extracted.policy_id,
        file_hash=extracted.source_file_hash or "",
        source_path=extracted.source_file,
        status=outcome.status,
        error_message=_issue_summary(outcome.issues),
    )
    return outcome


# ---------------------------------------------------------------------------
# Domain rules
# ---------------------------------------------------------------------------
def _domain_checks(extracted: ExtractedPolicy) -> list[PolicyValidationIssue]:
    issues: list[PolicyValidationIssue] = []

    # 1) 자치구 화이트리스트
    invalid_districts = filter_invalid_districts(extracted.target_districts)
    if invalid_districts:
        issues.append(_review(
            "target_districts",
            f"Unknown Seoul districts: {', '.join(invalid_districts)}",
        ))

    # 2) 카테고리 L1 화이트리스트 (있을 때만; 빈 리스트는 '전체 commerce' 의미로 허용)
    invalid_cats = filter_invalid_categories(extracted.benefit_categories)
    if invalid_cats:
        issues.append(_review(
            "benefit_categories",
            f"Unknown L1 categories: {', '.join(invalid_cats)}. "
            "Use one of: 식사, 카페, 디저트, 주점, 편의점, 마트, 미용, 쇼핑, 여가, 건강, 교육, 기타",
        ))

    # 3) 전국 범위 (서울 매핑 X) — review
    if is_national_scope([extracted.summary, extracted.title]):
        issues.append(_review(
            "scope",
            "Broad national scope detected — needs review before mapping to Seoul",
        ))

    # 4) 모호어
    if has_ambiguous_scope([
        *extracted.target_districts, *extracted.target_dongs,
        *extracted.benefit_categories, *extracted.target_groups,
    ]):
        issues.append(_review(
            "scope",
            "Policy scope contains ambiguous terms — needs human interpretation",
        ))

    return issues


# ---------------------------------------------------------------------------
# JSONL logs
# ---------------------------------------------------------------------------
def _append_extracted_policy_log(extracted: ExtractedPolicy, log_path: Path) -> None:
    _append_jsonl(log_path, {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "extracted_policy": extracted.dump_for_audit(),
        "confidence": extracted.confidence,
        "review_reasons": list(extracted.review_reasons),
    })


def _append_validation_outcome_log(
    extracted: ExtractedPolicy,
    outcome: PolicyValidationOutcome,
    log_path: Path,
) -> None:
    _append_jsonl(log_path, {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "policy_id": extracted.policy_id,
        "source_file": extracted.source_file,
        "source_file_hash": extracted.source_file_hash,
        "status": outcome.status.value,
        "issues": [i.model_dump(mode="json") for i in outcome.issues],
    })


def _append_jsonl(log_path: Path, payload: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _error(field: str, message: str) -> PolicyValidationIssue:
    return PolicyValidationIssue(field=field, message=message, severity=ValidationSeverity.ERROR)


def _review(field: str, message: str) -> PolicyValidationIssue:
    return PolicyValidationIssue(field=field, message=message, severity=ValidationSeverity.NEEDS_REVIEW)


def _issue_summary(issues: list[PolicyValidationIssue]) -> str | None:
    if not issues:
        return None
    return "; ".join(f"{i.field}: {i.message}" for i in issues)
