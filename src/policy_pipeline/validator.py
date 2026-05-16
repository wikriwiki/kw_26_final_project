"""
validator.py
============
ExtractedPolicy 에 **도메인 어휘 룰**을 적용해 다음 중 하나로 분기:

  - VALIDATED      → 모든 도메인 룰 통과 → ValidatedPolicy 생성
  - NEEDS_REVIEW   → 사람 검토 필요 (모호어, 비매핑 어휘 등)
  - FAILED         → 데이터 자체가 깨짐 (현재 이 단계에 도달했다면 거의 없음.
                      Pydantic 이 model_validator 단계에서 이미 잡아냄)

이 모듈은 **도메인 룰만** 본다. 데이터 자기일관성(date 순서, benefit 범위 등)은
`models.py` 의 Pydantic 검증이 책임지므로 여기서 중복 검사하지 않는다.

어휘 사전은 `vocabulary.py` 한 곳에서 import — 단일 진실의 원천.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from src.policy_pipeline.models import (
    ExtractedPolicy,
    ValidatedPolicy,
    to_validated_policy,
)
from src.policy_pipeline.state import PolicyStatus, append_policy_status
from src.policy_pipeline.vocabulary import (
    VALID_INDUSTRY_TERMS,
    VALID_TARGET_GROUP_TERMS,
    contains_any_term,
    filter_invalid_districts,
    has_ambiguous_scope,
    is_national_scope,
    is_seoul_wide_scope,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VALIDATION_LOG_PATH = PROJECT_ROOT / "output" / "logs" / "policy_validation.jsonl"
DEFAULT_EXTRACTED_POLICY_LOG_PATH = (
    PROJECT_ROOT / "output" / "logs" / "extracted_policies.jsonl"
)


# ---------------------------------------------------------------------------
# 결과 모델
# ---------------------------------------------------------------------------
class ValidationSeverity(str, Enum):
    ERROR = "error"                # 데이터 자체가 깨짐 → FAILED
    NEEDS_REVIEW = "needs_review"  # 사람 판단 필요


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
# 메인 API
# ---------------------------------------------------------------------------
def validate_policy(extracted: ExtractedPolicy) -> PolicyValidationOutcome:
    """도메인 룰 적용 결과를 outcome 으로 돌려준다.

    Pydantic 이 이미 거른 자기일관성 위반은 여기 도달 못 한다.
    """
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
    """검증 + JSONL 로그 + state.py status append 까지 한번에."""
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
# 도메인 룰
# ---------------------------------------------------------------------------
def _domain_checks(extracted: ExtractedPolicy) -> list[PolicyValidationIssue]:
    issues: list[PolicyValidationIssue] = []

    # 1) 자치구 화이트리스트
    invalid = filter_invalid_districts(extracted.target_districts)
    if invalid:
        issues.append(_review(
            "target_districts",
            f"Unknown Seoul district names: {', '.join(invalid)}",
        ))

    # 2) 지역 스코프가 서울로 매핑 가능한가
    if not extracted.target_districts and not is_seoul_wide_scope(extracted.target_regions):
        issues.append(_review(
            "target_regions",
            "Policy region is not mapped to Seoul districts or a clear Seoul-wide scope",
        ))

    # 3) 전국 범위 (서울 표현 없이) — 매핑 작업 필요
    if is_national_scope(extracted.target_regions):
        issues.append(_review(
            "target_regions",
            "Broad national scope needs review before mapping to Seoul simulation units",
        ))

    # 4) 모호어
    if has_ambiguous_scope(_all_scope_values(extracted)):
        issues.append(_review(
            "scope",
            "Policy scope is broad or vague and needs human interpretation",
        ))

    # 5) 업종 화이트리스트 (있을 때만)
    if extracted.target_industries and not contains_any_term(
        extracted.target_industries, VALID_INDUSTRY_TERMS,
    ):
        issues.append(_review(
            "target_industries",
            "Industry names do not match the current local policy vocabulary",
        ))

    # 6) 대상 그룹 화이트리스트 (있을 때만)
    if extracted.target_groups and not contains_any_term(
        extracted.target_groups, VALID_TARGET_GROUP_TERMS,
    ):
        issues.append(_review(
            "target_groups",
            "Target groups do not match the current local policy vocabulary",
        ))

    return issues


def _all_scope_values(extracted: ExtractedPolicy) -> list[str]:
    return [
        *extracted.target_regions,
        *extracted.target_districts,
        *extracted.target_industries,
        *extracted.target_groups,
    ]


# ---------------------------------------------------------------------------
# JSONL 로그 (raw_text 제외 — `dump_for_audit()` 사용)
# ---------------------------------------------------------------------------
def _append_extracted_policy_log(
    extracted: ExtractedPolicy,
    log_path: Path | None = None,
) -> None:
    _append_jsonl(log_path, {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "extracted_policy": extracted.dump_for_audit(),
        "confidence": extracted.confidence,
        "review_reasons": list(extracted.review_reasons),
    })


def _append_validation_outcome_log(
    extracted: ExtractedPolicy,
    outcome: PolicyValidationOutcome,
    log_path: Path | None = None,
) -> None:
    _append_jsonl(log_path, {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "policy_id": extracted.policy_id,
        "source_file": extracted.source_file,
        "source_file_hash": extracted.source_file_hash,
        "status": outcome.status.value,
        "issues": [issue.model_dump(mode="json") for issue in outcome.issues],
    })


def _append_jsonl(log_path: Path, payload: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 작은 헬퍼
# ---------------------------------------------------------------------------
def _error(field: str, message: str) -> PolicyValidationIssue:
    return PolicyValidationIssue(field=field, message=message, severity=ValidationSeverity.ERROR)


def _review(field: str, message: str) -> PolicyValidationIssue:
    return PolicyValidationIssue(field=field, message=message, severity=ValidationSeverity.NEEDS_REVIEW)


def _issue_summary(issues: list[PolicyValidationIssue]) -> str | None:
    if not issues:
        return None
    return "; ".join(f"{i.field}: {i.message}" for i in issues)
