"""
models.py
=========
정책 추출 파이프라인의 도메인 모델.

3단 계층:
  PolicyDocument   원문 + 해시 + 메타
       ↓ (LLM 추출)
  ExtractedPolicy  + confidence / ambiguous_fields / missing_fields / review_reasons
       ↓ (validator 통과)
  ValidatedPolicy  검증 메타만 남음. downstream(graph, simulation) 은 이 타입만 본다.

규칙:
- 이 파일은 **순수한 데이터 형식 + 자기일관성 검증**만 책임진다.
- 도메인 규칙(자치구 화이트리스트, 어휘 사전 등)은 `validator.py` 와
  `vocabulary.py` 가 담당한다. 여기서는 중복 정의하지 않는다.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# 원문
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 공통 정책 베이스
# ---------------------------------------------------------------------------
class PolicyModel(BaseModel):
    """추출/검증 단계 양쪽이 공유하는 정책의 핵심 필드.

    여기서는 **데이터 형태의 자기일관성**만 검증한다 (날짜 순서, 금액/비율 범위).
    도메인 어휘 검증("강남구"가 실제 서울 자치구인지 등)은 validator 책임.
    """
    model_config = ConfigDict(extra="ignore")

    policy_id: str
    title: str
    summary: str
    source_file: str
    source_file_hash: str | None = None

    # raw_text 는 audit 용. JSONL 로그에는 기본적으로 제외(`dump_for_audit()` 사용).
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
    def _require_non_blank(cls, value: str) -> str:
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
    def _strip_blank_items(cls, values: list[str]) -> list[str]:
        return [v.strip() for v in values if v and v.strip()]

    @model_validator(mode="after")
    def _check_date_range(self) -> PolicyModel:
        if (
            self.effective_start_date is not None
            and self.effective_end_date is not None
            and self.effective_start_date > self.effective_end_date
        ):
            raise ValueError("effective_start_date must be before effective_end_date")
        return self

    @model_validator(mode="after")
    def _check_benefit_values(self) -> PolicyModel:
        if self.benefit_amount is not None and self.benefit_amount < 0:
            raise ValueError("benefit_amount must be >= 0")
        if self.benefit_rate is not None and not 0 <= self.benefit_rate <= 1:
            raise ValueError("benefit_rate must be in [0, 1]")
        return self

    def dump_for_audit(self) -> dict:
        """JSONL 감사 로그용 직렬화. 원문 raw_text 는 해시 참조로 대체.

        원문 자체는 `data/policies/processed/`(성공) 또는 `failed/`(실패) 폴더에
        파일로 보존된다. 로그가 비대해지는 것을 막기 위함.
        """
        data = self.model_dump(mode="json", exclude={"raw_text"})
        data["raw_text_ref"] = self.source_file_hash
        return data


# ---------------------------------------------------------------------------
# LLM 추출 결과
# ---------------------------------------------------------------------------
MIN_CONFIDENCE = 0.7


class ExtractedPolicy(PolicyModel):
    """LLM 이 채워준 자기보고 메타가 붙은 타입.

    validator 가 검증을 통과시키지 못하면 다음 단계로 못 간다.
    """

    confidence: float = Field(ge=0, le=1)
    ambiguous_fields: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    requires_human_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)

    @field_validator("ambiguous_fields", "missing_fields", "review_reasons")
    @classmethod
    def _strip_blank_field_names(cls, values: list[str]) -> list[str]:
        return [v.strip() for v in values if v and v.strip()]

    @model_validator(mode="after")
    def _flag_low_confidence(self) -> ExtractedPolicy:
        """confidence 와 자기보고 필드를 보고 자동으로 review 플래그 부여.

        이 메서드는 **모델 차원의 최소 안전망**이다. 실제 도메인 룰(어휘 사전,
        자치구 화이트리스트 등)은 validator 가 추가로 reason 을 누적한다.
        """
        reasons: list[str] = list(self.review_reasons)

        if self.confidence < MIN_CONFIDENCE:
            reasons.append(
                f"confidence {self.confidence:.2f} < {MIN_CONFIDENCE} threshold"
            )
        if self.ambiguous_fields:
            reasons.append(f"ambiguous fields: {', '.join(self.ambiguous_fields)}")
        if self.missing_fields:
            reasons.append(f"missing fields: {', '.join(self.missing_fields)}")

        if reasons:
            # bypass re-validation by writing through __dict__
            self.__dict__["requires_human_review"] = True
            self.__dict__["review_reasons"] = _dedupe_keep_order(reasons)
        return self


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# 검증 완료 타입 (downstream 진입점)
# ---------------------------------------------------------------------------
class ValidatedPolicy(PolicyModel):
    validated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    validation_notes: list[str] = Field(default_factory=list)

    @field_validator("validation_notes")
    @classmethod
    def _strip_blank_notes(cls, values: list[str]) -> list[str]:
        return [v.strip() for v in values if v and v.strip()]


# ---------------------------------------------------------------------------
# 변환 헬퍼
# ---------------------------------------------------------------------------
def to_validated_policy(
    extracted: ExtractedPolicy,
    validation_notes: list[str] | None = None,
) -> ValidatedPolicy:
    """ExtractedPolicy → ValidatedPolicy.

    requires_human_review 가 True 면 변환 거부.
    호출자(검증기)는 이 조건을 미리 분기해야 한다.
    """
    if extracted.requires_human_review:
        raise ValueError("cannot promote a policy that requires human review")

    return ValidatedPolicy(
        **extracted.model_dump(
            exclude={
                "confidence",
                "ambiguous_fields",
                "missing_fields",
                "requires_human_review",
                "review_reasons",
            }
        ),
        validation_notes=validation_notes or [],
    )


# Backward-compat alias used by extractor.py (구 이름 유지)
validate_extracted_policy = to_validated_policy
