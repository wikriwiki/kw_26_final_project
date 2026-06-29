"""
models.py
=========
정책 추출 파이프라인 도메인 모델 — agent_persona 브랜치 Neo4j 스키마와 정합.

대응 표 (Pydantic ↔ Neo4j :Policy 노드 속성):

  policy_id           ↔  id                                (str, UNIQUE)
  title               ↔  name                              (str)
  policy_type         ↔  type                              ("subsidy" / "coupon" / ...)
  summary/description ↔  description                       (str)
  benefit_rate        ↔  benefit_rate                      (float 0..1)
  cap_per_agent       ↔  cap_per_agent                     (int, KRW)
  announce_date       ↔  announce_date                     (date)
  effective_from      ↔  effective_from                    (date)
  effective_until     ↔  effective_until                   (date)
  target_districts    ↔  :applied_to → (:District {name})
  benefit_categories  ↔  :targets    → (:Category {parent})   (L1 카테고리 이름)
  source_file_hash    ↔  raw_json_ref                      (파일 해시)

3단 Pydantic 계층 유지: PolicyDocument → ExtractedPolicy → ValidatedPolicy.
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


# ---------------------------------------------------------------------------
# 정책 유형 (Neo4j :Policy.type 와 매칭)
# ---------------------------------------------------------------------------
class PolicyType(str, Enum):
    SUBSIDY = "subsidy"        # 보조금/지원금/쿠폰
    COUPON = "coupon"
    DISCOUNT = "discount"
    TAX_RELIEF = "tax_relief"
    LOAN = "loan"
    VOUCHER = "voucher"
    GRANT = "grant"            # 현금 일시 지급 (소득별 차등 등)
    FACILITY = "facility"      # 시설 정책 (보행친화거리 등)
    OTHER = "other"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# 공통 베이스
# ---------------------------------------------------------------------------
class PolicyModel(BaseModel):
    """Neo4j :Policy 노드와 1:1 매핑되는 형태.

    데이터 자기일관성 검증만 여기서. 어휘/도메인 검증은 validator 책임.
    """
    model_config = ConfigDict(extra="ignore")

    policy_id: str
    title: str
    summary: str
    source_file: str
    source_file_hash: str | None = None
    raw_text: str | None = None

    policy_type: PolicyType = PolicyType.UNKNOWN

    # 발효 기간
    announce_date: date | None = None
    effective_from: date | None = None
    effective_until: date | None = None

    # 적용 범위 (Neo4j 엣지로 풀림)
    target_districts: list[str] = Field(default_factory=list)
    target_dongs: list[str] = Field(default_factory=list)
    benefit_categories: list[str] = Field(default_factory=list)  # L1 카테고리 이름

    # 혜택
    benefit_rate: float | None = None     # 0..1
    cap_per_agent: int | None = None      # KRW

    # 부가
    target_groups: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    restrictions: list[str] = Field(default_factory=list)

    @field_validator("policy_id", "title", "summary", "source_file")
    @classmethod
    def _require_non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator(
        "target_districts", "target_dongs",
        "benefit_categories", "target_groups",
        "conditions", "restrictions",
    )
    @classmethod
    def _strip_blank_items(cls, values: list[str]) -> list[str]:
        return [v.strip() for v in values if v and v.strip()]

    @model_validator(mode="after")
    def _check_date_range(self) -> PolicyModel:
        if (
            self.effective_from is not None
            and self.effective_until is not None
            and self.effective_from > self.effective_until
        ):
            raise ValueError("effective_from must be <= effective_until")
        return self

    @model_validator(mode="after")
    def _check_benefit_values(self) -> PolicyModel:
        if self.cap_per_agent is not None and self.cap_per_agent < 0:
            raise ValueError("cap_per_agent must be >= 0")
        if self.benefit_rate is not None and not 0 <= self.benefit_rate <= 1:
            raise ValueError("benefit_rate must be in [0, 1]")
        return self

    def dump_for_audit(self) -> dict:
        """JSONL 감사 로그용. raw_text 는 해시 참조로 대체."""
        data = self.model_dump(mode="json", exclude={"raw_text"})
        data["raw_text_ref"] = self.source_file_hash
        return data


# ---------------------------------------------------------------------------
# LLM 추출 결과
# ---------------------------------------------------------------------------
MIN_CONFIDENCE = 0.7


class ExtractedPolicy(PolicyModel):
    """LLM 자기보고 메타가 붙은 타입."""

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
        reasons: list[str] = list(self.review_reasons)

        if self.confidence < MIN_CONFIDENCE:
            reasons.append(f"confidence {self.confidence:.2f} < {MIN_CONFIDENCE}")
        if self.ambiguous_fields:
            reasons.append(f"ambiguous fields: {', '.join(self.ambiguous_fields)}")
        if self.missing_fields:
            reasons.append(f"missing fields: {', '.join(self.missing_fields)}")

        if reasons:
            self.__dict__["requires_human_review"] = True
            self.__dict__["review_reasons"] = _dedupe(reasons)
        return self


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ---------------------------------------------------------------------------
# 검증 통과 타입 (Neo4j 적재 진입점)
# ---------------------------------------------------------------------------
class ValidatedPolicy(PolicyModel):
    validated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    validation_notes: list[str] = Field(default_factory=list)

    @field_validator("validation_notes")
    @classmethod
    def _strip_blank_notes(cls, values: list[str]) -> list[str]:
        return [v.strip() for v in values if v and v.strip()]


# ---------------------------------------------------------------------------
# 변환
# ---------------------------------------------------------------------------
def to_validated_policy(
    extracted: ExtractedPolicy,
    validation_notes: list[str] | None = None,
) -> ValidatedPolicy:
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


# ---------------------------------------------------------------------------
# 외부 JSON (P007.json 등) → ExtractedPolicy 변환
# ---------------------------------------------------------------------------
def extracted_from_curated_json(payload: dict, *, source_file: str, file_hash: str) -> ExtractedPolicy:
    """수기로 검수된 JSON (data/neo4j_load/policies/*.json) 을 ExtractedPolicy 로.

    P007.json 같은 외부 입력 포맷을 우리 모델로 변환.
    LLM 을 호출하지 않고 manual injection path 에서 사용.
    """
    # 외부 포맷 → 우리 필드명
    title = payload.get("name") or payload.get("title") or "unnamed policy"
    summary = payload.get("description") or payload.get("summary") or title
    return ExtractedPolicy(
        policy_id=payload.get("id") or f"policy_{file_hash[:12]}",
        title=title,
        summary=summary,
        source_file=source_file,
        source_file_hash=file_hash,
        raw_text=None,
        policy_type=PolicyType(payload.get("type", "unknown")) if _is_known_type(payload.get("type")) else PolicyType.UNKNOWN,
        announce_date=_parse_date(payload.get("announce_date")),
        effective_from=_parse_date(payload.get("effective_from")),
        effective_until=_parse_date(payload.get("effective_until")),
        target_districts=list(payload.get("target_districts") or []),
        target_dongs=list(payload.get("target_dongs") or []),
        benefit_categories=list(payload.get("benefit_categories") or []),
        benefit_rate=payload.get("benefit_rate"),
        cap_per_agent=payload.get("cap_per_agent"),
        target_groups=list(payload.get("target_groups") or []),
        conditions=list(payload.get("conditions") or []),
        restrictions=list(payload.get("restrictions") or []),
        # 수기 검수 JSON 은 신뢰도 100% 로 간주
        confidence=1.0,
        ambiguous_fields=[],
        missing_fields=[],
    )


def _is_known_type(t) -> bool:
    if not isinstance(t, str):
        return False
    try:
        PolicyType(t)
        return True
    except ValueError:
        return False


def _parse_date(value):
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError(f"cannot parse date from {value!r}")


# Backward-compat alias
validate_extracted_policy = to_validated_policy
