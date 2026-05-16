"""
extractor.py
============
정책 문서(자연어) → ExtractedPolicy 변환.

Structured Output 사용 (`llm_client.call_chat_structured`). jinja2 의존성 제거,
프롬프트를 모듈 내 상수로.

입력 메타(`policy_id`, `source_file`, `source_file_hash`, `raw_text`) 는 LLM 응답을
무시하고 원문 기준으로 덮어쓴다 (환각 차단).
"""

from __future__ import annotations

import json
from typing import Protocol

from pydantic import ValidationError

from scripts.policy_pipeline.models import (
    MIN_CONFIDENCE,
    ExtractedPolicy,
    PolicyDocument,
    to_validated_policy,
)
from scripts.policy_pipeline.vocabulary import (
    REVIEW_TRIGGER_FIELDS,
    has_ambiguous_scope,
    is_national_scope,
)


# ---------------------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You extract structured policy data from Korean policy documents for a Seoul
commercial-behavior simulation.

Rules:
- Return JSON only.
- Use null for unknown scalars, [] for unknown lists.
- benefit_rate is a decimal in [0,1]. 10% → 0.1.
- Dates in ISO YYYY-MM-DD.
- target_districts MUST be Seoul 자치구 names (강남구, 마포구, ...).
- target_dongs are 행정동 names if explicitly stated. Otherwise leave empty.
- benefit_categories use the L1 vocabulary: 식사, 카페, 디저트, 주점,
  편의점, 마트, 미용, 쇼핑, 여가, 건강, 교육, 기타.
  If the policy applies to "all commerce" or doesn't restrict by category,
  return an empty list [].
- If the policy is national ("전국") with no clear Seoul mapping, set
  requires_human_review=true and explain in review_reasons.
- If the scope contains ambiguous words like "일부", "주요 상권", set
  requires_human_review=true.
"""


def build_user_prompt(document: PolicyDocument) -> str:
    return (
        f"Policy document metadata (do not invent — copy verbatim):\n"
        f'{{"policy_id": "policy_{document.file_hash[:12]}", '
        f'"source_file": {json.dumps(document.source_file, ensure_ascii=False)}, '
        f'"source_file_hash": {json.dumps(document.file_hash)}}}\n\n'
        f"Policy document text:\n{document.raw_text}\n"
    )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class PolicyExtractionError(RuntimeError):
    """추출 단계 전반의 베이스 에러."""


class PolicyJsonParseError(PolicyExtractionError):
    pass


class PolicyValidationError(PolicyExtractionError):
    pass


# ---------------------------------------------------------------------------
# LLM 클라이언트 Protocol — 테스트가 쉽게 mock 할 수 있도록
# ---------------------------------------------------------------------------
class StructuredLLMClient(Protocol):
    def call_chat_structured(
        self,
        response_model: type[ExtractedPolicy],
        system_prompt: str,
        user_prompt: str,
    ) -> ExtractedPolicy: ...


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------
def extract_policy(
    document: PolicyDocument,
    *,
    client: StructuredLLMClient | None = None,
) -> ExtractedPolicy:
    """원문 문서를 ExtractedPolicy 로 추출.

    `client` 가 None 이면 `llm_client.call_chat_structured` 를 직접 호출한다.
    """
    user_prompt = build_user_prompt(document)

    try:
        if client is not None:
            extracted = client.call_chat_structured(
                ExtractedPolicy, SYSTEM_PROMPT, user_prompt,
            )
        else:
            from scripts.policy_pipeline.llm_client import call_chat_structured
            extracted = call_chat_structured(
                ExtractedPolicy, SYSTEM_PROMPT, user_prompt,
            )
    except ValidationError as exc:
        raise PolicyValidationError(f"structured output failed validation: {exc}") from exc

    # 신원 메타 강제 (LLM 환각 차단)
    extracted.__dict__["policy_id"] = f"policy_{document.file_hash[:12]}"
    extracted.__dict__["source_file"] = document.source_file
    extracted.__dict__["source_file_hash"] = document.file_hash
    extracted.__dict__["raw_text"] = document.raw_text

    return _enrich_review_reasons(extracted)


def extract_and_validate_policy(
    document: PolicyDocument,
    *,
    client: StructuredLLMClient | None = None,
):
    """추출 + 모델 차원 변환만. 도메인 룰은 validator.validate_policy 가 한다."""
    extracted = extract_policy(document, client=client)
    return to_validated_policy(extracted)


# ---------------------------------------------------------------------------
# Review reason 누적 — LLM 자기보고 기반의 1차 필터
# ---------------------------------------------------------------------------
def _enrich_review_reasons(extracted: ExtractedPolicy) -> ExtractedPolicy:
    reasons: list[str] = list(extracted.review_reasons)

    trigger_amb = REVIEW_TRIGGER_FIELDS.intersection(extracted.ambiguous_fields)
    if trigger_amb:
        reasons.append(f"trigger-field ambiguous: {', '.join(sorted(trigger_amb))}")

    trigger_miss = REVIEW_TRIGGER_FIELDS.intersection(extracted.missing_fields)
    if trigger_miss:
        reasons.append(f"trigger-field missing: {', '.join(sorted(trigger_miss))}")

    scope_values = [
        *extracted.target_districts,
        *extracted.target_dongs,
        *extracted.benefit_categories,
        *extracted.target_groups,
    ]
    if has_ambiguous_scope(scope_values):
        reasons.append("scope contains ambiguous terms")

    # target_districts 비어 있고 dong 도 비어 있으면 매핑 불가 (서울 전역은 별도 처리)
    if not extracted.target_districts and not extracted.target_dongs:
        reasons.append("no target districts or dongs")

    if extracted.confidence < MIN_CONFIDENCE:
        reasons.append(f"confidence {extracted.confidence:.2f} < {MIN_CONFIDENCE}")

    if reasons:
        extracted.__dict__["requires_human_review"] = True
        seen: set[str] = set()
        deduped: list[str] = []
        for r in reasons:
            if r not in seen:
                seen.add(r)
                deduped.append(r)
        extracted.__dict__["review_reasons"] = deduped

    return extracted
