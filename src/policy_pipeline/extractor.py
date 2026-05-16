"""
extractor.py
============
정책 문서 → ExtractedPolicy 변환.

핵심 변경 (v2):
1. **Structured Output 사용**. `OpenAIChatClient.complete_structured()` 가 Pydantic
   모델을 그대로 strict JSON schema 로 강제하므로, 더 이상 정규식·코드펜스 파싱이
   필요 없다. 폴백 경로는 `LegacyJsonLLMClient` 로 격리.
2. **어휘 중복 제거**. AMBIGUOUS/NATIONAL/SEOUL_WIDE 상수와 `_contains_any_term`
   가 `vocabulary.py` 로 이동.
3. **review reason 누적**. 검토 필요 사유를 boolean 한 비트가 아니라
   `review_reasons` 리스트에 기록.
4. **파일 해시 dedup 은 호출자(pipeline) 책임**. extractor 는 순수 변환.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from jinja2 import Template
from pydantic import ValidationError

from src.policy_pipeline.models import (
    ExtractedPolicy,
    MIN_CONFIDENCE,
    PolicyDocument,
    to_validated_policy,
)
from src.policy_pipeline.vocabulary import (
    REVIEW_TRIGGER_FIELDS,
    has_ambiguous_scope,
    is_national_scope,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "src" / "prompts" / "policy_extraction.jinja2"


# ---------------------------------------------------------------------------
# LLM client interface
# ---------------------------------------------------------------------------
class StructuredLLMClient(Protocol):
    """Structured output 을 지원하는 LLM 클라이언트 (권장)."""

    def complete_structured(
        self,
        prompt: str,
        response_model: type[ExtractedPolicy],
        *,
        system_prompt: str | None = None,
    ) -> ExtractedPolicy:
        ...


class LLMClient(Protocol):
    """Structured output 미지원 환경용 폴백 (구버전 호환).

    JSON object 모드만 지원하는 클라이언트. 코드펜스/think 태그 제거 후 직접 파싱.
    """

    def complete(self, prompt: str) -> str:
        ...


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class PolicyExtractionError(RuntimeError):
    """추출 단계 전반의 베이스 에러."""


class PolicyJsonParseError(PolicyExtractionError):
    """LLM 응답이 JSON 으로 파싱 불가."""


class PolicyValidationError(PolicyExtractionError):
    """파싱된 JSON 이 ExtractedPolicy 스키마와 안 맞음."""


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_policy_extraction_prompt(
    document: PolicyDocument,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
) -> str:
    template = Template(prompt_path.read_text(encoding="utf-8"))
    return template.render(
        document=document,
        extracted_policy_schema=ExtractedPolicy.model_json_schema(),
    )


# ---------------------------------------------------------------------------
# Main entrypoints
# ---------------------------------------------------------------------------
def extract_policy(
    document: PolicyDocument,
    llm_client: StructuredLLMClient | LLMClient,
) -> ExtractedPolicy:
    """문서에서 ExtractedPolicy 를 뽑아낸다.

    클라이언트가 `complete_structured` 를 노출하면 structured output 경로를 쓰고,
    아니면 `complete` (JSON object 모드) 폴백 경로를 쓴다.
    """
    prompt = build_policy_extraction_prompt(document)

    if hasattr(llm_client, "complete_structured"):
        extracted = _extract_via_structured(llm_client, prompt, document)
    else:
        extracted = _extract_via_json_object(llm_client, prompt, document)

    return _enrich_review_reasons(extracted)


def extract_and_validate_policy(
    document: PolicyDocument,
    llm_client: StructuredLLMClient | LLMClient,
):
    """추출 → 모델 차원 검증까지. 도메인 룰 검증은 validator.validate_policy 가 한다."""
    extracted = extract_policy(document, llm_client)
    return to_validated_policy(extracted)


# ---------------------------------------------------------------------------
# Structured path
# ---------------------------------------------------------------------------
def _extract_via_structured(
    client: StructuredLLMClient,
    prompt: str,
    document: PolicyDocument,
) -> ExtractedPolicy:
    """OpenAI structured output 경로.

    스키마 강제는 OpenAI 측에서 해주지만, 식별 메타(`policy_id`, `source_file*`)는
    원문에서 가져와 강제로 덮어쓴다 — LLM 이 임의로 만들지 못하게.
    """
    try:
        parsed = client.complete_structured(prompt, ExtractedPolicy)
    except ValidationError as exc:
        raise PolicyValidationError("Structured output failed Pydantic validation") from exc
    except RuntimeError as exc:
        raise PolicyExtractionError(str(exc)) from exc

    # 신원 메타는 원문 기준으로 고정 (LLM 환각 차단).
    parsed.__dict__["policy_id"] = f"policy_{document.file_hash[:12]}"
    parsed.__dict__["source_file"] = document.source_file
    parsed.__dict__["source_file_hash"] = document.file_hash
    parsed.__dict__["raw_text"] = document.raw_text
    return parsed


# ---------------------------------------------------------------------------
# Legacy JSON object path
# ---------------------------------------------------------------------------
import json as _json  # noqa: E402  (intentional: keep heavy import below main API)


def _extract_via_json_object(
    client: LLMClient,
    prompt: str,
    document: PolicyDocument,
) -> ExtractedPolicy:
    response = client.complete(prompt)
    if not response.strip():
        raise PolicyExtractionError("LLM response was empty")

    try:
        payload = _parse_json_object(response)
    except _json.JSONDecodeError as exc:
        raise PolicyJsonParseError("LLM response was not valid JSON") from exc

    payload.setdefault("policy_id", f"policy_{document.file_hash[:12]}")
    payload.setdefault("source_file", document.source_file)
    payload.setdefault("source_file_hash", document.file_hash)
    payload.setdefault("raw_text", document.raw_text)

    try:
        return ExtractedPolicy.model_validate(payload)
    except ValidationError as exc:
        raise PolicyValidationError("LLM response did not match ExtractedPolicy") from exc


def _parse_json_object(response_text: str) -> dict:
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = _strip_code_fence(cleaned)

    try:
        parsed = _json.loads(cleaned)
    except _json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = _json.loads(cleaned[start : end + 1])

    if not isinstance(parsed, dict):
        raise PolicyJsonParseError("LLM response JSON must be an object")
    return parsed


# ---------------------------------------------------------------------------
# 호환 shim — scripts/check_policy_samples.py 가 직접 호출
# ---------------------------------------------------------------------------
def parse_extracted_policy_response(
    response_text: str,
    document: PolicyDocument,
) -> ExtractedPolicy:
    """LLM raw text(JSON 문자열) → ExtractedPolicy. 테스트/수동 스크립트용."""
    try:
        payload = _parse_json_object(response_text)
    except _json.JSONDecodeError as exc:
        raise PolicyJsonParseError("response was not valid JSON") from exc

    payload.setdefault("policy_id", f"policy_{document.file_hash[:12]}")
    payload.setdefault("source_file", document.source_file)
    payload.setdefault("source_file_hash", document.file_hash)
    payload.setdefault("raw_text", document.raw_text)

    try:
        extracted = ExtractedPolicy.model_validate(payload)
    except ValidationError as exc:
        raise PolicyValidationError("payload did not match ExtractedPolicy") from exc

    return _enrich_review_reasons(extracted)


def _strip_code_fence(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Review-reason enrichment (도메인 룰 일부 — validator 와 중복 X, 보완)
# ---------------------------------------------------------------------------
def _enrich_review_reasons(extracted: ExtractedPolicy) -> ExtractedPolicy:
    """확실히 검토가 필요한 경우(트리거 필드 누락/모호, 모호어, 비매핑 국가범위,
    필수 스코프 누락)만 reason 으로 추가.

    `vocabulary.py` 의 단일 진실의 원천을 사용한다.
    `validator.py` 도 도메인 룰을 추가로 적용하므로, 여기는 **LLM 자기보고 기반의
    가벼운 1차 필터**만 둔다.
    """
    reasons: list[str] = list(extracted.review_reasons)

    trigger_amb = REVIEW_TRIGGER_FIELDS.intersection(extracted.ambiguous_fields)
    if trigger_amb:
        reasons.append(f"trigger-field ambiguous: {', '.join(sorted(trigger_amb))}")

    trigger_miss = REVIEW_TRIGGER_FIELDS.intersection(extracted.missing_fields)
    if trigger_miss:
        reasons.append(f"trigger-field missing: {', '.join(sorted(trigger_miss))}")

    scope_values = [
        *extracted.target_regions,
        *extracted.target_districts,
        *extracted.target_industries,
        *extracted.target_groups,
    ]
    if has_ambiguous_scope(scope_values):
        reasons.append("scope contains ambiguous terms")

    if is_national_scope(extracted.target_regions):
        reasons.append("national scope not mapped to Seoul units")

    if not extracted.target_regions and not extracted.target_districts:
        reasons.append("no target regions or districts")

    if not extracted.target_industries:
        reasons.append("no target industries")

    if extracted.confidence < MIN_CONFIDENCE:
        # 이미 모델 validator 가 적었을 수 있으나, dedupe 가 해결.
        reasons.append(f"confidence {extracted.confidence:.2f} < {MIN_CONFIDENCE}")

    if reasons:
        extracted.__dict__["requires_human_review"] = True
        # 중복 제거
        seen: set[str] = set()
        deduped: list[str] = []
        for r in reasons:
            if r not in seen:
                seen.add(r)
                deduped.append(r)
        extracted.__dict__["review_reasons"] = deduped
    return extracted
