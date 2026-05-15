from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from jinja2 import Template
from pydantic import ValidationError

from src.policy_pipeline.models import (
    ExtractedPolicy,
    PolicyDocument,
    validate_extracted_policy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "src" / "prompts" / "policy_extraction.jinja2"
REVIEW_TRIGGER_FIELDS = {
    "target_regions",
    "target_districts",
    "target_industries",
    "effective_start_date",
    "effective_end_date",
    "benefit_amount",
    "benefit_rate",
    "conditions",
}
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
UNMAPPED_REGION_TERMS = {
    "전국",
    "전 지역",
    "대한민국",
    "국내 전체",
}
CLEAR_SEOUL_WIDE_TERMS = {
    "서울",
    "서울시",
    "서울 전체",
    "서울시 전체",
    "서울 전역",
}


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str:
        pass


class PolicyExtractionError(RuntimeError):
    pass


class PolicyJsonParseError(PolicyExtractionError):
    pass


class PolicyValidationError(PolicyExtractionError):
    pass


def build_policy_extraction_prompt(
    document: PolicyDocument,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
) -> str:
    template = Template(prompt_path.read_text(encoding="utf-8"))
    return template.render(
        document=document,
        extracted_policy_schema=ExtractedPolicy.model_json_schema(),
    )


def parse_extracted_policy_response(
    response_text: str,
    document: PolicyDocument,
) -> ExtractedPolicy:
    try:
        payload = _parse_json_object(response_text)
    except json.JSONDecodeError as exc:
        raise PolicyJsonParseError("LLM response was not valid JSON") from exc

    payload.setdefault("policy_id", f"policy_{document.file_hash[:12]}")
    payload.setdefault("source_file", document.source_file)
    payload.setdefault("source_file_hash", document.file_hash)
    payload.setdefault("raw_text", document.raw_text)

    try:
        extracted_policy = ExtractedPolicy.model_validate(payload)
    except ValidationError as exc:
        raise PolicyValidationError("LLM response did not match ExtractedPolicy") from exc

    if _requires_policy_review(extracted_policy):
        extracted_policy.requires_human_review = True

    return extracted_policy


def extract_policy(document: PolicyDocument, llm_client: LLMClient) -> ExtractedPolicy:
    prompt = build_policy_extraction_prompt(document)
    response_text = llm_client.complete(prompt)
    if not response_text.strip():
        raise PolicyExtractionError("LLM response was empty")

    return parse_extracted_policy_response(response_text, document)


def extract_and_validate_policy(
    document: PolicyDocument,
    llm_client: LLMClient,
):
    extracted_policy = extract_policy(document, llm_client)
    return validate_extracted_policy(extracted_policy)


def _parse_json_object(response_text: str) -> dict:
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = _strip_code_fence(cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])

    if not isinstance(parsed, dict):
        raise PolicyJsonParseError("LLM response JSON must be an object")

    return parsed


def _strip_code_fence(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _requires_policy_review(extracted_policy: ExtractedPolicy) -> bool:
    if extracted_policy.confidence < 0.7:
        return True
    if REVIEW_TRIGGER_FIELDS.intersection(extracted_policy.ambiguous_fields):
        return True
    if REVIEW_TRIGGER_FIELDS.intersection(extracted_policy.missing_fields):
        return True
    if _has_ambiguous_scope(extracted_policy):
        return True
    if _has_unmapped_region_scope(extracted_policy):
        return True
    if not extracted_policy.target_regions and not extracted_policy.target_districts:
        return True
    if not extracted_policy.target_industries:
        return True
    return False


def _has_ambiguous_scope(extracted_policy: ExtractedPolicy) -> bool:
    scope_values = [
        *extracted_policy.target_regions,
        *extracted_policy.target_districts,
        *extracted_policy.target_industries,
        *extracted_policy.target_groups,
    ]
    return _contains_any_term(scope_values, AMBIGUOUS_SCOPE_TERMS)


def _has_unmapped_region_scope(extracted_policy: ExtractedPolicy) -> bool:
    if not _contains_any_term(extracted_policy.target_regions, UNMAPPED_REGION_TERMS):
        return False

    return not _contains_any_term(extracted_policy.target_regions, CLEAR_SEOUL_WIDE_TERMS)


def _contains_any_term(values: list[str], terms: set[str]) -> bool:
    normalized_values = [value.replace(" ", "").lower() for value in values]
    normalized_terms = [term.replace(" ", "").lower() for term in terms]
    return any(
        term in value
        for value in normalized_values
        for term in normalized_terms
    )
