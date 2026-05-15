from __future__ import annotations

import json
from pathlib import Path

from src.policy_pipeline.extractor import (
    PolicyExtractionError,
    parse_extracted_policy_response,
)
from src.policy_pipeline.loader import load_policy_document
from src.policy_pipeline.validator import validate_and_record_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PROJECT_ROOT / "data" / "policies" / "samples"


def main() -> None:
    sample_cases = [
        (
            "policy_001_normal_consumption_coupon.txt",
            _normal_coupon_payload,
        ),
        (
            "policy_002_ambiguous_local_voucher.txt",
            _ambiguous_voucher_payload,
        ),
        (
            "policy_003_invalid_missing_fields.txt",
            _invalid_policy_payload,
        ),
    ]

    for filename, payload_factory in sample_cases:
        path = SAMPLE_DIR / filename
        document = load_policy_document(path)
        payload = payload_factory(document)
        print(f"\n[sample] {filename}")

        try:
            extracted_policy = parse_extracted_policy_response(
                json.dumps(payload, ensure_ascii=False),
                document,
            )
            outcome = validate_and_record_policy(extracted_policy)
        except PolicyExtractionError as exc:
            print(f"status=failed reason={exc}")
            continue
        except ValueError as exc:
            print(f"status=failed reason={exc}")
            continue

        print(f"status={outcome.status.value}")
        if outcome.issues:
            for issue in outcome.issues:
                print(f"- {issue.severity.value}: {issue.field}: {issue.message}")
        if outcome.validated_policy is not None:
            print(f"validated_policy={outcome.validated_policy.policy_id}")


def _base_payload(document) -> dict:
    return {
        "policy_id": f"policy_{document.file_hash[:12]}",
        "source_file": document.source_file,
        "source_file_hash": document.file_hash,
        "raw_text": document.raw_text,
    }


def _normal_coupon_payload(document) -> dict:
    payload = _base_payload(document)
    payload.update(
        {
            "title": "2026 Mapo local consumption coupon support",
            "summary": "Consumption coupons for Mapo local commercial areas.",
            "effective_start_date": "2026-06-01",
            "effective_end_date": "2026-08-31",
            "target_regions": ["\uc11c\uc6b8\uc2dc"],
            "target_districts": ["\ub9c8\ud3ec\uad6c"],
            "target_industries": ["\uc74c\uc2dd\uc810", "\uce74\ud398", "\uc804\ud1b5\uc2dc\uc7a5"],
            "target_groups": ["\uc11c\uc6b8\uc2dc\ubbfc", "\uad00\uad11\uac1d"],
            "benefit_type": "coupon",
            "benefit_amount": 30000,
            "benefit_rate": None,
            "conditions": ["Minimum payment amount is 20000 KRW."],
            "restrictions": ["Large marts and online payments are excluded."],
            "expected_behavior_effects": ["Increase local store visits."],
            "confidence": 0.92,
            "ambiguous_fields": [],
            "missing_fields": [],
            "requires_human_review": False,
        }
    )
    return payload


def _ambiguous_voucher_payload(document) -> dict:
    payload = _base_payload(document)
    payload.update(
        {
            "title": "Seoul local commerce voucher pilot",
            "summary": "Voucher pilot for selected local commercial areas.",
            "effective_start_date": None,
            "effective_end_date": None,
            "target_regions": ["\uc77c\ubd80 \uc9c0\uc5ed"],
            "target_districts": [],
            "target_industries": ["\uc0dd\ud65c\ubc00\ucc29\ud615 \uc5c5\uc885"],
            "target_groups": ["\uc11c\uc6b8\uc2dc\ubbfc"],
            "benefit_type": "voucher",
            "benefit_amount": None,
            "benefit_rate": None,
            "conditions": [],
            "restrictions": ["Large retailers and some gambling industries are excluded."],
            "expected_behavior_effects": ["Increase local store visits."],
            "confidence": 0.58,
            "ambiguous_fields": ["target_regions", "target_industries"],
            "missing_fields": [
                "effective_start_date",
                "effective_end_date",
                "benefit_amount",
                "conditions",
            ],
            "requires_human_review": True,
        }
    )
    return payload


def _invalid_policy_payload(document) -> dict:
    payload = _base_payload(document)
    payload.update(
        {
            "title": "Youth dining discount support draft",
            "summary": "Draft discount support for youth dining.",
            "effective_start_date": "2026-09-30",
            "effective_end_date": "2026-09-01",
            "target_regions": [],
            "target_districts": [],
            "target_industries": ["\uc74c\uc2dd\uc810"],
            "target_groups": ["\uccad\ub144"],
            "benefit_type": "discount",
            "benefit_amount": None,
            "benefit_rate": 1.2,
            "conditions": [],
            "restrictions": [],
            "expected_behavior_effects": [],
            "confidence": 0.4,
            "ambiguous_fields": ["target_groups"],
            "missing_fields": ["target_regions", "benefit_amount", "restrictions"],
            "requires_human_review": True,
        }
    )
    return payload


if __name__ == "__main__":
    main()
