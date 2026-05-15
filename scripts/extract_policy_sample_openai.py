from __future__ import annotations

import argparse
from pathlib import Path

from src.infra.llm.openai_client import OpenAIChatClient
from src.policy_pipeline.extractor import extract_policy
from src.policy_pipeline.loader import load_policy_document
from src.policy_pipeline.validator import validate_and_record_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "policies"
    / "samples"
    / "policy_001_normal_consumption_coupon.txt"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and validate one sample policy with OpenAI."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_SAMPLE_PATH,
        help="Policy .txt or .json file to extract.",
    )
    args = parser.parse_args()

    document = load_policy_document(args.path)
    llm_client = OpenAIChatClient()
    extracted_policy = extract_policy(document, llm_client)
    outcome = validate_and_record_policy(extracted_policy)

    print(f"policy_id={extracted_policy.policy_id}")
    print(f"title={extracted_policy.title}")
    print(f"status={outcome.status.value}")
    print(f"requires_human_review={extracted_policy.requires_human_review}")
    if outcome.issues:
        for issue in outcome.issues:
            print(f"- {issue.severity.value}: {issue.field}: {issue.message}")


if __name__ == "__main__":
    main()
