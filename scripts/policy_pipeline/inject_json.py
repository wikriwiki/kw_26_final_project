"""
inject_json.py
==============
수기 검수 JSON 정책 1건을 Neo4j 에 주입.

기존 `scripts/neo4j_load/load_p007.py` 와 같은 역할이지만 다음이 다르다:
  - 정책 ID/필드를 하드코딩하지 않음 — 임의의 JSON 파일 경로를 받음
  - Pydantic 검증 + 도메인 룰 검증을 거침
  - Scope 분석 → Neo4j 의 :applied_to → Dong/Category 까지 일관 처리
  - 처리 결과가 JSONL 감사 로그로 자동 기록

사용:
  # 기본 (P007 JSON, P001 비활성화 같은 효과)
  python -m scripts.policy_pipeline.inject_json \\
      data/neo4j_load/policies/P007.json \\
      --deactivate-others-from 2026-05-01

  # 단순 적재 (비활성화 없음)
  python -m scripts.policy_pipeline.inject_json data/neo4j_load/policies/P007.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scripts.policy_pipeline.loader import load_json_payload
from scripts.policy_pipeline.pipeline import inject_validated_payload
from scripts.policy_pipeline.state import calculate_file_hash, PolicyStatus


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_JSON = PROJECT_ROOT / "data" / "neo4j_load" / "policies" / "P007.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Inject a curated policy JSON into Neo4j.")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_POLICY_JSON,
        help=f"Policy JSON path (default: {DEFAULT_POLICY_JSON.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--deactivate-others-from",
        default=None,
        help="YYYY-MM-DD. Set effective_until of all other policies to one day before "
             "this date (mimics load_p007.py's P001-deactivation behavior).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate + scope but skip Neo4j write.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging level (DEBUG/INFO/WARNING/ERROR).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if not args.path.exists():
        print(f"[ERROR] policy JSON not found: {args.path}", file=sys.stderr)
        return 1

    payload = load_json_payload(args.path)
    file_hash = calculate_file_hash(args.path)

    print(f"[inject] file={args.path.name} id={payload.get('id')} hash={file_hash[:12]}")

    if args.dry_run:
        from scripts.policy_pipeline.models import extracted_from_curated_json
        from scripts.policy_pipeline.scope import NullGraphReader, analyze_scope
        from scripts.policy_pipeline.validator import validate_policy

        extracted = extracted_from_curated_json(
            payload, source_file=str(args.path), file_hash=file_hash,
        )
        outcome = validate_policy(extracted)
        print(f"[dry-run] validation status = {outcome.status.value}")
        for issue in outcome.issues:
            print(f"  - {issue.severity.value}: {issue.field}: {issue.message}")
        if outcome.validated_policy:
            scope = analyze_scope(outcome.validated_policy, NullGraphReader())
            print(f"[dry-run] scope.units = {[u.value for u in scope.scope_units]}")
            print(f"[dry-run] affected_districts = {len(scope.affected_districts)}")
            print(f"[dry-run] affected_categories = {len(scope.affected_categories)}")
        return 0

    result = inject_validated_payload(
        payload,
        source_file=str(args.path),
        file_hash=file_hash,
        deactivate_others_from=args.deactivate_others_from,
    )

    print(f"[inject] final_status={result.final_status.value}")
    if result.policy_id:
        print(f"  policy_id={result.policy_id}")
    if result.error:
        print(f"  error={result.error}")
    print(f"  neo4j: districts={result.neo4j_districts_linked} "
          f"dongs={result.neo4j_dongs_linked} categories={result.neo4j_categories_linked}")
    print(f"  cache_keys_invalidated={result.cache_keys_invalidated}")
    print(f"  summary_jobs_enqueued={result.summary_jobs_enqueued}")

    return 0 if result.final_status == PolicyStatus.APPLIED else 2


if __name__ == "__main__":
    raise SystemExit(main())
