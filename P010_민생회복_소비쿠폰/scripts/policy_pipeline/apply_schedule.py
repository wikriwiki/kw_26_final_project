"""
apply_schedule.py
=================
스케줄 파일을 읽어 등록된 정책들을 **시뮬 시작 전 일괄로** Neo4j 에 주입한다.

각 정책은 JSON 원본을 건드리지 않고, schedule.yaml 에 지정된 effective_from /
effective_until 로 override 되어 적재된다. 시뮬 도중 POLICY_CYPHER 가
`$today >= pol.effective_from AND $today <= pol.effective_until` 로 필터링하므로,
스케줄에 적힌 날짜가 도래해야 LLM 프롬프트에 노출된다.

사용:
  # 기본 스케줄 (data/policies/schedule.yaml)
  python -m scripts.policy_pipeline.apply_schedule

  # 다른 스케줄 파일
  python -m scripts.policy_pipeline.apply_schedule --schedule path/to/schedule.yaml

  # dry-run (Neo4j 안 건드리고 검증 + 해석된 날짜만 출력)
  python -m scripts.policy_pipeline.apply_schedule --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scripts.policy_pipeline.loader import load_json_payload
from scripts.policy_pipeline.pipeline import inject_validated_payload
from scripts.policy_pipeline.schedule import (
    Schedule,
    ScheduleError,
    load_schedule,
    override_payload_dates,
    resolve_dates,
    resolve_file_path,
)
from scripts.policy_pipeline.state import PolicyStatus, calculate_file_hash


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEDULE_PATH = PROJECT_ROOT / "data" / "policies" / "schedule.yaml"
DEFAULT_POLICY_BASE = PROJECT_ROOT / "data" / "neo4j_load" / "policies"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply a policy schedule (bulk inject before simulation start)."
    )
    parser.add_argument(
        "--schedule", type=Path, default=DEFAULT_SCHEDULE_PATH,
        help=f"Schedule YAML path (default: {DEFAULT_SCHEDULE_PATH.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--policy-base", type=Path, default=DEFAULT_POLICY_BASE,
        help="Base directory for relative `file:` paths in schedule",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve dates and validate, but skip Neo4j writes.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    try:
        schedule = resolve_dates(load_schedule(args.schedule))
    except ScheduleError as exc:
        print(f"[ERROR] schedule invalid: {exc}", file=sys.stderr)
        return 1

    print(f"[schedule] sim_start={schedule.sim_start} "
          f"entries={len(schedule.entries)}")

    failures = 0
    for idx, entry in enumerate(schedule.entries):
        path = resolve_file_path(entry.file, args.policy_base)
        if not path.exists():
            print(f"[{idx}] SKIP {entry.file} — file not found at {path}")
            failures += 1
            continue

        payload = load_json_payload(path)
        payload = override_payload_dates(payload, entry)
        file_hash = calculate_file_hash(path)

        eff_from = payload.get("effective_from")
        eff_until = payload.get("effective_until")
        print(f"[{idx}] {entry.file} id={payload.get('id')} "
              f"effective={eff_from}~{eff_until} "
              f"deactivate_others={entry.deactivate_others}")

        if args.dry_run:
            continue

        deactivate_from = eff_from if entry.deactivate_others else None
        result = inject_validated_payload(
            payload,
            source_file=str(path),
            file_hash=file_hash,
            deactivate_others_from=deactivate_from,
        )
        print(f"     status={result.final_status.value} "
              f"districts={result.neo4j_districts_linked} "
              f"dongs={result.neo4j_dongs_linked} "
              f"categories={result.neo4j_categories_linked}")
        if result.final_status != PolicyStatus.APPLIED:
            failures += 1
            if result.error:
                print(f"     error={result.error}")

    if failures:
        print(f"[schedule] DONE with {failures} failure(s)")
        return 2
    print(f"[schedule] DONE — all {len(schedule.entries)} policies applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
