"""
dedup.py
========
file_hash 가 이미 처리됐는지 확인 (state 로그 기준).

VALIDATED / APPLIED / NEEDS_REVIEW 중 하나라도 도달했으면 중복으로 간주.
FAILED 만 있었다면 재시도 허용.
"""

from __future__ import annotations

from pathlib import Path

from scripts.policy_pipeline.state import (
    DEFAULT_STATE_LOG_PATH,
    PolicyStatus,
    read_processing_records,
)


_TERMINAL_OR_REVIEW: frozenset[PolicyStatus] = frozenset({
    PolicyStatus.VALIDATED,
    PolicyStatus.APPLIED,
    PolicyStatus.NEEDS_REVIEW,
})


def is_duplicate_hash(
    file_hash: str,
    state_log_path: Path | None = None,
) -> bool:
    path = state_log_path if state_log_path is not None else DEFAULT_STATE_LOG_PATH
    if not path.exists():
        return False
    for r in read_processing_records(path):
        if r.file_hash == file_hash and r.status in _TERMINAL_OR_REVIEW:
            return True
    return False
