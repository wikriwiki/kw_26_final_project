"""
dedup.py
========
파일 해시 기반 중복 처리 방지.

같은 SHA-256 해시를 가진 파일이 이미 처리됐는지 확인. state.py 의 JSONL 로그를
읽어 file_hash 가 한 번이라도 등장했는지 본다.

판단 기준:
- 같은 hash 가 VALIDATED / APPLIED / NEEDS_REVIEW 상태로 한 번이라도 도달했으면
  중복 처리로 간주 (LLM 다시 부르지 않음).
- FAILED 만 있었다면 재시도 허용 (네트워크/일시 오류일 수 있으므로).
"""

from __future__ import annotations

from pathlib import Path

from src.policy_pipeline.state import (
    DEFAULT_STATE_LOG_PATH,
    PolicyStatus,
    read_processing_records,
)


# 한 번이라도 이 상태에 도달했으면 "이미 처리됨" 으로 본다.
_TERMINAL_OR_REVIEW_STATUSES: frozenset[PolicyStatus] = frozenset({
    PolicyStatus.VALIDATED,
    PolicyStatus.APPLIED,
    PolicyStatus.NEEDS_REVIEW,
})


def is_duplicate_hash(
    file_hash: str,
    state_log_path: Path | None = None,
) -> bool:
    """file_hash 가 이전에 의미 있게 처리됐는지.

    state_log_path 가 None 이면 호출 시점의 모듈 attr `DEFAULT_STATE_LOG_PATH` 를
    사용 (monkeypatch 친화).
    """
    path = state_log_path if state_log_path is not None else DEFAULT_STATE_LOG_PATH
    if not path.exists():
        return False

    for record in read_processing_records(path):
        if record.file_hash == file_hash and record.status in _TERMINAL_OR_REVIEW_STATUSES:
            return True
    return False
