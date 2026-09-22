"""
state.py
========
정책 처리 상태 머신 + JSONL 감사 로그.

상태 흐름:
  DETECTED → EXTRACTING → VALIDATED → APPLIED         (정상)
                       ↓
                    NEEDS_REVIEW                       (모호/필수 누락)
                       ↓
                    FAILED                             (데이터 깨짐)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_LOG_PATH = PROJECT_ROOT / "output" / "policy_pipeline" / "policy_processing.jsonl"


class PolicyStatus(str, Enum):
    DETECTED = "detected"
    EXTRACTING = "extracting"
    VALIDATED = "validated"
    NEEDS_REVIEW = "needs_review"
    APPLIED = "applied"              # Neo4j MERGE 완료
    FAILED = "failed"


class PolicyProcessingRecord(BaseModel):
    policy_id: str
    file_hash: str
    source_path: str
    version: int = 1
    status: PolicyStatus
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: str | None = None


def calculate_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_policy_id(file_hash: str) -> str:
    return f"policy_{file_hash[:12]}"


def create_processing_record(
    path: Path,
    status: PolicyStatus = PolicyStatus.DETECTED,
    version: int = 1,
    error_message: str | None = None,
) -> PolicyProcessingRecord:
    file_hash = calculate_file_hash(path)
    now = datetime.now(timezone.utc)
    return PolicyProcessingRecord(
        policy_id=build_policy_id(file_hash),
        file_hash=file_hash,
        source_path=str(path),
        version=version,
        status=status,
        created_at=now,
        updated_at=now,
        error_message=error_message,
    )


def append_processing_record(
    record: PolicyProcessingRecord,
    log_path: Path | None = None,
) -> None:
    path = log_path if log_path is not None else DEFAULT_STATE_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


def append_status_for_file(
    path: Path,
    status: PolicyStatus,
    log_path: Path | None = None,
    error_message: str | None = None,
) -> PolicyProcessingRecord:
    record = create_processing_record(path=path, status=status, error_message=error_message)
    append_processing_record(record, log_path)
    return record


def append_policy_status(
    policy_id: str,
    file_hash: str,
    source_path: str,
    status: PolicyStatus,
    log_path: Path | None = None,
    version: int = 1,
    error_message: str | None = None,
) -> PolicyProcessingRecord:
    now = datetime.now(timezone.utc)
    record = PolicyProcessingRecord(
        policy_id=policy_id,
        file_hash=file_hash,
        source_path=source_path,
        version=version,
        status=status,
        created_at=now,
        updated_at=now,
        error_message=error_message,
    )
    append_processing_record(record, log_path)
    return record


def read_processing_records(
    log_path: Path | None = None,
) -> list[PolicyProcessingRecord]:
    path = log_path if log_path is not None else DEFAULT_STATE_LOG_PATH
    if not path.exists():
        return []
    records: list[PolicyProcessingRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(PolicyProcessingRecord.model_validate(json.loads(line)))
    return records
