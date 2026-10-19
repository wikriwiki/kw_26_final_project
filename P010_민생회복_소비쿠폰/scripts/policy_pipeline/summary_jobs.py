"""
summary_jobs.py
===============
Global / Community Summary 재생성 잡 큐.

agent_persona 브랜치에서는 워커가 아직 없으므로 JSONL append-only 로 큐만 채운다.
docs/schedule_generation_plan/runtime_ontology.md 의 "Signal Sender → Celery 재생성"
자리를 위한 scaffolding.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from scripts.policy_pipeline.scope import PolicyScope, ScopeUnit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOB_LOG_PATH = PROJECT_ROOT / "output" / "policy_pipeline" / "summary_jobs.jsonl"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SummaryJobKind(str, Enum):
    COMMUNITY = "community_summary"
    INDUSTRY = "industry_summary"
    SEOUL_WIDE = "seoul_wide_summary"


class SummaryJob(BaseModel):
    job_id: str
    kind: SummaryJobKind
    target_id: str                     # dong_code / L1 cat name / "seoul"
    triggered_by_policy_id: str
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: str | None = None


def enqueue_summary_jobs_for_scope(
    scope: PolicyScope,
    log_path: Path | None = None,
) -> list[SummaryJob]:
    log_path = log_path or DEFAULT_JOB_LOG_PATH
    jobs: list[SummaryJob] = []

    if ScopeUnit.SEOUL_WIDE in scope.scope_units:
        jobs.append(_make(SummaryJobKind.SEOUL_WIDE, "seoul", scope.policy_id))
    else:
        for d in scope.affected_dongs:
            jobs.append(_make(SummaryJobKind.COMMUNITY, d, scope.policy_id))
        for c in scope.affected_categories:
            jobs.append(_make(SummaryJobKind.INDUSTRY, c, scope.policy_id))

    for job in jobs:
        _append(job, log_path)
    scope.summary_jobs_to_rebuild = [j.job_id for j in jobs]
    return jobs


def update_job_status(
    job_id: str,
    new_status: JobStatus,
    *,
    error_message: str | None = None,
    log_path: Path | None = None,
) -> None:
    log_path = log_path or DEFAULT_JOB_LOG_PATH
    payload = {
        "job_id": job_id,
        "status_transition": new_status.value,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "error_message": error_message,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _make(kind: SummaryJobKind, target_id: str, policy_id: str) -> SummaryJob:
    return SummaryJob(
        job_id=f"job_{uuid.uuid4().hex[:12]}",
        kind=kind,
        target_id=target_id,
        triggered_by_policy_id=policy_id,
    )


def _append(job: SummaryJob, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(job.model_dump(mode="json"), ensure_ascii=False) + "\n")
