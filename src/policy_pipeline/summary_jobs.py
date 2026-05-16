"""
summary_jobs.py
===============
Section 8 — Global/Community Summary 재생성 트리거 (job 큐).

영향 받은 동 / 업종 / 서울 전체 단위로 요약 재생성 잡을 등록.
실제 워커 구현은 별도 (이 파일은 큐 등록 + 상태 추적만).

상태 머신: queued → running → completed | failed.

저장소: JSONL append-only (`output/state/summary_jobs.jsonl`). 단순함 위주.
프로덕션에서는 Redis Streams 나 RQ/Celery 로 교체 권장.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from src.policy_pipeline.scope import PolicyScope, ScopeUnit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOB_LOG_PATH = PROJECT_ROOT / "output" / "state" / "summary_jobs.jsonl"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SummaryJobKind(str, Enum):
    COMMUNITY = "community_summary"       # 행정동 단위 요약
    INDUSTRY = "industry_summary"         # 업종 단위
    SEOUL_WIDE = "seoul_wide_summary"     # 서울 전체


class SummaryJob(BaseModel):
    job_id: str
    kind: SummaryJobKind
    target_id: str                       # dong_id / industry_id / "seoul"
    triggered_by_policy_id: str
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: str | None = None


# ---------------------------------------------------------------------------
# 메인 API: scope → job 등록
# ---------------------------------------------------------------------------
def enqueue_summary_jobs_for_scope(
    scope: PolicyScope,
    log_path: Path | None = None,
) -> list[SummaryJob]:
    log_path = log_path or DEFAULT_JOB_LOG_PATH
    """PolicyScope 에 따라 필요한 summary 재생성 잡을 enqueue.

    - SEOUL_WIDE → 1건의 seoul_wide_summary 잡
    - 그 외     → 영향 받은 동마다 community_summary, 영향 받은 업종마다 industry_summary

    중복 등록 방지는 향후 워커 측에서 (같은 target_id+context_version 이면 skip).
    이번 PR 은 큐만 채운다.
    """
    jobs: list[SummaryJob] = []

    if ScopeUnit.SEOUL_WIDE in scope.scope_units:
        jobs.append(_make_job(SummaryJobKind.SEOUL_WIDE, "seoul", scope.policy_id))
    else:
        for dong_id in scope.affected_dongs:
            jobs.append(_make_job(SummaryJobKind.COMMUNITY, dong_id, scope.policy_id))
        for industry in scope.affected_industries:
            jobs.append(_make_job(SummaryJobKind.INDUSTRY, industry, scope.policy_id))

    for job in jobs:
        _append_job_log(job, log_path)

    # scope 에도 반영
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
    """jobs 로그에 상태 전이 한 줄 append. 워커가 호출."""
    payload = {
        "job_id": job_id,
        "status_transition": new_status.value,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "error_message": error_message,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------
def _make_job(kind: SummaryJobKind, target_id: str, policy_id: str) -> SummaryJob:
    return SummaryJob(
        job_id=f"job_{uuid.uuid4().hex[:12]}",
        kind=kind,
        target_id=target_id,
        triggered_by_policy_id=policy_id,
    )


def _append_job_log(job: SummaryJob, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(job.model_dump(mode="json"), ensure_ascii=False) + "\n")
