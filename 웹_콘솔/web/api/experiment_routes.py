"""실험 콘솔 API — 레지스트리·실제 데이터·결과·인터뷰·집단 질의."""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from . import demo_guard, experiment_interview, experiment_query
from .experiment_sync import SyncJobs
from .experiments import ExperimentStore


class ExperimentInterviewRequest(BaseModel):
    arm_id: str
    question: str
    through_day: str | None = None
    history: list[dict[str, str]] = Field(default_factory=list)


class GroupFilterRequest(BaseModel):
    arm_id: str
    through_day: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)


class GroupEffectRequest(BaseModel):
    filters: dict[str, Any] = Field(default_factory=dict)


class GroupInterviewRequest(GroupFilterRequest):
    question: str
    sample: int = Field(default=10, ge=1, le=experiment_query.MAX_SAMPLE)
    seed: int = 0
    summarize: bool = True
    fetch_evidence: bool = False


def register(app: FastAPI, store: ExperimentStore | None = None) -> ExperimentStore:
    store = store or ExperimentStore()
    sync_jobs = SyncJobs(store)
    query_jobs = experiment_query.QueryJobs(store)
    app.state.experiments = store

    @app.get("/api/experiments")
    async def list_experiments() -> dict:
        return await asyncio.to_thread(store.list)

    @app.get("/api/experiments/{experiment_id}")
    async def experiment_detail(experiment_id: str) -> dict:
        return await asyncio.to_thread(store.detail, experiment_id)

    @app.get("/api/experiments/{experiment_id}/daily")
    async def experiment_daily(experiment_id: str) -> dict:
        return await asyncio.to_thread(store.daily, experiment_id)

    @app.get("/api/experiments/{experiment_id}/comparison")
    async def experiment_comparison(experiment_id: str) -> dict:
        return await asyncio.to_thread(store.comparison, experiment_id)

    @app.get("/api/experiments/{experiment_id}/agents")
    async def experiment_agents(experiment_id: str) -> dict:
        return await asyncio.to_thread(store.roster, experiment_id)

    @app.get("/api/experiments/{experiment_id}/agents/{agent_id}")
    async def experiment_agent(experiment_id: str, agent_id: str) -> dict:
        return await asyncio.to_thread(store.agent, experiment_id, agent_id)

    @app.post("/api/experiments/{experiment_id}/agents/{agent_id}/interview")
    async def experiment_agent_interview(experiment_id: str, agent_id: str, request: ExperimentInterviewRequest) -> dict:
        return await asyncio.to_thread(
            experiment_interview.interview_agent,
            store,
            experiment_id,
            request.arm_id,
            agent_id,
            request.question,
            request.history,
            request.through_day,
        )

    @app.get("/api/experiments/{experiment_id}/interviews")
    async def experiment_interviews(experiment_id: str, limit: int = Query(default=50, ge=1, le=500)) -> dict:
        return await asyncio.to_thread(experiment_interview.recent_interviews, store, experiment_id, limit)

    @app.post("/api/experiments/{experiment_id}/query/stats")
    async def experiment_group_stats(experiment_id: str, request: GroupFilterRequest) -> dict:
        return await asyncio.to_thread(
            experiment_query.group_stats, store, experiment_id, request.arm_id, request.filters, request.through_day
        )

    @app.post("/api/experiments/{experiment_id}/query/effect")
    async def experiment_group_effect(experiment_id: str, request: GroupEffectRequest) -> dict:
        return await asyncio.to_thread(experiment_query.group_effect, store, experiment_id, request.filters)

    @app.post("/api/experiments/{experiment_id}/query/interviews")
    async def experiment_group_interview(experiment_id: str, request: GroupInterviewRequest) -> dict:
        payload = request.model_dump()
        cap = demo_guard.max_sample()
        if cap is not None:
            payload["sample"] = min(payload["sample"], cap)
        return await asyncio.to_thread(query_jobs.start, experiment_id, payload)

    @app.get("/api/experiments/{experiment_id}/query/interviews/{job_id}")
    async def experiment_group_interview_status(experiment_id: str, job_id: str) -> dict:
        return await asyncio.to_thread(query_jobs.get, experiment_id, job_id)

    @app.get("/api/experiments/{experiment_id}/sync")
    async def experiment_sync_status(experiment_id: str) -> dict:
        return await asyncio.to_thread(sync_jobs.status, experiment_id)

    @app.post("/api/experiments/{experiment_id}/sync")
    async def experiment_sync_start(experiment_id: str) -> dict:
        return await asyncio.to_thread(sync_jobs.start, experiment_id)

    return store
