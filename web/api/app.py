"""FastAPI entrypoint for the policy simulation console."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import Body, FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from scripts.report import llm as report_llm

from .runner import RunLock, Runner
from .report_jobs import ReportJobManager
from .store import ArtifactStore, StoreError


REPO_ROOT = Path(__file__).resolve().parents[2]


class SpaFiles(StaticFiles):
    """빌드 산출물을 서빙하되, 없는 경로는 `index.html` 로 넘긴다.

    콘솔은 `BrowserRouter` 를 쓴다. 즉 `/runs/FINAL/report` 같은 주소는 **서버에
    파일로 존재하지 않고** 브라우저가 받아서 그린다. 기본 `StaticFiles` 는 그런
    경로를 404 로 돌려주기 때문에, 클릭해서 들어가면 되던 화면이 새로고침하거나
    링크로 열면 죽는다. 주소 하나로 링크가 완결되어야 한다는 요구(§9 deep-linking)가
    깨지는 자리다.

    다만 `/api/...` 는 넘기지 않는다. 없는 API 경로가 HTML 을 돌려주면 클라이언트가
    JSON 을 기대하다 엉뚱한 곳에서 터진다. API 는 API 답게 404 로 끝낸다.
    """

    async def get_response(self, path: str, scope):  # type: ignore[override]
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            # `path` 는 OS 구분자로 정규화돼 넘어온다(윈도우에서는 `api\nope`).
            # 그래서 판정은 원본 주소로 한다.
            request_path = scope.get("path", "")
            if exc.status_code != 404 or request_path.startswith("/api/"):
                raise
            return await super().get_response("index.html", scope)


class RunStartRequest(BaseModel):
    run_id: str
    policy_id: str
    # 새 정책을 실행과 **같은 요청 안에서** 주입할 수 있게 한다.
    # 값이 있으면 저장(preflight 통과 필수) → lock 획득 → 실행 순서로 처리한다.
    policy: dict[str, Any] | None = None
    start_day: str | None = None
    days: int | None = Field(default=None, ge=1, le=365)
    agents: int | None = Field(default=None, ge=1, le=100000)


class PolicyDraftRequest(BaseModel):
    policy: dict[str, Any]


class ReportStartRequest(BaseModel):
    run_id: str
    policy_id: str
    start: str
    days: int = Field(ge=1, le=365)
    policy_from: str | None = None
    analyses: list[str] = Field(default_factory=list)
    include_interview: bool = False
    engine: str = "v2"
    use_llm: bool = True


def create_app(*, store: ArtifactStore | None = None, runner: Runner | None = None) -> FastAPI:
    app = FastAPI(title="Policy Simulation Console API", version="0.1.0")
    app.state.store = store or ArtifactStore.from_environment(REPO_ROOT)
    app.state.runner = runner or Runner(
        repo_root=REPO_ROOT,
        lock=RunLock(Path(os.environ.get("SIM_LOCK_PATH", str(REPO_ROOT / "web" / ".run.lock")))),
    )
    app.state.report_jobs = ReportJobManager(repo_root=REPO_ROOT, store=app.state.store, runner=app.state.runner)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in os.environ.get("WEB_CORS_ORIGINS", "http://localhost:5173").split(",") if origin.strip()],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["*"],
    )

    @app.exception_handler(StoreError)
    async def store_error_handler(_request: Request, exc: StoreError) -> JSONResponse:
        body: dict[str, Any] = {"error": exc.message}
        if exc.detail:
            body["detail"] = exc.detail
        return JSONResponse(status_code=exc.status_code, content=body)

    def ensure_policy_writable() -> None:
        """Do not mutate the policy input while an owned simulation is alive."""
        lock_status = app.state.runner.lock.status()
        if lock_status.get("locked"):
            raise StoreError(
                409,
                "실행 중에는 정책을 저장할 수 없습니다",
                json.dumps(lock_status, ensure_ascii=False),
            )

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "contract_version": "s1.0.0", "unknown": []}

    @app.get("/api/runs")
    async def list_runs() -> dict:
        return await asyncio.to_thread(app.state.store.runs_index)

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict:
        return await asyncio.to_thread(app.state.store.run_detail, run_id)

    @app.get("/api/runs/{run_id}/days")
    async def get_days(run_id: str) -> dict:
        # status_scan is intentionally executed off the event loop and never
        # calls aggregate_day for the first-screen resource.
        return await asyncio.to_thread(app.state.store.run_days, run_id)

    @app.get("/api/runs/{run_id}/days/{day}")
    async def get_day(run_id: str, day: str) -> dict:
        return await asyncio.to_thread(app.state.store.day_detail, run_id, day)

    @app.get("/api/runs/{run_id}/days/{day}/bottlenecks")
    async def get_bottlenecks(run_id: str, day: str) -> dict:
        return await asyncio.to_thread(app.state.store.bottlenecks, run_id, day)

    @app.get("/api/runs/{run_id}/days/{day}/slow")
    async def get_slow(
        run_id: str,
        day: str,
        limit: int = Query(15, ge=1, le=100),
        offset: int = Query(0, ge=0),
    ) -> dict:
        return await asyncio.to_thread(app.state.store.slow, run_id, day, limit, offset)

    @app.get("/api/runs/{run_id}/days/{day}/failed")
    async def get_failed(run_id: str, day: str) -> dict:
        return await asyncio.to_thread(app.state.store.failed, run_id, day)

    @app.get("/api/runs/{run_id}/failures")
    async def get_failures(
        run_id: str,
        day: str | None = Query(None),
        limit: int = Query(12, ge=1, le=100),
    ) -> dict:
        return await asyncio.to_thread(app.state.store.failures, run_id, day=day, limit=limit)

    @app.get("/api/runs/{run_id}/events/summary")
    async def get_events_summary(run_id: str) -> dict:
        return await asyncio.to_thread(app.state.store.events_summary, run_id)

    @app.get("/api/runs/{run_id}/events")
    async def run_events(
        run_id: str,
        request: Request,
        interval: float = Query(2.0, ge=1.0, le=30.0),
        max_events: int = Query(30, ge=1, le=300),
    ) -> StreamingResponse:
        # Validate before opening the stream.
        await asyncio.to_thread(app.state.store.run_detail, run_id)

        async def stream() -> AsyncIterator[str]:
            previous: str | None = None
            for _ in range(max_events):
                if await request.is_disconnected():
                    break
                payload = await asyncio.to_thread(app.state.store.run_days, run_id)
                encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                if encoded != previous:
                    yield f"event: run\ndata: {encoded}\n\n"
                    previous = encoded
                if any(item.get("day_complete") is False for item in payload.get("items", [])):
                    await asyncio.sleep(interval)
                else:
                    break

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/policies")
    async def list_policies() -> dict:
        return await asyncio.to_thread(app.state.store.policy_index)

    @app.get("/api/policies/next-id")
    async def next_policy_id() -> dict:
        """새 정책을 만들 때 쓸 다음 ID. 기존 파일과 충돌하지 않는 값을 서버가 정한다."""

        def compute() -> dict:
            index = app.state.store.policy_index()
            used = set()
            for item in index.get("items", []):
                raw = str(item.get("id", ""))
                if raw.startswith("P") and raw[1:].isdigit():
                    used.add(int(raw[1:]))
            candidate = max(used) + 1 if used else 1
            while candidate in used:
                candidate += 1
            return {
                "policy_id": f"P{candidate:03d}",
                "existing": sorted(f"P{value:03d}" for value in used),
                "unknown": [],
            }

        return await asyncio.to_thread(compute)

    # 아래 `/api/policies/{policy_id}/validate` 보다 **먼저** 등록해야 한다.
    # 그렇지 않으면 "draft" 가 policy_id 로 잡힌다 (FastAPI 는 등록 순서로 매칭한다).
    @app.post("/api/policies/draft/validate")
    async def validate_policy_draft(request: PolicyDraftRequest) -> dict:
        """저장하지 않고 초안만 검증한다. 새 정책을 만드는 화면이 매 입력마다 부른다."""
        payload = dict(request.policy)
        policy_id = str(payload.get("id") or "")
        if not policy_id:
            raise StoreError(422, "정책 초안에 id 가 없습니다")
        return await asyncio.to_thread(app.state.store.validate_policy, policy_id, payload)

    @app.get("/api/policies/{policy_id}")
    async def get_policy(policy_id: str) -> dict:
        return await asyncio.to_thread(app.state.store.policy_detail, policy_id)

    @app.get("/api/policies/{policy_id}/validate")
    async def validate_policy(policy_id: str) -> dict:
        return await asyncio.to_thread(app.state.store.validate_policy, policy_id)

    @app.post("/api/policies/{policy_id}/validate")
    async def validate_policy_payload(policy_id: str, payload: dict[str, Any] = Body(...)) -> dict:
        if set(payload) == {"payload"} and isinstance(payload.get("payload"), dict):
            payload = payload["payload"]
        return await asyncio.to_thread(app.state.store.validate_policy, policy_id, payload)

    @app.post("/api/policies")
    async def create_policy(payload: dict[str, Any] = Body(...)) -> dict:
        ensure_policy_writable()
        if set(payload) == {"payload"} and isinstance(payload.get("payload"), dict):
            payload = payload["payload"]
        policy_id = str(payload.get("id", ""))
        return await asyncio.to_thread(app.state.store.save_policy, policy_id, payload)

    @app.put("/api/policies/{policy_id}")
    async def update_policy(policy_id: str, payload: dict[str, Any] = Body(...)) -> dict:
        ensure_policy_writable()
        if set(payload) == {"payload"} and isinstance(payload.get("payload"), dict):
            payload = payload["payload"]
        return await asyncio.to_thread(app.state.store.save_policy, policy_id, payload)

    @app.delete("/api/policies/{policy_id}")
    async def delete_policy(policy_id: str) -> dict:
        if app.state.runner.lock.status().get("locked"):
            raise StoreError(409, "실행 중에는 정책을 삭제할 수 없습니다")
        return await asyncio.to_thread(app.state.store.delete_policy, policy_id)

    @app.get("/api/runner/lock")
    async def runner_lock() -> dict:
        return app.state.runner.lock.status()

    @app.post("/api/runner/start")
    async def start_runner(request: RunStartRequest) -> dict:
        """실행을 시작한다. 새 정책이 함께 오면 **먼저 저장하고** 그 정책으로 실행한다.

        순서를 뒤집지 않는다. 정책 저장은 preflight 를 통과해야만 성공하므로,
        검증되지 않은 정책으로 시뮬레이션이 시작되는 경로가 없다.
        """

        def run() -> dict:
            injected = None
            if request.policy is not None:
                # 실행 lock 이 이미 있으면 정책 파일을 건드리지 않는다 (B3).
                ensure_policy_writable()
                payload = dict(request.policy)
                payload["id"] = request.policy_id
                injected = app.state.store.save_policy(request.policy_id, payload)
            result = app.state.runner.start(
                run_id=request.run_id,
                policy_id=request.policy_id,
                plan={
                    "start_day": request.start_day,
                    "days": request.days,
                    "agents": request.agents,
                },
            )
            if injected is not None:
                result = {**result, "injected_policy": injected}
            return result

        return await asyncio.to_thread(run)

    @app.get("/api/llm/status")
    async def llm_status() -> dict:
        return await asyncio.to_thread(report_llm.provider_status)

    @app.post("/api/llm/ping")
    async def llm_ping() -> dict:
        """실제 왕복 1회로 연결을 확인한다. 키를 넣은 뒤 바로 확인할 수 있게 한다."""
        return await asyncio.to_thread(report_llm.ping)

    @app.post("/api/runner/stop")
    async def stop_runner() -> dict:
        return await asyncio.to_thread(app.state.runner.request_stop)

    @app.post("/api/runner/release")
    async def release_runner() -> dict:
        # Release is only valid once the owned child is no longer alive. This
        # endpoint cannot be used to erase a live lock.
        return await asyncio.to_thread(app.state.runner.lock.release)

    @app.get("/api/reports/catalog")
    async def report_catalog(run_id: str = Query(...), policy_id: str = Query(...)) -> dict:
        return await asyncio.to_thread(app.state.report_jobs.catalog, run_id=run_id, policy_id=policy_id)

    @app.get("/api/reports/jobs")
    async def report_jobs(run_id: str | None = Query(None)) -> dict:
        return await asyncio.to_thread(app.state.report_jobs.list_jobs, run_id=run_id)

    @app.get("/api/reports/jobs/{job_id}")
    async def report_job(job_id: str) -> dict:
        return await asyncio.to_thread(app.state.report_jobs.get_job, job_id)

    @app.post("/api/reports/jobs")
    async def start_report_job(request: ReportStartRequest) -> dict:
        return await asyncio.to_thread(app.state.report_jobs.create, request.model_dump())

    @app.get("/api/artifacts")
    async def list_artifacts() -> dict:
        return await asyncio.to_thread(app.state.store.artifact_index)

    @app.get("/api/artifacts/{artifact_path:path}")
    async def get_artifact(artifact_path: str) -> FileResponse:
        path = await asyncio.to_thread(app.state.store.artifact, artifact_path)
        return FileResponse(path)

    @app.get("/api/runs/{run_id}/artifacts/{artifact_path:path}")
    async def get_run_artifact(run_id: str, artifact_path: str) -> FileResponse:
        path = await asyncio.to_thread(app.state.store.artifact, artifact_path, run_id=run_id)
        return FileResponse(path)

    ui_dist = REPO_ROOT / "web" / "ui" / "dist"
    if ui_dist.is_dir():
        # API routes are registered above; the root mount only serves the
        # already-built bundle and never replaces an API response.
        app.mount("/", SpaFiles(directory=ui_dist, html=True), name="ui")

    return app


app = create_app()
