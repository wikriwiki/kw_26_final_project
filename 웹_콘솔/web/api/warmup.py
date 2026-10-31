"""서버를 켜자마자 화면이 쓰는 집계를 한 번씩 미리 계산한다 (`WEB_WARMUP=1`).

시연용 노트북에서 정책 결과 화면을 처음 열면 비교(부트스트랩 500회)가 정책마다 2초 안팎
걸린다. 데이터 묶음을 막 풀었을 때는 하루 집계 캐시까지 새로 만들어야 해서 첫 목록이
1분 넘게 멈췄다. 계산이 도는 동안에는 같은 서버의 다른 요청(지도 조각·목록)도 함께 늦어진다.
그래서 브라우저를 열기 전에 서버가 혼자 순서대로 다 계산해 메모리에 올려 둔다.

계산 결과는 각 저장소의 기존 캐시에 그대로 들어간다. 여기서 새로 저장하는 값은 없다.
진행 상황은 `/api/warmup` 에 실린다. 실행 스크립트는 ``state == "done"`` 을
본 뒤에 브라우저를 연다.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

from fastapi import FastAPI


class Warmup:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state: dict[str, Any] = {"state": "off", "done": 0, "total": 0, "seconds": None, "failed": [], "slowest": []}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {**self.state, "failed": list(self.state["failed"]), "slowest": list(self.state["slowest"])}

    def _set(self, **values: Any) -> None:
        with self._lock:
            self.state.update(values)

    def start(self, app: FastAPI) -> None:
        self._set(state="running", done=0, total=0, seconds=None, failed=[], slowest=[])
        threading.Thread(target=self._run, args=(app,), name="warmup", daemon=True).start()

    def _tasks(self, app: FastAPI) -> list[tuple[str, Callable[[], Any]]]:
        from . import experiment_query

        tasks: list[tuple[str, Callable[[], Any]]] = []
        experiments = getattr(app.state, "experiments", None)
        if experiments is not None:
            # 목록이 모든 팔의 하루 집계를 읽는다 — 디스크 캐시가 없으면 여기서 만든다
            tasks.append(("experiments", experiments.list))
            for item in experiments.list().get("items") or []:
                eid = item.get("experiment_id")
                if not eid:
                    continue
                tasks += [
                    (f"{eid}/detail", lambda e=eid: experiments.detail(e)),
                    (f"{eid}/daily", lambda e=eid: experiments.daily(e)),
                    (f"{eid}/agents", lambda e=eid: experiments.roster(e)),
                    (f"{eid}/comparison", lambda e=eid: experiments.comparison(e)),
                ]
                if item.get("kind") == "policy_paired":
                    tasks.append((f"{eid}/effect", lambda e=eid: experiment_query.group_effect(experiments, e, {})))
        store = app.state.store
        try:
            runs = store.runs_index().get("items") or []
        except Exception:  # noqa: BLE001 — 실행 폴더가 없어도 실험 화면은 데워 둔다
            runs = []
        for run in runs:
            rid = run.get("run_id")
            if not rid:
                continue
            tasks += [
                (f"runs/{rid}", lambda r=rid: store.run_detail(r)),
                (f"runs/{rid}/days", lambda r=rid: store.run_days(r)),
                (f"runs/{rid}/events", lambda r=rid: store.events_summary(r)),
                (f"runs/{rid}/failures", lambda r=rid: store.failures(r)),
            ]
        return tasks

    def _run(self, app: FastAPI) -> None:
        started = time.perf_counter()
        try:
            tasks = self._tasks(app)
        except Exception as exc:  # noqa: BLE001
            self._set(state="done", seconds=round(time.perf_counter() - started, 1), failed=[f"목록: {exc}"])
            return
        self._set(total=len(tasks))
        timings: list[tuple[float, str]] = []
        for index, (label, task) in enumerate(tasks, start=1):
            began = time.perf_counter()
            try:
                task()
            except Exception as exc:  # noqa: BLE001 — 하나가 실패해도 나머지는 데운다. 화면은 그 요청에서 같은 오류를 보인다
                with self._lock:
                    self.state["failed"].append(f"{label}: {getattr(exc, 'message', exc)}")
            timings.append((time.perf_counter() - began, label))
            self._set(done=index)
        slowest = [f"{label} {seconds:.1f}s" for seconds, label in sorted(timings, reverse=True)[:5]]
        self._set(state="done", seconds=round(time.perf_counter() - started, 1), slowest=slowest)
