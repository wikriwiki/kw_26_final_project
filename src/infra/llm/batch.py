"""Batch controller — submits requests sorted by group_key (region) for prefix locality.

Same group_key → executes near in time → sglang's RadixAttention shares the L5
(region) prefix across all agents in that group. Concurrency is capped by the
active model's `max_running_requests` from `ModelProfile`.

[한글 요약]
6만 개의 요청을 효율적으로 sglang에 던지는 컨트롤러.

핵심 트릭 — group_key(보통 행정동 코드) 정렬:
- 같은 행정동 사람들을 시간상 모아서 보냄
- → sglang이 동네 정보 부분(L5)을 캐시에 두고 재사용
- → 같은 행정동 평균 120명 중 119명이 동네 정보 부분에서 캐시 적중

세마포어로 동시성 제한:
- Qwen 모드: 동시 256개
- EXAONE 모드: 동시 96개 (GPU 메모리 한도)
- 6만 명 한꺼번에 던지면 서버가 OOM으로 죽음

실패 격리:
- 한 명 요청이 실패해도 나머지는 계속 진행
- 결과 리스트의 해당 위치에 예외 객체가 들어감
- 호출자가 isinstance(r, Exception)으로 골라낼 수 있음
- 6만 명 7시간 시뮬 중 한 명 실패로 전체 중단되는 사고 방지
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from .engine_client import EngineClient
from .prompt_layers import PromptLayers
from .structured import generate_structured


@dataclass(slots=True)
class _Submitted:
    """Internal: one queued submission."""

    layers: PromptLayers
    schema: type[BaseModel] | None
    group_key: str
    max_tokens: int
    temperature: float


class BatchController:
    """Collects requests, sorts by group_key, executes under bounded concurrency.

    Usage:
        ctrl = BatchController(client)
        for agent in agents:
            ctrl.submit(layers, group_key=agent.dong_code, schema=PlanSchema)
        results = await ctrl.flush()  # list aligned with submission order

    Per-request exceptions are RETURNED in their slot, not raised. Inspect with
    `isinstance(r, Exception)` so one bad agent doesn't abort the whole batch.
    """

    def __init__(
        self,
        client: EngineClient,
        *,
        max_concurrent: int | None = None,
    ) -> None:
        self.client = client
        self.max_concurrent: int = max_concurrent or client.profile.max_running_requests
        self._queue: list[_Submitted] = []

    def __len__(self) -> int:
        return len(self._queue)

    def submit(
        self,
        layers: PromptLayers,
        *,
        group_key: str,
        schema: type[BaseModel] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> None:
        """Queue a request. Result available in submission order from `flush()`.

        `group_key` controls execution ordering — same key = same region prefix.
        Use `agent.dong_code` (행정동) for Dawn phase Plan generation.
        """
        # 큐에만 추가, 실제 호출은 일어나지 않음.
        # group_key는 보통 행정동 코드 — flush()가 이걸로 정렬해서 캐시 적중 극대화.
        self._queue.append(
            _Submitted(
                layers=layers,
                schema=schema,
                group_key=group_key,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )

    async def flush(self) -> list[Any]:
        """Execute all queued requests; return results aligned with submission order.

        Sorting happens internally so the wire-order matches group_key (region),
        but the returned list preserves the order in which `submit()` was called.
        """
        if not self._queue:
            return []

        # 큐를 로컬로 옮기고 즉시 초기화 — flush 중에 새 submit이 들어와도 안전.
        queue = self._queue
        self._queue = []

        # 세마포어로 동시 호출 수를 모델 한도 이하로 제한.
        sem = asyncio.Semaphore(self.max_concurrent)
        # 실행 순서: group_key 정렬된 인덱스 순. 결과 리스트는 원래 순서 유지.
        # 예) 제출 순서 [강남, 마포, 강남, 종로] → 실행 순서 [강남, 강남, 마포, 종로]
        #    → 같은 강남 두 명이 시간상 붙어서 L5 prefix 공유
        order = sorted(range(len(queue)), key=lambda i: queue[i].group_key)
        results: list[Any] = [None] * len(queue)

        async def _run(idx: int) -> None:
            req = queue[idx]
            # 세마포어 획득 — 동시 진행 수 제한
            async with sem:
                try:
                    # schema 있으면 구조화 출력, 없으면 plain text
                    if req.schema is not None:
                        results[idx] = await generate_structured(
                            self.client,
                            req.layers,
                            req.schema,
                            max_tokens=req.max_tokens,
                            temperature=req.temperature,
                        )
                    else:
                        results[idx] = await self.client.generate(
                            req.layers,
                            max_tokens=req.max_tokens,
                            temperature=req.temperature,
                        )
                except Exception as e:
                    # ※ 예외를 raise하지 않고 결과 리스트에 담음.
                    # asyncio.gather가 한 task 실패로 전체를 취소하는 걸 방지.
                    # 호출자는 isinstance(r, Exception)으로 골라냄.
                    results[idx] = e

        # 모든 task를 동시에 시작 — 세마포어가 실제 동시 실행 수를 제한.
        await asyncio.gather(*(_run(i) for i in order))
        return results
