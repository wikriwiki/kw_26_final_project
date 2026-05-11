"""Batch controller — submits requests sorted by group_key (region) for prefix locality.

Same group_key → executes near in time → sglang's RadixAttention shares the L5
(region) prefix across all agents in that group. Concurrency is capped by the
active model's `max_running_requests` from `ModelProfile`.
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

        queue = self._queue
        self._queue = []

        sem = asyncio.Semaphore(self.max_concurrent)
        order = sorted(range(len(queue)), key=lambda i: queue[i].group_key)
        results: list[Any] = [None] * len(queue)

        async def _run(idx: int) -> None:
            req = queue[idx]
            async with sem:
                try:
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
                    results[idx] = e

        await asyncio.gather(*(_run(i) for i in order))
        return results
