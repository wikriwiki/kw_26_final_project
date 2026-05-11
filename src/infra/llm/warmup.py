"""Prefix cache warmup.

Pushes shared prefixes into sglang's RadixAttention BEFORE the main batch arrives,
so the first agent in each region also gets a cache hit. Without warmup, the very
first request prefills the global prefix at full cost.
"""
from __future__ import annotations

from .engine_client import EngineClient
from .prompt_layers import PromptLayers


async def prewarm(
    client: EngineClient,
    layers: PromptLayers,
    *,
    max_tokens: int = 1,
) -> None:
    """Issue a 1-token completion to push `layers` into the radix cache.

    Patterns:
      - Day-start: fill L1-L4 (global) + empty L5/L6, call once → global prefix cached.
      - Region-start: fill L1-L5 (global + that region) + empty L6, call once → adds
        region branch to the cache tree.
    """
    await client.generate(layers, max_tokens=max_tokens, temperature=0.0)
