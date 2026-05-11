"""sglang `/metrics` (Prometheus text) scraper + local counter snapshots.

Use to compute cache hit rate, throughput, and latency over a simulation run.
Diff two snapshots to get per-phase deltas.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import httpx


_METRIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "cache_hit_tokens": re.compile(r"^sglang:cached_tokens_total\s+([\d.eE+\-]+)", re.MULTILINE),
    "prompt_tokens": re.compile(r"^sglang:prompt_tokens_total\s+([\d.eE+\-]+)", re.MULTILINE),
    "generation_tokens": re.compile(r"^sglang:generation_tokens_total\s+([\d.eE+\-]+)", re.MULTILINE),
    "running_requests": re.compile(r"^sglang:num_running_reqs\s+([\d.eE+\-]+)", re.MULTILINE),
}


@dataclass(slots=True)
class EngineMetrics:
    """Counter snapshot scraped from sglang's Prometheus endpoint.

    Counters are monotonic; subtract two snapshots to get per-phase deltas via
    `diff_metrics`. `running_requests` is a gauge (not delta'd).
    """

    cache_hit_tokens: float = 0.0
    prompt_tokens: float = 0.0
    generation_tokens: float = 0.0
    running_requests: float = 0.0
    scraped_at: float = field(default_factory=time.time)

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of input tokens served from cache. Returns 0.0 if no traffic."""
        if self.prompt_tokens == 0:
            return 0.0
        return self.cache_hit_tokens / self.prompt_tokens


async def scrape_engine_metrics(metrics_url: str) -> EngineMetrics:
    """Fetch sglang `/metrics` and parse known counter values.

    Missing counters default to 0 — useful when scraping a freshly-started server
    before any requests have completed.
    """
    async with httpx.AsyncClient(timeout=10.0) as cli:
        resp = await cli.get(metrics_url)
        resp.raise_for_status()
        body = resp.text

    parsed: dict[str, float] = {}
    for key, pattern in _METRIC_PATTERNS.items():
        match = pattern.search(body)
        parsed[key] = float(match.group(1)) if match else 0.0

    return EngineMetrics(**parsed)


def diff_metrics(before: EngineMetrics, after: EngineMetrics) -> EngineMetrics:
    """Compute monotonic-counter deltas (after - before). `running_requests` is the latest gauge."""
    return EngineMetrics(
        cache_hit_tokens=after.cache_hit_tokens - before.cache_hit_tokens,
        prompt_tokens=after.prompt_tokens - before.prompt_tokens,
        generation_tokens=after.generation_tokens - before.generation_tokens,
        running_requests=after.running_requests,
        scraped_at=after.scraped_at,
    )
