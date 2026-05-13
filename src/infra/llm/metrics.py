"""sglang `/metrics` (Prometheus text) scraper + local counter snapshots.

Use to compute cache hit rate, throughput, and latency over a simulation run.
Diff two snapshots to get per-phase deltas.

[한글 요약]
sglang 서버가 노출하는 Prometheus 형식 메트릭을 긁어와서, 캐시 적중률 등을
실측하기 위한 모듈.

쓰는 법:
  before = await scrape_engine_metrics("http://localhost:30000/metrics")
  ... 시뮬레이션 1시간 돌림 ...
  after = await scrape_engine_metrics("...")
  delta = diff_metrics(before, after)
  print(delta.cache_hit_rate)  # 0.65 = 65% 적중

왜 필요한가:
- 우리 설계가 정말 작동하는지 가설 검증
- 적중률이 50% 미만이면 prompt_layers 순서나 batch 정렬에 문제 있음을 의미
- 시뮬레이션 비용 추정에 직결 (적중률이 비용을 결정)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import httpx


# sglang이 노출하는 Prometheus 메트릭 이름. 버전 따라 바뀔 수 있어 정규식으로 추출.
# 매칭 실패 시 0.0으로 처리 (서버 첫 시작 직후 등).
# 메트릭 명이 sglang 새 버전에서 바뀌면 여기를 업데이트하면 됨.
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
        # 분모 0 가드 — 트래픽 없을 때 ZeroDivisionError 방지
        if self.prompt_tokens == 0:
            return 0.0
        return self.cache_hit_tokens / self.prompt_tokens


async def scrape_engine_metrics(metrics_url: str) -> EngineMetrics:
    """Fetch sglang `/metrics` and parse known counter values.

    Missing counters default to 0 — useful when scraping a freshly-started server
    before any requests have completed.
    """
    # sglang의 /metrics 엔드포인트는 Prometheus 텍스트 형식.
    # 예: "sglang:cached_tokens_total 1234567"
    async with httpx.AsyncClient(timeout=10.0) as cli:
        resp = await cli.get(metrics_url)
        resp.raise_for_status()
        body = resp.text

    # 정규식으로 카운터 값을 뽑음. 매칭 안 되면 0.0 (서버 갓 켜진 경우 등).
    parsed: dict[str, float] = {}
    for key, pattern in _METRIC_PATTERNS.items():
        match = pattern.search(body)
        parsed[key] = float(match.group(1)) if match else 0.0

    return EngineMetrics(**parsed)


def diff_metrics(before: EngineMetrics, after: EngineMetrics) -> EngineMetrics:
    """Compute monotonic-counter deltas (after - before). `running_requests` is the latest gauge."""
    # Prometheus 카운터는 단조 증가 → 두 시점의 차이가 구간별 변화량.
    # 시뮬레이션 시작 전 before, 끝나면 after를 찍어두면 그 구간만의 적중률을 알 수 있음.
    # 단, running_requests는 카운터가 아니라 게이지(현재 값)라 차이가 아니라 최신값 사용.
    return EngineMetrics(
        cache_hit_tokens=after.cache_hit_tokens - before.cache_hit_tokens,
        prompt_tokens=after.prompt_tokens - before.prompt_tokens,
        generation_tokens=after.generation_tokens - before.generation_tokens,
        running_requests=after.running_requests,
        scraped_at=after.scraped_at,
    )
