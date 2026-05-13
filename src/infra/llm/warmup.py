"""Prefix cache warmup.

Pushes shared prefixes into sglang's RadixAttention BEFORE the main batch arrives,
so the first agent in each region also gets a cache hit. Without warmup, the very
first request prefills the global prefix at full cost.

[한글 요약]
6만 명 호출 직전에 공통 프롬프트 부분을 미리 한 번 보내는 트릭.
이걸 안 하면 첫 번째 요청은 공통 부분도 모두 새로 계산 → 비싼 비용.

비유: 학교에서 매시간 똑같은 출석 내용을 적는다고 할 때,
- 워밍업 없이 시작 → 첫 학생부터 처음부터 다 적음
- 워밍업 있을 때 → 칠판에 공통 내용을 미리 적어두고 시작 → 모든 학생이 활용

사용 패턴 2가지:
1. 하루 시작 시: 시스템+작업+날짜+정책 등 L1~L4만 채우고 호출
   → 6만 명 전원이 공유하는 공통 prefix가 캐시에 적재
2. 행정동 처리 시작 시: L1~L5(행정동까지) 채우고 호출
   → 해당 행정동 첫 사람부터 동네 정보 부분 적중
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
    # max_tokens=1 → 1 토큰만 생성 (디코드 비용 거의 0)
    # temperature=0.0 → 결정적 생성 (캐시는 입력으로 결정되므로 출력 무관이지만 명시적으로)
    # 진짜 목적은 응답을 받는 게 아니라 sglang 측에 prefix를 등록하는 것.
    await client.generate(layers, max_tokens=max_tokens, temperature=0.0)
