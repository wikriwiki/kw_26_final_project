"""Prefix-cache-friendly prompt assembly.

Layer order (shared → unique) is CRITICAL for sglang RadixAttention hit rate.
Reordering fields silently destroys the cache contract — see `tests/unit/infra/llm/`.

Order (least → most unique):

    L1 system            shared by all 60K agents
    L_task task          shared by all agents in same task (PLAN_GENERATION etc.)
    L2 sim_state         shared by all agents in same day
    L3 policy_list       shared by all agents while policies stable
    L4 community_summary shared by all agents
    L5 region_context    shared by ~120 agents in same 행정동
    L6 agent_context     unique per agent (always cache miss)

Rendered as 2 chat messages (system + user). The single-user-message form is
chosen for tokenization stability — fewer role boundaries = fewer chat-template
tokens that could vary across calls.

[한글 요약]
이 파일이 본 LLM 모듈에서 가장 중요. 7개 정보를 sglang에 보내는 메시지로
변환하는데, **순서가 절대로 바뀌면 안 됨**.

순서 규칙: "공유 범위가 큰 것 → 작은 것" 순.
sglang의 캐시는 토큰 시퀀스의 앞부분이 같은지 비교 → 공유 정보가 앞에 있어야
6만 명 사이에서 캐시 적중이 일어남.

예) day=3, 강남구 역삼동 김씨 vs 박씨:
  - L1~L5는 둘 다 똑같음 → 김씨 호출 후 박씨 호출 시 L1~L5는 캐시 적중
  - L6(본인 정보)만 새로 계산
필드 순서를 잘못 바꾸면 캐시 적중이 0%가 됨 (silent failure — 알아채기 어려움).
이를 막기 위해 tests/unit/infra/llm/test_prompt_layers.py에서 순서 검증.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


# 섹션 헤더 — 모든 호출에서 바이트 단위로 동일해야 prefix가 깨지지 않음.
# (헤더 문자열 한 글자만 바뀌어도 토큰 시퀀스가 달라져 캐시 miss)
_TASK_HEADER = "## TASK"
_SIM_STATE_HEADER = "## SIMULATION STATE"
_POLICY_HEADER = "## ACTIVE POLICIES"
_COMMUNITY_HEADER = "## COMMUNITY SUMMARY"
_REGION_HEADER = "## REGION"
_AGENT_HEADER = "## AGENT"


@dataclass(frozen=True, slots=True)
class PromptLayers:
    """Ordered prompt layers. Field order IS the wire order — never reorder."""

    # frozen=True: 한 번 만든 후 수정 불가 → 객체를 dict 키나 캐시 키로 쓸 수 있음
    # slots=True: 메모리 절약 + 오타로 새 필드 추가 차단
    system: str
    task_instruction: str
    sim_state: str
    policy_list: str
    community_summary: str
    region_context: str
    agent_context: str

    def to_messages(self) -> list[dict[str, str]]:
        """Render as OpenAI Chat Completions messages.

        Returns exactly 2 messages: system + concatenated user content.
        Concatenation order is fixed by this method and must not change.
        """
        # 왜 user 메시지 하나로 합쳤나:
        # - 메시지 경계마다 chat template이 토큰을 추가/제거함 (모델/버전 따라 다름)
        # - 메시지 7개로 나누면 토크나이저 결과가 미세하게 흔들려 캐시 적중 깨짐
        # - 메시지 1개 안에서 텍스트로 구분하면 토큰 시퀀스가 결정적 → 캐시 안전
        user_body = "\n\n".join(
            [
                f"{_TASK_HEADER}\n{self.task_instruction}",
                f"{_SIM_STATE_HEADER}\n{self.sim_state}",
                f"{_POLICY_HEADER}\n{self.policy_list}",
                f"{_COMMUNITY_HEADER}\n{self.community_summary}",
                f"{_REGION_HEADER}\n{self.region_context}",
                f"{_AGENT_HEADER}\n{self.agent_context}",
            ]
        )
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user_body},
        ]

    def with_overrides(self, **changes: str) -> PromptLayers:
        """Return a new PromptLayers with the given fields replaced."""
        # 사용 예: 같은 행정동 사람들에게 agent_context만 다른 PromptLayers를 만들 때.
        # frozen이라 직접 수정 불가 → dataclasses.replace로 새 객체 생성.
        return replace(self, **changes)


def empty_layers(*, system: str = "", task: str = "") -> PromptLayers:
    """Construct a PromptLayers with all variable layers empty.

    Use for prefix warmup: fill `system` + `task` (the always-shared part) and
    leave the rest blank to push that prefix into the radix cache cheaply.
    """
    # 워밍업 전용 헬퍼. 진짜 6만 명 호출 전에 공통 prefix만 채워서 sglang에
    # 한 번 보내면, 캐시에 적재됨 → 이후 첫 호출부터 적중.
    # warmup.prewarm()과 짝으로 사용됨.
    return PromptLayers(
        system=system,
        task_instruction=task,
        sim_state="",
        policy_list="",
        community_summary="",
        region_context="",
        agent_context="",
    )
