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
"""
from __future__ import annotations

from dataclasses import dataclass, replace


# Section headers — must be byte-identical across calls or the prefix breaks.
_TASK_HEADER = "## TASK"
_SIM_STATE_HEADER = "## SIMULATION STATE"
_POLICY_HEADER = "## ACTIVE POLICIES"
_COMMUNITY_HEADER = "## COMMUNITY SUMMARY"
_REGION_HEADER = "## REGION"
_AGENT_HEADER = "## AGENT"


@dataclass(frozen=True, slots=True)
class PromptLayers:
    """Ordered prompt layers. Field order IS the wire order — never reorder."""

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
        return replace(self, **changes)


def empty_layers(*, system: str = "", task: str = "") -> PromptLayers:
    """Construct a PromptLayers with all variable layers empty.

    Use for prefix warmup: fill `system` + `task` (the always-shared part) and
    leave the rest blank to push that prefix into the radix cache cheaply.
    """
    return PromptLayers(
        system=system,
        task_instruction=task,
        sim_state="",
        policy_list="",
        community_summary="",
        region_context="",
        agent_context="",
    )
