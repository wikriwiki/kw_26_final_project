"""Tests for prompt_layers.

These guard the prefix-cache contract: section order and byte-identity.
Reordering will silently destroy cache hit rate, so these tests must NEVER be
relaxed without explicit team review.
"""
from __future__ import annotations

import pytest

from src.infra.llm import PromptLayers, empty_layers


def _full_layers() -> PromptLayers:
    return PromptLayers(
        system="SYS",
        task_instruction="TASK",
        sim_state="SIM",
        policy_list="POL",
        community_summary="COMM",
        region_context="REGION",
        agent_context="AGENT",
    )


def test_to_messages_returns_two_messages() -> None:
    messages = _full_layers().to_messages()
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_system_message_contains_only_system_field() -> None:
    messages = _full_layers().to_messages()
    assert messages[0]["content"] == "SYS"


def test_user_content_section_order_is_fixed() -> None:
    """Section order MUST be: task → sim_state → policy → community → region → agent.

    Reordering breaks the radix-cache prefix tree contract.
    """
    user_content = _full_layers().to_messages()[1]["content"]
    expected_order = [
        "TASK",
        "SIMULATION STATE",
        "ACTIVE POLICIES",
        "COMMUNITY SUMMARY",
        "REGION",
        "AGENT",
    ]
    positions = [user_content.index(section) for section in expected_order]
    assert positions == sorted(positions), (
        f"Section order broken — found positions {positions}. "
        "This is a prefix-cache regression."
    )


def test_identical_inputs_produce_byte_identical_messages() -> None:
    a = empty_layers(system="s", task="t")
    b = empty_layers(system="s", task="t")
    assert a.to_messages() == b.to_messages()


def test_with_overrides_preserves_other_fields() -> None:
    base = _full_layers()
    modified = base.with_overrides(agent_context="AGENT_2")
    assert modified.agent_context == "AGENT_2"
    assert modified.system == "SYS"
    assert modified.region_context == "REGION"


def test_with_overrides_returns_new_instance() -> None:
    base = _full_layers()
    modified = base.with_overrides(agent_context="X")
    assert base.agent_context == "AGENT"
    assert modified is not base


def test_frozen_layers_cannot_be_mutated() -> None:
    layers = empty_layers(system="s", task="t")
    with pytest.raises(AttributeError):
        layers.system = "different"  # type: ignore[misc]


def test_empty_layers_fills_only_system_and_task() -> None:
    layers = empty_layers(system="s", task="t")
    assert layers.system == "s"
    assert layers.task_instruction == "t"
    assert layers.sim_state == ""
    assert layers.policy_list == ""
    assert layers.community_summary == ""
    assert layers.region_context == ""
    assert layers.agent_context == ""
