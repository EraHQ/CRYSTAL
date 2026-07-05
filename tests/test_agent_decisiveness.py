"""C5 — agent decisiveness nudge test (cost + parity, 2026-06-17).

CC-D12 scoped C5 to a single static lever: a "Decisiveness" block in the
system prompt that tells the agent to act on what it has retrieved, not repeat
identical tool calls, batch independent lookups into one turn, and stop once it
can answer. Decisiveness is a behavioral property — its real validation is a
live eyeball on a fixed task (per the plan). This test only guards the prompt
*content*, so a future prompt edit can't silently drop the guidance.

R14 note: this assertion is verified by `pytest`.
"""
from __future__ import annotations

from typing import Any

from crystal_cache.agent import (
    build_system_prompt,
    get_registry,
    import_all_tools,
)


def test_system_prompt_includes_decisiveness_guidance(customer: Any):
    """build_system_prompt carries the Decisiveness block and its levers."""
    import_all_tools()
    tools = get_registry().list_for_context("agent")
    prompt = build_system_prompt(customer, tools)

    assert "Decisiveness" in prompt

    low = prompt.lower()
    # The four levers CC-D12 specified.
    assert "act on what you've already retrieved" in low  # use what you have
    assert "repeat a tool call" in low                    # no redundant calls
    assert "one turn" in low                              # batch / parallel
    assert "stop and give your answer" in low             # terminate early

    # Still a coherent POLICIES section (the block didn't displace anything).
    assert "Retrieval first" in prompt
    assert "Multi-turn awareness" in prompt
