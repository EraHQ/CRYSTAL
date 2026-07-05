"""C1 — prompt caching (CC-D1 "both", CC-D2 5-min ephemeral).

Verifies the cache breakpoints CRYS sends — the tools-array prefix (last
tool), the tools+system prefix (system block), and a moving breakpoint on the
conversation prefix (last message's last block) — that the agent's persisted
`working` list is never mutated with markers, and that the run result surfaces
cache-token totals (which C0 records + prices).

Pure adapter/helper tests + Agent-level tests via FakeAnthropic.
asyncio_mode=auto (bare async def).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crystal_cache.agent.adapters.anthropic import render_tools_for_anthropic
from crystal_cache.agent.agent import (
    Agent,
    _messages_with_cache_breakpoint,
    _system_blocks,
)

_EPH = {"type": "ephemeral"}


@dataclass
class _ToolStub:
    """Minimal stand-in for a registered Tool — only the attributes
    `_render_one` reads (name / description / parameters_schema)."""
    name: str
    description: str = "stub tool"
    parameters_schema: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


# --- adapter: only the last tool carries the breakpoint --------------------

def test_render_tools_marks_only_last_tool():
    rendered = render_tools_for_anthropic(
        [_ToolStub("a"), _ToolStub("b"), _ToolStub("c")]
    )
    assert "cache_control" not in rendered[0]
    assert "cache_control" not in rendered[1]
    assert rendered[-1]["cache_control"] == _EPH
    # The tool's own fields survive alongside the marker.
    assert rendered[-1]["name"] == "c"
    assert "input_schema" in rendered[-1]


def test_render_tools_empty_is_safe():
    assert render_tools_for_anthropic([]) == []


# --- system blocks ---------------------------------------------------------

def test_system_blocks_carry_cache_control():
    assert _system_blocks("You are CRYS.") == [{
        "type": "text",
        "text": "You are CRYS.",
        "cache_control": _EPH,
    }]


# --- conversation breakpoint (the moving prefix) ---------------------------

def test_breakpoint_converts_string_content_to_block():
    msgs = [{"role": "user", "content": "hello"}]
    out = _messages_with_cache_breakpoint(msgs)
    assert out[-1]["content"] == [
        {"type": "text", "text": "hello", "cache_control": _EPH}
    ]
    # The persisted original is untouched (no in-place mutation).
    assert msgs[0]["content"] == "hello"


def test_breakpoint_marks_only_last_block():
    msgs = [{
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "a"},
            {"type": "tool_result", "tool_use_id": "t2", "content": "b"},
        ],
    }]
    out = _messages_with_cache_breakpoint(msgs)
    blocks = out[-1]["content"]
    assert "cache_control" not in blocks[0]
    assert blocks[1]["cache_control"] == _EPH
    assert blocks[1]["tool_use_id"] == "t2"
    # Original blocks untouched.
    assert "cache_control" not in msgs[0]["content"][1]


def test_breakpoint_only_on_last_message():
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
        {"role": "user", "content": "q2"},
    ]
    out = _messages_with_cache_breakpoint(msgs)
    assert out[0]["content"] == "q1"  # unchanged
    assert "cache_control" not in out[1]["content"][0]
    assert out[-1]["content"][0]["cache_control"] == _EPH


def test_breakpoint_empty_messages_safe():
    assert _messages_with_cache_breakpoint([]) == []


# --- Agent level: the request carries all three breakpoints ----------------

async def test_agent_sends_cache_breakpoints(customer, tool_state, fake_anthropic):
    fake_anthropic.script_text("done")
    agent = Agent(
        customer=customer,
        llm=fake_anthropic,
        tool_state=tool_state,
    )
    await agent.run(messages=[{"role": "user", "content": "hi"}])

    call = fake_anthropic.assert_called_once()
    assert call["system"][0]["cache_control"] == _EPH
    assert call["tools"][-1]["cache_control"] == _EPH
    assert call["messages"][-1]["content"][-1]["cache_control"] == _EPH


async def test_working_history_never_carries_markers(
    customer, tool_state, fake_anthropic
):
    # Two iterations (unknown tool → error → text). The persisted trajectory
    # (result["messages"]) must carry NO cache_control: the breakpoint is
    # applied only to the per-call copy, never to `working`.
    fake_anthropic.script_tool_use("nonexistent_tool", {}, "tu_1")
    fake_anthropic.script_text("done")
    agent = Agent(
        customer=customer,
        llm=fake_anthropic,
        tool_state=tool_state,
    )
    result = await agent.run(messages=[{"role": "user", "content": "hi"}])

    def _has_marker(msg: dict) -> bool:
        c = msg.get("content")
        return isinstance(c, list) and any(
            isinstance(b, dict) and "cache_control" in b for b in c
        )

    assert not any(_has_marker(m) for m in result["messages"])


# --- result surfaces cache-token totals (feeds C0's cost row) --------------

async def test_result_accumulates_cache_tokens(customer, tool_state, fake_anthropic):
    fake_anthropic.script_tool_use(
        "nonexistent_tool", {}, "tu_1",
        cache_read_input_tokens=1000, cache_creation_input_tokens=300,
    )
    fake_anthropic.script_text(
        "done", cache_read_input_tokens=2000, cache_creation_input_tokens=0,
    )
    agent = Agent(
        customer=customer,
        llm=fake_anthropic,
        tool_state=tool_state,
    )
    result = await agent.run(messages=[{"role": "user", "content": "hi"}])
    assert result["cache_read_tokens"] == 3000
    assert result["cache_creation_tokens"] == 300
