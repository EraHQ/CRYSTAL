"""C4 — agent tool-output trimming tests (cost + parity, 2026-06-17).

Covers `crystal_cache.agent.agent._cap_tool_output` and its wiring into the
tool-dispatch loop behind CC_AGENT_TOOL_OUTPUT_MAX_CHARS. The cap bounds a
single tool_result's content in the model-facing trajectory (head + tail
around a truncation marker), while `tool_calls_log` keeps the full untrimmed
output. The integration tests compare the model-facing `tool_result` content
against `_cap_tool_output`'s own output, so they prove the wiring regardless of
the real tool's output size, and confirm the log copy is unaffected.

R14 note: these assertions are verified by `pytest`; they describe expected
behavior and have not yet been run at authoring time.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from crystal_cache import config
from crystal_cache.agent import Agent
from crystal_cache.agent.agent import _cap_tool_output


# ---------------------------------------------------------------------------
# Unit — _cap_tool_output
# ---------------------------------------------------------------------------

def test_under_cap_returned_unchanged():
    """Content at/under the cap is returned verbatim."""
    content = "a short tool result"
    assert _cap_tool_output(content, 1000) == content
    assert _cap_tool_output(content, len(content)) == content


def test_zero_or_negative_disables_capping():
    """max_chars <= 0 means 'no cap' (the codebase idiom)."""
    content = "x" * 5000
    assert _cap_tool_output(content, 0) == content
    assert _cap_tool_output(content, -10) == content


def test_over_cap_truncates_head_and_tail_with_marker():
    """A large output keeps its head and tail around a truncation marker."""
    content = "HEAD" + ("m" * 2000) + "TAIL"
    out = _cap_tool_output(content, 100)
    assert "truncated" in out
    assert out.startswith("HEAD")   # head preserved (structure)
    assert out.endswith("TAIL")     # tail preserved (recent lines)
    assert len(out) < len(content)
    # Retained content (head+tail) is bounded by max_chars; marker is extra.
    assert len(out) <= 100 + 64


def test_capping_never_grows_a_barely_over_output():
    """If the marker would cost more than it saves, leave the content as-is."""
    content = "y" * 55  # just over a 50-char cap; marker (~30) would grow it
    assert _cap_tool_output(content, 50) == content


def test_dropped_count_reflects_removed_chars():
    """The marker reports how many characters were dropped."""
    content = "z" * 1000
    out = _cap_tool_output(content, 100)
    # head 60 + tail 40 retained -> 900 dropped.
    assert "900 chars truncated" in out


# ---------------------------------------------------------------------------
# Integration — Agent.run wiring (off vs on)
# ---------------------------------------------------------------------------

def _extract_tool_result_content(fake: Any, call_index: int) -> str:
    """Pull the tool_result content string the model saw on a given call."""
    last = fake.calls[call_index]["messages"][-1]
    assert last["role"] == "user"
    block = last["content"][0]
    assert block["type"] == "tool_result"
    return block["content"]


@pytest.mark.asyncio
async def test_tool_output_uncapped_by_default(
    customer: Any,
    tool_state: dict[str, Any],
    fake_anthropic: Any,
):
    """With the knob at its default (0 = off), the model sees the full
    serialized tool output and the log holds the same full output."""
    fake_anthropic.script_tool_use(
        "knowledge_search", {"query": "anything"}, "tu_1",
    )
    fake_anthropic.script_text("done")

    agent = Agent(
        customer=customer,
        llm=fake_anthropic,
        tool_state=tool_state,
    )
    result = await agent.run(
        messages=[{"role": "user", "content": "look something up"}],
    )

    tc = result["tool_calls"][0]
    full = json.dumps(tc["output"], default=str)
    model_content = _extract_tool_result_content(fake_anthropic, 1)
    assert model_content == full
    assert "truncated" not in model_content


@pytest.mark.asyncio
async def test_tool_output_capped_when_knob_set(
    customer: Any,
    tool_state: dict[str, Any],
    fake_anthropic: Any,
    monkeypatch: Any,
):
    """With the knob set, the model-facing tool_result is capped to exactly
    what `_cap_tool_output` produces, while the full output stays in
    tool_calls_log untouched."""
    # Patch the setting BEFORE constructing the agent (it reads it in __init__).
    monkeypatch.setattr(config.settings, "agent_tool_output_max_chars", 10)

    fake_anthropic.script_tool_use(
        "knowledge_search", {"query": "anything"}, "tu_1",
    )
    fake_anthropic.script_text("done")

    agent = Agent(
        customer=customer,
        llm=fake_anthropic,
        tool_state=tool_state,
    )
    result = await agent.run(
        messages=[{"role": "user", "content": "look something up"}],
    )

    tc = result["tool_calls"][0]
    # Full output preserved in the telemetry log (a dict, not the capped str).
    assert isinstance(tc["output"], dict)
    assert "fact_count" in tc["output"]

    full = json.dumps(tc["output"], default=str)
    model_content = _extract_tool_result_content(fake_anthropic, 1)
    # Wiring: the model saw exactly the capped form, with the configured cap.
    assert model_content == _cap_tool_output(full, 10)
    # And capping actually engaged (the empty-bank result exceeds the cap).
    assert model_content != full
    assert "truncated" in model_content
