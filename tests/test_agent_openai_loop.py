"""Slice 5c: the agent loop is provider-routed through the seam.

Agent takes an `llm` client instead of a raw Anthropic SDK client and
calls complete_messages. Under an openai-provider client the loop must
send a PLAIN system string and unmarked messages (cache_control is
Anthropic-only decoration), accept shim blocks (plain dicts) back,
dispatch tools, and complete the cycle — proving the loop body is
provider-independent end to end.

R14 note: these assertions are verified by `pytest`; they describe
expected behavior and have not yet been run at authoring time.
"""
from __future__ import annotations

import copy
import json

from crystal_cache.agent import Agent
from crystal_cache.agent.adapters.openai import OpenAIChatShim
from crystal_cache import config


class _FakeOpenAISeam:
    """Seam-shaped fake playing the openai provider for the agent loop."""

    provider = "openai"

    def __init__(self, shims: list[OpenAIChatShim]):
        self._shims = list(shims)
        self.calls: list[dict] = []

    def is_ready(self) -> bool:
        return True

    def complete_messages(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        assert self._shims, "loop called more times than scripted"
        return self._shims.pop(0)


async def test_agent_openai_loop_runs_tool_cycle(
    customer, tool_state, fake_anthropic, monkeypatch,
):
    monkeypatch.setattr(config.settings, "agent_model", "")
    seam = _FakeOpenAISeam([
        OpenAIChatShim(
            content=[{
                "type": "tool_use", "id": "call_1",
                "name": "nonexistent_tool", "input": {},
            }],
            stop_reason="tool_use",
        ),
        OpenAIChatShim(
            content=[{"type": "text", "text": "done"}],
            stop_reason="end_turn",
        ),
    ])
    agent = Agent(customer=customer, llm=seam, tool_state=tool_state)

    result = await agent.run(messages=[{"role": "user", "content": "hi"}])

    # The loop completed the full cycle on shim responses.
    assert result["stop_reason"] == "end_turn"
    assert len(seam.calls) == 2

    # OpenAI path sends a PLAIN system string and unmarked messages:
    # no Anthropic-only cache_control decoration anywhere.
    first = seam.calls[0]
    assert isinstance(first["system"], str)
    assert "cache_control" not in json.dumps(first["messages"])
    assert "cache_control" not in json.dumps(seam.calls[1]["messages"])

    # Under the openai provider with no explicit/house model, the model
    # stays None so the seam resolves its large tier.
    assert first["model"] is None

    # The shim's tool_use dict flowed into history verbatim and the
    # dispatch produced a matching tool_result (error for unknown tool).
    second_msgs = seam.calls[1]["messages"]
    assistant_turn = second_msgs[1]
    assert assistant_turn["content"][0]["type"] == "tool_use"
    tool_turn = second_msgs[2]
    tr = tool_turn["content"][0]
    assert tr["type"] == "tool_result"
    assert tr["tool_use_id"] == "call_1"
    assert tr.get("is_error") is True


async def test_agent_model_none_under_openai_provider(
    customer, tool_state, monkeypatch,
):
    monkeypatch.setattr(config.settings, "agent_model", "")
    seam = _FakeOpenAISeam([])
    agent = Agent(customer=customer, llm=seam, tool_state=tool_state)
    assert agent.model is None


async def test_agent_explicit_model_wins_under_openai(
    customer, tool_state, monkeypatch,
):
    monkeypatch.setattr(config.settings, "agent_model", "")
    seam = _FakeOpenAISeam([])
    agent = Agent(
        customer=customer, llm=seam, tool_state=tool_state,
        model="qwen2.5-72b-instruct",
    )
    assert agent.model == "qwen2.5-72b-instruct"
