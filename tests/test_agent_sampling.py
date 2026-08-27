"""Audit (e) stage 1.11 — temperature on the agent turn (Q1=A / Q2=A).

The one sampling param ported from the proxy (retirement item 12).
Pins here cover the loop and the request schema; the seam's four routes
are pinned in test_llm_seam.py (present when set, absent when None).

- Loop: Agent passes its temperature to the seam — and passes NOTHING
  when unset, so every pre-1.11 fake signature stays valid (the
  conditional-kwarg contract).
- Request: AgentRequest range-checks 0..1; out-of-range is a validation
  error, in-range (including 0.0 — the reproducibility value) threads.

asyncio_mode=auto.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from crystal_cache.agent.agent import Agent
from crystal_cache.endpoints.agent import AgentRequest


class _CaptureLLM:
    """Seam fake capturing complete_messages kwargs.

    Deliberately DECLARES no temperature param — proving the loop omits
    the kwarg when unset (a pre-1.11 fake signature must keep working)
    — while **kw admits it when the loop sends one.
    """

    provider = "anthropic"

    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] = {}

    def complete_messages(self, **kw: Any) -> Any:
        self.last_kwargs = kw
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=1, output_tokens=1,
                cache_creation_input_tokens=None,
                cache_read_input_tokens=None,
            ),
        )


def _agent(customer: Any, **kw: Any) -> Agent:
    return Agent(
        customer=customer,
        llm=_CaptureLLM(),
        tool_state={"store": None},
        **kw,
    )


async def test_call_model_passes_temperature_when_set(customer):
    agent = _agent(customer, temperature=0.0)
    await agent._call_model(system="s", messages=[
        {"role": "user", "content": "q"},
    ], tools=[])
    # 0.0 is a real value — the reproducibility pin (Guarantee #6).
    assert agent.llm.last_kwargs["temperature"] == 0.0


async def test_call_model_omits_temperature_when_unset(customer):
    agent = _agent(customer)
    await agent._call_model(system="s", messages=[
        {"role": "user", "content": "q"},
    ], tools=[])
    # The conditional-kwarg contract: unset → the call shape is exactly
    # pre-1.11, so fakes without the param keep working.
    assert "temperature" not in agent.llm.last_kwargs


def test_agent_request_rejects_out_of_range_temperature():
    base = {"messages": [{"role": "user", "content": "q"}]}
    with pytest.raises(ValidationError):
        AgentRequest(**base, temperature=1.5)
    with pytest.raises(ValidationError):
        AgentRequest(**base, temperature=-0.1)


def test_agent_request_threads_in_range_temperature():
    base = {"messages": [{"role": "user", "content": "q"}]}
    assert AgentRequest(**base, temperature=0.0).temperature == 0.0
    assert AgentRequest(**base, temperature=1.0).temperature == 1.0
    assert AgentRequest(**base).temperature is None
