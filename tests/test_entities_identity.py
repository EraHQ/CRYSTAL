"""Entities layer slice A — identity composition + agent prompt threading.

Design gate 2026-07-22 (SESSION_HANDOFF 0c): Q1=B+C operator resolution,
Q3=C stable identity block in the cached prefix + per-query variance
tail after the C1 breakpoint, Q4 read posture (pinned digest carries
only stated/verified facts).

Two layers under test:
- crystal_cache.agent.identity (FastAPI-free): resolution edges, digest
  provenance, tail relevance, fail-safety — via duck-typed fakes.
- The Agent loop: identity_block renders into the stable (cached)
  system block; system_tail rides as a second uncached block; the
  no-identity configuration is byte-identical to before the layer
  existed.
"""

from __future__ import annotations

import pytest

from crystal_cache.agent import Agent
from crystal_cache.agent.identity import (
    compose_identity_context,
    resolve_operator,
)

from fakes import FakeAnthropic


# ---------------------------------------------------------------------------
# Duck-typed fakes for the composition layer
# ---------------------------------------------------------------------------

class _Op:
    def __init__(
        self, id, team, status="active", name="Anthony", role="admin",
    ):
        self.id = id
        self.team_id = team
        self.status = status
        self.display_name = name
        self.role = role


class _Ent:
    def __init__(self, crystal_id):
        self.crystal_id = crystal_id


class _Fact:
    def __init__(self, claim, sk="model_reasoning", vb=None, vec=None):
        self.claim_text = claim
        self.source_kind = sk
        self.verified_by = vb
        self.vector = vec or []


class _Store:
    def __init__(self, ops, entity=None, facts=None, boom=False):
        self._ops = ops
        self._entity = entity
        self._facts = facts or []
        self._boom = boom

    async def get_operator_by_id(self, oid):
        if self._boom:
            raise RuntimeError("db down")
        return next((o for o in self._ops if o.id == oid), None)

    async def list_operators_for_team(self, team):
        if self._boom:
            raise RuntimeError("db down")
        return [o for o in self._ops if o.team_id == team]

    async def get_entity_for_operator(self, oid):
        return self._entity

    async def list_facts_for_crystal(self, cid):
        return self._facts


class _Encoder:
    """Duck-typed native encoder for the relevance tail."""

    def encode_native(self, text):
        return [1.0, 0.0]


# ---------------------------------------------------------------------------
# Resolution (Q1=B+C)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sole_active_operator_resolves():
    op = await resolve_operator(
        store=_Store([_Op("op1", "c1")]), customer_id="c1", operator_id=None,
    )
    assert op is not None and op.id == "op1"


@pytest.mark.asyncio
async def test_zero_or_multiple_active_resolve_nothing():
    assert await resolve_operator(
        store=_Store([]), customer_id="c1", operator_id=None,
    ) is None
    assert await resolve_operator(
        store=_Store([_Op("a", "c1"), _Op("b", "c1")]),
        customer_id="c1", operator_id=None,
    ) is None


@pytest.mark.asyncio
async def test_suspended_operators_do_not_count():
    op = await resolve_operator(
        store=_Store([_Op("a", "c1"), _Op("b", "c1", status="suspended")]),
        customer_id="c1", operator_id=None,
    )
    assert op is not None and op.id == "a"


@pytest.mark.asyncio
async def test_explicit_id_is_team_and_status_checked():
    assert await resolve_operator(
        store=_Store([_Op("op1", "OTHER_TEAM")]),
        customer_id="c1", operator_id="op1",
    ) is None
    assert await resolve_operator(
        store=_Store([_Op("op1", "c1", status="suspended")]),
        customer_id="c1", operator_id="op1",
    ) is None
    op = await resolve_operator(
        store=_Store([_Op("op1", "c1"), _Op("op2", "c1")]),
        customer_id="c1", operator_id="op2",
    )
    assert op is not None and op.id == "op2"


# ---------------------------------------------------------------------------
# Composition (Q3 shapes, Q4 read posture)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_identity_block_shape_and_digest_provenance():
    facts = [
        _Fact("Prefers terse answers", sk="operator_stated"),
        _Fact("Works at Era HQ", vb="curator@era"),
        _Fact("Seems to work late"),  # inferred, unverified -> NOT pinned
    ]
    block, tail = await compose_identity_context(
        store=_Store([_Op("op1", "c1")], entity=_Ent("cry1"), facts=facts),
        customer_id="c1", operator_id=None, query_text="", encoder=None,
    )
    assert block is not None and tail is None
    assert "You are speaking with Anthony (admin)." in block
    assert "- Prefers terse answers" in block
    assert "- Works at Era HQ" in block
    assert "Seems to work late" not in block
    assert "dedicated memory crystal" in block


@pytest.mark.asyncio
async def test_tail_is_query_relevant_and_excludes_pinned_lines():
    facts = [
        _Fact("Prefers terse answers", sk="operator_stated", vec=[1.0, 0.0]),
        _Fact("Owns a dental practice", vec=[0.9, 0.1]),
        _Fact("Allergic to peanuts", vec=[0.0, 1.0]),
    ]
    block, tail = await compose_identity_context(
        store=_Store([_Op("op1", "c1")], entity=_Ent("cry1"), facts=facts),
        customer_id="c1", operator_id=None,
        query_text="what practice does he own", encoder=_Encoder(),
    )
    assert tail is not None and tail.startswith("OPERATOR CONTEXT")
    assert "- Owns a dental practice" in tail
    assert "terse" not in tail
    assert "- Owns a dental practice" not in block


@pytest.mark.asyncio
async def test_no_operator_and_store_failure_yield_none_none():
    assert await compose_identity_context(
        store=_Store([]), customer_id="c1", operator_id=None,
        query_text="q", encoder=None,
    ) == (None, None)
    # Fail-safe (P0.44): a raising store must never break the run.
    assert await compose_identity_context(
        store=_Store([_Op("op1", "c1")], boom=True), customer_id="c1",
        operator_id=None, query_text="q", encoder=None,
    ) == (None, None)


@pytest.mark.asyncio
async def test_entity_without_crystal_still_yields_identity():
    block, tail = await compose_identity_context(
        store=_Store([_Op("op1", "c1")], entity=_Ent(None)),
        customer_id="c1", operator_id=None, query_text="q", encoder=None,
    )
    assert block is not None and "Anthony" in block and tail is None


# ---------------------------------------------------------------------------
# Agent loop threading (identity into the cached prefix, tail after it)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_identity_block_renders_into_stable_cached_block(
    customer, tool_state, fake_anthropic,
):
    fake_anthropic.script_text("hi")
    agent = Agent(
        customer=customer, llm=fake_anthropic, tool_state=tool_state,
        identity_block="OPERATOR\n\nYou are speaking with Anthony (admin).",
    )
    await agent.run([{"role": "user", "content": "hello"}])
    sent_system = fake_anthropic.calls[0]["system"]
    assert isinstance(sent_system, list) and len(sent_system) == 1
    block = sent_system[0]
    assert block.get("cache_control"), "stable block carries the breakpoint"
    assert "You are speaking with Anthony (admin)." in block["text"]


@pytest.mark.asyncio
async def test_system_tail_is_second_uncached_block(
    customer, tool_state, fake_anthropic,
):
    fake_anthropic.script_text("hi")
    agent = Agent(
        customer=customer, llm=fake_anthropic, tool_state=tool_state,
        identity_block="OPERATOR\nAnthony.",
        system_tail="OPERATOR CONTEXT (relevant to this message):\n- x",
    )
    await agent.run([{"role": "user", "content": "hello"}])
    sent_system = fake_anthropic.calls[0]["system"]
    assert isinstance(sent_system, list) and len(sent_system) == 2
    stable, tail = sent_system
    assert stable.get("cache_control"), "breakpoint stays on the stable block"
    assert "cache_control" not in tail, "tail rides AFTER the breakpoint"
    assert tail["text"].startswith("OPERATOR CONTEXT")


@pytest.mark.asyncio
async def test_no_identity_no_tail_is_byte_identical(
    customer, tool_state, fake_anthropic,
):
    fake_anthropic.script_text("hi")
    agent = Agent(customer=customer, llm=fake_anthropic, tool_state=tool_state)
    await agent.run([{"role": "user", "content": "hello"}])
    baseline = fake_anthropic.calls[0]["system"]

    fake2 = FakeAnthropic()
    fake2.script_text("hi")
    agent2 = Agent(
        customer=customer, llm=fake2, tool_state=tool_state,
        identity_block=None, system_tail=None,
    )
    await agent2.run([{"role": "user", "content": "hello"}])
    assert fake2.calls[0]["system"] == baseline


@pytest.mark.asyncio
async def test_explicit_system_override_ignores_identity_block(
    customer, tool_state, fake_anthropic,
):
    """A caller-supplied `system` means full control: identity is not
    injected into it (the tail, a delivery concern, still applies)."""
    fake_anthropic.script_text("hi")
    agent = Agent(
        customer=customer, llm=fake_anthropic, tool_state=tool_state,
        identity_block="OPERATOR\nAnthony IDENTITY-SENTINEL.",
    )
    await agent.run(
        [{"role": "user", "content": "hello"}],
        system="CUSTOM OVERRIDE PROMPT",
    )
    block = fake_anthropic.calls[0]["system"][0]
    assert block["text"].startswith("CUSTOM OVERRIDE PROMPT")
    assert "IDENTITY-SENTINEL" not in block["text"]
