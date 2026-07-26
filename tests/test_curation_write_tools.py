"""0g (2026-07-25): the curation WRITE tools — resolve_conflict + record_gap.

Motivating incidents, all same-day, all the same shape: the agent could READ
what the bank contradicts and lacks and could not act on either, so it routed
around the missing drawer. It wrote a stronger-matching crystal for a
user-confirmed launch date (leaving the stale fact live and the conflict open
behind a better vector), and it wrote "GAP: X not documented" facts when asked
questions the bank could not answer. These pin the drawers, and the gate on
the first one: a conflict is settled only on an explicit user confirmation,
quoted.
"""

from __future__ import annotations

import pytest

from crystal_cache.agent.tool_registry import get_registry, import_all_tools
from crystal_cache.agent.tools.curation import record_gap, resolve_conflict
from crystal_cache.agent.tools.retrievers import set_tool_state


CONFIRM = "yes, September 22 is the right one"


async def _seed_facts(store, customer, tool_state):
    """Two real facts, so retirement is observable on the row."""
    _, fact_a = await store.add_pair_for_customer(
        customer_id=customer.id,
        prompt_text="When does Wren & Sparrow launch?",
        answer_text="The launch is September 15",
        pair_type="question_answer",
        encoder=tool_state["encoder"],
        vector_store=tool_state["vector_store"],
        vector_index=tool_state["vector_index"],
    )
    _, fact_b = await store.add_pair_for_customer(
        customer_id=customer.id,
        prompt_text="When does Wren & Sparrow launch?",
        answer_text="The launch is September 22",
        pair_type="question_answer",
        encoder=tool_state["encoder"],
        vector_store=tool_state["vector_store"],
        vector_index=tool_state["vector_index"],
    )
    return fact_a, fact_b


async def _seed_conflict(store, customer_id, fact_a_id, fact_b_id,
                         pair_key="pk_launch"):
    return await store.create_knowledge_conflict(
        customer_id,
        fact_a_id=fact_a_id,
        fact_b_id=fact_b_id,
        claim_a="The launch is September 15",
        claim_b="The launch is September 22",
        pair_key=pair_key,
        subject="launch date",
    )


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------

def test_write_tools_registered_agent_only():
    import_all_tools()
    reg = get_registry()
    for name in ("resolve_conflict", "record_gap"):
        tool = reg.get(name)
        assert tool is not None, f"{name} not registered"
        # Write-side: agent-only. Cognition writes through its commit gate.
        assert tool.contexts == frozenset({"agent"})
        assert "customer_id" not in tool.parameters_schema["properties"]
    assert set(reg.get("resolve_conflict").parameters_schema["required"]) == {
        "conflict_id", "resolution", "user_confirmation",
    }
    assert set(reg.get("record_gap").parameters_schema["required"]) == {
        "question", "disposition",
    }


# ---------------------------------------------------------------------------
# resolve_conflict — the confirmation gate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_blank_confirmation_refused_and_nothing_written(
    store, customer, tool_state,
):
    set_tool_state(tool_state)
    fact_a, fact_b = await _seed_facts(store, customer, tool_state)
    c = await _seed_conflict(store, customer.id, fact_a.id, fact_b.id)

    out = await resolve_conflict(customer.id, c.id, "superseded", "   ",
                                 loser="a")
    assert out["resolved"] is False
    assert "user_confirmation" in out["error"]

    # The conflict is untouched and the fact is still live.
    still = await store.list_knowledge_conflicts(customer.id, status="open")
    assert [x.id for x in still] == [c.id]
    assert (await store.get_fact(fact_a.id)).grating_strength == 1.0


@pytest.mark.asyncio
async def test_blacklist_and_unknown_resolutions_refused(
    store, customer, tool_state,
):
    set_tool_state(tool_state)
    fact_a, fact_b = await _seed_facts(store, customer, tool_state)
    c = await _seed_conflict(store, customer.id, fact_a.id, fact_b.id)

    # The STORE accepts blacklisted; the agent surface does not — that one
    # also suppresses the claim from ever being re-learned, and nothing on
    # the agent side undoes it, so it stays an operator click.
    out = await resolve_conflict(customer.id, c.id, "blacklisted", CONFIRM,
                                 loser="a")
    assert out["resolved"] is False
    assert "operator" in out["error"]

    out = await resolve_conflict(customer.id, c.id, "retire", CONFIRM)
    assert out["resolved"] is False
    for valid in ("superseded", "qualified", "dismissed"):
        assert valid in out["error"]

    assert (await store.get_fact(fact_a.id)).grating_strength == 1.0
    assert len(await store.list_knowledge_conflicts(customer.id,
                                                    status="open")) == 1


@pytest.mark.asyncio
async def test_superseded_requires_loser(store, customer, tool_state):
    set_tool_state(tool_state)
    fact_a, fact_b = await _seed_facts(store, customer, tool_state)
    c = await _seed_conflict(store, customer.id, fact_a.id, fact_b.id)

    out = await resolve_conflict(customer.id, c.id, "superseded", CONFIRM)
    assert out["resolved"] is False
    assert "loser" in out["error"]


@pytest.mark.asyncio
async def test_foreign_conflict_reads_as_not_found(store, customer,
                                                   tool_state):
    """Tenancy: another tenant's conflict is indistinguishable from a missing
    one — never an existence oracle."""
    set_tool_state(tool_state)
    other = await _seed_conflict(store, "cus_someone_else", "f1", "f2")

    out = await resolve_conflict(customer.id, other.id, "dismissed", CONFIRM)
    assert out["resolved"] is False
    assert "not found" in out["error"]

    survivors = await store.list_knowledge_conflicts("cus_someone_else",
                                                     status="open")
    assert [x.id for x in survivors] == [other.id]


# ---------------------------------------------------------------------------
# resolve_conflict — the arc (CONF-R warning stops travelling)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolution_retires_loser_and_clears_retrieval_warning(
    store, customer, tool_state,
):
    """The demo path end to end: the user says Sept 22 is right, the agent
    settles it with the tool, the stale fact is deactivated, and the CONTESTED
    warning stops riding retrieval. Peer of
    test_conflict_retrieval_signal.py::test_resolved_conflicts_stop_travelling,
    which pins the same arc at the store boundary."""
    set_tool_state(tool_state)
    fact_a, fact_b = await _seed_facts(store, customer, tool_state)
    c = await _seed_conflict(store, customer.id, fact_a.id, fact_b.id)

    # Before: the conflict travels with either fact.
    assert await store.open_conflicts_for_facts(customer.id, [fact_a.id])

    out = await resolve_conflict(customer.id, c.id, "superseded", CONFIRM,
                                 loser="a")
    assert out["resolved"] is True
    assert out["status"] == "resolved"
    assert out["resolution"] == "superseded"
    assert out["retired_claim"] == "The launch is September 15"

    # The loser is deactivated from retrieval; the winner is untouched.
    assert (await store.get_fact(fact_a.id)).grating_strength == 0.0
    assert (await store.get_fact(fact_b.id)).grating_strength == 1.0

    # And the warning is gone from BOTH sides — no shadowing, no residue.
    assert await store.open_conflicts_for_facts(
        customer.id, [fact_a.id, fact_b.id]) == {}
    assert await store.list_knowledge_conflicts(customer.id,
                                                status="open") == []


@pytest.mark.asyncio
async def test_qualified_closes_conflict_keeping_both_facts(
    store, customer, tool_state,
):
    set_tool_state(tool_state)
    fact_a, fact_b = await _seed_facts(store, customer, tool_state)
    c = await _seed_conflict(store, customer.id, fact_a.id, fact_b.id)

    out = await resolve_conflict(
        customer.id, c.id, "qualified",
        "both are right, one is the US date", loser="a",
    )
    assert out["resolved"] is True
    assert out["retired_claim"] is None
    # loser is ignored for qualified — BOTH claims stay live.
    assert (await store.get_fact(fact_a.id)).grating_strength == 1.0
    assert (await store.get_fact(fact_b.id)).grating_strength == 1.0
    assert await store.open_conflicts_for_facts(customer.id, [fact_a.id]) == {}


# ---------------------------------------------------------------------------
# record_gap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_record_gap_writes_agent_observed_row(store, customer,
                                                    tool_state):
    """The score-hit / answer-insufficient case: retrieval matched adjacent
    content, so the automatic detector recorded nothing."""
    set_tool_state(tool_state)
    out = await record_gap(
        customer.id,
        question="What is Meridian's return policy?",
        disposition="researchable",
        context="the pricing sheet came back but has no returns section",
        subject="Meridian returns",
        priority="high",
    )
    assert out["recorded"] is True
    assert out["status"] == "open"
    assert out["disposition"] == "researchable"

    gaps = await store.list_knowledge_gaps(customer.id, status="open")
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.source == "agent_observed"
    assert gap.disposition == "researchable"
    assert gap.priority == "high"
    assert gap.subject == "Meridian returns"
    assert gap.triggering_query == "What is Meridian's return policy?"
    assert "no returns section" in gap.missing


@pytest.mark.asyncio
async def test_record_gap_rejects_invalid_values_and_writes_nothing(
    store, customer, tool_state,
):
    set_tool_state(tool_state)

    empty = await record_gap(customer.id, "   ", "researchable")
    assert empty["recorded"] is False

    bad_disp = await record_gap(customer.id, "What is X?", "sometime")
    assert bad_disp["recorded"] is False
    for valid in ("researchable", "workable", "needs_document"):
        assert valid in bad_disp["error"]

    bad_prio = await record_gap(customer.id, "What is X?", "workable",
                                priority="urgent")
    assert bad_prio["recorded"] is False
    assert "low, medium, high" in bad_prio["error"]

    assert await store.list_knowledge_gaps(customer.id) == []


@pytest.mark.asyncio
async def test_recorded_gap_is_not_promoted_to_a_task(store, customer,
                                                      tool_state):
    """A gap is a request, not a task. Promotion to research stays a human
    click (POST /v1/gaps/{id}/research) — the tool must never enqueue."""
    set_tool_state(tool_state)
    await record_gap(customer.id, "What is Meridian's lead time?",
                     "researchable")
    assert await store.list_cognition_tasks(customer.id) == []
