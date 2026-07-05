"""WS C — MCP memory server: surface, registry promotion, and dispatch.

Three layers of coverage, all runnable under the existing in-memory
fixtures (no live server, no real LLM):

  1. surface       — the FastMCP server registers all 16 memory_* tools.
  2. promotion     — crystal_learn / knowledge_conflicts / knowledge_gaps
                     landed in the agent registry with the right contexts
                     (so the agent + cognition get them too).
  3. dispatch      — calling the memory_* wrappers directly, after setting
                     the auth contextvar + tool-state the way the ASGI
                     middleware + lifespan would, exercises the
                     bridge -> registry -> store path.

The store-backed tools (store / stats / list / export / import / forget /
conflicts / gaps) get strong deterministic assertions. The vector-backed
tools (search / recall / synthesize) are asserted at the plumbing level
(well-formed result, no crash) because the test encoder is a deterministic
stub, not a semantic model — retrieval QUALITY is covered by the live
connection smoke (scripts/mcp_smoke.py) against the real encoder.
"""
from __future__ import annotations

import pytest
import pytest_asyncio

from crystal_cache.agent import mcp_server as srv
from crystal_cache.agent.tool_registry import get_registry, import_all_tools
from crystal_cache.agent.tools.retrievers import set_tool_state


EXPECTED_TOOLS = {
    "memory_search", "memory_search_documents", "memory_outline",
    "memory_keys", "memory_synthesize", "memory_recall", "memory_store",
    "memory_forget", "memory_ingest", "memory_learn", "memory_stats",
    "memory_list", "memory_export", "memory_import",
    "memory_conflicts", "memory_gaps",
}


# ---------------------------------------------------------------------------
# 1. Surface
# ---------------------------------------------------------------------------

async def test_mcp_registers_all_memory_tools():
    names = {t.name for t in await srv.mcp.list_tools()}
    assert EXPECTED_TOOLS <= names, f"missing tools: {EXPECTED_TOOLS - names}"


# ---------------------------------------------------------------------------
# 2. Registry promotion (learn / conflicts / gaps now visible to the agent)
# ---------------------------------------------------------------------------

def test_curation_tools_promoted_into_registry():
    import_all_tools()
    reg = get_registry()

    learn = reg.get("crystal_learn")
    assert learn is not None
    assert "agent" in learn.contexts
    assert "cognition" not in learn.contexts  # write-side, agent-only

    for name in ("knowledge_conflicts", "knowledge_gaps"):
        tool = reg.get(name)
        assert tool is not None, f"{name} not registered"
        assert {"agent", "cognition"} <= tool.contexts  # read-side, shared


# ---------------------------------------------------------------------------
# 3. Dispatch — auth contextvar + tool-state, then call the wrappers
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def authed(customer, tool_state):
    """Set the MCP auth contextvar + tool-state the way the middleware and
    lifespan would, so the memory_* wrappers can be called directly."""
    set_tool_state(tool_state)
    token = srv._current_customer_id.set(customer.id)
    try:
        yield customer
    finally:
        srv._current_customer_id.reset(token)


async def test_store_returns_ids(authed):
    w = await srv.memory_store(key="Capital|France", value="Paris is the capital of France.")
    assert w["crystal_id"]
    assert w["fact_id"]
    assert w["pair_type"] == "question_answer"


async def test_search_and_recall_well_formed(authed):
    await srv.memory_store(key="Animal|Pangolin", value="A pangolin is a scaly mammal.")
    res = await srv.memory_search(query="pangolin", k=5)
    assert "fact_count" in res and isinstance(res["fact_count"], int)
    assert "injection_text" in res
    rec = await srv.memory_recall(query="pangolin", k=5)
    assert "count" in rec and isinstance(rec["count"], int)


async def test_stats_reflect_stored_data(authed):
    await srv.memory_store(key="Color|Sky", value="The sky is blue.")
    stats = await srv.memory_stats()
    assert stats["crystal_count"] >= 1
    assert stats["fact_count"] >= 1
    assert isinstance(stats["pair_type_distribution"], dict)


async def test_list_then_forget_crystal(authed):
    w = await srv.memory_store(key="Fruit|Apple", value="An apple is a fruit.")
    listing = await srv.memory_list()
    assert listing["total"] >= 1
    gone = await srv.memory_forget(crystal_id=w["crystal_id"])
    assert gone["deleted"] is True


async def test_forget_requires_exactly_one_id(authed):
    assert (await srv.memory_forget())["deleted"] is False
    assert (await srv.memory_forget(crystal_id="a", fact_id="b"))["deleted"] is False


async def test_export_import_roundtrip(authed):
    await srv.memory_store(key="Planet|Mars", value="Mars is the fourth planet.")
    dump = await srv.memory_export()
    assert dump["record_count"] >= 1
    assert dump["export_format"] == "jsonl"
    imp = await srv.memory_import(records=dump["data"], wipe=True)
    assert imp["records_processed"] >= 1
    assert imp["errors"] == 0


async def test_learn_success_caches(authed):
    r = await srv.memory_learn(prompt="2+2?", response="4", outcome="success")
    assert "crystals_written" in r
    assert "cached" in r


async def test_synthesize_does_not_crash(authed):
    # Deep-by-default; with the stub LLM the synthesis step degrades to
    # organized context (the router catches synth errors) — assert it
    # returns a well-formed result rather than raising.
    await srv.memory_store(key="Topic|Photosynthesis", value="Plants convert light to energy.")
    res = await srv.memory_synthesize(query="how does photosynthesis work")
    assert isinstance(res, dict)
    assert "injection_text" in res


async def test_conflicts_and_gaps_empty(authed):
    assert (await srv.memory_conflicts())["count"] == 0
    assert (await srv.memory_gaps())["count"] == 0


async def test_conflicts_surface_created_conflict(authed, store, customer):
    await store.create_knowledge_conflict(
        customer.id,
        fact_a_id="fa", fact_b_id="fb",
        claim_a="Rate is $120/hr.", claim_b="Rate is $95/hr.",
        pair_key="pk_test_rate", subject="Contract|Rate",
    )
    c = await srv.memory_conflicts()
    assert c["count"] == 1
    assert c["conflicts"][0]["subject"] == "Contract|Rate"
    assert c["conflicts"][0]["claim_a"] == "Rate is $120/hr."


async def test_tenant_isolation_on_reads(authed, store):
    # A second customer's conflict must not appear for the authed customer.
    other = await store.create_customer(
        provider="anthropic",
        model_id="claude-sonnet-4-5-20250929",
        api_key_ref="sk-test-other-mcp",
    )
    await store.create_knowledge_conflict(
        other.id, fact_a_id="x", fact_b_id="y",
        claim_a="A", claim_b="B", pair_key="pk_other", subject="Other|Thing",
    )
    assert (await srv.memory_conflicts())["count"] == 0
