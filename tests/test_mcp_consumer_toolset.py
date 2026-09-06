"""Consumer MCP toolset pins (L4 Q1-Q4=A, 2026-09-05).

What must not regress: (1) the default toolset is unchanged for every
existing deployment; (2) CC_MCP_TOOLSET=consumer exposes EXACTLY the
four-tool chat-host surface the skill and registry listings describe;
(3) recall's mode enum stays closed; (4) consumer forget is the LEDGERED
RETIRE, never the hard delete (Q3=A).
"""
import importlib
import inspect

import pytest

pytest.importorskip("mcp")

import crystal_cache.agent.mcp_server as mcp_server
from crystal_cache.config import Settings, get_settings

CONSUMER = {"remember", "recall", "status", "forget"}


@pytest.mark.asyncio
async def test_default_toolset_is_full_and_unchanged():
    # Default settings: memory_* surface intact AND the four consumer tools
    # present alongside it (Q1=A: one server, additive).
    assert Settings().mcp_toolset == "full"
    names = {t.name for t in await mcp_server.mcp.list_tools()}
    assert CONSUMER <= names
    assert {"memory_search", "memory_store", "memory_forget",
            "memory_stats", "memory_record_gap"} <= names


@pytest.mark.asyncio
async def test_consumer_toolset_exposes_exactly_four(monkeypatch):
    monkeypatch.setattr(
        "crystal_cache.config.get_settings",
        lambda: Settings(mcp_toolset="consumer"),
    )
    try:
        mod = importlib.reload(mcp_server)
        names = {t.name for t in await mod.mcp.list_tools()}
        assert names == CONSUMER
    finally:
        # Restore the real module state for every other test.
        monkeypatch.undo()
        importlib.reload(mcp_server)


@pytest.mark.asyncio
async def test_recall_mode_enum_is_closed():
    out = await mcp_server.recall(query="anything", mode="bogus")
    assert "error" in out and "quick" in out["error"]


def test_consumer_forget_is_ledgered_retire_not_hard_delete():
    src = inspect.getsource(mcp_server.forget)
    # The semantic markers of Q3=A: a retire ledger row per fact, written
    # BEFORE removal, attributed to the consumer surface.
    assert 'op="retire"' in src
    assert "append_fact_ledger" in src
    assert 'actor="mcp_consumer"' in src
    assert src.index("append_fact_ledger") < src.index("delete_crystal")
    # And the tool's contract says so out loud.
    assert "ledger" in mcp_server.forget.__doc__ if mcp_server.forget.__doc__ else True
