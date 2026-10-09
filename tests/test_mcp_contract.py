"""RC-15 (2026-10-07): the MCP contract matches the tools.

Asserted at the PROTOCOL level through an in-memory client session
(mcp 1.30: a raised ToolError reaches the client as isError=true; a
returned dict never does, verified in a sandbox):
- a refusal (an over-limit write) arrives as isError=true with a stable
  code in the text, naming memory_ingest, not an agent-only tool;
- recall results carry a `tier` from the documented set;
- stats carries the plan's fact cap and what is left of it;
- every refusal dict any tool can return has a code (the boundary needs
  one to raise).
"""
import inspect
import re

import pytest

from crystal_cache.agent import mcp_server
from crystal_cache.agent.tools.memory import CRYSTAL_WRITE_MAX_VALUE_CHARS

TIERS = {"whitelist", "neutral", "quarantine", "blacklist"}


@pytest.fixture
def mcp_state(store, customer, semantic_encoder_stub, vector_store, fact_vector_store, monkeypatch):
    from crystal_cache.agent.tools.retrievers import set_tool_state

    state = {
        "store": store, "encoder": semantic_encoder_stub, "vector_store": vector_store,
        "fact_vector_store": fact_vector_store, "vector_index": None,
    }
    # Both seams the tools use: the module-level getter and the registry.
    monkeypatch.setattr(mcp_server, "_get_state", lambda: state)
    set_tool_state(state)
    token = mcp_server._current_customer_id.set(customer.id)
    yield state
    mcp_server._current_customer_id.reset(token)


@pytest.mark.asyncio
async def test_over_limit_write_is_a_protocol_error_naming_memory_ingest(mcp_state):
    from mcp.shared.memory import create_connected_server_and_client_session

    async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as client:
        r = await client.call_tool(
            "memory_store", {"key": "k", "value": "x" * (CRYSTAL_WRITE_MAX_VALUE_CHARS + 1)},
        )
    assert r.isError is True
    text = r.content[0].text
    assert "[content_too_long]" in text
    assert "memory_ingest" in text  # the tool that exists on THIS surface


@pytest.mark.asyncio
async def test_recall_results_carry_a_documented_tier(mcp_state, store, customer, semantic_encoder_stub, vector_store):
    await store.add_pair_for_customer(
        customer_id=customer.id, prompt_text="Billing|terms", answer_text="Net 30",
        encoder=semantic_encoder_stub, vector_store=vector_store,
    )
    from mcp.shared.memory import create_connected_server_and_client_session

    async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as client:
        r = await client.call_tool("recall", {"query": "billing terms", "k": 5})
    assert r.isError is False
    import json

    out = json.loads(r.content[0].text)
    crystals = out.get("crystals") or out.get("results") or []
    assert crystals, out
    assert all(c.get("tier") in TIERS for c in crystals), crystals


@pytest.mark.asyncio
async def test_stats_carries_the_plan_caps(mcp_state, store, customer):
    await store.set_customer_subscription(customer.id, "free", None)
    from mcp.shared.memory import create_connected_server_and_client_session

    async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as client:
        r = await client.call_tool("status", {})
    import json

    out = json.loads(r.content[0].text)
    assert out["plan"]["tier"] == "free"
    assert out["plan"]["fact_cap"] == 2500
    assert out["plan"]["facts_remaining"] == 2500 - out["plan"]["facts_billable"]
    # Q4=A: no customer-facing dollars anywhere in the tool output.
    assert "micro_usd" not in r.content[0].text and "usd" not in r.content[0].text.lower()


def _codeless_returns(src: str, label: str) -> list[str]:
    import ast
    import textwrap

    offenders = []
    for node in ast.walk(ast.parse(textwrap.dedent(src))):
        if not (isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)):
            continue
        keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
        if "error" in keys and "code" not in keys:
            offenders.append(f"{label}:{node.lineno}")
    return offenders


def test_every_refusal_dict_in_the_tools_has_a_code():
    """The boundary raises only when both error and code are present; a
    refusal without a code would still slip through as isError=false.
    Walks the AST: every `return {...}` literal in mcp_server with an
    "error" key must also have a "code" key.

    Lockdown PR-4 (class 6, 2026-10-09): the walk also covers every
    registry tool mcp_server reaches through `_dispatch` (the names are
    read from mcp_server's own source, so a new dispatch is walked
    automatically) — key_scan, crystal_write, record_gap and
    crystal_learn each shipped codeless refusals the old walk never saw."""
    import re

    src = inspect.getsource(mcp_server)
    offenders = _codeless_returns(src, "mcp_server.py")

    names = sorted(set(re.findall(r'_dispatch\(\s*"([a-z_]+)"', src)))
    assert len(names) >= 11, names  # the walk must see the dispatched set
    registry = mcp_server.get_registry()
    for name in names:
        tool = registry.get(name)
        assert tool is not None, name
        impl = inspect.unwrap(tool.impl)
        offenders += _codeless_returns(inspect.getsource(impl), f"{name}@{impl.__module__}")
    assert offenders == [], offenders


@pytest.mark.asyncio
async def test_descriptions_no_longer_lie(mcp_state):
    """Read the descriptions the CLIENT receives, not the source (a
    description is assembled from several string literals)."""
    from mcp.shared.memory import create_connected_server_and_client_session

    async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as client:
        tools = {t.name: t.description or "" for t in (await client.list_tools()).tools}
    assert "read 'verified'" not in tools["recall"]
    assert "whitelist, neutral, quarantine, blacklist" in tools["recall"]
    assert "reversible" not in tools["forget"] and "no restore" in tools["forget"]
    assert "at most 800 characters" in tools["remember"] and "memory_ingest" in tools["remember"]
    assert "POST /v1/documents" not in " ".join(tools.values())
