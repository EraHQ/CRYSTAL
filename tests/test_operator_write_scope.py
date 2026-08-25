"""Phase 1.4 Gate 5 — write-side scope + owner stamps (Q3=C).

crystal_write reaches full parity with /v1/store: the deployment's
`default_ingest_scope` decides mode, the acting operator (request
context, Q1=A) is stamped as owner, an explicit per-request `scope`
overrides the default, unknown scopes are refused before anything is
written, viewers are refused (defense-in-depth under the doors), and
the no-operator system lane keeps the exact pre-gate stamps (team
0o640, unowned — sdk_store's identical fallback: personal is undefined
without an owner). The MCP memory_store wrapper forwards `scope`
verbatim to the registry impl.

Settings note: `get_settings()` is an lru_cache singleton returning the
same object as `crystal_cache.config.settings`, so monkeypatching an
attribute on the imported `settings` is visible through the
`get_settings()` call inside crystal_write.

asyncio_mode=auto.
"""
from __future__ import annotations

from types import SimpleNamespace

from crystal_cache.agent.principal import (
    reset_current_operator,
    set_current_operator,
)
from crystal_cache.agent.tools.memory import crystal_write
from crystal_cache.agent.tools.retrievers import set_tool_state
from crystal_cache.config import settings


def _state(store, encoder, vector_store) -> dict:
    return {
        "store": store,
        "encoder": encoder,
        "vector_store": vector_store,
        "vector_index": None,
    }


async def _write_as(operator, customer_id, store, encoder, vector_store, **kw):
    set_tool_state(_state(store, encoder, vector_store))
    if operator is None:
        return await crystal_write(customer_id, **kw)
    token = set_current_operator(operator)
    try:
        return await crystal_write(customer_id, **kw)
    finally:
        reset_current_operator(token)


async def test_default_personal_scope_stamps_owner(
    store, customer, semantic_encoder_stub, vector_store, monkeypatch,
):
    monkeypatch.setattr(settings, "default_ingest_scope", "personal")
    op, _ = await store.create_operator(team_id=customer.id, display_name="W")

    out = await _write_as(
        op, customer.id, store, semantic_encoder_stub, vector_store,
        key="Scoped|Alpha", value="alpha value",
    )
    assert "error" not in out
    assert out["scope"] == "personal"

    crystal = await store.get_crystal(out["crystal_id"])
    assert crystal.mode == 0o600
    assert crystal.owner_operator_id == op.id
    assert crystal.group_team_id == op.team_id


async def test_explicit_team_scope_overrides_personal_default(
    store, customer, semantic_encoder_stub, vector_store, monkeypatch,
):
    monkeypatch.setattr(settings, "default_ingest_scope", "personal")
    op, _ = await store.create_operator(team_id=customer.id, display_name="W")

    out = await _write_as(
        op, customer.id, store, semantic_encoder_stub, vector_store,
        key="Scoped|Beta", value="beta value", scope="team",
    )
    assert "error" not in out
    assert out["scope"] == "team"

    crystal = await store.get_crystal(out["crystal_id"])
    assert crystal.mode == 0o640
    assert crystal.owner_operator_id == op.id  # owned even when team-shared


async def test_unknown_scope_refused_nothing_written(
    store, customer, semantic_encoder_stub, vector_store,
):
    op, _ = await store.create_operator(team_id=customer.id, display_name="W")
    before = len(await store.list_all_facts_for_customer(customer.id))

    out = await _write_as(
        op, customer.id, store, semantic_encoder_stub, vector_store,
        key="Scoped|Bad", value="nope", scope="everyone",
    )
    assert "error" in out and "everyone" in out["error"]
    after = len(await store.list_all_facts_for_customer(customer.id))
    assert after == before


async def test_viewer_refused_nothing_written(
    store, customer, semantic_encoder_stub, vector_store,
):
    viewer = SimpleNamespace(
        id="op_view", role="viewer", team_id=customer.id, status="active",
    )
    before = len(await store.list_all_facts_for_customer(customer.id))

    out = await _write_as(
        viewer, customer.id, store, semantic_encoder_stub, vector_store,
        key="Scoped|View", value="denied",
    )
    assert out.get("code") == "viewer_forbidden"
    after = len(await store.list_all_facts_for_customer(customer.id))
    assert after == before


async def test_no_operator_context_keeps_pre_gate_stamps(
    store, customer, semantic_encoder_stub, vector_store, monkeypatch,
):
    """System lane (CLI local runtime, direct callers): the personal
    default CANNOT apply without an owner to stamp — behavior is the
    pre-gate team write exactly (Q2=A philosophy on the write side)."""
    monkeypatch.setattr(settings, "default_ingest_scope", "personal")

    out = await _write_as(
        None, customer.id, store, semantic_encoder_stub, vector_store,
        key="Scoped|Sys", value="system value",
    )
    assert "error" not in out
    assert out["scope"] == "team"

    crystal = await store.get_crystal(out["crystal_id"])
    assert crystal.mode == 0o640
    assert crystal.owner_operator_id is None
    # Spawn-fresh stamps group_team_id RAW (the `or customer_id` in
    # add_pair_for_customer belongs to the may_join boundary check only,
    # normalizing the INCOMING pair for comparison — it never touches
    # what's stored). None in → None stored: byte-identical to the
    # pre-gate defaulted write. Unstamped rows ride the legacy NULL-stamp
    # readability contract (test_permissions).
    assert crystal.group_team_id is None


async def test_mcp_memory_store_forwards_scope(monkeypatch):
    """The MCP wrapper hands `scope` to the registry impl verbatim."""
    from crystal_cache.agent import mcp_server

    captured: dict = {}

    async def _fake_dispatch(registry_name, **kwargs):
        captured["tool"] = registry_name
        captured["kwargs"] = kwargs
        return {"ok": True}

    monkeypatch.setattr(mcp_server, "_dispatch", _fake_dispatch)
    out = await mcp_server.memory_store(
        key="K", value="V", scope="personal",
    )
    assert out == {"ok": True}
    assert captured["tool"] == "crystal_write"
    assert captured["kwargs"]["scope"] == "personal"
