"""Task-scoped keys (Phase 3 G3, 2026-07-03, ratified).

The disposable box's ONLY credential: tenant-bound, budgeted from the
existing ledger (session_id = task_id), expiring, revocable, hash at
rest. Restriction is by ROUTING: only the chat proxy's dependencies
accept task keys — require_customer and resolve_principal never see
them, so the SDK/document/control surface rejects them naturally.

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from crystal_cache.ingress.auth import resolve_principal_or_task


# --- mint / resolve lifecycle -------------------------------------------------

async def test_mint_and_resolve_roundtrip(store, customer):
    raw, rec = await store.mint_task_key(
        customer.id, "task-1", budget_micro_usd=50_000, ttl_seconds=3600,
    )
    assert raw.startswith("ck_task_")
    assert rec.task_id == "task-1" and rec.customer_id == customer.id

    live = await store.resolve_task_key(raw)
    assert live is not None
    assert live.budget_micro_usd == 50_000


async def test_key_is_hashed_at_rest(store, customer):
    raw, _ = await store.mint_task_key(
        customer.id, "task-hash", budget_micro_usd=1, ttl_seconds=60,
    )
    from sqlalchemy import select
    from crystal_cache.infrastructure.schema import TaskKeyRow

    async with store.session() as session:
        row = await session.get(TaskKeyRow, "task-hash")
        assert row.key_hash != raw
        assert raw not in row.key_hash  # no plaintext anywhere in the row


async def test_unknown_revoked_expired_all_resolve_none(store, customer):
    # Unknown
    assert await store.resolve_task_key("ck_task_nope") is None
    # Revoked
    raw_r, _ = await store.mint_task_key(
        customer.id, "task-rev", budget_micro_usd=1, ttl_seconds=3600,
    )
    assert await store.revoke_task_key("task-rev") is True
    assert await store.resolve_task_key(raw_r) is None
    # Expired (ttl 0 = already past)
    raw_e, _ = await store.mint_task_key(
        customer.id, "task-exp", budget_micro_usd=1, ttl_seconds=0,
    )
    assert await store.resolve_task_key(raw_e) is None


async def test_revoke_is_idempotent_and_unknown_false(store, customer):
    await store.mint_task_key(
        customer.id, "task-i", budget_micro_usd=1, ttl_seconds=60,
    )
    assert await store.revoke_task_key("task-i") is True
    assert await store.revoke_task_key("task-i") is True   # second call fine
    assert await store.revoke_task_key("never-was") is False


# --- budget: the ledger IS the meter -------------------------------------------

async def test_task_spend_sums_only_this_task(store, customer):
    await store.record_llm_call(
        customer_id=customer.id, model="claude-haiku-4-5-20251001",
        input_tokens=1000, output_tokens=1000, session_id="task-a",
        origin="disposable_task",
    )
    await store.record_llm_call(
        customer_id=customer.id, model="claude-haiku-4-5-20251001",
        input_tokens=1000, output_tokens=1000, session_id="task-b",
        origin="disposable_task",
    )
    a = await store.task_spend_micro_usd("task-a")
    b = await store.task_spend_micro_usd("task-b")
    none = await store.task_spend_micro_usd("task-none")
    assert a > 0 and a == b
    assert none == 0


# --- the auth door --------------------------------------------------------------

def _req(token: str):
    return SimpleNamespace(
        headers={"Authorization": f"Bearer {token}"},
        state=SimpleNamespace(),
    )


async def test_door_accepts_live_key_and_stashes_task_id(store, customer):
    raw, _ = await store.mint_task_key(
        customer.id, "task-door", budget_micro_usd=1_000_000, ttl_seconds=600,
    )
    req = _req(raw)
    team, operator = await resolve_principal_or_task(req, store)
    assert team.id == customer.id
    assert operator is not None            # P1: always an acting operator
    assert req.state.task_key_task_id == "task-door"


async def test_door_rejects_dead_key(store, customer):
    raw, _ = await store.mint_task_key(
        customer.id, "task-dead", budget_micro_usd=1, ttl_seconds=600,
    )
    await store.revoke_task_key("task-dead")
    with pytest.raises(HTTPException) as e:
        await resolve_principal_or_task(_req(raw), store)
    assert e.value.status_code == 401


async def test_door_429_when_budget_exhausted(store, customer):
    raw, _ = await store.mint_task_key(
        customer.id, "task-broke", budget_micro_usd=1, ttl_seconds=600,
    )
    # One real ledger row under this task blows the 1-micro-usd budget.
    await store.record_llm_call(
        customer_id=customer.id, model="claude-haiku-4-5-20251001",
        input_tokens=1000, output_tokens=1000, session_id="task-broke",
        origin="disposable_task",
    )
    with pytest.raises(HTTPException) as e:
        await resolve_principal_or_task(_req(raw), store)
    assert e.value.status_code == 429


# --- restriction by routing ------------------------------------------------------

async def test_sdk_surface_rejects_task_keys(store, customer):
    """require_customer never resolves task keys: the whole SDK surface is
    closed to them without any per-endpoint flagging."""
    raw, _ = await store.mint_task_key(
        customer.id, "task-sdk", budget_micro_usd=1_000, ttl_seconds=600,
    )
    assert await store.get_customer_by_api_key(raw) is None  # not a customer

    from crystal_cache.ingress.auth import require_customer

    with pytest.raises(HTTPException) as e:
        await require_customer(_req(raw), store)
    assert e.value.status_code == 401
