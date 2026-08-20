"""Task-scoped keys — the AUTH-SIDE validation (Phase 3 G3, 2026-07-03).

Restriction is by ROUTING: only the chat proxy's dependencies accept
task keys — require_customer and resolve_principal never see them, so
the SDK/document/control surface rejects them naturally.

The minting side (mint_task_key / revoke_task_key) was retired with the
hosted execution plane (2026-08-19). These tests construct task-key rows
directly against the schema, because resolve_task_key and the budget
door remain live production code that must keep validating any row that
exists at rest.

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from crystal_cache.infrastructure.credentials import hash_api_key
from crystal_cache.infrastructure.schema import TaskKeyRow
from crystal_cache.ingress.auth import resolve_principal_or_task


async def _insert_task_key(
    store,
    customer_id: str,
    task_id: str,
    *,
    budget_micro_usd: int,
    ttl_seconds: int,
    revoked: bool = False,
) -> str:
    """Write a task-key row directly (hash at rest), returning the raw key.

    Mirrors what the retired minter persisted, so resolve_task_key sees
    exactly the at-rest shape production auth must validate.
    """
    raw = "ck_task_" + secrets.token_hex(32)
    now = datetime.now(timezone.utc)
    row = TaskKeyRow(
        task_id=task_id,
        key_hash=hash_api_key(raw),
        customer_id=customer_id,
        budget_micro_usd=int(budget_micro_usd),
        expires_at=now + timedelta(seconds=int(ttl_seconds)),
        revoked_at=(now if revoked else None),
        created_at=now,
    )
    async with store.session() as session:
        session.add(row)
    return raw


# --- resolve: the live-record gate ---------------------------------------------

async def test_resolve_returns_live_record(store, customer):
    raw = await _insert_task_key(
        store, customer.id, "task-1", budget_micro_usd=50_000, ttl_seconds=3600,
    )
    live = await store.resolve_task_key(raw)
    assert live is not None
    assert live.task_id == "task-1" and live.customer_id == customer.id
    assert live.budget_micro_usd == 50_000


async def test_unknown_revoked_expired_all_resolve_none(store, customer):
    # Unknown
    assert await store.resolve_task_key("ck_task_nope") is None
    # Revoked
    raw_r = await _insert_task_key(
        store, customer.id, "task-rev", budget_micro_usd=1, ttl_seconds=3600,
        revoked=True,
    )
    assert await store.resolve_task_key(raw_r) is None
    # Expired (ttl 0 = already past)
    raw_e = await _insert_task_key(
        store, customer.id, "task-exp", budget_micro_usd=1, ttl_seconds=0,
    )
    assert await store.resolve_task_key(raw_e) is None


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
    raw = await _insert_task_key(
        store, customer.id, "task-door", budget_micro_usd=1_000_000,
        ttl_seconds=600,
    )
    req = _req(raw)
    team, operator = await resolve_principal_or_task(req, store)
    assert team.id == customer.id
    assert operator is not None            # P1: always an acting operator
    assert req.state.task_key_task_id == "task-door"


async def test_door_rejects_dead_key(store, customer):
    raw = await _insert_task_key(
        store, customer.id, "task-dead", budget_micro_usd=1, ttl_seconds=600,
        revoked=True,
    )
    with pytest.raises(HTTPException) as e:
        await resolve_principal_or_task(_req(raw), store)
    assert e.value.status_code == 401


async def test_door_429_when_budget_exhausted(store, customer):
    raw = await _insert_task_key(
        store, customer.id, "task-broke", budget_micro_usd=1, ttl_seconds=600,
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
    raw = await _insert_task_key(
        store, customer.id, "task-sdk", budget_micro_usd=1_000, ttl_seconds=600,
    )
    assert await store.get_customer_by_api_key(raw) is None  # not a customer

    from crystal_cache.ingress.auth import require_customer

    with pytest.raises(HTTPException) as e:
        await require_customer(_req(raw), store)
    assert e.value.status_code == 401
