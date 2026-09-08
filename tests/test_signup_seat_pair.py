"""L2-S1 pins (Q3=A, 2026-09-06): signup creates the linked owner SEAT.

The AS (Q2=A) maps OAuth tokens -> an operator; writes stamp owners. Both
depend on the pair existing from signup, so the pair is pinned at the
store level (round-trip of the join columns) and at the route level
(signup creates exactly one admin operator, idempotently).
"""
from typing import Any, Optional

import pytest

from crystal_cache.config import Settings
from crystal_cache.endpoints import me as me_mod
from crystal_cache.ingress import auth as auth_mod


@pytest.mark.asyncio
async def test_create_operator_round_trips_join_columns(store, customer):
    op, raw = await store.create_operator(
        team_id=customer.id,
        display_name="anthony",
        role="admin",
        email="anthony@test.dev",
        user_id="uid_join_1",
    )
    assert raw and op.email == "anthony@test.dev"
    got = await store.get_operator_by_id(op.id)
    assert got is not None
    assert got.user_id == "uid_join_1"
    assert got.email == "anthony@test.dev"
    assert got.role == "admin"


class _StubRequest:
    def __init__(self, token: str, body: Optional[dict] = None):
        self.headers = {"authorization": f"Bearer {token}"}
        self._body = body

    async def json(self) -> Any:
        if self._body is None:
            raise ValueError("no body")
        return self._body


@pytest.mark.asyncio
async def test_signup_creates_linked_owner_seat_idempotently(store, monkeypatch):
    fake_jwt = "eyJx.eyJy.sig"  # passes the shape check; verification stubbed
    # Patch where the route LOOKS UP its settings — me.py binds the name at
    # import (same lesson as the disabled-tools pin, 2026-09-03).
    monkeypatch.setattr(
        me_mod, "get_settings",
        lambda: Settings(firebase_project_id="test-proj"),
    )
    monkeypatch.setattr(
        auth_mod, "_verify_firebase_jwt",
        lambda tok, proj: {"sub": "uid_seat_1", "email": "seat1@test.dev"},
    )

    r1 = await me_mod.signup(_StubRequest(fake_jwt), store)
    assert r1["created"] is True
    assert r1["api_key"]  # the one-time reveal
    # The returned seat IS the team's default admin (deterministic id —
    # ensure_default_admin is an idempotent get, so this comparison is
    # exact, not a prefix guess).
    default = await store.ensure_default_admin(r1["customer_id"])
    assert r1["operator_id"] == default.id

    # The seat is the team's DEFAULT ADMIN (P1 identity chain), linked in
    # place — not a second operator (the first version of this pin caught
    # signup minting a duplicate, 2026-09-07).
    op = await store.get_operator_by_id(r1["operator_id"])
    assert op is not None
    assert op.team_id == r1["customer_id"]
    assert op.user_id == "uid_seat_1"
    assert op.email == "seat1@test.dev"
    assert op.role == "admin"

    # Idempotent second signup: same identity back, NO second seat — the
    # tenant has exactly ONE operator: the linked default admin.
    r2 = await me_mod.signup(_StubRequest(fake_jwt), store)
    assert r2["created"] is False and r2["api_key"] is None
    ops = await store.list_operators_for_team(r1["customer_id"])
    assert len(ops) == 1
    assert ops[0].id == r1["operator_id"]
    assert ops[0].user_id == "uid_seat_1"
