"""HTTP-layer tests for the operator endpoints (Foundation F1).

Follows the suite's endpoint-test convention (see test_phase9c): call the
endpoint functions directly with their FastAPI dependencies pre-resolved
(customer / operator / store passed in) rather than spinning up a
TestClient, and use a minimal FakeRequest for the header-reading auth
dependency. asyncio_mode=auto (pyproject) means async tests need no marker.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from crystal_cache.endpoints.operators import (
    create_operator,
    get_me,
    get_operator,
    list_operators,
    set_operator_role,
    set_operator_status,
)
from crystal_cache.ingress.auth import require_operator, require_role
from crystal_cache.ingress.schema import (
    CreateOperatorRequest,
    SetOperatorRoleRequest,
    SetOperatorStatusRequest,
)


class _FakeRequest:
    """Minimal Request stand-in carrying just headers (all require_* reads)."""

    def __init__(self, authorization: str | None = None) -> None:
        self.headers = {"authorization": authorization} if authorization else {}


# ---------------------------------------------------------------------------
# Team-key management endpoints
# ---------------------------------------------------------------------------

async def test_create_operator_returns_key_once_and_lists(store, customer):
    resp = await create_operator(
        CreateOperatorRequest(display_name="Ada", role="admin"),
        principal=(customer, None),
        store=store,
    )
    assert resp.team_id == customer.id
    assert resp.display_name == "Ada"
    assert resp.role == "admin"
    assert resp.status == "active"
    assert resp.api_key.startswith("cc_sk_")

    listed = await list_operators(customer=customer, store=store)
    # P1 (2026-07-02): every team is born with its Default Admin, so the
    # roster is Ada + the default admin.
    assert listed.total == 2
    by_name = {o.display_name: o for o in listed.operators}
    assert by_name["Ada"].id == resp.id
    assert by_name["Default Admin"].role == "admin"
    # Read responses never carry a key.
    assert not hasattr(listed.operators[0], "api_key")


async def test_get_operator_is_team_scoped(store, customer):
    resp = await create_operator(
        CreateOperatorRequest(display_name="Grace"),
        principal=(customer, None),
        store=store,
    )
    got = await get_operator(resp.id, customer=customer, store=store)
    assert got.id == resp.id

    # An operator on another team is invisible (404), not leaked.
    other = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="y",
    )
    with pytest.raises(HTTPException) as exc:
        await get_operator(resp.id, customer=other, store=store)
    assert exc.value.status_code == 404


async def test_set_role_and_status(store, customer):
    resp = await create_operator(
        CreateOperatorRequest(display_name="Lin"),
        principal=(customer, None),
        store=store,
    )
    after_role = await set_operator_role(
        resp.id, SetOperatorRoleRequest(role="viewer"),
        principal=(customer, None), store=store,
    )
    assert after_role.role == "viewer"

    after_status = await set_operator_status(
        resp.id, SetOperatorStatusRequest(status="suspended"),
        principal=(customer, None), store=store,
    )
    assert after_status.status == "suspended"


async def test_manage_rejects_cross_team_role_change(store, customer):
    resp = await create_operator(
        CreateOperatorRequest(display_name="Ada"),
        principal=(customer, None), store=store,
    )
    other = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="z",
    )
    with pytest.raises(HTTPException) as exc:
        await set_operator_role(
            resp.id, SetOperatorRoleRequest(role="admin"),
            principal=(other, None), store=store,
        )
    assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# Operator-key self endpoint + require_operator
# ---------------------------------------------------------------------------

async def test_require_operator_resolves_active_key_and_me(store, customer):
    operator, raw_key = await store.create_operator(
        team_id=customer.id, display_name="Ada", role="admin",
    )
    resolved = await require_operator(_FakeRequest(f"Bearer {raw_key}"), store)
    assert resolved.id == operator.id

    me = await get_me(operator=resolved)
    assert me.id == operator.id
    assert me.role == "admin"
    assert me.team_id == customer.id


async def test_require_operator_rejects_unknown_and_suspended(store, customer):
    # Unknown key -> 401.
    with pytest.raises(HTTPException) as exc401:
        await require_operator(_FakeRequest("Bearer cc_sk_nope"), store)
    assert exc401.value.status_code == 401

    # Suspended operator -> 403 (the row still resolves; the boundary denies).
    operator, raw_key = await store.create_operator(
        team_id=customer.id, display_name="Sus",
    )
    await store.set_operator_status(operator.id, "suspended")
    with pytest.raises(HTTPException) as exc403:
        await require_operator(_FakeRequest(f"Bearer {raw_key}"), store)
    assert exc403.value.status_code == 403


async def test_require_operator_missing_header_401(store):
    with pytest.raises(HTTPException) as exc:
        await require_operator(_FakeRequest(None), store)
    assert exc.value.status_code == 401


# ---------------------------------------------------------------------------
# require_role gate matrix (Foundation F1)
# ---------------------------------------------------------------------------
# require_role enforces at the dependency boundary, so these exercise the
# dependency directly (a direct endpoint call bypasses the gate, by design
# of FastAPI's Depends). Ranks: viewer(0) < operator(1) < admin(2); the
# team key is root and outranks every named role.

async def test_require_role_admits_team_key_as_root(store, customer):
    """A team (customer) key is the team root credential -- admitted by any
    gate, with operator None in the returned principal."""
    dep = require_role("admin")
    got_customer, got_operator = await dep(
        _FakeRequest(f"Bearer {customer.api_key}"), store
    )
    assert got_customer.id == customer.id
    # P1 (2026-07-02): the team key acts as the Default Admin — root within
    # its team — rather than an operator-less principal.
    assert got_operator is not None
    assert got_operator.role == "admin"
    assert got_operator.display_name == "Default Admin"


async def test_require_role_admits_admin_operator(store, customer):
    """An admin operator key clears an admin gate, returning its principal."""
    operator, raw_key = await store.create_operator(
        team_id=customer.id, display_name="Ada", role="admin",
    )
    dep = require_role("admin")
    got_customer, got_operator = await dep(
        _FakeRequest(f"Bearer {raw_key}"), store
    )
    assert got_customer.id == customer.id
    assert got_operator is not None
    assert got_operator.id == operator.id


async def test_require_role_rejects_operator_on_admin_gate(store, customer):
    """An operator-role key is below admin -> 403 on an admin gate."""
    _operator, raw_key = await store.create_operator(
        team_id=customer.id, display_name="Otto", role="operator",
    )
    dep = require_role("admin")
    with pytest.raises(HTTPException) as exc:
        await dep(_FakeRequest(f"Bearer {raw_key}"), store)
    assert exc.value.status_code == 403


async def test_require_role_rejects_viewer_on_admin_gate(store, customer):
    """A viewer key is denied the admin gate that guards operator mutation
    (create / role / status) -- the F1 'viewer denied a destructive action
    with a clean 403' guarantee, enforced at the boundary."""
    _operator, raw_key = await store.create_operator(
        team_id=customer.id, display_name="Vee", role="viewer",
    )
    dep = require_role("admin")
    with pytest.raises(HTTPException) as exc:
        await dep(_FakeRequest(f"Bearer {raw_key}"), store)
    assert exc.value.status_code == 403


async def test_require_role_rejects_viewer_on_operator_gate(store, customer):
    """A viewer key is below operator -> 403 on an operator gate."""
    _operator, raw_key = await store.create_operator(
        team_id=customer.id, display_name="Val", role="viewer",
    )
    dep = require_role("operator")
    with pytest.raises(HTTPException) as exc:
        await dep(_FakeRequest(f"Bearer {raw_key}"), store)
    assert exc.value.status_code == 403


async def test_require_role_admits_operator_on_operator_gate(store, customer):
    """An operator-role key clears an operator gate (rank == min)."""
    operator, raw_key = await store.create_operator(
        team_id=customer.id, display_name="Opal", role="operator",
    )
    dep = require_role("operator")
    _got_customer, got_operator = await dep(
        _FakeRequest(f"Bearer {raw_key}"), store
    )
    assert got_operator is not None
    assert got_operator.id == operator.id
