"""P1 identity chain (ratified 2026-07-02).

Every team has a default admin operator; the bare team API key ACTS AS
that operator. Child operator keys resolve to their own operators. No
request is operator-less anymore — the owner invariant every scope and
ACL feature stands on.

R14 note: these assertions are verified by `pytest`; they describe
expected behavior and have not yet been run at authoring time.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

from crystal_cache.ingress.auth import resolve_principal


def _default_admin_id(team_id: str) -> str:
    return "opdef_" + hashlib.sha1(team_id.encode("utf-8")).hexdigest()[:16]


def _request_with_bearer(key: str):
    return SimpleNamespace(headers={"authorization": f"Bearer {key}"})


async def test_customer_creation_births_the_default_admin(store, customer):
    admin = await store.get_operator_by_id(_default_admin_id(customer.id))

    assert admin is not None
    assert admin.role == "admin"
    assert admin.team_id == customer.id
    assert admin.display_name == "Default Admin"
    assert admin.api_key_hash is None  # authenticates ONLY via the team key


async def test_ensure_default_admin_is_idempotent(store, customer):
    first = await store.ensure_default_admin(customer.id)
    second = await store.ensure_default_admin(customer.id)

    assert first.id == second.id == _default_admin_id(customer.id)


async def test_team_key_resolves_to_the_default_admin(store, customer):
    team, operator = await resolve_principal(
        _request_with_bearer(customer.api_key), store,
    )

    assert team.id == customer.id
    assert operator is not None
    assert operator.id == _default_admin_id(customer.id)
    assert operator.role == "admin"


async def test_operator_key_resolves_to_that_operator(store, customer):
    child, raw_key = await store.create_operator(
        customer.id, display_name="Sarah", role="operator",
    )

    team, operator = await resolve_principal(
        _request_with_bearer(raw_key), store,
    )

    assert team.id == customer.id
    assert operator.id == child.id
    assert operator.role == "operator"
