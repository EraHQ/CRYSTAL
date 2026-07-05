"""Foundation F1 identity layer: operator CRUD + customer-key hashing.

Operators sit beneath the customer/team entity (CustomerRow IS the team);
credentials are stored hashed, never in plaintext. These exercise the
store CRUD added to metadata_store.py against the in-memory `store`
fixture. asyncio_mode=auto (see pyproject) means async tests need no
marker.
"""
from __future__ import annotations

from crystal_cache.infrastructure.credentials import hash_api_key


# ---------------------------------------------------------------------------
# Customer key hashing — no plaintext at rest
# ---------------------------------------------------------------------------

async def test_create_customer_returns_raw_key_once(store):
    customer = await store.create_customer(
        provider="anthropic",
        model_id="claude-sonnet-4-5-20250929",
        api_key_ref="sk-upstream",
    )
    # The raw Key A is returned exactly once, at creation.
    assert customer.api_key is not None
    assert customer.api_key.startswith("cc_sk_")


async def test_raw_key_authenticates_via_hash_lookup(store):
    customer = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="sk-upstream",
    )
    raw = customer.api_key

    found = await store.get_customer_by_api_key(raw)
    assert found is not None
    assert found.id == customer.id
    # Reads never expose the raw key — only the hash lives in the DB.
    assert found.api_key is None


async def test_wrong_key_does_not_authenticate(store):
    await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="x",
    )
    assert await store.get_customer_by_api_key("cc_sk_not_a_real_key") is None


# ---------------------------------------------------------------------------
# Operator CRUD
# ---------------------------------------------------------------------------

async def test_create_operator_returns_raw_key_and_stores_hash(store, customer):
    operator, raw_key = await store.create_operator(
        team_id=customer.id,
        display_name="Ada",
        role="admin",
    )
    assert raw_key.startswith("cc_sk_")
    assert operator.team_id == customer.id
    assert operator.display_name == "Ada"
    assert operator.role == "admin"
    assert operator.status == "active"
    # Only the hash is carried/stored — never the raw key.
    assert operator.api_key_hash == hash_api_key(raw_key)
    assert operator.api_key_hash != raw_key


async def test_get_operator_by_api_key_round_trip(store, customer):
    operator, raw_key = await store.create_operator(
        team_id=customer.id, display_name="Grace",
    )
    found = await store.get_operator_by_api_key(raw_key)
    assert found is not None
    assert found.id == operator.id
    # Wrong key -> None.
    assert await store.get_operator_by_api_key("cc_sk_wrong") is None


async def test_get_operator_by_id(store, customer):
    operator, _ = await store.create_operator(
        team_id=customer.id, display_name="Lin",
    )
    found = await store.get_operator_by_id(operator.id)
    assert found is not None
    assert found.id == operator.id
    assert await store.get_operator_by_id("op_missing") is None


async def test_list_operators_for_team_is_scoped(store, customer):
    o1, _ = await store.create_operator(team_id=customer.id, display_name="A")
    o2, _ = await store.create_operator(team_id=customer.id, display_name="B")

    other = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="y",
    )
    o3, _ = await store.create_operator(team_id=other.id, display_name="C")

    team_ops = await store.list_operators_for_team(customer.id)
    ids = {o.id for o in team_ops}
    # P1 (2026-07-02): the team's Default Admin is always on the roster.
    default_admin = await store.ensure_default_admin(customer.id)
    assert ids == {o1.id, o2.id, default_admin.id}
    assert o3.id not in ids


async def test_set_operator_role_and_status(store, customer):
    operator, _ = await store.create_operator(
        team_id=customer.id, display_name="A",
    )
    assert await store.set_operator_role(operator.id, "viewer") is True
    assert await store.set_operator_status(operator.id, "suspended") is True

    reloaded = await store.get_operator_by_id(operator.id)
    assert reloaded.role == "viewer"
    assert reloaded.status == "suspended"

    # Unknown id -> False (nothing updated).
    assert await store.set_operator_role("op_missing", "admin") is False
    assert await store.set_operator_status("op_missing", "active") is False
