"""Growth G4 — shard ledger + vetting (marketplace/crediting.py +
ShardExtensionsMixin).

The append-only economy's invariants: the ledger is idempotent (a replayed
interaction never double-credits), balance nets credits against
debits/clawbacks, citation crediting excludes self-traffic + non-marketplace
crystals, and expert authorization is a simple active/revoked registry. Direct
against the in-memory store fixture; asyncio_mode=auto.
"""
from __future__ import annotations

from crystal_cache.marketplace.crediting import (
    is_marketplace_crystal,
    is_self_traffic,
    shards_from_weight,
    split_weight,
)


# --- pure crediting policy -------------------------------------------------

def test_is_marketplace_crystal():
    assert is_marketplace_crystal("general:python", None) is True
    assert is_marketplace_crystal(None, None) is True          # null customer = general
    assert is_marketplace_crystal("customer:foo", "cust_1") is False
    # Customer-owned with no general signal is NOT a marketplace crystal.
    assert is_marketplace_crystal(None, "cust_1") is False


def test_is_marketplace_crystal_customer_owned_is_not_general():
    assert is_marketplace_crystal("customer:legacy", "cust_9") is False


def test_is_self_traffic():
    assert is_self_traffic("team_1", "team_1") is True
    assert is_self_traffic("team_1", "team_2") is False
    assert is_self_traffic(None, "team_2") is False
    assert is_self_traffic("team_1", None) is False


def test_split_and_shards():
    assert split_weight(1.0, 2) == 0.5
    assert split_weight(1.0, 1) == 1.0
    assert split_weight(1.0, 0) == 1.0
    assert shards_from_weight(1.0) == 1
    assert shards_from_weight(0.25) == 1   # any positive weight → 1 shard (D7 placeholder)
    assert shards_from_weight(0.0) == 0
    assert shards_from_weight(-1.0) == 0


# --- ledger idempotency + balance ------------------------------------------

async def test_append_idempotent_no_double_credit(store):
    e1 = await store.append_shard_event(
        event_type="credit", owner_operator_id="op1", crystal_id="cz",
        consuming_team_id="t2", interaction_id="ix1",
        raw_weight=1.0, shards_credited=1,
    )
    e2 = await store.append_shard_event(
        event_type="credit", owner_operator_id="op1", crystal_id="cz",
        consuming_team_id="t2", interaction_id="ix1",
        raw_weight=1.0, shards_credited=1,
    )
    # The replay returns the SAME row, and the balance reflects ONE credit.
    assert e1["id"] == e2["id"]
    assert await store.shard_balance("op1") == 1


async def test_credit_then_clawback_nets_to_zero(store):
    await store.append_shard_event(
        event_type="credit", owner_operator_id="op2", crystal_id="cy",
        interaction_id="ix2", shards_credited=1,
    )
    # A clawback coexists with the credit (distinct event_type) and nets out.
    await store.clawback_citation(
        crystal_id="cy", interaction_id="ix2", owner_operator_id="op2", shards=1,
    )
    assert await store.shard_balance("op2") == 0


async def test_record_citation_credit_happy_then_idempotent(store):
    res = await store.record_citation_credit(
        crystal_id="cg", owner_operator_id="opZ",
        crystal_group_team_id=None, crystal_type="general:py",
        crystal_customer_id=None, consuming_team_id="t2",
        interaction_id="ixg", raw_weight=1.0,
    )
    assert res is not None
    assert res["event_type"] == "credit"
    assert await store.shard_balance("opZ") == 1

    # Re-grounding the same answer (same interaction+crystal) does not
    # double-credit.
    await store.record_citation_credit(
        crystal_id="cg", owner_operator_id="opZ",
        crystal_group_team_id=None, crystal_type="general:py",
        crystal_customer_id=None, consuming_team_id="t2",
        interaction_id="ixg", raw_weight=1.0,
    )
    assert await store.shard_balance("opZ") == 1


async def test_record_citation_credit_excludes_self_traffic(store):
    res = await store.record_citation_credit(
        crystal_id="cs", owner_operator_id="opX",
        crystal_group_team_id="t1", crystal_type="general:py",
        crystal_customer_id=None, consuming_team_id="t1",  # same team → self
        interaction_id="ixs", raw_weight=1.0,
    )
    assert res is None
    assert await store.shard_balance("opX") == 0


async def test_record_citation_credit_skips_non_marketplace(store):
    res = await store.record_citation_credit(
        crystal_id="cp", owner_operator_id="opY",
        crystal_group_team_id="t1", crystal_type="customer:foo",
        crystal_customer_id="cust1", consuming_team_id="t2",
        interaction_id="ixp", raw_weight=1.0,
    )
    assert res is None
    assert await store.shard_balance("opY") == 0


async def test_spend_is_a_debit(store):
    await store.append_shard_event(
        event_type="credit", owner_operator_id="opS", crystal_id="cc",
        interaction_id="ixc", shards_credited=5,
    )
    await store.spend_shards("opS", 2)
    assert await store.shard_balance("opS") == 3


# --- expert vetting --------------------------------------------------------

async def test_expert_authorization_lifecycle(store, customer):
    op, _ = await store.create_operator(
        team_id=customer.id, display_name="Expert",
    )
    auth = await store.authorize_expert(op.id, customer.id, "general:python")
    assert auth["status"] == "active"
    assert await store.is_expert_authorized(op.id, "general:python") is True
    assert await store.is_expert_authorized(op.id, "general:rust") is False

    listed = await store.list_expert_authorizations(customer.id)
    assert any(a["operator_id"] == op.id for a in listed)

    assert await store.revoke_expert(op.id, "general:python") is True
    assert await store.is_expert_authorized(op.id, "general:python") is False
    # Revoking a non-existent authorization is a clean False.
    assert await store.revoke_expert(op.id, "general:nope") is False
