"""Growth G4 — marketplace API (endpoints/marketplace.py) tests.

Direct-call convention (principal injected). Balance reads are role-gated to
operator+ and authorize/revoke to admin at the auth layer; the ledger
invariants are covered in test_shard_ledger.py. These focus on endpoint
wiring: response shape, the cross-team operator 404 guard, and the vetting
lifecycle.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from crystal_cache.endpoints.marketplace import (
    AuthorizeExpertRequest,
    RevokeExpertRequest,
    authorize_expert,
    list_experts,
    revoke_expert,
    shard_balance,
    shard_ledger,
)


async def _seed_credit(store, operator_id, interaction_id="ix1"):
    await store.append_shard_event(
        event_type="credit", owner_operator_id=operator_id, crystal_id="cg",
        interaction_id=interaction_id, shards_credited=1,
    )


async def test_shard_balance_endpoint(store, customer):
    op, _ = await store.create_operator(
        team_id=customer.id, display_name="Expert",
    )
    await _seed_credit(store, op.id)
    resp = await shard_balance(
        principal=(customer, None), store=store, operator_id=op.id,
    )
    assert resp["operator_id"] == op.id
    assert resp["shard_balance"] == 1


async def test_shard_balance_cross_team_404(store, customer):
    other = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="x",
    )
    foreign_op, _ = await store.create_operator(
        team_id=other.id, display_name="Outsider",
    )
    with pytest.raises(HTTPException) as exc:
        await shard_balance(
            principal=(customer, None), store=store, operator_id=foreign_op.id,
        )
    assert exc.value.status_code == 404


async def test_shard_ledger_endpoint(store, customer):
    op, _ = await store.create_operator(
        team_id=customer.id, display_name="Expert",
    )
    await _seed_credit(store, op.id)
    resp = await shard_ledger(
        principal=(customer, None), store=store, operator_id=op.id,
    )
    assert len(resp["events"]) == 1


async def test_authorize_list_revoke_expert(store, customer):
    op, _ = await store.create_operator(
        team_id=customer.id, display_name="Expert",
    )
    auth = await authorize_expert(
        body=AuthorizeExpertRequest(operator_id=op.id, domain="general:python"),
        principal=(customer, None),
        store=store,
    )
    assert auth["authorization"]["status"] == "active"

    listed = await list_experts(principal=(customer, None), store=store)
    assert any(a["operator_id"] == op.id for a in listed["experts"])

    revoked = await revoke_expert(
        body=RevokeExpertRequest(operator_id=op.id, domain="general:python"),
        principal=(customer, None),
        store=store,
    )
    assert revoked["revoked"] is True


async def test_authorize_expert_cross_team_404(store, customer):
    other = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="x",
    )
    foreign_op, _ = await store.create_operator(
        team_id=other.id, display_name="Outsider",
    )
    with pytest.raises(HTTPException) as exc:
        await authorize_expert(
            body=AuthorizeExpertRequest(
                operator_id=foreign_op.id, domain="general:python",
            ),
            principal=(customer, None),
            store=store,
        )
    assert exc.value.status_code == 404


async def test_revoke_nonexistent_authorization_404(store, customer):
    op, _ = await store.create_operator(
        team_id=customer.id, display_name="Expert",
    )
    # The operator is in-team (passes the scope guard) but has no
    # authorization for this domain → 404 from the endpoint.
    with pytest.raises(HTTPException) as exc:
        await revoke_expert(
            body=RevokeExpertRequest(operator_id=op.id, domain="general:nope"),
            principal=(customer, None),
            store=store,
        )
    assert exc.value.status_code == 404
