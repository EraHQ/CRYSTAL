"""Foundation F2.2a — operator-scoped writes + the operator auth boundary.

Covers:
  - resolve_principal: operator key → (team, operator); team key →
    (customer, None); unknown → 401; suspended → 403; orphan team → 401.
  - add_pair_for_customer stamps POSIX ownership on spawn-fresh, and leaves
    an existing crystal's owner untouched on bond.
  - auto-split inherits the parent's ownership (no privacy leak on split).
  - sdk_store rejects a viewer (read-only) before doing any work.

asyncio_mode=auto (pyproject) — async tests need no marker.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi import HTTPException

from crystal_cache.endpoints.sdk import sdk_store
from crystal_cache.ingress.auth import resolve_principal
from crystal_cache.ingress.schema import StoreRequest
from crystal_cache.models import Crystal, CrystalType


class _FakeRequest:
    def __init__(self, authorization: str | None = None) -> None:
        self.headers = {"authorization": authorization} if authorization else {}


# ---------------------------------------------------------------------------
# resolve_principal
# ---------------------------------------------------------------------------

async def test_resolve_principal_operator_key(store, customer):
    operator, raw_key = await store.create_operator(
        team_id=customer.id, display_name="Ada",
    )
    team, op = await resolve_principal(_FakeRequest(f"Bearer {raw_key}"), store)
    assert team.id == customer.id
    assert op is not None and op.id == operator.id


async def test_resolve_principal_team_key(store, customer):
    # create_customer returns the raw key once on the Customer record.
    team, op = await resolve_principal(
        _FakeRequest(f"Bearer {customer.api_key}"), store,
    )
    assert team.id == customer.id
    # P1 identity chain (2026-07-02): the team key resolves to the team's
    # DEFAULT ADMIN operator — no request is operator-less anymore.
    assert op is not None
    assert op.role == "admin"
    assert op.display_name == "Default Admin"
    assert op.team_id == customer.id


async def test_resolve_principal_unknown_key_401(store):
    with pytest.raises(HTTPException) as exc:
        await resolve_principal(_FakeRequest("Bearer cc_sk_nope"), store)
    assert exc.value.status_code == 401


async def test_resolve_principal_suspended_operator_403(store, customer):
    operator, raw_key = await store.create_operator(
        team_id=customer.id, display_name="Sus",
    )
    await store.set_operator_status(operator.id, "suspended")
    with pytest.raises(HTTPException) as exc:
        await resolve_principal(_FakeRequest(f"Bearer {raw_key}"), store)
    assert exc.value.status_code == 403


async def test_resolve_principal_orphan_team_401(store):
    # Operator whose team_id points at no customer row (SQLite doesn't
    # enforce the FK) → integrity 401, not a silent pass.
    operator, raw_key = await store.create_operator(
        team_id="cus_ghost", display_name="Orphan",
    )
    with pytest.raises(HTTPException) as exc:
        await resolve_principal(_FakeRequest(f"Bearer {raw_key}"), store)
    assert exc.value.status_code == 401


# ---------------------------------------------------------------------------
# Write-side ownership stamping
# ---------------------------------------------------------------------------

async def test_add_pair_stamps_ownership_on_spawn(
    store, customer, semantic_encoder_stub, vector_store,
):
    op, _ = await store.create_operator(team_id=customer.id, display_name="A")
    crystal, _fact = await store.add_pair_for_customer(
        customer_id=customer.id,
        prompt_text="alpha topic",
        answer_text="alpha answer",
        encoder=semantic_encoder_stub,
        vector_store=vector_store,
        owner_operator_id=op.id,
        group_team_id=op.team_id,
        mode=0o600,
    )
    assert crystal.owner_operator_id == op.id
    assert crystal.group_team_id == op.team_id
    assert crystal.mode == 0o600


async def test_add_pair_unowned_defaults(
    store, customer, semantic_encoder_stub, vector_store,
):
    crystal, _ = await store.add_pair_for_customer(
        customer_id=customer.id,
        prompt_text="beta topic",
        answer_text="beta answer",
        encoder=semantic_encoder_stub,
        vector_store=vector_store,
    )
    assert crystal.owner_operator_id is None
    assert crystal.group_team_id is None
    assert crystal.mode == 0o640


async def test_bond_preserves_existing_owner(
    store, customer, semantic_encoder_stub, vector_store,
):
    """SUPERSEDED BEHAVIOR RECORDED (keystone, ratified 2026-07-02): this
    test originally asserted op_b's TEAM pair bonding into op_a's PERSONAL
    crystal with ownership preserved — which is precisely the scope leak
    the merge boundary now forbids (op_b could write a fact they couldn't
    read). Scope-mismatched pairs now SPAWN. Ownership preservation on
    legitimate (same-scope) cross-author bonds is covered in
    test_scope_sharing.test_team_pairs_join_across_authors."""
    op_a, _ = await store.create_operator(team_id=customer.id, display_name="A")
    op_b, _ = await store.create_operator(team_id=customer.id, display_name="B")
    first, _ = await store.add_pair_for_customer(
        customer_id=customer.id,
        prompt_text="shared topic",
        answer_text="answer one",
        encoder=semantic_encoder_stub,
        vector_store=vector_store,
        bond_threshold=0.5,
        owner_operator_id=op_a.id,
        group_team_id=op_a.team_id,
        mode=0o600,
    )
    # Same prompt routes to `first` (cosine ~1.0 ≥ 0.5), but the scope
    # identities differ (personal/op_a vs team/op_b) — the keystone spawns.
    second, _ = await store.add_pair_for_customer(
        customer_id=customer.id,
        prompt_text="shared topic",
        answer_text="answer two",
        encoder=semantic_encoder_stub,
        vector_store=vector_store,
        bond_threshold=0.5,
        owner_operator_id=op_b.id,
        group_team_id=op_b.team_id,
        mode=0o640,
    )
    assert second.id != first.id
    assert second.owner_operator_id == op_b.id
    assert second.mode == 0o640
    # op_a's personal crystal is untouched.
    assert (await store.get_crystal(first.id)).mode == 0o600
    assert (await store.get_crystal(first.id)).owner_operator_id == op_a.id


async def test_autosplit_inherits_ownership(
    store, customer, semantic_encoder_stub,
):
    # capacity-1 type so the 2nd write to a crystal auto-splits.
    await store.upsert_crystal_type(CrystalType(
        id="test:cap1", display_name="Cap1", scope="customer",
        capacity_default=1,
    ))
    await store.upsert_crystal(Crystal(
        id="crys_owned", customer_id=customer.id, summary_vector=[],
        crystal_type="test:cap1",
        owner_operator_id="op_a", group_team_id=customer.id, mode=0o600,
    ))
    await store.add_pair_to_crystal(
        crystal_id="crys_owned", prompt_text="q1", answer_text="a1",
        encoder=semantic_encoder_stub,
    )
    # Second write exceeds capacity 1 → auto-split into a sibling.
    fact2 = await store.add_pair_to_crystal(
        crystal_id="crys_owned", prompt_text="q2", answer_text="a2",
        encoder=semantic_encoder_stub,
    )
    assert fact2.crystal_id != "crys_owned"  # redirected to a sibling
    sibling = await store.get_crystal(fact2.crystal_id)
    assert sibling.owner_operator_id == "op_a"
    assert sibling.group_team_id == customer.id
    assert sibling.mode == 0o600


# ---------------------------------------------------------------------------
# sdk_store viewer rejection
# ---------------------------------------------------------------------------

async def test_sdk_store_rejects_viewer(store, customer):
    viewer, _ = await store.create_operator(
        team_id=customer.id, display_name="V", role="viewer",
    )
    body = StoreRequest(key="k", value="v")
    # The viewer check fires before any encoder/LLM work, so a bare request
    # with no app.state is fine.
    with pytest.raises(HTTPException) as exc:
        await sdk_store(body, _FakeRequest(), (customer, viewer), store)
    assert exc.value.status_code == 403


# ---------------------------------------------------------------------------
# F2.2b — crystal-level retrieval is permission-filtered
# ---------------------------------------------------------------------------

async def test_vector_search_excludes_other_operators_private(
    store, customer, vector_store,
):
    op_a, _ = await store.create_operator(team_id=customer.id, display_name="A")
    op_b, _ = await store.create_operator(team_id=customer.id, display_name="B")
    # Two crystals with identical routing vectors: one private to op_a
    # (0o600), one team-readable (0o640). Setting routing_vector directly
    # keeps the test independent of encoder geometry — both score 1.0
    # against the query, so only the permission filter decides the result.
    rv = [1.0, 0.0, 0.0, 0.0]
    await store.upsert_crystal(Crystal(
        id="crys_priv", customer_id=customer.id, summary_vector=[],
        routing_vector=rv, crystal_type="customer:legacy",
        owner_operator_id=op_a.id, group_team_id=customer.id, mode=0o600,
    ))
    await store.upsert_crystal(Crystal(
        id="crys_team", customer_id=customer.id, summary_vector=[],
        routing_vector=rv, crystal_type="customer:legacy",
        owner_operator_id=op_a.id, group_team_id=customer.id, mode=0o640,
    ))
    qvec = np.asarray(rv, dtype=np.float32)

    # op_b (teammate, not owner): the 0o600 crystal is hidden, the 0o640 one
    # is visible via the group read bit.
    ids_b = {cid for cid, _ in await vector_store.search(
        customer_id=customer.id, query_vector=qvec, k=5,
        crystal_type="customer:legacy", operator=op_b,
    )}
    assert "crys_priv" not in ids_b
    assert "crys_team" in ids_b

    # op_a (owner): sees its own private crystal.
    ids_a = {cid for cid, _ in await vector_store.search(
        customer_id=customer.id, query_vector=qvec, k=5,
        crystal_type="customer:legacy", operator=op_a,
    )}
    assert "crys_priv" in ids_a

    # No operator → today's unfiltered behavior: both present.
    ids_none = {cid for cid, _ in await vector_store.search(
        customer_id=customer.id, query_vector=qvec, k=5,
        crystal_type="customer:legacy",
    )}
    assert {"crys_priv", "crys_team"} <= ids_none
