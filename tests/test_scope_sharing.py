"""P2 scope + sharing (ratified 2026-07-02).

Personal-by-default ingest (CC_DEFAULT_INGEST_SCOPE), the share primitive
(one reversible mode write), the owner-or-admin share endpoint, and the
can_read journey: personal → invisible to teammates → shared → visible →
unshared → invisible again. The endpoint-level ingest-default precedence
(scope > private > knob) is three inline lines exercised by the live
smoke; the knob value itself is pinned here.

R14 note: these assertions are verified by `pytest`; they describe
expected behavior and have not yet been run at authoring time.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from crystal_cache.config import get_settings
from crystal_cache.endpoints.sdk import sdk_set_crystal_scope
from crystal_cache.infrastructure.permissions import can_read, mode_for_scope
from crystal_cache.infrastructure.schema import CrystalRow


async def _seed_crystal(
    store, customer_id: str, cid: str, *, owner: str, mode: int,
):
    async with store.session() as s:
        s.add(CrystalRow(
            id=cid, customer_id=customer_id,
            crystal_type="customer:legacy", summary_vector=[],
            owner_operator_id=owner, group_team_id=customer_id, mode=mode,
        ))


def test_scope_vocabulary():
    assert mode_for_scope("personal") == 0o600
    assert mode_for_scope("team") == 0o640
    with pytest.raises(ValueError):
        mode_for_scope("public")


def test_deployment_default_is_personal():
    assert get_settings().default_ingest_scope == "personal"


async def test_set_crystal_scope_flips_the_mode(store, customer):
    await _seed_crystal(store, customer.id, "c_flip", owner="op_x", mode=0o600)

    assert await store.set_crystal_scope("c_flip", customer.id, "team") is True
    assert (await store.get_crystal("c_flip")).mode == 0o640

    assert await store.set_crystal_scope("c_flip", customer.id, "personal") is True
    assert (await store.get_crystal("c_flip")).mode == 0o600


async def test_set_crystal_scope_is_customer_guarded(store, customer):
    await _seed_crystal(store, "cus_other", "c_theirs", owner="op_x", mode=0o600)

    assert await store.set_crystal_scope("c_theirs", customer.id, "team") is False
    assert await store.set_crystal_scope("c_missing", customer.id, "team") is False


async def test_share_journey_owner_teammate_and_back(store, customer):
    """The whole point: personal is invisible to teammates, one share makes
    it visible, one unshare hides it again — no copies anywhere."""
    owner, _ = await store.create_operator(customer.id, display_name="Own")
    mate, _ = await store.create_operator(customer.id, display_name="Mate")
    await _seed_crystal(store, customer.id, "c_j", owner=owner.id, mode=0o600)

    crystal = await store.get_crystal("c_j")
    assert can_read(crystal, owner) is True
    assert can_read(crystal, mate) is False  # personal: teammate blocked

    await store.set_crystal_scope("c_j", customer.id, "team")
    crystal = await store.get_crystal("c_j")
    assert can_read(crystal, mate) is True   # shared: one mode write

    await store.set_crystal_scope("c_j", customer.id, "personal")
    crystal = await store.get_crystal("c_j")
    assert can_read(crystal, mate) is False  # reversible

    # The Default Admin (P1) is root within its team throughout.
    default_admin = await store.ensure_default_admin(customer.id)
    assert can_read(crystal, default_admin) is True


async def test_share_endpoint_authorization(store, customer):
    owner, _ = await store.create_operator(customer.id, display_name="Own")
    other, _ = await store.create_operator(customer.id, display_name="Oth")
    await _seed_crystal(store, customer.id, "c_ep", owner=owner.id, mode=0o600)

    # Owner may share.
    resp = await sdk_set_crystal_scope(
        "c_ep", {"scope": "team"}, (customer, owner), store,
    )
    assert resp.status_code == 200
    assert (await store.get_crystal("c_ep")).mode == 0o640

    # A non-owner, non-admin teammate may not.
    with pytest.raises(HTTPException) as exc:
        await sdk_set_crystal_scope(
            "c_ep", {"scope": "personal"}, (customer, other), store,
        )
    assert exc.value.status_code == 403

    # The Default Admin (team key) may — admin is root within its team.
    default_admin = await store.ensure_default_admin(customer.id)
    resp = await sdk_set_crystal_scope(
        "c_ep", {"scope": "personal"}, (customer, default_admin), store,
    )
    assert resp.status_code == 200
    assert (await store.get_crystal("c_ep")).mode == 0o600


async def test_share_endpoint_rejects_bad_scope_and_foreign(store, customer):
    owner, _ = await store.create_operator(customer.id, display_name="Own")
    await _seed_crystal(store, customer.id, "c_v", owner=owner.id, mode=0o600)
    await _seed_crystal(store, "cus_other", "c_f", owner="op_z", mode=0o600)

    with pytest.raises(HTTPException) as exc:
        await sdk_set_crystal_scope(
            "c_v", {"scope": "public"}, (customer, owner), store,
        )
    assert exc.value.status_code == 422

    with pytest.raises(HTTPException) as exc:
        await sdk_set_crystal_scope(
            "c_f", {"scope": "team"}, (customer, owner), store,
        )
    assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# KEYSTONE — scope is a merge boundary (may_join + the add_pair guard)
# ---------------------------------------------------------------------------

async def _add(store, customer, encoder, vector_store, *, text, owner, mode):
    crystal, _ = await store.add_pair_for_customer(
        customer_id=customer.id,
        prompt_text=text,
        answer_text=f"{text} answer",
        encoder=encoder,
        vector_store=vector_store,
        owner_operator_id=owner,
        group_team_id=customer.id,
        mode=mode,
    )
    return crystal


async def test_two_operators_personal_facts_never_share_a_crystal(
    store, customer, semantic_encoder_stub, vector_store,
):
    op_a, _ = await store.create_operator(customer.id, display_name="A")
    op_b, _ = await store.create_operator(customer.id, display_name="B")

    c1 = await _add(store, customer, semantic_encoder_stub, vector_store,
                    text="gamma topic", owner=op_a.id, mode=0o600)
    c2 = await _add(store, customer, semantic_encoder_stub, vector_store,
                    text="gamma topic", owner=op_b.id, mode=0o600)

    assert c1.id != c2.id
    assert c2.owner_operator_id == op_b.id


async def test_personal_pair_never_joins_a_team_crystal(
    store, customer, semantic_encoder_stub, vector_store,
):
    op_a, _ = await store.create_operator(customer.id, display_name="A")

    team_c = await _add(store, customer, semantic_encoder_stub, vector_store,
                        text="delta topic", owner=op_a.id, mode=0o640)
    personal_c = await _add(store, customer, semantic_encoder_stub, vector_store,
                            text="delta topic", owner=op_a.id, mode=0o600)

    assert personal_c.id != team_c.id
    assert personal_c.mode == 0o600
    assert (await store.get_crystal(team_c.id)).mode == 0o640


async def test_team_pairs_join_across_authors(
    store, customer, semantic_encoder_stub, vector_store,
):
    """Team mode deliberately allows cross-author joins — contributing to
    a shared crystal doesn't change who owns it."""
    op_a, _ = await store.create_operator(customer.id, display_name="A")
    op_b, _ = await store.create_operator(customer.id, display_name="B")

    c1 = await _add(store, customer, semantic_encoder_stub, vector_store,
                    text="epsilon topic", owner=op_a.id, mode=0o640)
    c2 = await _add(store, customer, semantic_encoder_stub, vector_store,
                    text="epsilon topic", owner=op_b.id, mode=0o640)

    assert c1.id == c2.id
    assert (await store.get_crystal(c1.id)).owner_operator_id == op_a.id


async def test_legacy_unstamped_pairs_still_join(
    store, customer, semantic_encoder_stub, vector_store,
):
    """Pre-scope callers (no stamps) keep today's bonding — the boundary
    never splits legacy banks."""
    c1, _ = await store.add_pair_for_customer(
        customer_id=customer.id, prompt_text="zeta topic",
        answer_text="zeta one",
        encoder=semantic_encoder_stub, vector_store=vector_store,
    )
    c2, _ = await store.add_pair_for_customer(
        customer_id=customer.id, prompt_text="zeta topic",
        answer_text="zeta two",
        encoder=semantic_encoder_stub, vector_store=vector_store,
    )
    assert c1.id == c2.id
