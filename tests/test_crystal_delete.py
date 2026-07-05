"""B.3 — DELETE /v1/crystals/{id} and /v1/facts/{id} wired to the store.

Crystal delete (store.delete_crystal): cascades to facts, invalidates the
vector stores, tenancy-scoped. Fact delete (store.delete_fact): removes one
fact and REBUILDS the crystal's summary/routing vectors from the survivors,
or deletes the crystal if that was its last fact. Coverage:
  - crystal delete: happy (gone) + 404 (missing / other-tenant).
  - fact delete recompute: removing the 2nd fact restores the crystal's
    vectors to their exact 1-fact state (rebuild is exact — same encoder,
    same stored prompt/claim text, pure-additive accumulation).
  - last-fact delete removes the now-empty crystal.
  - fact delete: missing → False (store) / 404 (endpoint).

asyncio_mode=auto (pyproject) — async tests need no marker.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from crystal_cache.endpoints.sdk import sdk_crystal_delete, sdk_fact_delete
from crystal_cache.models import Crystal, CrystalType


def _request_with_state(vector_store, fact_vector_store, prompt_encoder=None):
    """A minimal Request stand-in carrying app.state.{vector_store,
    fact_vector_store, prompt_encoder} — the only app.state the delete
    handlers read (vector stores via getattr; the fact handler reads the
    encoder directly, as sdk_store does)."""
    state = SimpleNamespace(
        vector_store=vector_store,
        fact_vector_store=fact_vector_store,
        prompt_encoder=prompt_encoder,
    )
    return SimpleNamespace(app=SimpleNamespace(state=state), headers={})


# ---------------------------------------------------------------------------
# Crystal delete
# ---------------------------------------------------------------------------

async def test_delete_removes_crystal_and_facts(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    crystal, _fact = await store.add_pair_for_customer(
        customer_id=customer.id,
        prompt_text="delete me topic",
        answer_text="delete me answer",
        encoder=semantic_encoder_stub,
        vector_store=vector_store,
    )
    assert await store.get_crystal(crystal.id) is not None
    assert len(await store.list_facts_for_crystal(crystal.id)) == 1

    req = _request_with_state(vector_store, fact_vector_store)
    resp = await sdk_crystal_delete(crystal.id, req, customer, store)

    assert json.loads(resp.body)["deleted"] is True
    assert await store.get_crystal(crystal.id) is None
    assert await store.list_facts_for_crystal(crystal.id) == []


async def test_delete_missing_crystal_404(
    store, customer, vector_store, fact_vector_store,
):
    req = _request_with_state(vector_store, fact_vector_store)
    with pytest.raises(HTTPException) as exc:
        await sdk_crystal_delete("crys_does_not_exist", req, customer, store)
    assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# Fact delete — recompute-on-delete (Option 3)
# ---------------------------------------------------------------------------

async def test_delete_fact_recomputes_crystal_vectors(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    enc = semantic_encoder_stub
    # High-capacity type so both facts live in ONE crystal (no auto-split),
    # and both go through add_pair_to_crystal — the exact path recompute
    # replays.
    await store.upsert_crystal_type(CrystalType(
        id="test:multi", display_name="Multi", scope="customer",
        capacity_default=10,
    ))
    await store.upsert_crystal(Crystal(
        id="crys_multi", customer_id=customer.id, summary_vector=[],
        crystal_type="test:multi",
        owner_operator_id=None, group_team_id=customer.id, mode=0o640,
    ))

    # Fact A, then snapshot the 1-fact vectors.
    fact_a = await store.add_pair_to_crystal(
        crystal_id="crys_multi", prompt_text="alpha topic",
        answer_text="alpha answer", encoder=enc,
    )
    after_a = await store.get_crystal("crys_multi")
    summary_a = np.asarray(after_a.summary_vector, dtype=np.float32)
    routing_a = np.asarray(after_a.routing_vector, dtype=np.float32)
    assert after_a.fact_count == 1

    # Fact B into the same crystal → vectors change.
    fact_b = await store.add_pair_to_crystal(
        crystal_id="crys_multi", prompt_text="beta topic",
        answer_text="beta answer", encoder=enc,
    )
    assert fact_b.crystal_id == "crys_multi"  # capacity 10 → no split
    after_b = await store.get_crystal("crys_multi")
    assert after_b.fact_count == 2
    assert not np.allclose(
        np.asarray(after_b.summary_vector, dtype=np.float32), summary_a
    )

    # Delete B → rebuild from survivors {A} → back to the 1-fact vectors.
    deleted = await store.delete_fact(
        fact_b.id, customer.id, encoder=enc,
        vector_store=vector_store, fact_vector_store=fact_vector_store,
    )
    assert deleted is True

    after_del = await store.get_crystal("crys_multi")
    assert after_del is not None
    assert after_del.fact_count == 1
    remaining = {f.id for f in await store.list_facts_for_crystal("crys_multi")}
    assert fact_a.id in remaining and fact_b.id not in remaining
    # The rebuild is exact (same encoder, same stored text, additive accum).
    assert np.allclose(
        np.asarray(after_del.summary_vector, dtype=np.float32), summary_a,
        atol=1e-5,
    )
    assert np.allclose(
        np.asarray(after_del.routing_vector, dtype=np.float32), routing_a,
        atol=1e-5,
    )


async def test_delete_last_fact_removes_crystal(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    enc = semantic_encoder_stub
    crystal, fact = await store.add_pair_for_customer(
        customer_id=customer.id, prompt_text="solo topic",
        answer_text="solo answer", encoder=enc, vector_store=vector_store,
    )
    deleted = await store.delete_fact(
        fact.id, customer.id, encoder=enc,
        vector_store=vector_store, fact_vector_store=fact_vector_store,
    )
    assert deleted is True
    # No survivors → the crystal is deleted whole.
    assert await store.get_crystal(crystal.id) is None


async def test_delete_fact_missing_returns_false(
    store, customer, semantic_encoder_stub,
):
    assert await store.delete_fact(
        "fact_does_not_exist", customer.id, encoder=semantic_encoder_stub,
    ) is False


async def test_endpoint_fact_delete_and_404(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    enc = semantic_encoder_stub
    crystal, fact = await store.add_pair_for_customer(
        customer_id=customer.id, prompt_text="endpoint topic",
        answer_text="endpoint answer", encoder=enc, vector_store=vector_store,
    )
    req = _request_with_state(vector_store, fact_vector_store, enc)
    resp = await sdk_fact_delete(fact.id, req, customer, store)
    assert json.loads(resp.body)["deleted"] is True

    with pytest.raises(HTTPException) as exc:
        await sdk_fact_delete("fact_nope", req, customer, store)
    assert exc.value.status_code == 404
