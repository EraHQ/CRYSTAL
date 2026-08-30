"""L7a gate 3 (ratified 2026-08-29: Q1=A, Q2=A, Q3=A): incremental bank
update instead of invalidate + reload, and a routing-only cold load.

Before: every add_pair_for_customer ended with `invalidate(customer_id)`.
In-memory, the next routing search reloaded EVERY crystal of the
customer (full rows, both 10k-float JSON vectors through Pydantic). On
Qdrant, `invalidate` marked both lanes stale, so the next search deleted
all the customer's mirrored points and re-upserted them from a full DB
read — per pair. Now the write calls `note_pair_written(customer_id,
crystal, fact)` on the index: the loaded routing bank gets that one
crystal's row replaced-or-appended in place, the loaded fact bank gets
the one new fact appended, Qdrant gets one point per loaded lane. Banks
that are not loaded are untouched. A write redirected by an auto-split
(two crystals changed) falls back to `invalidate`. Every other mutation
site keeps `invalidate`. Cold loads use a routing-only projection.

Pinned here:
  1. The projection returns (id, type, routing_vector) for exactly the
     crystals the full list returns, honours the type filter, and its
     vectors equal the crystals' routing_vectors.
  2. Equivalence: after a run of writes through add_pair_for_customer
     the in-memory bank object was never dropped, and its ids and rows
     equal a fresh bank loaded from the DB.
  3. VectorStore.note_pair_written replaces an existing row, appends a
     new one (also into an empty-cached bank), leaves unloaded banks
     alone, and skips recall-gated crystals.
  4. FactVectorStore.note_pair_written appends the new fact to a loaded
     bank and matches a fresh load; the in-memory VectorIndex seam
     drives both lanes from one write.
  5. A redirected write (fact landed elsewhere than targeted) takes the
     invalidate path, not the in-place one.
  6. Qdrant: with a customer loaded on a lane, one upsert per lane with
     the loader's point shape; with nothing loaded, no client call.
"""
from __future__ import annotations

import numpy as np
import pytest

from crystal_cache.infrastructure.fact_vector_store import FactVectorStore
from crystal_cache.infrastructure.vector_index import InMemoryVectorIndex
from crystal_cache.infrastructure.vector_store import VectorStore


CT = "customer:legacy"
KEYS = [
    ("Topic|Ingest|batch", "one native pass per distinct text"),
    ("Topic|Ingest|batch", "buckets keep short texts short"),        # bonds to the first
    ("Topic|Lane|priority", "interactive ahead of bulk"),
    ("Topic|Worker|approve", "the worker runs the write leg"),
    ("Topic|Lane|priority", "fifo within a class"),                 # bonds to the third
    ("Topic|Bank|incremental", "one row, not a reload"),
]


async def _write(store, customer_id, encoder, vector_store, key, answer, **kw):
    return await store.add_pair_for_customer(
        customer_id=customer_id, prompt_text=key, answer_text=answer,
        encoder=encoder, vector_store=vector_store, crystal_type=CT, **kw,
    )


def _cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0.0 or nb == 0.0 else float(a @ b / (na * nb))


# ---------------------------------------------------------------------------
# 1. The projection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_routing_projection_matches_full_list(store, customer, semantic_encoder_stub, vector_store):
    for key, answer in KEYS[:4]:
        await _write(store, customer.id, semantic_encoder_stub, vector_store, key, answer)
    full = await store.list_crystals_for_customer(customer.id, include_recall_gated=False)
    proj = await store.list_routing_vectors_for_customer(customer.id)
    assert {cid for cid, _, _ in proj} == {c.id for c in full if c.routing_vector}
    by_id = {c.id: c for c in full}
    for cid, ctype, vec in proj:
        assert ctype == by_id[cid].crystal_type == CT
        assert vec == list(by_id[cid].routing_vector)
    assert await store.list_routing_vectors_for_customer(customer.id, "other:type") == []
    assert await store.list_routing_vectors_for_customer("cus_nobody") == []


# ---------------------------------------------------------------------------
# 2. Equivalence with a reload
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_incremental_bank_equals_fresh_load(store, customer, semantic_encoder_stub, vector_store):
    key = (customer.id, CT)
    bank_obj = None
    for k, a in KEYS:
        await _write(store, customer.id, semantic_encoder_stub, vector_store, k, a)
        bank = vector_store._banks[key]
        if bank_obj is None:
            bank_obj = bank
        assert bank is bank_obj                           # never dropped and reloaded

    fresh = VectorStore(store)
    fresh_bank = await fresh._ensure_loaded(customer.id, CT)
    assert set(bank_obj.crystal_ids) == set(fresh_bank.crystal_ids)
    assert len(bank_obj.crystal_ids) == 4                # 6 writes, 2 bonds
    for cid in bank_obj.crystal_ids:
        a = bank_obj.matrix[bank_obj.crystal_ids.index(cid)]
        b = fresh_bank.matrix[fresh_bank.crystal_ids.index(cid)]
        assert _cos(a, b) > 0.9999


# ---------------------------------------------------------------------------
# 3. VectorStore.note_pair_written edge cases
# ---------------------------------------------------------------------------

class _Cr:
    def __init__(self, id, vec, crystal_type=CT, recall_gated=False):
        self.id, self.routing_vector, self.crystal_type, self.recall_gated = id, vec, crystal_type, recall_gated


@pytest.mark.asyncio
async def test_note_pair_written_replace_append_skip(store):
    vs = VectorStore(store)
    cid = "cus_x"
    empty = await vs._ensure_loaded(cid, CT)              # empty bank cached as (0,0)
    assert empty.matrix.size == 0

    await vs.note_pair_written(cid, _Cr("c1", [1.0, 0.0, 0.0]))
    bank = vs._banks[(cid, CT)]
    assert bank.crystal_ids == ["c1"] and bank.matrix.shape == (1, 3)

    await vs.note_pair_written(cid, _Cr("c2", [0.0, 2.0, 0.0]))
    assert bank.crystal_ids == ["c1", "c2"]
    assert np.allclose(bank.matrix[1], [0.0, 1.0, 0.0])   # unit-normalized

    await vs.note_pair_written(cid, _Cr("c1", [0.0, 0.0, 3.0]))   # replace, no growth
    assert bank.crystal_ids == ["c1", "c2"]
    assert np.allclose(bank.matrix[0], [0.0, 0.0, 1.0])

    await vs.note_pair_written(cid, _Cr("gated", [1.0, 1.0, 1.0], recall_gated=True))
    assert bank.crystal_ids == ["c1", "c2"]
    await vs.note_pair_written(cid, _Cr("c3", [1.0, 1.0]))          # dim mismatch: left to a reload
    assert bank.crystal_ids == ["c1", "c2"]
    await vs.note_pair_written(cid, _Cr("c4", [1.0, 0.0, 0.0], crystal_type="other:type"))
    assert (cid, "other:type") not in vs._banks           # unloaded bank untouched


# ---------------------------------------------------------------------------
# 4. Fact lane in place, through the in-memory VectorIndex seam
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fact_bank_appends_in_place_via_index_seam(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    index = InMemoryVectorIndex(fact_store=fact_vector_store, vector_store=vector_store, metadata_store=store)
    await _write(store, customer.id, semantic_encoder_stub, vector_store, *KEYS[0], vector_index=index)
    fact_bank = await fact_vector_store._ensure_loaded(customer.id)   # load once
    n0 = len(fact_bank.entries)

    _, fact = await _write(store, customer.id, semantic_encoder_stub, vector_store, *KEYS[2], vector_index=index)
    assert fact_vector_store._banks[customer.id] is fact_bank          # not dropped
    assert len(fact_bank.entries) == n0 + 1
    assert fact_bank.entries[-1].fact_id == fact.id
    assert _cos(fact_bank.matrix[-1], fact.vector) > 0.9999

    fresh = FactVectorStore(store)
    fresh_bank = await fresh._ensure_loaded(customer.id)
    assert {e.fact_id for e in fresh_bank.entries} == {e.fact_id for e in fact_bank.entries}


# ---------------------------------------------------------------------------
# 5. Redirect falls back to invalidate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_redirected_write_invalidates(store, customer, semantic_encoder_stub, vector_store, monkeypatch):
    crystal_a, _ = await _write(store, customer.id, semantic_encoder_stub, vector_store, *KEYS[0])

    orig = store.add_pair_to_crystal
    async def redirected(**kw):
        fact = await orig(**kw)
        return fact.model_copy(update={"crystal_id": crystal_a.id})   # landed elsewhere
    monkeypatch.setattr(store, "add_pair_to_crystal", redirected)

    calls: list[str] = []
    orig_note = vector_store.note_pair_written
    async def note(*a, **k):
        calls.append("note"); await orig_note(*a, **k)
    monkeypatch.setattr(vector_store, "note_pair_written", note)
    monkeypatch.setattr(vector_store, "invalidate", lambda cid: calls.append("invalidate"))

    await _write(store, customer.id, semantic_encoder_stub, vector_store, *KEYS[2])   # spawns B, redirected to A
    assert calls == ["invalidate"]


# ---------------------------------------------------------------------------
# 6. Qdrant: one point per loaded lane, nothing when not loaded
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_qdrant_note_pair_written_mirrors_one_point_per_loaded_lane(store):
    pytest.importorskip("qdrant_client")
    from crystal_cache.infrastructure.qdrant_vector_index import QdrantVectorIndex

    class _Client:
        def __init__(self):
            self.upserts: list[tuple[str, list]] = []
        async def upsert(self, collection, points):
            self.upserts.append((collection, points))

    client = _Client()
    idx = QdrantVectorIndex(client=client, metadata_store=store)
    crystal = _Cr("c1", [1.0, 0.0, 0.0, 0.0])

    class _Fact:
        id, crystal_id, pair_type, prompt_text, vector = "f1", "c1", "question_answer", "k", [0.0, 1.0]

    await idx.note_pair_written("cus_q", crystal, _Fact())
    assert client.upserts == []                          # nothing loaded -> nothing mirrored

    idx._routing_loaded.add("cus_q"); idx._routing_dim = 4
    idx._loaded.add("cus_q"); idx._dim = 2
    await idx.note_pair_written("cus_q", crystal, _Fact())
    assert [c for c, _ in client.upserts] == [idx._routing_collection, idx._collection]
    rp, fp = client.upserts[0][1][0], client.upserts[1][1][0]
    assert rp.id == idx._pid("c1") and rp.payload["crystal_id"] == "c1" and rp.payload["customer_id"] == "cus_q"
    assert fp.id == idx._pid("f1") and fp.payload["fact_id"] == "f1" and fp.payload["pair_type"] == "question_answer"
    assert rp.payload == {"scope": "customer", "customer_id": "cus_q", "crystal_id": "c1", "crystal_type": CT}
