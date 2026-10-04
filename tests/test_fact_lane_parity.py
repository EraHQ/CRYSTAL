"""RC-01 (2026-10-04): fact and routing indexes cannot drift from the DB.

Two processes share one database but each holds its own in-memory (or
Qdrant-mirror) caches. Before this fix a write in one process was
invisible to the other forever, and even inside one process each write
path told whichever index its author had in hand. Now:

- every bank write or delete goes through MetadataStore.bank_changed,
  which bumps bank_generations and notifies every attached index;
- every index compares the generation it loaded against the DB on each
  search and reloads when behind.

The pin stands up TWO sets of indexes over the same store. Set A is
attached (the writer's process). Set B is not attached and only ever
learns of changes through the generation (the other process). Every
assertion drives the real search lane, not a helper.
"""
import numpy as np
import pytest

from crystal_cache.infrastructure.fact_vector_store import FactVectorStore
from crystal_cache.infrastructure.schema import CrystalRow, FactRow
from crystal_cache.infrastructure.vector_store import VectorStore

D = 8  # small vectors: the pin is about freshness, not geometry


def _unit(i: int) -> list[float]:
    v = [0.0] * D
    v[i % D] = 1.0
    return v


async def _write_crystal(store, cid, crystal_id, dim_idx, crystal_type="customer:legacy"):
    async with store.session() as s:
        s.add(CrystalRow(
            id=crystal_id, customer_id=cid, crystal_type=crystal_type,
            summary_vector=_unit(dim_idx), routing_vector=_unit(dim_idx),
            summary_text=crystal_id, recall_gated=False, quality_tier="neutral",
        ))


async def _write_fact(store, crystal_id, fact_id, dim_idx):
    async with store.session() as s:
        s.add(FactRow(
            id=fact_id, crystal_id=crystal_id, pair_type="question_answer",
            prompt_text=fact_id, claim_text="c", vector=_unit(dim_idx),
        ))


async def _delete_fact(store, fact_id):
    async with store.session() as s:
        row = await s.get(FactRow, fact_id)
        if row is not None:
            await s.delete(row)


def _ids(results):
    return {r[0] for r in results}


@pytest.mark.asyncio
async def test_other_process_sees_writes_and_deletes_through_the_generation(store, customer):
    cid = customer.id
    # Process A (writer): attached. Process B (reader): NOT attached.
    facts_a, routing_a = FactVectorStore(store), VectorStore(store)
    facts_b, routing_b = FactVectorStore(store), VectorStore(store)
    store.attach_indexes(facts_a, routing_a)

    await _write_crystal(store, cid, "cr_1", 0)
    await _write_fact(store, "cr_1", "f_1", 0)
    await store.bank_changed(cid)

    q = np.array(_unit(0), dtype=np.float32)
    assert _ids(await facts_b.search(cid, q, k=5)) == {"f_1"}
    assert [c for c, _ in await routing_b.search(cid, q, k=5, crystal_type="customer:legacy")] == ["cr_1"]

    # A writes a second fact and a second crystal; B must see both on its
    # next search with no notification of any kind.
    await _write_crystal(store, cid, "cr_2", 1)
    await _write_fact(store, "cr_2", "f_2", 1)
    await store.bank_changed(cid)
    assert _ids(await facts_b.search(cid, q, k=5)) == {"f_1", "f_2"}
    assert {c for c, _ in await routing_b.search(cid, q, k=5, crystal_type="customer:legacy")} == {"cr_1", "cr_2"}

    # A deletes f_1; B must stop returning it.
    await _delete_fact(store, "f_1")
    await store.bank_changed(cid)
    assert _ids(await facts_b.search(cid, q, k=5)) == {"f_2"}

    # And A's own attached indexes agree with B's.
    assert _ids(await facts_a.search(cid, q, k=5)) == {"f_2"}


@pytest.mark.asyncio
async def test_incremental_update_stamps_the_generation_and_does_not_reload(store, customer):
    """note_pair_written appends in place; the stamp afterwards must keep
    the cache valid at the new generation, or every in-process write
    would trigger the full reload the incremental path exists to avoid."""
    cid = customer.id
    facts = FactVectorStore(store)
    store.attach_indexes(facts)

    await _write_crystal(store, cid, "cr_s", 2)
    await _write_fact(store, "cr_s", "f_s1", 2)
    await store.bank_changed(cid)
    q = np.array(_unit(2), dtype=np.float32)
    await facts.search(cid, q, k=5)
    bank_before = facts._banks[cid]

    # The incremental path: crystal + fact given, index appends in place.
    crystal = await store.get_crystal("cr_s")
    await _write_fact(store, "cr_s", "f_s2", 3)
    async with store.session() as s:
        fact_row = await s.get(FactRow, "f_s2")
    from crystal_cache.infrastructure.metadata_store import _fact_from_row

    await store.bank_changed(cid, crystal, _fact_from_row(fact_row))
    assert facts._banks[cid] is bank_before  # not dropped
    assert facts._banks[cid].generation == await store.bank_generation(cid)

    results = await facts.search(cid, np.array(_unit(3), dtype=np.float32), k=5)
    assert "f_s2" in _ids(results)
    assert facts._banks[cid] is bank_before  # still the same object: no reload


@pytest.mark.asyncio
async def test_every_bank_mutation_in_the_store_goes_through_bank_changed():
    """Structural: no store code path may notify an index directly."""
    import inspect
    import re

    from crystal_cache.infrastructure import metadata_store as ms
    from crystal_cache.infrastructure import metadata_store_erase_ext as erase
    from crystal_cache.ingestion import document_pipeline
    from crystal_cache.maintenance import promotion_service

    for mod in (ms, erase, document_pipeline, promotion_service):
        src = inspect.getsource(mod)
        direct = re.findall(
            r"(?:vector_store|fact_vector_store|vector_index|_index|idx|cache)\.(?:invalidate|note_pair_written)\(",
            src,
        )
        assert direct == [], f"{mod.__name__}: {direct}"
