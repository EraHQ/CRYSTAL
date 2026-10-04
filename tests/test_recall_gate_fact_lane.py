"""RC-03 (2026-10-04): the recall gate holds on the FACT lane.

"Held out of recall until you approve" was true on the crystal routing
lane and false on the fact lane every MCP read tool uses, so unapproved
assumptions reached customers. Now the fact loader, the key scan, the
Qdrant mirror and the sqlite_vec partition all exclude recall-gated
crystals, and the gate setters notify the indexes (RC-01). This pin
walks an assumption through its life on the real lanes: invisible while
pending, visible once approved, gone once superseded.
"""
import numpy as np
import pytest

from crystal_cache.infrastructure.fact_vector_store import FactVectorStore
from crystal_cache.infrastructure.schema import CrystalRow, FactRow


async def _seed_evidence(store, customer_id, encoder):
    vec = lambda t: [float(x) for x in np.asarray(encoder.encode_native(t)).tolist()]  # noqa: E731
    async with store.session() as s:
        s.add(CrystalRow(id="ev_a", customer_id=customer_id, summary_vector=[],
                         summary_text="Invoices are sent on the 1st"))
        s.add(FactRow(id="evf_a", crystal_id="ev_a", claim_text="Invoices go out on the 1st",
                      prompt_text="Billing|invoice day", pair_type="question_answer",
                      vector=vec("Billing|invoice day")))
        s.add(CrystalRow(id="ev_b", customer_id=customer_id, summary_vector=[],
                         summary_text="Net-30 terms apply"))
        s.add(FactRow(id="evf_b", crystal_id="ev_b", claim_text="Payment terms are net 30",
                      prompt_text="Billing|terms", pair_type="question_answer",
                      vector=vec("Billing|terms")))


def _fact_ids(results):
    return {r[0] for r in results}


@pytest.mark.asyncio
async def test_assumption_is_gated_then_visible_then_superseded_on_every_lane(
    store, customer, semantic_encoder_stub
):
    cid = customer.id
    await _seed_evidence(store, cid, semantic_encoder_stub)
    facts = FactVectorStore(store)
    store.attach_indexes(facts)

    out = await store.create_assumption_crystal(
        cid, statement="Payments are due on the 31st", subject="billing",
        parent_a_id="ev_a", parent_b_id="ev_b", confidence=0.6,
        encoder=semantic_encoder_stub,
    )
    asm = out["crystal_id"]
    asm_fact_ids = {f.id for f in await store.list_facts_for_crystal(asm, include_deactivated=True)}
    assert asm_fact_ids

    async def visible_on_fact_lane():
        q = np.asarray(semantic_encoder_stub.encode_native("payments due"), dtype=np.float32)
        return bool(asm_fact_ids & _fact_ids(await facts.search(cid, q, k=50)))

    async def visible_on_key_scan():
        rows = await store.list_facts_by_key_prefix(cid, key_prefix="Assumptions|", limit=100)
        return any(getattr(f, "crystal_id", None) == asm for f in rows)

    # Pending: invisible on both fact lanes.
    assert not await visible_on_fact_lane()
    assert not await visible_on_key_scan()

    # Approved: visible, with no manual invalidation anywhere.
    assert await store.set_crystal_recall_gate(asm, cid, False)
    assert await visible_on_fact_lane()
    assert await visible_on_key_scan()

    # Superseded: gone again.
    assert await store.supersede_assumptions(cid, [asm], by_id="asm_other") == 1
    assert not await visible_on_fact_lane()
    assert not await visible_on_key_scan()


@pytest.mark.asyncio
async def test_fact_born_under_a_gated_crystal_is_not_appended_in_place(store, customer):
    """note_pair_written must hold a gated crystal's fact out exactly as
    the loader does, or an in-process write could leak it until the
    next reload."""
    from types import SimpleNamespace

    cid = customer.id
    async with store.session() as s:
        s.add(CrystalRow(id="cr_open", customer_id=cid, summary_vector=[], summary_text="open",
                         recall_gated=False))
        s.add(FactRow(id="f_open", crystal_id="cr_open", claim_text="x", prompt_text="k",
                      vector=[1.0, 0.0, 0.0, 0.0]))
    facts = FactVectorStore(store)
    await facts.search(cid, np.array([1.0, 0, 0, 0], dtype=np.float32), k=5)  # load

    gated = SimpleNamespace(id="cr_gated", recall_gated=True)
    fact = SimpleNamespace(id="f_gated", crystal_id="cr_gated", pair_type="question_answer",
                           prompt_text="k2", vector=[0.0, 1.0, 0.0, 0.0])
    await facts.note_pair_written(cid, gated, fact)
    results = await facts.search(cid, np.array([0, 1.0, 0, 0], dtype=np.float32), k=5)
    assert "f_gated" not in _fact_ids(results)
