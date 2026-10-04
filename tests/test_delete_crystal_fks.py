"""RC-02 (2026-10-04): a crystal can be deleted whatever points at it.

Eleven columns carry a foreign key to crystals.id. delete_crystal cleaned
up four of them by hand; on Postgres any of the others (a chat that
routed to the crystal, a sharing grant, a contribution, a diagnostic, an
edit) refused the delete, so a memory a chat had ever used could not be
forgotten. SQLite tests never saw it because foreign keys were off.

This pin turns foreign keys ON for its connections, so SQLite refuses the
same deletes Postgres refuses, then forgets a crystal that is referenced
by every FK-bearing table the schema has, through the real store methods
(delete_crystal and the last-fact branch of delete_fact).
"""
import pytest
from sqlalchemy import event, func, select

from crystal_cache.infrastructure.schema import (
    Base, CrystalAclRow, CrystalRow, FactRow, QueryLogRow,
)


def _fk_tables_to_crystals():
    out = []
    for table in Base.metadata.sorted_tables:
        for col in table.c:
            if any(fk.column.table.name == "crystals" and fk.column.name == "id" for fk in col.foreign_keys):
                if not (table.name == "crystals" and col.name == "id"):
                    out.append((table.name, col.name, col.nullable))
    return out


def test_the_schema_has_the_eleven_references_the_walk_must_cover():
    refs = _fk_tables_to_crystals()
    names = {t for t, _, _ in refs}
    assert {"facts", "crystal_chains", "crystal_edges", "crystal_acls", "query_logs"} <= names
    assert len(refs) >= 11, refs


async def _enable_fks(store):
    """The real store with SQLite foreign keys enforced, so FK violations
    raise exactly as on Postgres. The fixture DB is in-memory, so the
    pragma goes on the live connection (never dispose it) and a listener
    covers any connection the pool opens later."""
    from sqlalchemy import text

    def _on_connect(dbapi_conn, _record):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    event.listen(store.engine.sync_engine, "connect", _on_connect)
    async with store.engine.begin() as conn:
        await conn.execute(text("PRAGMA foreign_keys=ON"))
        on = (await conn.execute(text("PRAGMA foreign_keys"))).scalar()
    assert on == 1, "SQLite did not enable foreign keys"
    return store


async def _seed_referenced_crystal(fk_store, customer_id, crystal_id):
    async with fk_store.session() as s:
        s.add(CrystalRow(id=crystal_id, customer_id=customer_id, summary_vector=[],
                         summary_text=crystal_id))
        s.add(CrystalRow(id=f"{crystal_id}_child", customer_id=customer_id, summary_vector=[],
                         summary_text="child", parent_crystal_id=crystal_id))
        await s.flush()
        s.add(FactRow(id=f"{crystal_id}_f", crystal_id=crystal_id, claim_text="c",
                      prompt_text="k", pair_type="question_answer", vector=[1.0, 0.0]))
        s.add(QueryLogRow(id=f"{crystal_id}_q", customer_id=customer_id, query_text="q",
                          match_type="miss", routed_crystal_id=crystal_id))
        s.add(CrystalAclRow(crystal_id=crystal_id, principal_type="operator",
                            principal_id="op_x", grant="read"))


async def _count(fk_store, model, **where):
    async with fk_store.session() as s:
        stmt = select(func.count()).select_from(model)
        for k, v in where.items():
            stmt = stmt.where(getattr(model, k) == v)
        return (await s.execute(stmt)).scalar_one()


@pytest.mark.asyncio
async def test_delete_crystal_detaches_every_reference_under_enforced_fks(store, customer):
    fk_store = await _enable_fks(store)
    await _seed_referenced_crystal(fk_store, customer.id, "cr_ref")
    assert await _count(fk_store, QueryLogRow, routed_crystal_id="cr_ref") == 1
    assert await _count(fk_store, CrystalAclRow, crystal_id="cr_ref") == 1

    assert await fk_store.delete_crystal("cr_ref", customer.id) is True

    assert await fk_store.get_crystal("cr_ref") is None
    assert await _count(fk_store, FactRow, crystal_id="cr_ref") == 0
    assert await _count(fk_store, CrystalAclRow, crystal_id="cr_ref") == 0
    # The chat log survives with its routing pointer cleared, not deleted.
    assert await _count(fk_store, QueryLogRow, id="cr_ref_q") == 1
    assert await _count(fk_store, QueryLogRow, routed_crystal_id="cr_ref") == 0
    child = await fk_store.get_crystal("cr_ref_child")
    assert child is not None and child.parent_crystal_id is None


@pytest.mark.asyncio
async def test_last_fact_delete_detaches_references_too(store, customer, semantic_encoder_stub):
    fk_store = await _enable_fks(store)
    await _seed_referenced_crystal(fk_store, customer.id, "cr_last")
    out = await fk_store.delete_fact("cr_last_f", customer.id, encoder=semantic_encoder_stub)
    assert out
    assert await fk_store.get_crystal("cr_last") is None
    assert await _count(fk_store, CrystalAclRow, crystal_id="cr_last") == 0
    assert await _count(fk_store, QueryLogRow, routed_crystal_id="cr_last") == 0
