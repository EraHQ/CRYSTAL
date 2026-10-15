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


# ---------------------------------------------------------------------------
# Lockdown PR-8 (Q57): the gate on the lanes the audit named and the
# first pin did not drive — the Qdrant mirror, the sqlite_vec partition,
# the MCP read tools and the HTTP retrieve.
# ---------------------------------------------------------------------------

async def _assumption(store, cid, encoder):
    out = await store.create_assumption_crystal(
        cid, statement="Payments are due on the 31st", subject="billing",
        parent_a_id="ev_a", parent_b_id="ev_b", confidence=0.6,
        encoder=encoder,
    )
    asm = out["crystal_id"]
    fact_ids = {f.id for f in await store.list_facts_for_crystal(asm, include_deactivated=True)}
    assert fact_ids
    return asm, fact_ids


@pytest.mark.asyncio
async def test_the_gate_holds_on_the_qdrant_lane(store, customer, semantic_encoder_stub):
    """The Qdrant mirror excludes gated crystals on load and follows the
    gate setters through the bank generation (RC-01), with no manual
    invalidate anywhere."""
    pytest.importorskip("qdrant_client")
    from qdrant_client import AsyncQdrantClient

    from crystal_cache.infrastructure.qdrant_vector_index import QdrantVectorIndex

    cid = customer.id
    await _seed_evidence(store, cid, semantic_encoder_stub)
    qvi = QdrantVectorIndex(client=AsyncQdrantClient(location=":memory:"), metadata_store=store)
    store.attach_indexes(qvi)
    asm, asm_fact_ids = await _assumption(store, cid, semantic_encoder_stub)
    q = np.asarray(semantic_encoder_stub.encode_native("payments due"), dtype=np.float32)

    async def visible():
        got = await qvi.search_facts(customer_id=cid, query_vector=q, k=50)
        return bool(asm_fact_ids & _fact_ids(got))

    assert not await visible()
    assert await store.set_crystal_recall_gate(asm, cid, False)
    assert await visible()
    assert await store.supersede_assumptions(cid, [asm], by_id="asm_other") == 1
    assert not await visible()


@pytest.mark.asyncio
async def test_the_gate_holds_on_the_sqlite_vec_lane(tmp_path, semantic_encoder_stub):
    """The sqlite_vec partition (a file-backed store, as that index needs)."""
    pytest.importorskip("sqlite_vec")
    from crystal_cache.config import Settings
    from crystal_cache.infrastructure import MetadataStore
    from crystal_cache.infrastructure.sqlite_vec_index import SqliteVecIndex

    settings = Settings(database_url=f"sqlite+aiosqlite:///{tmp_path.as_posix()}/gate.db")
    store = MetadataStore(settings_override=settings)
    await store.init()
    await store._seed_legacy_crystal_types_for_tests()
    try:
        customer = await store.create_customer(provider="anthropic", model_id="claude-x", api_key_ref="ref")
        cid = customer.id
        await _seed_evidence(store, cid, semantic_encoder_stub)
        index = SqliteVecIndex(metadata_store=store)
        store.attach_indexes(index)
        asm, asm_fact_ids = await _assumption(store, cid, semantic_encoder_stub)
        q = np.asarray(semantic_encoder_stub.encode_native("payments due"), dtype=np.float32)

        async def visible():
            got = await index.search_facts(customer_id=cid, query_vector=q, k=50)
            return bool(asm_fact_ids & _fact_ids(got))

        assert not await visible()
        assert await store.set_crystal_recall_gate(asm, cid, False)
        assert await visible()
        assert await store.supersede_assumptions(cid, [asm], by_id="asm_other") == 1
        assert not await visible()
    finally:
        await store.dispose()


_HTTP_QUERY = "payments due on the 31st"


async def _routable_gated_crystal(store, cid, encoder, crystal_id):
    """A crystal born recall-gated on the lane POST /v1/retrieve reads:
    the routing lane (crystal routing vectors, type-scoped). Its routing
    vector is the stub's encoding of the exact query text, so the real
    pipeline routes to it with cosine 1.0 the moment the gate opens.
    Assumptions never sit on this lane (type `assumption`, no routing
    vector) — they reach a customer through the fact lane the MCP tools
    read — so the HTTP half of this pin uses a crystal the lane can
    hold."""
    hdc = [float(x) for x in np.asarray(encoder.encode(_HTTP_QUERY)).tolist()]
    native = [float(x) for x in np.asarray(encoder.encode_native(_HTTP_QUERY)).tolist()]
    async with store.session() as s:
        s.add(CrystalRow(
            id=crystal_id, customer_id=cid, crystal_type="customer:legacy",
            summary_vector=hdc, routing_vector=hdc,
            summary_text="Payments are due on the 31st",
            recall_gated=True, quality_tier="neutral",
        ))
        s.add(FactRow(
            id=f"{crystal_id}_f", crystal_id=crystal_id, pair_type="question_answer",
            prompt_text="Billing|due day", claim_text="Payments are due on the 31st",
            vector=native,
        ))


@pytest.mark.asyncio
async def test_the_gate_holds_on_the_mcp_and_http_read_tools(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store, monkeypatch,
):
    """memory_recall / memory_keys (the fact lane) and POST /v1/retrieve
    (the routing lane) — the surfaces a customer's client actually reads
    through — never show a recall-gated crystal and show it once the
    gate opens, with no manual invalidation anywhere (RC-01: the gate
    setter notifies the indexes)."""
    from httpx import ASGITransport, AsyncClient

    from crystal_cache.agent import mcp_server
    from crystal_cache.agent.tools.retrievers import set_tool_state
    try:
        from tests.test_endpoint_smoke import _build_app
    except ModuleNotFoundError:
        from test_endpoint_smoke import _build_app

    from crystal_cache.infrastructure.vector_index import InMemoryVectorIndex

    cid = customer.id
    await _seed_evidence(store, cid, semantic_encoder_stub)
    store.attach_indexes(fact_vector_store, vector_store)
    # What runtime.py builds on the default backend; the MCP state and the
    # HTTP app (via _build_app) carry one each, as production does.
    vector_index = InMemoryVectorIndex(
        fact_store=fact_vector_store, vector_store=vector_store, metadata_store=store,
    )
    state = {"store": store, "encoder": semantic_encoder_stub, "vector_store": vector_store,
             "fact_vector_store": fact_vector_store, "vector_index": vector_index}
    monkeypatch.setattr(mcp_server, "_get_state", lambda: state)
    set_tool_state(state)
    token = mcp_server._current_customer_id.set(cid)
    app = _build_app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    headers = {"Authorization": f"Bearer {customer.api_key}"}
    try:
        asm, asm_fact_ids = await _assumption(store, cid, semantic_encoder_stub)
        await _routable_gated_crystal(store, cid, semantic_encoder_stub, "cr_http")

        async def mcp_sees():
            out = await mcp_server.memory_recall.fn(query="payments due on the 31st", k=50)
            recall_ids = {f["fact_id"] for c in out["crystals"] for f in c["facts"]}
            keys = await mcp_server.memory_keys.fn(key_prefix="Assumptions|")
            key_ids = {f["crystal_id"] for f in keys["findings"]}
            return bool(asm_fact_ids & recall_ids), asm in key_ids

        async def http_sees():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.post("/v1/retrieve", headers=headers,
                                 json={"query": _HTTP_QUERY, "k": 20})
            assert r.status_code == 200, r.text[:300]
            return "cr_http" in r.json()["matched_crystal_ids"]

        # Gated: invisible on both surfaces.
        assert await mcp_sees() == (False, False)
        assert not await http_sees()
        # Gate opened: visible on both, no invalidate called by anyone.
        assert await store.set_crystal_recall_gate(asm, cid, False)
        assert await store.set_crystal_recall_gate("cr_http", cid, False)
        assert await mcp_sees() == (True, True)
        assert await http_sees()
        # Superseded (assumption) / re-gated (crystal): gone again.
        assert await store.supersede_assumptions(cid, [asm], by_id="asm_other") == 1
        assert await store.set_crystal_recall_gate("cr_http", cid, True)
        assert await mcp_sees() == (False, False)
        assert not await http_sees()
    finally:
        mcp_server._current_customer_id.reset(token)
