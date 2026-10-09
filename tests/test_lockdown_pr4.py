"""Lockdown PR-4 (2026-10-09): the Part A items the verify report reopened.

  - RC-16: key presence is the approve switch; an explicit empty item
    selection with chunks kept writes zero items and only the chunks;
  - Q54: forgetting the last crystal born from an upload scrubs the
    upload's text, chunks and items on EVERY delete path (the scrub lives
    in the two delete primitives), matched by the upload id recorded on
    the crystal; a row still being written is left alone;
  - Q58: the spend view gives tenants a percent and platform admins the
    dollars;
  - RC-01 D-A1-1: a generation stamp never hides another process's write;
  - RC-01 D-A1-2: a fresh Qdrant process clears stale points on first load;
  - RC-08: the stale-claim window starts when a row's own work starts;
  - RC-13: only an integrity error on the event id is "already recorded";
  - RC-14: one shutdown budget inside Cloud Run's window (pinned in
    test_no_sync_seam_in_async.py);
  - class 3 / class 6: pinned in test_every_call_ledgered.py and
    test_mcp_contract.py.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from crystal_cache.agent import mcp_server
from crystal_cache.endpoints import admin as admin_mod
from crystal_cache.endpoints import documents as docs_mod
from crystal_cache.endpoints import sdk as sdk_mod
from crystal_cache.infrastructure.schema import CrystalRow, DocumentUploadRow, FactRow
from crystal_cache.ingress.schema import ImportRequest
from crystal_cache.workers import crystallization as wk


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class _Req:
    def __init__(self, body=None, *, bearer=None, path="/v1/x", app_state=None, method="GET"):
        self._body = body if body is not None else {}
        self.headers = {"content-type": "application/json"}
        if bearer:
            self.headers["authorization"] = f"Bearer {bearer}"
        self.url = SimpleNamespace(path=path)
        self.method = method
        self.app = SimpleNamespace(state=app_state or SimpleNamespace())
        self.state = SimpleNamespace()

    async def json(self):
        return self._body


def _state(encoder, vector_store, fact_vector_store):
    return SimpleNamespace(
        prompt_encoder=encoder, vector_store=vector_store,
        fact_vector_store=fact_vector_store, vector_index=None,
    )


def _chunk(i, text):
    return {"index": i, "label": f"Section {i}", "text": text, "char_count": len(text),
            "locator": f"Section {i}", "subject": None, "doc_type": "general",
            "injection_hits": []}


def _mcp(monkeypatch, store, encoder, vector_store, fact_vector_store):
    monkeypatch.setattr(mcp_server, "_get_state", lambda: {
        "store": store, "encoder": encoder, "vector_store": vector_store,
        "fact_vector_store": fact_vector_store, "vector_index": None,
    })


def _tool(name):
    fn = getattr(mcp_server, name)
    return getattr(fn, "fn", fn)


class _As:
    def __init__(self, cid, operator=None):
        self.cid, self.op = cid, operator

    def __enter__(self):
        self._t = mcp_server._current_customer_id.set(self.cid)
        self._o = mcp_server.set_current_operator(self.op)
        return self

    def __exit__(self, *a):
        mcp_server.reset_current_operator(self._o)
        mcp_server._current_customer_id.reset(self._t)


async def _crystallized_doc(store, customer, encoder, vector_store, fact_vector_store,
                            label, *, items=1, chunks=1):
    """A document written through the real write leg: returns the row and
    the crystal ids it produced (file crystal + item crystals)."""
    doc = await store.create_document_upload(customer.id, label, "THE RAW TEXT")
    await store.mark_document_review_ready(
        doc.id, detected_type="general",
        content_chunks=[_chunk(i, f"{label} chunk {i}") for i in range(chunks)],
        extracted_items=[{"key": f"{label} key {i}", "value": f"value {i}", "type": "fact",
                          "sparse_key": f"Docs|{label}|{i}"} for i in range(items)],
        items_extracted_count=items,
    )
    won = await store.save_approval_edits_and_mark_crystallizing(
        doc.id,
        items=[{"key": f"{label} key {i}", "value": f"value {i}", "type": "fact",
                "sparse_key": f"Docs|{label}|{i}"} for i in range(items)],
        content_chunks=[_chunk(i, f"{label} chunk {i}") for i in range(chunks)],
        approved_via="console",
    )
    assert won
    await wk.write_approved_document(
        store=store, encoder=encoder, vector_store=vector_store,
        fact_vector_store=fact_vector_store, document_id=doc.id, customer_id=customer.id,
    )
    row = await store.get_document_upload(doc.id, customer.id)
    assert row.status == "crystallized" and row.text == "THE RAW TEXT"
    assert row.content_chunks and row.crystal_ids
    assert len(row.crystal_ids) == items + 1
    for cid in row.crystal_ids:
        assert (await store.get_crystal(cid)).source_document_id == doc.id
    return row, list(row.crystal_ids)


async def _scrubbed(store, doc_id):
    async with store.session() as s:
        up = await s.get(DocumentUploadRow, doc_id)
    return (up is not None and up.text == "" and up.content_chunks is None
            and up.extracted_items is None and up.status == "forgotten")


async def _intact(store, doc_id):
    async with store.session() as s:
        up = await s.get(DocumentUploadRow, doc_id)
    return up is not None and up.text == "THE RAW TEXT" and up.status == "crystallized"


# ---------------------------------------------------------------------------
# RC-16
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approve_key_presence_is_the_switch(store, customer, monkeypatch):
    from crystal_cache.config import settings

    monkeypatch.setattr(settings, "ingest_mode", "worker")

    async def _doc():
        doc = await store.create_document_upload(customer.id, "notes.txt", "raw")
        await store.mark_document_review_ready(
            doc.id, detected_type="general",
            content_chunks=[_chunk(0, "alpha"), _chunk(1, "beta")],
            extracted_items=[{"key": "k", "value": "v", "type": "fact"}],
            items_extracted_count=1,
        )
        return doc.id

    # The reopened case: no items, chunks kept -> zero items, both chunks.
    d = await _doc()
    out = await docs_mod.sdk_approve_document(d, _Req({"items": [], "include_chunks": True}), customer, store)
    assert out.status_code == 202
    row = await store.get_document_upload(d, customer.id)
    assert row.status == "approved" and row.extracted_items == []
    assert [c["index"] for c in row.content_chunks] == [0, 1]

    # include_chunks false drops every chunk; the items stay.
    d = await _doc()
    await docs_mod.sdk_approve_document(
        d, _Req({"items": [{"key": "k", "value": "v2", "type": "fact"}], "include_chunks": False}),
        customer, store,
    )
    row = await store.get_document_upload(d, customer.id)
    assert row.extracted_items == [{"key": "k", "value": "v2", "type": "fact"}]
    assert row.content_chunks == []

    # A chosen chunk subset with the items as extracted.
    d = await _doc()
    await docs_mod.sdk_approve_document(d, _Req({"content_chunks": [{"index": 1}]}), customer, store)
    row = await store.get_document_upload(d, customer.id)
    assert row.extracted_items == [{"key": "k", "value": "v", "type": "fact"}]
    assert [c["index"] for c in row.content_chunks] == [1]

    # Nothing selected at all is still refused; omitted keys still mean "as extracted".
    d = await _doc()
    with pytest.raises(HTTPException) as e:
        await docs_mod.sdk_approve_document(d, _Req({"items": [], "include_chunks": False}), customer, store)
    assert e.value.status_code == 400
    with pytest.raises(HTTPException) as e:
        await docs_mod.sdk_approve_document(d, _Req({"items": [], "content_chunks": []}), customer, store)
    assert e.value.status_code == 400
    assert (await store.get_document_upload(d, customer.id)).status == "review"
    out = await docs_mod.sdk_approve_document(d, _Req({}), customer, store)
    assert out.status_code == 202
    row = await store.get_document_upload(d, customer.id)
    assert len(row.extracted_items) == 1 and len(row.content_chunks) == 2


# ---------------------------------------------------------------------------
# Q54: the scrub, on every path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_forget_scrubs_the_upload_on_every_delete_path(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store, monkeypatch
):
    _mcp(monkeypatch, store, semantic_encoder_stub, vector_store, fact_vector_store)
    admin = await store.ensure_default_admin(customer.id)
    app_state = _state(semantic_encoder_stub, vector_store, fact_vector_store)
    mk = lambda label, **kw: _crystallized_doc(  # noqa: E731
        store, customer, semantic_encoder_stub, vector_store, fact_vector_store, label, **kw)

    # 1. MCP memory_forget by crystal: the row survives the first delete
    #    (one crystal from it remains) and is scrubbed by the last.
    row, ids = await mk("mcp-crystal", items=1, chunks=1)
    with _As(customer.id, admin):
        assert (await _tool("memory_forget")(crystal_id=ids[0]))["deleted"] is True
        assert await _intact(store, row.id)
        assert (await _tool("memory_forget")(crystal_id=ids[1]))["deleted"] is True
    assert await _scrubbed(store, row.id)

    # 2. MCP memory_forget by fact: the last fact takes the crystal with it.
    row, ids = await mk("mcp-fact", items=0, chunks=1)
    (fact,) = await store.list_facts_for_crystal(ids[0])
    with _As(customer.id, admin):
        assert (await _tool("memory_forget")(fact_id=fact.id))["deleted"] is True
    assert await store.get_crystal(ids[0]) is None
    assert await _scrubbed(store, row.id)

    # 3. MCP forget.
    row, ids = await mk("mcp-forget", items=0, chunks=1)
    with _As(customer.id, admin):
        assert (await _tool("forget")(ids[0]))["retired"] is True
    assert await _scrubbed(store, row.id)

    # 4. HTTP DELETE /v1/crystals/{id} and /v1/facts/{id}.
    row, ids = await mk("http", items=1, chunks=1)
    await sdk_mod.sdk_crystal_delete(ids[0], _Req(app_state=app_state), customer, store)
    assert await _intact(store, row.id)
    (fact,) = await store.list_facts_for_crystal(ids[1])
    await sdk_mod.sdk_fact_delete(fact.id, _Req(app_state=app_state), customer, store)
    assert await _scrubbed(store, row.id)

    # 5. The console deletes (platform-admin seat: no tenant pin).
    row, ids = await mk("console", items=1, chunks=1)
    await admin_mod.admin_delete_crystal(_Req(app_state=app_state), ids[0], store)
    assert await _intact(store, row.id)
    (fact,) = await store.list_facts_for_crystal(ids[1])
    await admin_mod.admin_retire_fact(_Req(app_state=app_state), ids[1], fact.id, store)
    assert await _scrubbed(store, row.id)

    # 6. HTTP import wipe and MCP import wipe.
    async def _open(**kw):
        return None

    monkeypatch.setattr(mcp_server, "_write_admission_block", _open)
    row, ids = await mk("wipe-http", items=1, chunks=1)
    await sdk_mod.sdk_import(
        ImportRequest(records=[{"key": "k", "value": "v"}], wipe=True),
        _Req(app_state=app_state), (customer, admin), store,
    )
    assert await _scrubbed(store, row.id)
    row, ids = await mk("wipe-mcp", items=1, chunks=1)
    with _As(customer.id, admin):
        out = await _tool("memory_import")(records=[{"key": "k", "value": "v"}], wipe=True)
    assert out.get("errors") == 0, out
    assert await _scrubbed(store, row.id)

    # 7. The pipeline's own replace: a changed re-upload retires the old
    #    version's crystals, so the old row is scrubbed too.
    old, old_ids = await mk("replaced.txt", items=0, chunks=1)
    new = await store.create_document_upload(customer.id, "replaced.txt", "THE RAW TEXT")
    await store.mark_document_review_ready(
        new.id, detected_type="general", content_chunks=[_chunk(0, "a changed body")],
        extracted_items=[], items_extracted_count=0,
    )
    await store.save_approval_edits_and_mark_crystallizing(
        new.id, items=[], content_chunks=[_chunk(0, "a changed body")], approved_via="console",
    )
    await wk.write_approved_document(
        store=store, encoder=semantic_encoder_stub, vector_store=vector_store,
        fact_vector_store=fact_vector_store, document_id=new.id, customer_id=customer.id,
    )
    assert await store.get_crystal(old_ids[0]) is None
    assert await _scrubbed(store, old.id)
    assert await _intact(store, new.id)


@pytest.mark.asyncio
async def test_scrub_matches_by_upload_id_and_falls_back_to_the_base_uri(store, customer):
    cid = customer.id
    async with store.session() as s:
        # A row still being written: never scrubbed, whatever its crystals do.
        s.add(DocumentUploadRow(id="up_live", customer_id=cid, label="live", text="LIVE",
                                status="crystallizing", content_chunks=[{"index": 0}],
                                extracted_items=[{"key": "k"}]))
        # A pre-column row whose crystals carved fragments (#sheet=): the
        # old scrub never matched them; the base URI does.
        s.add(DocumentUploadRow(id="up_old", customer_id=cid, label="old", text="OLD",
                                status="crystallized", source_uri="upload://up_old",
                                content_chunks=[{"index": 0}], extracted_items=[{"key": "k"}]))
        s.add(CrystalRow(id="cr_sheet", customer_id=cid, summary_vector=[],
                         source_uri="upload://up_old#sheet=Q1"))
    assert await store.scrub_upload_if_orphaned(cid, document_id="up_live") == 0
    async with store.session() as s:
        assert (await s.get(DocumentUploadRow, "up_live")).text == "LIVE"
    # The fragment crystal still exists: nothing scrubbed.
    assert await store.scrub_upload_if_orphaned(cid, source_uri="upload://up_old#sheet=Q1") == 0
    async with store.session() as s:
        await s.delete(await s.get(CrystalRow, "cr_sheet"))
    assert await store.scrub_upload_if_orphaned(cid, source_uri="upload://up_old#sheet=Q1") == 1
    async with store.session() as s:
        up = await s.get(DocumentUploadRow, "up_old")
    assert up.text == "" and up.content_chunks is None and up.extracted_items is None
    assert up.status == "forgotten"
    # A forgotten row still lists (the status is in the contract).
    assert (await store.get_document_upload("up_old", cid)).status == "forgotten"


# ---------------------------------------------------------------------------
# Q58
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spend_view_gives_tenants_a_percent_and_admins_the_dollars(store, customer, monkeypatch):
    from crystal_cache.config import Settings
    from crystal_cache.ingress import auth as auth_mod

    settings = Settings(environment="development", admin_api_key="adm-secret",
                        api_key_pepper="", firebase_project_id="proj-x")
    monkeypatch.setattr(auth_mod, "get_settings", lambda: settings)
    await store.set_customer_inference_mode(customer.id, "managed")
    path = f"/admin/api/customers/{customer.id}/spend"

    tenant = await admin_mod.get_customer_spend(
        customer.id, _Req(bearer=customer.api_key, path=path), store,
    )
    assert "managed_capacity_used_percent" in tenant
    assert 0 <= tenant["managed_capacity_used_percent"] <= 100
    assert not any("usd" in k for k in tenant), tenant
    assert "totals" not in tenant

    admin = await admin_mod.get_customer_spend(
        customer.id, _Req(bearer="adm-secret", path=path), store,
    )
    assert "managed_month_to_date_micro_usd" in admin and "managed_monthly_cap_micro_usd" in admin
    assert "totals" in admin
    assert admin["managed_capacity_used_percent"] == tenant["managed_capacity_used_percent"]


# ---------------------------------------------------------------------------
# RC-01 D-A1-1 and D-A1-2
# ---------------------------------------------------------------------------

D = 8


def _unit(i):
    v = [0.0] * D
    v[i % D] = 1.0
    return v


async def _raw_crystal(store, cid, crystal_id, dim):
    async with store.session() as s:
        s.add(CrystalRow(id=crystal_id, customer_id=cid, crystal_type="customer:legacy",
                         summary_vector=_unit(dim), routing_vector=_unit(dim),
                         summary_text=crystal_id, recall_gated=False, quality_tier="neutral"))


async def _raw_fact(store, crystal_id, fact_id, dim):
    async with store.session() as s:
        s.add(FactRow(id=fact_id, crystal_id=crystal_id, pair_type="question_answer",
                      prompt_text=fact_id, claim_text="c", vector=_unit(dim)))


@pytest.mark.asyncio
async def test_a_stamp_never_hides_another_processes_write(store, customer):
    """API cache at G; the worker writes G+1 behind its back; the API's own
    incremental write at G+2 must reload and surface the worker's fact."""
    from crystal_cache.infrastructure.fact_vector_store import FactVectorStore
    from crystal_cache.infrastructure.metadata_store import _fact_from_row
    from crystal_cache.infrastructure.vector_store import VectorStore

    cid = customer.id
    facts, routing = FactVectorStore(store), VectorStore(store)
    store.attach_indexes(facts, routing)
    await _raw_crystal(store, cid, "cr_1", 0)
    await _raw_fact(store, "cr_1", "f_1", 0)
    await store.bank_changed(cid)
    q = np.array(_unit(0), dtype=np.float32)
    assert {r[0] for r in await facts.search(cid, q, k=5)} == {"f_1"}
    assert [c for c, _ in await routing.search(cid, q, k=5, crystal_type="customer:legacy")] == ["cr_1"]
    g = await store.bank_generation(cid)

    # Another process: a row and a bump this process is never told about.
    await _raw_crystal(store, cid, "cr_worker", 1)
    await _raw_fact(store, "cr_worker", "f_worker", 1)
    assert await store.bump_bank_generation(cid) == g + 1

    # This process's incremental write (the path that used to stamp G+2).
    await _raw_fact(store, "cr_1", "f_api", 2)
    async with store.session() as s:
        fact_row = await s.get(FactRow, "f_api")
    await store.bank_changed(cid, await store.get_crystal("cr_1"), _fact_from_row(fact_row))
    assert await store.bank_generation(cid) == g + 2

    assert {r[0] for r in await facts.search(cid, q, k=5)} == {"f_1", "f_worker", "f_api"}
    assert {c for c, _ in await routing.search(cid, q, k=5, crystal_type="customer:legacy")} == {"cr_1", "cr_worker"}
    assert facts._banks[cid].generation == g + 2


@pytest.mark.asyncio
async def test_stamp_at_exactly_the_previous_generation_keeps_the_cache(store, customer):
    from crystal_cache.infrastructure.fact_vector_store import FactVectorStore
    from crystal_cache.infrastructure.metadata_store import _fact_from_row

    cid = customer.id
    facts = FactVectorStore(store)
    store.attach_indexes(facts)
    await _raw_crystal(store, cid, "cr_s", 2)
    await _raw_fact(store, "cr_s", "f_s1", 2)
    await store.bank_changed(cid)
    await facts.search(cid, np.array(_unit(2), dtype=np.float32), k=5)
    before = facts._banks[cid]
    await _raw_fact(store, "cr_s", "f_s2", 3)
    async with store.session() as s:
        fact_row = await s.get(FactRow, "f_s2")
    await store.bank_changed(cid, await store.get_crystal("cr_s"), _fact_from_row(fact_row))
    assert facts._banks[cid] is before  # the in-place append stands


@pytest.mark.asyncio
async def test_fresh_qdrant_process_clears_stale_points_on_first_load(store, customer):
    pytest.importorskip("qdrant_client")
    from qdrant_client import AsyncQdrantClient

    from crystal_cache.infrastructure.qdrant_vector_index import QdrantVectorIndex

    cid = customer.id
    client = AsyncQdrantClient(location=":memory:")  # the shared, persistent collection
    await _raw_crystal(store, cid, "cr_q", 0)
    await _raw_fact(store, "cr_q", "f_keep", 0)
    await _raw_fact(store, "cr_q", "f_gone", 1)
    await store.bank_changed(cid)

    first = QdrantVectorIndex(client=client, metadata_store=store)  # process 1
    q = np.array(_unit(1), dtype=np.float32)
    assert "f_gone" in {r[0] for r in await first.search_facts(customer_id=cid, query_vector=q, k=5)}

    # The fact is deleted and the bank bumped while process 1 is gone.
    async with store.session() as s:
        await s.delete(await s.get(FactRow, "f_gone"))
    await store.bank_changed(cid)

    second = QdrantVectorIndex(client=client, metadata_store=store)  # a fresh process
    ids = {r[0] for r in await second.search_facts(customer_id=cid, query_vector=q, k=5)}
    assert "f_gone" not in ids and "f_keep" in ids


# ---------------------------------------------------------------------------
# RC-08 and RC-13
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_claim_is_refreshed_when_a_rows_work_starts(store, customer, monkeypatch):
    doc = await store.create_document_upload(customer.id, "slow.txt", "raw")
    (claimed,) = await store.claim_pending_documents_batch(limit=1)
    assert claimed.id == doc.id
    stale = datetime.now(timezone.utc) - timedelta(minutes=45)
    async with store.session() as s:
        (await s.get(DocumentUploadRow, doc.id)).claimed_at = stale  # waited on the semaphore
    assert await store.touch_document_claim(doc.id) is True
    async with store.session() as s:
        row = await s.get(DocumentUploadRow, doc.id)
    fresh = row.claimed_at if row.claimed_at.tzinfo else row.claimed_at.replace(tzinfo=timezone.utc)
    assert fresh > stale + timedelta(minutes=40)
    # Nothing to reclaim now: the row's own work just started.
    assert await store.reclaim_stale_document_claims(older_than_minutes=30) == 0
    assert (await store.get_document_upload(doc.id, customer.id)).status == "crystallizing"
    # Not crystallizing -> no touch.
    await store.mark_document_error(doc.id, "boom")
    assert await store.touch_document_claim(doc.id) is False


@pytest.mark.asyncio
async def test_billing_event_duplicate_is_only_an_integrity_error(store, customer, monkeypatch):
    assert await store.record_billing_event("evt_1", customer_id=customer.id, event_type="x", created=1) is True
    assert await store.record_billing_event("evt_1", customer_id=customer.id, event_type="x", created=1) is False

    class _Boom(RuntimeError):
        pass

    real_session = store.session

    class _Broken:
        def __init__(self, inner):
            self._inner = inner

        async def __aenter__(self):
            self._s = await self._inner.__aenter__()
            outer = self

            class _S:
                def __getattr__(self, name):
                    return getattr(outer._s, name)

                async def commit(self):
                    raise _Boom("db down")

            return _S()

        async def __aexit__(self, *a):
            return await self._inner.__aexit__(*a)

    monkeypatch.setattr(store, "session", lambda: _Broken(real_session()))
    with pytest.raises(_Boom):
        await store.record_billing_event("evt_2", customer_id=customer.id, event_type="x", created=2)
