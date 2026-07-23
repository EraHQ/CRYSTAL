"""Gate G slice 1 (2026-07-22): the source-schema registry + schema
fingerprinting + awaiting_schema parking + the watch-event feed.

Design record: G-Q1=B (record-window fragments — G2 territory),
G-Q2=A (the status column IS the review queue), G-Q3=A
(awaiting_schema parking on document_uploads, released by approval in
one update), G-Q4=A (durable watch events). C5's core contract pinned
here: the fingerprint ignores values, order, and array length, and
notices structure and types.
"""

from __future__ import annotations

import pytest

from crystal_cache.ingestion.schema_hash import (
    parse_json_source,
    schema_hash,
    schema_key_paths,
)
from crystal_cache.infrastructure.metadata_store_schema_ext import (
    STATUS_AWAITING_SCHEMA,
    STATUS_SCHEMA_REJECTED,
)


# ---------------------------------------------------------------------------
# The fingerprint (C5)
# ---------------------------------------------------------------------------

def test_hash_ignores_values_order_and_length():
    a = [{"id": 1, "name": "x", "tags": ["a"]},
         {"id": 2, "name": "y", "tags": []}]
    b = [{"name": "DIFFERENT", "id": 999, "tags": ["z", "w", "q"]}]
    assert schema_hash(a) == schema_hash(b)
    assert schema_hash(a) == schema_hash(list(reversed(a)))


def test_hash_notices_structure_and_types():
    a = [{"id": 1, "name": "x", "tags": ["a"]}]
    missing_field = [{"id": 1, "name": "x"}]
    type_changed = [{"id": "1", "name": "x", "tags": ["a"]}]
    assert schema_hash(a) != schema_hash(missing_field)
    assert schema_hash(a) != schema_hash(type_changed)


def test_optional_fields_union_across_sample():
    e = [{"id": 1}, {"id": 2, "nick": "z"}]
    assert "[].nick:string" in schema_key_paths(e)


def test_root_shape_and_json_types():
    obj = {"id": 1, "name": "x"}
    assert schema_hash(obj) != schema_hash([obj])
    assert "[].ok:boolean" in schema_key_paths([{"ok": True}])
    assert "[].n:number" in schema_key_paths([{"n": 1}])
    assert "[].f:number" in schema_key_paths([{"f": 1.5}])
    nested = {"user": {"address": {"city": "Raleigh"}}}
    assert "user.address.city:string" in schema_key_paths(nested)


def test_jsonl_parses_to_array_and_hashes_identically():
    a = [{"id": 1, "name": "x", "tags": ["a"]},
         {"id": 2, "name": "y", "tags": []}]
    jl = '{"id": 1, "name": "x", "tags": ["a"]}\n' \
         '{"id": 2, "name": "y", "tags": []}\n'
    payload, shape = parse_json_source(jl)
    assert shape == "array"
    assert schema_hash(payload) == schema_hash(a)


def test_parse_rejects_empty_and_garbage():
    with pytest.raises(ValueError):
        parse_json_source("   ")
    with pytest.raises(Exception):
        parse_json_source("not json at all {{{")


# ---------------------------------------------------------------------------
# The registry (G-Q2=A: status IS the queue)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_proposal_lifecycle_and_idempotent_first_contact(
    store, customer,
):
    proposal = await store.create_source_schema_proposal(
        customer_id=customer.id,
        schema_hash="hash_a",
        mapping={"[].name": {"role": "fact_key"}},
        sample=[{"name": "x"}],
    )
    assert proposal.status == "proposed"
    assert proposal.id.startswith("schema_")

    # Re-arrival of the same shape returns the SAME row — never a
    # second proposal ("one judgment per shape, ever").
    again = await store.create_source_schema_proposal(
        customer_id=customer.id,
        schema_hash="hash_a",
        mapping={"different": "mapping"},
        sample=[],
    )
    assert again.id == proposal.id
    assert again.mapping == proposal.mapping

    got = await store.get_source_schema(customer.id, "hash_a")
    assert got is not None and got.id == proposal.id

    listed = await store.list_source_schemas(customer.id, status="proposed")
    assert [s.id for s in listed] == [proposal.id]


@pytest.mark.asyncio
async def test_approval_releases_parked_documents(store, customer):
    proposal = await store.create_source_schema_proposal(
        customer_id=customer.id, schema_hash="hash_b",
        mapping={}, sample=[],
    )
    # Park two documents against the shape (G-Q3=A).
    d1 = await store.create_document_upload(
        customer.id, label="export-1.json", text='[{"a": 1}]',
    )
    d2 = await store.create_document_upload(
        customer.id, label="export-2.json", text='[{"a": 2}]',
    )
    await store.park_document_for_schema(d1.id, "hash_b")
    await store.park_document_for_schema(d2.id, "hash_b")
    parked1 = await store.get_document_upload(d1.id, customer.id)
    assert parked1.status == STATUS_AWAITING_SCHEMA

    released = await store.approve_source_schema(proposal.id)
    assert released == 2
    assert (await store.get_source_schema(customer.id, "hash_b")).status \
        == "approved"
    assert (await store.get_document_upload(d1.id, customer.id)).status \
        == "pending"
    assert (await store.get_document_upload(d2.id, customer.id)).status \
        == "pending"


@pytest.mark.asyncio
async def test_rejection_parks_terminally(store, customer):
    proposal = await store.create_source_schema_proposal(
        customer_id=customer.id, schema_hash="hash_c",
        mapping={}, sample=[],
    )
    d = await store.create_document_upload(
        customer.id, label="feed.json", text='[{"z": 1}]',
    )
    await store.park_document_for_schema(d.id, "hash_c")

    parked = await store.reject_source_schema(proposal.id)
    assert parked == 1
    assert (await store.get_source_schema(customer.id, "hash_c")).status \
        == "rejected"
    assert (await store.get_document_upload(d.id, customer.id)).status \
        == STATUS_SCHEMA_REJECTED


@pytest.mark.asyncio
async def test_mapping_edit_is_forward_only_metadata(store, customer):
    proposal = await store.create_source_schema_proposal(
        customer_id=customer.id, schema_hash="hash_d",
        mapping={"v": 1}, sample=[],
    )
    await store.approve_source_schema(proposal.id)
    await store.update_source_schema_mapping(proposal.id, {"v": 2})
    got = await store.get_source_schema(customer.id, "hash_d")
    assert got.mapping == {"v": 2}
    assert got.status == "approved"     # edits never demote


# ---------------------------------------------------------------------------
# Watch events (G-Q4=A)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_g3_admin_read_methods(store, customer):
    """G3 surface reads: tenancy-checked by-id fetch, parked counts
    per shape, and the label lookup that names a shape card."""
    proposal = await store.create_source_schema_proposal(
        customer_id=customer.id, schema_hash="hash_g3",
        mapping={}, sample=[],
    )
    # Tenancy: right id + wrong customer = None.
    got = await store.get_source_schema_by_id(proposal.id, customer.id)
    assert got is not None and got.id == proposal.id
    assert await store.get_source_schema_by_id(
        proposal.id, "cus_other",
    ) is None

    d1 = await store.create_document_upload(
        customer.id, label="exports/staff.json", text="[]",
    )
    d2 = await store.create_document_upload(
        customer.id, label="exports/staff-2.json", text="[]",
    )
    await store.park_document_for_schema(d1.id, "hash_g3")
    await store.park_document_for_schema(d2.id, "hash_g3")

    counts = await store.parked_counts_by_schema(customer.id)
    assert counts == {"hash_g3": 2}

    # Newest upload carrying the hash names the card.
    label = await store.label_for_schema_hash(customer.id, "hash_g3")
    assert label == "exports/staff-2.json"

    # Released docs keep the hash: the label survives approval.
    await store.approve_source_schema(proposal.id)
    assert await store.parked_counts_by_schema(customer.id) == {}
    assert await store.label_for_schema_hash(
        customer.id, "hash_g3",
    ) == "exports/staff-2.json"


@pytest.mark.asyncio
async def test_watch_event_feed_newest_first(store, customer):
    w = await store.create_source_watch(
        customer.id, scheme="folder", source_name="drop", config={},
    )
    await store.record_watch_event(
        w.id, customer_id=customer.id, event_type="sync_started",
        label="cycle 1",
    )
    await store.record_watch_event(
        w.id, customer_id=customer.id, event_type="file_ingested",
        label="fees.xlsx", payload={"facts": 11},
    )
    await store.record_watch_event(
        w.id, customer_id=customer.id, event_type="cycle_completed",
        label="cycle 1 done",
    )
    events = await store.list_watch_events(w.id, limit=10)
    assert [e["event_type"] for e in events] == [
        "cycle_completed", "file_ingested", "sync_started",
    ]
    assert events[1]["payload"] == {"facts": 11}
    assert all(e["watch_id"] == w.id for e in events)
