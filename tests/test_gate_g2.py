"""Gate G slice 2 (2026-07-23): JSON detection, #records carving,
mapping inference + mechanical application, and the pipeline's
schema gate (park/apply/terminal flows).

Mapping format v1 ratified 2026-07-23: role-per-path table
(key/value/locator/timestamp/skip) + subject path + domain phrase.
"""

from __future__ import annotations

import json

import pytest

from crystal_cache.ingestion.document_chunker import (
    chunk_document,
    detect_document_type,
)
from crystal_cache.ingestion.schema_mapping import (
    apply_mapping,
    propose_mapping,
    validate_mapping,
)
from crystal_cache.workers.crystallization import _json_schema_gate
from crystal_cache.infrastructure.metadata_store_schema_ext import (
    STATUS_AWAITING_SCHEMA,
    STATUS_SCHEMA_REJECTED,
)


GOOD_MAPPING_JSON = json.dumps({
    "version": 1,
    "roles": {
        "[].name": "key", "[].role": "value", "[].email": "value",
        "[].id": "locator",
    },
    "subject": "[].name",
    "domain": "Team Directory",
})


class _FakeResult:
    def __init__(self, text):
        self.text = text
        self.model = "fake-small"
        self.input_tokens = 100
        self.output_tokens = 40
        self.cache_creation_tokens = 0
        self.cache_read_tokens = 0


class _FakeClient:
    """Duck-typed seam client: complete_detailed preferred (ledger)."""

    def __init__(self, text=GOOD_MAPPING_JSON):
        self._text = text
        self.calls = 0

    def complete_detailed(self, **kwargs):
        self.calls += 1
        return _FakeResult(self._text)


STAFF_JSON = json.dumps([
    {"name": "Maria Lopez", "role": "Office Manager",
     "email": "m@x.co", "id": "u_17"},
    {"name": "Devon Price", "role": "Hygienist",
     "email": "d@x.co", "id": "u_18"},
])


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_json_extensions_detect_as_json():
    assert detect_document_type('[{"a": 1}]', "export.json") == "json"
    assert detect_document_type('{"a": 1}\n{"a": 2}', "log.jsonl") == "json"
    assert detect_document_type('{"a": 1}', "feed.ndjson") == "json"


def test_non_json_detection_unaffected():
    assert detect_document_type("a,b\n1,2", "data.csv") == "tabular"
    assert detect_document_type("plain prose about nothing", "note.txt") \
        != "json"


# ---------------------------------------------------------------------------
# Carving (G-Q1=B)
# ---------------------------------------------------------------------------

def test_array_carves_record_windows():
    big = json.dumps([{"id": i} for i in range(600)])
    chunks = chunk_document(big, "json", "feed.json")
    assert len(chunks) == 2
    assert chunks[0]["label"] == "feed records 1-500"
    assert chunks[0]["record_start"] == 0
    assert chunks[1]["label"] == "feed records 501-600"
    assert chunks[1]["record_start"] == 500
    assert len(json.loads(chunks[0]["text"])) == 500


def test_object_root_stays_whole_file():
    chunks = chunk_document('{"a": {"b": 1}}', "json", "config.json")
    assert len(chunks) == 1
    assert chunks[0]["doc_type"] == "json"
    assert "record_start" not in chunks[0]


def test_jsonl_carves_like_its_array():
    jl = "\n".join(json.dumps({"id": i}) for i in range(3))
    chunks = chunk_document(jl, "json", "log.jsonl")
    assert chunks[0]["label"] == "log records 1-3"


# ---------------------------------------------------------------------------
# Mechanical application
# ---------------------------------------------------------------------------

def test_apply_mapping_mechanics():
    mapping = validate_mapping(json.loads(GOOD_MAPPING_JSON))
    items = apply_mapping(json.loads(STAFF_JSON), mapping, label="staff.json")
    assert len(items) == 2
    first = items[0]
    assert first["key"] == "Maria Lopez"
    assert first["value"] == "role: Office Manager; email: m@x.co"
    assert first["sparse_key"] == \
        "staff | u_17 | Maria Lopez | Team Directory"
    assert first["citation"] == "u_17"
    assert first["type"] == "fact"


def test_apply_mapping_skips_valueless_and_falls_back():
    mapping = {
        "version": 1,
        "roles": {"[].title": "key", "[].body": "value"},
        "subject": "[].title", "domain": "Notes",
    }
    payload = [
        {"title": "Plan", "body": "Ship G2"},
        {"title": "Empty"},                       # no value -> skipped
    ]
    items = apply_mapping(payload, mapping, label="notes.json")
    assert len(items) == 1
    assert items[0]["citation"] == "record 1"     # locator fallback


def test_validate_mapping_coerces_and_rejects():
    ok = validate_mapping({
        "version": 1,
        "roles": {"[].a": "key", "[].b": "banana"},
        "subject": "[].a", "domain": " Ops ",
    })
    assert ok["roles"]["[].b"] == "skip"
    assert ok["domain"] == "Ops"
    assert validate_mapping({"roles": {}}) is None
    assert validate_mapping("garbage") is None


@pytest.mark.asyncio
async def test_propose_mapping_parses_and_fails_safe():
    good = await propose_mapping(
        _FakeClient(), key_paths=["[].name:string"], sample_records=[{}],
    )
    assert good is not None and good["roles"]["[].name"] == "key"

    fenced = await propose_mapping(
        _FakeClient("```json\n" + GOOD_MAPPING_JSON + "\n```"),
        key_paths=[], sample_records=[],
    )
    assert fenced is not None and fenced["domain"] == "Team Directory"

    garbage = await propose_mapping(
        _FakeClient("no json here"), key_paths=[], sample_records=[],
    )
    assert garbage is None

    class _Boom:
        def complete_detailed(self, **kwargs):
            raise RuntimeError("api down")

    boom = await propose_mapping(_Boom(), key_paths=[], sample_records=[])
    assert boom is None


# ---------------------------------------------------------------------------
# The pipeline gate (park / apply / terminal)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_first_contact_proposes_and_parks(store, customer):
    doc = await store.create_document_upload(
        customer.id, label="staff.json", text=STAFF_JSON,
    )
    client = _FakeClient()
    parked, items = await _json_schema_gate(
        store=store, document_id=doc.id, customer_id=customer.id,
        text=STAFF_JSON, label="staff.json", client=client,
    )
    assert parked is True and items == []
    assert client.calls == 1                       # the ONE inference call
    schemas = await store.list_source_schemas(customer.id, status="proposed")
    assert len(schemas) == 1
    assert schemas[0].mapping["roles"]["[].name"] == "key"
    assert len(schemas[0].sample) == 2             # both records sampled
    row = await store.get_document_upload(doc.id, customer.id)
    assert row.status == STATUS_AWAITING_SCHEMA

    # Second arrival of the same shape: parks, NO second proposal,
    # NO second inference call.
    doc2 = await store.create_document_upload(
        customer.id, label="staff-2.json", text=STAFF_JSON,
    )
    client2 = _FakeClient()
    parked2, _ = await _json_schema_gate(
        store=store, document_id=doc2.id, customer_id=customer.id,
        text=STAFF_JSON, label="staff-2.json", client=client2,
    )
    assert parked2 is True
    # The proposed row already exists — the first-contact branch (and
    # its inference call) must not re-enter:
    assert client2.calls == 0
    assert len(await store.list_source_schemas(customer.id)) == 1


@pytest.mark.asyncio
async def test_approved_shape_applies_mechanically(store, customer):
    doc = await store.create_document_upload(
        customer.id, label="staff.json", text=STAFF_JSON,
    )
    parked, _ = await _json_schema_gate(
        store=store, document_id=doc.id, customer_id=customer.id,
        text=STAFF_JSON, label="staff.json", client=_FakeClient(),
    )
    assert parked is True
    schema = (await store.list_source_schemas(customer.id))[0]
    released = await store.approve_source_schema(schema.id)
    assert released == 1

    # A fresh arrival now applies with ZERO model involvement.
    doc2 = await store.create_document_upload(
        customer.id, label="staff-new.json", text=STAFF_JSON,
    )

    class _NeverCalled:
        def complete_detailed(self, **kwargs):
            raise AssertionError("approved shapes must not call the model")

    parked2, items = await _json_schema_gate(
        store=store, document_id=doc2.id, customer_id=customer.id,
        text=STAFF_JSON, label="staff-new.json", client=_NeverCalled(),
    )
    assert parked2 is False
    assert len(items) == 2
    assert items[0]["key"] == "Maria Lopez"


@pytest.mark.asyncio
async def test_rejected_shape_parks_terminally(store, customer):
    doc = await store.create_document_upload(
        customer.id, label="feed.json", text=STAFF_JSON,
    )
    await _json_schema_gate(
        store=store, document_id=doc.id, customer_id=customer.id,
        text=STAFF_JSON, label="feed.json", client=_FakeClient(),
    )
    schema = (await store.list_source_schemas(customer.id))[0]
    await store.reject_source_schema(schema.id)

    # New arrival of the rejected shape: terminal, immediately.
    doc2 = await store.create_document_upload(
        customer.id, label="feed-2.json", text=STAFF_JSON,
    )
    parked, items = await _json_schema_gate(
        store=store, document_id=doc2.id, customer_id=customer.id,
        text=STAFF_JSON, label="feed-2.json", client=_FakeClient(),
    )
    assert parked is True and items == []
    row = await store.get_document_upload(doc2.id, customer.id)
    assert row.status == STATUS_SCHEMA_REJECTED
