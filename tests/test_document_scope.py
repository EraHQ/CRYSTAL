"""P2 scope-on-sources (ratified 2026-07-02).

A document is a SOURCE: it carries scope + owner from upload, and every
crystal born from it inherits the stamps. Legacy uploads (NULL scope)
produce team-scoped unowned crystals — today's exact behavior.

R14 note: these assertions are verified by `pytest`; they describe
expected behavior and have not yet been run at authoring time.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from crystal_cache.endpoints.documents import (
    _resolve_source_scope,
    sdk_upload_document,
)
from crystal_cache.ingestion.document_pipeline import stamps_for_source
from crystal_cache.ingress.schema import DocumentUploadRequest


def test_stamps_for_source_vocabulary():
    assert stamps_for_source(None, "op_a", "cus_1") == {}  # legacy: untouched
    assert stamps_for_source("personal", "op_a", "cus_1") == {
        "owner_operator_id": "op_a", "group_team_id": "cus_1", "mode": 0o600,
    }
    assert stamps_for_source("team", "op_a", "cus_1") == {
        "owner_operator_id": "op_a", "group_team_id": "cus_1", "mode": 0o640,
    }
    with pytest.raises(ValueError):
        stamps_for_source("public", "op_a", "cus_1")


async def test_resolve_source_scope_defaults_and_guards(store, customer):
    op, _ = await store.create_operator(customer.id, display_name="A")

    # Deployment default (personal) when omitted.
    scope, owner = _resolve_source_scope(None, op)
    assert scope == "personal" and owner == op.id

    # Explicit team honored.
    scope, owner = _resolve_source_scope("team", op)
    assert scope == "team" and owner == op.id

    # Bad scope → 422; viewer → 403.
    with pytest.raises(HTTPException) as exc:
        _resolve_source_scope("public", op)
    assert exc.value.status_code == 422

    viewer, _ = await store.create_operator(
        customer.id, display_name="V", role="viewer",
    )
    with pytest.raises(HTTPException) as exc:
        _resolve_source_scope(None, viewer)
    assert exc.value.status_code == 403


async def test_upload_stamps_the_source(store, customer):
    """The JSON upload route stamps the resolved scope + owner onto the
    document row — the source carries its scope from birth."""
    op, _ = await store.create_operator(customer.id, display_name="A")

    body = DocumentUploadRequest(text="some knowledge", label="doc")
    resp = await sdk_upload_document(body, (customer, op), store)
    assert resp.status_code == 200

    docs = await store.list_document_uploads(customer.id)
    doc = next(d for d in docs if d.label == "doc")
    assert doc.scope == "personal"          # deployment default
    assert doc.owner_operator_id == op.id

    body2 = DocumentUploadRequest(text="shared knowledge", label="tdoc",
                                  scope="team")
    await sdk_upload_document(body2, (customer, op), store)
    docs = await store.list_document_uploads(customer.id)
    tdoc = next(d for d in docs if d.label == "tdoc")
    assert tdoc.scope == "team"
    assert tdoc.owner_operator_id == op.id


async def test_create_document_upload_roundtrips_stamps(store, customer):
    doc = await store.create_document_upload(
        customer_id=customer.id, label="x", text="t",
        scope="personal", owner_operator_id="op_z",
    )
    fetched = await store.get_document_upload(doc.id, customer.id)
    assert fetched.scope == "personal"
    assert fetched.owner_operator_id == "op_z"

    legacy = await store.create_document_upload(
        customer_id=customer.id, label="y", text="t",
    )
    fetched = await store.get_document_upload(legacy.id, customer.id)
    assert fetched.scope is None
    assert fetched.owner_operator_id is None
