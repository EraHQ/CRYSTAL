"""Share-source (P4, ratified 2026-07-02).

One call flips a document AND everything derived from it: provenance
resolution (extracted-item crystal ids + content-chunk source_path
stamps), owner-or-admin authorization, the reversible journey, and the
row restamp so future crystallization inherits.

R14 note: these assertions are verified by `pytest`; they describe
expected behavior and have not yet been run at authoring time.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from crystal_cache.endpoints.documents import sdk_set_document_scope
from crystal_cache.infrastructure.schema import CrystalRow


def _json_request(payload: dict):
    async def _json():
        return payload
    return SimpleNamespace(
        headers={"content-type": "application/json"}, json=_json,
    )


async def _seed_crystal(
    store, customer_id, cid, *, owner, mode=0o600, source_path=None,
):
    async with store.session() as s:
        s.add(CrystalRow(
            id=cid, customer_id=customer_id,
            crystal_type="customer:legacy", summary_vector=[],
            owner_operator_id=owner, group_team_id=customer_id,
            mode=mode, source_path=source_path,
        ))


async def _seed_document(store, customer, owner_id):
    """A crystallized personal document: one knowledge-item crystal
    (recorded on extracted_items) + one chunk crystal (source_path)."""
    doc = await store.create_document_upload(
        customer_id=customer.id, label="handbook", text="t",
        scope="personal", owner_operator_id=owner_id,
    )
    await _seed_crystal(store, customer.id, "c_item", owner=owner_id)
    await _seed_crystal(store, customer.id, "c_chunk", owner=owner_id,
                        source_path="handbook")
    await store.update_document_review_edits(
        doc.id, customer.id,
        extracted_items=[{"key": "k", "value": "v", "crystal_id": "c_item"}],
        content_chunks=[{"text": "x", "source_path": "handbook"}],
    )
    return doc


async def test_share_source_flips_the_whole_derived_set(store, customer):
    owner, _ = await store.create_operator(customer.id, display_name="Own")
    doc = await _seed_document(store, customer, owner.id)

    resp = await sdk_set_document_scope(
        doc.id, _json_request({"scope": "team"}), (customer, owner), store,
    )
    assert resp.status_code == 200
    payload = json.loads(resp.body)
    assert payload["crystals_flipped"] == 2
    assert set(payload["crystal_ids"]) == {"c_item", "c_chunk"}

    assert (await store.get_crystal("c_item")).mode == 0o640
    assert (await store.get_crystal("c_chunk")).mode == 0o640
    # The source row is restamped — future crystallization inherits.
    fetched = await store.get_document_upload(doc.id, customer.id)
    assert fetched.scope == "team"

    # Reversible: unshare puts everything back.
    await sdk_set_document_scope(
        doc.id, _json_request({"scope": "personal"}), (customer, owner), store,
    )
    assert (await store.get_crystal("c_item")).mode == 0o600
    assert (await store.get_crystal("c_chunk")).mode == 0o600
    fetched = await store.get_document_upload(doc.id, customer.id)
    assert fetched.scope == "personal"


async def test_share_source_authorization(store, customer):
    owner, _ = await store.create_operator(customer.id, display_name="Own")
    other, _ = await store.create_operator(customer.id, display_name="Oth")
    doc = await _seed_document(store, customer, owner.id)

    with pytest.raises(HTTPException) as exc:
        await sdk_set_document_scope(
            doc.id, _json_request({"scope": "team"}), (customer, other), store,
        )
    assert exc.value.status_code == 403

    # The Default Admin (team key) may.
    admin = await store.ensure_default_admin(customer.id)
    resp = await sdk_set_document_scope(
        doc.id, _json_request({"scope": "team"}), (customer, admin), store,
    )
    assert resp.status_code == 200

    with pytest.raises(HTTPException) as exc:
        await sdk_set_document_scope(
            doc.id, _json_request({"scope": "public"}), (customer, owner), store,
        )
    assert exc.value.status_code == 422

    with pytest.raises(HTTPException) as exc:
        await sdk_set_document_scope(
            "doc_missing", _json_request({"scope": "team"}),
            (customer, owner), store,
        )
    assert exc.value.status_code == 404


async def test_share_source_provenance_reads(store, customer):
    await _seed_crystal(store, customer.id, "c_p1", owner="op_x",
                        source_path="a.md")
    await _seed_crystal(store, customer.id, "c_p2", owner="op_x",
                        source_path="b.md")
    await _seed_crystal(store, "cus_other", "c_pf", owner="op_z",
                        source_path="a.md")

    ids = await store.list_crystal_ids_for_source_paths(
        customer.id, ["a.md", "b.md"],
    )
    assert set(ids) == {"c_p1", "c_p2"}  # foreign tenant excluded
    assert await store.list_crystal_ids_for_source_paths(customer.id, []) == []
