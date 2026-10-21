"""2026-10-21: one owner per document row, no regression past review,
provenance from the fact's own document, and an erase that lets watched
sources re-deliver. Found on the demo tenant: the repo sync and the
crystallization worker both extracted every watched file, the late
extraction put the crystallized row back in review, and approving that
review wrote every fact twice."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from crystal_cache.infrastructure.schema import (
    CrystalRow, DocumentUploadRow, FactRow, SourceWatchRow,
)


async def _status(store, doc_id: str) -> str:
    async with store.session() as s:
        return (await s.get(DocumentUploadRow, doc_id)).status


async def test_a_row_born_claimed_is_invisible_to_the_worker(store, customer):
    mine = await store.create_document_upload(
        customer.id, "watched file", "text of the watched file", claimed=True,
    )
    theirs = await store.create_document_upload(
        customer.id, "uploaded file", "text of the uploaded file",
    )
    assert mine.status == "crystallizing" and theirs.status == "pending"
    assert await _status(store, mine.id) == "crystallizing"

    claimed = await store.claim_pending_documents_batch(limit=10)
    assert [d.id for d in claimed] == [theirs.id]
    async with store.session() as s:
        row = await s.get(DocumentUploadRow, mine.id)
        assert row.claimed_from == "pending" and row.claimed_at is not None


async def test_review_ready_moves_only_a_row_still_in_flight(store, customer):
    doc = await store.create_document_upload(customer.id, "d", "t", claimed=True)
    moved = await store.mark_document_review_ready(
        document_id=doc.id, detected_type="general", content_chunks=[],
        extracted_items=[{"key": "a", "value": "1"}], items_extracted_count=1,
    )
    assert moved is True and await _status(store, doc.id) == "review"

    await store.mark_document_crystallized(
        document_id=doc.id, crystals_written=1, items_extracted=1,
        crystallized_at=datetime.now(timezone.utc),
    )
    assert await _status(store, doc.id) == "crystallized"

    # A late second extraction cannot put it back in review.
    moved = await store.mark_document_review_ready(
        document_id=doc.id, detected_type="general", content_chunks=[],
        extracted_items=[{"key": "a", "value": "1 again"}], items_extracted_count=1,
    )
    assert moved is False
    assert await _status(store, doc.id) == "crystallized"
    async with store.session() as s:
        row = await s.get(DocumentUploadRow, doc.id)
        assert row.extracted_items == [{"key": "a", "value": "1"}]


async def test_conflict_side_names_the_facts_own_document(store, customer):
    seed = await store.create_document_upload(customer.id, "Kerning_Code/adr/0002-pro-price-15.md", "t1")
    other = await store.create_document_upload(customer.id, "Kerning_Code/adr/0003-roles.md", "t2")
    async with store.session() as s:
        s.add(CrystalRow(
            id="cX", customer_id=customer.id, crystal_type="customer:legacy",
            summary_vector=[], source_path="Kerning_Code/adr/0002-pro-price-15.md",
        ))
        s.add(FactRow(
            id="f_seed", crystal_id="cX", pair_type="question_answer",
            prompt_text="ADR 0002|Pricing", claim_text="Pro is $15",
            source_kind="document_chunk", source_doc_id=seed.id, vector=[],
            created_at=datetime.now(timezone.utc),
        ))
        s.add(FactRow(
            id="f_other", crystal_id="cX", pair_type="question_answer",
            prompt_text="ADR 0003|Roles", claim_text="Two roles before launch",
            source_kind="document_chunk", source_doc_id=other.id, vector=[],
            created_at=datetime.now(timezone.utc),
        ))
        s.add(FactRow(
            id="f_stated", crystal_id="cX", pair_type="question_answer",
            prompt_text="Kerning|Status", claim_text="Launch is November 12",
            source_kind="model_reasoning", source_doc_id=None, vector=[],
            created_at=datetime.now(timezone.utc),
        ))
    details = await store.conflict_fact_details(["f_seed", "f_other", "f_stated"])
    assert details["f_seed"]["source_path"] == "Kerning_Code/adr/0002-pro-price-15.md"
    assert details["f_other"]["source_path"] == "Kerning_Code/adr/0003-roles.md"
    # No document of its own: the cluster's path is the fallback.
    assert details["f_stated"]["source_path"] == "Kerning_Code/adr/0002-pro-price-15.md"


async def test_erase_keeps_the_watch_but_forgets_what_it_delivered(store, customer):
    w = await store.create_source_watch(
        customer.id, scheme="git", source_name="Kerning code",
        config={"repo": "owner/kerning", "branch": "main"},
    )
    await store.update_source_watch_state(
        w.id, customer.id, last_state={"head": "abc123"}, last_error="1 file skipped",
    )
    counts = await store.erase_tenant_bank(customer.id)
    assert counts.get("source_watches_reset") == 1
    async with store.session() as s:
        row = (await s.execute(
            select(SourceWatchRow).where(SourceWatchRow.id == w.id)
        )).scalar_one()
        assert row.status == "active"
        assert row.last_state is None and row.last_checked_at is None
        assert row.last_error is None
    # Due again on the next cycle.
    due = await store.list_source_watches_due(datetime.now(timezone.utc))
    assert [x.id for x in due] == [w.id]
