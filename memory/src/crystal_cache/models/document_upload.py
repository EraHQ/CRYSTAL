"""DocumentUpload — document ingestion pipeline state.

Tracks documents from upload through chunking and crystallization. The
ingestion pipeline (workers/crystallization.py) reads pending rows,
chunks them, extracts items, and writes crystals; the inspector reads
this table to surface "what's pending, what's failed, what's done."

Distinct from `Document` (which is the v1 source-of-truth table for
ingested content): `DocumentUpload` is the v2 staging/pipeline table,
populated by /v1/documents/upload and consumed by the crystallization
worker. A row may produce many Documents downstream.

Source tracking columns (source_file_id, source_modified_at,
source_connection_id) enable dedup when the Drive sync worker
re-discovers a file we've already crystallized — we look up by
source_file_id rather than re-importing.

Status lifecycle (matches v1 verbatim per Phase 6.5 P0.1 decision):
  pending → crystallizing → crystallized   (happy path, auto-approved)
  pending → crystallizing → review → crystallizing → crystallized
                                              (human-in-the-loop path)
  pending → crystallizing → error           (extraction failure)

State strings are public contracts (frontend, admin queries, external
SDK consumers filter on them); they MUST NOT be renamed without an
explicit ledger decision per CLAUDE.md R3.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# Verbatim v1 names. Per Phase 6.5 P0.1 / CLAUDE.md R3, renames here
# require an explicit ledger decision; the v2 names introduced
# inadvertently in earlier Phase 4 (`processing`, `review_required`,
# `failed`, `complete`) were reverted to the v1 contract.
DocumentUploadStatus = Literal[
    "pending", "crystallizing", "review", "error", "crystallized"
]


class DocumentUpload(BaseModel):
    id: str
    customer_id: str

    label: str = ""
    text: str
    status: DocumentUploadStatus = "pending"
    crystal_type: str = "customer:legacy"

    char_count: int = 0
    crystals_written: int = 0
    items_extracted: int = 0
    error_message: Optional[str] = None

    # Source tracking for Drive sync dedup. Populated when this upload
    # came from Drive; NULL for direct API uploads.
    source_file_id: Optional[str] = None
    source_modified_at: Optional[datetime] = None
    source_connection_id: Optional[str] = None

    # P2 scope-on-sources (ratified 2026-07-02): a document is a SOURCE
    # and carries its own scope; crystals born from it inherit the
    # stamps. None = legacy → team-scoped unowned (today's behavior).
    scope: Optional[str] = None
    owner_operator_id: Optional[str] = None

    # Review workflow columns. Populated when status='review';
    # consumed by the inspector's review UI.
    extracted_items: Optional[list[dict[str, Any]]] = None
    detected_type: Optional[str] = None
    confirmed_type: Optional[str] = None

    # Content chunks for traditional RAG path (verbatim retrieval).
    # Populated alongside crystallization for content_chunk pair_types.
    content_chunks: Optional[list[dict[str, Any]]] = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    crystallized_at: Optional[datetime] = None
