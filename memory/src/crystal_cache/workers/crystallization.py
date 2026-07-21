"""Crystallization worker — processes pending document uploads.

Polls for `DocumentUpload` rows with status='pending', runs them
through the chunk → extract pipeline, and either marks them
ready for human review or commits the crystals depending on the
ingestion settings.

Status lifecycle (matches v1 verbatim per Phase 6.5 P0.1):
  pending → crystallizing → review → crystallized   (happy path)
  pending → crystallizing → error                   (extraction failed)

v1 layout (replaced by this module):
  - lifespan._crystallization_worker: the poll loop
  - lifespan._crystallization_worker._process_one: per-doc handler
  - _crystallize_document (module-level helper): manual + auto-
    crystallize endpoint shared path. AN-2 said this becomes a
    workflow function in workers/. That's `crystallize_document`
    below.

Two public entry points:

1. `run_crystallization_worker(...)` — the background poll loop,
   called from the FastAPI lifespan. Runs until shutdown_event is set.

2. `crystallize_document(...)` — synchronous workflow that takes a
   single document_id and runs the full pipeline against it. Called
   by the SDK's manual /v1/documents/{id}/crystallize endpoint and by
   the cognition commit path. Same logic as the worker's per-doc
   handler, exposed as a callable.

AN-4 (CU-10 CLOSED, verified Gate M slice 6, 2026-07-18):
`MetadataStore.claim_pending_documents_batch` is the atomic claim
primitive. On SQLite the single-writer SERIALIZABLE transaction
makes SELECT+mark atomic; on Postgres the SELECT takes `FOR UPDATE
SKIP LOCKED`, so concurrent workers claim disjoint batches. This
matters now: the source-sync worker (Gate M) can enqueue bursts of
uploads that multiple workers may drain.
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import structlog

if TYPE_CHECKING:
    from ..encoding.base import TextEncoder
    from ..infrastructure.metadata_store import MetadataStore
    from ..infrastructure.vector_store import VectorStore

logger = structlog.get_logger(__name__)


async def crystallize_document(
    *,
    store: "MetadataStore",
    encoder: "TextEncoder",
    vector_store: "VectorStore",
    document_id: str,
    client: Optional[object] = None,
) -> None:
    """Run the chunking + extraction pipeline against one document.

    Reads the DocumentUpload row, chunks the text, calls the LLM
    extraction step, and writes the result back as status='review'
    (waiting for user approval) OR status='error' on failure. The
    state transitions are owned by the mixin methods this function
    calls (`mark_document_review_ready`, `mark_document_error`),
    not by string literals here.

    This is the workflow that AN-2 said should move from app.py's
    `_crystallize_document` helper into the workers package. It's
    called from both the background poll loop (via the worker's
    `_process_one` helper) and from the manual SDK endpoint.

    No tenancy check at this layer — the caller is responsible for
    customer scoping (SDK endpoint has the customer dep; the poll
    loop is cross-tenant).
    """
    from ..ingestion.document_chunker import chunk_document, detect_document_type
    from ..ingestion.document_pipeline import DocumentPipeline

    # Read the doc — cross-tenant get, doesn't validate customer
    # because the caller already did or because this is cross-tenant
    # background work.
    from ..infrastructure.schema import DocumentUploadRow
    async with store.session() as session:
        row = await session.get(DocumentUploadRow, document_id)
        if row is None:
            logger.warning("crystallize_document.not_found", document_id=document_id)
            return
        # Capture the fields we need outside the session
        doc_text = row.text
        doc_label = row.label or ""
        doc_crystal_type = row.crystal_type or "customer:legacy"
        # Preserve the ORIGINAL provenance marker. Cognition/background-
        # worker output arrives tagged detected_type='inferred_knowledge'
        # (recall-gate birth attribution, 2026-07-03). The re-detection
        # below picks a CHUNKING strategy, which would otherwise overwrite
        # that marker and lose the provenance the approve step keys on. So
        # we carry it forward explicitly.
        original_detected_type = row.detected_type

    try:
        logger.info(
            "crystallize_document.extracting",
            document_id=document_id,
            label=doc_label,
        )

        # Phase 1: Content chunking (no LLM). detected_type here is a
        # CHUNKING strategy; it does not overwrite an inferred_knowledge
        # provenance marker (see review_type below).
        detected_type = detect_document_type(doc_text, doc_label)
        chunks = chunk_document(doc_text, detected_type, doc_label)
        content_chunks = [
            {
                "index": i,
                "label": c["label"],
                "text": c["text"],
                "char_count": len(c["text"]),
                "locator": c.get("locator", c["label"]),
                "subject": c.get("subject"),
                "doc_type": detected_type,
            }
            for i, c in enumerate(chunks)
        ]

        # Gate D4 (ratified 2026-07-17, option C; built 2026-07-18): the
        # C2 screen runs at CHUNK time so its findings reach the REVIEW
        # surface — the curator sees "instruction-shaped text" warnings
        # before approving, and the approve becomes the un-quarantine.
        # The stamped key is the sentinel: present = screened and
        # surfaceable; absent (legacy rows, direct paths) = the pipeline
        # screens at write exactly as before. Fail-safe: a screen error
        # stamps nothing and write-time behavior stands.
        try:
            from ..ingestion.injection_screen import (
                scan_for_injection as _scan,
            )
            for cc in content_chunks:
                cc["injection_hits"] = _scan(cc["text"])
        except Exception as _screen_err:  # noqa: BLE001
            logger.warning(
                "crystallize_document.chunk_screen_failed",
                document_id=document_id, error=str(_screen_err),
            )

        # Code descriptions (CC_ENABLE_CODE_DESCRIPTIONS): index each code
        # chunk by a functional NL description instead of its raw source, so
        # conceptual queries match. One model call per file, attached to the
        # chunk dicts here and threaded to add_pair_*'s embed_text at approve
        # time. Best-effort — undescribed chunks fall back to encoding the
        # verbatim body. Historically only the eval harness passed a client
        # (the poll loop skipped describing); since the Gate D2 wiring fix
        # below, the flag alone is sufficient on every path.
        from ..config import settings as _settings
        # Gate D2 wiring fix (2026-07-17): no production call site passes
        # a client (only the A/B eval harness ever did), so the describer
        # was unreachable in deployed ingest regardless of the flag. When
        # the flag is on and no client was injected, resolve the seam
        # client here — fail-safe, an unconfigured provider must never
        # break crystallization. An explicitly injected client (the eval)
        # still wins.
        if (
            _settings.enable_code_descriptions
            and detected_type == "code"
            and client is None
        ):
            try:
                from ..llm import get_llm_client
                client = get_llm_client()
            except Exception as _cl_err:  # noqa: BLE001
                logger.warning(
                    "crystallize_document.describer_client_unavailable",
                    document_id=document_id, error=str(_cl_err),
                )
        if (
            _settings.enable_code_descriptions
            and detected_type == "code"
            and client is not None
        ):
            from ..ingestion.code_describer import describe_code_file
            desc = await describe_code_file(
                file_text=doc_text, chunks=content_chunks,
                client=client, file_label=doc_label,
                customer_id=getattr(row, "customer_id", None), store=store,
            )
            by_index = desc["by_index"]
            file_summary = desc["file_summary"]
            for cc in content_chunks:
                idx = cc["index"]
                if idx in by_index:
                    cc["description"] = by_index[idx]
                elif file_summary and "::<module" in str(cc.get("locator", "")):
                    cc["description"] = f"{doc_label}: {file_summary}"
            logger.info(
                "crystallize_document.described",
                document_id=document_id,
                described=sum(1 for c in content_chunks if c.get("description")),
                total=len(content_chunks),
            )

        # Phase 2: Knowledge extraction (LLM).
        # Code skips the prose extractor — the verbatim per-symbol chunks
        # ARE the code knowledge. Code-aware summaries are a later phase.
        if detected_type == "code":
            extracted_items: list[dict] = []
        else:
            pipeline = DocumentPipeline(
                store=store,
                encoder=encoder,
                vector_store=vector_store,
                client=client,
            )
            extracted = await pipeline.extract_items(
                text=doc_text,
                label=doc_label,
                crystal_type=doc_crystal_type,
                # Gate A (2026-07-16): structure-fed extraction — the
                # Phase-1 chunks carry locators into the prompts; the
                # profile follows PROVENANCE (a cognition report stays
                # inferred_knowledge even though its content re-detects
                # as something else).
                content_chunks=content_chunks,
                detected_type=(
                    "inferred_knowledge"
                    if original_detected_type == "inferred_knowledge"
                    else detected_type
                ),
                # Gate B (2026-07-16): extraction stamps the ledger under
                # the document's customer.
                customer_id=getattr(row, "customer_id", None),
                store=store,
            )
            extracted_items = [
                {
                    "key": item.key,
                    "sparse_key": item.sparse_key,
                    "value": item.value,
                    "type": item.item_type,
                    "citation": item.citation,
                }
                for item in extracted
            ]

        # Mark ready for review via v2 store method. Preserve the
        # inferred_knowledge provenance marker if that's what the document
        # arrived as, so the approve step can birth its crystals
        # recall_gated; otherwise record the chunking type as before.
        review_type = (
            "inferred_knowledge"
            if original_detected_type == "inferred_knowledge"
            else detected_type
        )
        await store.mark_document_review_ready(
            document_id=document_id,
            detected_type=review_type,
            content_chunks=content_chunks,
            extracted_items=extracted_items,
            items_extracted_count=len(extracted_items),
        )

        logger.info(
            "crystallize_document.ready_for_review",
            document_id=document_id,
            detected_type=detected_type,
            content_chunks=len(content_chunks),
            extracted_items=len(extracted_items),
        )

    except Exception as e:
        # Best-effort error marking: the worker recovers and keeps
        # polling other docs even if one fails.
        await store.mark_document_error(document_id, str(e))
        logger.error(
            "crystallize_document.failed",
            document_id=document_id,
            error=str(e),
            error_type=type(e).__name__,
        )


async def run_crystallization_worker(
    *,
    store: "MetadataStore",
    encoder: "TextEncoder",
    vector_store: "VectorStore",
    shutdown_event: asyncio.Event,
) -> None:
    """Background worker poll loop.

    Reads `CC_CRYSTALLIZE_CONCURRENCY` (default 1) and
    `CC_CRYSTALLIZE_POLL_SECONDS` (default 10) from env. Runs until
    `shutdown_event` is set; on shutdown, finishes any in-flight
    docs before exiting.

    Per-iteration:
      1. Claim a batch of pending docs via the atomic claim primitive,
         which marks claimed rows as 'crystallizing' inside one
         transaction (AN-4, resolved-for-SQLite per Phase 6.5).
      2. Spawn one task per doc, bounded by a semaphore.
      3. Wait for all tasks.
      4. Sleep until next poll or shutdown.
    """
    concurrency = int(os.environ.get("CC_CRYSTALLIZE_CONCURRENCY", "1"))
    poll_interval = int(os.environ.get("CC_CRYSTALLIZE_POLL_SECONDS", "10"))
    sem = asyncio.Semaphore(concurrency)

    logger.info(
        "crystallization_worker.started",
        concurrency=concurrency,
        poll_interval=poll_interval,
    )

    async def _process_one(document_id: str) -> None:
        async with sem:
            await crystallize_document(
                store=store,
                encoder=encoder,
                vector_store=vector_store,
                document_id=document_id,
            )

    while not shutdown_event.is_set():
        try:
            # Cost 1c: the bank stops thinking before it stops
            # answering — no new describe/extract work past the daily
            # background budget.
            from .budget import llm_budget_exhausted
            if await llm_budget_exhausted(store):
                await asyncio.sleep(poll_interval)
                continue
            # AN-4: atomic claim. The store method marks claimed rows
            # as 'crystallizing' inside the same transaction so two
            # workers running concurrently see disjoint sets under
            # SQLite. Postgres-multi-worker scope-fix is CU-10.
            claimed = await store.claim_pending_documents_batch(
                limit=concurrency * 2,
            )

            if claimed:
                logger.info(
                    "crystallization_worker.claimed_batch",
                    count=len(claimed),
                )
                # Per-customer budget: release claimed docs whose
                # customer's daily subsidy is spent — back to pending,
                # retried when the day (or the plan) allows.
                from .budget import customer_llm_budget_exhausted
                runnable = []
                for doc in claimed:
                    if await customer_llm_budget_exhausted(
                        store, doc.customer_id,
                    ):
                        await store.release_document_to_pending(doc.id)
                    else:
                        runnable.append(doc)
                tasks = [_process_one(doc.id) for doc in runnable]
                await asyncio.gather(*tasks, return_exceptions=True)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(
                "crystallization_worker.poll_error",
                error=str(e),
                error_type=type(e).__name__,
            )

        # Sleep for poll_interval OR until shutdown signaled
        try:
            await asyncio.wait_for(
                shutdown_event.wait(),
                timeout=poll_interval,
            )
            break  # shutdown signaled
        except asyncio.TimeoutError:
            pass  # normal poll-interval expiry, continue

    logger.info("crystallization_worker.stopped")
