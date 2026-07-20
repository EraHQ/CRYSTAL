"""Source-sync worker — Gate M slice 4, the spine.

The loop that makes watching real: every cycle, ask the store which
watches are due, dispatch each to its scheme handler, and turn what
changed into crystals through the SAME ingestion path every manual
upload takes — identity (C1/D6), chunk-time screening (D4), code
comprehension + chains (D2), reconciliation order-independence.

M-Q3 routing, mechanically: every changed file becomes a pending
upload and runs the standard chunk/describe/stamp step
(`crystallize_document`). A `gated` watch stops there — the doc sits
in review like any manual upload. An `auto` watch immediately runs
the approve pipeline with curator_reviewed=False: born QUARANTINE per
D4-A, earning promotion via the scans — the tier system is the
reviewer for unattended ingest. No new branches in existing code;
auto is just machine-approved review.

Deletions delete (M design statement): removed paths retire their
crystals by exact source_uri — facts and chains die via the D2
cascade; the repo is the source of truth.

Crash/partial-failure safety: last_state advances ONLY after a cycle
with zero hard failures. A re-poll re-finds the same changes, and
content-hash dedup makes re-ingesting the already-landed files a
cheap skip — idempotent by the replace semantics Gate D built.
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from ..ingestion.git_handler import GitSourceHandler
from ..ingestion.source_handlers import (
    get_handler,
    register_handler,
    resolve_watch_token,
)

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore

import structlog

logger = structlog.get_logger(__name__)


def register_builtin_handlers() -> None:
    """The registry's standing tenants. Called at worker start;
    idempotent. Future schemes (folder, unified drive) register here."""
    register_handler(GitSourceHandler())


async def run_source_sync_worker(
    *,
    store: "MetadataStore",
    encoder,
    vector_store,
    fact_vector_store=None,
    llm_client=None,
    shutdown_event: asyncio.Event,
) -> None:
    """Poll loop. CC_SOURCE_SYNC_INTERVAL_SECONDS (default 300)."""
    register_builtin_handlers()
    poll_interval = int(
        os.environ.get("CC_SOURCE_SYNC_INTERVAL_SECONDS", "300")
    )
    logger.info("source_sync_worker.started", poll_interval=poll_interval)
    while not shutdown_event.is_set():
        try:
            await _sync_due_watches(
                store, encoder, vector_store, fact_vector_store, llm_client,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.error("source_sync_worker.cycle_error",
                         error=str(e), error_type=type(e).__name__)
        try:
            await asyncio.wait_for(
                shutdown_event.wait(), timeout=poll_interval,
            )
        except asyncio.TimeoutError:
            pass
    logger.info("source_sync_worker.stopped")


async def _sync_due_watches(
    store, encoder, vector_store, fact_vector_store, llm_client,
) -> None:
    now = datetime.now(timezone.utc)
    for watch in await store.list_source_watches_due(now):
        try:
            await sync_one_watch(
                store, encoder, vector_store, fact_vector_store,
                llm_client, watch,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.error("source_sync.watch_failed",
                         watch_id=watch.id, error=str(e))
            try:
                await store.update_source_watch_state(
                    watch.id, watch.customer_id, last_error=str(e)[:2000],
                )
            except Exception:  # noqa: BLE001
                pass


async def sync_one_watch(
    store, encoder, vector_store, fact_vector_store, llm_client, watch,
) -> dict:
    """One watch, one cycle. Returns a small result dict (tested)."""
    handler = get_handler(watch.scheme)
    if handler is None:
        await store.update_source_watch_state(
            watch.id, watch.customer_id,
            last_error=f"no handler for scheme {watch.scheme!r}",
        )
        return {"error": "no_handler"}

    try:
        token = await resolve_watch_token(store, watch)
    except ValueError as e:
        await store.update_source_watch_state(
            watch.id, watch.customer_id,
            last_error=f"token decrypt failed: {e}",
        )
        return {"error": "token"}

    changeset = await handler.check(watch, token)
    if changeset is None:
        # Unchanged — touch checked_at, clear any stale error.
        await store.update_source_watch_state(
            watch.id, watch.customer_id, last_error=None,
        )
        return {"unchanged": True}

    ingested = 0
    retired = 0
    failures = 0

    # Deletions delete: exact-URI retirement, cascade does the rest.
    for path in changeset.removed:
        uri = f"repo://{watch.source_name}/{path}"
        try:
            crystals = await store.list_crystals_for_customer(
                watch.customer_id
            )
            for c in crystals:
                if getattr(c, "source_uri", None) == uri:
                    if await store.delete_crystal(
                        c.id, watch.customer_id,
                        vector_store=vector_store,
                        fact_vector_store=fact_vector_store,
                    ):
                        retired += 1
                        logger.info("source_sync.crystal_retired",
                                    watch_id=watch.id, source_uri=uri)
        except Exception as e:  # noqa: BLE001
            failures += 1
            logger.error("source_sync.retire_failed",
                         watch_id=watch.id, path=path, error=str(e))

    for path in changeset.changed:
        try:
            envelope = await handler.fetch(watch, path, token)
            await _ingest_envelope(
                store, encoder, vector_store, fact_vector_store,
                llm_client, watch, envelope,
            )
            ingested += 1
        except Exception as e:  # noqa: BLE001
            failures += 1
            logger.error("source_sync.ingest_failed",
                         watch_id=watch.id, path=path, error=str(e))

    if failures == 0:
        # The cycle landed whole — advance the state.
        await store.update_source_watch_state(
            watch.id, watch.customer_id,
            last_state=changeset.new_state, last_error=None,
        )
    else:
        # Partial cycle: DON'T advance — the next poll re-finds the
        # same changes and dedup skips what already landed.
        await store.update_source_watch_state(
            watch.id, watch.customer_id,
            last_error=f"{failures} item(s) failed; state not advanced",
        )
    logger.info("source_sync.cycle_done",
                watch_id=watch.id, ingested=ingested,
                retired=retired, failures=failures)
    return {"ingested": ingested, "retired": retired, "failures": failures}


async def _ingest_envelope(
    store, encoder, vector_store, fact_vector_store, llm_client,
    watch, envelope,
) -> None:
    """Envelope -> upload -> chunk/describe/stamp -> M-Q3 routing."""
    from .crystallization import crystallize_document

    # Binary formats can't be utf-8-decoded into sense — route them
    # through the same extractors the upload endpoint uses (Gate E
    # fixed this for xlsx and closed the latent pdf/docx hole too).
    lower = (envelope.label or envelope.source_uri).lower()
    if lower.endswith((".xlsx", ".pdf", ".docx")):
        from ..ingestion.file_extract import extract_text_from_file
        text = extract_text_from_file(
            envelope.payload_bytes, lower,
        )
    else:
        text = envelope.payload_bytes.decode("utf-8", errors="replace")
    doc = await store.create_document_upload(
        watch.customer_id,
        envelope.label or envelope.source_uri,
        text,
        source_modified_at=envelope.source_modified_at,
        source_connection_id=envelope.connection_id,
        source_uri=envelope.source_uri,
    )
    # The standard chunk/describe/stamp step every upload takes —
    # chunk-time injection findings included (D4).
    await crystallize_document(
        store=store, encoder=encoder, vector_store=vector_store,
        document_id=doc.id, client=llm_client,
    )
    if watch.review_mode != "auto":
        return  # gated: the doc sits in review like any manual upload

    # Auto (M-Q3): machine-approved review. curator_reviewed=False ->
    # born quarantine (D4-A) — the tier system reviews unattended
    # ingest.
    row = await store.get_document_upload(doc.id, watch.customer_id)
    if row is None or row.status != "review":
        return  # chunking failed or errored; leave as-is for triage
    from ..ingestion.document_pipeline import DocumentPipeline
    pipeline = DocumentPipeline(
        store=store, encoder=encoder, vector_store=vector_store,
        fact_vector_store=fact_vector_store,
    )
    result = await pipeline.approve_and_crystallize(
        customer_id=watch.customer_id,
        document_id=doc.id,
        items=list(row.extracted_items or []),
        content_chunks=list(row.content_chunks or []),
        crystal_type=row.confirmed_type or row.crystal_type,
        origin="direct",
        curator_reviewed=False,
    )
    await store.mark_document_crystallized(
        document_id=doc.id,
        crystals_written=result.crystals_written,
        items_extracted=row.items_extracted or 0,
        crystallized_at=datetime.now(timezone.utc),
    )
