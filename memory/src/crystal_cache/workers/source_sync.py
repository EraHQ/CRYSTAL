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
from typing import TYPE_CHECKING

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

# RC-12 (2026-10-05): a file that keeps failing is tried this many times
# per repository head, then skipped until the head moves.
MAX_FILE_ATTEMPTS = 3
# The private key in watch.last_state that carries per-file attempt
# counters beside the handler's own "head" (the only key it reads).
FILE_FAILURES_KEY = "_file_failures"
# A last_error that starts with this is a standing skip note, not a
# transient error; the unchanged-poll branch keeps it.
SKIP_NOTE_PREFIX = "skipped: "
# Lockdown PR-2 (AUDIT_LAUNCH_VERIFY B2-12): a cycle ingests at most this
# many files per watch and re-checks the tenant's capacity every
# BUDGET_RECHECK_EVERY files; the rest wait for the next cycle. Progress
# under an unfinished head lives in last_state under DONE_PATHS_KEY so a
# file is never ingested (and paid for) twice.
MAX_FILES_PER_CYCLE = 200
BUDGET_RECHECK_EVERY = 20
DONE_PATHS_KEY = "_done_paths"


def register_builtin_handlers(store=None) -> None:
    """The registry's standing tenants. Called at worker start;
    idempotent. Git needs no store; the Drive handler resolves
    connections/tokens through it, so gdrive registers only when the
    caller can supply one (2026-07-24, DRIVE-Q1=B)."""
    register_handler(GitSourceHandler())
    if store is not None:
        from ..ingestion.drive_handler import DriveSourceHandler
        register_handler(DriveSourceHandler(store))


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
    register_builtin_handlers(store=store)
    poll_interval = int(
        os.environ.get("CC_SOURCE_SYNC_INTERVAL_SECONDS", "300")
    )
    logger.info("source_sync_worker.started", poll_interval=poll_interval)
    await report_watches_without_token(store)
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


async def report_watches_without_token(store) -> list:
    """Lockdown PR-2 (Q53=A): git watches never fall back to a platform
    token any more, so a watch created without its own token stops
    syncing. Name every such watch at worker start (WARNING, so the
    ERROR alert stays quiet) and return them (tested)."""
    try:
        watches = await store.list_active_source_watches()
    except Exception as e:  # noqa: BLE001  (an inventory must never stop the worker)
        logger.warning("source_sync.inventory_failed", error=str(e)[:200])
        return []
    missing = [w for w in watches if w.scheme == "git" and not w.encrypted_token]
    for w in missing:
        logger.warning(
            "source_sync.watch_without_token",
            watch_id=w.id, customer_id=w.customer_id, source_name=w.source_name,
            note="add a GitHub token to this watch in the console; it will not sync until then",
        )
    return missing


async def _sync_due_watches(
    store, encoder, vector_store, fact_vector_store, llm_client,
) -> None:
    # Cost 1c: watched-source ingest is background spend (describe +
    # extract per file) — it waits when the daily budget is gone.
    from .budget import llm_budget_exhausted
    if await llm_budget_exhausted(store):
        return
    now = datetime.now(timezone.utc)
    for watch in await store.list_source_watches_due(now):
        # Launch knob Q1=B (2026-09-28): gdrive watches stay recorded
        # but DORMANT while the Drive watcher is disabled — flipping
        # CC_DRIVE_WATCHER_ENABLED=true resumes them untouched. The
        # acquisition routes in endpoints/drive.py honor the same knob.
        if watch.scheme == "gdrive":
            from ..config import get_settings
            if not get_settings().drive_watcher_enabled:
                continue
        # Per-customer budget: this customer's subsidy is spent for
        # today — their watches wait; other customers' proceed.
        from .budget import customer_llm_budget_exhausted
        if await customer_llm_budget_exhausted(store, watch.customer_id):
            continue
        # RC-12 / RC-09 (2026-10-05): the TENANT door. A sync runs paid
        # extraction per file; a tenant over the fact cap or out of daily
        # capacity waits (visible in last_error), spends nothing, and is
        # retried next cadence. The legacy per-customer knob above
        # defaults to None and gated nobody.
        try:
            _tenant = await store.get_customer_by_id(watch.customer_id)
            if _tenant is not None:
                from ..ingress.auth import require_write_capacity

                await require_write_capacity(_tenant, store, spend=True)
        except Exception as wall:  # noqa: BLE001  (HTTPException or a store error: either way, do not spend)
            detail = getattr(wall, "detail", None) or type(wall).__name__
            await store.update_source_watch_state(
                watch.id, watch.customer_id,
                last_error=f"waiting on plan capacity: {str(detail)[:200]}",
            )
            logger.info("source_sync.watch_waiting_on_plan",
                        watch_id=watch.id, customer_id=watch.customer_id)
            continue
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


async def _tenant_out_of_capacity(store, customer_id: str) -> bool:
    """PR-2 (B2-12): the mid-cycle re-check, the same two gates the
    watch loop applies before a cycle starts (the legacy per-customer
    budget and the tenant's plan door). A gate that cannot be evaluated
    counts as closed: no spend on an unknown answer."""
    from .budget import customer_llm_budget_exhausted

    try:
        if await customer_llm_budget_exhausted(store, customer_id):
            return True
        tenant = await store.get_customer_by_id(customer_id)
        if tenant is not None:
            from ..ingress.auth import require_write_capacity

            await require_write_capacity(tenant, store, spend=True)
    except Exception:  # noqa: BLE001  (a wall, or a store error: do not spend)
        return True
    return False


async def _emit(store, watch, event_type: str, label: str = "",
                payload: dict | None = None) -> None:
    """G-Q4=A: append to the durable watch-event feed. Best-effort by
    design — the feed must never break a sync cycle. Unchanged polls
    emit nothing (a 15-min cadence would bloat the table with noise;
    the drawer's sync_state chip covers idle)."""
    try:
        await store.record_watch_event(
            watch.id,
            customer_id=watch.customer_id,
            event_type=event_type,
            label=label,
            payload=payload,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("source_sync.event_emit_failed",
                       watch_id=watch.id, event_type=event_type,
                       error=str(e))


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
        # Unchanged — touch checked_at, clear any stale (transient) error.
        # RC-12: a skip note is not transient; it stands until the head
        # moves, so the console keeps showing which file is being skipped.
        keep = (watch.last_error or "").startswith(SKIP_NOTE_PREFIX)
        await store.update_source_watch_state(
            watch.id, watch.customer_id,
            last_error=watch.last_error if keep else None,
        )
        return {"unchanged": True}

    ingested = 0
    retired = 0
    failures = 0
    await _emit(store, watch, "sync_started",
                label=f"{len(changeset.changed)} changed, "
                      f"{len(changeset.removed)} removed")

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
                        await _emit(store, watch, "file_retired", label=path,
                                    payload={"source_uri": uri})
        except Exception as e:  # noqa: BLE001
            failures += 1
            logger.error("source_sync.retire_failed",
                         watch_id=watch.id, path=path, error=str(e))

    # RC-12 (2026-10-05): a file that fails extraction or encoding was
    # retried every cycle with no attempt counter, at a cadence as low as
    # one minute, spending on every try. Attempts live beside the head
    # in last_state under a private key (the git handler reads only
    # "head"; verified); a NEW head resets them, so a fixed file is
    # retried and an unchanged poisoned one is skipped after
    # MAX_FILE_ATTEMPTS and the cycle advances without it.
    new_head = (changeset.new_state or {}).get("head")
    file_failures: dict = dict((watch.last_state or {}).get(FILE_FAILURES_KEY) or {})
    skipped = 0
    progress = (watch.last_state or {}).get(DONE_PATHS_KEY) or {}
    done_paths: set = set(progress.get("paths") or []) if progress.get("head") == new_head else set()
    attempted_this_cycle = 0
    deferred = 0
    changed = list(changeset.changed)

    def _capped_out(p: str) -> bool:
        e = file_failures.get(p) or {}
        return e.get("head") == new_head and int(e.get("attempts", 0)) >= MAX_FILE_ATTEMPTS

    for idx, path in enumerate(changed):
        if path in done_paths:
            continue  # landed in an earlier cycle of this same head
        entry = file_failures.get(path) or {}
        if _capped_out(path):
            skipped += 1
            continue
        if attempted_this_cycle >= MAX_FILES_PER_CYCLE:
            deferred += 1
            continue
        if attempted_this_cycle and attempted_this_cycle % BUDGET_RECHECK_EVERY == 0:
            if await _tenant_out_of_capacity(store, watch.customer_id):
                # Everything from here on waits for the next cycle.
                deferred += sum(
                    1 for p in changed[idx:]
                    if p not in done_paths and not _capped_out(p)
                )
                logger.info("source_sync.cycle_paused_on_capacity",
                            watch_id=watch.id, attempted=attempted_this_cycle)
                break
        attempted_this_cycle += 1
        try:
            envelope = await handler.fetch(watch, path, token)
            await _ingest_envelope(
                store, encoder, vector_store, fact_vector_store,
                llm_client, watch, envelope,
            )
            ingested += 1
            done_paths.add(path)
            file_failures.pop(path, None)
            await _emit(store, watch, "file_ingested", label=path)
        except Exception as e:  # noqa: BLE001
            failures += 1
            # S7 (2026-09-30): the persisted sync event carries the safe
            # message; the redacted detail is logged under its reference.
            from ..hygiene import safe_error
            from ..ingestion.file_extract import DocumentTooLarge, UnsupportedFileType
            ref, message = safe_error(
                "source_sync.ingest_failed", e,
                user_message="This file could not be ingested.",
                watch_id=watch.id, path=path,
            )
            attempts = (int(entry.get("attempts", 0)) if entry.get("head") == new_head else 0) + 1
            if isinstance(e, (DocumentTooLarge, UnsupportedFileType)):
                # PR-1: a file past a hard limit, or of a type ingestion
                # does not accept, will not change by retrying; its
                # message is user-facing by construction.
                message = e.public_message
                attempts = MAX_FILE_ATTEMPTS
            file_failures[path] = {"attempts": attempts, "head": new_head, "ref": ref}
            await _emit(store, watch, "error", label=path,
                        payload={"error": message, "ref": ref, "attempts": attempts})
            if attempts >= MAX_FILE_ATTEMPTS:
                await _emit(store, watch, "file_skipped", label=path,
                            payload={"ref": ref, "attempts": attempts,
                                     "note": "will not be retried until the file changes"})

    # Failures that have reached the attempt cap are skipped from here on
    # and no longer hold the state back: the cycle advances without them.
    live_failures = failures - sum(
        1 for p, e in file_failures.items()
        if p in changeset.changed and e.get("head") == new_head
        and int(e.get("attempts", 0)) >= MAX_FILE_ATTEMPTS
    )
    carried_state = dict(changeset.new_state or {})
    if file_failures:
        carried_state[FILE_FAILURES_KEY] = file_failures

    if deferred > 0:
        # PR-2 (B2-12): the cycle stopped at the per-cycle cap or at the
        # capacity re-check. The head does NOT advance; what landed is
        # remembered so the next cycle continues from here instead of
        # paying for the same files again.
        await store.update_source_watch_state(
            watch.id, watch.customer_id,
            last_state={
                **(watch.last_state or {}),
                FILE_FAILURES_KEY: file_failures,
                DONE_PATHS_KEY: {"head": new_head, "paths": sorted(done_paths)},
            },
            last_error=f"{deferred} file(s) remaining; continuing next cycle",
        )
        logger.info("source_sync.cycle_partial",
                    watch_id=watch.id, ingested=ingested, deferred=deferred)
        await _emit(store, watch, "cycle_completed",
                    label=f"{ingested} ingested, {retired} retired, "
                          f"{failures} failed, {deferred} remaining",
                    payload={"ingested": ingested, "retired": retired,
                             "failures": failures, "deferred": deferred})
        return {"ingested": ingested, "retired": retired,
                "failures": failures, "deferred": deferred}

    if live_failures <= 0:
        # The cycle landed whole (poisoned files excepted) — advance.
        await store.update_source_watch_state(
            watch.id, watch.customer_id,
            last_state=carried_state,
            last_error=(
                f"{SKIP_NOTE_PREFIX}{skipped + (failures - live_failures)} file(s) skipped after "
                f"{MAX_FILE_ATTEMPTS} failed attempts; see sync events"
                if (skipped or failures > live_failures) else None
            ),
        )
    else:
        # Partial cycle: DON'T advance the head — the next poll re-finds
        # the same changes and dedup skips what already landed. The
        # attempt counters still persist.
        await store.update_source_watch_state(
            watch.id, watch.customer_id,
            last_state={
                **(watch.last_state or {}),
                FILE_FAILURES_KEY: file_failures,
                DONE_PATHS_KEY: {"head": new_head, "paths": sorted(done_paths)},
            },
            last_error=f"{live_failures} item(s) failed; state not advanced",
        )
    logger.info("source_sync.cycle_done",
                watch_id=watch.id, ingested=ingested,
                retired=retired, failures=failures)
    await _emit(store, watch, "cycle_completed",
                label=f"{ingested} ingested, {retired} retired, "
                      f"{failures} failed",
                payload={"ingested": ingested, "retired": retired,
                         "failures": failures})
    return {"ingested": ingested, "retired": retired, "failures": failures}


async def _ingest_envelope(
    store, encoder, vector_store, fact_vector_store, llm_client,
    watch, envelope,
) -> None:
    """Envelope -> upload -> chunk/describe/stamp -> M-Q3 routing."""
    from .crystallization import crystallize_document

    # Binary formats can't be utf-8-decoded into sense — route them
    # through the same extractors the upload endpoint uses (Gate E
    # fixed this for xlsx; Gate H adds its adapter batch — rtf/ipynb
    # decode as text but still want their proper rendering).
    lower = (envelope.label or envelope.source_uri).lower()
    # Lockdown PR-1 (B2-5): the same byte and character caps as the
    # upload route; a file past either is skipped for good (see the
    # caller), not retried.
    from ..config import get_settings
    from ..ingestion.file_extract import (
        DocumentTooLarge, UnsupportedFileType, extract_text_from_file, looks_binary,
    )
    _settings = get_settings()
    max_bytes = int(_settings.upload_max_bytes)
    max_chars = int(_settings.document_max_chars)
    if len(envelope.payload_bytes or b"") > max_bytes:
        raise DocumentTooLarge(
            f"The file is larger than {max_bytes // 2**20} MB. Split it."
        )
    if lower.endswith((
        ".xlsx", ".pdf", ".docx",
        ".pptx", ".rtf", ".odt", ".epub", ".ipynb",
    )):
        text = await asyncio.to_thread(  # RC-14: CPU-bound extraction off the loop
            extract_text_from_file,
            envelope.payload_bytes, lower,
            mime=getattr(envelope, "mime_type", None),
            max_chars=max_chars + 1,
        )
    else:
        if looks_binary(envelope.payload_bytes or b""):
            raise UnsupportedFileType("Binary files are not ingested.")
        text = envelope.payload_bytes.decode("utf-8", errors="replace")
    if len(text) > max_chars:
        raise DocumentTooLarge(
            f"Extracted text exceeds {max_chars:,} characters; split the file."
        )
    from ..ingestion.file_extract import sanitize_label

    doc = await store.create_document_upload(
        watch.customer_id,
        # Lockdown PR-5 (Q45): the same label rules as every other lane.
        # Path separators survive (they are the source path identity).
        sanitize_label(envelope.label or envelope.source_uri, text=text),
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
