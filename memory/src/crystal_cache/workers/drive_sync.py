"""Drive sync worker — polls watched folders + files for changes.

Periodically queries Google Drive for new or modified files inside
watched folders and standalone watched files, imports them as
pending DocumentUploads, and logs PHI access for resources flagged
as containing protected health information.

v1 layout (replaced by this module):
  - lifespan._drive_sync_worker: the poll loop with inline SQLAlchemy
    queries against DriveConnectionRow, WatchedFolderRow,
    WatchedFileRow, DocumentUploadRow, PhiAccessLogRow.

v2 changes:
  - All DB access goes through MetadataStore methods (Phase 5):
      list_active_watched_folders_due_for_sync
      list_active_watched_files_due_for_sync
      get_drive_connection
      update_drive_connection_status
      find_existing_doc_for_drive_file
      create_document_upload
      update_watched_folder_after_check
      update_watched_file_after_check
      log_phi_access
  - AN-1 resolved: both Drive-sync branches (folder-discovered files
    and standalone watched files) call the same
    `create_document_upload` method. The two-code-paths-with-slightly-
    different-field-shapes problem from v1 is gone.
  - The session-per-write pattern from v1 is preserved (each store
    method opens its own session) so the worker's behavior matches
    v1 verbatim. BD-1 (split-session cleanup) is still deferred.
"""
from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore

logger = structlog.get_logger(__name__)


async def run_drive_sync_worker(
    *,
    store: "MetadataStore",
    shutdown_event: asyncio.Event,
) -> None:
    """Background worker poll loop for Drive sync.

    Reads `CC_DRIVE_SYNC_INTERVAL_SECONDS` (default 300) from env.
    Runs until `shutdown_event` is set.
    """
    poll_interval = int(os.environ.get("CC_DRIVE_SYNC_INTERVAL_SECONDS", "300"))
    logger.info("drive_sync_worker.started", poll_interval=poll_interval)

    while not shutdown_event.is_set():
        try:
            await _sync_watched_folders(store)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(
                "drive_sync_worker.folder_poll_error",
                error=str(e),
                error_type=type(e).__name__,
            )

        try:
            await _sync_watched_files(store)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(
                "drive_sync_worker.file_poll_error",
                error=str(e),
                error_type=type(e).__name__,
            )

        # Sleep for poll_interval OR until shutdown
        try:
            await asyncio.wait_for(
                shutdown_event.wait(),
                timeout=poll_interval,
            )
            break  # shutdown signaled
        except asyncio.TimeoutError:
            pass

    logger.info("drive_sync_worker.stopped")


async def _sync_watched_folders(store: "MetadataStore") -> None:
    """One pass: check each due folder for new/modified files."""
    from ..infrastructure.drive_connector import (
        refresh_access_token,
        list_folder_files,
        read_file_text,
    )

    now = datetime.now(timezone.utc)
    due_folders = await store.list_active_watched_folders_due_for_sync(now)

    for folder in due_folders:
        try:
            # Get the connection. The folder.customer_id IS the
            # tenancy guard; get_drive_connection enforces it.
            conn = await store.get_drive_connection(
                folder.connection_id, folder.customer_id,
            )
            if conn is None or conn.status != "active":
                continue

            # Refresh access token. v1 closed the read-side session
            # before this call to keep the session short; v2's
            # store methods already do this implicitly (each method
            # opens its own session).
            try:
                access_token = await refresh_access_token(
                    conn.encrypted_refresh_token,
                    conn.token_nonce,
                )
            except Exception as e:
                logger.error(
                    "drive_sync.token_refresh_failed",
                    connection_id=conn.id,
                    error=str(e),
                )
                await store.update_drive_connection_status(
                    conn.id,
                    status="expired",
                    error_message=f"Token refresh failed: {e}",
                )
                continue

            # List files in the folder
            drive_files = await list_folder_files(
                access_token=access_token,
                folder_id=folder.folder_id,
            )

            imported_count = 0
            skipped_count = 0

            for drive_file in drive_files:
                imported = await _import_drive_file(
                    store=store,
                    access_token=access_token,
                    drive_file=drive_file,
                    customer_id=folder.customer_id,
                    connection_id=folder.connection_id,
                    contains_phi=folder.contains_phi,
                )
                if imported:
                    imported_count += 1
                else:
                    skipped_count += 1

            # Update folder tracking + connection last_synced_at
            await store.update_watched_folder_after_check(
                folder.id,
                last_checked_at=now,
                last_file_count=len(drive_files),
            )
            await store.update_drive_connection_status(
                folder.connection_id,
                status="active",
                last_synced_at=now,
                error_message=None,
            )

            if imported_count > 0:
                logger.info(
                    "drive_sync.folder_synced",
                    folder_id=folder.folder_id,
                    folder_name=folder.folder_name,
                    customer_id=folder.customer_id,
                    imported=imported_count,
                    skipped=skipped_count,
                    total_files=len(drive_files),
                )

        except Exception as e:
            logger.error(
                "drive_sync.folder_error",
                folder_id=folder.folder_id,
                error=str(e),
                error_type=type(e).__name__,
            )


async def _sync_watched_files(store: "MetadataStore") -> None:
    """One pass: check each due standalone watched file."""
    from ..infrastructure.drive_connector import (
        refresh_access_token,
        read_file_text,
        get_file_metadata,
    )

    now = datetime.now(timezone.utc)
    due_files = await store.list_active_watched_files_due_for_sync(now)

    # Group by connection to avoid repeated token refreshes
    by_conn: dict[str, list] = defaultdict(list)
    for wf in due_files:
        by_conn[wf.connection_id].append(wf)

    for conn_id, files_batch in by_conn.items():
        try:
            # Need any one watched file's customer_id to tenancy-check
            # the connection lookup. Files in this batch share a
            # connection_id by construction, and a connection has one
            # owning customer, so any file's customer_id is correct.
            sample_customer = files_batch[0].customer_id
            conn = await store.get_drive_connection(conn_id, sample_customer)
            if conn is None or conn.status != "active":
                continue

            try:
                access_token = await refresh_access_token(
                    conn.encrypted_refresh_token,
                    conn.token_nonce,
                )
            except Exception:
                continue

            for wf in files_batch:
                try:
                    # Get current metadata from Drive
                    meta = await get_file_metadata(access_token, wf.file_id)
                    modified_str = meta.get("modifiedTime", "")
                    file_modified_at = None
                    if modified_str:
                        try:
                            file_modified_at = datetime.fromisoformat(
                                modified_str.replace("Z", "+00:00")
                            )
                        except ValueError:
                            pass

                    # Decide if we need to import
                    last_mod = wf.last_modified_at
                    if last_mod is not None and hasattr(last_mod, "tzinfo") and last_mod.tzinfo is None:
                        last_mod = last_mod.replace(tzinfo=timezone.utc)

                    needs_import = (
                        last_mod is None
                        or (file_modified_at is not None and file_modified_at > last_mod)
                    )

                    if not needs_import:
                        # Just touch last_checked_at and move on
                        await store.update_watched_file_after_check(
                            wf.id,
                            last_checked_at=now,
                        )
                        continue

                    # Read and import
                    mime = wf.mime_type or meta.get("mimeType", "")
                    text = await read_file_text(access_token, wf.file_id, mime)

                    if not text or not text.strip():
                        continue

                    await store.create_document_upload(
                        customer_id=wf.customer_id,
                        label=wf.file_name,
                        text=text,
                        crystal_type="customer:legacy",
                        source_file_id=wf.file_id,
                        source_modified_at=file_modified_at,
                        source_connection_id=wf.connection_id,
                    )

                    # PHI audit if flagged
                    if wf.contains_phi:
                        await store.log_phi_access(
                            customer_id=wf.customer_id,
                            action="read_drive_file",
                            resource_type="drive_file",
                            resource_id=wf.file_id,
                            resource_name=wf.file_name,
                            contains_phi=True,
                            source_connection_id=wf.connection_id,
                        )

                    # Update watched file tracking
                    await store.update_watched_file_after_check(
                        wf.id,
                        last_checked_at=now,
                        last_modified_at=file_modified_at,
                    )

                    logger.info(
                        "drive_sync.file_synced",
                        file_id=wf.file_id,
                        file_name=wf.file_name,
                        customer_id=wf.customer_id,
                    )

                except Exception as e:
                    logger.warning(
                        "drive_sync.watched_file_error",
                        file_id=wf.file_id,
                        error=str(e),
                    )

        except Exception as e:
            logger.error(
                "drive_sync.file_conn_error",
                connection_id=conn_id,
                error=str(e),
            )


async def _import_drive_file(
    *,
    store: "MetadataStore",
    access_token: str,
    drive_file: dict,
    customer_id: str,
    connection_id: str,
    contains_phi: bool,
) -> bool:
    """Import one Drive file as a pending DocumentUpload.

    Returns True if the file was imported (new or modified), False if
    it was skipped (already imported, unchanged, or empty).

    AN-1 resolved: this is the single code path for "create document
    upload from Drive file." v1 had two near-identical paths in the
    folder-sync worker; v2 funnels them through one helper that calls
    the unified MetadataStore method.
    """
    from ..infrastructure.drive_connector import read_file_text

    file_id = drive_file["id"]
    file_name = drive_file.get("name", "Untitled")
    mime_type = drive_file.get("mimeType", "")
    modified_time_str = drive_file.get("modifiedTime", "")

    # Parse modifiedTime (RFC 3339)
    file_modified_at = None
    if modified_time_str:
        try:
            file_modified_at = datetime.fromisoformat(
                modified_time_str.replace("Z", "+00:00")
            )
        except ValueError:
            pass

    # Dedup: have we seen this file before?
    existing = await store.find_existing_doc_for_drive_file(
        customer_id=customer_id,
        source_file_id=file_id,
    )

    if existing is not None:
        existing_modified = existing.source_modified_at
        if existing_modified is not None and existing_modified.tzinfo is None:
            existing_modified = existing_modified.replace(tzinfo=timezone.utc)
        if (
            existing_modified is not None
            and file_modified_at is not None
            and file_modified_at <= existing_modified
        ):
            return False  # already imported, unchanged
        logger.info(
            "drive_sync.file_updated",
            file_id=file_id,
            file_name=file_name,
            customer_id=customer_id,
        )
    else:
        logger.info(
            "drive_sync.new_file",
            file_id=file_id,
            file_name=file_name,
            customer_id=customer_id,
        )

    # Read file content
    try:
        text = await read_file_text(
            access_token=access_token,
            file_id=file_id,
            mime_type=mime_type,
        )
    except Exception as e:
        logger.warning(
            "drive_sync.read_failed",
            file_id=file_id,
            error=str(e),
        )
        return False

    if not text or not text.strip():
        return False

    # Insert as pending document. Single store method call replaces
    # v1's two slightly-different inline INSERT shapes.
    await store.create_document_upload(
        customer_id=customer_id,
        label=file_name,
        text=text,
        crystal_type="customer:legacy",
        source_file_id=file_id,
        source_modified_at=file_modified_at,
        source_connection_id=connection_id,
    )

    # PHI audit if the parent folder is flagged
    if contains_phi:
        await store.log_phi_access(
            customer_id=customer_id,
            action="read_drive_file",
            resource_type="drive_file",
            resource_id=file_id,
            resource_name=file_name,
            contains_phi=True,
            source_connection_id=connection_id,
        )

    return True
