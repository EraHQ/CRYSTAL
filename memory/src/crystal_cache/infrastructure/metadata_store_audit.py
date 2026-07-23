"""Audit-table CRUD methods — Phase 5 of the v2 port.

This module provides the 42 narrow methods that handle the eight
"audit tables" identified in Phase 1's AUDIT.md: document_uploads,
drive_connections, watched_folders, watched_files, baa_tracking,
phi_access_log (dict-shaped, no Pydantic model), push_review_queue,
knowledge_gaps, and cognition_tasks.

Architectural note (per ledger AN-7 — "If metadata_store.py becomes
unwieldy, we split into multiple files under a metadata_store/
package later, behind the same MetadataStore facade"): rather than
extend metadata_store.py from ~117 KB to ~150 KB, we put these
methods in a mixin class. The mixin's methods are bound onto
MetadataStore at import time via setattr in
infrastructure/__init__.py (see D12). The mixin is NOT in
MetadataStore's MRO — the binding is attribute-level, not
inheritance-level. The public surface is still unchanged:
`store.create_document_upload(...)` works exactly as if the method
were defined inline on MetadataStore.

Why setattr and not class inheritance:
  - Avoids editing the existing 117 KB metadata_store.py
  - Easier diff review for the Phase 5 work
  - D12 documents the tradeoff: type checkers and some IDEs won't
    see the bound methods as MetadataStore attributes without a
    .pyi stub. CU-6 tracks the eventual fix.

Per D1: `phi_access_log` stays dict-in/dict-out at this boundary.
No Pydantic model; the `log_phi_access` method takes keyword
arguments and returns nothing.

Per AN-4 (corrected during Phase 6.5 gloss audit; CU-10 closed): the
`claim_pending_documents_batch` and `claim_pending_cognition_task`
claims now SELECT with `FOR UPDATE SKIP LOCKED`, so concurrent
Postgres workers lock their rows and skip rows another worker already
holds — claims are disjoint. On SQLite the hint is a no-op (the
dialect omits it) and the single-writer SERIALIZABLE transaction
already made the SELECT+mutation atomic.

Per AN-6: `list_knowledge_gaps_with_filled_content` encapsulates
the 1+N FactRow lookup that admin.py does inline in v1. The N+1
read pattern survives — it's just hidden behind the store
boundary so the admin handler is clean. Limit=50 makes it
acceptable.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import structlog
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import (
    BaaTracking,
    CognitionTask,
    DocumentUpload,
    DriveConnection,
    KnowledgeGap,
    PushReviewItem,
    WatchedFile,
    WatchedFolder,
)
from .schema import (
    BaaTrackingRow,
    CognitionTaskRow,
    DocumentUploadRow,
    DriveConnectionRow,
    FactRow,
    KnowledgeGapRow,
    PhiAccessLogRow,
    PushReviewQueueRow,
    WatchedFileRow,
    WatchedFolderRow,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Mixin: 42 audit-table CRUD methods + 1 dict-shaped PHI log method
# ---------------------------------------------------------------------------

class AuditTablesMixin:
    """Methods for the eight audit tables, bound onto MetadataStore.

    The binding happens in `infrastructure/__init__.py` at module
    import time via `setattr(MetadataStore, name, method)` for every
    public method on this class. The mixin is NOT in MetadataStore's
    MRO; `self.session()` inside a mixin method resolves to
    MetadataStore.session via normal attribute lookup on the bound
    callable.

    The mixin doesn't need any state of its own; it only calls
    `self.session()` which the concrete class provides.
    """

    # =================================================================
    # document_uploads — 12 methods
    # =================================================================

    async def create_document_upload(
        self,
        customer_id: str,
        label: str,
        text: str,
        crystal_type: str = "customer:legacy",
        *,
        source_file_id: Optional[str] = None,
        source_modified_at: Optional[datetime] = None,
        source_connection_id: Optional[str] = None,
        source_uri: Optional[str] = None,
        detected_type: Optional[str] = None,
        scope: Optional[str] = None,
        owner_operator_id: Optional[str] = None,
    ) -> DocumentUpload:
        """Insert a new document with status='pending'.

        The crystallization worker picks it up on next poll. Used by:
        SDK upload endpoints (file + JSON paths), Drive sync worker
        (per-file and watched-file branches), gdrive_import_files
        batch endpoint, and cognition's commit path
        (detected_type='inferred_knowledge').

        AN-1: this method unifies the two Drive-sync code paths that
        v1 had for the same write. Callers pass `source_connection_id`
        or omit it; the row gets None when omitted.
        """
        import hashlib
        import uuid
        doc_id = f"doc_{uuid.uuid4().hex[:16]}"
        now = datetime.now(timezone.utc)
        char_count = len(text)
        # Gate D (C1 ratified): the identity pair. Location identity is
        # scheme-qualified — a Drive file keeps its Drive identity across
        # re-syncs; a plain upload is its own place. Content identity is
        # the sha256 of the EXTRACTED TEXT (survives PDF re-saves).
        # Gate M: an envelope-supplied URI is the authority (the
        # watcher KNOWS the canonical identity — repo://<name>/<path>);
        # derivation is the fallback for direct uploads.
        source_uri = source_uri or (
            f"gdrive://{source_file_id}" if source_file_id
            else f"upload://{doc_id}"
        )
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        async with self.session() as session:  # type: ignore[attr-defined]
            row = DocumentUploadRow(
                id=doc_id,
                customer_id=customer_id,
                label=label,
                text=text,
                status="pending",
                crystal_type=crystal_type,
                char_count=char_count,
                crystals_written=0,
                items_extracted=0,
                source_file_id=source_file_id,
                source_modified_at=source_modified_at,
                source_connection_id=source_connection_id,
                source_uri=source_uri,
                content_hash=content_hash,
                detected_type=detected_type,
                scope=scope,
                owner_operator_id=owner_operator_id,
                created_at=now,
            )
            session.add(row)

        return DocumentUpload(
            id=doc_id,
            customer_id=customer_id,
            label=label,
            text=text,
            status="pending",
            crystal_type=crystal_type,
            char_count=char_count,
            source_file_id=source_file_id,
            source_modified_at=source_modified_at,
            source_connection_id=source_connection_id,
            source_uri=source_uri,
            content_hash=content_hash,
            detected_type=detected_type,
            scope=scope,
            owner_operator_id=owner_operator_id,
            created_at=now,
        )

    async def get_document_upload(
        self, document_id: str, customer_id: str
    ) -> Optional[DocumentUpload]:
        """Get by id with tenancy check. None if not found OR
        belongs to another tenant. The tenancy check is the win
        over v1's inline pattern where every handler did this
        manually."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(DocumentUploadRow, document_id)
            if row is None or row.customer_id != customer_id:
                return None
            return _document_upload_from_row(row)

    async def list_document_uploads(
        self,
        customer_id: str,
        *,
        status: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[DocumentUpload]:
        """List by customer, optional status filter, ordered by
        created_at DESC."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(DocumentUploadRow)
                .where(DocumentUploadRow.customer_id == customer_id)
                .order_by(DocumentUploadRow.created_at.desc())
            )
            if status is not None:
                stmt = stmt.where(DocumentUploadRow.status == status)
            if limit is not None:
                stmt = stmt.limit(limit)
            result = await session.execute(stmt)
            return [
                _document_upload_from_row(r)
                for r in result.scalars().all()
            ]

    async def list_pending_documents_for_processing(
        self,
        *,
        limit: int,
    ) -> list[DocumentUpload]:
        """Cross-tenant: used by the crystallization worker's poll
        loop. Status='pending', any customer, ordered by created_at
        ASC so the oldest pending docs process first."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(DocumentUploadRow)
                .where(DocumentUploadRow.status == "pending")
                .order_by(DocumentUploadRow.created_at.asc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return [
                _document_upload_from_row(r)
                for r in result.scalars().all()
            ]

    async def find_existing_doc_for_drive_file(
        self,
        customer_id: str,
        source_file_id: str,
    ) -> Optional[DocumentUpload]:
        """Drive sync dedup — find the latest doc with this
        source_file_id. Returns None if never imported. Used by the
        per-file Drive sync branch to avoid re-importing files that
        haven't changed."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(DocumentUploadRow)
                .where(DocumentUploadRow.customer_id == customer_id)
                .where(DocumentUploadRow.source_file_id == source_file_id)
                .order_by(DocumentUploadRow.created_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            return _document_upload_from_row(row) if row else None

    async def mark_document_crystallizing(
        self, document_id: str
    ) -> None:
        """Status pending → crystallizing. Used to claim a single
        document for processing. Race-free per-row update via
        Postgres-native semantics; SQLite users have no concurrent
        writers so this is fine.

        Per Phase 6.5 P0.1, the status string is the v1 name
        ('crystallizing'), not 'processing' as an earlier port
        attempt used."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(DocumentUploadRow, document_id)
            if row is not None:
                row.status = "crystallizing"

    async def release_document_to_pending(self, document_id: str) -> None:
        """Release a claimed document back to 'pending' (cost 1c
        per-customer gate: the claim was made before the customer's
        budget was known; releasing keeps the claim atomic and the
        retry free)."""
        from sqlalchemy import update
        async with self.session() as session:
            await session.execute(
                update(DocumentUploadRow)
                .where(DocumentUploadRow.id == document_id)
                .values(status="pending")
            )
            await session.commit()

    async def claim_pending_documents_batch(
        self, limit: int
    ) -> list[DocumentUpload]:
        """Atomic-on-SQLite: select N pending docs AND mark them
        'crystallizing' in one transaction. Returns the claimed rows.

        AN-4 (CU-10 closed): the SELECT takes `FOR UPDATE SKIP
        LOCKED`, so under Postgres with multiple workers each claim
        locks its rows and skips rows another worker already holds —
        claims are disjoint. On SQLite the hint is a no-op and the
        session's SERIALIZABLE single-writer transaction already made
        the SELECT+mutation atomic.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            # Select pending docs (locking: skip rows another worker
            # already holds, so concurrent Postgres workers claim
            # disjoint batches; no-op on SQLite).
            stmt = (
                select(DocumentUploadRow)
                .where(DocumentUploadRow.status == "pending")
                .order_by(DocumentUploadRow.created_at.asc())
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
            result = await session.execute(stmt)
            rows = list(result.scalars().all())

            # Mark each claimed in the same transaction
            for row in rows:
                row.status = "crystallizing"

            return [_document_upload_from_row(r) for r in rows]

    async def mark_document_review_ready(
        self,
        document_id: str,
        *,
        detected_type: str,
        content_chunks: list[dict[str, Any]],
        extracted_items: list[dict[str, Any]],
        items_extracted_count: int,
    ) -> None:
        """Worker finished extraction; document waits for user
        review/approval. Status crystallizing → review."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(DocumentUploadRow, document_id)
            if row is not None:
                row.status = "review"
                row.detected_type = detected_type
                row.content_chunks = content_chunks
                row.extracted_items = extracted_items
                row.items_extracted = items_extracted_count

    async def mark_document_error(
        self, document_id: str, error_message: str
    ) -> None:
        """Status → error. Used by worker on extraction failure
        AND by manual crystallize on pipeline failure."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(DocumentUploadRow, document_id)
            if row is not None:
                row.status = "error"
                row.error_message = error_message

    async def mark_document_crystallized(
        self,
        document_id: str,
        *,
        crystals_written: int,
        items_extracted: int,
        crystallized_at: datetime,
    ) -> None:
        """Final state. Status → crystallized.

        Per Phase 6.5 P0.1, the status string is the v1 name
        ('crystallized'), not 'complete' as an earlier port attempt
        used. The DocumentUploadStatus Literal in the Pydantic
        model is the contract; matches v1 verbatim."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(DocumentUploadRow, document_id)
            if row is not None:
                row.status = "crystallized"
                row.crystals_written = crystals_written
                row.items_extracted = items_extracted
                row.crystallized_at = crystallized_at

    async def set_document_scope(
        self, document_id: str, customer_id: str, scope: str,
    ) -> bool:
        """Restamp a document source's scope (share-source, P4). Future
        crystallization of this document inherits the new scope; the
        already-born crystal set is flipped by the endpoint via
        set_crystal_scope. Tenancy-checked."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(DocumentUploadRow, document_id)
            if row is None or row.customer_id != customer_id:
                return False
            row.scope = scope
            return True

    async def update_document_review_edits(
        self,
        document_id: str,
        customer_id: str,
        *,
        extracted_items: Optional[list[dict[str, Any]]] = None,
        content_chunks: Optional[list[dict[str, Any]]] = None,
        confirmed_type: Optional[str] = None,
    ) -> None:
        """User edits during review. Only updates provided fields.
        Tenancy-checked: silently no-ops if row belongs to another
        customer (callers should get_document_upload first to
        detect that case)."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(DocumentUploadRow, document_id)
            if row is None or row.customer_id != customer_id:
                return
            if extracted_items is not None:
                row.extracted_items = extracted_items
            if content_chunks is not None:
                row.content_chunks = content_chunks
            if confirmed_type is not None:
                row.confirmed_type = confirmed_type

    async def save_approval_edits_and_mark_crystallizing(
        self,
        document_id: str,
        *,
        items: list[dict[str, Any]],
        content_chunks: list[dict[str, Any]],
    ) -> None:
        """Atomic step at start of approval: save the final edits
        AND transition to crystallizing in one go. Replaces v1's
        two-step update which had a window between save and
        transition."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(DocumentUploadRow, document_id)
            if row is not None:
                row.extracted_items = items
                row.content_chunks = content_chunks
                row.status = "crystallizing"

    async def delete_document_upload(
        self, document_id: str, customer_id: str
    ) -> None:
        """Hard delete. Tenancy-checked: silent no-op if the row
        belongs to another customer."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(DocumentUploadRow, document_id)
            if row is not None and row.customer_id == customer_id:
                await session.delete(row)

    # =================================================================
    # drive_connections — 5 methods
    # =================================================================

    async def create_oauth_state(self, state: str, customer_id: str) -> None:
        """Persist one issued OAuth state nonce (F1 CSRF fix, 2026-07-03)."""
        from .schema import OAuthStateRow

        async with self.session() as session:
            session.add(OAuthStateRow(
                state=state,
                customer_id=customer_id,
                created_at=datetime.now(timezone.utc),
            ))

    async def consume_oauth_state(
        self, state: str, *, max_age_seconds: int = 600,
    ) -> Optional[str]:
        """Single-use redemption of an OAuth state nonce.

        Returns the customer_id when the state exists and is younger than
        max_age_seconds, else None. The row is DELETED either way when
        found — a state can never be redeemed twice, and stale rows clean
        themselves up on the failed attempt.
        """
        from .schema import OAuthStateRow

        async with self.session() as session:
            row = await session.get(OAuthStateRow, state)
            if row is None:
                return None
            created = row.created_at
            if created is not None and created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            customer_id = row.customer_id
            await session.delete(row)
            if created is None:
                return None
            age = (datetime.now(timezone.utc) - created).total_seconds()
            if age > max_age_seconds:
                return None
            return customer_id

    async def create_drive_connection(
        self,
        customer_id: str,
        *,
        connection_id: str,
        email: Optional[str],
        encrypted_refresh_token: str,
        token_nonce: str,
        scopes: Optional[str],
        provider: str = "google_drive",
    ) -> DriveConnection:
        """Persist a new OAuth connection after token exchange.
        The caller (OAuth callback handler) is responsible for
        encrypting the refresh token via token_crypto BEFORE calling
        this method. We never see the plaintext."""
        now = datetime.now(timezone.utc)
        async with self.session() as session:  # type: ignore[attr-defined]
            row = DriveConnectionRow(
                id=connection_id,
                customer_id=customer_id,
                provider=provider,
                email=email,
                encrypted_refresh_token=encrypted_refresh_token,
                token_nonce=token_nonce,
                scopes=scopes,
                status="active",
                created_at=now,
                updated_at=now,
            )
            session.add(row)

        return DriveConnection(
            id=connection_id,
            customer_id=customer_id,
            provider=provider,
            email=email,
            encrypted_refresh_token=encrypted_refresh_token,
            token_nonce=token_nonce,
            scopes=scopes,
            status="active",
            created_at=now,
            updated_at=now,
        )

    async def get_drive_connection(
        self, connection_id: str, customer_id: str
    ) -> Optional[DriveConnection]:
        """Get by id with tenancy check. The DriveConnection model
        exposes encrypted_refresh_token + token_nonce verbatim;
        callers needing the plaintext call into token_crypto with
        these values."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(DriveConnectionRow, connection_id)
            if row is None or row.customer_id != customer_id:
                return None
            return _drive_connection_from_row(row)

    async def list_drive_connections(
        self, customer_id: str
    ) -> list[DriveConnection]:
        """All connections for a customer, ordered by created_at
        DESC. Most recently authorized first."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(DriveConnectionRow)
                .where(DriveConnectionRow.customer_id == customer_id)
                .order_by(DriveConnectionRow.created_at.desc())
            )
            result = await session.execute(stmt)
            return [
                _drive_connection_from_row(r)
                for r in result.scalars().all()
            ]

    async def update_drive_connection_status(
        self,
        connection_id: str,
        *,
        status: str,
        error_message: Optional[str] = None,
        last_synced_at: Optional[datetime] = None,
    ) -> None:
        """Status transitions and sync tracking. Used by the drive
        sync worker on:
          - token refresh success: status='active',
            last_synced_at=now
          - token refresh failure: status='expired' or 'error',
            error_message set
          - manual user disconnect: status='revoked'

        No tenancy check — workers operate cross-tenant. The
        connection_id is itself a strong identifier."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(DriveConnectionRow, connection_id)
            if row is None:
                return
            row.status = status
            row.updated_at = datetime.now(timezone.utc)
            if error_message is not None:
                row.error_message = error_message
            if last_synced_at is not None:
                row.last_synced_at = last_synced_at

    async def delete_drive_connection_with_watches(
        self, connection_id: str, customer_id: str
    ) -> None:
        """Cascading delete: removes watched_folders + watched_files
        + the connection itself. Tenancy-checked at the connection
        level.

        All three deletes happen in one session/transaction; either
        the whole connection unwinds or none of it does."""
        async with self.session() as session:  # type: ignore[attr-defined]
            # Verify the connection exists and belongs to this customer
            conn_row = await session.get(DriveConnectionRow, connection_id)
            if conn_row is None or conn_row.customer_id != customer_id:
                return

            # Delete child watched_files
            files_stmt = select(WatchedFileRow).where(
                WatchedFileRow.connection_id == connection_id
            )
            for file_row in (await session.execute(files_stmt)).scalars():
                await session.delete(file_row)

            # Delete child watched_folders
            folders_stmt = select(WatchedFolderRow).where(
                WatchedFolderRow.connection_id == connection_id
            )
            for folder_row in (await session.execute(folders_stmt)).scalars():
                await session.delete(folder_row)

            # Delete the connection itself
            await session.delete(conn_row)

    # =================================================================
    # watched_folders — 5 methods
    # =================================================================

    async def create_watched_folder(
        self,
        *,
        watch_id: str,
        connection_id: str,
        customer_id: str,
        folder_id: str,
        folder_name: str,
        folder_path: Optional[str],
        contains_phi: bool,
        sync_interval_minutes: int,
    ) -> WatchedFolder:
        """Add a Drive folder to the sync watch list."""
        now = datetime.now(timezone.utc)
        async with self.session() as session:  # type: ignore[attr-defined]
            row = WatchedFolderRow(
                id=watch_id,
                connection_id=connection_id,
                customer_id=customer_id,
                folder_id=folder_id,
                folder_name=folder_name,
                folder_path=folder_path,
                contains_phi=contains_phi,
                sync_interval_minutes=sync_interval_minutes,
                status="active",
                created_at=now,
            )
            session.add(row)

        return WatchedFolder(
            id=watch_id,
            connection_id=connection_id,
            customer_id=customer_id,
            folder_id=folder_id,
            folder_name=folder_name,
            folder_path=folder_path,
            contains_phi=contains_phi,
            sync_interval_minutes=sync_interval_minutes,
            status="active",
            created_at=now,
        )

    async def get_watched_folder(
        self, watch_id: str, customer_id: str
    ) -> Optional[WatchedFolder]:
        """Get by id with tenancy check."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(WatchedFolderRow, watch_id)
            if row is None or row.customer_id != customer_id:
                return None
            return _watched_folder_from_row(row)

    async def list_watched_folders_for_connection(
        self, connection_id: str, customer_id: str
    ) -> list[WatchedFolder]:
        """All watched folders under one connection. Tenancy-checked
        via the customer_id filter; the connection's customer match
        is enforced by the connection's own tenancy."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(WatchedFolderRow)
                .where(WatchedFolderRow.connection_id == connection_id)
                .where(WatchedFolderRow.customer_id == customer_id)
                .order_by(WatchedFolderRow.created_at.desc())
            )
            result = await session.execute(stmt)
            return [
                _watched_folder_from_row(r)
                for r in result.scalars().all()
            ]

    # ------------------------------------------------------------------
    # Source watches (Gate M, 2026-07-18) — the GENERAL registration.
    # Git is the first tenant; every future scheme lands here too.
    # ------------------------------------------------------------------

    async def create_source_watch(
        self, customer_id: str, *, scheme: str, source_name: str,
        config: dict, cadence_minutes: int = 15,
        review_mode: str = "auto", encrypted_token: "str | None" = None,
    ) -> "SourceWatch":
        import uuid as _uuid
        from ..models.source_watch import SourceWatch
        from .schema import SourceWatchRow
        watch_id = f"watch_{_uuid.uuid4().hex[:16]}"
        now = datetime.now(timezone.utc)
        async with self.session() as session:
            session.add(SourceWatchRow(
                id=watch_id, customer_id=customer_id, scheme=scheme,
                source_name=source_name, config=config,
                cadence_minutes=cadence_minutes, review_mode=review_mode,
                encrypted_token=encrypted_token, status="active",
                created_at=now,
            ))
            await session.commit()
        return SourceWatch(
            id=watch_id, customer_id=customer_id, scheme=scheme,
            source_name=source_name, config=config,
            cadence_minutes=cadence_minutes, review_mode=review_mode,
            encrypted_token=encrypted_token, status="active",
            created_at=now,
        )

    async def get_source_watch(
        self, watch_id: str, customer_id: str,
    ) -> "SourceWatch | None":
        from .schema import SourceWatchRow
        async with self.session() as session:
            row = (await session.execute(
                select(SourceWatchRow).where(
                    SourceWatchRow.id == watch_id,
                    SourceWatchRow.customer_id == customer_id,
                )
            )).scalar_one_or_none()
        return _source_watch_from_row(row) if row else None

    async def list_source_watches(
        self, customer_id: str,
    ) -> "list[SourceWatch]":
        from .schema import SourceWatchRow
        async with self.session() as session:
            rows = (await session.execute(
                select(SourceWatchRow)
                .where(SourceWatchRow.customer_id == customer_id)
                .order_by(SourceWatchRow.created_at)
            )).scalars().all()
        return [_source_watch_from_row(r) for r in rows]

    async def list_source_watches_due(
        self, now: datetime,
    ) -> "list[SourceWatch]":
        """Active watches whose cadence has elapsed (or never checked).
        The sync worker's due-cycle query — same shape as the Drive
        folders' due listing, scheme-agnostic."""
        from .schema import SourceWatchRow
        async with self.session() as session:
            rows = (await session.execute(
                select(SourceWatchRow).where(
                    SourceWatchRow.status == "active",
                )
            )).scalars().all()
        due = []
        for r in rows:
            if r.last_checked_at is None:
                due.append(r)
                continue
            last = r.last_checked_at
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            elapsed = (now - last).total_seconds() / 60.0
            if elapsed >= max(1, r.cadence_minutes):
                due.append(r)
        return [_source_watch_from_row(r) for r in due]

    async def update_source_watch_state(
        self, watch_id: str, customer_id: str, *,
        last_state: "dict | None" = None,
        last_error: "str | None" = None,
        checked_at: "datetime | None" = None,
    ) -> None:
        from .schema import SourceWatchRow
        async with self.session() as session:
            row = (await session.execute(
                select(SourceWatchRow).where(
                    SourceWatchRow.id == watch_id,
                    SourceWatchRow.customer_id == customer_id,
                )
            )).scalar_one_or_none()
            if row is None:
                return
            if last_state is not None:
                row.last_state = last_state
            row.last_error = last_error
            row.last_checked_at = checked_at or datetime.now(timezone.utc)
            await session.commit()

    async def set_source_watch_status(
        self, watch_id: str, customer_id: str, status: str,
    ) -> bool:
        from .schema import SourceWatchRow
        async with self.session() as session:
            row = (await session.execute(
                select(SourceWatchRow).where(
                    SourceWatchRow.id == watch_id,
                    SourceWatchRow.customer_id == customer_id,
                )
            )).scalar_one_or_none()
            if row is None:
                return False
            row.status = status
            await session.commit()
            return True

    async def delete_source_watch(
        self, watch_id: str, customer_id: str,
    ) -> bool:
        from .schema import SourceWatchRow
        async with self.session() as session:
            row = (await session.execute(
                select(SourceWatchRow).where(
                    SourceWatchRow.id == watch_id,
                    SourceWatchRow.customer_id == customer_id,
                )
            )).scalar_one_or_none()
            if row is None:
                return False
            await session.delete(row)
            await session.commit()
            return True

    async def find_chat_crystals_with_text(
        self, customer_id: str, needle: str, limit: int = 5,
    ) -> "list[str]":
        """Crystal ids whose facts contain `needle` (Gate F slice 2:
        message-id lookup for email reply chains). Bounded LIKE scan —
        callers apply the unique-match discipline on the result."""
        from .schema import CrystalRow, FactRow
        async with self.session() as session:
            rows = (await session.execute(
                select(FactRow.crystal_id)
                .join(CrystalRow, CrystalRow.id == FactRow.crystal_id)
                .where(
                    CrystalRow.customer_id == customer_id,
                    FactRow.claim_text.like(f"%{needle}%"),
                )
                .distinct()
                .limit(limit)
            )).scalars().all()
        return list(rows)

    async def count_inflight_uploads_by_label_prefix(
        self, customer_id: str, prefix: str,
    ) -> int:
        """How many of a watch's files are mid-pipeline right now —
        the 'syncing' signal for the panel's state chip."""
        from sqlalchemy import func
        async with self.session() as session:
            n = (await session.execute(
                select(func.count(DocumentUploadRow.id)).where(
                    DocumentUploadRow.customer_id == customer_id,
                    DocumentUploadRow.label.like(prefix + "%"),
                    DocumentUploadRow.status.in_(
                        ("pending", "crystallizing")
                    ),
                )
            )).scalar_one()
        return int(n or 0)

    async def list_active_watched_folders_due_for_sync(
        self, now: datetime
    ) -> list[WatchedFolder]:
        """Drive sync worker entry point. Returns folders with
        status='active' where last_checked_at + sync_interval has
        passed (or last_checked_at IS NULL meaning never synced).

        Cross-tenant — workers operate on all customers.

        Implementation note: SQLAlchemy's date arithmetic across
        dialects is awkward. We filter `status='active'` AND
        `(last_checked_at IS NULL OR last_checked_at < now)` in SQL,
        then post-filter by sync interval in Python. For dev/MVP scale
        this is fine; production scale would push interval comparison
        into SQL via DATEADD/INTERVAL functions per dialect."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(WatchedFolderRow)
                .where(WatchedFolderRow.status == "active")
            )
            result = await session.execute(stmt)
            all_folders = list(result.scalars().all())

        # Post-filter by sync interval in Python
        due = []
        for row in all_folders:
            if row.last_checked_at is None:
                due.append(row)
                continue
            # Compute time-since-last-check
            elapsed = (now - row.last_checked_at).total_seconds() / 60.0
            if elapsed >= row.sync_interval_minutes:
                due.append(row)

        return [_watched_folder_from_row(r) for r in due]

    async def update_watched_folder_after_check(
        self,
        watch_id: str,
        *,
        last_checked_at: datetime,
        last_file_count: int,
    ) -> None:
        """Sync worker updates after scanning a folder."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(WatchedFolderRow, watch_id)
            if row is not None:
                row.last_checked_at = last_checked_at
                row.last_file_count = last_file_count

    async def delete_watched_folder(
        self, watch_id: str, customer_id: str
    ) -> None:
        """Remove a folder from the watch list. Tenancy-checked."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(WatchedFolderRow, watch_id)
            if row is not None and row.customer_id == customer_id:
                await session.delete(row)

    # =================================================================
    # watched_files — 5 methods (parallel to watched_folders)
    # =================================================================

    async def create_watched_file(
        self,
        *,
        watch_id: str,
        connection_id: str,
        customer_id: str,
        file_id: str,
        file_name: str,
        mime_type: Optional[str],
        contains_phi: bool,
        sync_interval_minutes: int,
    ) -> WatchedFile:
        """Add a Drive file to the per-file watch list (distinct
        from being inside a watched folder)."""
        now = datetime.now(timezone.utc)
        async with self.session() as session:  # type: ignore[attr-defined]
            row = WatchedFileRow(
                id=watch_id,
                connection_id=connection_id,
                customer_id=customer_id,
                file_id=file_id,
                file_name=file_name,
                mime_type=mime_type,
                contains_phi=contains_phi,
                sync_interval_minutes=sync_interval_minutes,
                status="active",
                created_at=now,
            )
            session.add(row)

        return WatchedFile(
            id=watch_id,
            connection_id=connection_id,
            customer_id=customer_id,
            file_id=file_id,
            file_name=file_name,
            mime_type=mime_type,
            contains_phi=contains_phi,
            sync_interval_minutes=sync_interval_minutes,
            status="active",
            created_at=now,
        )

    async def get_watched_file(
        self, watch_id: str, customer_id: str
    ) -> Optional[WatchedFile]:
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(WatchedFileRow, watch_id)
            if row is None or row.customer_id != customer_id:
                return None
            return _watched_file_from_row(row)

    async def list_watched_files_for_connection(
        self, connection_id: str, customer_id: str
    ) -> list[WatchedFile]:
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(WatchedFileRow)
                .where(WatchedFileRow.connection_id == connection_id)
                .where(WatchedFileRow.customer_id == customer_id)
                .order_by(WatchedFileRow.created_at.desc())
            )
            result = await session.execute(stmt)
            return [
                _watched_file_from_row(r)
                for r in result.scalars().all()
            ]

    async def list_active_watched_files_due_for_sync(
        self, now: datetime
    ) -> list[WatchedFile]:
        """Same shape as list_active_watched_folders_due_for_sync."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(WatchedFileRow)
                .where(WatchedFileRow.status == "active")
            )
            result = await session.execute(stmt)
            all_files = list(result.scalars().all())

        due = []
        for row in all_files:
            if row.last_checked_at is None:
                due.append(row)
                continue
            elapsed = (now - row.last_checked_at).total_seconds() / 60.0
            if elapsed >= row.sync_interval_minutes:
                due.append(row)

        return [_watched_file_from_row(r) for r in due]

    async def update_watched_file_after_check(
        self,
        watch_id: str,
        *,
        last_checked_at: datetime,
        last_modified_at: Optional[datetime] = None,
    ) -> None:
        """Sync worker updates after fetching a file's metadata."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(WatchedFileRow, watch_id)
            if row is not None:
                row.last_checked_at = last_checked_at
                if last_modified_at is not None:
                    row.last_modified_at = last_modified_at

    async def delete_watched_file(
        self, watch_id: str, customer_id: str
    ) -> None:
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(WatchedFileRow, watch_id)
            if row is not None and row.customer_id == customer_id:
                await session.delete(row)

    # =================================================================
    # baa_tracking — 2 methods
    # =================================================================

    async def get_baa_record(
        self, customer_id: str
    ) -> Optional[BaaTracking]:
        """One row per customer (UNIQUE constraint on customer_id).
        Returns None if no BAA has been recorded yet."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = select(BaaTrackingRow).where(
                BaaTrackingRow.customer_id == customer_id
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            return _baa_tracking_from_row(row) if row else None

    async def upsert_baa_record(
        self,
        customer_id: str,
        *,
        baa_signed: Optional[bool] = None,
        baa_signed_date: Optional[datetime] = None,
        baa_document_ref: Optional[str] = None,
        phi_data_sources: Optional[list[str]] = None,
        hipaa_contact_email: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> BaaTracking:
        """Create if missing, patch the supplied fields if exists.
        Returns the post-state. Used by admin's compliance UI."""
        import uuid
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = select(BaaTrackingRow).where(
                BaaTrackingRow.customer_id == customer_id
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()

            now = datetime.now(timezone.utc)
            if row is None:
                row = BaaTrackingRow(
                    id=f"baa_{uuid.uuid4().hex[:16]}",
                    customer_id=customer_id,
                    baa_signed=baa_signed or False,
                    baa_signed_date=baa_signed_date,
                    baa_document_ref=baa_document_ref,
                    phi_data_sources=phi_data_sources,
                    hipaa_contact_email=hipaa_contact_email,
                    notes=notes,
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)
            else:
                if baa_signed is not None:
                    row.baa_signed = baa_signed
                if baa_signed_date is not None:
                    row.baa_signed_date = baa_signed_date
                if baa_document_ref is not None:
                    row.baa_document_ref = baa_document_ref
                if phi_data_sources is not None:
                    row.phi_data_sources = phi_data_sources
                if hipaa_contact_email is not None:
                    row.hipaa_contact_email = hipaa_contact_email
                if notes is not None:
                    row.notes = notes
                row.updated_at = now

            # session.flush() not needed; commit happens at context exit
            return _baa_tracking_from_row(row)

    # =================================================================
    # phi_access_log — 1 method (dict-shaped per D1)
    # =================================================================

    async def log_phi_access(
        self,
        *,
        customer_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        resource_name: Optional[str] = None,
        contains_phi: bool = True,
        source_connection_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> None:
        """Fire-and-forget HIPAA audit log write.

        Per D1: dict-in (kwargs), no return value. Caller cannot
        introspect the row from this method; reads against
        phi_access_log go through a separate dict-returning method
        if/when needed.

        Never raises on DB failure — the audit is best-effort. A
        failed log write logs a warning and continues. This is
        deliberate: blocking the actual operation on audit-write
        success would be worse than a missing audit entry."""
        import uuid
        try:
            async with self.session() as session:  # type: ignore[attr-defined]
                row = PhiAccessLogRow(
                    id=f"phi_{uuid.uuid4().hex[:16]}",
                    customer_id=customer_id,
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    resource_name=resource_name,
                    contains_phi=contains_phi,
                    source_connection_id=source_connection_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    timestamp=datetime.now(timezone.utc),
                )
                session.add(row)
        except Exception as e:
            logger.warning(
                "phi_access_log.write_failed",
                customer_id=customer_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                error=str(e),
                note=(
                    "PHI audit log write failed but the operation it "
                    "audits will continue. Audit gap may be visible "
                    "to compliance reviewers."
                ),
            )

    # =================================================================
    # push_review_queue — 5 methods
    # =================================================================

    async def create_push_review_item(
        self,
        customer_id: str,
        *,
        key: str,
        value: str,
        confidence: float,
        source: str = "llm_observation",
        source_query_id: Optional[str] = None,
    ) -> PushReviewItem:
        """Queue a medium-confidence LLM observation for human review.
        Created when v3_signal_handler sees a crystal_push_store call
        with confidence in the medium band (default 0.5..0.9)."""
        import uuid
        item_id = f"prv_{uuid.uuid4().hex[:16]}"
        now = datetime.now(timezone.utc)

        async with self.session() as session:  # type: ignore[attr-defined]
            row = PushReviewQueueRow(
                id=item_id,
                customer_id=customer_id,
                key=key,
                value=value,
                confidence=confidence,
                source=source,
                status="pending",
                source_query_id=source_query_id,
                created_at=now,
            )
            session.add(row)

        return PushReviewItem(
            id=item_id,
            customer_id=customer_id,
            key=key,
            value=value,
            confidence=confidence,
            source=source,  # type: ignore[arg-type]
            status="pending",
            source_query_id=source_query_id,
            created_at=now,
        )

    async def get_push_review_item(
        self, item_id: str, customer_id: str
    ) -> Optional[PushReviewItem]:
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(PushReviewQueueRow, item_id)
            if row is None or row.customer_id != customer_id:
                return None
            return _push_review_from_row(row)

    async def list_push_review_items(
        self,
        customer_id: str,
        *,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[PushReviewItem]:
        """Paginated review queue for the inspector. Ordered by
        created_at DESC so newest pending items appear first."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(PushReviewQueueRow)
                .where(PushReviewQueueRow.customer_id == customer_id)
                .order_by(PushReviewQueueRow.created_at.desc())
                .limit(limit)
            )
            if status is not None:
                stmt = stmt.where(PushReviewQueueRow.status == status)
            result = await session.execute(stmt)
            return [
                _push_review_from_row(r)
                for r in result.scalars().all()
            ]

    async def mark_push_review_approved(
        self, item_id: str, *, crystal_id: str, reviewed_at: datetime
    ) -> None:
        """Operator approved this push. crystal_id is the crystal the
        approval write landed in; caller is responsible for performing
        the actual add_pair_for_customer call before calling this."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(PushReviewQueueRow, item_id)
            if row is not None:
                row.status = "approved"
                row.crystal_id = crystal_id
                row.reviewed_at = reviewed_at

    async def mark_push_review_rejected(
        self, item_id: str, *, reviewed_at: datetime
    ) -> None:
        """Operator rejected this push. Item stays in the queue with
        status='rejected' for audit; not deleted."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(PushReviewQueueRow, item_id)
            if row is not None:
                row.status = "rejected"
                row.reviewed_at = reviewed_at

    # =================================================================
    # knowledge_gaps — 5 methods (4 from audit + cross-tenant lister)
    # =================================================================

    async def create_knowledge_gap(
        self,
        customer_id: str,
        *,
        domain: Optional[str],
        subject: Optional[str],
        missing: str,
        priority: str = "medium",
        source: str = "llm_observation",
        full_key: Optional[str] = None,
        triggering_query: Optional[str] = None,
        disposition: Optional[str] = None,
    ) -> KnowledgeGap:
        """Record a knowledge gap. The cognition worker / inspector
        can fill it later. S3 (2026-07-08): full_key carries the complete
        sparse key when the gap is anchored to one; triggering_query the
        demand that missed."""
        import uuid
        gap_id = f"gap_{uuid.uuid4().hex[:16]}"
        now = datetime.now(timezone.utc)

        async with self.session() as session:  # type: ignore[attr-defined]
            row = KnowledgeGapRow(
                id=gap_id,
                customer_id=customer_id,
                domain=domain,
                subject=subject,
                missing=missing,
                full_key=full_key,
                triggering_query=triggering_query,
                disposition=disposition,
                priority=priority,
                status="open",
                source=source,
                created_at=now,
            )
            session.add(row)

        return KnowledgeGap(
            id=gap_id,
            customer_id=customer_id,
            domain=domain,
            subject=subject,
            missing=missing,
            full_key=full_key,
            triggering_query=triggering_query,
            disposition=disposition,  # type: ignore[arg-type]
            priority=priority,  # type: ignore[arg-type]
            status="open",
            source=source,  # type: ignore[arg-type]
            created_at=now,
        )

    async def list_knowledge_gaps(
        self,
        customer_id: str,
        *,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[KnowledgeGap]:
        """Plain paginated list. Use list_knowledge_gaps_with_filled_content
        when you need the filling-crystal content too."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(KnowledgeGapRow)
                .where(KnowledgeGapRow.customer_id == customer_id)
                .order_by(KnowledgeGapRow.created_at.desc())
                .limit(limit)
            )
            if status is not None:
                stmt = stmt.where(KnowledgeGapRow.status == status)
            result = await session.execute(stmt)
            return [
                _knowledge_gap_from_row(r)
                for r in result.scalars().all()
            ]

    async def list_open_knowledge_gaps_cross_tenant(
        self, *, limit: int = 50
    ) -> list[KnowledgeGap]:
        """Cross-tenant: all open knowledge gaps across all
        customers, ordered by created_at ASC (oldest first).

        Added in Phase 6.5 P3.5 / CU-11 to support the cognition
        worker's `_fill_open_gaps` background sweep. v1's worker
        did this with raw SQL across customers; this method
        encapsulates the cross-tenant query inside the store
        boundary, preserving the convention that no SQL lives
        outside metadata_store.py or this mixin.

        The caller (cognition worker) is expected to iterate the
        returned gaps and scope subsequent operations to each
        gap's customer_id. Tenancy enforcement happens at the
        worker boundary, not here."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(KnowledgeGapRow)
                .where(KnowledgeGapRow.status == "open")
                .order_by(KnowledgeGapRow.created_at.asc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return [
                _knowledge_gap_from_row(r)
                for r in result.scalars().all()
            ]

    async def list_knowledge_gaps_with_filled_content(
        self, customer_id: str, *, limit: int = 50
    ) -> list[tuple[KnowledgeGap, Optional[str]]]:
        """Admin inspector variant: for each gap with
        filled_by_crystal_id set, also fetch a snippet of the
        filling crystal's first Fact.claim_text (truncated to 500
        chars). Returns [(gap, snippet_or_None), ...].

        AN-6: this encapsulates the 1+N FactRow lookup that v1
        does inline in admin.py. The N+1 read pattern survives —
        it's just hidden behind the store boundary. Acceptable for
        limit=50; not designed for scale."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(KnowledgeGapRow)
                .where(KnowledgeGapRow.customer_id == customer_id)
                .order_by(KnowledgeGapRow.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            gaps = list(result.scalars().all())

            enriched: list[tuple[KnowledgeGap, Optional[str]]] = []
            for gap_row in gaps:
                snippet: Optional[str] = None
                if gap_row.filled_by_crystal_id is not None:
                    fact_stmt = (
                        select(FactRow)
                        .where(
                            FactRow.crystal_id ==
                            gap_row.filled_by_crystal_id
                        )
                        .order_by(FactRow.created_at.asc())
                        .limit(1)
                    )
                    fact_result = await session.execute(fact_stmt)
                    first_fact = fact_result.scalar_one_or_none()
                    if first_fact is not None:
                        text = first_fact.claim_text or ""
                        snippet = text[:500]

                enriched.append(
                    (_knowledge_gap_from_row(gap_row), snippet)
                )

            return enriched

    async def update_knowledge_gap_disposition(
        self, gap_id: str, disposition: str
    ) -> None:
        """S10 (2026-07-08): verdict writeback. When research concludes
        the bank can't answer (needs_capability), the gap's disposition
        flips to needs_document — which durably parks it from the fill
        sweep, moves it to Your Tasks (S5), and hides the Research
        button. The verdict becomes state instead of a log line."""
        async with self.session() as session:
            row = await session.get(KnowledgeGapRow, gap_id)
            if row is not None:
                row.disposition = disposition

    async def mark_knowledge_gap_filled(
        self,
        gap_id: str,
        *,
        filled_by_crystal_id: str,
        resolved_at: datetime,
    ) -> None:
        """Mark a gap resolved by a specific crystal. Used by the
        cognition fill_gap worker (Phase 7+) and by manual operator
        resolution via the inspector."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(KnowledgeGapRow, gap_id)
            if row is not None:
                row.status = "filled"
                row.filled_by_crystal_id = filled_by_crystal_id
                row.resolved_at = resolved_at

    # =================================================================
    # cognition_tasks — 6 methods
    # =================================================================

    async def create_cognition_task(
        self,
        customer_id: str,
        *,
        task_type: str,
        payload: Optional[dict[str, Any]],
        priority: str = "background",
        source_query_id: Optional[str] = None,
    ) -> CognitionTask:
        """Enqueue a cognition workflow. The cognition worker picks
        up pending tasks; the agent's synchronous cognition_run tool
        will use priority='urgent' (per AGENT_ARCHITECTURE.md)."""
        import uuid
        task_id = f"cog_{uuid.uuid4().hex[:16]}"
        now = datetime.now(timezone.utc)

        async with self.session() as session:  # type: ignore[attr-defined]
            row = CognitionTaskRow(
                id=task_id,
                customer_id=customer_id,
                task_type=task_type,
                payload=payload,
                priority=priority,
                status="pending",
                source_query_id=source_query_id,
                created_at=now,
            )
            session.add(row)

        return CognitionTask(
            id=task_id,
            customer_id=customer_id,
            task_type=task_type,  # type: ignore[arg-type]
            payload=payload,
            priority=priority,  # type: ignore[arg-type]
            status="pending",
            source_query_id=source_query_id,
            created_at=now,
        )

    async def get_cognition_task(
        self, task_id: str
    ) -> Optional[CognitionTask]:
        """No customer_id check — the cognition worker is
        cross-tenant. Tenancy enforcement happens at the worker
        boundary by reading the row's customer_id and scoping
        subsequent operations to it."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(CognitionTaskRow, task_id)
            return _cognition_task_from_row(row) if row else None

    async def list_cognition_tasks(
        self,
        customer_id: str,
        *,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[CognitionTask]:
        """Paginated task list per customer, for the inspector."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(CognitionTaskRow)
                .where(CognitionTaskRow.customer_id == customer_id)
                .order_by(CognitionTaskRow.created_at.desc())
                .limit(limit)
            )
            if status is not None:
                stmt = stmt.where(CognitionTaskRow.status == status)
            result = await session.execute(stmt)
            return [
                _cognition_task_from_row(r)
                for r in result.scalars().all()
            ]

    async def list_open_research_topics(
        self, customer_id: str, *, limit: int = 50
    ) -> list[str]:
        """Return the topics of OPEN (pending|running) research tasks.

        Used by the signal handler to dedup near-identical concurrent
        research before enqueueing a new cognition task (C5). 'Open'
        deliberately excludes completed/failed tasks, so a topic that
        finished long ago can be researched again — only genuinely
        concurrent duplicates are suppressed. The topic lives in the
        task payload under the 'topic' key (written by the signal
        handler's _handle_research).
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(CognitionTaskRow)
                .where(CognitionTaskRow.customer_id == customer_id)
                .where(CognitionTaskRow.task_type == "research")
                .where(CognitionTaskRow.status.in_(["pending", "running"]))
                .order_by(CognitionTaskRow.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            topics: list[str] = []
            for row in result.scalars().all():
                payload = row.payload if isinstance(row.payload, dict) else None
                topic = payload.get("topic") if payload else None
                if topic:
                    topics.append(topic)
            return topics

    async def claim_pending_cognition_task(
        self,
    ) -> Optional[CognitionTask]:
        """Atomic-on-SQLite: pick the oldest pending task and mark
        it 'running' in one transaction. Cross-tenant.

        Same scope as claim_pending_documents_batch: the SELECT takes
        `FOR UPDATE SKIP LOCKED` so Postgres-with-multiple-workers
        claims are disjoint; on SQLite the hint is a no-op and the
        SERIALIZABLE single-writer transaction already makes it atomic.
        CU-10 closed.

        Returns None if no pending tasks. Caller scopes subsequent
        operations to the returned task's customer_id."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(CognitionTaskRow)
                .where(CognitionTaskRow.status == "pending")
                # Urgent first (agent-enqueued cognition_run tasks; a
                # person is waiting on the other end), then FIFO.
                # boolean asc sorts False<True on both PG and SQLite.
                .order_by(
                    (CognitionTaskRow.priority != "urgent").asc(),
                    CognitionTaskRow.created_at.asc(),
                )
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return None

            row.status = "running"
            row.started_at = datetime.now(timezone.utc)
            return _cognition_task_from_row(row)

    async def mark_cognition_task_complete(
        self,
        task_id: str,
        *,
        result: dict[str, Any],
        result_crystal_id: Optional[str] = None,
        completed_at: datetime,
    ) -> None:
        """Cognition workflow finished and validator approved.
        result_crystal_id is set if the deliverable was written as a
        new crystal (e.g., research result crystallized)."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(CognitionTaskRow, task_id)
            if row is not None:
                row.status = "complete"
                row.result = result
                row.result_crystal_id = result_crystal_id
                row.completed_at = completed_at

    async def requeue_cognition_task(self, task_id: str) -> bool:
        """Cognition cycles (2026-07-16): flip a terminal task back to
        pending — the SAME row, so trigger identity (task.id) is
        preserved and prior verdicts + open critiques flow into the
        next run. Used by the worker's auto-recycle and by the manual
        Re-run endpoint. Returns False if the task doesn't exist or is
        already pending (a no-op). A RUNNING task may flip — that is
        the worker's own recycle path (the task it is processing is
        legitimately running); the manual endpoint separately 409s on
        pending/running before calling this."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(CognitionTaskRow, task_id)
            if row is None or row.status == "pending":
                return False
            row.status = "pending"
            row.completed_at = None
            row.error_message = None
            return True

    async def mark_cognition_task_failed(
        self,
        task_id: str,
        *,
        error_message: str,
        completed_at: datetime,
    ) -> None:
        """Cognition workflow failed or validator rejected after
        max retries."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(CognitionTaskRow, task_id)
            if row is not None:
                row.status = "failed"
                row.error_message = error_message
                row.completed_at = completed_at


# ---------------------------------------------------------------------------
# Row → Pydantic converters
# ---------------------------------------------------------------------------
#
# Each converter mirrors the pattern in metadata_store.py's bottom
# section: take a row, return a Pydantic model. The `# type: ignore`
# annotations on Literal fields match the pattern in the existing
# converters there (e.g. _customer_from_row, _crystal_type_from_row).

def _source_watch_from_row(row) -> "SourceWatch":
    from ..models.source_watch import SourceWatch
    return SourceWatch(
        id=row.id, customer_id=row.customer_id, scheme=row.scheme,
        source_name=row.source_name, config=row.config or {},
        cadence_minutes=row.cadence_minutes,
        last_state=row.last_state, review_mode=row.review_mode,
        encrypted_token=row.encrypted_token, status=row.status,
        last_checked_at=row.last_checked_at, last_error=row.last_error,
        created_at=row.created_at,
    )


def _document_upload_from_row(row: DocumentUploadRow) -> DocumentUpload:
    return DocumentUpload(
        id=row.id,
        customer_id=row.customer_id,
        label=row.label or "",
        text=row.text,
        status=row.status,  # type: ignore[arg-type]
        crystal_type=row.crystal_type or "customer:legacy",
        char_count=row.char_count or 0,
        crystals_written=row.crystals_written or 0,
        items_extracted=row.items_extracted or 0,
        error_message=row.error_message,
        source_file_id=row.source_file_id,
        source_modified_at=row.source_modified_at,
        source_connection_id=row.source_connection_id,
        source_uri=getattr(row, "source_uri", None),
        content_hash=getattr(row, "content_hash", None),
        scope=row.scope,
        owner_operator_id=row.owner_operator_id,
        extracted_items=row.extracted_items,
        detected_type=row.detected_type,
        confirmed_type=row.confirmed_type,
        content_chunks=row.content_chunks,
        created_at=row.created_at,
        crystallized_at=row.crystallized_at,
    )


def _drive_connection_from_row(row: DriveConnectionRow) -> DriveConnection:
    return DriveConnection(
        id=row.id,
        customer_id=row.customer_id,
        provider=row.provider or "google_drive",
        email=row.email,
        encrypted_refresh_token=row.encrypted_refresh_token,
        token_nonce=row.token_nonce,
        scopes=row.scopes,
        status=row.status,  # type: ignore[arg-type]
        last_synced_at=row.last_synced_at,
        error_message=row.error_message,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _watched_folder_from_row(row: WatchedFolderRow) -> WatchedFolder:
    return WatchedFolder(
        id=row.id,
        connection_id=row.connection_id,
        customer_id=row.customer_id,
        folder_id=row.folder_id,
        folder_name=row.folder_name,
        folder_path=row.folder_path,
        contains_phi=row.contains_phi,
        sync_interval_minutes=row.sync_interval_minutes or 60,
        last_checked_at=row.last_checked_at,
        last_file_count=row.last_file_count,
        status=row.status,  # type: ignore[arg-type]
        created_at=row.created_at,
    )


def _watched_file_from_row(row: WatchedFileRow) -> WatchedFile:
    return WatchedFile(
        id=row.id,
        connection_id=row.connection_id,
        customer_id=row.customer_id,
        file_id=row.file_id,
        file_name=row.file_name,
        mime_type=row.mime_type,
        contains_phi=row.contains_phi,
        sync_interval_minutes=row.sync_interval_minutes or 60,
        last_checked_at=row.last_checked_at,
        last_modified_at=row.last_modified_at,
        status=row.status,  # type: ignore[arg-type]
        created_at=row.created_at,
    )


def _baa_tracking_from_row(row: BaaTrackingRow) -> BaaTracking:
    return BaaTracking(
        id=row.id,
        customer_id=row.customer_id,
        baa_signed=row.baa_signed,
        baa_signed_date=row.baa_signed_date,
        baa_document_ref=row.baa_document_ref,
        phi_data_sources=row.phi_data_sources,
        hipaa_contact_email=row.hipaa_contact_email,
        notes=row.notes,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _push_review_from_row(row: PushReviewQueueRow) -> PushReviewItem:
    return PushReviewItem(
        id=row.id,
        customer_id=row.customer_id,
        key=row.key,
        value=row.value,
        confidence=row.confidence,
        source=row.source,  # type: ignore[arg-type]
        status=row.status,  # type: ignore[arg-type]
        crystal_id=row.crystal_id,
        source_query_id=row.source_query_id,
        reviewed_at=row.reviewed_at,
        created_at=row.created_at,
    )


def _knowledge_gap_from_row(row: KnowledgeGapRow) -> KnowledgeGap:
    return KnowledgeGap(
        id=row.id,
        customer_id=row.customer_id,
        domain=row.domain,
        subject=row.subject,
        missing=row.missing,
        full_key=getattr(row, "full_key", None),
        triggering_query=getattr(row, "triggering_query", None),
        disposition=getattr(row, "disposition", None),
        priority=row.priority,  # type: ignore[arg-type]
        status=row.status,  # type: ignore[arg-type]
        source=row.source,  # type: ignore[arg-type]
        filled_by_crystal_id=row.filled_by_crystal_id,
        created_at=row.created_at,
        resolved_at=row.resolved_at,
    )


def _cognition_task_from_row(row: CognitionTaskRow) -> CognitionTask:
    return CognitionTask(
        id=row.id,
        customer_id=row.customer_id,
        task_type=row.task_type,  # type: ignore[arg-type]
        payload=row.payload,
        priority=row.priority,  # type: ignore[arg-type]
        status=row.status,  # type: ignore[arg-type]
        result=row.result,
        result_crystal_id=row.result_crystal_id,
        source_query_id=row.source_query_id,
        error_message=row.error_message,
        created_at=row.created_at,
        started_at=row.started_at,
        completed_at=row.completed_at,
    )
