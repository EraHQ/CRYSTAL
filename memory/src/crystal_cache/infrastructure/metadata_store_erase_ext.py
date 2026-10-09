"""Tenant erasure and account deletion (2026-10-03; Q36=A, Q37=B).

Two scopes, both schema-driven so a table added later cannot be missed:

- erase_tenant_bank(cid): "Erase all memories". Deletes every row the
  tenant's bank is made of, in dependency order, and leaves the ACCOUNT
  intact: the customer row, users, Key A, the tenant encryption key,
  connections and grants, spend budgets, the ledger, config.
- purge_tenant(cid): the end of "Delete my account". Erases the bank,
  then removes the account itself: users, operators (Key A hashes),
  tenant key (crypto-shred, immediate), connections, OAuth grants and
  states, groups, budgets, and the customer row. The LLM ledger is the
  one thing that survives, anonymized to a tombstone id, because it is
  an accounting record.

Between asking and purging (seven days, Q37=B) the account is LOCKED:
schedule_account_deletion stamps deletion_scheduled_at/purge_after; the
auth layer refuses every route but /v1/me and /v1/me/restore; the purge
worker runs purge_tenant once purge_after has passed.

How the walk works: every table in Base.metadata is visited in
reverse dependency order (children before parents). A table with a
customer_id column loses the tenant's rows; a table with a crystal_id
column (and no customer_id) loses rows whose crystal belongs to the
tenant; a table with neither is untouched. Keep-lists name the tables a
scope must NOT touch. Nothing is hand-ordered.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog
from sqlalchemy import delete, select, update

from .schema import Base, CrystalRow, CustomerRow, LlmCallRow, OAuthRecordRow, OperatorRow

logger = structlog.get_logger(__name__)

PURGE_GRACE_DAYS = 7

# "Erase all memories" keeps the account. Everything on this list
# survives an erase; everything else keyed by the tenant goes.
ERASE_KEEPS: frozenset[str] = frozenset({
    "customers", "users", "operators", "tenant_keys", "drive_connections",
    "source_watches", "source_schemas", "oauth_records", "oauth_states",
    "task_keys", "groups", "group_members", "spend_budgets", "llm_calls",
    "dsl_configs", "baa_tracking", "phi_access_log", "expert_authorizations",
    # Config the re-import depends on, and user-authored rules.
    "crystal_types", "system_rules", "mandatory_rules",
})

# Purge keeps only the ledger (anonymized) and the customer row until the
# very end, where it is deleted last.
PURGE_KEEPS: frozenset[str] = frozenset({"llm_calls", "customers"})


def _tombstone(customer_id: str) -> str:
    return "deleted:" + hashlib.sha256(customer_id.encode()).hexdigest()[:16]


def _aware(dt: Optional[datetime]) -> Optional[datetime]:
    """SQLite hands back naive datetimes for timezone=True columns;
    Postgres hands back aware ones. Normalize to UTC-aware."""
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _crystal_ref_columns(table):
    """Columns that point at a crystal: crystal_id, source_crystal_id,
    target_crystal_id, parent_crystal_id, result_crystal_id, ...,
    crystal_a_id, crystal_b_id."""
    return [
        c for c in table.c
        if c.name.endswith("crystal_id") or c.name in ("crystal_a_id", "crystal_b_id")
    ]


class ErasureExtensionsMixin:
    """Bound onto MetadataStore by _bind_mixin_methods (see
    infrastructure/__init__.py)."""

    async def scrub_upload_if_orphaned(
        self, customer_id: str, *,
        document_id: Optional[str] = None, source_uri: Optional[str] = None,
    ) -> int:
        """Lockdown PR-4 (Q54=A, 2026-10-09; supersedes the RC-05 scrub):
        forgetting the last crystal born from an upload blanks that
        upload's `text`, `content_chunks` and `extracted_items` — the
        row stays as the record that something was uploaded (status
        'forgotten'), the fact ledger keeps the before-text for audit.

        Called from the two delete primitives (`delete_crystal`, and
        `delete_fact` when the last fact takes the crystal), so every
        delete path — MCP forget tools, HTTP DELETE, the console deletes,
        both wipes, the pipeline's own replace — scrubs by construction.

        Matching: by `document_id` (crystals.source_document_id, the
        upload the crystal was born from) when the crystal carried one;
        else by the legacy `source_uri` match for pre-column rows, with
        the fragment (#sheet=, #msg-window=) stripped so a carved source
        still finds its row. Only rows that finished ('crystallized')
        are scrubbed: a row still being written (the pipeline deletes an
        empty file crystal of the document it is writing) is left alone.
        Returns how many uploads were scrubbed."""
        from .schema import DocumentUploadRow

        async with self.session() as session:  # type: ignore[attr-defined]
            if document_id:
                still = (await session.execute(
                    select(CrystalRow.id)
                    .where(CrystalRow.customer_id == customer_id)
                    .where(CrystalRow.source_document_id == document_id)
                    .limit(1)
                )).first()
                if still is not None:
                    return 0
                where = (DocumentUploadRow.id == document_id)
            elif source_uri:
                base_uri = source_uri.split("#", 1)[0]
                still = (await session.execute(
                    select(CrystalRow.id)
                    .where(CrystalRow.customer_id == customer_id)
                    .where(
                        (CrystalRow.source_uri == base_uri)
                        | (CrystalRow.source_uri.like(base_uri + "#%"))
                    )
                    .limit(1)
                )).first()
                if still is not None:
                    return 0
                where = (DocumentUploadRow.source_uri == base_uri)
            else:
                return 0
            res = await session.execute(
                update(DocumentUploadRow)
                .where(DocumentUploadRow.customer_id == customer_id)
                .where(where)
                .where(DocumentUploadRow.status == "crystallized")
                .values(text="", content_chunks=None, extracted_items=None,
                        status="forgotten")
            )
            await session.commit()
            return int(res.rowcount or 0)

    async def scrub_upload_text_if_orphaned(self, customer_id: str, source_uri: Optional[str]) -> int:
        """RC-05 name kept for callers; the Q54 scrub does the work."""
        return await self.scrub_upload_if_orphaned(customer_id, source_uri=source_uri)

    async def erase_tenant_bank(self, customer_id: str) -> dict[str, int]:
        """Q36=A, immediate: every bank row the tenant owns, account kept.
        Returns rows deleted per table. Callers invalidate the vector
        caches afterwards (they are caches over these rows)."""
        return await self.erase_rows(customer_id, keeps=ERASE_KEEPS)

    async def schedule_account_deletion(
        self, customer_id: str, *, now: Optional[datetime] = None,
    ) -> Optional[datetime]:
        """Q37=B: lock the account and set the purge date. Idempotent.
        Returns purge_after, or None if the customer does not exist."""
        now = now or datetime.now(timezone.utc)
        purge_after = now + timedelta(days=PURGE_GRACE_DAYS)
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(CustomerRow, customer_id)
            if row is None:
                return None
            if row.deletion_scheduled_at is None:
                row.deletion_scheduled_at = now
                row.purge_after = purge_after
            else:
                purge_after = _aware(row.purge_after)
            await session.commit()
        return purge_after

    async def cancel_account_deletion(self, customer_id: str) -> bool:
        """Restore within the grace period. True when a lock was cleared."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(CustomerRow, customer_id)
            if row is None or row.deletion_scheduled_at is None:
                return False
            row.deletion_scheduled_at = None
            row.purge_after = None
            await session.commit()
            return True

    async def list_due_purges(self, *, now: Optional[datetime] = None) -> list[str]:
        """Customers whose purge_after has passed. Compared in Python after
        normalizing, because SQLite stores naive timestamps and a mixed
        comparison in SQL is a string comparison."""
        now = _aware(now or datetime.now(timezone.utc))
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(
                select(CustomerRow.id, CustomerRow.purge_after)
                .where(CustomerRow.purge_after.is_not(None))
            )).all()
            return [r.id for r in rows if _aware(r.purge_after) <= now]

    async def revoke_tenant_access(self, customer_id: str) -> dict[str, int]:
        """At lock time: Key A rotated to a value nobody is shown, and every
        OAuth grant held by one of the tenant's operators (MCP
        connections) revoked. The console session (Firebase) stays, so
        the owner can restore."""
        counts = {"api_key_rotated": 0, "oauth_revoked": 0}
        try:
            if await self.rotate_customer_api_key(customer_id) is not None:  # type: ignore[attr-defined]
                counts["api_key_rotated"] = 1
        except Exception:  # noqa: BLE001
            logger.warning("erase.key_rotate_failed", customer_id=customer_id)
        async with self.session() as session:  # type: ignore[attr-defined]
            subjects = [
                r.id for r in (await session.execute(
                    select(OperatorRow.id).where(OperatorRow.team_id == customer_id)
                )).all()
            ]
            if subjects:
                res = await session.execute(
                    update(OAuthRecordRow)
                    .where(OAuthRecordRow.subject.in_(subjects))
                    .where(OAuthRecordRow.revoked_at_utc.is_(None))
                    .values(revoked_at_utc=datetime.now(timezone.utc))
                )
                counts["oauth_revoked"] = int(res.rowcount or 0)
            await session.commit()
        return counts

    async def purge_tenant(self, customer_id: str) -> dict[str, int]:
        """The end of account deletion. Bank, then account, then the
        customer row; the ledger survives under a tombstone id."""
        counts = await self.erase_rows(customer_id, keeps=PURGE_KEEPS)
        async with self.session() as session:  # type: ignore[attr-defined]
            # OAuth records are keyed by operator id, not tenant: drop the
            # ones the walk cannot see.
            subjects = [
                r.id for r in (await session.execute(
                    select(OperatorRow.id).where(OperatorRow.team_id == customer_id)
                )).all()
            ]
            if subjects:
                res = await session.execute(
                    delete(OAuthRecordRow).where(OAuthRecordRow.subject.in_(subjects))
                )
                counts["oauth_records"] = int(res.rowcount or 0)
            res = await session.execute(
                delete(OperatorRow).where(OperatorRow.team_id == customer_id)
            )
            if res.rowcount:
                counts["operators"] = int(res.rowcount)
            await session.commit()
        # Crypto-shred the tenant encryption key (immediate: the owner
        # asked, the grace period has passed).
        try:
            if await self.destroy_tenant_dek(customer_id, immediate=True):  # type: ignore[attr-defined]
                counts["tenant_keys"] = counts.get("tenant_keys", 0) + 1
        except Exception:  # noqa: BLE001
            logger.warning("erase.dek_destroy_failed", customer_id=customer_id)
        async with self.session() as session:  # type: ignore[attr-defined]
            # Ledger rows become accounting records with no tenant identity.
            res = await session.execute(
                update(LlmCallRow)
                .where(LlmCallRow.customer_id == customer_id)
                .values(customer_id=_tombstone(customer_id))
            )
            counts["llm_calls_anonymized"] = int(res.rowcount or 0)
            row = await session.get(CustomerRow, customer_id)
            if row is not None:
                await session.delete(row)
                counts["customers"] = 1
            await session.commit()
        logger.info("erase.account_purged", customer_id=customer_id, counts=counts)
        return counts

    async def erase_rows(self, customer_id: str, *, keeps: frozenset[str]) -> dict[str, int]:
        """The schema-driven walk both scopes share. Public because the
        mixin binder skips underscore names."""
        counts: dict[str, int] = {}
        async with self.session() as session:  # type: ignore[attr-defined]
            # Materialize the tenant's crystal ids ONCE: a table that
            # references crystals without a foreign key can sort after the
            # crystals table, where a live subquery would already be empty.
            crystal_ids = [
                r.id for r in (await session.execute(
                    select(CrystalRow.id).where(CrystalRow.customer_id == customer_id)
                )).all()
            ]
            for table in reversed(Base.metadata.sorted_tables):
                if table.name in keeps:
                    continue
                if "customer_id" in table.c:
                    stmt = delete(table).where(table.c.customer_id == customer_id)
                elif table.name == "operators":
                    continue  # team_id-keyed; purge_tenant handles it explicitly
                else:
                    refs = _crystal_ref_columns(table)
                    if not refs or not crystal_ids:
                        continue
                    cond = refs[0].in_(crystal_ids)
                    for c in refs[1:]:
                        cond = cond | c.in_(crystal_ids)
                    stmt = delete(table).where(cond)
                res = await session.execute(stmt)
                n = int(res.rowcount or 0)
                if n:
                    counts[table.name] = n
            await session.commit()
        logger.info("erase.rows_deleted", customer_id=customer_id, counts=counts)
        return counts
