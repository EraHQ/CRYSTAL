"""Connectors and the Beta list (2026-10-24): the SQL for beta_features,
beta_invites, app_connections and watched_items.

Anthony: "Make a beta list for private invites to features like this.
Everything but keep Gmail private." A person's apps are reached through
Nango (the tokens live there, never here); a connection row records what
Crystal may do with each app, and watched_items keeps what the apps
showed for 7 days (Q108 A). Same binding pattern as the other mixins
(infrastructure/__init__ setattrs the public methods onto MetadataStore).
R9: these tables' SQL lives here and nowhere else.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

import structlog
from sqlalchemy import delete, func, or_, select

from .schema import AppConnectionRow, BetaFeatureRow, BetaInviteRow, UserRow, WatchedItemRow

logger = structlog.get_logger(__name__)

DEFAULT_USES = {"watch": True, "remember": False, "act": False}
_CONNECTION_FIELDS = {
    "uses", "remember_rules", "watch_cadence_minutes", "status", "account_label",
    "sync_due", "cursors", "last_synced_at", "last_error",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _aware(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _feature_dict(r: BetaFeatureRow) -> dict[str, Any]:
    return {"key": r.key, "title": r.title, "description": r.description or "", "status": r.status}


def _invite_dict(r: BetaInviteRow) -> dict[str, Any]:
    return {
        "id": r.id, "feature_key": r.feature_key, "customer_id": r.customer_id, "email": r.email,
        "invited_by": r.invited_by or "", "note": r.note or "",
        "created_at": _iso(r.created_at), "accepted_at": _iso(r.accepted_at),
    }


def connection_dict(r: AppConnectionRow) -> dict[str, Any]:
    uses = {**DEFAULT_USES, **dict(r.uses or {})}
    return {
        "id": r.id, "customer_id": r.customer_id, "integration": r.integration,
        "nango_connection_id": r.nango_connection_id, "account_label": r.account_label or "",
        "status": r.status, "uses": {k: bool(uses.get(k)) for k in DEFAULT_USES},
        "remember_rules": list(r.remember_rules or []), "watch_cadence_minutes": int(r.watch_cadence_minutes or 15),
        "cursors": dict(r.cursors or {}), "sync_due": bool(r.sync_due),
        "last_synced_at": _iso(r.last_synced_at), "last_error": r.last_error,
        "created_at": _iso(r.created_at), "updated_at": _iso(r.updated_at),
    }


def watched_item_dict(r: WatchedItemRow) -> dict[str, Any]:
    return {
        "id": r.id, "customer_id": r.customer_id, "connection_id": r.connection_id, "kind": r.kind,
        "external_id": r.external_id, "title": r.title or "", "author": r.author or "",
        "occurred_at": _iso(r.occurred_at), "url": r.url, "snippet": r.snippet or "", "body": r.body or "",
        "meta": dict(r.meta or {}), "remembered_document_id": r.remembered_document_id,
        "deleted_at": _iso(r.deleted_at), "expires_at": _iso(r.expires_at),
        "created_at": _iso(r.created_at), "updated_at": _iso(r.updated_at),
    }


class ConnectorExtensionsMixin:
    """beta_* / app_connections / watched_items, bound onto MetadataStore."""

    # ----------------------------------------------------------- beta list

    async def ensure_beta_features(self, defaults: Iterable[dict[str, str]]) -> None:
        """Insert any default feature that is missing; never changes an
        existing row (a platform admin's status choice stands)."""
        async with self.session() as session:  # type: ignore[attr-defined]
            have = set((await session.execute(select(BetaFeatureRow.key))).scalars().all())
            for d in defaults:
                if d["key"] not in have:
                    session.add(BetaFeatureRow(key=d["key"], title=d["title"],
                                               description=d.get("description", ""),
                                               status=d.get("status", "invite"), created_at=_now()))
            await session.commit()

    async def list_beta_features(self) -> list[dict[str, Any]]:
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(select(BetaFeatureRow).order_by(BetaFeatureRow.key))).scalars().all()
            return [_feature_dict(r) for r in rows]

    async def set_beta_feature_status(self, key: str, status: str) -> Optional[dict[str, Any]]:
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(BetaFeatureRow, key)
            if row is None:
                return None
            row.status = status
            await session.commit()
            return _feature_dict(row)

    async def create_beta_invite(self, feature_key: str, *, customer_id: Optional[str] = None,
                                 email: Optional[str] = None, invited_by: str = "",
                                 note: str = "") -> dict[str, Any]:
        email = (email or "").strip().lower() or None
        if not customer_id and not email:
            raise ValueError("an invite needs a customer or an email")
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = select(BetaInviteRow).where(BetaInviteRow.feature_key == feature_key)
            if customer_id:
                stmt = stmt.where(BetaInviteRow.customer_id == customer_id)
            else:
                stmt = stmt.where(BetaInviteRow.email == email)
            existing = (await session.execute(stmt)).scalars().first()
            if existing is not None:
                return _invite_dict(existing)
            row = BetaInviteRow(id=f"inv_{uuid.uuid4().hex[:16]}", feature_key=feature_key,
                                customer_id=customer_id, email=email, invited_by=invited_by[:320],
                                note=note[:256], created_at=_now(),
                                accepted_at=_now() if customer_id else None)
            session.add(row)
            await session.commit()
            return _invite_dict(row)

    async def list_beta_invites(self, feature_key: Optional[str] = None) -> list[dict[str, Any]]:
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = select(BetaInviteRow).order_by(BetaInviteRow.created_at.desc())
            if feature_key:
                stmt = stmt.where(BetaInviteRow.feature_key == feature_key)
            return [_invite_dict(r) for r in (await session.execute(stmt)).scalars().all()]

    async def delete_beta_invite(self, invite_id: str) -> bool:
        async with self.session() as session:  # type: ignore[attr-defined]
            res = await session.execute(delete(BetaInviteRow).where(BetaInviteRow.id == invite_id))
            await session.commit()
            return bool(res.rowcount)

    async def beta_keys_for(self, customer_id: str, emails: Iterable[str] = ()) -> set[str]:
        """Every feature this tenant (or one of its people, by email) was
        invited to."""
        mails = [e.strip().lower() for e in emails if (e or "").strip()]
        async with self.session() as session:  # type: ignore[attr-defined]
            cond = BetaInviteRow.customer_id == customer_id
            if mails:
                cond = or_(cond, BetaInviteRow.email.in_(mails))
            rows = (await session.execute(select(BetaInviteRow.feature_key).where(cond))).scalars().all()
            return set(rows)

    async def list_user_emails_for_customer(self, customer_id: str) -> list[str]:
        """The emails of a tenant's people (an email invite reaches them)."""
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(
                select(UserRow.email).where(UserRow.customer_id == customer_id)
            )).scalars().all()
            return [e for e in rows if e]

    async def bind_beta_invites(self, customer_id: str, emails: Iterable[str]) -> int:
        """An invite made by email before the person had an account is
        bound to their tenant at sign-in. Returns how many were bound."""
        mails = [e.strip().lower() for e in emails if (e or "").strip()]
        if not mails:
            return 0
        n = 0
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(
                select(BetaInviteRow).where(BetaInviteRow.email.in_(mails)).where(BetaInviteRow.customer_id.is_(None))
            )).scalars().all()
            for r in rows:
                r.customer_id = customer_id
                r.accepted_at = _now()
                n += 1
            await session.commit()
        return n

    # --------------------------------------------------------- connections

    async def upsert_app_connection(self, customer_id: str, *, integration: str, nango_connection_id: str,
                                    account_label: str = "", created_by_operator_id: Optional[str] = None
                                    ) -> dict[str, Any]:
        """Bind a Nango connection to a tenant (first sign-in) or refresh
        it (a re-authorization): status back to active, due for a sync."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = (await session.execute(
                select(AppConnectionRow)
                .where(AppConnectionRow.customer_id == customer_id)
                .where(AppConnectionRow.nango_connection_id == nango_connection_id)
            )).scalars().first()
            now = _now()
            if row is None:
                row = AppConnectionRow(
                    id=f"conn_{uuid.uuid4().hex[:16]}", customer_id=customer_id, integration=integration,
                    nango_connection_id=nango_connection_id, account_label=account_label[:320],
                    status="active", uses=dict(DEFAULT_USES), remember_rules=[], watch_cadence_minutes=15,
                    cursors={}, sync_due=True, created_by_operator_id=created_by_operator_id,
                    created_at=now, updated_at=now,
                )
                session.add(row)
            else:
                row.status = "active"
                row.sync_due = True
                row.last_error = None
                if account_label:
                    row.account_label = account_label[:320]
                row.updated_at = now
            await session.commit()
            return connection_dict(row)

    async def list_app_connections(self, customer_id: str) -> list[dict[str, Any]]:
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(
                select(AppConnectionRow).where(AppConnectionRow.customer_id == customer_id)
                .order_by(AppConnectionRow.created_at)
            )).scalars().all()
            return [connection_dict(r) for r in rows]

    async def get_app_connection(self, customer_id: str, connection_id: str) -> Optional[dict[str, Any]]:
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(AppConnectionRow, connection_id)
            if row is None or row.customer_id != customer_id:
                return None
            return connection_dict(row)

    async def find_app_connection_by_nango(self, nango_connection_id: str) -> Optional[dict[str, Any]]:
        """The webhook's lookup (Nango names the connection, not the tenant)."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = (await session.execute(
                select(AppConnectionRow).where(AppConnectionRow.nango_connection_id == nango_connection_id)
            )).scalars().first()
            return connection_dict(row) if row is not None else None

    async def update_app_connection(self, customer_id: str, connection_id: str, **fields: Any
                                    ) -> Optional[dict[str, Any]]:
        unknown = set(fields) - _CONNECTION_FIELDS
        if unknown:
            raise ValueError(f"unknown connection fields {sorted(unknown)}")
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(AppConnectionRow, connection_id)
            if row is None or row.customer_id != customer_id:
                return None
            for k, v in fields.items():
                setattr(row, k, v)
            row.updated_at = _now()
            await session.commit()
            return connection_dict(row)

    async def delete_app_connection(self, customer_id: str, connection_id: str) -> bool:
        """The connection and everything watched through it. Memory made
        from it stays (it is the person's knowledge now)."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(AppConnectionRow, connection_id)
            if row is None or row.customer_id != customer_id:
                return False
            await session.execute(delete(WatchedItemRow).where(WatchedItemRow.connection_id == connection_id))
            await session.delete(row)
            await session.commit()
            return True

    async def list_due_app_connections(self, *, now: Optional[datetime] = None, limit: int = 50
                                       ) -> list[dict[str, Any]]:
        """Active connections Nango flagged (sync_due) or whose own cadence
        has passed, oldest first."""
        now = now or _now()
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(
                select(AppConnectionRow).where(AppConnectionRow.status == "active")
                .order_by(AppConnectionRow.last_synced_at.is_not(None), AppConnectionRow.last_synced_at)
            )).scalars().all()
        out = []
        for r in rows:
            last = _aware(r.last_synced_at)
            cadence = timedelta(minutes=max(5, int(r.watch_cadence_minutes or 15)))
            if r.sync_due or last is None or now - last >= cadence:
                out.append(connection_dict(r))
            if len(out) >= limit:
                break
        return out

    # ------------------------------------------------------- watched items

    async def upsert_watched_item(self, customer_id: str, connection_id: str, *, kind: str, external_id: str,
                                  title: str = "", author: str = "", occurred_at: Optional[datetime] = None,
                                  url: Optional[str] = None, snippet: str = "", body: str = "",
                                  meta: Optional[dict[str, Any]] = None, ttl_days: int = 7) -> dict[str, Any]:
        """Insert or refresh one item; a refresh does not extend its life
        past 7 days from when it was first seen."""
        now = _now()
        async with self.session() as session:  # type: ignore[attr-defined]
            row = (await session.execute(
                select(WatchedItemRow).where(WatchedItemRow.connection_id == connection_id)
                .where(WatchedItemRow.kind == kind).where(WatchedItemRow.external_id == external_id)
            )).scalars().first()
            if row is None:
                row = WatchedItemRow(
                    id=f"wi_{uuid.uuid4().hex[:16]}", customer_id=customer_id, connection_id=connection_id,
                    kind=kind, external_id=external_id[:256], expires_at=now + timedelta(days=ttl_days),
                    created_at=now, meta={},
                )
                session.add(row)
            row.title = (title or "")[:512]
            row.author = (author or "")[:320]
            row.occurred_at = occurred_at
            row.url = (url or None) and url[:1024]
            row.snippet = (snippet or "")[:2000]
            row.body = (body or "")[:200_000]
            row.meta = dict(meta or {})
            row.deleted_at = None
            row.updated_at = now
            await session.commit()
            return watched_item_dict(row)

    async def mark_watched_item_deleted(self, connection_id: str, kind: str, external_id: str
                                        ) -> Optional[dict[str, Any]]:
        async with self.session() as session:  # type: ignore[attr-defined]
            row = (await session.execute(
                select(WatchedItemRow).where(WatchedItemRow.connection_id == connection_id)
                .where(WatchedItemRow.kind == kind).where(WatchedItemRow.external_id == external_id)
            )).scalars().first()
            if row is None:
                return None
            row.deleted_at = _now()
            row.updated_at = row.deleted_at
            await session.commit()
            return watched_item_dict(row)

    async def set_watched_item_remembered(self, item_id: str, document_id: str) -> None:
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(WatchedItemRow, item_id)
            if row is not None:
                row.remembered_document_id = document_id
                await session.commit()

    async def get_watched_item(self, customer_id: str, item_id: str) -> Optional[dict[str, Any]]:
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(WatchedItemRow, item_id)
            if row is None or row.customer_id != customer_id:
                return None
            return watched_item_dict(row)

    async def list_watched_items(self, customer_id: str, *, connection_id: Optional[str] = None,
                                 kind: Optional[str] = None, since: Optional[datetime] = None,
                                 query: Optional[str] = None, include_deleted: bool = False,
                                 limit: int = 50) -> list[dict[str, Any]]:
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = select(WatchedItemRow).where(WatchedItemRow.customer_id == customer_id)
            if connection_id:
                stmt = stmt.where(WatchedItemRow.connection_id == connection_id)
            if kind:
                stmt = stmt.where(WatchedItemRow.kind == kind)
            if since is not None:
                stmt = stmt.where(func.coalesce(WatchedItemRow.occurred_at, WatchedItemRow.created_at) >= since)
            if not include_deleted:
                stmt = stmt.where(WatchedItemRow.deleted_at.is_(None))
            if query:
                q = f"%{query.strip().lower()}%"
                stmt = stmt.where(or_(func.lower(WatchedItemRow.title).like(q), func.lower(WatchedItemRow.snippet).like(q),
                                      func.lower(WatchedItemRow.author).like(q)))
            stmt = stmt.order_by(func.coalesce(WatchedItemRow.occurred_at, WatchedItemRow.created_at).desc()).limit(
                max(1, min(int(limit or 50), 500)))
            return [watched_item_dict(r) for r in (await session.execute(stmt)).scalars().all()]

    async def purge_expired_watched_items(self, *, now: Optional[datetime] = None) -> int:
        """Q108 A: what the apps showed is kept 7 days, then deleted."""
        now = now or _now()
        async with self.session() as session:  # type: ignore[attr-defined]
            res = await session.execute(delete(WatchedItemRow).where(WatchedItemRow.expires_at <= now))
            await session.commit()
            return int(res.rowcount or 0)
