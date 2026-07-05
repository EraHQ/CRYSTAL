"""MetadataStore web-search-log extension (launch-prep sweep, 2026-07-02).

Same binding pattern as ConflictExtensionsMixin: bound onto MetadataStore
by infrastructure/__init__._bind_mixin_methods. One writer + one reader
over web_search_logs — the goldmine's raw side (see schema.py's table
comment for the rationale and the provenance join).
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

import structlog
from sqlalchemy import select

from .schema import WebSearchLogRow

logger = structlog.get_logger(__name__)


class WebSearchExtensionsMixin:
    """web_search_logs CRUD bound onto MetadataStore."""

    async def write_web_search_log(
        self,
        customer_id: str,
        *,
        query: str,
        provider: str,
        results: list[dict[str, Any]],
        origin: str = "tool",
    ) -> str:
        """Record one search interaction. Results are stripped to
        title/url/snippet — extracted content never lands in the log."""
        row_id = f"wsl_{uuid.uuid4().hex[:16]}"
        slim = [
            {
                "title": str(r.get("title") or ""),
                "url": str(r.get("url") or ""),
                "snippet": str(r.get("snippet") or "")[:500],
            }
            for r in results
        ]
        async with self.session() as session:  # type: ignore[attr-defined]
            session.add(WebSearchLogRow(
                id=row_id,
                customer_id=customer_id,
                query=query[:4000],
                provider=provider,
                n_results=len(slim),
                results=slim,
                origin=origin,
            ))
        return row_id

    async def list_web_search_logs(
        self,
        customer_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Per-customer search history, newest first."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(WebSearchLogRow)
                .where(WebSearchLogRow.customer_id == customer_id)
                .order_by(WebSearchLogRow.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [
                {
                    "id": r.id,
                    "query": r.query,
                    "provider": r.provider,
                    "n_results": r.n_results,
                    "results": r.results or [],
                    "origin": r.origin,
                    "created_at": r.created_at,
                }
                for r in rows
            ]
