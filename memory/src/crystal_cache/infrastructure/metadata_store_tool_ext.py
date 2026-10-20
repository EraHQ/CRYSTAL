"""tool_invocations: every tool call from a connected client, and the
per-call model ledger reads the Activity trace needs (2026-10-19).
Bound onto MetadataStore like every other extension; all SQL lives
here (R9). Arguments are shortened at the door (`shorten_args`) so a
pasted document never lands in the log whole.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select

from .schema import LlmCallRow, ToolInvocationRow

ARG_CHARS = 300
# Argument names that are credentials, never memory content. A bare
# "key" is a memory key ("Rate|Now") and stays visible.
_SECRET_KEYS = ("api_key", "apikey", "token", "secret", "password", "authorization")


def shorten(value: Any, *, chars: int = ARG_CHARS, depth: int = 0) -> Any:
    """A JSON-safe, bounded copy of a tool argument or result: strings
    cut to `chars`, lists to their first five items (with a count),
    dicts one level deep, everything else as str."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= chars else value[:chars] + f"… (+{len(value) - chars} chars)"
    if isinstance(value, (list, tuple)):
        head = [shorten(v, chars=chars, depth=depth + 1) for v in list(value)[:5]]
        return head if len(value) <= 5 else head + [f"… {len(value) - 5} more"]
    if isinstance(value, dict):
        if depth >= 1:
            return f"{{{len(value)} keys}}"
        return {str(k): shorten(v, chars=chars, depth=depth + 1) for k, v in list(value.items())[:20]}
    return str(value)[:chars]


def shorten_args(kwargs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in kwargs.items():
        if v is None:
            continue
        lk = str(k).lower()
        out[str(k)] = "(hidden)" if any(s in lk.replace("-", "_") for s in _SECRET_KEYS) else shorten(v)
    return out


def summarize_result(out: Any) -> str:
    """One line about what came back: list sizes and the small scalars
    of a dict, or the head of a string."""
    if isinstance(out, dict):
        parts: list[str] = []
        for k, v in list(out.items())[:12]:
            if isinstance(v, (list, tuple)):
                parts.append(f"{k}: {len(v)}")
            elif isinstance(v, (bool, int, float)) or (isinstance(v, str) and len(v) <= 80):
                parts.append(f"{k}: {v}")
            elif isinstance(v, str):
                parts.append(f"{k}: {len(v)} chars")
            elif isinstance(v, dict):
                parts.append(f"{k}: {{{len(v)} keys}}")
        return "; ".join(parts)[:500]
    if isinstance(out, str):
        return out[:500]
    if isinstance(out, (list, tuple)):
        return f"{len(out)} items"
    return ""


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _invocation_to_dict(r: ToolInvocationRow) -> dict[str, Any]:
    return {
        "id": r.id, "customer_id": r.customer_id, "surface": r.surface, "client": r.client,
        "tool": r.tool, "args": r.args, "ok": bool(r.ok), "code": r.code,
        "result_summary": r.result_summary, "duration_ms": r.duration_ms,
        "query_log_id": r.query_log_id, "created_at": _iso(r.created_at),
    }


def llm_call_to_dict(r: LlmCallRow) -> dict[str, Any]:
    return {
        "id": r.id, "session_id": r.session_id, "parent_session_id": r.parent_session_id,
        "origin": r.origin, "model": r.model,
        "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
        "cache_creation_tokens": r.cache_creation_tokens, "cache_read_tokens": r.cache_read_tokens,
        "cost_micro_usd": r.computed_cost_micro_usd, "created_at": _iso(r.created_at),
    }


class ToolInvocationsMixin:
    async def record_tool_invocation(
        self, customer_id: str, *, tool: str, surface: str = "mcp", client: Optional[str] = None,
        args: Optional[dict[str, Any]] = None, ok: bool = True, code: Optional[str] = None,
        result_summary: Optional[str] = None, duration_ms: Optional[int] = None,
        query_log_id: Optional[str] = None,
    ) -> str:
        row = ToolInvocationRow(
            id=f"tcall_{uuid.uuid4().hex[:16]}", customer_id=customer_id, surface=surface,
            client=(client or None), tool=tool[:64], args=args, ok=ok, code=(code or None),
            result_summary=(result_summary or None), duration_ms=duration_ms,
            query_log_id=query_log_id, created_at=datetime.now(timezone.utc),
        )
        async with self.session() as session:  # type: ignore[attr-defined]
            session.add(row)
        return row.id

    async def list_tool_invocations(
        self, customer_id: str, *, limit: int = 200, since: Optional[datetime] = None,
        query_log_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = select(ToolInvocationRow).where(ToolInvocationRow.customer_id == customer_id)
            if since is not None:
                stmt = stmt.where(ToolInvocationRow.created_at >= since)
            if query_log_id:
                stmt = stmt.where(ToolInvocationRow.query_log_id == query_log_id)
            stmt = stmt.order_by(ToolInvocationRow.created_at.desc()).limit(limit)
            rows = (await session.execute(stmt)).scalars().all()
            return [_invocation_to_dict(r) for r in rows]

    async def list_llm_calls(
        self, customer_id: str, *, session_id: Optional[str] = None,
        since: Optional[datetime] = None, until: Optional[datetime] = None, limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Per-call rows of the model ledger, newest first (the aggregates
        live in the cost extension)."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = select(LlmCallRow).where(LlmCallRow.customer_id == customer_id)
            if session_id:
                stmt = stmt.where(LlmCallRow.session_id == session_id)
            if since is not None:
                stmt = stmt.where(LlmCallRow.created_at >= since)
            if until is not None:
                stmt = stmt.where(LlmCallRow.created_at <= until)
            stmt = stmt.order_by(LlmCallRow.created_at.desc()).limit(limit)
            rows = (await session.execute(stmt)).scalars().all()
            return [llm_call_to_dict(r) for r in rows]
