"""Conversation-persistence store primitives — CRYS session continuity (P5).

The CRUD behind the `agent_conversations` table: per-scope conversation
transcripts so CRYS can resume context across exit/relaunch. Mode-agnostic —
`conversation_key` is the resolved project_dir for the CLI coding mode, a
thread id for the future general/web mode (the Inspector chat playground
becoming CRYS).

Same binding pattern as the other extension mixins (setattr in
infrastructure/__init__.py via _bind_mixin_methods). Methods return plain
dicts — a conversation is operational state, not a domain entity (the
agent_sessions / agent_tasks precedent).

UPSERT-BY-SCOPE: `upsert_conversation` is select-then-update-or-insert on
(customer_id, conversation_key). The CLI reuses the project_dir key, so a
relaunch resumes the SAME rolling conversation (overwrite, not a new row);
the web mode uses unique thread ids for many threads. Atomic under SQLite's
single-writer transaction; the `ux_agent_conversations_scope` unique index is
the backstop. The CALLER caps the transcript before writing (the store
persists what it is given).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import structlog
from sqlalchemy import select

from .schema import AgentConversationRow

logger = structlog.get_logger(__name__)


def _conversation_to_dict(row: AgentConversationRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "customer_id": row.customer_id,
        "conversation_key": row.conversation_key,
        "mode": row.mode,
        "title": row.title,
        "transcript": row.transcript or [],
        "turn_count": row.turn_count,
        "last_summary": row.last_summary,
        "meta": row.meta,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


class ConversationExtensionsMixin:
    """agent_conversations CRUD, bound onto MetadataStore."""

    async def upsert_conversation(
        self,
        customer_id: str,
        *,
        conversation_key: str,
        transcript: list,
        turn_count: int,
        last_summary: Optional[str] = None,
        mode: str = "coding",
        title: Optional[str] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Create or overwrite the conversation for (customer, scope).

        If a row exists for this (customer_id, conversation_key) it is
        updated in place (updated_at refreshes via onupdate); otherwise a
        fresh row is inserted. Returns the row as a dict.
        """
        import uuid

        async with self.session() as session:  # type: ignore[attr-defined]
            existing = (
                await session.execute(
                    select(AgentConversationRow)
                    .where(AgentConversationRow.customer_id == customer_id)
                    .where(
                        AgentConversationRow.conversation_key == conversation_key
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()

            if existing is not None:
                existing.transcript = transcript
                existing.turn_count = turn_count
                existing.last_summary = last_summary
                existing.mode = mode
                if title is not None:
                    existing.title = title
                existing.meta = meta
                return _conversation_to_dict(existing)

            row = AgentConversationRow(
                id=f"conv_{uuid.uuid4().hex[:16]}",
                customer_id=customer_id,
                conversation_key=conversation_key,
                mode=mode,
                title=title,
                transcript=transcript,
                turn_count=turn_count,
                last_summary=last_summary,
                meta=meta,
            )
            session.add(row)
            return _conversation_to_dict(row)

    async def get_conversation(
        self, customer_id: str, *, conversation_key: str
    ) -> Optional[dict[str, Any]]:
        """The saved conversation for (customer, scope), or None."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = (
                await session.execute(
                    select(AgentConversationRow)
                    .where(AgentConversationRow.customer_id == customer_id)
                    .where(
                        AgentConversationRow.conversation_key == conversation_key
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()
            return _conversation_to_dict(row) if row is not None else None

    async def delete_conversation(
        self, customer_id: str, *, conversation_key: str
    ) -> bool:
        """Delete the conversation for (customer, scope). Returns True if a
        row was removed (the /reset path). False if nothing matched."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = (
                await session.execute(
                    select(AgentConversationRow)
                    .where(AgentConversationRow.customer_id == customer_id)
                    .where(
                        AgentConversationRow.conversation_key == conversation_key
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()
            if row is None:
                return False
            await session.delete(row)
            return True

    async def list_conversations(
        self, customer_id: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        """A customer's conversations, most-recently-updated first. Backs the
        future web thread list; unused by the CLI's single-project resume."""
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (
                await session.execute(
                    select(AgentConversationRow)
                    .where(AgentConversationRow.customer_id == customer_id)
                    .order_by(AgentConversationRow.updated_at.desc())
                    .limit(limit)
                )
            ).scalars().all()
            return [_conversation_to_dict(r) for r in rows]

    async def get_conversation_model(
        self, customer_id: str, *, conversation_key: str
    ) -> Optional[str]:
        """The model saved for (customer, scope), or None.

        The read half of per-conversation model selection (C6). Returns the
        sticky model the conversation was last set to; None when no row exists
        or the row predates a model choice — the caller then falls back to the
        CC_AGENT_MODEL house default and finally the built-in DEFAULT_MODEL.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            row = (
                await session.execute(
                    select(AgentConversationRow)
                    .where(AgentConversationRow.customer_id == customer_id)
                    .where(
                        AgentConversationRow.conversation_key == conversation_key
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()
            return row.model if row is not None else None

    async def set_conversation_model(
        self, customer_id: str, *, conversation_key: str, model: str
    ) -> None:
        """Persist the model for (customer, scope) — the write half of C6.

        Last-writer-wins: the client's explicit model becomes this
        conversation's sticky model, reused on later turns from any device.
        FOCUSED — touches only the `model` column on an existing row, never the
        transcript/meta, so it cannot clobber a resumed conversation's state
        (the reason model is a typed column and not a `meta` key). Inserts a
        thin row (empty transcript, mode='general') when the conversation
        doesn't exist yet — the common case for the web chat, where the model
        choice is the first backend state the conversation persists.
        """
        import uuid

        async with self.session() as session:  # type: ignore[attr-defined]
            existing = (
                await session.execute(
                    select(AgentConversationRow)
                    .where(AgentConversationRow.customer_id == customer_id)
                    .where(
                        AgentConversationRow.conversation_key == conversation_key
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()

            if existing is not None:
                existing.model = model
                return

            row = AgentConversationRow(
                id=f"conv_{uuid.uuid4().hex[:16]}",
                customer_id=customer_id,
                conversation_key=conversation_key,
                mode="general",
                transcript=[],
                turn_count=0,
                model=model,
            )
            session.add(row)
