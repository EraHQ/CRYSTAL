"""Control-plane command channel — the Growth G2 store surface.

The outbound-poll model (G2): an operator's command (approve/deny an agent's
approval gate, or terminate a session / dependency) is written to
`control_commands`; the agent POLLS its session for pending commands and
acts. Nothing ever connects inbound to the agent — NAT-safe, and faithful to
"coordinate through a shared store, not direct RPC."

The decision is SIGNED by the operator's key (signature / nonce / signed_at)
and the AGENT verifies it against the pinned public key before acting (see
control/signing.py). The server — and this store — is a courier: it persists
and relays the signed blob but cannot forge a decision.

Three failure modes are designed into the state machine here:
  - first-wins (split-brain: a local terminal AND a remote operator both
    answer) — `claim_next_command_for_session` flips pending→consumed in one
    transaction; the agent acts on the first command it claims and ignores
    decisions for a request it is no longer awaiting.
  - crash-while-awaiting — `void_pending_commands_for_session` lets the F4
    staleness sweep void a crashed session's pending commands.
  - no-decision → the agent's own timeout defaults to deny (control/signing.py
    DECISION_TIMEOUT_SECONDS); nothing to persist here.

R9 keeps the SQL here, bound onto MetadataStore via the __init__ mixin
pattern. Methods return plain dicts (operational rows the control plane +
Inspector branch on), mirroring the session-registry mixin.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import structlog
from sqlalchemy import select, update

from .schema import ControlCommandRow

logger = structlog.get_logger(__name__)

# Command lifecycle.
PENDING = "pending"
CONSUMED = "consumed"
VOIDED = "voided"

# Command types.
APPROVAL_DECISION = "approval_decision"
TERMINATE = "terminate"
TERMINATE_DEPENDENCY = "terminate_dependency"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _command_to_dict(row: ControlCommandRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "session_id": row.session_id,
        "customer_id": row.customer_id,
        "request_id": row.request_id,
        "command_type": row.command_type,
        "decision": row.decision,
        "dependency_id": row.dependency_id,
        "payload": row.payload,
        # The signed-authorization envelope the agent verifies. Surfaced so
        # the agent's poll gets everything it needs to verify locally.
        "signature": row.signature,
        "nonce": row.nonce,
        "signed_at": row.signed_at,
        "issued_by_operator_id": row.issued_by_operator_id,
        "status": row.status,
        "created_at": row.created_at,
        "consumed_at": row.consumed_at,
    }


class ControlExtensionsMixin:
    """control_commands CRUD + claim state machine bound onto MetadataStore."""

    async def create_control_command(
        self,
        session_id: str,
        customer_id: str,
        request_id: str,
        command_type: str,
        *,
        decision: Optional[str] = None,
        dependency_id: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
        signature: Optional[str] = None,
        nonce: Optional[str] = None,
        signed_at: Optional[datetime] = None,
        issued_by_operator_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Write a pending command targeting a session.

        command_type ∈ approval_decision | terminate | terminate_dependency.
        For approval_decision, `decision` is approve|deny and the
        signature/nonce/signed_at carry the operator's authorization for the
        agent to verify. The row lands `pending`; the agent claims it.
        """
        cmd_id = f"cmd_{uuid.uuid4().hex[:16]}"
        async with self.session() as session:  # type: ignore[attr-defined]
            row = ControlCommandRow(
                id=cmd_id,
                session_id=session_id,
                customer_id=customer_id,
                request_id=request_id,
                command_type=command_type,
                decision=decision,
                dependency_id=dependency_id,
                payload=payload,
                signature=signature,
                nonce=nonce,
                signed_at=signed_at,
                issued_by_operator_id=issued_by_operator_id,
                status=PENDING,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            result = _command_to_dict(row)
        logger.info(
            "control_plane.command_created",
            command_id=cmd_id,
            session_id=session_id,
            command_type=command_type,
            decision=decision,
        )
        return result

    async def claim_next_command_for_session(
        self, session_id: str
    ) -> Optional[dict[str, Any]]:
        """Atomically claim the oldest pending command for a session.

        First-wins: the claim flips pending→consumed (stamping consumed_at) in
        one transaction, so a command is delivered to the agent exactly once
        even if two pollers race (SQLite is single-writer; the
        select-then-update is atomic within the transaction). Returns the
        claimed command dict, or None when nothing is pending.

        The agent calls this on its poll loop and acts on the returned command
        (verifying the signature first for approval_decision). A second
        operator's late decision for the same request stays pending until
        claimed, but the agent — no longer awaiting that request — ignores it;
        a periodic void or the staleness sweep reclaims it.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            row = (await session.execute(
                select(ControlCommandRow)
                .where(
                    ControlCommandRow.session_id == session_id,
                    ControlCommandRow.status == PENDING,
                )
                .order_by(ControlCommandRow.created_at)
                .limit(1)
            )).scalar_one_or_none()
            if row is None:
                return None
            row.status = CONSUMED
            row.consumed_at = _utcnow()
            await session.commit()
            await session.refresh(row)
            return _command_to_dict(row)

    async def list_commands_for_session(
        self,
        session_id: str,
        *,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """A session's commands, newest first. Optionally filtered by status.
        Read-only — the Inspector's view of the control channel."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = select(ControlCommandRow).where(
                ControlCommandRow.session_id == session_id
            )
            if status is not None:
                stmt = stmt.where(ControlCommandRow.status == status)
            stmt = stmt.order_by(ControlCommandRow.created_at.desc()).limit(limit)
            rows = (await session.execute(stmt)).scalars().all()
            return [_command_to_dict(r) for r in rows]

    async def void_pending_commands_for_session(self, session_id: str) -> int:
        """Void all pending commands for a session (e.g. when F4 staleness
        presumes it crashed). Returns the number voided."""
        async with self.session() as session:  # type: ignore[attr-defined]
            result = await session.execute(
                update(ControlCommandRow)
                .where(
                    ControlCommandRow.session_id == session_id,
                    ControlCommandRow.status == PENDING,
                )
                .values(status=VOIDED, consumed_at=_utcnow())
            )
            await session.commit()
            voided = result.rowcount or 0
        if voided:
            logger.info(
                "control_plane.commands_voided",
                session_id=session_id,
                count=voided,
            )
        return voided
