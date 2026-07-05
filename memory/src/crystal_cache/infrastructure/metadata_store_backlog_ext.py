"""Unified backlog read-model — Never-Idle Convergence (D6).

`list_backlog` is the single ranked view over everything a customer's bank
has waiting for action, aggregating the fragmented work queues:

  knowledge_gaps        (status 'open')      — incl. agent-run gaps
                                                (source='agent_run'); they live
                                                in knowledge_gaps, so the gaps
                                                source already covers them.
  knowledge_conflicts   (status 'open')      — the contradiction scan's output
  cognition_tasks       (status 'pending')   — queued research
  agent_tasks           (status 'queued')    — queued coding work
  push_review_queue     (status 'pending')   — observations awaiting review
  verification_tasks    (status 'pending')   — claims awaiting verification

Each row is normalized to a common shape — {kind, id, subject, status,
priority_score, created_at} — merged, and ranked by priority then age
(oldest-first within a priority, so the longest-waiting item of a given
importance rises). The backlog surfaces items AWAITING ACTION; in-progress
(running/retrying) and terminal (filled/resolved/done/approved) states are
deliberately excluded — "nothing waiting" is a correct, empty backlog.

This is a READ-MODEL: no producer is rewired, no new physical queue table.
A single session runs six small SELECTs (uniform, robust to sibling
list-method signatures, one round of converters). R9 is satisfied — the SQL
lives here, in a bound store mixin.

v1 scope: per-customer (D8). priority_score is a coarse 1–3 derived from each
source's own priority field (conflicts/agent-tasks have none → medium);
query-frequency worth ranking is a P5 refinement. Returns plain dicts (the
agent-task / session-registry precedent — a work-queue view is operational,
not a domain entity).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import structlog
from sqlalchemy import select

from .schema import (
    AgentTaskRow,
    CognitionTaskRow,
    KnowledgeConflictRow,
    KnowledgeGapRow,
    PushReviewQueueRow,
    VerificationTaskRow,
)

logger = structlog.get_logger(__name__)

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

# Coarse 1–3 priority from a source's textual priority field.
_TEXT_PRIORITY_SCORES = {
    "high": 3, "urgent": 3, "critical": 3,
    "medium": 2, "normal": 2,
    "low": 1, "background": 1,
}
# Default when a source has no priority field or an unrecognized value.
_DEFAULT_PRIORITY_SCORE = 2


def _text_priority_score(priority: Optional[str]) -> int:
    return _TEXT_PRIORITY_SCORES.get((priority or "").lower(), _DEFAULT_PRIORITY_SCORE)


def _float_priority_score(priority: Optional[float]) -> int:
    """Map a float priority (verification_tasks) onto the 1–3 scale."""
    p = priority or 0.0
    if p >= 0.66:
        return 3
    if p >= 0.33:
        return 2
    return 1


def _truncate(text: Optional[str], n: int = 80) -> str:
    t = " ".join((text or "").split())
    return t[:n]


def _item(
    *, kind: str, item_id: str, subject: str, status: str,
    priority_score: int, created_at: Optional[datetime],
) -> dict[str, Any]:
    return {
        "kind": kind,
        "id": item_id,
        "subject": subject,
        "status": status,
        "priority_score": priority_score,
        "created_at": created_at,
    }


class BacklogExtensionsMixin:
    """The unified backlog read-model, bound onto MetadataStore."""

    async def list_backlog(
        self,
        customer_id: str,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Ranked, normalized view of a customer's waiting work across all
        queues. Highest priority first; oldest-first within a priority.

        Args:
            customer_id: tenant scope (per-customer, D8).
            limit: cap applied per-source AND to the final ranked result.

        Returns:
            list of {kind, id, subject, status, priority_score, created_at}.
        """
        items: list[dict[str, Any]] = []

        async with self.session() as session:  # type: ignore[attr-defined]
            # knowledge_gaps (open) — includes agent-run gaps.
            gap_rows = (await session.execute(
                select(KnowledgeGapRow)
                .where(
                    KnowledgeGapRow.customer_id == customer_id,
                    KnowledgeGapRow.status == "open",
                )
                .order_by(KnowledgeGapRow.created_at.desc())
                .limit(limit)
            )).scalars().all()
            for r in gap_rows:
                subject = r.subject or r.domain or _truncate(r.missing)
                items.append(_item(
                    kind="gap", item_id=r.id, subject=subject or "(gap)",
                    status=r.status, priority_score=_text_priority_score(r.priority),
                    created_at=r.created_at,
                ))

            # knowledge_conflicts (open) — no priority field → medium in v1.
            conflict_rows = (await session.execute(
                select(KnowledgeConflictRow)
                .where(
                    KnowledgeConflictRow.customer_id == customer_id,
                    KnowledgeConflictRow.status == "open",
                )
                .order_by(KnowledgeConflictRow.created_at.desc())
                .limit(limit)
            )).scalars().all()
            for r in conflict_rows:
                items.append(_item(
                    kind="conflict", item_id=r.id,
                    subject=r.subject or "(contradiction)", status=r.status,
                    priority_score=_DEFAULT_PRIORITY_SCORE, created_at=r.created_at,
                ))

            # cognition_tasks (pending).
            cog_rows = (await session.execute(
                select(CognitionTaskRow)
                .where(
                    CognitionTaskRow.customer_id == customer_id,
                    CognitionTaskRow.status == "pending",
                )
                .order_by(CognitionTaskRow.created_at.desc())
                .limit(limit)
            )).scalars().all()
            for r in cog_rows:
                topic = None
                if isinstance(r.payload, dict):
                    topic = r.payload.get("topic")
                items.append(_item(
                    kind="cognition_task", item_id=r.id,
                    subject=_truncate(topic) or r.task_type, status=r.status,
                    priority_score=_text_priority_score(r.priority),
                    created_at=r.created_at,
                ))

            # agent_tasks (queued) — no priority field → medium.
            agent_rows = (await session.execute(
                select(AgentTaskRow)
                .where(
                    AgentTaskRow.customer_id == customer_id,
                    AgentTaskRow.status == "queued",
                )
                .order_by(AgentTaskRow.created_at.desc())
                .limit(limit)
            )).scalars().all()
            for r in agent_rows:
                items.append(_item(
                    kind="agent_task", item_id=r.id, subject=_truncate(r.task),
                    status=r.status, priority_score=_DEFAULT_PRIORITY_SCORE,
                    created_at=r.created_at,
                ))

            # push_review_queue (pending) — review work → medium.
            push_rows = (await session.execute(
                select(PushReviewQueueRow)
                .where(
                    PushReviewQueueRow.customer_id == customer_id,
                    PushReviewQueueRow.status == "pending",
                )
                .order_by(PushReviewQueueRow.created_at.desc())
                .limit(limit)
            )).scalars().all()
            for r in push_rows:
                items.append(_item(
                    kind="push_review", item_id=r.id, subject=_truncate(r.key),
                    status=r.status, priority_score=_DEFAULT_PRIORITY_SCORE,
                    created_at=r.created_at,
                ))

            # verification_tasks (pending) — float priority → 1–3.
            verif_rows = (await session.execute(
                select(VerificationTaskRow)
                .where(
                    VerificationTaskRow.customer_id == customer_id,
                    VerificationTaskRow.status == "pending",
                )
                .order_by(VerificationTaskRow.created_at.desc())
                .limit(limit)
            )).scalars().all()
            for r in verif_rows:
                items.append(_item(
                    kind="verification", item_id=r.id,
                    subject=_truncate(r.candidate_claim), status=r.status,
                    priority_score=_float_priority_score(r.priority),
                    created_at=r.created_at,
                ))

        # Rank: priority desc, then oldest-first within a priority.
        items.sort(key=lambda it: (-it["priority_score"], it["created_at"] or _EPOCH))
        return items[:limit]
