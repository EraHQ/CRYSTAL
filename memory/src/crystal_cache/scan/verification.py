"""Assumption verification — influence-driven spawn (C4 Q1=A,
ratified 2026-08-11).

The passive verification loop already exists: approved assumptions
accrue grounded citations and the tier-promotion sweep raises them;
the convergence scans surface conflicts against them (C3 made
assumption facts visible to those scans). This pass adds the ACTIVE
half for the assumptions the bank is actually LEANING ON: an approved
assumption still in quarantine/neutral whose grounded-citation count
has reached the threshold is influencing real answers while
unverified — exactly the inference worth spending research on. It is
enqueued as a cognition task whose goal asks for evidence that
confirms OR refutes (Q3=A: refutation framed as equally valuable —
no confirmation-bias goal).

Substrate-pure by design (Q3=A): this pass spawns and witnesses;
it never judges. The research crystal lands in the bank through the
normal review-gated scratchpad, where the C3 segment scans bring it
face to face with the assumption — support becomes citations for the
promotion sweep, refutation surfaces a knowledge_conflict into the
ratified invalidation path. No second promotion or invalidation
authority is created here.

Never spawned for: recall-gated assumptions (not yet approved — the
curator hasn't put them in play), blacklist (already invalidated),
whitelist (the passive loop already finished its job), assumptions
with OPEN conflicts (already in the invalidation path — verifying a
disputed inference re-buys what the conflict surface already knows),
or assumptions already tagged `verification_task:` (durable
once-per-assumption idempotence; the manual endpoint can respawn).

Spend discipline (Q2=B): the scan itself makes NO model calls —
enqueue only. The WORKER caller gates each tenant through
function_budget_allows(function="assumption_verification",
origin="assumption_verification") BEFORE calling this, so no task is
enqueued that can't afford to run; the task's model calls land in the
ledger under their own origin (the engine threads env.origin), which
IS the meter. Default cap 0 = OFF: the manual-by-default posture.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import structlog

from ..config import settings

if TYPE_CHECKING:
    from ..infrastructure import MetadataStore

logger = structlog.get_logger(__name__)

VERIFICATION_TASK_TYPE = "assumption_verification"
VERIFICATION_TAG_PREFIX = "verification_task:"

# Q3=A neutral goal framing: confirmation and refutation are the same
# deliverable — what the evidence actually shows.
_GOAL_TEMPLATE = (
    'Find evidence that CONFIRMS OR REFUTES this inference: "{statement}". '
    "Refutation is exactly as valuable as confirmation - report what the "
    "evidence actually shows, including partial support or contradiction. "
    "If the available material cannot settle it, say precisely what is "
    "missing."
)


@dataclass
class VerificationScanResult:
    """Outcome of one verification-spawn pass for one customer."""

    customer_id: str
    assumptions_scanned: int
    tasks_spawned: int
    skipped_tagged: int
    skipped_conflicted: int


def has_verification_tag(diagnostic_tags: Optional[list]) -> bool:
    """True when a spawn already happened for this assumption (durable
    idempotence — the tag survives restarts, unlike task-table scans)."""
    return any(
        str(t).startswith(VERIFICATION_TAG_PREFIX)
        for t in (diagnostic_tags or [])
    )


def verification_goal(statement: str) -> str:
    """The task's goal text for an assumption's statement."""
    return _GOAL_TEMPLATE.format(statement=(statement or "").strip())


async def run_assumption_verification_scan(
    *,
    store: "MetadataStore",
    customer_id: str,
    min_recalls: Optional[int] = None,
    per_cycle: Optional[int] = None,
    log: Any = None,
) -> VerificationScanResult:
    """One spawn pass over a customer's assumptions (C4 Q1=A).

    Trigger, all conditions required:
      - crystal_type == assumption, recall gate CLEARED (approved)
      - quality_tier in (quarantine, neutral) — whitelist means the
        passive loop already promoted it; blacklist means invalidated
      - grounded citations >= min_recalls (the influence meter —
        the same signal the tier-promotion sweep reads)
      - zero OPEN conflicts (a disputed assumption is already in the
        invalidation path)
      - no prior `verification_task:` tag

    Spawns at most `per_cycle` tasks; knobs default from settings.
    Budget gating is the CALLER's job (see module docstring) — this
    function spends no model calls and checks no budgets.
    """
    log = log or logger
    if min_recalls is None:
        min_recalls = settings.assumption_verification_min_recalls
    if per_cycle is None:
        per_cycle = settings.assumption_verification_per_cycle

    rows = await store.list_assumption_crystals(customer_id)
    scanned = 0
    spawned = 0
    skipped_tagged = 0
    skipped_conflicted = 0

    for r in rows:
        if spawned >= max(0, per_cycle):
            break
        scanned += 1
        if r["recall_gated"] or r["quality_tier"] not in (
            "quarantine", "neutral",
        ):
            continue
        if has_verification_tag(r["diagnostic_tags"]):
            skipped_tagged += 1
            continue
        cites = await store.count_grounded_citations_for_crystal(
            customer_id, r["id"],
        )
        if cites < min_recalls:
            continue
        conflicts = await store.count_open_conflicts_for_crystal(
            customer_id, r["id"],
        )
        if conflicts > 0:
            skipped_conflicted += 1
            continue

        statement = (r["statement"] or "").strip()
        task = await store.create_cognition_task(
            customer_id,
            task_type=VERIFICATION_TASK_TYPE,
            payload={
                "topic": verification_goal(statement),
                "assumption_crystal_id": r["id"],
                "statement": statement,
            },
            priority="background",
        )
        await store.tag_assumption_verification(
            customer_id, r["id"], task.id,
        )
        spawned += 1
        # C4 witness (the C2 activity feed): the system saying, out
        # loud, "I am spending money to test my own inference." Best-
        # effort by contract.
        try:
            await store.record_curation_event(
                customer_id,
                event_type="verification_spawned",
                subject_id=r["id"],
                label=(
                    f"Verification queued - {statement}"[:256]
                    if statement else "Verification queued"
                ),
                payload={
                    "task_id": task.id,
                    "grounded_citations": cites,
                },
            )
        except Exception:  # noqa: BLE001 — witness never breaks the spawn
            log.debug("curation_event.emit_failed", exc_info=True)
        log.info(
            "assumption_verification.spawned",
            customer_id=customer_id,
            crystal_id=r["id"],
            task_id=task.id,
            grounded_citations=cites,
        )

    return VerificationScanResult(
        customer_id=customer_id,
        assumptions_scanned=scanned,
        tasks_spawned=spawned,
        skipped_tagged=skipped_tagged,
        skipped_conflicted=skipped_conflicted,
    )
