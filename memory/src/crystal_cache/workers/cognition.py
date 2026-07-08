"""Cognition worker — processes research tasks and fills knowledge gaps.

Polls `CognitionTask` rows with status='pending' and runs each one
through the cognition engine (orchestrator → workers → validator →
commit gate). Also opportunistically fills open `KnowledgeGap` rows
when no urgent tasks are queued.

v1 layout (replaced by this module):
  - lifespan._cognition_worker: the poll loop, inline.

v2 changes:
  - All DB access goes through Phase 5 MetadataStore methods:
      claim_pending_cognition_task (atomic-on-SQLite per Phase 5 design)
      mark_cognition_task_complete
      mark_cognition_task_failed
      list_open_knowledge_gaps_cross_tenant (added in Phase 6.5 P3.5)
      mark_knowledge_gap_filled
  - Depends on `cognition.engine.run_cognition_workflow` (ported in
    Wave C of Phase 6). Import is at function-call time so Wave A
    of Phase 6 produces importable code without cognition fully
    ported yet — the worker just won't successfully run until
    Wave C lands.

Agent-reframe note (per AGENT_ARCHITECTURE.md): in v2 agent mode,
`cognition_run` is one of the agent's tools and the agent calls it
synchronously. This worker remains for:
  - Push/pull protocol research tasks (proxy mode), where the
    LLM emits a `crystal_research` tool call that lands as a row
    here.
  - Background gap-fill, which neither mode triggers synchronously.

Phase 7.5 will revisit whether this worker is still needed once the
agent path is the flagship — at minimum it stays for proxy-mode
push/pull.
"""
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from ..config import settings
from ..llm import get_llm_client
from .idle import is_quiet
from ..scan import (
    discover_gaps,
    run_tier_promotion_scan,
    run_topic_seeding,
    scan_for_contradictions,
    scan_for_duplicates,
)

if TYPE_CHECKING:
    from ..encoding.base import TextEncoder
    from ..infrastructure.fact_vector_store import FactVectorStore
    from ..infrastructure.metadata_store import MetadataStore

logger = structlog.get_logger(__name__)


# Env var name for the cognition worker's poll interval.
#
# Phase 10C (P0.88, CU-24) introduced the rename from
# `CC_METACOGNITION_INTERVAL_SECONDS` (misnamed pre-MCR — this
# worker handles research tasks, not the metacognitive layer)
# to `CC_COGNITION_WORKER_INTERVAL_SECONDS` with a deprecation
# alias for one release cycle.
#
# **Phase 11 (P0.98) retired the alias.** Only the new name is
# honored; the old name is now silently ignored. Operators who
# missed the Phase 10C deprecation warning will fall through to
# the default 600. The retirement was scheduled by P0.88's note
# ("removed in Phase 11 or 11.5").
_NEW_INTERVAL_ENV_VAR = "CC_COGNITION_WORKER_INTERVAL_SECONDS"
_DEFAULT_INTERVAL_SECONDS = 600

# Gap-fill backoff (idle-log storm fix, 2026-06-08; front-tuned 2026-06-09).
# A gap that fails to auto-fill is retried on a schedule and parked after
# GAP_MAX_ATTEMPTS, so the worker stops re-running the same unfillable gap
# every poll cycle.
#
# The schedule has two phases:
#   1. Quick-retry phase: the first GAP_QUICK_RETRIES failures cool down for
#      only GAP_QUICK_DELAY_SECONDS each. This catches *transient* failures
#      (a flaky SLM call, or a document that finishes crystallizing moments
#      later) cheaply — with C2's answerability gate, an empty-bank retry now
#      runs only the cheap orchestrator + retrieval, not the full pipeline.
#   2. Exponential phase: subsequent failures back off from
#      GAP_BACKOFF_BASE_SECONDS (~1h), doubling each time, until the gap is
#      parked. This handles the genuinely-unfillable gap without a storm.
#
# State is process-local (see run_cognition_worker): gaps stay 'open' in the
# DB so the inspector still surfaces them, and a restart re-evaluates.
# Persisting the attempt count as a knowledge_gaps column is a clean future
# upgrade now that Alembic is wired.
GAP_MAX_ATTEMPTS = 4
GAP_QUICK_RETRIES = 2
GAP_QUICK_DELAY_SECONDS = 300
GAP_BACKOFF_BASE_SECONDS = 3600


def _resolve_cognition_poll_interval() -> int:
    """Resolve the cognition worker's poll interval from env.

    Honors `CC_COGNITION_WORKER_INTERVAL_SECONDS` (the post-rename
    name from Phase 10C P0.88); falls back to default 600 when
    unset. The old name `CC_METACOGNITION_INTERVAL_SECONDS` is
    silently ignored as of Phase 11 (P0.98) — its deprecation
    cycle ended.
    """
    new_value = os.environ.get(_NEW_INTERVAL_ENV_VAR)
    if new_value is not None:
        return int(new_value)
    return _DEFAULT_INTERVAL_SECONDS


def _record_gap_failure(
    gap_backoff: dict[str, dict], gap_id: str, now: float, *, permanent: bool = False
) -> dict:
    """Record a failed gap-fill attempt and schedule the next retry.

    Two-phase schedule (see the GAP_* constants):
      * The first GAP_QUICK_RETRIES failures cool down for
        GAP_QUICK_DELAY_SECONDS each — quick probes that catch
        transient failures.
      * After that, exponential backoff starting at
        GAP_BACKOFF_BASE_SECONDS (~1h) and doubling each time, until
        GAP_MAX_ATTEMPTS is reached — after which the gap is parked
        for the life of this process.

    permanent=True skips the schedule and parks the gap immediately
    (for the life of this process). Used when the engine reports
    `needs_capability`: the C2 gate already established the bank has
    no grounding and no external tool can supply it, so retrying on a
    timer re-buys the same verdict with fresh orchestrator tokens.
    The gap row stays open in the DB — a process restart (which
    typically accompanies new documents or new capabilities) makes it
    eligible again.

    Returns the gap's backoff entry so the caller can log
    attempts/parked state.
    """
    bo = gap_backoff.setdefault(gap_id, {"attempts": 0, "next_eligible": 0.0})
    bo["attempts"] += 1
    if permanent:
        bo["attempts"] = max(bo["attempts"], GAP_MAX_ATTEMPTS)
        bo["next_eligible"] = float("inf")
        return bo
    attempts = bo["attempts"]
    if attempts <= GAP_QUICK_RETRIES:
        delay = GAP_QUICK_DELAY_SECONDS
    else:
        # First post-quick attempt backs off by the base (~1h); each
        # further attempt doubles it.
        exponent = attempts - GAP_QUICK_RETRIES - 1
        delay = GAP_BACKOFF_BASE_SECONDS * (2 ** exponent)
    bo["next_eligible"] = now + delay
    return bo


async def run_cognition_worker(
    *,
    store: "MetadataStore",
    fact_vector_store: "FactVectorStore",
    encoder: "TextEncoder",
    shutdown_event: asyncio.Event,
) -> None:
    """Background poll loop.

    Reads `CC_COGNITION_WORKER_INTERVAL_SECONDS` (default 600) from
    env. Phase 11 (P0.98) retired the Phase 10C deprecation alias
    for the old name `CC_METACOGNITION_INTERVAL_SECONDS`; the old
    name is now silently ignored. The current new name is also
    distinct from `CC_METACOG_WORKER_INTERVAL_SECONDS`, which
    controls the separate Phase 10B metacognition worker.
    """
    poll_interval = _resolve_cognition_poll_interval()
    logger.info("cognition_worker.started", poll_interval=poll_interval)

    # Process-local gap-fill backoff state (see GAP_MAX_ATTEMPTS).
    gap_backoff: dict[str, dict] = {}

    # Process-local never-idle contradiction-scan state (Phase 3): the daily
    # discriminator-call counter (reset at the UTC day boundary) and the
    # round-robin customer offset. Same process-local posture as gap_backoff;
    # a restart resets the daily ceiling, which is acceptable for v1.
    scan_state: dict = {"day": None, "calls_today": 0, "cust_offset": 0}

    while not shutdown_event.is_set():
        try:
            # Phase 1: process up to N pending research tasks. Cognition
            # model calls route through the provider-neutral seam; tasks
            # fail loudly when no provider is configured.
            processed_count = await _process_pending_tasks(
                store=store,
                fact_vector_store=fact_vector_store,
                encoder=encoder,
                max_tasks=5,
            )

            # Phase 2: if no tasks were processed this cycle, opportunistically
            # fill open gaps and run the Phase 3 convergence scans. Both
            # route through the seam and gate on its readiness — and on the
            # load-aware idle gate (workers/idle.py): a deployment actively
            # serving /v1/* traffic is not idle, whatever the cognition
            # queue says (Core Principle #1).
            if processed_count == 0 and is_quiet(settings.idle_quiet_seconds):
                if get_llm_client().is_ready():
                    await _fill_open_gaps(
                        store=store,
                        fact_vector_store=fact_vector_store,
                        encoder=encoder,
                        max_gaps=3,
                        gap_backoff=gap_backoff,
                    )

                # Phase 3 (Never-Idle Convergence): when idle and enabled,
                # scan a rotating slice of customers for contradictions and
                # surface knowledge_conflicts. Budget-bounded (per cycle + per
                # UTC day); surfacing-only. The autonomous path is gated by
                # enable_convergence_scan (OFF by default); the admin endpoint
                # runs on demand regardless. The scans route through the
                # provider-neutral seam and no-op when it is not configured.
                if settings.enable_convergence_scan:
                    await _run_contradiction_scan(
                        store=store,
                        scan_state=scan_state,
                        customers_per_cycle=settings.convergence_customers_per_cycle,
                        max_candidate_pairs=settings.convergence_max_pairs_per_scan,
                        max_calls_per_cycle=settings.convergence_max_calls_per_cycle,
                        max_calls_per_day=settings.convergence_max_calls_per_day,
                    )

                # Dedup scan (P5): same candidate set + shared budget as the
                # contradiction scan; a DUPLICATE verdict surfaces a
                # knowledge_conflict (detector='dedup_scan'), resolvable through
                # the same gate. Independently gated, OFF by default.
                if settings.enable_dedup_scan:
                    await _run_dedup_scan(
                        store=store,
                        scan_state=scan_state,
                        customers_per_cycle=settings.convergence_customers_per_cycle,
                        max_candidate_pairs=settings.convergence_max_pairs_per_scan,
                        max_calls_per_cycle=settings.convergence_max_calls_per_cycle,
                        max_calls_per_day=settings.convergence_max_calls_per_day,
                    )

                # Gap-discovery scan (P5): per-subject "what's missing?" → a
                # knowledge_gap (source='gap_discovery') the Phase-2 fill sweep
                # can act on. Independently gated, OFF by default; shares the
                # same daily call ceiling.
                if settings.enable_gap_discovery:
                    await _run_gap_discovery(
                        store=store,
                        scan_state=scan_state,
                        customers_per_cycle=settings.convergence_customers_per_cycle,
                        max_subjects_per_cycle=settings.gap_discovery_max_subjects_per_cycle,
                        max_calls_per_day=settings.convergence_max_calls_per_day,
                    )

                # Tier promotion (launch-prep sweep): quality tiers that
                # MOVE. No model calls — pure store signals — so it takes
                # no budget from the shared daily ceiling and needs no
                # seam gate. Walks every customer each idle cycle.
                if settings.enable_tier_promotion:
                    await _run_tier_promotion(store=store)

                # Outbound review (2026-07-03, RATIFIED): the high-tier
                # model half of the review gate for background-worker
                # memory. Stamps outbound_scan_passed / failed verdicts on
                # gated crystals; runs BEFORE the system-rules pass so a
                # fresh verdict and the user's promotion rule can land in
                # the same idle cycle. Explicit opt-in — spends frontier
                # calls.
                if settings.enable_outbound_scan:
                    await _run_outbound_review(store=store)

                # System rules (2026-07-03): the user-owned judgment-
                # automation pass. Evaluates each customer's enabled
                # promotion rules against their recall-gated crystals,
                # clearing gates where the user's conditions hold.
                # Gate-clearing and tier accrual are orthogonal, so order
                # relative to tier promotion doesn't matter; it sits here
                # with the other no-model-call idle passes. A no-op until a
                # user writes a rule (human approval stays the default).
                if settings.enable_system_rules:
                    await _run_system_promotion_rules(store=store)

                # Topic seeding (§3 remainder): research seeds without model
                # calls — thin crystals + the operator topic list write
                # knowledge_gaps the Phase-2 fill sweep consumes. Flood-
                # guarded per customer; spends nothing itself.
                if settings.enable_topic_seeding:
                    await _run_topic_seeding(store=store)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(
                "cognition_worker.poll_error",
                error=str(e),
                error_type=type(e).__name__,
            )

        # Sleep for poll_interval OR until shutdown
        try:
            await asyncio.wait_for(
                shutdown_event.wait(),
                timeout=poll_interval,
            )
            break
        except asyncio.TimeoutError:
            pass

    logger.info("cognition_worker.stopped")


async def _process_pending_tasks(
    *,
    store: "MetadataStore",
    fact_vector_store: "FactVectorStore",
    encoder: "TextEncoder",
    max_tasks: int,
) -> int:
    """Claim and process up to `max_tasks` pending tasks.

    Returns the number of tasks attempted (whether they succeeded or
    failed). Each claim is atomic via the Phase 5
    `claim_pending_cognition_task` method.
    """
    # Cognition engine import is lazy so Wave A produces importable
    # code without the cognition package fully ported yet.
    try:
        from ..cognition.engine import run_cognition_workflow
    except ImportError:
        logger.warning(
            "cognition_worker.engine_not_available",
            note=(
                "cognition.engine.run_cognition_workflow not yet ported "
                "(expected during Phase 6 Wave A; lands in Wave C). "
                "Worker is idling."
            ),
        )
        return 0

    processed = 0
    for _ in range(max_tasks):
        task = await store.claim_pending_cognition_task()
        if task is None:
            break

        if not get_llm_client().is_ready():
            await store.mark_cognition_task_failed(
                task.id,
                error_message="No LLM provider configured (set CC_LLM_API_KEY or ANTHROPIC_API_KEY)",
                completed_at=datetime.now(timezone.utc),
            )
            processed += 1
            continue

        try:
            payload = task.payload or {}
            topic = payload.get("topic", "")
            conv_context = payload.get("conversation_context", "")

            cog_result = await run_cognition_workflow(
                goal=topic,
                customer_id=task.customer_id,
                store=store,
                fact_store=fact_vector_store,
                encoder=encoder,
                conversation_context=conv_context,
                source_crystal_id=payload.get("source_crystal_id", ""),
                output_type="crystal",
                trigger_type="research",
                trigger_id=task.id,
            )

            result = {
                "topic": topic,
                "findings": cog_result.text[:2000] if cog_result.text else "",
                "source": "cognition_engine",
                "tokens_used": cog_result.tokens_used,
                "cost_usd": cog_result.cost_usd,
                "confidence": cog_result.confidence,
            }

            if cog_result.success:
                result["action"] = "inferred_fact_created"
                result["crystal_id"] = cog_result.crystal_id
                # S4: a task promoted FROM a gap (manual Research click)
                # closes that gap on success — same terminal state the
                # auto sweep writes.
                _gap_id = (task.payload or {}).get("gap_id")
                if _gap_id and cog_result.crystal_id:
                    try:
                        await store.mark_knowledge_gap_filled(
                            _gap_id,
                            filled_by_crystal_id=cog_result.crystal_id,
                            resolved_at=datetime.now(timezone.utc),
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "cognition_worker.gap_close_failed",
                            gap_id=_gap_id, error=str(e),
                        )
                result["confidence"] = cog_result.confidence
                logger.info(
                    "cognition_worker.research_complete",
                    task_id=task.id,
                    crystal_id=cog_result.crystal_id,
                    confidence=cog_result.confidence,
                    tokens=cog_result.tokens_used,
                    cost=cog_result.cost_usd,
                    topic=topic[:60],
                )
            else:
                result["action"] = "no_actionable_findings"
                result["reason"] = cog_result.reason
                result["recommendation"] = (
                    "Additional documents may be needed to answer this question."
                )
                # S10: verdict writeback — a needs_capability conclusion
                # on a gap-promoted task flips the GAP to needs_document
                # (durable: sweep skips it, S5 moves it to Your Tasks,
                # the Research button stops re-offering itself).
                _gap_id = (task.payload or {}).get("gap_id")
                if _gap_id and (cog_result.reason or "").startswith(
                    "needs_capability"
                ):
                    try:
                        await store.update_knowledge_gap_disposition(
                            _gap_id, "needs_document"
                        )
                        result["gap_disposition"] = "needs_document"
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "cognition_worker.gap_writeback_failed",
                            gap_id=_gap_id, error=str(e),
                        )

            await store.mark_cognition_task_complete(
                task.id,
                result=result,
                result_crystal_id=cog_result.crystal_id if cog_result.success else None,
                completed_at=datetime.now(timezone.utc),
            )
            processed += 1

        except Exception as e:
            await store.mark_cognition_task_failed(
                task.id,
                error_message=str(e),
                completed_at=datetime.now(timezone.utc),
            )
            logger.error(
                "cognition_worker.task_failed",
                task_id=task.id,
                error=str(e),
                error_type=type(e).__name__,
            )
            processed += 1

    return processed


async def _fill_open_gaps(
    *,
    store: "MetadataStore",
    fact_vector_store: "FactVectorStore",
    encoder: "TextEncoder",
    max_gaps: int,
    gap_backoff: dict[str, dict],
) -> None:
    """Try to fill open knowledge gaps via cognition workflows.

    Iterates the oldest open gaps cross-tenant (up to max_gaps) and
    runs the cognition engine against each gap's `missing` text. On
    success the gap is marked filled with the resulting crystal_id.

    Phase 6.5 P3.5: uses the new
    `list_open_knowledge_gaps_cross_tenant` store method (CU-11
    closed). Before P3.5, this function was a no-op stub that
    logged "skipped" without actually filling anything — see
    CLAUDE.md R6 on stubs disguised as real code.
    """
    try:
        from ..cognition.engine import run_cognition_workflow
    except ImportError:
        # Engine not yet ported (Wave C will land it). Silently skip.
        return

    open_gaps = await store.list_open_knowledge_gaps_cross_tenant(limit=max_gaps)
    if not open_gaps:
        return

    # Skip gaps that are cooling down or parked after repeated failures
    # (see GAP_MAX_ATTEMPTS). This is what stops the worker from
    # re-running the same unfillable gap every poll cycle.
    now = time.time()
    # S4 (2026-07-08, ratified B-1: MANUAL BY DEFAULT): the fill sweep —
    # which IS auto-research — is budget-gated PER TENANT via the
    # spend_budgets substrate. No auto_research row (and a zero config
    # default) = the tenant's gaps are skipped; the manual Research
    # button remains. Self-host single-tenant deploys can enable via
    # CC_AUTO_RESEARCH_DEFAULT_MONTHLY_CAP_MICRO_USD without touching
    # the table. Dispositions: only researchable gaps (or pre-S4 NULL
    # rows, for continuity) are auto-researched — workable and
    # needs_document are never burned by this sweep.
    from ..control.admission import function_budget_allows

    budget_verdicts: dict[str, bool] = {}
    eligible = []
    for gap in open_gaps:
        if gap.disposition not in (None, "researchable"):
            continue
        bo = gap_backoff.get(gap.id)
        if bo is not None:
            if bo["attempts"] >= GAP_MAX_ATTEMPTS:
                continue  # parked for this process
            if now < bo["next_eligible"]:
                continue  # still cooling down
        if gap.customer_id not in budget_verdicts:
            customer = await store.get_customer_by_id(gap.customer_id)
            budget_verdicts[gap.customer_id] = (
                customer is not None
                and await function_budget_allows(
                    store,
                    customer,
                    "auto_research",
                    origin="cognition",
                    default_cap_micro_usd=(
                        settings.auto_research_default_monthly_cap_micro_usd
                    ),
                )
            )
            if not budget_verdicts[gap.customer_id]:
                logger.debug(
                    "cognition_worker.gap_fill_budget_skip",
                    customer_id=gap.customer_id,
                )
        if not budget_verdicts[gap.customer_id]:
            continue
        eligible.append(gap)

    if not eligible:
        return

    logger.info("cognition_worker.gap_fill_starting", count=len(eligible))

    for gap in eligible:
        try:
            # The gap's `missing` field is a natural-language description
            # of what's not in the bank. Treat it as the cognition goal.
            cog_result = await run_cognition_workflow(
                goal=gap.missing,
                customer_id=gap.customer_id,
                store=store,
                fact_store=fact_vector_store,
                encoder=encoder,
                conversation_context="",
                source_crystal_id="",
                output_type="crystal",
                trigger_type="fill_gap",
                trigger_id=gap.id,
            )

            if cog_result.success and cog_result.crystal_id:
                await store.mark_knowledge_gap_filled(
                    gap.id,
                    filled_by_crystal_id=cog_result.crystal_id,
                    resolved_at=datetime.now(timezone.utc),
                )
                gap_backoff.pop(gap.id, None)
                logger.info(
                    "cognition_worker.gap_filled",
                    gap_id=gap.id,
                    customer_id=gap.customer_id,
                    crystal_id=cog_result.crystal_id,
                )
            else:
                _reason = cog_result.reason if cog_result else "no_result"
                # needs_capability is the engine's C2 verdict: zero grounding
                # in the bank and no external tool to supply it. Retrying on
                # the backoff schedule can't change that — park immediately
                # instead of burning an orchestrator call per retry.
                _permanent = bool(_reason and _reason.startswith("needs_capability"))
                if _permanent:
                    # S10: durable park — the verdict writes to the gap
                    # row (needs_document) instead of only an in-memory
                    # backoff that resets on worker restart.
                    try:
                        await store.update_knowledge_gap_disposition(
                            gap.id, "needs_document"
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "cognition_worker.gap_writeback_failed",
                            gap_id=gap.id, error=str(e),
                        )
                bo = _record_gap_failure(
                    gap_backoff, gap.id, now, permanent=_permanent
                )
                logger.info(
                    "cognition_worker.gap_unfilled",
                    gap_id=gap.id,
                    customer_id=gap.customer_id,
                    reason=_reason,
                    attempts=bo["attempts"],
                    parked=bo["attempts"] >= GAP_MAX_ATTEMPTS,
                    parked_permanently=_permanent,
                )

        except Exception as e:
            bo = _record_gap_failure(gap_backoff, gap.id, now)
            logger.warning(
                "cognition_worker.gap_fill_error",
                gap_id=gap.id,
                customer_id=gap.customer_id,
                error=str(e),
                error_type=type(e).__name__,
                attempts=bo["attempts"],
                parked=bo["attempts"] >= GAP_MAX_ATTEMPTS,
            )


def _utc_day() -> str:
    """Current UTC day as YYYY-MM-DD (the daily-budget reset key)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _remaining_daily_budget(scan_state: dict, max_calls_per_day: int) -> int:
    """Remaining SHARED convergence-call budget for the current UTC day.

    Resets scan_state['calls_today'] at the UTC day boundary, then returns
    max_calls_per_day - calls_today. Shared by all three convergence scans
    (contradiction / dedup / gap-discovery): whichever pass runs first on a new
    day does the reset; the others see the same day and read the running total,
    so the daily ceiling bounds TOTAL convergence cost regardless of how many
    generators are enabled. The day-reset lives here (not in any one pass) so
    it runs no matter which generators are switched on.
    """
    today = _utc_day()
    if scan_state.get("day") != today:
        scan_state["day"] = today
        scan_state["calls_today"] = 0
    return max_calls_per_day - scan_state.get("calls_today", 0)


async def _run_topic_seeding(*, store: "MetadataStore") -> None:
    """One idle-cycle topic-seeding pass over every customer.

    No model calls (store signals only); flood-guarded inside the scan, so
    like tier promotion it takes no slice rotation and no daily budget.
    Fail-safe per customer.
    """
    customers = await store.list_customers(limit=1000)
    for c in customers:
        try:
            await run_topic_seeding(store=store, customer_id=c.id)
        except Exception as e:
            logger.warning(
                "cognition_worker.topic_seeding_failed",
                customer_id=c.id,
                error=str(e),
            )


async def _run_tier_promotion(*, store: "MetadataStore") -> None:
    """One idle-cycle tier-promotion pass over every customer.

    Cheap by construction (no model calls; two count queries per crystal,
    capped per customer inside the scan), so unlike the discriminator scans
    it takes no slice rotation and no daily budget. Fail-safe per customer:
    one bad bank never blocks the rest.
    """
    customers = await store.list_customers(limit=1000)
    for c in customers:
        try:
            await run_tier_promotion_scan(store=store, customer_id=c.id)
        except Exception as e:
            logger.warning(
                "cognition_worker.tier_promotion_failed",
                customer_id=c.id,
                error=str(e),
            )


async def _run_outbound_review(*, store: "MetadataStore") -> None:
    """One idle-cycle outbound-review pass over every customer.

    The ratified high-tier reviewer for background-worker memory: walks
    each customer's recall-gated background_worker crystals lacking a
    verdict and stamps outbound_scan_passed / failed via the scan. The
    frontier-tier client comes from the provider-neutral seam; when it
    isn't configured the scan can still FAIL crystals deterministically
    but never stamps a pass. Fail-safe per customer.
    """
    from ..scan.outbound_review import run_outbound_review_scan

    try:
        from ..llm import get_llm_client
        client = get_llm_client()
    except Exception:  # noqa: BLE001 — no provider configured
        client = None

    customers = await store.list_customers(limit=1000)
    for c in customers:
        try:
            await run_outbound_review_scan(
                store, c.id,
                client=client,
                max_crystals=settings.outbound_scan_max_crystals_per_cycle,
            )
        except Exception as e:
            logger.warning(
                "cognition_worker.outbound_review_failed",
                customer_id=c.id,
                error=str(e),
            )


async def _run_system_promotion_rules(*, store: "MetadataStore") -> None:
    """One idle-cycle system-rules promotion pass over every customer.

    Evaluates each customer's enabled 'promotion' rules against their
    recall-gated crystals via system_rules.store.run_promotion_rules,
    clearing gates where the user's conditions hold. Cheap by construction
    (no model calls; short-circuits when a customer has no rules or no
    gated crystals), so like tier promotion it takes no slice rotation and
    no daily budget. Fail-safe per customer: one bad bank never blocks the
    rest.
    """
    from ..system_rules.store import run_promotion_rules

    customers = await store.list_customers(limit=1000)
    for c in customers:
        try:
            await run_promotion_rules(store, c.id)
        except Exception as e:
            logger.warning(
                "cognition_worker.system_rules_failed",
                customer_id=c.id,
                error=str(e),
            )


async def _run_contradiction_scan(
    *,
    store: "MetadataStore",
    scan_state: dict,
    customers_per_cycle: int,
    max_candidate_pairs: int,
    max_calls_per_cycle: int,
    max_calls_per_day: int,
) -> int:
    """One idle-cycle contradiction-scan pass (Never-Idle Convergence Phase 3).

    Surfacing-only. Rotates through customers across cycles (round-robin offset
    in scan_state, so every tenant gets scanned over time rather than only the
    first few) and spends at most max_calls_per_cycle discriminator calls this
    cycle and max_calls_per_day per UTC day. scan_state carries the
    process-local daily counter + the rotation offset (the gap_backoff
    process-local precedent). Returns the number of discriminator calls spent
    this cycle.

    Each customer is scanned via scan_for_contradictions with the REMAINING
    cycle budget, so the per-cycle cap is honored across the customer slice and
    never exceeded.
    """
    # Daily budget (shared across all convergence scans) resets at the UTC
    # day boundary; bail if today's ceiling is already spent.
    remaining_today = _remaining_daily_budget(scan_state, max_calls_per_day)
    if remaining_today <= 0:
        return 0

    customers = await store.list_customers(limit=1000)
    if not customers:
        return 0

    n = len(customers)
    start = scan_state.get("cust_offset", 0) % n
    slice_count = min(customers_per_cycle, n)
    selected = [customers[(start + i) % n] for i in range(slice_count)]
    scan_state["cust_offset"] = (start + slice_count) % n

    cycle_budget = min(max_calls_per_cycle, remaining_today)
    calls_this_cycle = 0
    for c in selected:
        if calls_this_cycle >= cycle_budget:
            break
        result = await scan_for_contradictions(
            store=store,
            customer_id=c.id,
            max_candidate_pairs=max_candidate_pairs,
            max_discriminator_calls=cycle_budget - calls_this_cycle,
        )
        calls_this_cycle += result.pairs_evaluated
        if result.conflicts_found:
            logger.info(
                "cognition_worker.contradictions_surfaced",
                customer_id=c.id,
                conflicts=result.conflicts_found,
                pairs_evaluated=result.pairs_evaluated,
            )
    scan_state["calls_today"] = scan_state.get("calls_today", 0) + calls_this_cycle
    if calls_this_cycle:
        logger.info(
            "cognition_worker.contradiction_scan_cycle",
            customers_scanned=len(selected),
            calls_this_cycle=calls_this_cycle,
            calls_today=scan_state["calls_today"],
        )
    return calls_this_cycle


async def _run_dedup_scan(
    *,
    store: "MetadataStore",
    scan_state: dict,
    customers_per_cycle: int,
    max_candidate_pairs: int,
    max_calls_per_cycle: int,
    max_calls_per_day: int,
) -> int:
    """One idle-cycle dedup-scan pass (Never-Idle Convergence Phase 3, P5).

    Mirrors _run_contradiction_scan exactly — same per-cycle + shared daily
    budget, surfacing-only — but calls scan_for_duplicates (DUPLICATE →
    knowledge_conflict with detector='dedup_scan') and rotates customers on its
    OWN offset ('dedup_offset' in scan_state), so it round-robins independently
    of the contradiction scan. Returns the discriminator calls spent this
    cycle.
    """
    remaining_today = _remaining_daily_budget(scan_state, max_calls_per_day)
    if remaining_today <= 0:
        return 0

    customers = await store.list_customers(limit=1000)
    if not customers:
        return 0

    n = len(customers)
    start = scan_state.get("dedup_offset", 0) % n
    slice_count = min(customers_per_cycle, n)
    selected = [customers[(start + i) % n] for i in range(slice_count)]
    scan_state["dedup_offset"] = (start + slice_count) % n

    cycle_budget = min(max_calls_per_cycle, remaining_today)
    calls_this_cycle = 0
    for c in selected:
        if calls_this_cycle >= cycle_budget:
            break
        result = await scan_for_duplicates(
            store=store,
            customer_id=c.id,
            max_candidate_pairs=max_candidate_pairs,
            max_discriminator_calls=cycle_budget - calls_this_cycle,
        )
        calls_this_cycle += result.pairs_evaluated
        if result.duplicates_found:
            logger.info(
                "cognition_worker.duplicates_surfaced",
                customer_id=c.id,
                duplicates=result.duplicates_found,
                pairs_evaluated=result.pairs_evaluated,
            )
    scan_state["calls_today"] = scan_state.get("calls_today", 0) + calls_this_cycle
    if calls_this_cycle:
        logger.info(
            "cognition_worker.dedup_scan_cycle",
            customers_scanned=len(selected),
            calls_this_cycle=calls_this_cycle,
            calls_today=scan_state["calls_today"],
        )
    return calls_this_cycle


async def _run_gap_discovery(
    *,
    store: "MetadataStore",
    scan_state: dict,
    customers_per_cycle: int,
    max_subjects_per_cycle: int,
    max_calls_per_day: int,
) -> int:
    """One idle-cycle gap-discovery pass (Never-Idle Convergence Phase 3, P5).

    Mirrors _run_contradiction_scan but spends per-SUBJECT model calls (not
    pairwise): each subject the model evaluates costs one call against the
    SHARED daily ceiling. Rotates customers on its own offset ('gap_offset').
    The per-cycle budget is max_subjects_per_cycle, bounded by what's left of
    the shared daily ceiling. Surfacing-only (gaps with source='gap_discovery').
    Returns the model calls spent this cycle.
    """
    remaining_today = _remaining_daily_budget(scan_state, max_calls_per_day)
    if remaining_today <= 0:
        return 0

    customers = await store.list_customers(limit=1000)
    if not customers:
        return 0

    n = len(customers)
    start = scan_state.get("gap_offset", 0) % n
    slice_count = min(customers_per_cycle, n)
    selected = [customers[(start + i) % n] for i in range(slice_count)]
    scan_state["gap_offset"] = (start + slice_count) % n

    cycle_budget = min(max_subjects_per_cycle, remaining_today)
    calls_this_cycle = 0
    for c in selected:
        if calls_this_cycle >= cycle_budget:
            break
        result = await discover_gaps(
            store=store,
            customer_id=c.id,
            max_subjects=cycle_budget - calls_this_cycle,
        )
        calls_this_cycle += result.subjects_evaluated
        if result.gaps_found:
            logger.info(
                "cognition_worker.gaps_discovered",
                customer_id=c.id,
                gaps=result.gaps_found,
                subjects_evaluated=result.subjects_evaluated,
            )
    scan_state["calls_today"] = scan_state.get("calls_today", 0) + calls_this_cycle
    if calls_this_cycle:
        logger.info(
            "cognition_worker.gap_discovery_cycle",
            customers_scanned=len(selected),
            calls_this_cycle=calls_this_cycle,
            calls_today=scan_state["calls_today"],
        )
    return calls_this_cycle
