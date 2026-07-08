"""Metacognition scheduler worker — Phase 10B per P0.76 + P0.80.

Background polling loop that automates the metacognitive layer.
Each cycle does TWO passes:

  Pass 1 (shadow review): scans traces with an agent_self critique
    but no shadow critique yet; for each, calls
    `agent.shadow_critic.shadow_review_trace` (which applies its
    own sampling policy). Respects the per-customer shadow cost
    cap (P0.80).

  Pass 2 (synthesis): scans traces with at least one critique and
    no synthesis row, settled (age > settling_seconds); for each,
    calls
    `metacognition.engine.compute_alignment_and_synthesis_for_trace`.

The two passes share one polling interval. Pass ordering is
intentional: Pass 1 first so newly-attached shadow critiques have
a chance to be visible to Pass 2 in the SAME cycle. Pass 2's
settling guard prevents racing the shadow's LLM call when Pass 1
is in flight.

Env var: `CC_METACOG_WORKER_INTERVAL_SECONDS` (default 300 = 5min).
**Notably distinct** from `CC_METACOGNITION_INTERVAL_SECONDS` which
is misnamed and controls the COGNITION (research-task) worker —
see workers/cognition.py. The misnaming predates MCR's
appropriation of "metacognition"; CU-24 (logged in Phase 10B
close-out) tracks the rename.

Cost cap: `settings.shadow_max_per_customer_per_day` (default 100)
is the GLOBAL default. Phase 12 (CU-27) adds a per-customer
override via `CustomerRow.shadow_max_per_day` (NULL = use global).
The shadow pass resolves each customer's effective cap, counts
shadow critiques created in the last 24h per customer, and skips
when at/over that cap (P0.80 + P0.111).

Following the workers/cognition.py pattern: NEVER raises per cycle,
catches asyncio.CancelledError + general Exception, sleeps via
asyncio.wait_for(shutdown_event.wait(), timeout=interval).
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import structlog

from ..agent.shadow_critic import shadow_review_trace
from ..control.admission import function_budget_allows
from ..metacognition.structural import run_structural_ingestion_scan
from ..config import settings
from ..llm import get_llm_client
from ..metacognition.engine import compute_alignment_and_synthesis_for_trace

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore

logger = structlog.get_logger(__name__)


# Env var name — intentionally distinct from cognition worker's
# CC_METACOGNITION_INTERVAL_SECONDS (which controls research tasks,
# not the metacognitive layer; CU-24 tracks the rename).
_INTERVAL_ENV_VAR = "CC_METACOG_WORKER_INTERVAL_SECONDS"
_DEFAULT_INTERVAL_SECONDS = 300

# Per-cycle batch sizes. Tuned for typical operator workloads;
# operators with heavier traffic may raise these via a settings
# extension in Phase 11+.
_SHADOW_BATCH_SIZE = 10
_SYNTHESIS_BATCH_SIZE = 20

# Pass-2 settling guard: traces younger than this haven't given
# Pass 1's LLM call enough time to complete. 60 seconds is
# conservative; the shadow critique typically takes 5-20 seconds.
_SYNTHESIS_SETTLING_SECONDS = 60


async def run_metacognition_worker(
    *,
    store: "MetadataStore",
    shutdown_event: asyncio.Event,
) -> None:
    """Background poll loop. Reads `CC_METACOG_WORKER_INTERVAL_SECONDS`
    from env (default 300).

    Each cycle runs `_run_one_cycle` which executes Pass 1 (shadow)
    then Pass 2 (synthesis). Per-cycle errors are caught and logged;
    the loop continues.

    Pass 1 routes through the provider-neutral seam and no-ops when no
    provider is configured. Pass 2 only needs `store` and runs
    regardless.
    """
    poll_interval = int(
        os.environ.get(_INTERVAL_ENV_VAR, str(_DEFAULT_INTERVAL_SECONDS))
    )
    logger.info(
        "metacog_worker.started",
        poll_interval=poll_interval,
        provider_ready=get_llm_client().is_ready(),
    )

    while not shutdown_event.is_set():
        try:
            await _run_one_cycle(
                store=store,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(
                "metacog_worker.cycle_error",
                error=str(e),
                error_type=type(e).__name__,
            )

        try:
            await asyncio.wait_for(
                shutdown_event.wait(),
                timeout=poll_interval,
            )
            break  # shutdown triggered
        except asyncio.TimeoutError:
            pass  # interval elapsed; next cycle

    logger.info("metacog_worker.stopped")


async def _run_one_cycle(
    *,
    store: "MetadataStore",
    shadow_max_per_day: int | None = None,
    settling_seconds: int = _SYNTHESIS_SETTLING_SECONDS,
) -> dict[str, int]:
    """Execute one full cycle: shadow pass then synthesis pass.

    Returns aggregate counts. Test-friendly: parameters override env
    defaults so tests can pass a low cap and zero settling to exercise
    the worker without time manipulation.
    """
    cap = (
        shadow_max_per_day
        if shadow_max_per_day is not None
        else _resolve_shadow_cap()
    )

    shadow_result = await _shadow_pass(
        store=store,
        shadow_max_per_day=cap,
    )
    synth_result = await _synthesis_pass(
        store=store,
        settling_seconds=settling_seconds,
    )

    # Pass 3 (S6, 2026-07-08): structural critics — store-signal scans
    # over the bank's ARTIFACTS (no model calls, no budget needed).
    # First critic: the blob-fact ingestion detector. Never raises.
    structural_result = await run_structural_ingestion_scan(store=store)

    out = {
        "shadow_shadowed": shadow_result["shadowed"],
        "shadow_skipped_cost_cap": shadow_result["skipped_cost_cap"],
        "shadow_skipped_other": shadow_result["skipped_other"],
        "structural_found": structural_result["found"],
        "structural_filed": structural_result["filed"],
        "synthesis_synthesized": synth_result["synthesized"],
        "synthesis_skipped": synth_result["skipped"],
    }
    logger.info("metacog_worker.cycle_complete", **out)
    return out


def _resolve_shadow_cap() -> int:
    """Resolve the per-customer shadow cost cap from settings.

    Defaults to 100 if the setting is not present (which happens in
    test environments where Settings is constructed without the new
    field). See P0.80.
    """
    return getattr(settings, "shadow_max_per_customer_per_day", 100)


async def _shadow_pass(
    *,
    store: "MetadataStore",
    shadow_max_per_day: int,
) -> dict[str, int]:
    """Pass 1: find un-shadowed traces, run shadow_review_trace on each.

    Per-customer cost cap (P0.80 + CU-27/P0.111): counts shadow
    critiques in the last 24 hours per customer; skips when the count
    is at/over that customer's EFFECTIVE cap. The effective cap is the
    customer's `shadow_max_per_day` override when set, else the
    resolver-injected `shadow_max_per_day` global default. This lets
    the operator tune metacognition (R&D) spend per customer tier
    without affecting customers who have no override.
    """
    out = {"shadowed": 0, "skipped_cost_cap": 0, "skipped_other": 0}

    # The shadow critique itself routes through the provider-neutral seam
    # (shadow_review_trace -> run rationale via get_llm_client); no-op when
    # no provider is configured.
    if not get_llm_client().is_ready():
        return out

    traces = await store.list_traces_needing_shadow_review(
        limit=_SHADOW_BATCH_SIZE,
    )

    if not traces:
        return out

    # Per-customer 24h-window counter cache: avoid re-querying the
    # critique count for every trace from the same customer.
    counter_cache: dict[str, int] = {}
    budget_gate_cache: dict[str, bool] = {}
    # Per-customer effective-cap cache (CU-27 / P0.111): resolve each
    # customer's cap once. The effective cap is the customer's
    # shadow_max_per_day override when set, else the injected default.
    cap_cache: dict[str, int] = {}
    since = datetime.now(timezone.utc) - timedelta(hours=24)

    for trace in traces:
        # Resolve this customer's effective cap (per-customer override
        # or the global/injected default). Cached per customer.
        if trace.customer_id not in cap_cache:
            try:
                override = await store.get_customer_shadow_cap_override(
                    trace.customer_id
                )
            except Exception as e:
                logger.warning(
                    "metacog_worker.cap_resolve_failed",
                    customer_id=trace.customer_id,
                    error=str(e),
                )
                override = None
            cap_cache[trace.customer_id] = (
                override if override is not None else shadow_max_per_day
            )
        effective_cap = cap_cache[trace.customer_id]

        # Cost cap check.
        if trace.customer_id not in counter_cache:
            try:
                recent_shadows = await store.list_critiques_by_role(
                    customer_id=trace.customer_id,
                    critic_role="shadow",
                    since=since,
                )
                counter_cache[trace.customer_id] = len(recent_shadows)
            except Exception as e:
                logger.warning(
                    "metacog_worker.cost_check_failed",
                    customer_id=trace.customer_id,
                    error=str(e),
                )
                counter_cache[trace.customer_id] = 0

        if counter_cache[trace.customer_id] >= effective_cap:
            out["skipped_cost_cap"] += 1
            logger.info(
                "metacog_worker.shadow_cost_cap_hit",
                customer_id=trace.customer_id,
                recent_shadows=counter_cache[trace.customer_id],
                cap=effective_cap,
            )
            continue

        # S6 (2026-07-08): the DOLLAR cap — a spend_budgets row for
        # 'shadow_critic', when present, hard-caps ledger spend (the
        # shadow stamps origin='shadow_critic' as of S6). No row = the
        # count cap above remains the only governor (live behavior
        # preserved; MCR §11 Q10 closed for tenants who set a budget).
        if trace.customer_id not in budget_gate_cache:
            allowed = True
            try:
                customer = await store.get_customer_by_id(trace.customer_id)
                budget = (
                    await store.get_spend_budget(
                        trace.customer_id, function="shadow_critic"
                    )
                    if customer is not None
                    else None
                )
                if customer is not None and budget is not None:
                    allowed = await function_budget_allows(
                        store, customer, "shadow_critic",
                        origin="shadow_critic",
                    )
            except Exception as e:
                logger.warning(
                    "metacog_worker.budget_check_failed",
                    customer_id=trace.customer_id, error=str(e),
                )
            budget_gate_cache[trace.customer_id] = allowed
        if not budget_gate_cache[trace.customer_id]:
            out["skipped_cost_cap"] += 1
            logger.info(
                "metacog_worker.shadow_budget_cap_hit",
                customer_id=trace.customer_id,
            )
            continue

        # Run the shadow review. shadow_review_trace handles its own
        # sampling policy — may decline; either outcome counts here.
        try:
            result = await shadow_review_trace(
                store=store,
                trace_id=trace.id,
            )
        except Exception as e:
            logger.warning(
                "metacog_worker.shadow_call_failed",
                trace_id=trace.id,
                error=str(e),
                error_type=type(e).__name__,
            )
            out["skipped_other"] += 1
            continue

        if result.get("reason") == "shadowed":
            out["shadowed"] += 1
            # Bump the cache so further iterations on this customer
            # observe the increment.
            counter_cache[trace.customer_id] += 1
        else:
            out["skipped_other"] += 1

    return out


async def _synthesis_pass(
    *,
    store: "MetadataStore",
    settling_seconds: int,
) -> dict[str, int]:
    """Pass 2: find un-synthesized traces, run
    compute_alignment_and_synthesis_for_trace on each.

    The synthesis engine is idempotent at the item level (P0.74) so
    racing Pass 1's shadow attachment is safe — a future cycle re-
    synthesizes any newly-pending items.
    """
    out = {"synthesized": 0, "skipped": 0}

    traces = await store.list_traces_needing_synthesis(
        limit=_SYNTHESIS_BATCH_SIZE,
        settling_seconds=settling_seconds,
    )

    if not traces:
        return out

    for trace in traces:
        try:
            result = await compute_alignment_and_synthesis_for_trace(
                store=store,
                trace_id=trace.id,
            )
        except Exception as e:
            logger.warning(
                "metacog_worker.synthesis_call_failed",
                trace_id=trace.id,
                error=str(e),
                error_type=type(e).__name__,
            )
            out["skipped"] += 1
            continue

        if result.get("synthesis_id"):
            out["synthesized"] += 1
        else:
            out["skipped"] += 1

    return out
