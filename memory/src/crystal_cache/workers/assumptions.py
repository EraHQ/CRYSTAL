"""Assumptions worker — the standalone role for bridging inference.

Slice 2 of the Assumptions build (RQ1=B, ratified 2026-07-20 + refresh
2026-07-31): assumptions run as their OWN worker role ('assumptions'
in CC_WORKER_ROLES), not another pass inside the cognition worker's
idle phase — the queue-latency ledger item (45-minute pending→running
behind minutes of idle small-model calls) is exactly the failure mode
piggybacking would inherit. Deploy shape: CC_WORKER_ROLES +=
assumptions on crystal-worker only; the role gate is the enable
switch (no separate settings flag, matching the metacognition
worker's posture).

Each cycle scans a ROTATING slice of customers
(settings.assumptions_customers_per_cycle, the convergence fairness
pattern) via scan.assumptions.run_assumptions_scan, which reads its
own pairs/gaps/threshold knobs from settings. Gates, in order:

  1. Daily background budget (workers/budget.llm_budget_exhausted) —
     assumptions are the definition of background spend.
  2. Load-aware idle gate (workers/idle.is_quiet) — a deployment
     actively serving /v1/* traffic is not idle (Core Principle #1).
     Inert on the split-process worker service (per-process stamp,
     documented in idle.py); protective when this loop runs in the
     API lifespan.
  3. Seam readiness (checked once per cycle in _run_one_cycle to
     avoid N pointless store reads; the scan re-checks per call).

Following the workers/metacognition.py pattern: NEVER raises per
cycle, catches asyncio.CancelledError + general Exception, sleeps via
asyncio.wait_for(shutdown_event.wait(), timeout=interval).

Env var: `CC_ASSUMPTIONS_WORKER_INTERVAL_SECONDS` (default 600,
matching the cognition worker's cadence — assumptions are idle
curation, not latency-sensitive).
"""
from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any, Optional

import structlog

from ..config import settings
from ..llm import get_llm_client
from ..scan.assumptions import run_assumptions_scan
from ..scan.pairing_funnel import FunnelState, run_pairing_funnel

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore

logger = structlog.get_logger(__name__)

_INTERVAL_ENV_VAR = "CC_ASSUMPTIONS_WORKER_INTERVAL_SECONDS"
_DEFAULT_INTERVAL_SECONDS = 600


async def run_assumptions_worker(
    *,
    store: "MetadataStore",
    encoder: Any,
    shutdown_event: asyncio.Event,
) -> None:
    """Background poll loop. Reads `CC_ASSUMPTIONS_WORKER_INTERVAL_SECONDS`
    from env (default 600).

    `encoder` is required by the write path (assumption vectors ride
    the serialized encoder lane) — both wiring sites pass the shared
    runtime encoder (app.state.prompt_encoder / core.encoder). Per-
    cycle errors are caught and logged; the loop continues.
    """
    poll_interval = int(
        os.environ.get(_INTERVAL_ENV_VAR, str(_DEFAULT_INTERVAL_SECONDS))
    )
    logger.info(
        "assumptions_worker.started",
        poll_interval=poll_interval,
        provider_ready=get_llm_client().is_ready(),
    )

    # Process-local round-robin offset (the scan_state posture from the
    # convergence scans: a restart resets rotation, acceptable for v1).
    rotation_state: dict = {"cust_offset": 0}

    while not shutdown_event.is_set():
        try:
            # Slice 4 (Q3=B): parent-death sweep for out-of-band
            # deletions — store-only, no model spend, so it runs
            # BEFORE the budget/idle gates (the reclaim posture:
            # integrity work must not wait on a spend cap).
            swept = await store.sweep_orphaned_assumptions(limit=50)
            if swept:
                logger.info(
                    "assumptions_worker.sweep_invalidated", count=swept
                )

            from .budget import llm_budget_exhausted
            from .idle import is_quiet
            if (
                not await llm_budget_exhausted(store)
                and is_quiet(settings.idle_quiet_seconds)
            ):
                await _run_one_cycle(
                    store=store,
                    encoder=encoder,
                    state=rotation_state,
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(
                "assumptions_worker.cycle_error",
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

    logger.info("assumptions_worker.stopped")


async def _run_one_cycle(
    *,
    store: "MetadataStore",
    encoder: Any,
    state: Optional[dict] = None,
    customers_per_cycle: Optional[int] = None,
    slm_client: Any = None,
) -> dict[str, int]:
    """Scan one rotating slice of customers; return aggregate counts.

    Test-friendly: `slm_client` threads a fake straight into the scan
    (None -> the provider-neutral seam), `customers_per_cycle`
    overrides the settings knob, and `state` carries the round-robin
    offset so tests can assert rotation across cycles without a loop.
    """
    state = state if state is not None else {}
    per_cycle = (
        settings.assumptions_customers_per_cycle
        if customers_per_cycle is None else customers_per_cycle
    )

    out = {
        "customers_scanned": 0,
        "pairs_evaluated": 0,
        "assumptions_written": 0,
        "skipped_existing": 0,
    }

    # One readiness check per cycle (the _shadow_pass posture): with no
    # provider and no injected client there is nothing to spend, so
    # skip the customer enumeration entirely.
    if slm_client is None and not get_llm_client().is_ready():
        return out

    customers = await store.list_customers(limit=1000)
    n = len(customers)
    if n == 0:
        return out

    k = min(max(per_cycle, 0), n)
    if k == 0:
        return out
    offset = int(state.get("cust_offset", 0)) % n
    cycle_slice = [customers[(offset + i) % n] for i in range(k)]
    state["cust_offset"] = (offset + k) % n

    funnel_states: dict = state.setdefault("funnel", {})
    for customer in cycle_slice:
        # Funnel F1 (Q6=A): score/reinforce the customer's crystal_edges
        # from every recorded usage signal BEFORE spending verdicts —
        # free (store reads + edge writes, no model calls). F2 rewires
        # the scan to read these edges; until then the scan's own
        # pairing still drives the spend and the graph accumulates.
        funnel_state = funnel_states.setdefault(
            customer.id, FunnelState()
        )
        try:
            await run_pairing_funnel(
                store=store,
                customer_id=customer.id,
                state=funnel_state,
            )
        except Exception as e:  # fail-safe: the funnel never costs a scan
            logger.warning(
                "assumptions_worker.funnel_failed",
                customer_id=customer.id,
                error=str(e),
                error_type=type(e).__name__,
            )
        result = await run_assumptions_scan(
            store=store,
            slm_client=slm_client,
            customer_id=customer.id,
            encoder=encoder,
        )
        out["customers_scanned"] += 1
        out["pairs_evaluated"] += result.pairs_evaluated
        out["assumptions_written"] += result.assumptions_written
        out["skipped_existing"] += result.skipped_existing

    # Sibling log discipline: found-nothing cycles are debug.
    _log_fn = logger.info if out["assumptions_written"] else logger.debug
    _log_fn("assumptions_worker.cycle_complete", **out)
    return out
