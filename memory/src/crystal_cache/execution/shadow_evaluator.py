"""ShadowEvaluator — §4 of BUILD_PROPOSAL.md (RESEARCH-GROUNDED).

On a sampled fraction of MEDIUM-match requests, run a BASELINE (no-injection)
pass in parallel and compare outputs. Capture shadow_delta in QueryLog.

WHY
---
Without shadow evaluation, we never know whether injection actually helped
on live traffic. We only know whether the user re-queried (implicit
negative signal, unreliable). Shadow provides a ground-truth label for
every sampled event:

    shadow_delta > 0  → injection produced a meaningfully different response;
                         treat as "helped" unless hurt metric proves otherwise
    shadow_delta = 0  → injection had negligible effect; neutral
    shadow_delta < 0  → baseline was shorter / cleaner; treat as "hurt"

The sign convention reads "positive = injection made a difference worth
noting." The magnitude is implementation-dependent — string-match,
length-delta, semantic-similarity, task-specific judges. We keep the
metric pluggable.

SAMPLING POLICY
---------------
For a given request:
  - If customer.shadow_sample_rate (in [0, 1]) fires, AND
  - The match was "medium" (high-match is trusted, low-match is pass-through),
  - Then run shadow.

Plus a mandatory rule for quarantined crystals: every injection from a
quarantined-tier crystal gets a shadow, until we have enough labels to
promote or blacklist.

COST
----
Shadow doubles compute for sampled requests. Typical customer config:
5-10% sample rate. A quarantined crystal gets 100% for its first N
matches. Anthropic's recommendation in the proposal is to bound this
by a per-customer budget so a single high-volume tenant can't spike
billing.

DEFAULT METRIC (placeholder)
----------------------------
`length_delta_metric` computes
    (len(baseline) - len(injected)) / max(len(baseline), len(injected), 1)

which is in [-1, 1]. Length alone is a weak proxy for quality; it captures
"injection made the model terser" as a proxy for "injection helped focus
the answer". Operators should replace this with a domain-specific metric:

  - Exact-answer domains (math, code): answer extraction + string match
  - Reasoning: semantic similarity via a judge model
  - Code: AST diff

The metric function's signature is:
    metric_fn(injected_text: str, baseline_text: str) -> float in [-1, 1]

You swap by passing `metric_fn=my_metric` to `ShadowEvaluator(...)`.
"""
from __future__ import annotations

import asyncio
import random
from typing import Any, Awaitable, Callable, Optional

import structlog

from ..models import Customer
from .upstream_client import UpstreamClient, UpstreamResponse


logger = structlog.get_logger(__name__)


# Signed float in [-1, 1]. Pluggable — see module docstring.
MetricFn = Callable[[str, str], float]


def length_delta_metric(injected: str, baseline: str) -> float:
    """Baseline-minus-injected length, normalized.

    Positive → injection made the answer shorter (proxy for "focus helped").
    Negative → injection made the answer longer.
    Zero    → same length or both empty.

    This is a weak proxy. It's used as the default because it's
    domain-agnostic and cheap. Replace with task-specific metrics for
    real evaluations.
    """
    li = len(injected or "")
    lb = len(baseline or "")
    denom = max(li, lb, 1)
    return (lb - li) / denom


class ShadowEvaluator:
    """Decides whether to run shadow, runs it, and computes delta.

    Usage in the pipeline (simplified):

        if evaluator.should_shadow(customer, match_type, quality_tier):
            shadow_resp = await evaluator.run_shadow(
                client, original_messages, model, temperature, max_tokens
            )
            delta = evaluator.compute_delta(injected_text, shadow_resp.assistant_text)
            # write shadow_ran=True, shadow_delta=delta to QueryLog

    The evaluator does NOT write to QueryLog itself. Separating the
    compute from the persistence keeps this class focused and lets the
    caller batch multiple telemetry writes.
    """

    def __init__(
        self,
        *,
        metric_fn: Optional[MetricFn] = None,
        quarantine_always_shadow: bool = True,
        random_seed: Optional[int] = None,
    ) -> None:
        self._metric_fn: MetricFn = metric_fn or length_delta_metric
        self._quarantine_always_shadow = quarantine_always_shadow
        # A private Random lets tests inject determinism without
        # polluting the global seed.
        self._rng = random.Random(random_seed)

    # -----------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------

    def should_shadow(
        self,
        customer: Customer,
        match_type: str,
        crystal_quality_tier: Optional[str] = None,
    ) -> bool:
        """Return True if this request should get a shadow evaluation.

        Rules (first-match wins):
          1. match_type not in ("medium", "high"): never shadow. Low and
             none didn't inject, so there's nothing to compare.
          2. crystal_quality_tier == "quarantine" AND quarantine_always_shadow:
             always shadow. Quarantined crystals need evidence fast.
          3. Otherwise: Bernoulli trial at customer.shadow_sample_rate.
        """
        if match_type not in ("medium", "high"):
            return False
        if (
            self._quarantine_always_shadow
            and crystal_quality_tier == "quarantine"
        ):
            return True
        rate = customer.shadow_sample_rate
        if rate <= 0:
            return False
        if rate >= 1:
            return True
        return self._rng.random() < rate

    # -----------------------------------------------------------------
    # Running the baseline
    # -----------------------------------------------------------------

    async def run_shadow(
        self,
        client: UpstreamClient,
        original_messages: list[dict[str, Any]],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **extra: Any,
    ) -> Optional[UpstreamResponse]:
        """Call upstream WITHOUT any crystal injection. Returns None on failure.

        Swallows exceptions rather than propagating — shadow failure
        must never break the user's primary request. Logs the failure
        so operators can see shadow-call error rates in the dashboard.
        """
        try:
            return await client.complete(
                messages=original_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra,
            )
        except Exception as e:
            logger.warning(
                "shadow.upstream_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

    # -----------------------------------------------------------------
    # Delta computation
    # -----------------------------------------------------------------

    def compute_delta(
        self, injected_response: str, baseline_response: str
    ) -> float:
        """Compute the signed metric comparing injected vs baseline."""
        try:
            return float(self._metric_fn(injected_response, baseline_response))
        except Exception as e:
            # A broken metric shouldn't poison every request's QueryLog.
            logger.warning(
                "shadow.metric_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            return 0.0


async def run_shadow_in_parallel(
    evaluator: ShadowEvaluator,
    primary_coro: Awaitable[UpstreamResponse],
    shadow_coro_factory: Callable[[], Awaitable[Optional[UpstreamResponse]]],
) -> tuple[UpstreamResponse, Optional[UpstreamResponse]]:
    """Run the primary injection call and the baseline shadow call concurrently.

    The primary call's result is returned as-is. The shadow call's
    result is returned as None on failure so the caller can skip
    writing shadow_delta rather than fabricating one.

    Why a factory for the shadow: we want to construct the shadow
    coroutine lazily (only after we've decided to shadow) but also
    gather it concurrently with the primary. Accepting a factory lets
    the caller defer construction until inside asyncio.gather.
    """
    # Pre-create the shadow coroutine so both run concurrently.
    shadow_coro = shadow_coro_factory()
    primary, shadow = await asyncio.gather(
        primary_coro, shadow_coro, return_exceptions=False,
    )
    return primary, shadow
