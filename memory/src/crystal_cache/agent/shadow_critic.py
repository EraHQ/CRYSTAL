"""MCR shadow critic — Phase 9.5 (2026-05-27).

The shadow critic is the SECOND MCR critic (MCR_FRAMEWORK.md §5.2,
D-MCR-10). Phase 9A landed the agent self-critique (critic_role=
"agent_self"); Phase 9.5 lands the shadow (critic_role="shadow").

WHAT MAKES THE SHADOW DIFFERENT FROM THE SELF-CRITIQUE
------------------------------------------------------
1. It runs SAMPLED, not always-on (D-MCR-10). The sampling policy is
   `ShadowSamplingPolicy` below: always-sample when the self-critique
   flagged >=1 observation, else Bernoulli at
   `settings.shadow_critic_sample_rate`.

2. It runs on a FRONTIER model (§5.2) — `settings.shadow_critic_model`,
   default Opus — not the Haiku tier the self-critique uses. A
   same-tier second opinion adds little; the value is a stronger model
   reviewing a weaker model's reasoning AND the weaker model's
   self-assessment.

3. It reviews the trace AND the agent's self-critique (§5.2: "the
   shadow critiques the self-critique too, not just the reasoning").
   The shadow can flag where the self-critique missed its own border
   crossings, or where the self-critique over-flagged.

4. It runs against PERSISTED artifacts, not live request state
   (P0.63). `shadow_review_trace(trace_id)` loads the trace + its
   agent_self critique(s) from the store. This means:
   - No frontier-model latency on the live request path.
   - The shadow critique's `trace_id` is a HARD FK pointer (the trace
     already exists when the shadow runs), resolving CU-19 for the
     shadow path — no `update_critique_trace_id` upgrade method is
     needed.

WHAT THIS MODULE IS *NOT*
-------------------------
- NOT `execution/shadow_evaluator.py`. That module is v1's
  response-quality comparator (no-injection baseline + length-delta
  metric written to QueryLog.shadow_delta). It is a DIFFERENT,
  orthogonal signal still wired into chat_proxy. Phase 9.5 leaves it
  untouched (P0.62). The MCR_FRAMEWORK.md §8.1 language about the
  shadow critic "replacing" shadow_evaluator predates the Phase 9C
  wiring of shadow_delta into the proxy; replacing it wholesale would
  break that telemetry for no benefit. The two coexist: response-
  quality delta and reasoning-trace critique are independent.

- NOT the automatic background scheduler. Phase 9.5 ships the critic
  itself + sampling + a manual/triggered entry point
  (`shadow_review_trace`). The scheduled/idle-triggered scan that
  finds un-shadowed traces and runs the shadow on a cadence is
  Phase 10's job (MCR §11 Q5 — review cadence is a metacognitive-layer
  concern).

REUSE OF mcr_emitter HELPERS (P0.67)
------------------------------------
The shadow's OUTPUT contract is IDENTICAL to the self-critique's:
observations + action_items + summary_text JSON in the P0.40
vocabulary. So this module imports the parser
(`_parse_self_critique_response`), the confidence coercer
(`_coerce_confidence`), and the vocabulary tuples
(`_OBSERVATION_TYPES`, `_ACTION_TYPES`) from `mcr_emitter`. Sharing
the parser GUARANTEES both critics speak the same vocabulary, which
is what makes Phase 10's item-alignment (self vs. shadow) possible.

Only the PROMPT and the user-message builder differ, because the
shadow's review task differs (review trace + self-critique vs.
review raw reasoning).

FAILURE MODES (inherit mcr_emitter's NEVER-raises discipline, P0.44)
--------------------------------------------------------------------
- Trace not found → log warning, return with shadow_critique_id=None.
- Self-critique LLM call failure → persist a shadow critique with
  empty observations + a failure summary for forensic review.
- Parse failure → persist critique with raw response as summary.
- Critique persistence failure → log warning, return None id.
- Action-item persistence failure (per-item) → log warning, skip
  that item, continue.
"""
from __future__ import annotations

import asyncio
import json
import random
from typing import TYPE_CHECKING, Any, Optional

import structlog

from ..config import settings
from ..cost.emit import record_model_call
from ..llm import get_llm_client
from .mcr_emitter import (
    _ACTION_TYPES,  # noqa: F401  (re-exported vocabulary, used by prompt)
    _OBSERVATION_TYPES,  # noqa: F401
    _coerce_confidence,  # noqa: F401  (kept available for symmetry/tests)
    _parse_self_critique_response,
)

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore
    from ..models import Critique, ReasoningTrace

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Sampling policy (P0.65, MCR §11 Q3)
# ---------------------------------------------------------------------------

class ShadowSamplingPolicy:
    """Decides whether a given trace gets a shadow review.

    Rules (first-match wins):
      1. The agent_self critique flagged >=1 observation → ALWAYS
         shadow. The agent self-flagged uncertainty; a second opinion
         is most valuable exactly here.
      2. Otherwise → Bernoulli trial at `sample_rate`. A clean
         self-critique gets a shadow only on the random sample, to
         catch cases where the agent's introspection MISSED something.

    Cost budgeting (§11 Q10) is a SOFT target here: the rate bounds
    expected spend, but no hard per-customer-per-window cap is
    enforced. The hard cap lands with Phase 10's scheduler.
    """

    def __init__(
        self,
        *,
        sample_rate: Optional[float] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        # Resolve the rate at construction: explicit arg > settings.
        self._sample_rate = (
            sample_rate
            if sample_rate is not None
            else settings.shadow_critic_sample_rate
        )
        # Private RNG so tests inject determinism without polluting the
        # global seed (same pattern as ShadowEvaluator).
        self._rng = random.Random(random_seed)

    def should_shadow_trace(
        self,
        self_critique_observations: list[dict[str, Any]],
    ) -> bool:
        """Return True if this trace should get a shadow review.

        Args:
            self_critique_observations: the observations list from the
                agent_self critique for this trace. A non-empty list
                means the agent self-flagged something → always shadow.

        Returns:
            True to run the shadow review, False to skip.
        """
        # Rule 1: agent self-flagged → always shadow.
        if self_critique_observations:
            return True
        # Rule 2: clean self-critique → Bernoulli sample.
        rate = self._sample_rate
        if rate <= 0:
            return False
        if rate >= 1:
            return True
        return self._rng.random() < rate


# ---------------------------------------------------------------------------
# Shadow critique prompt (P0.64, P0.67)
# ---------------------------------------------------------------------------

_SHADOW_CRITIQUE_SYSTEM_PROMPT = """\
You are a SHADOW critic for an AI assistant called Crystal Cache. You are \
a second, independent reviewer running on a more capable model than the \
assistant. Your job is to review BOTH:

  1. The assistant's reasoning in a single response (which crystals it \
used, which tools it called, what it inferred, where it crossed from \
evidence to guesswork), AND

  2. The assistant's OWN self-critique of that reasoning. The assistant \
already reviewed itself; you are reviewing the reasoning AND checking \
whether the self-critique was complete and honest. Did the self-critique \
miss border crossings the assistant made? Did it over-flag things that \
were actually fine? Did it paper over a gap that you can see?

You are NOT grading the answer's correctness. You are grading the \
reasoning process and the quality of the self-assessment. You have no \
authority over the self-critique — when you disagree with it, that \
disagreement is itself a useful signal, not a verdict. Be specific about \
WHERE the reasoning or the self-critique was weak.

Output STRICTLY valid JSON in this exact shape:
{
  "observations": [
    {
      "type": "<one of the allowed observation types>",
      "text": "<one-sentence description>",
      "confidence": <float 0.0-1.0>,
      "anchors": []
    },
    ...
  ],
  "action_items": [
    {
      "action_type": "<one of the allowed action types>",
      "content": {<free-form dict with action-specific keys>},
      "critic_confidence": <float 0.0-1.0>
    },
    ...
  ],
  "summary_text": "<one-paragraph overall assessment, including whether \
you agreed with the self-critique>"
}

ALLOWED observation types (use EXACTLY these strings):
  assumption_identified        — agent assumed something not in evidence
  generalization_from_thin_evidence — agent generalized from too little data
  source_contradiction         — agent's claim contradicts a source it consulted
  tool_output_questionable     — agent treated a tool output as reliable when it wasn't
  gap_papered_over             — agent (or its self-critique) talked around a gap
  border_crossing_unflagged    — agent crossed from evidence to inference without flagging
  reasoning_skip               — agent skipped a load-bearing reasoning step
  substrate_complaint          — the SYSTEM made good work hard: retrieval quality, a
                                 tool lacking a capability you wished for, ingestion
                                 artifacts, prompt guidance, or metacognition that
                                 should have caught something and did not. Anything
                                 in the surrounding system that affected the outcome
                                 is fair game — name the subsystem.

ALLOWED action_item types (use EXACTLY these strings):
  research_task          — content: {topic, scope, why_needed}
  verification_task      — content: {crystal_id, claim_to_verify}
  evidence_gathering     — content: {topic, suggested_sources}
  gap_declaration        — content: {want, why_needed, domain, subject}
  edit_proposal          — content: {crystal_id, proposed_change, rationale}
  substrate_observation  — content: {subsystem, complaint, severity}
  escalation             — content: {issue, suggested_handler}

If both the reasoning AND the self-critique were sound, return empty \
observations and action_items with a brief summary saying so and noting \
your agreement with the self-critique. NEVER invent observations to seem \
thorough. An empty shadow critique that agrees with a clean self-critique \
is a valid and useful outcome.

Output ONLY the JSON object. No prose before or after.\
"""


def _build_shadow_review_message(
    *,
    user_query: str,
    agent_final_text: str,
    tool_calls_log: list[dict[str, Any]],
    crystals_used: list[str],
    self_critique_observations: list[dict[str, Any]],
    self_critique_summary: Optional[str],
) -> str:
    """Render the user-message body for the shadow critique LLM call.

    Per P0.64, the payload carries the trace data AND the agent_self
    critique so the shadow can review both. Long tool outputs are
    trimmed the same way the self-critique builder trims them.
    """
    def _trim(obj: Any, limit: int = 600) -> Any:
        if isinstance(obj, str) and len(obj) > limit:
            return obj[:limit] + "...(truncated)"
        if isinstance(obj, dict):
            return {k: _trim(v, limit) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_trim(x, limit) for x in obj[:5]]
        return obj

    payload = {
        "user_query": user_query[:2000],
        "agent_response": agent_final_text[:2000],
        "tool_calls": [
            {
                "tool_name": tc.get("tool_name"),
                "input": _trim(tc.get("input")),
                "output_summary": _trim(tc.get("output")),
                "is_error": tc.get("is_error", False),
            }
            for tc in tool_calls_log
        ],
        "crystals_used": crystals_used,
        "agent_self_critique": {
            "observations": self_critique_observations,
            "summary_text": self_critique_summary or "",
        },
    }
    return (
        "Review BOTH the reasoning AND the assistant's self-critique in "
        "this Crystal Cache interaction. The agent's self-critique is "
        "included under `agent_self_critique` — assess whether it was "
        "complete and honest:\n\n"
        + json.dumps(payload, default=str, indent=2)
    )


# ---------------------------------------------------------------------------
# Shadow critique LLM call (P0.66, P0.67)
# ---------------------------------------------------------------------------

async def run_shadow_critique(
    *,
    anthropic_client: Any = None,
    user_query: str,
    agent_final_text: str,
    tool_calls_log: list[dict[str, Any]],
    crystals_used: list[str],
    self_critique_observations: list[dict[str, Any]],
    self_critique_summary: Optional[str],
    model: Optional[str] = None,
    customer_id: Optional[str] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Run the shadow critique LLM call on a frontier model.

    Returns: (observations, action_items, summary_text). Empty lists +
    empty summary represent "reasoning and self-critique both sound."

    On any failure (network, parse, etc.), returns empty observations
    + empty action_items and surfaces the failure in summary_text.
    NEVER raises (inherits mcr_emitter's P0.44 discipline) — a shadow
    failure must not break anything; the shadow is an after-the-fact
    background review.

    The response is parsed by the SAME `_parse_self_critique_response`
    the self-critique uses (P0.67), so the vocabulary is guaranteed
    identical across critics.
    """
    chosen_model = model or settings.shadow_critic_model

    user_message = _build_shadow_review_message(
        user_query=user_query,
        agent_final_text=agent_final_text,
        tool_calls_log=tool_calls_log,
        crystals_used=crystals_used,
        self_critique_observations=self_critique_observations,
        self_critique_summary=self_critique_summary,
    )

    def _call():
        client = anthropic_client if anthropic_client is not None else get_llm_client()
        kwargs = dict(
            system=_SHADOW_CRITIQUE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=2048,
            temperature=0.0,
            tier="frontier",
            model=model,
        )
        # Prefer the usage-bearing variant (S6: the ledger stamp needs
        # token counts); fall back for clients that only expose
        # complete() — injected fakes, thin provider shims.
        fn = getattr(client, "complete_with_usage", None)
        if fn is not None:
            return fn(**kwargs)
        return client.complete(**kwargs)

    try:
        _result = await asyncio.to_thread(_call)
        raw_text = _result if isinstance(_result, str) else _result.text
        # S6 (2026-07-08): the shadow critic stamps the ledger like every
        # other spender — origin='shadow_critic' is the meter the
        # spend_budgets dollar cap reads (closes MCR §11 Q10).
        if customer_id and not isinstance(_result, str):
            await record_model_call(
                customer_id=customer_id,
                model=getattr(_result, "model", None) or chosen_model,
                origin="shadow_critic",
                input_tokens=getattr(_result, "input_tokens", None),
                output_tokens=getattr(_result, "output_tokens", None),
                cache_read_tokens=getattr(_result, "cache_read_tokens", None),
                cache_creation_tokens=getattr(_result, "cache_creation_tokens", None),
            )
    except Exception as e:
        logger.warning(
            "shadow_critic.call_failed",
            error=str(e),
            error_type=type(e).__name__,
            model=chosen_model,
        )
        return (
            [],
            [],
            f"shadow critique call failed: {type(e).__name__}: {e}",
        )

    if not raw_text.strip():
        logger.warning("shadow_critic.empty_response")
        return ([], [], "shadow critique returned empty text")

    return _parse_self_critique_response(raw_text)


# ---------------------------------------------------------------------------
# Trace → review payload reconstruction (P0.63, P0.64)
# ---------------------------------------------------------------------------

def _extract_final_text_from_trace(trace: "ReasoningTrace") -> str:
    """Pull the agent's final response text from the trace events.

    The mcr_emitter appends a trailing event
        {type: "final_text", text: ..., stop_reason: ...}
    to trace.events (see build_trace_from_agent_result). We read it
    back. Returns "" if no such event exists.
    """
    for event in reversed(trace.events or []):
        if isinstance(event, dict) and event.get("type") == "final_text":
            txt = event.get("text", "")
            return txt if isinstance(txt, str) else ""
    return ""


def _extract_user_query_from_trace(trace: "ReasoningTrace") -> str:
    """Best-effort user-query reconstruction from the trace.

    Phase 9.5 traces do not persist the raw user query as a dedicated
    field (the trace schema captures the agent's epistemic state, not
    the prompt). For the shadow review we fall back to "" when no
    query is recoverable — the shadow can still review the reasoning
    and self-critique from the tool calls + crystals + final text.

    Phase 10 may add a `user_query` column to reasoning_traces if the
    metacognitive layer needs it; for now the shadow degrades
    gracefully without it.
    """
    # No dedicated field today. Return empty; callers may override by
    # passing user_query explicitly to shadow_review_trace.
    return ""


# ---------------------------------------------------------------------------
# Top-level: review one persisted trace (P0.62, P0.63)
# ---------------------------------------------------------------------------

async def shadow_review_trace(
    *,
    store: "MetadataStore",
    trace_id: str,
    anthropic_client: Any = None,
    sampling_policy: Optional[ShadowSamplingPolicy] = None,
    user_query: Optional[str] = None,
    shadow_model: Optional[str] = None,
    force: bool = False,
) -> dict[str, Any]:
    """Review a persisted reasoning trace with the shadow critic.

    The manual/triggered entry point for Phase 9.5 (P0.62). Phase 10's
    metacognitive scheduler will call this on a cadence; for now it is
    invoked manually or by tests.

    Steps:
      1. Load the trace (`store.get_reasoning_trace`).
      2. Load the agent_self critique(s) for the trace
         (`store.list_critiques_for_trace`), take the first
         agent_self critique as the one to review.
      3. Apply the sampling policy (unless `force=True`). When the
         policy declines, return early with sampled=False.
      4. Run the shadow critique LLM call.
      5. Persist Critique(critic_role="shadow") + ActionItems.

    Args:
        store: MetadataStore with the McrExtensionsMixin bound.
        trace_id: the trace to review.
        anthropic_client: the (frontier-model) Anthropic client.
        sampling_policy: the policy deciding whether to shadow.
            Defaults to a fresh ShadowSamplingPolicy() reading
            settings.shadow_critic_sample_rate.
        user_query: optional explicit user query (the trace doesn't
            persist it; pass it if known for a richer review).
        shadow_model: override settings.shadow_critic_model.
        force: skip the sampling decision and always run (used by
            tests and by explicit operator-triggered reviews).

    Returns: a dict with keys:
        sampled: whether the shadow ran (False when the policy
            declined and force=False).
        shadow_critique_id: the persisted shadow critique id, or None
            if it didn't run / failed.
        action_item_ids: list of persisted action_item ids.
        reason: short string explaining the outcome (for logs/tests).

    NEVER raises (P0.44 discipline).
    """
    out: dict[str, Any] = {
        "sampled": False,
        "shadow_critique_id": None,
        "action_item_ids": [],
        "reason": "",
    }

    # --- 1. Load the trace. ----------------------------------------
    try:
        trace = await store.get_reasoning_trace(trace_id)
    except Exception as e:
        logger.warning(
            "shadow_critic.trace_load_failed",
            trace_id=trace_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        out["reason"] = "trace_load_failed"
        return out

    if trace is None:
        logger.warning("shadow_critic.trace_not_found", trace_id=trace_id)
        out["reason"] = "trace_not_found"
        return out

    # --- 2. Load the agent_self critique for this trace. -----------
    self_obs: list[dict[str, Any]] = []
    self_summary: Optional[str] = None
    try:
        critiques = await store.list_critiques_for_trace(trace_id)
    except Exception as e:
        logger.warning(
            "shadow_critic.self_critique_load_failed",
            trace_id=trace_id,
            error=str(e),
        )
        critiques = []

    # Take the first agent_self critique as the one to review. There
    # is normally exactly one per trace (Phase 9A emits one). If a
    # shadow critique already exists, we still proceed — re-shadowing
    # is allowed (Phase 10 may re-review), and we only READ the
    # agent_self critique here.
    for c in critiques:
        if c.critic_role == "agent_self":
            self_obs = list(c.observations or [])
            self_summary = c.summary_text
            break

    # --- 3. Sampling decision. -------------------------------------
    policy = sampling_policy or ShadowSamplingPolicy()
    if not force and not policy.should_shadow_trace(self_obs):
        out["reason"] = "not_sampled"
        logger.info(
            "shadow_critic.skipped_by_sampling",
            trace_id=trace_id,
            self_observation_count=len(self_obs),
        )
        return out

    out["sampled"] = True

    # --- 4. Reconstruct the review payload + run the shadow LLM. ---
    final_text = _extract_final_text_from_trace(trace)
    resolved_query = (
        user_query
        if user_query is not None
        else _extract_user_query_from_trace(trace)
    )

    observations, action_items, summary_text = await run_shadow_critique(
        anthropic_client=anthropic_client,
        user_query=resolved_query,
        customer_id=trace.customer_id,
        agent_final_text=final_text,
        tool_calls_log=list(trace.tool_calls or []),
        crystals_used=list(trace.crystals_used or []),
        self_critique_observations=self_obs,
        self_critique_summary=self_summary,
        model=shadow_model,
    )

    chosen_model = shadow_model or settings.shadow_critic_model

    # --- 5. Persist the shadow critique. ---------------------------
    # trace_id is a HARD FK pointer here (P0.63) — the trace exists.
    # This is the CU-19 resolution for the shadow path: no
    # update_critique_trace_id needed, because the shadow always runs
    # after the trace is persisted.
    try:
        critique = await store.create_critique(
            customer_id=trace.customer_id,
            critic_role="shadow",
            critic_model=chosen_model,
            trace_id=trace.id,
            sequence_id=trace.sequence_id,
            turn_index=trace.turn_index,
            observations=observations,
            summary_text=summary_text or None,
            total_action_items=len(action_items),
        )
        out["shadow_critique_id"] = critique.id
    except Exception as e:
        logger.warning(
            "shadow_critic.critique_persist_failed",
            trace_id=trace_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        out["reason"] = "critique_persist_failed"
        return out

    # --- 6. Persist action items. ----------------------------------
    persisted_ids: list[str] = []
    for item_data in action_items:
        try:
            item = await store.create_action_item(
                critique_id=critique.id,
                customer_id=trace.customer_id,
                action_type=item_data["action_type"],
                content=item_data.get("content", {}),
                critic_confidence=item_data.get("critic_confidence"),
            )
            persisted_ids.append(item.id)
        except Exception as e:
            logger.warning(
                "shadow_critic.action_item_persist_failed",
                trace_id=trace_id,
                critique_id=critique.id,
                action_type=item_data.get("action_type"),
                error=str(e),
                error_type=type(e).__name__,
            )
            continue

    out["action_item_ids"] = persisted_ids
    out["reason"] = "shadowed"

    logger.info(
        "shadow_critic.review_complete",
        trace_id=trace_id,
        customer_id=trace.customer_id,
        shadow_critique_id=critique.id,
        observation_count=len(observations),
        action_item_count=len(persisted_ids),
        critic_model=chosen_model,
    )

    return out
