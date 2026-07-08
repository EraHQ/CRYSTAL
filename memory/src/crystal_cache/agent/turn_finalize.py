"""Shared post-turn signal layer for CRYS surfaces that run `Agent.run`.

`Agent.run` is the only seam shared by the agent endpoint and both coding-agent
turns, and it deliberately emits none of a turn's signals — so each surface had
grown its own wrapper and its own subset, and they drifted (see
docs/LENS_PARITY_AUDIT.md). This module is the fix: ONE function,
`finalize_agent_turn`, that emits the *universal* post-turn signal set every
CRYS answer should produce regardless of which lens drove it. Both CRYS surfaces
call it; lens-specific steps (verify, reflection, resync, the Agents-view
session timeline, shadow eval) keep layering on top and this function does not
know about them.

The universal set (and where each piece already lived):

  1. **Cost-ledger row** — `record_agent_llm_cost` → `store.record_llm_call`.
     The turn's aggregate tokens + computed micro-USD, attributed to the team.
  2. **Citations + grounding + marketplace credit + coverage-gap** —
     `ground_agent_citations`. Grounds each crystal the retrieval tools
     surfaced against the final answer; records all, credits the grounded
     marketplace ones (G4), files a `uncited_answer` knowledge-gap when a
     substantive answer grounds to nothing (G1c).
  3. **MCR** — `emit_mcr_artifacts` (in `mcr_emitter.py`): a deterministic
     reasoning trace + a Haiku self-critique + action items.

WHY THIS MODULE IS FRAMEWORK-FREE: the first two helpers used to live in
`endpoints/agent.py`, which imports FastAPI — so the coding agent (a CLI, no web
framework) could not call them without dragging the framework in. They moved
here, a sibling of `agent.py` / `mcr_emitter.py`, so the endpoint *and* the
coding agent both import cleanly. `endpoints/agent.py` re-exports them so
existing import paths (e.g. `tests/test_agent_cost.py`,
`tests/test_agent_citations.py`) keep working.

IMPORT-SAFETY NOTE (load-bearing): `ground_agent_citations` imports
`ground_sources_against_answer` + `CitationSource` *locally, inside the function
body*. The citations test patches `ground_sources_against_answer` at its source
module (`crystal_cache.retrieval.citation_grounding`); a call-time local import
resolves the patched attribute, a module-level import here would capture an
unpatched reference and break the test. Keep these imports local.

FAIL-SAFETY: every step is individually fail-safe — `record_agent_llm_cost`
catches and returns None, `ground_agent_citations` catches internally,
`emit_mcr_artifacts` never raises (P0.44). `finalize_agent_turn` therefore adds
no outer try/except: it cannot raise from these steps, and an outer catch would
only mask real bugs. This mirrors the agent endpoint's existing posture exactly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import structlog

from ..scan.gap_disposition import (
    classify_gap_disposition as _classify_gap_disposition,
)
from ..config import settings
from .agent import DEFAULT_MODEL
from .mcr_emitter import emit_mcr_artifacts

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore
    from ..models import Customer

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Last-user-query extraction
# ---------------------------------------------------------------------------

def _extract_last_user_query(
    messages_dicts: list[dict[str, Any]],
) -> str:
    """Pull the text of the latest user message from the request body.

    The message list is Anthropic-shaped: each entry has `role` and
    `content`, where content is either a string or a list of content
    blocks. We walk in reverse, returning the first user-role
    message's text.

    For content lists (the multi-block shape), we concatenate the
    text blocks. This catches both the simple case (string content)
    and the rare case where the caller sent a multi-block user
    message (e.g. tool_result + text reply).

    Returns "" when no user message is found — emit_mcr_artifacts
    handles empty user_query gracefully.
    """
    for msg in reversed(messages_dicts):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    txt = block.get("text", "")
                    if isinstance(txt, str):
                        parts.append(txt)
            if parts:
                return "\n".join(parts)
        # Unknown content shape — skip, keep looking.
    return ""


# ---------------------------------------------------------------------------
# Cost ledger (C0 cost-parity wiring)
# ---------------------------------------------------------------------------

async def record_agent_llm_cost(
    *,
    store: "MetadataStore",
    customer_id: str,
    result: dict[str, Any],
    sequence_id: Optional[str],
    origin: str = "agent",
    billing: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Record one cost row for a CRYS agent run — the C0 cost-parity wiring.

    The proxy emits a `record_llm_call` row per turn (chat_proxy.py, Growth
    G3); the agent surface did not, so a CRYS run was invisible to the cost
    ledger and the Inspector's spend views. This closes that gap: the agent
    loop's aggregate tokens + computed cost, attributed to the team with
    `session_id=sequence_id` (the conversation is the agent unit the per-agent
    rollups group by).

    `origin` attributes the row to a surface (the cost column is free text):
    the HTTP agent endpoint passes ``"agent"`` (the default, preserving the C0
    behavior); the coding REPL passes ``"coding"`` and the background/daemon
    passes ``"coding-bg"``, so coding-agent spend is attributable instead of
    invisible.

    Returns the persisted row dict (which carries ``computed_cost_micro_usd``)
    so a caller can reuse the figure — e.g. the coding agent feeds it straight
    into its Agents-timeline `turn_completed` event rather than recomputing.

    Flag-gated (`settings.enable_cost_accounting`) and fail-safe (P0.44): when
    the flag is off or the write fails, returns None and the agent response is
    unaffected. Cache-token fields are read from `result` and default to 0, so
    once C1 (prompt caching) surfaces `cache_creation_tokens` /
    `cache_read_tokens` on the run result they are captured + priced here with
    no change to this helper.

    Scope: the MCR self-critique's Haiku call (`emit_mcr_artifacts`) is a
    separate model invocation and is NOT metered here yet — that waits on the
    emitter surfacing its own token usage. The agent-loop row recorded here is
    the figure the C1 caching win is measured against.
    """
    if not settings.enable_cost_accounting:
        return None
    try:
        from ..cost.pricing import price_table_from_settings
        return await store.record_llm_call(
            customer_id,
            model=result.get("model") or DEFAULT_MODEL,
            input_tokens=int(result.get("prompt_tokens") or 0),
            output_tokens=int(result.get("completion_tokens") or 0),
            cache_creation_tokens=int(result.get("cache_creation_tokens") or 0),
            cache_read_tokens=int(result.get("cache_read_tokens") or 0),
            session_id=sequence_id,
            operator_id=None,  # the operator layer lands in Foundation F1
            origin=origin,
            billing=billing,
            price_table=price_table_from_settings(
                settings.llm_price_table_overrides
            ),
        )
    except Exception as e:
        logger.warning(
            "cost.record_failed", customer_id=customer_id, error=str(e),
        )
        return None


# ---------------------------------------------------------------------------
# Citations (P3, CC-D11 = grounding-based implicit credit)
# ---------------------------------------------------------------------------

# Minimum answer length (chars) for the G1c uncited-answer gap dual. A
# substantive answer that retrieved knowledge yet grounded to nothing is a
# coverage signal; trivial "Done."-style turns shouldn't generate gap noise.
_AGENT_UNCITED_GAP_MIN_CHARS = 80


async def ground_agent_citations(
    *,
    store: "MetadataStore",
    encoder: Any,
    customer: "Customer",
    result: dict[str, Any],
    user_query: str,
    sequence_id: Optional[str],
) -> dict[str, Any]:
    """Attribute + meter an agent run's grounded sources (P3, CC-D11 = B).

    Returns {surfaced_crystals, grounded_count, matched_fact_ids} (zeros/
    empty on any internal failure — C2 uses these for the agent's
    query_log row; the never-raises discipline is unchanged).

    Grounding-based implicit credit. The agent surfaces crystals through its
    retrieval tools rather than emitting ``[[cc:N]]`` markers, so this grounds
    each SURFACED crystal against the final answer and records/credits the
    grounded ones — reusing the proxy's record_citations +
    record_citation_credit rail (data parity per the FOUNDATION_AND_GROWTH
    "lens" principle) without mechanism parity. No markers means no answer
    rewrite and no Sources footer; v1 is the metering rail, not user-facing
    provenance display (that would want explicit markers and is a deferred
    follow-up).

    Flag-gated (settings.enable_citations / enable_marketplace_metering) and
    fail-safe (P0.44): citation processing never breaks the agent's response.
    query_log_id is None — the agent endpoint writes no query_log row — so
    citations land crystal-scoped; G4 credit dedupes on interaction_id (the
    agent run id).
    """
    if not settings.enable_citations:
        return {"surfaced_crystals": 0, "grounded_count": 0, "matched_fact_ids": []}
    try:
        final_text = result.get("final_text") or ""

        # Collect surfaced crystals (+ their surfaced fact ids) from retrieval
        # tool outputs. Non-retrieval tools (llm_invoke, crystal_write) carry
        # no matched_crystal_ids and are skipped.
        surfaced: dict[str, set[str]] = {}
        for call in (result.get("tool_calls") or []):
            output = call.get("output")
            if not isinstance(output, dict):
                continue
            for cid in (output.get("matched_crystal_ids") or []):
                if not cid:
                    continue
                bucket = surfaced.setdefault(cid, set())
                for fid in (output.get("matched_fact_ids") or []):
                    if fid:
                        bucket.add(fid)
        _all_fact_ids = sorted({f for fids in surfaced.values() for f in fids})
        if not surfaced:
            return {"surfaced_crystals": 0, "grounded_count": 0, "matched_fact_ids": []}

        # One source per surfaced crystal; its text is the surfaced facts'
        # content (fallback: all of the crystal's facts). Handles are
        # positional only — no markers are emitted, so they're just row labels.
        from ..retrieval.citations import CitationSource
        from ..retrieval.citation_grounding import (
            ground_sources_against_answer,
        )

        sources: list[tuple[CitationSource, str]] = []
        for i, (cid, fact_ids) in enumerate(surfaced.items(), start=1):
            try:
                facts = await store.list_facts_for_crystal(cid)
            except Exception:  # noqa: BLE001
                facts = []
            texts: list[str] = []
            for f in facts:
                if fact_ids and f.id not in fact_ids:
                    continue
                val = (getattr(f, "claim_text", None)
                       or getattr(f, "answer_value", None) or "")
                if val:
                    texts.append(val)
            if not texts:  # matched ids didn't resolve — fall back to all facts
                for f in facts:
                    val = (getattr(f, "claim_text", None)
                           or getattr(f, "answer_value", None) or "")
                    if val:
                        texts.append(val)
            sources.append((
                CitationSource(handle=str(i), crystal_id=cid),
                "\n".join(texts)[:2000],
            ))

        grounded_results = await ground_sources_against_answer(
            encoder, final_text, sources,
            threshold=settings.agent_citation_grounding_threshold,
        )
        grounded_count = sum(1 for r in grounded_results if r["grounded"])

        # Record all (grounded + ungrounded) for the ledger rail + telemetry;
        # grounded gates G4 credit.
        await store.record_citations(
            customer.id,
            query_log_id=None,
            citations=[
                {
                    "crystal_id": r["source"].crystal_id,
                    "version": r["source"].version,
                    "handle": r["source"].handle,
                    "claim_span": r["claim_span"],
                    "grounding_score": r["grounding_score"],
                    "grounded": r["grounded"],
                }
                for r in grounded_results
            ],
        )
        logger.info(
            "agent.citations_processed",
            customer_id=customer.id,
            surfaced=len(sources),
            grounded=grounded_count,
        )

        # G4 marketplace metering: a grounded citation of a marketplace crystal
        # mints a shard credit for its owner. Idempotent on (interaction,
        # crystal); self-traffic + non-marketplace excluded inside
        # record_citation_credit. interaction_id = the agent run id.
        if settings.enable_marketplace_metering and grounded_count:
            interaction_id = result.get("id") or sequence_id or ""
            for r in grounded_results:
                if not r["grounded"]:
                    continue
                try:
                    cry = await store.get_crystal(r["source"].crystal_id)
                    if cry is None:
                        continue
                    await store.record_citation_credit(
                        crystal_id=r["source"].crystal_id,
                        owner_operator_id=getattr(cry, "owner_operator_id", None),
                        crystal_group_team_id=getattr(cry, "group_team_id", None),
                        crystal_type=getattr(cry, "crystal_type", None),
                        crystal_customer_id=getattr(cry, "customer_id", None),
                        consuming_team_id=customer.id,
                        interaction_id=interaction_id,
                        raw_weight=1.0,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "agent.marketplace_credit_failed",
                        customer_id=customer.id, error=str(e),
                    )

        # G1c dual: the agent retrieved knowledge but nothing grounded to the
        # answer (and it's a substantive answer) -> a coverage-gap candidate.
        # Gated on surfaced>=1 so a no-retrieval answer (model used its own
        # knowledge) doesn't wrongly fire it.
        if (
            grounded_count == 0
            and len(final_text.strip()) >= _AGENT_UNCITED_GAP_MIN_CHARS
        ):
            try:
                await store.create_knowledge_gap(
                    customer.id,
                    domain=None,
                    subject=(user_query[:256] or None),
                    missing=(
                        "The agent retrieved knowledge but no surfaced crystal "
                        "grounded to its answer — the bank may lack coverage "
                        "for this query."
                    ),
                    source="uncited_answer",
                    # S3: the demand that missed, untruncated.
                    triggering_query=(user_query or None),
                    # S4: capability-aware disposition.
                    disposition=_classify_gap_disposition(),
                )
                logger.info("agent.uncited_gap", customer_id=customer.id)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "agent.uncited_gap_failed",
                    customer_id=customer.id, error=str(e),
                )
        return {
            "surfaced_crystals": len(surfaced),
            "grounded_count": grounded_count,
            "matched_fact_ids": _all_fact_ids,
        }
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "agent.citations_post_response_failed",
            customer_id=customer.id,
            error=str(e), error_type=type(e).__name__,
        )
        return {
            "surfaced_crystals": 0,
            "grounded_count": 0,
            "matched_fact_ids": [],
        }


# ---------------------------------------------------------------------------
# The shared post-turn entry point
# ---------------------------------------------------------------------------

async def finalize_agent_turn(
    *,
    store: "MetadataStore",
    encoder: Any,
    customer: "Customer",
    anthropic_client: Any = None,
    result: dict[str, Any],
    user_query: str,
    sequence_id: Optional[str],
    origin: str = "agent",
    turn_index: Optional[int] = None,
    query_log_id: Optional[str] = None,
    skip_self_critique: bool = False,
) -> dict[str, Any]:
    """Emit one CRYS turn's universal post-turn signal set.

    Called by both CRYS surfaces AFTER `Agent.run(...)` returns (the HTTP agent
    endpoint and, once wired, the coding REPL + background/daemon). Runs the
    three steps in the same order the agent endpoint has always used — cost,
    then citations, then MCR — so the refactor is behavior-preserving for the
    endpoint:

      1. `record_agent_llm_cost`   (cost-ledger row; origin attributes surface)
      2. `ground_agent_citations`  (citations + credit + uncited-answer gap)
      3. `emit_mcr_artifacts`      (reasoning trace + self-critique + actions)

    Args:
      store, encoder, customer, anthropic_client: the same dependencies the
        surface already holds (encoder grounds citations; anthropic_client is
        reused for the MCR self-critique call).
      result: the dict `Agent.run(...)` returned (its tokens, tool_calls,
        final_text drive all three steps).
      user_query: the last user message text (callers use
        `_extract_last_user_query` on their message list).
      sequence_id: the conversation id (None is fine).
      origin: cost-ledger surface attribution — "agent" (HTTP, default),
        "coding" (REPL), "coding-bg" (background/daemon).
      turn_index, query_log_id: passed through to MCR (and the citations
        rail's query_log link). The stateless HTTP endpoint passes None.
      skip_self_critique: MCR cost control — when True, persists the trace only
        and skips the extra Haiku self-critique call.

    Returns a dict the caller can reuse:

      {
        "cost": <record_llm_call row dict | None>,
        "cost_micro_usd": <int | None>,   # cost["computed_cost_micro_usd"]
        "mcr": {"trace_id", "critique_id", "action_item_ids"},
      }

    Citations are intentionally NOT in the return — `ground_agent_citations`
    persists rows the caller (or an observer) reads back from the store, which
    is the more honest signal-check anyway. Surfacing a grounded-count via the
    return is a deferred enhancement.

    NEVER raises: each step is individually fail-safe, so this is too (no outer
    try/except, to avoid masking real bugs).
    """
    cost = await record_agent_llm_cost(
        store=store,
        customer_id=customer.id,
        result=result,
        sequence_id=sequence_id,
        origin=origin,
        # E4 (2026-07-06): agent rows carry the same billing dimension as
        # proxy rows — a managed tenant's agent turns are rebillable spend
        # and count against the monthly cap.
        billing=(
            "managed"
            if getattr(customer, "inference_mode", "byok") == "managed"
            else None
        ),
    )

    citation_stats = await ground_agent_citations(
        store=store,
        encoder=encoder,
        customer=customer,
        result=result,
        user_query=user_query,
        sequence_id=sequence_id,
    ) or {"surfaced_crystals": 0, "grounded_count": 0, "matched_fact_ids": []}

    # C2 (2026-07-08): the agent surface writes query_logs too — the Logs
    # tab was proxy-only, so a tenant chatting through the playground saw
    # an empty audit trail. match_type maps from grounding: grounded
    # answer = high; retrieval surfaced but ungrounded = medium; no
    # retrieval = none. Never raises.
    import uuid as _uuid

    from ..models.query_log import QueryLog as _QueryLog

    try:
        await store.write_query_log(_QueryLog(
            id=f"ql_{_uuid.uuid4().hex[:16]}",
            customer_id=customer.id,
            query_text=user_query or "",
            match_type=(
                "high" if citation_stats["grounded_count"] > 0
                else "medium" if citation_stats["surfaced_crystals"] > 0
                else "none"
            ),
            injection_method="agent_tools",
            matched_facts=list(citation_stats["matched_fact_ids"]),
            response_text=(result.get("final_text") or None),
            upstream_call_made=True,
            prompt_tokens=(
                int(result["prompt_tokens"])
                if result.get("prompt_tokens") is not None else None
            ),
            completion_tokens=(
                int(result["completion_tokens"])
                if result.get("completion_tokens") is not None else None
            ),
        ))
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "agent.query_log_failed", customer_id=customer.id, error=str(e)
        )

    mcr = await emit_mcr_artifacts(
        store=store,
        customer_id=customer.id,
        user_query=user_query,
        agent_result=result,
        anthropic_client=anthropic_client,
        sequence_id=sequence_id,
        turn_index=turn_index,
        query_log_id=query_log_id,
        skip_self_critique=skip_self_critique,
    )

    return {
        "cost": cost,
        "cost_micro_usd": (cost or {}).get("computed_cost_micro_usd"),
        "mcr": mcr,
    }
