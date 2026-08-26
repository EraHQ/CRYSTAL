"""Curation tools — learn + the self-curation surface, in the registry.

WS C step 4 promoted the first three into the registry, exposed BOTH on the
external MCP memory surface (memory_learn / memory_conflicts / memory_gaps
bridge to them) AND to the agent loop and cognition, so the agent can teach
memory from outcomes and consult what its memory contradicts / lacks. Gate 0g
(2026-07-25) added the two WRITE halves: reading what the bank contradicts or
lacks without being able to settle or record anything left the agent routing
around the missing drawer (see the incident notes on each tool below). The
implementations live HERE (single source of truth); the MCP server bridges to
them like every other registry tool.

Contexts:
  - crystal_learn         agent-only (write-side; cognition writes via its
                          commit gate, like crystal_write).
  - knowledge_conflicts   agent + cognition (read-only self-curation surface).
  - knowledge_gaps        agent + cognition (read-only self-curation surface).
  - resolve_conflict      agent-only (write-side; gated on explicit in-chat
                          user confirmation, quoted verbatim).
  - record_gap            agent-only (write-side). Records a request, never a
                          task: gap -> research promotion stays a human click.

Mode-agnostic: nothing here assumes the caller writes code. State (store,
encoder, vector_store) is injected the same way the retriever tools get it —
via set_tool_state / _get_state.
"""
from __future__ import annotations

from typing import Any, Optional

import structlog

from ..tool_registry import register_tool
from .retrievers import _get_state
from ...llm import get_llm_client
from ...scan.assumptions import infer_bridging_assumption

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# crystal_learn — teach from an outcome (write-side)
# ---------------------------------------------------------------------------

@register_tool(
    name="crystal_learn",
    description=(
        "Teach memory from an outcome. outcome='success' caches a "
        "prompt -> solution pair for fast future recall; outcome='fail' records "
        "a correction (pass 'signal' describing what went wrong) so the system "
        "learns from the mistake. Use after you find out whether a past answer "
        "was right or wrong. Write-side: agent-only."
    ),
    contexts={"agent"},
    parameters_schema={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The task/question prompt the outcome is about.",
            },
            "response": {
                "type": "string",
                "description": "The answer/solution that was produced.",
            },
            "outcome": {
                "type": "string",
                "description": "'success' (cache the solution) or 'fail' (record a correction). Default 'success'.",
                "default": "success",
            },
            "signal": {
                "type": "string",
                "description": "On failure, a short description of what was wrong (optional).",
            },
            "crystal_type": {
                "type": "string",
                "description": "Crystal type id. Default 'customer:legacy'.",
                "default": "customer:legacy",
            },
        },
        "required": ["prompt", "response"],
    },
    returns_description=(
        "{'crystals_written': int, 'cached'?: bool, 'reflection'?: str, "
        "'knowledge'?: str, 'category'?: str, 'error'?: str}"
    ),
)
async def crystal_learn(
    customer_id: str,
    prompt: str,
    response: str,
    outcome: str = "success",
    signal: Optional[str] = None,
    crystal_type: str = "customer:legacy",
) -> dict[str, Any]:
    from ...learning import LearningService

    state = _get_state()
    svc = LearningService(
        store=state["store"],
        encoder=state["encoder"],
        vector_store=state["vector_store"],
        vector_index=state.get("vector_index"),
    )
    if outcome == "fail":
        result = await svc.learn_from_failure(
            customer_id=customer_id,
            prompt=prompt,
            response=response,
            failure_signal=signal or "User indicated this response was incorrect",
            crystal_type=crystal_type,
        )
        return {
            "crystals_written": result.crystals_written,
            "reflection": result.reflection,
            "knowledge": result.knowledge,
            "category": result.category,
            "error": result.error,
        }
    cached = await svc.cache_success(
        customer_id=customer_id,
        prompt=prompt,
        solution=response,
        crystal_type=crystal_type,
    )
    return {"crystals_written": 1 if cached else 0, "cached": cached}


# ---------------------------------------------------------------------------
# knowledge_conflicts — what the memory contradicts itself on (read)
# ---------------------------------------------------------------------------

@register_tool(
    name="knowledge_conflicts",
    description=(
        "List contradictions the system has detected in its own memory — pairs "
        "of stored facts that can't both be true. Returns each conflict's "
        "subject and the two conflicting claims. Use to check what the memory "
        "disagrees with itself on before trusting it. Read-only."
    ),
    contexts={"agent", "cognition"},
    parameters_schema={
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "Filter by status: open / resolved / dismissed. Default 'open'.",
                "default": "open",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum conflicts to return. Default 50.",
                "default": 50,
            },
        },
    },
    returns_description=(
        "{'conflicts': [{'id','subject','claim_a','claim_b','status',"
        "'detector','created_at'}], 'count': int}"
    ),
)
async def knowledge_conflicts(
    customer_id: str,
    status: str = "open",
    limit: int = 50,
) -> dict[str, Any]:
    state = _get_state()
    store = state["store"]
    conflicts = await store.list_knowledge_conflicts(
        customer_id, status=status or None, limit=limit,
    )
    return {
        "conflicts": [
            {
                "id": c.id,
                "subject": c.subject,
                "claim_a": c.claim_a,
                "claim_b": c.claim_b,
                "status": c.status,
                "detector": c.detector,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in conflicts
        ],
        "count": len(conflicts),
    }


# ---------------------------------------------------------------------------
# knowledge_gaps — what the memory is missing (read)
# ---------------------------------------------------------------------------

@register_tool(
    name="knowledge_gaps",
    description=(
        "List gaps the system has identified in its own memory — things it was "
        "asked about or expected to know but doesn't. Returns each gap's subject "
        "and a description of what's missing. Use to see what the memory lacks "
        "(and might need taught). Read-only."
    ),
    contexts={"agent", "cognition"},
    parameters_schema={
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "Filter by status: open / filled / closed. Default 'open'.",
                "default": "open",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum gaps to return. Default 50.",
                "default": 50,
            },
        },
    },
    returns_description=(
        "{'gaps': [{'id','subject','domain','missing','priority','status',"
        "'source','created_at'}], 'count': int}"
    ),
)
async def knowledge_gaps(
    customer_id: str,
    status: str = "open",
    limit: int = 50,
) -> dict[str, Any]:
    state = _get_state()
    store = state["store"]
    gaps = await store.list_knowledge_gaps(
        customer_id, status=status or None, limit=limit,
    )
    return {
        "gaps": [
            {
                "id": g.id,
                "subject": g.subject,
                "domain": g.domain,
                "missing": g.missing,
                "priority": g.priority,
                "status": g.status,
                "source": g.source,
                "created_at": g.created_at.isoformat() if g.created_at else None,
            }
            for g in gaps
        ],
        "count": len(gaps),
    }


# ---------------------------------------------------------------------------
# resolve_conflict — settle a contradiction the USER has adjudicated (write)
# ---------------------------------------------------------------------------
# 0g (ratified 2026-07-25). Motivating incident: asked to act on a confirmed
# launch date, the agent had a reader for conflicts and no way to settle one,
# so it wrote a stronger-matching crystal for the winning claim. Retrieval
# then found the new fact first while the old fact stayed live and the
# conflict stayed open — the bank silently disagreeing with itself behind a
# better vector. The missing drawer, not the model, was the defect.

# Resolutions the AGENT may pass. The store accepts a fourth, `blacklisted`,
# which ALSO records a blacklisted_reflections row so the claim is never
# re-learned or re-surfaced. Nothing on the agent surface undoes that, so
# durable suppression stays an operator click in the Conflicts console.
AGENT_RESOLUTIONS = ("superseded", "qualified", "dismissed")


@register_tool(
    name="resolve_conflict",
    description=(
        "Settle a knowledge conflict the user has adjudicated in this "
        "conversation. Call this ONLY after the user has told you which side "
        "is right - never on your own judgement, and never on a guess about "
        "what they meant. resolution='superseded' retires the outdated claim "
        "(pass loser='a' or 'b'); 'qualified' closes the conflict keeping "
        "BOTH claims, for when they are true under different conditions; "
        "'dismissed' closes a pairing that was never a real contradiction. "
        "Pass the user's own confirming words verbatim in user_confirmation. "
        "This is the ONLY correct way to act on a confirmed conflict: writing "
        "a new, better-matching fact instead leaves the outdated fact live "
        "and the conflict open, and the memory silently disagrees with "
        "itself. Write-side: agent-only."
    ),
    contexts={"agent"},
    parameters_schema={
        "type": "object",
        "properties": {
            "conflict_id": {
                "type": "string",
                "description": "The conflict's id, from knowledge_conflicts.",
            },
            "resolution": {
                "type": "string",
                "description": (
                    "One of: superseded (one claim is outdated - requires "
                    "loser), qualified (both true under different "
                    "conditions), dismissed (not a real conflict)."
                ),
            },
            "user_confirmation": {
                "type": "string",
                "description": (
                    "The user's own words confirming which side is right, "
                    "quoted verbatim from their message. Not your paraphrase, "
                    "not your inference. Required: without an explicit "
                    "confirmation from the user, do not call this tool - ask "
                    "them first."
                ),
            },
            "loser": {
                "type": "string",
                "description": (
                    "'a' or 'b' - which of the two claims the user says is "
                    "outdated. Required for superseded; ignored otherwise. "
                    "claim_a and claim_b come from knowledge_conflicts."
                ),
            },
        },
        "required": ["conflict_id", "resolution", "user_confirmation"],
    },
    returns_description=(
        "{'resolved': bool, 'conflict_id'?: str, 'status'?: str, "
        "'resolution'?: str, 'retired_claim'?: str | None, 'error'?: str}"
    ),
)
async def resolve_conflict(
    customer_id: str,
    conflict_id: str,
    resolution: str,
    user_confirmation: str,
    loser: Optional[str] = None,
) -> dict[str, Any]:
    from datetime import datetime, timezone

    if not (user_confirmation or "").strip():
        return {
            "resolved": False,
            "error": (
                "resolve_conflict requires the user's own confirming words in "
                "user_confirmation. Surface both claims, ask which is right, "
                "wait for their answer, then call this quoting what they said."
            ),
        }
    if resolution not in AGENT_RESOLUTIONS:
        return {
            "resolved": False,
            "error": (
                f"resolution must be one of "
                f"{', '.join(AGENT_RESOLUTIONS)}; got {resolution!r}. "
                "Flagging a claim as wrong-and-never-relearn (blacklisted) is "
                "an operator action in the Conflicts console."
            ),
        }
    if resolution == "superseded" and loser not in ("a", "b"):
        return {
            "resolved": False,
            "error": (
                "resolution 'superseded' requires loser='a' or loser='b' - "
                "which of the two claims the user says is outdated."
            ),
        }

    state = _get_state()
    store = state["store"]
    try:
        updated = await store.apply_conflict_resolution(
            conflict_id,
            resolution=resolution,
            loser=loser if resolution == "superseded" else None,
            resolved_at=datetime.now(timezone.utc),
            # Tenancy: a conflict belonging to another customer is
            # indistinguishable from a missing one - never an existence
            # oracle (the posture the admin route already takes).
            customer_id=customer_id,
        )
    except ValueError as e:
        return {"resolved": False, "error": str(e)}

    if updated is None:
        return {
            "resolved": False,
            "error": f"conflict {conflict_id!r} not found",
        }

    retired = None
    if resolution == "superseded":
        retired = updated.claim_a if loser == "a" else updated.claim_b

    # The confirming turn IS the provenance: this call's arguments are
    # persisted with the turn in tool_calls_log, so the quote that authorized
    # the retirement stays auditable without a schema change.
    logger.info(
        "curation.conflict_resolved",
        customer_id=customer_id,
        conflict_id=conflict_id,
        resolution=resolution,
        loser=loser,
        user_confirmation=(user_confirmation or "").strip()[:200],
    )
    return {
        "resolved": True,
        "conflict_id": updated.id,
        "status": updated.status,
        "resolution": updated.resolution,
        "retired_claim": retired,
    }


# ---------------------------------------------------------------------------
# record_gap — name what the memory could not answer (write)
# ---------------------------------------------------------------------------
# 0g (ratified 2026-07-25). The automatic miss-detector keys on retrieval
# SCORES, so an adjacent-content hit that scores well records no gap even
# when it answered nothing. The agent already makes the insufficiency
# judgement; this is the drawer for it. Without one it wrote the absence as
# a fact ("GAP: X is not documented"), which the Memory discipline rule in
# the system prompt forbids and which becomes a stored falsehood the moment
# the answer arrives.

# S4 taxonomy (models/knowledge_gap.py): who can close this gap, cheapest
# capable actor first. Required on this tool per Anthony's amendment to
# GAP-Q4 - the agent making the judgement is the one who can tell which
# kind it is, so it chooses rather than inheriting a default.
GAP_DISPOSITIONS = ("researchable", "workable", "needs_document")
GAP_PRIORITIES = ("low", "medium", "high")


@register_tool(
    name="record_gap",
    description=(
        "Record a question the memory could not answer. Call this when you "
        "searched, results came back, and they still did not answer what was "
        "asked - the automatic gap detector keys on retrieval SCORES, so a "
        "near-miss that scores well records nothing unless you record it. You "
        "are the one who can tell that an answer was insufficient; this is "
        "the drawer for that judgement. Never write a fact describing the "
        "absence ('X is not documented') - that becomes a stored falsehood "
        "the moment the answer arrives. disposition says who can close it: "
        "'researchable' (could be found by searching), 'workable' (could be "
        "settled by doing or trying it), 'needs_document' (genuinely private "
        "- only the operator has it). Recording a gap is a request, not a "
        "task: a human decides whether it becomes research. Write-side: "
        "agent-only."
    ),
    contexts={"agent"},
    parameters_schema={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "What was asked and not answered, as a question."
                ),
            },
            "disposition": {
                "type": "string",
                "description": (
                    "Who can close this: 'researchable' (findable by "
                    "searching), 'workable' (settled by doing it), "
                    "'needs_document' (only the operator has it). Required - "
                    "pick the cheapest capable actor."
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "Optional: what you did find and why it fell short. Kept "
                    "with the gap so whoever fills it knows what was already "
                    "tried."
                ),
            },
            "subject": {
                "type": "string",
                "description": "Optional subject the gap is about.",
            },
            "domain": {
                "type": "string",
                "description": "Optional domain label for the gap.",
            },
            "priority": {
                "type": "string",
                "description": "One of: low, medium, high. Default medium.",
                "default": "medium",
            },
        },
        "required": ["question", "disposition"],
    },
    returns_description=(
        "{'recorded': bool, 'gap_id'?: str, 'status'?: str, "
        "'disposition'?: str, 'priority'?: str, 'error'?: str}"
    ),
)
async def record_gap(
    customer_id: str,
    question: str,
    disposition: str,
    context: Optional[str] = None,
    subject: Optional[str] = None,
    domain: Optional[str] = None,
    priority: str = "medium",
) -> dict[str, Any]:
    question = (question or "").strip()
    if not question:
        return {
            "recorded": False,
            "error": "question must not be empty - name what went unanswered.",
        }
    if disposition not in GAP_DISPOSITIONS:
        return {
            "recorded": False,
            "error": (
                f"disposition must be one of "
                f"{', '.join(GAP_DISPOSITIONS)}; got {disposition!r}."
            ),
        }
    if priority not in GAP_PRIORITIES:
        return {
            "recorded": False,
            "error": (
                f"priority must be one of {', '.join(GAP_PRIORITIES)}; "
                f"got {priority!r}."
            ),
        }

    state = _get_state()
    store = state["store"]

    missing = question
    if (context or "").strip():
        missing = f"{question} (searched, not answered: {context.strip()})"

    gap = await store.create_knowledge_gap(
        customer_id,
        domain=(domain or "").strip() or None,
        subject=(subject or "").strip() or None,
        missing=missing,
        priority=priority,
        source="agent_observed",
        triggering_query=question,
        disposition=disposition,
    )
    logger.info(
        "curation.gap_recorded",
        customer_id=customer_id,
        gap_id=gap.id,
        disposition=disposition,
        priority=priority,
    )
    return {
        "recorded": True,
        "gap_id": gap.id,
        "status": gap.status,
        "disposition": disposition,
        "priority": priority,
    }


# ---------------------------------------------------------------------------
# assume — bridging inference over two crystals (write-side, gated birth)
# ---------------------------------------------------------------------------

@register_tool(
    name="assume",
    description=(
        "Test whether two crystals jointly support a bridging assumption - "
        "a plausible inference NEITHER states alone. Ephemeral by default: "
        "you get the verdict (statement, subject, confidence, reasoning) to "
        "reason with in this conversation and nothing is stored. Pass "
        "persist=true only when the assumption is worth keeping across "
        "sessions: it is stored as its own quarantined, recall-gated "
        "assumption crystal chained to both parent crystals - held out of "
        "recall until verified or promoted, so persisted speculation cannot "
        "contaminate answers. persist is YOUR judgment call and is honored "
        "even below the background worker's confidence threshold (the gated "
        "birth is the guard, not the number). Both crystals must belong to "
        "this tenant's own bank. Never store an assumption as a fact via "
        "other write tools; this is the drawer for speculation. Write-side: "
        "agent-only."
    ),
    contexts={"agent"},
    parameters_schema={
        "type": "object",
        "properties": {
            "crystal_a_id": {
                "type": "string",
                "description": "First parent crystal id.",
            },
            "crystal_b_id": {
                "type": "string",
                "description": (
                    "Second parent crystal id (distinct from the first)."
                ),
            },
            "persist": {
                "type": "boolean",
                "description": (
                    "Store the assumption as a quarantined, recall-gated "
                    "crystal chained to both parents. Default false = "
                    "ephemeral (verdict only, nothing written)."
                ),
                "default": False,
            },
        },
        "required": ["crystal_a_id", "crystal_b_id"],
    },
    returns_description=(
        "{'assumption_exists': bool, 'statement'?: str, 'subject'?: str, "
        "'confidence'?: float, 'reasoning'?: str, 'persisted': bool, "
        "'crystal_id'?: str, 'error'?: str}"
    ),
)
async def assume(
    customer_id: str,
    crystal_a_id: str,
    crystal_b_id: str,
    persist: bool = False,
) -> dict[str, Any]:
    a = (crystal_a_id or "").strip()
    b = (crystal_b_id or "").strip()
    if not a or not b:
        return {
            "assumption_exists": False,
            "persisted": False,
            "error": "both crystal ids are required.",
        }
    if a == b:
        return {
            "assumption_exists": False,
            "persisted": False,
            "error": "pass two DISTINCT crystals - a bridge needs two ends.",
        }

    client = get_llm_client()
    if not client.is_ready():
        return {
            "assumption_exists": False,
            "persisted": False,
            "error": (
                "no model provider configured; assumptions need one "
                "small-tier call."
            ),
        }

    state = _get_state()
    store = state["store"]

    # Tenancy pre-check with a CLEAR error (the inference core also
    # refuses foreign crystals at hydration - defense in depth; this
    # check exists so the agent hears WHY instead of a bare failure).
    for cid in (a, b):
        crystal = await store.get_crystal(cid)
        if crystal is None or crystal.customer_id != customer_id:
            return {
                "assumption_exists": False,
                "persisted": False,
                "error": (
                    f"crystal {cid!r} was not found in this tenant's bank."
                ),
            }

    verdict = await infer_bridging_assumption(
        client, store,
        customer_id=customer_id,
        crystal_a_id=a,
        crystal_b_id=b,
    )
    if verdict is None:
        return {
            "assumption_exists": False,
            "persisted": False,
            "error": "inference failed or returned an unusable verdict.",
        }

    out: dict[str, Any] = {
        "assumption_exists": verdict.assumption_exists,
        "reasoning": verdict.reasoning,
        "persisted": False,
    }
    if not verdict.assumption_exists:
        return out

    out["statement"] = verdict.statement.strip()
    out["subject"] = verdict.subject.strip()
    out["confidence"] = verdict.confidence

    if not persist:
        return out
    if not out["statement"] or not out["subject"]:
        out["error"] = (
            "verdict lacked a statement/subject; nothing persisted."
        )
        return out

    try:
        written = await store.create_assumption_crystal(
            customer_id,
            statement=out["statement"],
            subject=out["subject"],
            parent_a_id=a,
            parent_b_id=b,
            confidence=verdict.confidence,
            encoder=state["encoder"],
        )
        out["persisted"] = True
        out["crystal_id"] = written["crystal_id"]
        logger.info(
            "curation.assumption_persisted",
            customer_id=customer_id,
            crystal_id=written["crystal_id"],
            parent_a=a,
            parent_b=b,
            confidence=verdict.confidence,
        )
    except ValueError as e:
        out["error"] = str(e)
    except Exception as e:  # fail-safe: the verdict survives a bad write
        out["error"] = f"persist failed: {e}"
        logger.warning(
            "curation.assumption_persist_failed",
            customer_id=customer_id,
            parent_a=a,
            parent_b=b,
            error=str(e),
            error_type=type(e).__name__,
        )
    return out


# ---------------------------------------------------------------------------
# propose_correction — flag a stored value as wrong, on the agent's own
# judgement (write-side, proposal only)
# ---------------------------------------------------------------------------
# Audit (e) stage 1.9 (Q1=A / Q2=A / Q3=A, 2026-08-26). Port of the proxy's
# crystal_push_correct — the one proxy capability with no registry analog
# (retirement doc item 1). Two deliberate departures from the proxy:
#
#   1b payload: the proxy's ActionItem content ({key, old_value, new_value})
#   carries neither of the fields metacognition/alignment.py reads —
#   _canonical_key(edit_proposal) resolves `crystal_id` and the contradiction
#   rule reads `proposed_change` — so proxy corrections were never visible to
#   the classifier. This tool produces the agent's canonical edit_proposal
#   shape {crystal_id, proposed_change, rationale}, the same vocabulary the
#   Haiku self-critique already emits (agent/mcr_emitter.py).
#
#   S2-214: the proxy wrote its Critique inline with trace_id=None on a soft
#   join nothing implements — every one orphaned from the review loop. Here
#   the tool VALIDATES and acks only; emit_mcr_artifacts persists the
#   Critique(source_contradiction) + ActionItem(edit_proposal) pair in a
#   deterministic walk AFTER the reasoning trace exists, so the pair carries
#   the real trace_id. The tool-call log (persisted verbatim in the trace)
#   is the accumulator — the same provenance argument resolve_conflict makes
#   for user_confirmation.

_STORED_VALUE_ANCHOR_CHARS = 200  # proxy anchor discipline (old_value[:200])


@register_tool(
    name="propose_correction",
    description=(
        "Propose a correction to a stored value you believe is wrong, on "
        "your own judgement. This EDITS NOTHING: the proposal enters "
        "metacognitive review as an edit_proposal, where it is weighed "
        "against other critiques before anything changes. Use it when "
        "retrieval surfaced a crystal whose content you have good reason to "
        "believe is outdated or incorrect - and say why in rationale; a "
        "proposal without a real justification is noise to the reviewer. "
        "Boundaries: if an EXISTING conflict row is on the table and the "
        "user has adjudicated it, use resolve_conflict; if the memory is "
        "MISSING something rather than wrong about it, use record_gap; this "
        "tool is for disagreement with a stored value, on your judgement, "
        "without user adjudication. Write-side: agent-only."
    ),
    contexts={"agent"},
    parameters_schema={
        "type": "object",
        "properties": {
            "crystal_id": {
                "type": "string",
                "description": (
                    "The crystal holding the wrong value, from what "
                    "retrieval surfaced in this conversation."
                ),
            },
            "proposed_change": {
                "type": "string",
                "description": (
                    "The corrected statement - complete enough to apply "
                    "without this conversation's context."
                ),
            },
            "rationale": {
                "type": "string",
                "description": (
                    "Why you believe the stored value is wrong. Required - "
                    "the reviewer weighs the proposal on this."
                ),
            },
            "disputed_claim": {
                "type": "string",
                "description": (
                    "Optional: which specific stored claim is wrong, for "
                    "crystals holding more than one fact."
                ),
            },
        },
        "required": ["crystal_id", "proposed_change", "rationale"],
    },
    returns_description=(
        "{'proposed': bool, 'crystal_id'?: str, 'stored_value'?: str, "
        "'error'?: str}"
    ),
)
async def propose_correction(
    customer_id: str,
    crystal_id: str,
    proposed_change: str,
    rationale: str,
    disputed_claim: Optional[str] = None,
) -> dict[str, Any]:
    cid = (crystal_id or "").strip()
    change = (proposed_change or "").strip()
    why = (rationale or "").strip()
    if not cid:
        return {
            "proposed": False,
            "error": "crystal_id is required - name the crystal you dispute.",
        }
    if not change:
        return {
            "proposed": False,
            "error": (
                "proposed_change must not be empty - state the corrected "
                "value, complete enough to apply on its own."
            ),
        }
    if not why:
        return {
            "proposed": False,
            "error": (
                "rationale must not be empty - the reviewer weighs this "
                "proposal on WHY you believe the stored value is wrong."
            ),
        }

    state = _get_state()
    store = state["store"]

    # Tenancy check with a CLEAR error, same posture as assume(): a
    # foreign crystal is indistinguishable from a missing one - never
    # an existence oracle.
    crystal = await store.get_crystal(cid)
    if crystal is None or crystal.customer_id != customer_id:
        return {
            "proposed": False,
            "error": f"crystal {cid!r} was not found in this tenant's bank.",
        }

    # Q3=A: the anchor snippet is the crystal's ACTUAL stored value as
    # it stood when the agent disputed it (not the model's paraphrase),
    # captured here and carried on the tool's return so the finalize
    # walk lifts it without a re-read.
    stored_value = (
        (crystal.summary_text or crystal.answer_value or "")
        [:_STORED_VALUE_ANCHOR_CHARS]
    )

    logger.info(
        "curation.correction_proposed",
        customer_id=customer_id,
        crystal_id=cid,
        proposed_change=change[:80],
        rationale=why[:80],
    )
    return {
        "proposed": True,
        "crystal_id": cid,
        "stored_value": stored_value,
    }
