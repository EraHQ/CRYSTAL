"""V3 Signal Handler — processes push/pull tool calls from the LLM.

After the LLM response is received, crystal tool calls are extracted
and dispatched here. The handler processes each signal type and returns
tool_result messages to complete the tool call cycle.

PUSH signals:
  - crystal_push_store: Queue or auto-commit new knowledge
  - crystal_push_gap: Record knowledge gaps
  - crystal_push_correct: Flag facts for correction

PULL signals:
  - crystal_pull_research: Queue SLM agent task
  - crystal_pull_expand: Update session routing state

v2 port (Phase 6 Wave D): three SQL refactors against v1 verbatim,
all dissolving into existing Phase 5 store methods (no new store
methods required):

  1. _handle_store's review-queue branch:
       was: inline PushReviewQueueRow construction + session.add
       now: store.create_push_review_item(...)

  2. handle_signals' gap-persistence loop:
       was: inline KnowledgeGapRow construction + session.add (with
            a dead-write to a `payload` column that doesn't exist
            on the schema — v1 bug, see BD-11)
       now: store.create_knowledge_gap(...)

  3. handle_signals' research-task persistence loop:
       was: inline CognitionTaskRow construction + session.add
       now: store.create_cognition_task(...) with priority mapping

Per P0.3 (Phase 6 Wave D decision): we keep v1's two-loop pattern
rather than collapse into a single combined persist method. The two
loops write to different tables and have independent error semantics;
a wrapper would obscure rather than clarify.

Per P0.4 (Phase 6 Wave D decision): the v1 tool surface accepts
`priority ∈ {immediate, background, idle}` (per CRYSTAL_TOOLS in
v3_push_pull.py — wire-format contract per R3). The v2 Pydantic
TaskPriority Literal only accepts {urgent, background} (per Phase 4
D1 — match v1 model verbatim). We map at the boundary:
  - immediate → urgent
  - idle      → background
  - background→ background
  - (anything else) → background
Tool definition stays v1-verbatim per R3; v2 model strictness stays
per D1.

Phase 9B (2026-05-27): MCR integration per P0.41–P0.54.
====================================================================
Resolves BD-3 and BD-11 as committed in Phase 8.5's P0.39 deferral.
The chat_proxy code path that calls `handle_signals` already runs
AFTER the upstream LLM has produced its response. The push/pull
signals ARE the agent's act of self-correction (push_correct) and
gap-flagging (push_gap); Phase 9B captures both as structured
MCR critiques + action items.

P0.48: scope is `retrieval/v3_signal_handler.py` only. chat_proxy
is untouched in Phase 9B (Phase 9C wires the new kwargs through).

P0.49: NO separate self-critique LLM call. The signal IS the
critique observation. Phase 9A's Haiku self-critique grades the
agent's REASONING after-the-fact; Phase 9B captures explicit
self-corrections the agent ITSELF emitted DURING reasoning. Both
are valid `agent_self` critiques; only Phase 9A's needs a second
LLM pass.

P0.50: MCR writes happen in the existing Pass 2 persistence loops.
push_gap: each gap entry produces parallel KnowledgeGapRow + MCR
rows (Critique + ActionItem(gap_declaration)). push_correct: new
Pass 2 loop walks `_correct_data` and writes Critique + ActionItem
(edit_proposal). push_store / push_research / push_expand are
unaffected — they continue with their existing artifact classes.

P0.51: new kwargs on `handle_signals` are keyword-only with safe
defaults — `sequence_id=None`, `turn_index=None`, `agent_model=None`,
`mcr_enabled=False`. The existing chat_proxy call site continues
to work unchanged; Phase 9C flips `mcr_enabled=True` when the proxy
is ready to emit traces.

P0.52: `critic_model` resolves from the `agent_model` kwarg (the
upstream model that produced the signal). Default `"unknown"` —
honest about not knowing rather than guessing. `critic_role` is
hardcoded `agent_self` — signal handler emissions are never
shadow.

P0.53: action item content schemas locked.
  gap_declaration: {want, why_needed, domain, conversation_context}
  edit_proposal:   {key, old_value, new_value, rationale}

R9: per the v2 SQL quarantine, all DB writes go through
`store.create_critique(...)` and `store.create_action_item(...)`
on the McrExtensionsMixin (Phase 8.5). No inline SQLAlchemy added.
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Optional

import structlog

from ..scan.gap_disposition import (
    classify_gap_disposition as _classify_gap_disposition,
)

from .sparse_key import format_key
from ..encoding.executor import encode_native_async
from .v3_push_pull import ParsedSignals
from ..llm import get_llm_client

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore

logger = structlog.get_logger(__name__)


# v1-tool-wire-priority → v2-Pydantic-TaskPriority. See module docstring
# / P0.4 for why this mapping exists.
_PRIORITY_MAP = {
    "immediate": "urgent",
    "background": "background",
    "idle": "background",
}


# Phase 9B (P0.43, P0.52): default confidence for the
# source_contradiction observation that push_correct emits. The agent
# is asserting that an existing value is wrong — high but not certain.
# Phase 11.5 may surface a confidence kwarg through the
# crystal_push_correct tool definition itself.
_PUSH_CORRECT_CONFIDENCE = 0.8

# Phase 9B (P0.52): default critic_model when the caller did not
# pass `agent_model`. Honest about not-knowing rather than guessing
# at a sensible default that may drift from reality.
_UNKNOWN_CRITIC_MODEL = "unknown"


def _map_priority(wire_value: str) -> str:
    """Map a v1 wire-protocol priority value to the v2 TaskPriority Literal.

    Unknown values fall through to 'background' (the conservative
    default; matches v1's default when priority is absent).
    """
    return _PRIORITY_MAP.get(wire_value, "background")


# C5: research-task dedup. Two near-identical concurrent research
# requests each spin up a full cognition env (~$0.09 + ~25k tokens per
# the idle-log analysis). Two topics count as duplicates when their
# token sets are equal OR overlap at/above this Jaccard threshold. The
# threshold is high on purpose: it merges re-phrasings of the SAME ask
# without collapsing distinct-but-related topics, which are legitimately
# separate research.
_RESEARCH_DEDUP_JACCARD = 0.8
_TOPIC_WORD_RE = re.compile(r"[a-z0-9_]+")
# Function words dropped before comparison so two phrasings that differ
# only by filler ("... in the codebase" vs "... in codebase") compare
# equal, and content words carry the similarity signal. Kept to true
# grammatical stopwords — action verbs like 'locate'/'find' and entity
# words stay, since they distinguish what is actually being researched.
_TOPIC_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or",
    "is", "are", "was", "were", "be", "this", "that", "these", "those",
    "it", "its", "with", "from", "as", "at", "by", "into", "about",
    "what", "where", "which", "how", "does", "do", "did",
})


def _topic_tokens(topic: str) -> frozenset:
    return frozenset(
        w for w in _TOPIC_WORD_RE.findall((topic or "").lower())
        if w not in _TOPIC_STOPWORDS
    )


def _topics_duplicate(a: str, b: str) -> bool:
    """True if two research topics are normalized-equal or have a token
    Jaccard >= _RESEARCH_DEDUP_JACCARD."""
    ta, tb = _topic_tokens(a), _topic_tokens(b)
    if not ta or not tb:
        return False
    if ta == tb:
        return True
    inter = len(ta & tb)
    union = len(ta | tb)
    return union > 0 and (inter / union) >= _RESEARCH_DEDUP_JACCARD


async def run_inline_research(
    topic: str,
    customer_id: str,
    store: "MetadataStore",
    vector_index: Any = None,
    encoder: Any = None,
    conversation_context: str = "",
) -> str:
    """Run immediate research by searching the crystal bank.

    If an SLM client is available, feeds the raw findings + conversation
    context to Haiku for reasoning and structured output.

    Args:
        topic: The research question or task description
        conversation_context: Last N turns of chat so the SLM understands
            what was asked and what the user actually wants

    Returns a text result the LLM can use to continue generating.
    """
    if not vector_index or not encoder:
        return f"Research unavailable: no fact store configured. Topic: {topic}"

    try:
        # Encode the topic as a query vector
        query_vector = await encode_native_async(encoder, topic)

        # Search all fact types
        results = await vector_index.search_facts(
            customer_id=customer_id,
            query_vector=query_vector,
            pair_types=["content_chunk", "entity_attribute", "question_answer", "entity_relationship"],
            k=10,
        )

        if not results:
            return f"No information found for: {topic}"

        # Build raw context from the top results
        lines = []
        seen = set()
        for fact_id, crystal_id, pair_type, score in results[:8]:
            if fact_id in seen:
                continue
            seen.add(fact_id)
            facts = await store.list_facts_for_crystal(crystal_id)
            for f in facts:
                if f.id == fact_id:
                    if pair_type == "content_chunk":
                        content = f.claim_text or f.answer_value or ""
                        lines.append(content[:800])
                    else:
                        key = f.prompt_text or ""
                        val = f.claim_text or f.answer_value or ""
                        if key and val:
                            lines.append(f"{key}: {val}")
                    break

        if not lines:
            return f"No relevant content found for: {topic}"

        raw_context = "\n\n".join(lines)

        # If SLM is available, use it to reason about the findings
        if get_llm_client().is_ready() and len(raw_context) > 50:
            try:
                # Build a context-aware prompt
                slm_prompt_parts = []
                slm_prompt_parts.append(
                    "You are a research analyst. Your job is to carefully read "
                    "source material, reason about what it contains, and produce "
                    "a thorough, structured answer to the research question."
                )
                if conversation_context:
                    slm_prompt_parts.append(
                        f"\nCONVERSATION CONTEXT (what led to this research):\n"
                        f"{conversation_context[:1000]}"
                    )
                slm_prompt_parts.append(
                    f"\nRESEARCH TASK: {topic}"
                )
                slm_prompt_parts.append(
                    f"\nSOURCE MATERIAL:\n{raw_context[:3000]}"
                )
                slm_prompt_parts.append(
                    "\nINSTRUCTIONS:\n"
                    "1. Read the source material carefully\n"
                    "2. Extract every relevant detail that answers the research task\n"
                    "3. Organize your answer with clear structure\n"
                    "4. If the task asks you to infer or create something (like a prop list, "
                    "summary, analysis), do that reasoning now and produce the output\n"
                    "5. If the source material doesn't fully answer the question, "
                    "note specifically what's missing\n"
                    "6. Be thorough. This output will be stored as knowledge."
                )

                slm_text = get_llm_client().complete(
                    tier="small",
                    temperature=0.0,
                    max_tokens=1200,
                    system=None,
                    messages=[{
                        "role": "user",
                        "content": "\n\n".join(slm_prompt_parts),
                    }],
                )
                logger.info(
                    "push_pull.slm_research_complete",
                    customer_id=customer_id,
                    topic=topic[:60],
                    raw_chars=len(raw_context),
                    slm_chars=len(slm_text),
                )
                return slm_text
            except Exception as e:
                logger.warning("push_pull.slm_research_failed", error=str(e))
                # Fall through to raw results

        result = raw_context
        logger.info(
            "push_pull.inline_research_complete",
            customer_id=customer_id,
            topic=topic[:60],
            results=len(lines),
            chars=len(result),
        )
        return result

    except Exception as e:
        logger.warning("push_pull.inline_research_failed", error=str(e), topic=topic[:60])
        return f"Research failed: {str(e)}. Topic: {topic}"


AUTO_COMMIT_THRESHOLD = 0.9
REVIEW_QUEUE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Phase 9B: MCR write helpers
# ---------------------------------------------------------------------------

async def _write_gap_mcr_pair(
    store: "MetadataStore",
    *,
    customer_id: str,
    sequence_id: Optional[str],
    turn_index: Optional[int],
    critic_model: str,
    gap_data: dict[str, Any],
) -> tuple[Optional[str], Optional[str]]:
    """Write a Critique + ActionItem(gap_declaration) pair for one gap.

    Per P0.42, the conversation_context lives inside the action item's
    content JSON (option A — no schema change). The critique carries
    a single observation of type `gap_papered_over` — the closest fit
    from the locked P0.40 vocabulary for "agent flagged a gap";
    `assumption_identified` would mis-describe (agent didn't assume,
    agent disclaimed). The action item's `action_type` is
    `gap_declaration`.

    Returns: (critique_id, action_item_id). Either may be None on
    persistence failure; failures are logged and the caller continues.
    NEVER raises (Phase 9A P0.44 discipline).
    """
    domain = gap_data.get("domain", "") or ""
    subject = gap_data.get("subject", "") or ""
    missing = gap_data.get("missing", "") or ""
    conversation_context = gap_data.get("conversation_context", "") or ""

    # --- Critique row ---
    observation = {
        "type": "gap_papered_over",
        "text": (
            f"Agent flagged a knowledge gap via crystal_push_gap "
            f"(subject={subject!r}): {missing[:200]}"
        ),
        "confidence": _PUSH_CORRECT_CONFIDENCE,  # 0.8 — agent's own signal
        "anchors": [
            {"domain": domain, "subject": subject} if domain or subject else {}
        ],
    }

    critique_id: Optional[str] = None
    try:
        critique = await store.create_critique(
            customer_id=customer_id,
            critic_role="agent_self",
            critic_model=critic_model,
            trace_id=None,  # Trace doesn't exist yet in Phase 9B
                            # (proxy emission is Phase 9C); soft-join
                            # via (customer_id, sequence_id, turn_index)
                            # resolves the relationship.
            sequence_id=sequence_id,
            turn_index=turn_index,
            observations=[observation],
            summary_text=(
                f"Agent self-flagged a knowledge gap "
                f"(domain={domain!r}, subject={subject!r})"
            ),
            total_action_items=1,
        )
        critique_id = critique.id
    except Exception as e:
        logger.warning(
            "push_pull.gap_critique_persist_failed",
            customer_id=customer_id,
            subject=subject[:60],
            error=str(e),
            error_type=type(e).__name__,
        )
        return (None, None)

    # --- ActionItem(gap_declaration) per P0.53 ---
    # Schema: {want, why_needed, domain, conversation_context}
    # (Note: P0.42's text mentions "subject" in the content; we
    # use why_needed=subject per the gap_declaration content schema
    # that mcr_emitter establishes in Phase 9A. The `subject`
    # value lives in `why_needed`. The `domain` stays separate.)
    item_id: Optional[str] = None
    try:
        item = await store.create_action_item(
            critique_id=critique_id,
            customer_id=customer_id,
            action_type="gap_declaration",
            content={
                "want": missing,
                "why_needed": subject,
                "domain": domain,
                "conversation_context": conversation_context[:500],
            },
            critic_confidence=_PUSH_CORRECT_CONFIDENCE,
        )
        item_id = item.id
    except Exception as e:
        logger.warning(
            "push_pull.gap_action_item_persist_failed",
            customer_id=customer_id,
            critique_id=critique_id,
            subject=subject[:60],
            error=str(e),
            error_type=type(e).__name__,
        )
        # Critique persisted but item did not — total_action_items
        # on the critique is now stale (says 1 but no item exists).
        # Phase 10 reconciliation handles drift per Phase 8.5
        # honest-disclosures.
        return (critique_id, None)

    return (critique_id, item_id)


async def _write_correct_mcr_pair(
    store: "MetadataStore",
    *,
    customer_id: str,
    sequence_id: Optional[str],
    turn_index: Optional[int],
    critic_model: str,
    correct_data: dict[str, Any],
) -> tuple[Optional[str], Optional[str]]:
    """Write a Critique(source_contradiction) + ActionItem(edit_proposal)
    pair for one push_correct call. Resolves BD-3 per P0.43.

    The Critique's observation captures the contradiction the agent
    is asserting (old_value disagrees with the agent's knowledge);
    the ActionItem carries the proposed edit (replace old_value
    with new_value). Both share a default confidence of 0.8.

    Returns: (critique_id, action_item_id). Either may be None on
    persistence failure; failures are logged and the caller continues.
    NEVER raises (Phase 9A P0.44 discipline).
    """
    key = correct_data.get("key", "") or ""
    old_value = correct_data.get("old_value", "") or ""
    new_value = correct_data.get("new_value", "") or ""

    # --- Critique row ---
    observation = {
        "type": "source_contradiction",
        "text": (
            f"Agent flagged a contradiction at key={key!r} via "
            f"crystal_push_correct: stored old_value disagrees with "
            f"agent's current knowledge"
        ),
        "confidence": _PUSH_CORRECT_CONFIDENCE,
        "anchors": [
            {"key": key, "old_value": old_value[:200]},
        ],
    }

    critique_id: Optional[str] = None
    try:
        critique = await store.create_critique(
            customer_id=customer_id,
            critic_role="agent_self",
            critic_model=critic_model,
            trace_id=None,  # See _write_gap_mcr_pair comment.
            sequence_id=sequence_id,
            turn_index=turn_index,
            observations=[observation],
            summary_text=(
                f"Agent self-correction at key={key!r}: "
                f"proposes replacement"
            ),
            total_action_items=1,
        )
        critique_id = critique.id
    except Exception as e:
        logger.warning(
            "push_pull.correct_critique_persist_failed",
            customer_id=customer_id,
            key=key[:60],
            error=str(e),
            error_type=type(e).__name__,
        )
        return (None, None)

    # --- ActionItem(edit_proposal) per P0.53 ---
    # Schema: {key, old_value, new_value, rationale}
    item_id: Optional[str] = None
    try:
        item = await store.create_action_item(
            critique_id=critique_id,
            customer_id=customer_id,
            action_type="edit_proposal",
            content={
                "key": key,
                "old_value": old_value,
                "new_value": new_value,
                "rationale": "agent self-correction via crystal_push_correct",
            },
            critic_confidence=_PUSH_CORRECT_CONFIDENCE,
        )
        item_id = item.id
    except Exception as e:
        logger.warning(
            "push_pull.correct_action_item_persist_failed",
            customer_id=customer_id,
            critique_id=critique_id,
            key=key[:60],
            error=str(e),
            error_type=type(e).__name__,
        )
        return (critique_id, None)

    return (critique_id, item_id)


async def handle_signals(
    signals: ParsedSignals,
    customer_id: str,
    store: "MetadataStore",
    *,
    encoder: Any = None,
    vector_store: Any = None,
    vector_index: Any = None,
    session_state: Optional[dict] = None,
    conversation_context: str = "",
    sequence_id: Optional[str] = None,
    turn_index: Optional[int] = None,
    agent_model: Optional[str] = None,
    mcr_enabled: bool = False,
    query_text: Optional[str] = None,
) -> dict[str, Any]:
    """Process parsed push/pull signals from tool calls.

    Returns stats dict and builds tool_result messages for each
    crystal tool call so the LLM's tool call cycle completes cleanly.

    Persistence happens in passes (P0.3 — keep v1's loop pattern):
      Pass 1 (per-signal dispatch below): builds tool_result messages
        and accumulates internal `_gap_data` / `_research_data` /
        `_correct_data` lists.
      Pass 2 (after dispatch): walks each accumulated list and calls
        the corresponding store create method. Failures in pass 2
        log a warning and skip; one bad row does not abort the rest.

    Phase 9B kwargs (P0.51):
      sequence_id, turn_index — populate the soft-join key on MCR
        rows (Critique, ActionItem). The chat_proxy passes these
        when Phase 9C wires them through. Default None — the soft
        join still works with NULL keys, but resolution to a trace
        is impossible until both are populated. Phase 9A's agent
        endpoint does NOT call handle_signals, so its trace
        emission path is unaffected.
      agent_model — used as critic_model on emitted Critique rows
        (P0.52). Default None → critic_model="unknown". Phase 9C
        passes body.model from the chat_proxy request.
      mcr_enabled — feature flag. False (default) means Phase 9B's
        new code paths are inert and the function behaves exactly
        like Phase 6 Wave D. True means the new MCR writes fire
        alongside the existing artifact writes. Tests pass True
        explicitly. Phase 9C flips to True in the chat_proxy when
        ready.
    """
    if not signals.has_signals:
        return {"processed": 0, "tool_results": []}

    stats: dict[str, Any] = {
        "processed": 0,
        "auto_committed": 0,
        "queued_for_review": 0,
        "gaps_recorded": 0,
        "corrections_flagged": 0,
        "research_queued": 0,
        "expand_applied": 0,
        "errors": 0,
        # Phase 9B counters. Populated only when mcr_enabled=True.
        "mcr_critiques_written": 0,
        "mcr_action_items_written": 0,
        "tool_results": [],
    }

    # --- Pass 1: dispatch each tool call, build results + accumulate ---
    for tc in signals.raw_tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        tool_call_id = tc.get("id", "")
        args_str = func.get("arguments", "{}")

        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            args = {}

        result_content = ""

        try:
            if name == "crystal_push_store":
                result_content = await _handle_store(
                    args, customer_id, store, encoder, vector_store, stats
                )
            elif name == "crystal_push_gap":
                result_content = _handle_gap(args, customer_id, stats, conversation_context)
            elif name == "crystal_push_correct":
                result_content = _handle_correct(args, customer_id, stats)
            elif name == "crystal_pull_research":
                result_content = _handle_research(args, customer_id, stats, conversation_context)
            elif name == "crystal_pull_expand":
                result_content = _handle_expand(args, customer_id, session_state, stats)

            stats["processed"] += 1

        except Exception as e:
            stats["errors"] += 1
            result_content = f"Error processing {name}: {str(e)}"
            logger.warning("push_pull.handle_error", name=name, error=str(e))

        # Build the tool_result message
        if tool_call_id:
            stats["tool_results"].append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result_content,
            })

    logger.info("push_pull.handled", customer_id=customer_id, **{
        k: v for k, v in stats.items() if k != "tool_results" and not k.startswith("_")
    })

    # Phase 9B: resolve critic_model once for all MCR writes in this
    # call. The same upstream model produced every signal in this
    # `handle_signals` invocation; one resolution is correct.
    critic_model = agent_model or _UNKNOWN_CRITIC_MODEL

    # --- Pass 2: persist gaps and research tasks ---
    #
    # P0.3: two loops, two tables. Each loop routes through its
    # corresponding store create method (Phase 5 AuditTablesMixin).
    # Independent error handling per row so one bad write doesn't
    # poison the others.
    #
    # Phase 9B (P0.50): when mcr_enabled=True, the gap loop ALSO
    # writes parallel MCR rows (Critique + ActionItem(gap_declaration))
    # per BD-11 resolution. The KnowledgeGapRow write still happens
    # first; MCR is additive.

    # Persist knowledge gaps.
    #
    # Note (BD-11): v1 captured `conversation_context` into the
    # gap_data dict here and attempted to write it to a non-existent
    # `payload` column on KnowledgeGapRow. The v1 guard always
    # evaluated False so the context was silently dropped. **Phase 9B
    # resolves BD-11 per P0.42 option A**: when mcr_enabled=True, an
    # ActionItem(gap_declaration) writes alongside the KnowledgeGapRow
    # carrying conversation_context in its content JSON. The
    # KnowledgeGapRow schema is unchanged (no migration).
    for gap_data in stats.get("_gap_data", []):
        # Existing KnowledgeGapRow write (Phase 6 Wave D path,
        # unchanged).
        try:
            await store.create_knowledge_gap(
                customer_id,
                domain=gap_data.get("domain", "") or None,
                subject=gap_data.get("subject", "") or None,
                missing=gap_data.get("missing", ""),
                source="llm_observation",
                # S3: the model may name the key it looked for; the
                # conversation's query is the demand.
                full_key=gap_data.get("key") or None,
                triggering_query=query_text,
                # S4: the model may name workable/needs_document; else
                # capability decides.
                disposition=_classify_gap_disposition(
                    gap_data.get("disposition")
                ),
            )
        except Exception as e:
            logger.warning("push_pull.gap_persist_failed", error=str(e))

        # Phase 9B (P0.42): parallel MCR rows.
        if mcr_enabled:
            critique_id, item_id = await _write_gap_mcr_pair(
                store,
                customer_id=customer_id,
                sequence_id=sequence_id,
                turn_index=turn_index,
                critic_model=critic_model,
                gap_data=gap_data,
            )
            if critique_id is not None:
                stats["mcr_critiques_written"] += 1
            if item_id is not None:
                stats["mcr_action_items_written"] += 1

    # Persist research tasks.
    #
    # P0.4: map v1 wire priority → v2 Pydantic TaskPriority at this
    # boundary. The payload dict still includes the original wire
    # priority (under the "priority" key) so downstream cognition
    # tooling can inspect what the LLM actually requested if needed.
    #
    # C5: dedup near-identical concurrent research before enqueueing.
    # Compare each requested topic against (a) topics already pending or
    # running for this customer and (b) topics enqueued earlier in THIS
    # batch — one turn can emit several overlapping research calls.
    # Distinct-but-related topics fall below the Jaccard threshold and
    # are left alone. A lookup failure degrades to "no dedup" (still
    # enqueues); it never blocks the enqueue.
    _open_topics: list[str] = []
    if stats.get("_research_data"):
        try:
            _open_topics = await store.list_open_research_topics(customer_id)
        except Exception as e:
            logger.warning("push_pull.research_dedup_lookup_failed", error=str(e))
    _enqueued_topics: list[str] = []
    for research_data in stats.get("_research_data", []):
        topic = research_data.get("topic", "") or ""
        if topic and any(
            _topics_duplicate(topic, t) for t in (_open_topics + _enqueued_topics)
        ):
            stats["research_deduped"] = stats.get("research_deduped", 0) + 1
            logger.info(
                "push_pull.research_deduped",
                customer_id=customer_id,
                topic=topic[:80],
            )
            continue
        try:
            wire_priority = research_data.get("priority", "background")
            mapped_priority = _map_priority(wire_priority)
            await store.create_cognition_task(
                customer_id,
                task_type="research",
                payload=research_data,
                priority=mapped_priority,
            )
            _enqueued_topics.append(topic)
        except Exception as e:
            logger.warning("push_pull.task_persist_failed", error=str(e))

    # Phase 9B (P0.50): third Pass-2 loop for push_correct MCR writes.
    # Walks `_correct_data` accumulated in Pass 1 by `_handle_correct`.
    # Only runs when mcr_enabled=True; otherwise correction is logged
    # only (matches v1 behavior + Phase 6 Wave D port).
    if mcr_enabled:
        for correct_data in stats.get("_correct_data", []):
            critique_id, item_id = await _write_correct_mcr_pair(
                store,
                customer_id=customer_id,
                sequence_id=sequence_id,
                turn_index=turn_index,
                critic_model=critic_model,
                correct_data=correct_data,
            )
            if critique_id is not None:
                stats["mcr_critiques_written"] += 1
            if item_id is not None:
                stats["mcr_action_items_written"] += 1

    # Clean up internal accumulator keys from stats before returning
    # — callers shouldn't see leading-underscore keys.
    stats.pop("_gap_data", None)
    stats.pop("_research_data", None)
    stats.pop("_correct_data", None)

    return stats


async def _handle_store(
    args: dict,
    customer_id: str,
    store: Any,
    encoder: Any,
    vector_store: Any,
    stats: dict,
) -> str:
    """Handle crystal_push_store tool call.

    Three confidence bands:
      - >= AUTO_COMMIT_THRESHOLD (0.9): immediate add_pair_for_customer
      - >= REVIEW_QUEUE_THRESHOLD (0.5): queue for human review
      - below: log only, no persistence

    v2 port: the review-queue branch now routes through
    `store.create_push_review_item(...)` instead of inline
    `PushReviewQueueRow` construction. Same field semantics; the
    store method generates the id internally.

    Phase 9B: push_store does NOT emit MCR rows per P0.50.
    Storing knowledge is not a critique; the artifact is the
    Crystal/Fact or PushReviewItem row, not an MCR row.
    """
    key = args.get("key", "")
    value = args.get("value", "")
    confidence = float(args.get("confidence", 0.7))

    # Canonicalize the LLM-provided key before ANY persistence (auto-
    # commit or review queue). format_key parses the '|' path, sanitizes
    # each segment, drops empties, and caps lengths — so a malformed
    # LLM key can't land verbatim as prompt_text.
    key = format_key(key)

    if not key or not value:
        return "Error: key and value are required"

    if confidence >= AUTO_COMMIT_THRESHOLD and encoder and vector_store:
        crystal, fact = await store.add_pair_for_customer(
            customer_id=customer_id,
            prompt_text=key,
            answer_text=value,
            pair_type="question_answer",
            encoder=encoder,
            vector_store=vector_store,
            vector_index=vector_index,
            crystal_type="customer:legacy",
            source_kind="model_reasoning",  # closest valid enum value
        )
        stats["auto_committed"] += 1
        logger.info(
            "push_pull.auto_committed",
            customer_id=customer_id,
            key=key[:60],
            crystal_id=crystal.id,
            confidence=confidence,
        )
        return f"Stored: {key} (crystal {crystal.id})"

    elif confidence >= REVIEW_QUEUE_THRESHOLD:
        # v2 port: store method instead of inline SQL.
        item = await store.create_push_review_item(
            customer_id,
            key=key,
            value=value,
            confidence=confidence,
            source="llm_observation",
        )
        stats["queued_for_review"] += 1
        logger.info(
            "push_pull.queued_for_review",
            customer_id=customer_id,
            review_id=item.id,
            key=key[:60],
            confidence=confidence,
        )
        return f"Queued for review: {key} (id: {item.id})"

    else:
        logger.debug("push_pull.low_confidence", key=key[:60], confidence=confidence)
        return f"Noted but not stored (confidence {confidence} below threshold)"


def _handle_gap(args: dict, customer_id: str, stats: dict, conversation_context: str = "") -> str:
    """Handle crystal_push_gap tool call.

    Accumulates into stats['_gap_data']; persistence happens in
    handle_signals' second pass (P0.3).

    Phase 9B (P0.42, BD-11 resolution): the `conversation_context`
    captured here is now PERSISTED — when mcr_enabled=True in the
    handle_signals caller, the gap loop writes a parallel
    ActionItem(gap_declaration) whose content carries the context.
    The KnowledgeGapRow itself stays unchanged (no schema change);
    the context lives on the action item side per P0.42 option A.
    """
    domain = args.get("domain", "")
    subject = args.get("subject", "")
    missing = args.get("missing", "")

    stats["gaps_recorded"] += 1
    stats.setdefault("_gap_data", []).append({
        "domain": domain,
        "subject": subject,
        "missing": missing,
        "conversation_context": conversation_context[:500],
    })
    logger.info(
        "push_pull.gap_identified",
        customer_id=customer_id,
        domain=domain,
        subject=subject,
        missing=missing[:100],
    )
    return f"Gap recorded: {missing[:80]} (subject: {subject})"


def _handle_correct(args: dict, customer_id: str, stats: dict) -> str:
    """Handle crystal_push_correct tool call.

    Phase 9B (P0.43, BD-3 resolution): accumulates into
    stats['_correct_data']; persistence happens in handle_signals'
    third Pass-2 loop. When mcr_enabled=True, writes a parallel
    Critique(source_contradiction) + ActionItem(edit_proposal)
    pair. When mcr_enabled=False (default), behavior matches v1
    exactly: log only, no persistence.

    The 0.8 confidence is the default for the source_contradiction
    observation per _PUSH_CORRECT_CONFIDENCE.
    """
    key = args.get("key", "")
    old_value = args.get("old_value", "")
    new_value = args.get("new_value", "")

    stats["corrections_flagged"] += 1

    # Phase 9B (P0.43): accumulator for the Pass-2 MCR write loop.
    stats.setdefault("_correct_data", []).append({
        "key": key,
        "old_value": old_value,
        "new_value": new_value,
    })

    logger.info(
        "push_pull.correction_flagged",
        customer_id=customer_id,
        key=key[:60],
        old=old_value[:50],
        new=new_value[:50],
    )
    return f"Correction flagged for review: {key}"


def _handle_research(args: dict, customer_id: str, stats: dict, conversation_context: str = "") -> str:
    """Handle crystal_pull_research tool call.

    For immediate priority: search the crystal bank NOW and return
    the results as the tool response. The LLM gets the info it needs
    to continue generating. (The actual inline search happens via
    `run_inline_research` invoked by the caller — this handler only
    queues the request and flags it for immediate processing.)

    For background/idle priority: queue for the cognition worker.

    Accumulates into stats['_research_data']; persistence happens in
    handle_signals' second pass with priority mapping per P0.4.

    Phase 9B: push_research does NOT emit MCR rows per P0.50. The
    research task IS the action; the cognition_task row is the
    canonical artifact. Phase 9.5 may revisit if the shadow critic
    surfaces "agent over-researched" patterns worth tracking.
    """
    topic = args.get("topic", "")
    scope = args.get("scope", "")
    priority = args.get("priority", "background")

    stats["research_queued"] += 1
    stats.setdefault("_research_data", []).append({
        "topic": topic,
        "scope": scope,
        "priority": priority,
        "conversation_context": conversation_context[:500],
    })

    # For immediate requests, mark for inline processing.
    # The actual search happens in the caller after handle_signals
    # returns (the caller reads stats['_immediate_research']).
    if priority == "immediate":
        stats.setdefault("_immediate_research", []).append({
            "topic": topic,
            "scope": scope,
        })

    logger.info(
        "push_pull.research_queued",
        customer_id=customer_id,
        topic=topic[:80],
        scope=scope[:60],
        priority=priority,
    )
    return f"Research queued: {topic[:80]} (priority: {priority})"


def _handle_expand(
    args: dict,
    customer_id: str,
    session_state: Optional[dict],
    stats: dict,
) -> str:
    """Handle crystal_pull_expand tool call.

    Mutates session_state in place when present; otherwise logs and
    returns a 'no active session' message. No persistence — expand
    state is per-session, not per-customer.

    Phase 9B: push_expand does NOT emit MCR rows per P0.50. Expand
    is a session-state mutation, not a critique signal.
    """
    key_pattern = args.get("key_pattern", "")
    reason = args.get("reason", "")

    stats["expand_applied"] += 1

    if session_state is not None:
        session_state["expand_pattern"] = key_pattern
        session_state["expand_reason"] = reason
        logger.info(
            "push_pull.expand_applied",
            customer_id=customer_id,
            pattern=key_pattern[:60],
            reason=reason[:60],
        )
        return f"Context expanded: {key_pattern}"
    else:
        logger.info(
            "push_pull.expand_no_session",
            customer_id=customer_id,
            pattern=key_pattern[:60],
        )
        return f"Expand noted: {key_pattern} (no active session)"
