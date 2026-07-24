"""Retriever tools — the four V3 routers exposed as flat agent tools.

Per D-A3, the agent sees four separate retrieval tools, not one
collapsed `retrieve(mode=...)` tool. Per D-A4, three of them return
raw structured results and let the agent decide how to compose;
`depth_search` is the exception that synthesizes internally because
cross-crystal reasoning is what makes "depth" different from
"knowledge."

All four read from the FactVectorStore + MetadataStore (via lazy
imports against the router classes from `retrieval/v3_*.py`, which
are verbatim ports of v1 — see Wave 7A in PROJECT_LEDGER.md).

WIRE-FORMAT NAMING (P0.26):
- Agent-side names follow D-A3: content_search, knowledge_search,
  navigation_search, depth_search.
- Two of them (content_search, knowledge_search) carry
  `cognition_action_alias` mapping to the v1 cognition StepAction
  enum values (crystal_search for knowledge_search, crystal_key_scan
  for navigation_search). This preserves R3 wire-format compatibility
  with persisted cognition_tasks rows while letting the agent
  address tools by their design names. (Cognition's worker
  dispatcher after the §6.5.5 refactor looks up by StepAction.value.)
- web_search lives in tools/external.py (not here) but follows the
  same alias pattern.

CONTEXT ASSIGNMENTS:
- All four retrievers are read-side shared — agent ✅, cognition ✅.
- Two of them have direct StepAction analogues (cognition crystal_search
  ↔ agent knowledge_search; cognition crystal_key_scan ↔ agent
  navigation_search). The other two (content_search, depth_search)
  are agent-side innovations that cognition can grow into using
  via the shared registry without StepAction enum extension —
  cognition just calls them by their agent name, no alias needed.

ACCESS TO APP STATE:
Tools need access to the FastAPI app.state (for VectorStore,
FactVectorStore, prompt_encoder). The Agent class injects these at
call time; tool implementations receive them via a context-bound
function (see `_get_state` below). This keeps the tool signatures
free of FastAPI-specific types while still letting tools reach into
process-wide singletons.
"""
from __future__ import annotations

from typing import Any, Optional

import structlog

from ..tool_registry import register_tool
from ...encoding.executor import encode_native_async
from ...retrieval.tier_signal import conflict_note, tier_map, tier_note

logger = structlog.get_logger(__name__)


async def _apply_tier_signal(
    store: Any, customer_id: str, payload: dict[str, Any],
) -> dict[str, Any]:
    """Tier-as-epistemic-signal (RATIFIED 2026-07-02): annotate retrieval
    results with their crystals' quality tiers plus an action note the
    model can act on (verify / ask the user). NEVER changes ranking —
    tiers are a sign, not a weight. Fail-safe: a tier lookup hiccup never
    breaks a search."""
    try:
        tiers = await tier_map(
            store, customer_id, payload.get("matched_crystal_ids") or [],
        )
        payload["crystal_tiers"] = tiers
        payload["tier_note"] = tier_note(tiers)
    except Exception as e:  # noqa: BLE001 — annotation never breaks retrieval
        logger.warning("tier_signal.failed", error=str(e))
        payload.setdefault("crystal_tiers", {})
        payload.setdefault("tier_note", None)
    # CONF-R (2026-07-23): the read path from the idle machinery to
    # answer time — facts under an OPEN conflict arrive marked, with
    # the other side's claim attached, so the model reasons about the
    # disagreement in the moment instead of answering on half of it.
    # Same discipline as tiers: a sign, never a filter. Fail-safe.
    try:
        contested = await store.open_conflicts_for_facts(
            customer_id, payload.get("matched_fact_ids") or [],
        )
        payload["contested_facts"] = contested
        payload["conflict_note"] = conflict_note(contested)
    except Exception as e:  # noqa: BLE001 — annotation never breaks retrieval
        logger.warning("conflict_signal.failed", error=str(e))
        payload.setdefault("contested_facts", {})
        payload.setdefault("conflict_note", None)
    return payload


# ---------------------------------------------------------------------------
# Shared state injection
# ---------------------------------------------------------------------------
#
# Tools need access to objects that live on FastAPI's app.state
# (vector_store, fact_vector_store, prompt_encoder, metadata_store,
# settings.anthropic_api_key). We can't import app.state directly
# because of circular-import concerns and because tools are also
# called from non-FastAPI contexts (tests, scripts, the cognition
# worker).
#
# The pattern: the Agent class (in agent/agent.py) sets a module-level
# state-holder via `set_tool_state(state)` before invoking any tool
# from the registry. Tools call `_get_state()` to retrieve it. Tests
# and direct callers can set their own state dict.

_tool_state: dict[str, Any] = {}


def set_tool_state(state: dict[str, Any]) -> None:
    """Inject the tool execution state.

    Called by the Agent class once at construction time and by tests
    that want to override individual dependencies.

    Expected keys:
      - 'store': MetadataStore
      - 'vector_store': VectorStore (for the V2 crystal-level
        retrieval that crystal_recall uses)
      - 'fact_vector_store': FactVectorStore (for the V3 fact-level
        retrieval that the four routers use)
      - 'encoder': prompt_encoder (semantic text encoder)
    """
    global _tool_state
    _tool_state = state


def _get_state() -> dict[str, Any]:
    """Read the tool execution state. Raises if not initialized.

    Tools should call this lazily inside their implementation so that
    import-time registration doesn't depend on state being set yet.
    """
    if not _tool_state:
        raise RuntimeError(
            "Tool state not initialized. Call set_tool_state(...) "
            "before invoking tools from the registry. The Agent class "
            "in agent/agent.py does this automatically."
        )
    return _tool_state


# ---------------------------------------------------------------------------
# content_search
# ---------------------------------------------------------------------------

@register_tool(
    name="content_search",
    description=(
        "Find verbatim document chunks matching a query. Returns the "
        "raw text of the top matching chunks plus sparse-key locators. "
        "Best for: 'what does the document say about X', 'find the "
        "passage about Y', 'show me the section on Z'. Bad for: "
        "counting, listing, or questions about knowledge structure "
        "(use navigation_search for those)."
    ),
    contexts={"agent", "cognition"},
    parameters_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language query for the content lookup.",
            },
            "k": {
                "type": "integer",
                "description": "Maximum number of chunks to return. Default 5.",
                "default": 5,
            },
            "hints": {
                "type": "object",
                "description": (
                    "Optional classifier hints. Recognized keys: "
                    "'locator_prefix' (e.g. 'Scene 5'), 'subject', "
                    "'domain'."
                ),
            },
        },
        "required": ["query"],
    },
    returns_description=(
        "{'injection_text': str | None, 'matched_fact_ids': [str], "
        "'matched_crystal_ids': [str], 'crystal_tiers': {id: tier}, "
        "'tier_note': str | None, 'top_score': float, "
        "'fact_count': int, 'voicing': str}"
    ),
)
async def content_search(
    customer_id: str,
    query: str,
    k: int = 5,
    hints: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    from ...retrieval.v3_routers import ContentRouter

    state = _get_state()
    router = ContentRouter(
        vector_index=state["vector_index"],
        metadata_store=state["store"],
    )
    query_vector = await encode_native_async(state["encoder"], query)
    result = await router.search(
        customer_id=customer_id,
        query_vector=query_vector,
        k=k,
        hints=hints,
    )
    return await _apply_tier_signal(state["store"], customer_id, {
        "injection_text": result.injection_text,
        "matched_fact_ids": result.matched_fact_ids,
        "matched_crystal_ids": result.matched_crystal_ids,
        "top_score": result.top_score,
        "fact_count": result.fact_count,
        "voicing": result.voicing,
    })


# ---------------------------------------------------------------------------
# knowledge_search
# ---------------------------------------------------------------------------

@register_tool(
    name="knowledge_search",
    description=(
        "Find crystals (entities, Q&A pairs, relationships) matching "
        "a query. Returns top facts with their keys and values. Best "
        "for: 'what do we know about X', 'find the answer to Y', "
        "'lookup Z'. Returns small, structured pieces of information "
        "(not full document chunks — use content_search for those). "
        "The bank holds TWO kinds of knowledge, searched together: "
        "project-specific facts (keys like 'Code|path|symbol') and "
        "general engineering patterns (keys under 'General|...') — so "
        "one query can answer 'how should I write this, given how this "
        "codebase already does it'. When both apply, weigh the "
        "project's own conventions over the general pattern. "
        "Keys under 'Reflections|...' are lessons you yourself learned "
        "from past runs in this project that failed verification and "
        "were then fixed — hard-won, project-specific, and worth the "
        "same trust as the project's own conventions."
    ),
    contexts={"agent", "cognition"},
    parameters_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language query for the knowledge lookup.",
            },
            "k": {
                "type": "integer",
                "description": "Maximum number of facts to consider. Default 10.",
                "default": 10,
            },
            "hints": {
                "type": "object",
                "description": "Optional classifier hints (subject, locator_prefix, domain).",
            },
        },
        "required": ["query"],
    },
    cognition_action_alias="crystal_search",
    returns_description=(
        "{'injection_text': str | None, 'matched_fact_ids': [str], "
        "'matched_crystal_ids': [str], 'crystal_tiers': {id: tier}, "
        "'tier_note': str | None (present when non-whitelist knowledge "
        "contributed - consider verifying or asking the user), "
        "'top_score': float, 'fact_count': int, 'voicing': str}"
    ),
)
async def knowledge_search(
    customer_id: str,
    query: str,
    k: int = 10,
    hints: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    from ...retrieval.v3_routers import KnowledgeRouter

    state = _get_state()
    router = KnowledgeRouter(
        vector_index=state["vector_index"],
        metadata_store=state["store"],
    )
    query_vector = await encode_native_async(state["encoder"], query)
    result = await router.search(
        customer_id=customer_id,
        query_vector=query_vector,
        k=k,
        hints=hints,
    )
    return await _apply_tier_signal(state["store"], customer_id, {
        "injection_text": result.injection_text,
        "matched_fact_ids": result.matched_fact_ids,
        "matched_crystal_ids": result.matched_crystal_ids,
        "top_score": result.top_score,
        "fact_count": result.fact_count,
        "voicing": result.voicing,
    })


# ---------------------------------------------------------------------------
# navigation_search
# ---------------------------------------------------------------------------

@register_tool(
    name="navigation_search",
    description=(
        "Scan the sparse-key registry to answer 'what do you know?' "
        "and enumeration queries. Returns a structured summary of "
        "what knowledge exists for a given subject/domain, including "
        "gap detection ('we have Scenes 1-4 and 6-68 but no Scene "
        "5'). No vector search — operates directly on key structure. "
        "Best for: counting ('how many scenes'), listing ('what "
        "chapters exist'), enumerating ('all characters mentioned'), "
        "structural questions. Bad for: semantic meaning (use "
        "knowledge_search or content_search for those)."
    ),
    contexts={"agent", "cognition"},
    parameters_schema={
        "type": "object",
        "properties": {
            "query_text": {
                "type": "string",
                "description": "Original query text (for context; navigation does not vector-search it).",
                "default": "",
            },
            "hints": {
                "type": "object",
                "description": (
                    "Filter hints. Recognized keys: 'subject' "
                    "(e.g. 'Corporate Mistletoe'), 'domain', "
                    "'locator_prefix'."
                ),
            },
        },
    },
    # cognition_action_alias intentionally NOT set here. Cognition's
    # crystal_key_scan routes to the `key_scan` enumeration tool below
    # (raw findings), not to this overview router — the two have
    # different output contracts and filter models (see B / §6.5.5
    # unification). navigation_search stays the agent's
    # what-do-I-know overview tool.
    returns_description=(
        "{'injection_text': str | None, 'total_keys': int, "
        "'matching_keys': int, 'sources': dict, 'gaps': [str], "
        "'subjects': [str], 'domains': [str], 'matched_fact_ids': "
        "[str], 'matched_crystal_ids': [str], 'crystal_tiers': "
        "{id: tier}, 'tier_note': str | None, 'top_score': float, "
        "'fact_count': int, 'voicing': str}"
    ),
)
async def navigation_search(
    customer_id: str,
    query_text: str = "",
    hints: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    from ...retrieval.v3_navigation import NavigationRouter

    state = _get_state()
    router = NavigationRouter(
        fact_store=state["fact_vector_store"],
        metadata_store=state["store"],
    )
    result = await router.search(
        customer_id=customer_id,
        hints=hints,
        query_text=query_text,
    )
    return await _apply_tier_signal(state["store"], customer_id, {
        "injection_text": result.injection_text,
        "matched_fact_ids": result.matched_fact_ids,
        "matched_crystal_ids": result.matched_crystal_ids,
        "top_score": result.top_score,
        "fact_count": result.fact_count,
        "voicing": result.voicing,
        "total_keys": result.total_keys,
        "matching_keys": result.matching_keys,
        "sources": result.sources,
        "gaps": result.gaps,
        "subjects": result.subjects,
        "domains": result.domains,
    })


# ---------------------------------------------------------------------------
# depth_search
# ---------------------------------------------------------------------------

@register_tool(
    name="depth_search",
    description=(
        "Cross-crystal analytical synthesis. Searches relationship "
        "and entity facts about subjects, finds content chunks for "
        "scene references, organizes results chronologically using "
        "sparse-key locators, and (if an LLM client is configured) "
        "pre-digests the raw context into an analytical summary "
        "before returning. Best for: 'how does X relate to Y', "
        "'compare X and Y across the document', 'what's the "
        "throughline for X'. Returns a synthesized paragraph (the "
        "exception to D-A4's raw-results rule). Bad for: simple "
        "fact lookup (use knowledge_search) or verbatim content "
        "(use content_search)."
    ),
    contexts={"agent", "cognition"},
    parameters_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language query for the analytical synthesis.",
            },
            "k": {
                "type": "integer",
                "description": "Maximum number of facts to consider per channel. Default 20.",
                "default": 20,
            },
            "hints": {
                "type": "object",
                "description": "Optional classifier hints (subject, locator_prefix, domain).",
            },
        },
        "required": ["query"],
    },
    returns_description=(
        "{'injection_text': str | None  # synthesized paragraph, "
        "'matched_fact_ids': [str], 'matched_crystal_ids': [str], "
        "'top_score': float, 'fact_count': int, 'voicing': str}"
    ),
)
async def depth_search(
    customer_id: str,
    query: str,
    k: int = 20,
    hints: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    from ...retrieval.v3_depth import DepthRouter

    state = _get_state()
    router = DepthRouter(
        vector_index=state["vector_index"],
        metadata_store=state["store"],
    )
    query_vector = await encode_native_async(state["encoder"], query)
    result = await router.search(
        customer_id=customer_id,
        query_vector=query_vector,
        k=k,
        hints=hints,
        query_text=query,
    )
    return {
        "injection_text": result.injection_text,
        "matched_fact_ids": result.matched_fact_ids,
        "matched_crystal_ids": result.matched_crystal_ids,
        "top_score": result.top_score,
        "fact_count": result.fact_count,
        "voicing": result.voicing,
    }


# ---------------------------------------------------------------------------
# key_scan
# ---------------------------------------------------------------------------
#
# The enumeration primitive behind cognition's crystal_key_scan. Unlike
# navigation_search (which loads all facts and emits a what-do-I-know
# overview), key_scan does a targeted sparse-key prefix scan via the
# store method `list_facts_by_key_prefix` and returns the RAW matching
# facts in cognition's findings shape. Registered as a shared-registry
# tool with the `crystal_key_scan` cognition alias so the worker
# dispatcher resolves it through the registry like every other retrieval
# action. Agent context added 2026-06-11: the coding agent's bank demo
# surfaced an identity query ('what does <file> define') that resemblance
# top-matching answers incompletely — the agent needs raw enumeration,
# which is exactly this tool's contract.

@register_tool(
    name="key_scan",
    description=(
        "Enumerate facts whose sparse key matches. Sparse keys are "
        "wide->specific paths (e.g. 'Film|Corporate Mistletoe|Script|Scene 5'). "
        "key_prefix matches the WIDE (left) end; subject_contains matches "
        "ANY segment (find something regardless of where it sits in the "
        "path). Returns the raw matching facts, not a summary. Best for: "
        "counting, listing, and 'where is X defined' identity lookups that "
        "need the actual facts. General engineering patterns live under "
        "the 'General|' namespace (e.g. key_prefix 'General|Python|' lists "
        "every Python pattern you hold) — scan it when you want your "
        "general knowledge on a domain, not just this project's facts. "
        "This is the enumeration primitive behind "
        "cognition's crystal_key_scan; navigation_search gives a high-level "
        "overview instead."
    ),
    contexts={"agent", "cognition"},
    parameters_schema={
        "type": "object",
        "properties": {
            "key_prefix": {
                "type": "string",
                "description": (
                    "Wide-end (left) prefix of the sparse-key path to "
                    "match (e.g. 'Film|Corporate Mistletoe|Script'). "
                    "Empty matches all keys (pair with subject_contains)."
                ),
                "default": "",
            },
            "subject_contains": {
                "type": "string",
                "description": (
                    "Optional substring that must appear in ANY segment "
                    "of the key. An empty key_prefix with this set scans "
                    "all keys for the substring (enter-anywhere)."
                ),
                "default": "",
            },
        },
    },
    cognition_action_alias="crystal_key_scan",
    returns_description=(
        "{'key_prefix': str, 'subject_contains': str, 'results_count': "
        "int, 'findings': [{'fact_id','crystal_id','key','pair_type',"
        "'content_preview','content_length'}], 'content_text': str, "
        "'matched_fact_ids': [str], 'matched_crystal_ids': [str], "
        "'fact_count': int}"
    ),
)
async def key_scan(
    customer_id: str,
    key_prefix: str = "",
    subject_contains: str = "",
) -> dict[str, Any]:
    state = _get_state()
    store = state["store"]

    # A scan needs at least one filter (mirrors the v1 worker guard):
    # either a key prefix OR a subject substring. Returning an empty,
    # well-formed result (rather than raising) keeps the dispatcher's
    # downstream shape stable.
    if not key_prefix and not subject_contains:
        return {
            "key_prefix": key_prefix,
            "subject_contains": subject_contains,
            "results_count": 0,
            "findings": [],
            "content_text": "",
            "matched_fact_ids": [],
            "matched_crystal_ids": [],
            "fact_count": 0,
            "error": "key_scan needs a key_prefix or subject_contains",
        }

    facts = await store.list_facts_by_key_prefix(
        customer_id,
        key_prefix=key_prefix,
        subject_contains=subject_contains or None,
    )

    findings = []
    for fact in facts:
        content = fact.claim_text or fact.answer_value or ""
        findings.append({
            "fact_id": fact.id,
            "crystal_id": fact.crystal_id,
            "key": fact.prompt_text or "",
            "pair_type": fact.pair_type or "",
            "content_preview": content[:300],
            "content_length": len(content),
        })

    key_list = "\n".join(f"- {f['key']}" for f in findings)
    content_text = (
        f"Key prefix scan for '{key_prefix}' found {len(findings)} facts:\n"
        f"{key_list}"
    )

    return {
        "key_prefix": key_prefix,
        "subject_contains": subject_contains,
        "results_count": len(findings),
        "findings": findings,
        "content_text": content_text,
        "matched_fact_ids": [f["fact_id"] for f in findings],
        "matched_crystal_ids": list({f["crystal_id"] for f in findings}),
        "fact_count": len(findings),
    }
