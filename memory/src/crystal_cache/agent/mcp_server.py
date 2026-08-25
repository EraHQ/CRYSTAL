"""HTTP MCP server — the external memory tool surface (WS C).

Exposes Crystal Cache's memory operations to ANY MCP-speaking client over
an in-process HTTP endpoint mounted on the main FastAPI app at /mcp
(option A — no separate process). Built on the official `mcp` SDK's
FastMCP for spec-conformance and broad client compatibility, rather than
a hand-rolled JSON-RPC layer.

MODE-AGNOSTIC SURFACE: memory is one product; a coding agent is just one
possible caller. The tools are named `memory_*` and described WITHOUT
assuming the caller writes code. This module is the EXTERNAL product
surface; the internal agent tool registry keeps its own names
(knowledge_search, crystal_write, ...). The bridge maps each memory_*
tool to its registry implementation, so there is one source of truth for
behavior and two views over it (the agent loop and this server).

THIS FILE (step 1 of the WS C build): the read/store bridge —
memory_search, memory_search_documents, memory_outline, memory_keys,
memory_synthesize, memory_recall, memory_store. The remaining tools
(forget / ingest / learn / stats / list / export / import / conflicts /
gaps) land in subsequent steps.

CUSTOMER IDENTITY (P0.23 — never trust the model): the customer is
resolved from the Crystal Cache customer API key on the HTTP request
(Bearer), exactly as ingress/auth.py::require_customer does
(store.get_customer_by_api_key hashes + looks up — no plaintext key is
ever stored). An ASGI middleware does the resolution once per request and
stashes the customer_id in a contextvar the tools read. Tool *arguments*
never carry identity.

INTEGRATION (proven end-to-end against mcp 1.28.1):
- FastMCP(stateless_http=True, json_response=True) → a plain
  request/response HTTP MCP server suitable for in-process mounting.
- build_mcp_asgi_app() returns the auth-wrapped Starlette sub-app, which
  app.py mounts at /mcp.
- A mounted sub-app's lifespan does NOT fire on its own, so app.py MUST
  wrap its own lifespan yield with `mcp.session_manager.run()` (see the
  lifespan in app.py). Without it every call 500s with "task group is not
  initialized."

NOTE ON ANNOTATIONS: this module intentionally does NOT use
`from __future__ import annotations`. FastMCP reflects each tool's JSON
schema from its real, runtime type hints; deferred (string) annotations
are an unnecessary risk for a schema-reflected surface.
"""

import contextvars
import json
from typing import Any, Optional

import structlog
from mcp.server.fastmcp import FastMCP

from ..infrastructure.metadata_store import get_metadata_store
from .principal import (
    get_current_operator,
    reset_current_operator,
    set_current_operator,
)
from .tool_registry import get_registry, import_all_tools
from .tools.retrievers import _get_state

logger = structlog.get_logger(__name__)

# Populate the registry so the bridge can resolve tool implementations.
# Idempotent at the import level (re-importing tool modules is a no-op).
import_all_tools()


# ---------------------------------------------------------------------------
# Customer identity (set by middleware from the API key; never from args)
# ---------------------------------------------------------------------------

_current_customer_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "mcp_current_customer_id", default=None
)


def _customer_id() -> str:
    """Return the request's authenticated customer_id, or raise.

    The auth middleware sets this from the Bearer key before any tool
    runs; a missing value means a tool was reached without auth, which
    the middleware is supposed to make impossible — so this is a hard
    error, not a silent fallback.
    """
    cid = _current_customer_id.get()
    if not cid:
        raise RuntimeError(
            "No authenticated customer in context. The MCP auth middleware "
            "must resolve the customer key before any tool runs."
        )
    return cid


async def _dispatch(registry_name: str, **kwargs: Any) -> dict:
    """Call a registry tool's implementation with the authed customer_id.

    The memory_* wrappers below are deliberately thin: they exist to give
    FastMCP a typed signature to reflect a schema from, then delegate to
    the single source of truth (the agent tool registry).
    """
    tool = get_registry().get(registry_name)
    if tool is None:  # pragma: no cover - import_all_tools ran at module load
        raise RuntimeError(
            f"Registry tool {registry_name!r} not found; import_all_tools() "
            f"should have registered it."
        )
    return await tool.impl(_customer_id(), **kwargs)


# ---------------------------------------------------------------------------
# Auth middleware: Bearer customer key -> customer_id contextvar (or 401)
# ---------------------------------------------------------------------------

class _CustomerKeyAuthMiddleware:
    """ASGI middleware resolving the full Crystal Cache principal per request.

    Phase 1.4 (Q1=A + Q4=C, ratified 2026-08-24) — mirrors
    ingress/auth.py::resolve_principal: the Bearer token may be an
    OPERATOR key (tried FIRST, so an operator never falls through to the
    team path) or a team customer key (which acts as the Default Admin
    per P1). Suspended operators get 403; anything unresolvable gets 401.
    The customer_id and the acting operator land in contextvars the tools
    read — identity never rides tool arguments (P0.23). Non-HTTP scopes
    (e.g. lifespan) pass through untouched.

    Exception posture unchanged (S3-59 noted, not worsened): any store
    failure during resolution still surfaces as a 401.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        token = _bearer_from_scope(scope)
        if not token:
            await _send_401(send, "Missing or malformed Authorization header")
            return

        customer = None
        operator = None
        presented_operator_key = False
        try:
            store = get_metadata_store()
            # Operator keys FIRST (resolve_principal's order), so an
            # operator never falls through to the team path.
            operator = await store.get_operator_by_api_key(token)
            if operator is not None:
                presented_operator_key = True
                customer = await store.get_customer_by_id(operator.team_id)
            else:
                customer = await store.get_customer_by_api_key(token)
                if customer is not None:
                    # P1 identity chain: the team key ACTS AS the Default
                    # Admin, so every request has an acting operator and
                    # every write can stamp an owner.
                    operator = await store.ensure_default_admin(customer.id)
        except Exception:  # noqa: BLE001 - any resolution failure is an auth failure here
            logger.warning("mcp.auth.resolve_failed", exc_info=True)
            customer = None
            operator = None
            presented_operator_key = False

        if presented_operator_key and operator.status != "active":
            # Same boundary semantics as require_operator/resolve_principal:
            # the row survives a suspension; THIS door is what denies it —
            # 'suspended' stays distinguishable from 'unknown key'. Only
            # checked for presented operator keys (the team-key path never
            # status-checks its Default Admin, mirroring resolve_principal).
            await _send_auth_error(send, 403, "Operator is suspended")
            return
        if customer is None:
            await _send_401(send, "Invalid api_key")
            return

        tok = _current_customer_id.set(customer.id)
        tok_op = set_current_operator(operator)
        try:
            await self.app(scope, receive, send)
        finally:
            reset_current_operator(tok_op)
            _current_customer_id.reset(tok)


def _bearer_from_scope(scope: dict) -> str:
    """Extract a Bearer token from the ASGI scope headers, or '' if absent."""
    for raw_name, raw_value in scope.get("headers", []):
        if raw_name == b"authorization":
            value = raw_value.decode("latin-1")
            parts = value.split(None, 1)
            if len(parts) == 2 and parts[0].lower() == "bearer":
                return parts[1].strip()
            return ""
    return ""


async def _send_auth_error(send: Any, status_code: int, detail: str) -> None:
    """Emit a minimal JSON auth-error response over ASGI (401 or 403).
    WWW-Authenticate rides only the 401 — a 403 is a recognized-but-denied
    credential, so a challenge header would be wrong there."""
    error = "unauthorized" if status_code == 401 else "forbidden"
    body = json.dumps({"error": error, "detail": detail}).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    if status_code == 401:
        headers.append((b"www-authenticate", b"Bearer"))
    await send({
        "type": "http.response.start",
        "status": status_code,
        "headers": headers,
    })
    await send({"type": "http.response.body", "body": body})


async def _send_401(send: Any, detail: str) -> None:
    """Emit a minimal JSON 401 response over ASGI."""
    await _send_auth_error(send, 401, detail)


def _viewer_write_block() -> Optional[dict]:
    """Q4=C (ratified 2026-08-24): viewers get the read tools; every
    mutating memory_* tool refuses them. Enforced at tool grain because
    JSON-RPC has no per-tool HTTP status — the structured error result
    IS this surface's 403. Returns the error dict to hand back, or None
    when the caller may write (any non-viewer operator, incl. the
    Default Admin a team key resolves to; None = system lane)."""
    op = get_current_operator()
    if op is not None and getattr(op, "role", None) == "viewer":
        return {
            "error": "operator role 'viewer' cannot write to memory",
            "code": "viewer_forbidden",
        }
    return None


# ---------------------------------------------------------------------------
# FastMCP server + the memory_* tool surface
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="crystal-cache",
    instructions=(
        "Crystal Cache memory tools. Search, recall, and store knowledge in "
        "a self-curating memory bank scoped to your account. Your identity "
        "is taken from your API key, never from tool arguments."
    ),
    stateless_http=True,
    json_response=True,
    streamable_http_path="/",
)


@mcp.tool(
    name="memory_search",
    description=(
        "Search the memory bank for relevant knowledge (facts, entities, "
        "Q&A pairs, relationships) and return the top matches as small, "
        "structured pieces with their keys and values. Use for 'what do we "
        "know about X', 'find the answer to Y', 'look up Z'. For verbatim "
        "passages from ingested documents use memory_search_documents "
        "instead; for counting or listing use memory_outline."
    ),
)
async def memory_search(
    query: str,
    k: int = 10,
    hints: Optional[dict] = None,
) -> dict:
    return await _dispatch("knowledge_search", query=query, k=k, hints=hints)


@mcp.tool(
    name="memory_search_documents",
    description=(
        "Search for verbatim chunks of ingested documents matching a query "
        "and return their raw text plus locators. Use for 'what does the "
        "document say about X' or 'find the passage about Y'. For small "
        "structured facts rather than document text, use memory_search."
    ),
)
async def memory_search_documents(
    query: str,
    k: int = 5,
    hints: Optional[dict] = None,
) -> dict:
    return await _dispatch("content_search", query=query, k=k, hints=hints)


@mcp.tool(
    name="memory_outline",
    description=(
        "Summarize what the memory bank knows about a subject by scanning "
        "its key structure — counts, listings, and gaps (e.g. 'items 1-4 "
        "and 6-10 exist but not 5'). No semantic search. Use for 'how "
        "many...', 'what ... exist', 'list all ...', and other structural "
        "questions. Narrow the scan with hints (subject, domain, "
        "locator_prefix)."
    ),
)
async def memory_outline(
    query_text: str = "",
    hints: Optional[dict] = None,
) -> dict:
    return await _dispatch("navigation_search", query_text=query_text, hints=hints)


@mcp.tool(
    name="memory_keys",
    description=(
        "Enumerate stored items whose hierarchical key matches. 'key_prefix' "
        "matches the wide (left) end of the key path; 'subject_contains' "
        "matches any segment of it. Returns the raw matching items, not a "
        "summary. Use for precise listing or counting and 'where is X "
        "recorded' lookups. Provide at least one of key_prefix or "
        "subject_contains."
    ),
)
async def memory_keys(
    key_prefix: str = "",
    subject_contains: str = "",
) -> dict:
    return await _dispatch(
        "key_scan", key_prefix=key_prefix, subject_contains=subject_contains
    )


@mcp.tool(
    name="memory_synthesize",
    description=(
        "Cross-item analytical synthesis: gathers related knowledge about "
        "the subjects in the query, organizes it, and — when the server has "
        "an LLM configured — returns a synthesized summary rather than raw "
        "hits. Use for 'how does X relate to Y', 'compare X and Y', or "
        "'what's the throughline for X'. Heavier than memory_search; prefer "
        "it only when you actually need synthesis."
    ),
)
async def memory_synthesize(
    query: str,
    k: int = 20,
    hints: Optional[dict] = None,
) -> dict:
    # memory_synthesize exists to RETURN a synthesized summary, so it opts into
    # the underlying router's "deep" mode by default — that is what triggers the
    # LLM pre-digest (when the server has a key). The agent's own depth tool
    # defaults to shallow to save cost; here the caller explicitly chose
    # synthesis, so deep is the right default. An explicit hint still wins, so
    # a caller can pass {"depth_mode": "shallow"} for organized-but-unsummarized
    # output.
    merged = dict(hints or {})
    merged.setdefault("depth_mode", "deep")
    return await _dispatch("depth_search", query=query, k=k, hints=merged)


@mcp.tool(
    name="memory_recall",
    description=(
        "Convenience 'what do we know about X' lookup that returns the top "
        "matching items grouped by the entity they belong to. A simple entry "
        "point for when you don't want to choose among the more specific "
        "search tools."
    ),
)
async def memory_recall(
    query: str,
    k: int = 10,
) -> dict:
    return await _dispatch("crystal_recall", query=query, k=k)


@mcp.tool(
    name="memory_store",
    description=(
        "Store a (key, value) pair in the memory bank for future recall. Use "
        "when you have produced or confirmed knowledge worth retaining. "
        "pair_type defaults to 'question_answer'; set answer_value to make a "
        "key eligible for cache-hit short-circuiting on future lookups. The "
        "pair is stored under the deployment's default visibility; pass "
        "scope='personal' (only you and admins can retrieve it) or "
        "scope='team' (the whole team can) to override for this write."
    ),
)
async def memory_store(
    key: str,
    value: str,
    pair_type: str = "question_answer",
    crystal_type: str = "customer:legacy",
    source_kind: str = "model_reasoning",
    answer_value: Optional[str] = None,
    scope: Optional[str] = None,
) -> dict:
    denied = _viewer_write_block()
    if denied:
        return denied
    return await _dispatch(
        "crystal_write",
        key=key,
        value=value,
        pair_type=pair_type,
        crystal_type=crystal_type,
        source_kind=source_kind,
        answer_value=answer_value,
        scope=scope,
    )


# ---------------------------------------------------------------------------
# MCP-layer-only memory tools (WS C step 2): forget / ingest / learn
# ---------------------------------------------------------------------------
# These wrap store methods + services directly (not the agent registry) so that
# building the external memory surface does NOT change the agent's own tool
# surface. learn (and, later, conflicts/gaps) are ALSO promoted into the
# registry separately so the agent gets them; that promotion calls the same
# underlying service.


@mcp.tool(
    name="memory_forget",
    description=(
        "Permanently delete stored knowledge. Provide exactly ONE of: "
        "crystal_id (removes a whole memory cluster and all of its facts) or "
        "fact_id (removes a single fact and rebuilds its cluster from the "
        "survivors). IDs come from memory_search / memory_list / memory_keys "
        "results. This cannot be undone."
    ),
)
async def memory_forget(
    crystal_id: Optional[str] = None,
    fact_id: Optional[str] = None,
) -> dict:
    denied = _viewer_write_block()
    if denied:
        return denied
    if bool(crystal_id) == bool(fact_id):
        return {
            "deleted": False,
            "error": "provide exactly one of crystal_id or fact_id",
        }
    state = _get_state()
    store = state["store"]
    cid = _customer_id()
    if crystal_id:
        deleted = await store.delete_crystal(
            crystal_id,
            cid,
            vector_store=state["vector_store"],
            fact_vector_store=state.get("fact_vector_store"),
        )
        return {"deleted": bool(deleted), "crystal_id": crystal_id}
    deleted = await store.delete_fact(
        fact_id,
        cid,
        encoder=state["encoder"],
        vector_store=state["vector_store"],
        fact_vector_store=state.get("fact_vector_store"),
    )
    return {"deleted": bool(deleted), "fact_id": fact_id}


@mcp.tool(
    name="memory_ingest",
    description=(
        "Ingest a document's text into memory and make it searchable in one "
        "call. Chunks the text, extracts knowledge, and writes it to the bank "
        "synchronously (no separate human-approval step), then returns counts. "
        "Use for 'remember this document / note / page'. Inputs over the "
        "deployment's character ceiling (default 200,000) are refused — send "
        "very large documents through the async upload endpoint instead."
    ),
)
async def memory_ingest(
    text: str,
    label: str = "Untitled",
    crystal_type: str = "customer:legacy",
) -> dict:
    denied = _viewer_write_block()
    if denied:
        return denied
    if not text.strip():
        return {"crystals_written": 0, "error": "text is required"}

    # Audit item (d) (Q2=A, ratified 2026-08-25): synchronous ingest runs
    # one small-tier extraction call per chunk, so unbounded text was
    # unbounded model spend — metered on the ledger but never refused.
    # Refuse BEFORE any row is written or any model is called.
    from ..config import get_settings
    _max_chars = get_settings().mcp_ingest_max_chars
    if _max_chars and len(text) > _max_chars:
        return {
            "crystals_written": 0,
            "error": (
                f"text is {len(text):,} characters — over this deployment's "
                f"{_max_chars:,}-character synchronous-ingest ceiling. Send "
                "large documents through the async upload endpoint "
                "(POST /v1/documents), which chunks and reviews them out of "
                "band."
            ),
            "code": "ingest_too_large",
            "max_chars": _max_chars,
            "received_chars": len(text),
        }

    state = _get_state()
    store = state["store"]
    cid = _customer_id()

    # 1. Create the upload row (pending).
    doc = await store.create_document_upload(
        customer_id=cid, label=label, text=text, crystal_type=crystal_type,
    )

    # 2. Chunk + extract -> status 'review' (nothing in the bank yet). Same
    #    workflow the manual /crystallize endpoint runs; client=None matches
    #    the background worker (code-description enrichment stays off here).
    from ..workers.crystallization import crystallize_document
    await crystallize_document(
        store=store,
        encoder=state["encoder"],
        vector_store=state["vector_store"],
        document_id=doc.id,
    )

    doc2 = await store.get_document_upload(doc.id, cid)
    if doc2 is None:
        return {"document_id": doc.id, "status": "unknown", "crystals_written": 0}
    if doc2.status == "error":
        return {
            "document_id": doc.id,
            "status": "error",
            "error": doc2.error_message,
            "crystals_written": 0,
        }

    # 3. Auto-approve -> write the chunks + extracted items to the bank. A
    #    programmatic ingest skips the human review gate by design (the caller
    #    asked to remember this now), so we approve whatever was extracted.
    from datetime import datetime, timezone
    from ..ingestion.document_pipeline import DocumentPipeline
    pipeline = DocumentPipeline(
        store=store,
        encoder=state["encoder"],
        vector_store=state["vector_store"],
        vector_index=state.get("vector_index"),
        fact_vector_store=state.get("fact_vector_store"),
    )
    try:
        # Recall-gate birth attribution (2026-07-03): inferred_knowledge
        # documents (cognition/background-worker output) birth recall_gated
        # crystals; user/agent-uploaded docs remain born usable.
        _origin = (
            "background_worker"
            if doc2.detected_type == "inferred_knowledge"
            else "direct"
        )
        result = await pipeline.approve_and_crystallize(
            customer_id=cid,
            document_id=doc.id,
            items=doc2.extracted_items or [],
            content_chunks=doc2.content_chunks or [],
            crystal_type=doc2.confirmed_type or doc2.crystal_type,
            scope=doc2.scope,
            owner_operator_id=doc2.owner_operator_id,
            origin=_origin,
        )
        await store.mark_document_crystallized(
            document_id=doc.id,
            crystals_written=result.crystals_written,
            items_extracted=result.items_extracted,
            crystallized_at=datetime.now(timezone.utc),
        )
        # Share-source provenance (P4): persist the pipeline-stamped item
        # crystal ids so the document knows its crystal set.
        await store.update_document_review_edits(
            doc.id, cid, extracted_items=doc2.extracted_items or [],
        )
    except Exception as e:  # noqa: BLE001 - report failure to the caller, don't 500
        await store.mark_document_error(doc.id, str(e))
        logger.warning("mcp.memory_ingest.crystallize_failed", error=str(e))
        return {
            "document_id": doc.id,
            "status": "error",
            "error": str(e),
            "crystals_written": 0,
        }

    return {
        "document_id": doc.id,
        "status": "crystallized",
        "crystals_written": result.crystals_written,
        "items_extracted": result.items_extracted,
        "errors": result.errors,
    }


@mcp.tool(
    name="memory_learn",
    description=(
        "Teach memory from an outcome. outcome='success' caches a "
        "prompt -> solution pair for fast future recall; outcome='fail' "
        "records a correction (pass 'signal' describing what was wrong) so the "
        "system learns from the mistake. Use after you find out whether a past "
        "answer was right or wrong."
    ),
)
async def memory_learn(
    prompt: str,
    response: str,
    outcome: str = "success",
    signal: Optional[str] = None,
    crystal_type: str = "customer:legacy",
) -> dict:
    denied = _viewer_write_block()
    if denied:
        return denied
    # Promoted into the agent registry as crystal_learn (WS C step 4), so this
    # bridges to it like the other registry-backed tools — one implementation.
    return await _dispatch(
        "crystal_learn",
        prompt=prompt,
        response=response,
        outcome=outcome,
        signal=signal,
        crystal_type=crystal_type,
    )


# ---------------------------------------------------------------------------
# MCP-layer-only memory tools (WS C step 3): stats / list / export / import
# ---------------------------------------------------------------------------


@mcp.tool(
    name="memory_stats",
    description=(
        "Return summary statistics for the memory bank: total clusters and "
        "facts, distributions by cluster type / quality tier / source / fact "
        "type, and how many entries are cache-hit eligible. Use to gauge how "
        "much is stored and of what kind."
    ),
)
async def memory_stats() -> dict:
    from collections import Counter

    state = _get_state()
    store = state["store"]
    cid = _customer_id()

    crystal_count = await store.count_crystals_for_customer(cid)
    crystals = await store.list_crystals_for_customer(cid)

    quality_dist: Counter = Counter()
    type_dist: Counter = Counter()
    source_dist: Counter = Counter()
    cache_hit_eligible = 0
    for c in crystals:
        if c.quality_tier:
            quality_dist[c.quality_tier] += 1
        if c.crystal_type:
            type_dist[c.crystal_type] += 1
        if c.source_kind:
            source_dist[c.source_kind] += 1
        if c.answer_value:
            cache_hit_eligible += 1

    pair_type_dist: Counter = Counter()
    total_facts = 0
    for c in crystals:
        facts = await store.list_facts_for_crystal(c.id)
        total_facts += len(facts)
        for f in facts:
            if f.pair_type:
                pair_type_dist[f.pair_type] += 1

    total_query_logs, _ = await store.list_query_logs_for_customer(
        customer_id=cid, limit=1, offset=0,
    )

    return {
        "crystal_count": crystal_count,
        "fact_count": total_facts,
        "quality_distribution": dict(quality_dist),
        "crystal_type_distribution": dict(type_dist),
        "pair_type_distribution": dict(pair_type_dist),
        "source_kind_distribution": dict(source_dist),
        "cache_hit_eligible": cache_hit_eligible,
        "total_query_logs": total_query_logs,
    }


@mcp.tool(
    name="memory_list",
    description=(
        "Browse stored memory. With no arguments, returns a paginated list of "
        "clusters (id, type, summary, fact count). Pass crystal_id to get one "
        "cluster's full detail and its facts. Distinct from memory_search: "
        "this lists / inspects rather than ranking by relevance."
    ),
)
async def memory_list(
    crystal_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    state = _get_state()
    store = state["store"]
    cid = _customer_id()

    if crystal_id:
        crystal = await store.get_crystal(crystal_id)
        if crystal is None or crystal.customer_id != cid:
            return {"error": "crystal not found", "crystal_id": crystal_id}
        facts = await store.list_facts_for_crystal(crystal_id)
        return {
            "crystal": {
                "id": crystal.id,
                "crystal_type": crystal.crystal_type,
                "summary_text": crystal.summary_text,
                "fact_count": crystal.fact_count,
                "created_at": crystal.created_at.isoformat(),
            },
            "facts": [
                {
                    "id": f.id,
                    "claim_text": f.claim_text,
                    "pair_type": f.pair_type,
                    "prompt_text": f.prompt_text,
                }
                for f in facts
            ],
        }

    total, crystals = await store.list_crystals_for_customer_paginated(
        customer_id=cid, limit=limit, offset=offset,
    )
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "crystals": [
            {
                "id": c.id,
                "crystal_type": c.crystal_type,
                "summary_text": c.summary_text,
                "fact_count": c.fact_count,
                "quality_tier": c.quality_tier,
                "created_at": c.created_at.isoformat(),
            }
            for c in crystals
        ],
    }


@mcp.tool(
    name="memory_export",
    description=(
        "Export the memory bank as fact-level records "
        "{key, value, pair_type, source_kind, answer_value, crystal_type} — "
        "the inverse of memory_import. Use for backup or moving a bank to "
        "another account. PAGINATED: returns up to `limit` records per call "
        "(default 1000) in a stable order; walk pages by advancing `offset` "
        "until has_more is false. Each page's records import cleanly on "
        "their own."
    ),
)
async def memory_export(limit: int = 1000, offset: int = 0) -> dict:
    state = _get_state()
    store = state["store"]
    cid = _customer_id()

    # Audit item (d) (Q3=A, ratified 2026-08-25): the whole-bank single
    # response is gone — fact-grain pagination over a deterministic order
    # (the store method's (created_at, id) sort), with the crystal fields
    # joined via a per-call cache so a page costs one query plus one
    # get_crystal per DISTINCT crystal on the page.
    limit = max(1, min(int(limit), 1000))
    offset = max(0, int(offset))
    total, facts = await store.list_facts_for_customer_paginated(
        cid, limit=limit, offset=offset,
    )
    crystal_cache: dict = {}
    records: list = []
    for f in facts:
        c = crystal_cache.get(f.crystal_id)
        if c is None:
            c = await store.get_crystal(f.crystal_id)
            crystal_cache[f.crystal_id] = c
        records.append({
            "key": f.prompt_text,
            "value": f.claim_text,
            "pair_type": f.pair_type,
            "source_kind": c.source_kind if c else None,
            "answer_value": c.answer_value if c else None,
            "crystal_type": c.crystal_type if c else None,
        })
    return {
        "record_count": len(records),
        "total_records": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(records) < total,
        "export_format": "jsonl",
        "data": records,
    }


@mcp.tool(
    name="memory_import",
    description=(
        "Import fact-level records (the shape memory_export produces) into the "
        "bank. Each record's key is re-sparsified and re-indexed via the same "
        "path as memory_store. Set wipe=true to replace the existing bank "
        "first. Per-record failures are counted, not fatal. Note: fact-faithful, "
        "not cluster-topology-exact."
    ),
)
async def memory_import(
    records: list,
    wipe: bool = False,
    crystal_type: str = "customer:legacy",
) -> dict:
    denied = _viewer_write_block()
    if denied:
        return denied
    from ..encoding.sparse_keys import generate_sparse_key_metered

    state = _get_state()
    store = state["store"]
    cid = _customer_id()
    encoder = state["encoder"]
    vector_store = state["vector_store"]
    vector_index = state.get("vector_index")
    fact_vector_store = state.get("fact_vector_store")

    if wipe:
        existing = await store.list_crystals_for_customer(cid)
        for c in existing:
            try:
                await store.delete_crystal(
                    c.id, cid,
                    vector_store=vector_store,
                    fact_vector_store=fact_vector_store,
                )
            except Exception as e:  # noqa: BLE001 - one bad delete can't abort the wipe
                logger.warning("mcp.memory_import.wipe_failed", crystal_id=c.id, error=str(e))

    processed = 0
    errors = 0
    seen: set = set()
    for rec in records:
        try:
            key = (rec.get("key") or "").strip()
            value = rec.get("value") or ""
            if not key or not value:
                errors += 1
                continue
            sparse_key = await generate_sparse_key_metered(
                key, customer_id=cid, store=store,
            )
            crystal, _fact = await store.add_pair_for_customer(
                customer_id=cid,
                prompt_text=sparse_key,
                answer_text=value,
                pair_type=rec.get("pair_type") or "question_answer",
                encoder=encoder,
                vector_store=vector_store,
                vector_index=vector_index,
                crystal_type=rec.get("crystal_type") or crystal_type or "customer:legacy",
                source_kind=rec.get("source_kind") or "model_reasoning",
                answer_value=rec.get("answer_value"),
            )
            processed += 1
            seen.add(crystal.id)
        except Exception as e:  # noqa: BLE001 - per-record failure is counted, never fatal
            errors += 1
            logger.warning("mcp.memory_import.record_failed", error=str(e))

    return {
        "records_processed": processed,
        "crystals_written": len(seen),
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# MCP-layer memory tools (WS C step 4): conflicts / gaps
# ---------------------------------------------------------------------------
# These bridge to the curation tools that step 4 promoted into the agent
# registry (so the agent gets them too) — one implementation in
# agent/tools/curation.py, two views over it.


@mcp.tool(
    name="memory_conflicts",
    description=(
        "List contradictions the system has detected in its own memory — pairs "
        "of stored facts that can't both be true, with the conflicting claims. "
        "Use to see where the memory disagrees with itself before trusting it."
    ),
)
async def memory_conflicts(status: str = "open", limit: int = 50) -> dict:
    return await _dispatch("knowledge_conflicts", status=status, limit=limit)


@mcp.tool(
    name="memory_gaps",
    description=(
        "List gaps the system has identified in its own memory — things it was "
        "asked about or expected to know but doesn't, with a description of "
        "what's missing. Use to see what the memory lacks and might need taught."
    ),
)
async def memory_gaps(status: str = "open", limit: int = 50) -> dict:
    return await _dispatch("knowledge_gaps", status=status, limit=limit)


# 0g (2026-07-25): the write half of the gaps drawer. memory_gaps could show
# what the memory lacks with no way to add to it, so a client that noticed an
# unanswered question had nowhere to put it. resolve_conflict is deliberately
# NOT bridged: its gate is an explicit in-chat human confirmation quoted
# verbatim, and an MCP caller has no human turn to quote - the provenance
# field would be fiction.
@mcp.tool(
    name="memory_record_gap",
    description=(
        "Record a question the memory could not answer. Use when a search "
        "returned results that still did not answer what was asked - gaps are "
        "detected automatically only when retrieval SCORES a miss, so a "
        "near-miss that ranks well goes unrecorded unless you record it. "
        "disposition says who can close it: 'researchable' (findable by "
        "searching), 'workable' (settled by doing it), 'needs_document' (only "
        "the operator has it). A recorded gap is a request, not a task: a "
        "human decides whether it becomes research."
    ),
)
async def memory_record_gap(
    question: str,
    disposition: str,
    context: str = "",
    subject: str = "",
    domain: str = "",
    priority: str = "medium",
) -> dict:
    denied = _viewer_write_block()
    if denied:
        return denied
    return await _dispatch(
        "record_gap",
        question=question,
        disposition=disposition,
        context=context,
        subject=subject,
        domain=domain,
        priority=priority,
    )


# ---------------------------------------------------------------------------
# ASGI app factory (mounted by app.py at /mcp)
# ---------------------------------------------------------------------------

def build_mcp_asgi_app() -> Any:
    """Return the auth-wrapped streamable-HTTP MCP app for mounting.

    Calling streamable_http_app() here also creates mcp.session_manager,
    which app.py's lifespan enters via `async with mcp.session_manager.run()`.
    Call this exactly once, at mount time.
    """
    return _CustomerKeyAuthMiddleware(mcp.streamable_http_app())
