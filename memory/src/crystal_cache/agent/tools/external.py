"""External tools — web_search, document_upload, decompose.

Per §4.6: tools that reach outside the crystal bank. Three of them
matter for Phase 7.5:

- web_search: placeholder for now. Cognition's v1 worker had a
  stub for this; we expose it as a first-class agent tool with the
  same stub behavior, so the agent can declare the intent and the
  Phase 8+ work fills in the real search backend.

- document_upload: lets the agent route a customer-supplied
  document through the chunking + crystallization pipeline. Wraps
  the same flow as the /v1/documents/upload HTTP endpoint.

- decompose: converts free text to structured intent. Wraps the
  Decomposer protocol. Most of the time the agent's own reasoning
  replaces this, but the tool exists for cases where the agent
  wants to hand structured intent to a downstream consumer
  (e.g. a customer's app via MCP).

CONTEXT ASSIGNMENTS:
- web_search and decompose are read-side shared (agent ✅, cognition ✅).
- document_upload is write-side agent-only (cognition workers don't
  ingest customer documents).
"""
from __future__ import annotations

from typing import Any, Optional

import structlog

from ..tool_registry import register_tool
from .retrievers import _get_state

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

@register_tool(
    name="web_search",
    description=(
        "Search the web for current or external information. Use when the "
        "answer cannot be found in the crystal bank (always check "
        "crystals first via knowledge_search or crystal_recall). "
        "Requires the operator to configure a search provider "
        "(CC_WEB_SEARCH_PROVIDER); unconfigured calls return an explicit "
        "error result. Depending on the provider, results carry either "
        "snippets only or extracted page content per result."
    ),
    contexts={"agent", "cognition"},
    parameters_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query.",
            },
        },
        "required": ["query"],
    },
    cognition_action_alias="web_search",
    returns_description=(
        "{'query': str, 'provider': str, 'results': [{'title', 'url', "
        "'snippet', 'content'|None}]} on success; {'error': str, 'query': "
        "str, 'results': []} when no provider is configured"
    ),
)
async def web_search(
    customer_id: str,
    query: str,
) -> dict[str, Any]:
    import asyncio

    from ...search import get_web_search_client

    client = get_web_search_client()
    if not client.is_configured():
        logger.info(
            "web_search.unconfigured", customer_id=customer_id, query=query[:80],
        )
        return {
            "error": (
                "web_search is not configured. Set CC_WEB_SEARCH_PROVIDER to "
                "searxng (with CC_WEB_SEARCH_URL) or tavily (with "
                "CC_WEB_SEARCH_API_KEY). Answer from the crystal bank and "
                "your own knowledge instead."
            ),
            "query": query,
            "results": [],
        }

    payload = await asyncio.to_thread(client.search, query)

    # The goldmine's raw side: log the interaction (title/url/snippet only).
    # Fail-safe — a logging hiccup never breaks the search itself.
    try:
        state = _get_state()
        store = state.get("store")
        if store is not None:
            await store.write_web_search_log(
                customer_id,
                query=query,
                provider=payload.get("provider", ""),
                results=payload.get("results", []),
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("web_search.log_failed", error=str(e))

    logger.info(
        "web_search.completed",
        customer_id=customer_id,
        provider=payload.get("provider"),
        n_results=len(payload.get("results", [])),
    )
    return payload


# ---------------------------------------------------------------------------
# source_lookup
# ---------------------------------------------------------------------------

@register_tool(
    name="source_lookup",
    description=(
        "Read ACTUAL source code to ground a claim instead of "
        "reconstructing it from memory. Three ops: 'read' returns a "
        "file's contents, 'list' returns a directory's entries, "
        "'search' finds a string/symbol across files (path + line + "
        "snippet). Use for 'where is X defined', 'what does the code at "
        "path P do', or to verify a path exists before asserting it. "
        "Requires a configured source backend; returns available=false "
        "otherwise (never fabricate paths or code)."
    ),
    contexts={"cognition"},
    parameters_schema={
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "enum": ["read", "list", "search"],
                "description": "Which operation to run.",
            },
            "path": {
                "type": "string",
                "description": (
                    "File path (op=read) or directory path (op=list), "
                    "relative to the source root."
                ),
                "default": "",
            },
            "query": {
                "type": "string",
                "description": "String/symbol to find (op=search).",
                "default": "",
            },
            "path_prefix": {
                "type": "string",
                "description": "Optional path prefix to scope an op=search.",
                "default": "",
            },
        },
        "required": ["op"],
    },
    cognition_action_alias="source_lookup",
    returns_description=(
        "read: {op,backend,path,content,truncated,size} | "
        "list: {op,backend,path,entries:[{name,type,size}]} | "
        "search: {op,backend,query,matches:[{path,line,text}],truncated}. "
        "When no backend is configured: {available: false, error}."
    ),
)
async def source_lookup(
    customer_id: str,
    op: str,
    path: str = "",
    query: str = "",
    path_prefix: str = "",
) -> dict[str, Any]:
    # Lazy imports: source_connector pulls httpx; keep it off the
    # import-time path. The connector can be injected via tool state
    # (tests) or built from settings (normal operation).
    from ...config import settings
    from ...infrastructure.source_connector import build_source_connector

    state = _get_state()
    conn = state.get("source_connector") or build_source_connector(settings)
    if conn is None:
        return {
            "available": False,
            "op": op,
            "error": (
                "no source backend configured "
                "(set CC_SOURCE_BACKEND to local_fs or github)"
            ),
        }

    if op == "read":
        return await conn.read(path)
    if op == "list":
        return await conn.list(path)
    if op == "search":
        return await conn.search(query, path_prefix=path_prefix)
    return {
        "available": True,
        "op": op,
        "error": f"unknown op {op!r} (use read|list|search)",
    }


# ---------------------------------------------------------------------------
# document_upload
# ---------------------------------------------------------------------------

@register_tool(
    name="document_upload",
    description=(
        "Submit a document for chunking and crystallization. The "
        "document lands in the crystallization queue with status "
        "'pending'; the background worker chunks it and extracts "
        "knowledge items. Use this when the user provides a "
        "document they want the agent to learn from. Returns the "
        "document upload id; the worker processes asynchronously, "
        "so the agent does not block on extraction completion."
    ),
    contexts={"agent"},
    parameters_schema={
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "description": (
                    "Human-readable label for the document (filename, "
                    "title, or descriptive name)."
                ),
            },
            "text": {
                "type": "string",
                "description": "The document text. Plain text, markdown, or similar.",
            },
            "crystal_type": {
                "type": "string",
                "description": (
                    "Crystal type id to scope extracted knowledge "
                    "under. Default 'customer:legacy'."
                ),
                "default": "customer:legacy",
            },
            "detected_type": {
                "type": "string",
                "description": (
                    "Optional pre-detected document type "
                    "(e.g. 'screenplay', 'spec', 'report'). When "
                    "omitted, the chunking pipeline detects "
                    "automatically."
                ),
            },
        },
        "required": ["label", "text"],
    },
    returns_description="{'document_id': str, 'status': str, 'label': str}",
)
async def document_upload(
    customer_id: str,
    label: str,
    text: str,
    crystal_type: str = "customer:legacy",
    detected_type: Optional[str] = None,
) -> dict[str, Any]:
    state = _get_state()
    store = state["store"]

    doc = await store.create_document_upload(
        customer_id=customer_id,
        label=label,
        text=text,
        crystal_type=crystal_type,
        detected_type=detected_type or "",
    )
    return {
        "document_id": doc.id,
        "status": doc.status,
        "label": doc.label,
    }


# ---------------------------------------------------------------------------
# decompose
# ---------------------------------------------------------------------------

@register_tool(
    name="decompose",
    description=(
        "Convert free text to a structured intent payload. The "
        "Decomposer runs an LLM call to parse the input into typed "
        "fields (subject, locator, action, etc.). Use this when the "
        "agent needs structured intent to hand to a downstream "
        "consumer (e.g. a customer app via MCP, a concept-path "
        "config). Most of the time the agent's own reasoning "
        "replaces this — call it explicitly when the structured "
        "shape matters for the consumer."
    ),
    contexts={"agent", "cognition"},
    parameters_schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Free text to decompose.",
            },
            "config_id": {
                "type": "string",
                "description": (
                    "Optional DSL config id to apply. When omitted, "
                    "the customer's default decomposer config is "
                    "used. Returns 'decomposer not configured' when "
                    "no GROQ_API_KEY is set."
                ),
            },
        },
        "required": ["text"],
    },
    returns_description=(
        "{'fields': dict, 'config_id': str | None, 'error': str | None}"
    ),
)
async def decompose(
    customer_id: str,
    text: str,
    config_id: Optional[str] = None,
) -> dict[str, Any]:
    state = _get_state()
    decomposer = state.get("decomposer")
    if decomposer is None:
        return {
            "fields": {},
            "config_id": config_id,
            "error": (
                "decomposer not configured "
                "(GROQ_API_KEY/CC_GROQ_API_KEY missing)"
            ),
        }

    # Decomposer protocol takes (text, context) where context carries
    # the tenant id at minimum. Phase 11 may extend the context shape
    # for per-customer config_id resolution; for now we pass the
    # customer id and let the decomposer pick up its default config.
    context = {"tenant_id": customer_id}
    if config_id:
        context["config_id"] = config_id

    try:
        result = await decomposer.decompose(text, context)
    except Exception as e:
        logger.error(
            "decompose.failed",
            customer_id=customer_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return {
            "fields": {},
            "config_id": config_id,
            "error": str(e),
        }

    # The decomposer protocol returns a DecomposeResult-like object
    # with .fields (dict) and .config_id (str). Coerce defensively
    # because customer Decomposer implementations vary.
    if hasattr(result, "fields"):
        fields = result.fields
        used_config = getattr(result, "config_id", config_id)
    elif isinstance(result, dict):
        fields = result.get("fields", {})
        used_config = result.get("config_id", config_id)
    else:
        fields = {}
        used_config = config_id

    return {
        "fields": fields,
        "config_id": used_config,
        "error": None,
    }
