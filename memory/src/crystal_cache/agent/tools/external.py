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
# web_fetch paging cache (2026-07-25)
# ---------------------------------------------------------------------------
# Extracted text, keyed by requested URL. Exists ONLY so that walking a long
# document costs one download and one parse instead of one per page —
# pdfplumber on a full HTS chapter is slow enough that stateless paging would
# make reading page four cost four parses. Deliberately tiny and short-lived:
# this is a paging aid, not a content cache, and staleness within one research
# turn is the point.
_HTML_CAP_CHARS = 12_000
_PDF_CAP_CHARS = 40_000
_PAGE_CACHE_MAX = 8
_PAGE_CACHE_TTL_SECONDS = 600.0
_PAGE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def _page_cache_get(url: str) -> Optional[dict[str, Any]]:
    import time

    hit = _PAGE_CACHE.get(url)
    if hit is None:
        return None
    expires_at, page = hit
    if time.monotonic() >= expires_at:
        _PAGE_CACHE.pop(url, None)
        return None
    return page


def _page_cache_put(url: str, page: dict[str, Any]) -> None:
    import time

    if len(_PAGE_CACHE) >= _PAGE_CACHE_MAX:
        oldest = min(_PAGE_CACHE, key=lambda k: _PAGE_CACHE[k][0])
        _PAGE_CACHE.pop(oldest, None)
    _PAGE_CACHE[url] = (time.monotonic() + _PAGE_CACHE_TTL_SECONDS, page)


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
    name="web_fetch",
    description=(
        "Fetch a specific URL and return its extracted main text. Use "
        "when the user names a site or page to visit (e.g. 'go to "
        "example.com and tell me...') or to read a promising URL from "
        "web_search results in full. Reads HTML and PDFs — use it on "
        "official PDF sources (regulations, tariff schedules, filings, "
        "notices) rather than citing a URL you only saw in a search "
        "snippet: a citation you did not read is not a verified one. "
        "Long documents come back one window at a time; when the result "
        "carries next_offset, call again with that offset to continue "
        "reading, and keep going until you have the section you need. "
        "Only public http/https URLs — private and internal addresses "
        "are refused by the SSRF guard."
    ),
    contexts={"agent", "cognition"},
    parameters_schema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": (
                    "Absolute http(s) URL to fetch. A bare domain like "
                    "'example.com' is accepted and treated as https."
                ),
            },
            "offset": {
                "type": "integer",
                "description": (
                    "Character offset to start reading from. Omit for the "
                    "first window; pass the next_offset from the previous "
                    "result to continue through a long document."
                ),
                "default": 0,
            },
        },
        "required": ["url"],
    },
    returns_description=(
        "{'url': final_url, 'title': str, 'content': str, 'total_chars': "
        "int, 'next_offset': int | None, 'truncated': bool} on success; "
        "{'error': str, 'url': str} on guard refusal or fetch failure"
    ),
)
async def web_fetch(
    customer_id: str,
    url: str,
    offset: int = 0,
) -> dict[str, Any]:
    """Visit one URL (2026-07-07 — the browsing half of the search+fetch
    pair; web_search discovers, web_fetch reads). Rides the SAME
    SSRF-guarded fetcher as result enrichment (search/fetch.py): scheme
    allowlist, full resolved-address-set public check, per-hop redirect
    re-guarding, pinned connect (B6), size cap, textual-or-PDF only.

    Paging (2026-07-25): PDFs of primary sources routinely exceed any
    sane single-response budget, so the caller gets a window plus
    total_chars / next_offset and can walk the document. Extracted text
    is cached briefly so paging costs one download and one parse, not
    one per page.
    """
    import asyncio

    from ...search.fetch import FetchGuardError, fetch_and_extract

    target = (url or "").strip()
    if not target:
        return {"error": "url is required", "url": url}
    if not target.lower().startswith(("http://", "https://")):
        target = f"https://{target}"
    try:
        offset = max(0, int(offset))
    except (TypeError, ValueError):
        return {"error": "offset must be a non-negative integer",
                "url": target}

    page = _page_cache_get(target)
    if page is None:
        try:
            page = await asyncio.to_thread(fetch_and_extract, target)
        except FetchGuardError as e:
            return {"error": f"refused: {e}", "url": target}
        except Exception as e:  # noqa: BLE001 — transport errors -> tool error
            return {"error": f"fetch failed: {e}", "url": target}
        _page_cache_put(target, page)

    full = page.get("content") or ""
    total = len(full)
    cap = _PDF_CAP_CHARS if page.get("kind") == "pdf" else _HTML_CAP_CHARS
    base = {"url": page.get("url", target), "title": page.get("title", ""),
            "total_chars": total}

    if total and offset >= total:
        return {**base, "content": "", "next_offset": None,
                "truncated": False,
                "note": (f"offset {offset} is past the end of this "
                         f"document ({total} chars)")}

    window = full[offset:offset + cap]
    end = offset + len(window)
    more = end < total
    return {**base, "content": window,
            "next_offset": end if more else None,
            "truncated": more}


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
        "Submit content for chunking and crystallization — THE learn "
        "path for anything bigger than one atomic fact. Use when the "
        "user provides a document, says 'learn this' / 'remember "
        "this' about substantive content, or when you want the bank "
        "to absorb a fetched page or produced report. The worker "
        "chunks it and extracts individual knowledge items (facts "
        "stay individually retrievable; the full context is kept as "
        "chunks). Returns the upload id immediately; extraction is "
        "asynchronous — say so honestly rather than claiming facts "
        "already exist."
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
