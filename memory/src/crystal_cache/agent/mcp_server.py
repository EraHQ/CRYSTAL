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
import re
from typing import Any, Optional

import structlog
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

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
# Console Refine PR 11 (Q65=A): which tool is calling (User-Agent family
# from the request; stateless HTTP leaves no handshake clientInfo).
_current_client: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "mcp_current_client", default=None
)
# Activity trace (2026-10-19): the query log row a recall-class call just
# wrote, so the tool_invocations row that _tool_boundary records can link
# to it. Set by _log_recall, read and cleared by the boundary.
_last_query_log_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "mcp_last_query_log_id", default=None
)

# Recall-class registry tools: every call writes a query log row so the
# Query log shows what a connected tool asked and which memories
# answered. Writes (memory_store, learn, …) are not questions.
RECALL_TOOLS: frozenset[str] = frozenset({
    "crystal_recall", "knowledge_search", "content_search",
    "navigation_search", "depth_search",
})

# v108 (Q15=A): records per memory_import call. 500 x ~$0.00065 (one
# Haiku sparse-key call each) is about $0.33 per call at the ceiling.
MEMORY_IMPORT_MAX_RECORDS = 500


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
    try:
        import time as _time

        _t0 = _time.monotonic()
        result = await tool.impl(_customer_id(), **kwargs)
        if registry_name in RECALL_TOOLS and isinstance(result, dict) and "error" not in result:
            await _log_recall(registry_name, kwargs, result, int((_time.monotonic() - _t0) * 1000))
        return result
    except Exception as e:  # noqa: BLE001
        # S3 (2026-09-30): an unhandled exception used to reach the
        # customer's AI client as FastMCP's default "Error executing
        # tool: <str(e)>", i.e. raw SDK/httpx/SQL text. The client now
        # gets a fixed message plus a reference; the redacted detail is
        # logged under that reference.
        from ..hygiene import safe_error
        ref, message = safe_error(
            "mcp.tool_failed", e, user_message=f"{registry_name} failed.",
            tool=registry_name, customer_id=_current_customer_id.get(),
        )
        return {"error": message, "code": "tool_failed", "ref": ref}


def recall_summary(registry_name: str, result: dict) -> tuple[list[str], float]:
    """(matched fact ids, top score) from a recall tool's result.
    crystal_recall groups facts under crystals with per-fact scores; the
    V3 router tools return matched_fact_ids + top_score."""
    if registry_name == "crystal_recall":
        ids: list[str] = []
        top = 0.0
        for c in result.get("crystals") or []:
            for f in c.get("facts") or []:
                fid = f.get("fact_id")
                if fid:
                    ids.append(fid)
                try:
                    top = max(top, float(f.get("score") or 0.0))
                except (TypeError, ValueError):
                    pass
        return ids, top
    ids = [i for i in (result.get("matched_fact_ids") or []) if i]
    try:
        top = float(result.get("top_score") or 0.0)
    except (TypeError, ValueError):
        top = 0.0
    return ids, top


async def _log_recall(registry_name: str, kwargs: dict, result: dict, latency_ms: int) -> None:
    """Console Refine PR 11: one query log row per recall-class MCP call.
    Best-effort by contract: a logging failure never reaches the client."""
    try:
        import uuid as _uuid

        from ..models.query_log import QueryLog as _QueryLog
        from ..retrieval.match_classifier import MatchClassifier

        store = get_metadata_store()
        cid = _customer_id()
        ids, top = recall_summary(registry_name, result)
        if ids:
            customer = await store.get_customer_by_id(cid)
            thresholds = getattr(customer, "retrieval_thresholds", None)
            match = MatchClassifier().classify(top, thresholds) if thresholds is not None else "medium"
        else:
            match = "none"
        ql_id = f"qlog_{_uuid.uuid4().hex[:16]}"
        await store.write_query_log(_QueryLog(
            id=ql_id,
            customer_id=cid,
            query_text=str(kwargs.get("query") or "")[:4000],
            match_type=match,
            injection_method="mcp_tool",
            matched_facts=ids[:50],
            upstream_call_made=False,
            latency_ms=latency_ms,
            client=_current_client.get(),
        ))
        _last_query_log_id.set(ql_id)
    except Exception:  # noqa: BLE001
        logger.debug("mcp.recall_log_failed", tool=registry_name, exc_info=True)


async def _record_invocation(
    name: str, kwargs: dict, *, out: Any, error: Optional[str], code: Optional[str], duration_ms: int,
) -> None:
    """Activity trace (2026-10-19): one tool_invocations row per MCP tool
    call, whatever the tool and however it ended. Best-effort by contract:
    a logging failure never reaches the client."""
    try:
        from ..infrastructure.metadata_store_tool_ext import shorten_args, summarize_result

        cid = _current_customer_id.get()
        if not cid:
            return
        ql_id = _last_query_log_id.get()
        _last_query_log_id.set(None)
        await get_metadata_store().record_tool_invocation(
            cid, tool=name, surface="mcp", client=_current_client.get(),
            args=shorten_args(kwargs), ok=error is None, code=code,
            result_summary=(error if error is not None else summarize_result(out)),
            duration_ms=duration_ms, query_log_id=ql_id,
        )
    except Exception:  # noqa: BLE001
        logger.debug("mcp.invocation_log_failed", tool=name, exc_info=True)


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
            await _send_401(send, "Missing or malformed Authorization header", scope)
            return

        customer = None
        operator = None
        presented_operator_key = False
        try:
            store = get_metadata_store()
            if token.startswith("cco_"):
                # L2-S5c (2026-09-24): an OAuth access token from our own
                # authorization server. subject IS the operator; the
                # tenant is that operator's team. The prefix is ours
                # alone, so an invalid/expired cco_ token never falls
                # through to the static-key paths — it just 401s.
                from ..control.oauth_provider import CrystalOAuthProvider

                at = await CrystalOAuthProvider(store).load_access_token(token)
                if at is not None and at.subject:
                    operator = await store.get_operator_by_id(at.subject)
                    if operator is not None:
                        # OAuth acts as the operator, so the operator's
                        # suspension boundary applies to it too.
                        presented_operator_key = True
                        customer = await store.get_customer_by_id(
                            operator.team_id
                        )
            else:
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
            await _send_401(send, "Invalid api_key", scope)
            return
        if getattr(customer, "deletion_scheduled_at", None) is not None:
            # Lockdown PR-3 (B4-5, 2026-10-09): a tenant scheduled for
            # deletion is locked. Keys and OAuth tokens reach nothing
            # here; the owner's console session keeps its lock-time
            # rights (export, restore) on the HTTP surface.
            await _send_auth_error(
                send, 401,
                "This workspace is scheduled for deletion. Restore it from "
                "Settings to continue.",
                code="account_locked", scope=scope,
            )
            return

        # T2a (2026-09-25): the first-contact signal for onboarding —
        # throttled inside the store; NEVER allowed to break serving.
        try:
            await store.stamp_mcp_seen(customer.id)
        except Exception:
            pass

        tok = _current_customer_id.set(customer.id)
        tok_op = set_current_operator(operator)
        from ..ingress.client import client_from_scope

        tok_client = _current_client.set(client_from_scope(scope))
        try:
            await self.app(scope, receive, send)
        finally:
            _current_client.reset(tok_client)
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


async def _send_auth_error(
    send: Any, status_code: int, detail: str, *, code: Optional[str] = None,
    scope: Optional[dict] = None,
) -> None:
    """Emit a minimal JSON auth-error response over ASGI (401 or 403).
    WWW-Authenticate rides only the 401 — a 403 is a recognized-but-denied
    credential, so a challenge header would be wrong there. `code` (PR-3)
    names the refusal for clients that branch on it."""
    error = "unauthorized" if status_code == 401 else "forbidden"
    payload: dict = {"error": error, "detail": detail}
    if code:
        payload["code"] = code
    body = json.dumps(payload).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    if status_code == 401:
        # L2-S5c: when the OAuth AS is armed, the challenge advertises
        # the RFC 9728 resource metadata — this is the breadcrumb
        # Claude's connector dialog follows to discover sign-in.
        challenge = b"Bearer"
        try:
            from ..config import get_settings

            s = get_settings()
            if s.oauth_enabled:
                # PR 12b: the breadcrumb names the host the client used
                # (allow-listed), so discovery stays self-consistent on
                # the directory's raw Cloud Run URL and on the product host.
                from ..ingress.oauth_hosts import host_from_scope, issuer_for

                base = issuer_for(host_from_scope(scope) if scope else None, s)
                rm = base + "/.well-known/oauth-protected-resource/mcp"
                challenge = f'Bearer resource_metadata="{rm}"'.encode("ascii")
        except Exception:
            pass
        headers.append((b"www-authenticate", challenge))
    await send({
        "type": "http.response.start",
        "status": status_code,
        "headers": headers,
    })
    await send({"type": "http.response.body", "body": body})


async def _send_401(send: Any, detail: str, scope: Optional[dict] = None) -> None:
    """Emit a minimal JSON 401 response over ASGI."""
    await _send_auth_error(send, 401, detail, scope=scope)


def _not_yours(**ids: Any) -> dict:
    """Lockdown PR-3 (B4-4): the refusal a member gets for a crystal they
    may read but do not own. One shape for forget / memory_forget."""
    return {
        "error": "only the owner of this memory or a workspace admin can remove it",
        "code": "not_owner",
        **ids,
    }


# Q46 (PR-6b, 2026-10-13): the tools implemented here (not through the
# registry) bound their own ids; the registry tools are bounded by their
# schemas at _dispatch. An id is 1..64 of [A-Za-z0-9_.:-] (ingress ID64).
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
MCP_LIST_MAX_LIMIT = 200
MCP_LIST_MAX_OFFSET = 100_000


def _invalid_id(name: str, value: Any) -> Optional[dict]:
    """The refusal for an id argument that cannot name a row, or None."""
    if value is None:
        return None
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        return {"error": f"{name} must be an id (1 to 64 letters, digits, _ . : -)",
                "code": "invalid_argument"}
    return None


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


async def _write_admission_block(*, spend: bool = False, incoming: int = 0) -> Optional[dict]:
    """The MCP write door. Every memory-creating tool passes the fact-cap
    wall (memory_full). Tools that RUN MODELS (ingest, import) also pass
    the daily allowance (daily_capacity) with spend=True. Plain writes
    (store, remember, learn, forget, record_gap) never do: Q30=A
    (2026-10-01), remembering is the promise, the fact cap bounds it.

    Combined write admission for the mutating tools (L2-S2 wiring=A,
    2026-09-07): the viewer role check (Q4=C, 2026-08-24) plus the
    trial-expiry check (Q4=B). Read tools never call this by
    construction. Returns the structured denial dict — this surface's
    HTTP status — or None. Expiry pauses WRITES only: the denial says so
    out loud, because a memory product never holds memories hostage."""
    denied = _viewer_write_block()
    if denied:
        return denied
    # Read the contextvar NON-raising: an empty value here is the system
    # lane (keyless in-process callers, direct-call tests) — exactly the
    # lane the trial gate must not touch. _customer_id()'s hard raise
    # stays for tools that require an authenticated tenant.
    cid = _current_customer_id.get()
    if not cid:
        return None  # system lane
    from ..ingress.auth import trial_expired

    try:
        customer = await _get_state()["store"].get_customer_by_id(cid)
    except Exception:  # noqa: BLE001
        return _admission_unavailable()  # RC-05: fail CLOSED
    if customer is not None and trial_expired(customer):
        return {
            "error": (
                "trial expired: memory writes are paused — your memories "
                "are safe and recall stays fully available; upgrade in "
                "the console to resume writing"
            ),
            "code": "trial_expired",
        }
    # T1 (Q5=B, 2026-09-24): the crystal-cap wall, same grace contract as
    # the HTTP surfaces — warnings live in status/meters, the denial
    # fires only past cap × GRACE_FACTOR. Tier None never caps.
    tier_name = getattr(customer, "subscription_tier", None) if customer else None
    if customer is not None and tier_name:
        from ..control.admission import fact_admission, resolve_tier

        # 2026-09-30: counted in billable crystal facts (unit switch).
        tier = resolve_tier(tier_name)
        if tier.fact_cap is not None:
            try:
                count = await _get_state()["store"].count_billable_facts(cid)
            except Exception:  # noqa: BLE001
                return _admission_unavailable()  # RC-05: fail CLOSED
            # PR-2 (B3-7): a batch is judged on bank + batch.
            if fact_admission(count + max(0, int(incoming)), tier) == "blocked":
                return {
                    "error": (
                        f"memory is full ({count:,} crystal facts; this "
                        f"plan holds {tier.fact_cap:,}); everything stored "
                        "stays recallable and exportable; upgrade in the "
                        "console to keep remembering"
                    ),
                    "code": "memory_full",
                }
    # v108 (Q11=A, Q12=A), scoped by Q30=A (2026-10-01): the daily AI
    # allowance, checked ONCE at the door of the tools that run models.
    # An admitted job finishes in full (Q17=A); the overage is carried
    # into tomorrow by daily_capacity.
    if spend and customer is not None:
        from ..control.admission import daily_capacity_block

        try:
            denied = await daily_capacity_block(_get_state()["store"], customer)
        except Exception:  # noqa: BLE001
            return _admission_unavailable()  # RC-05: fail CLOSED
        if denied:
            return denied
    return None


def _admission_unavailable() -> dict:
    """RC-05 (2026-10-04): when a wall cannot be evaluated (the store
    raised), the write is REFUSED, never admitted. The old comment said
    "admission must not add a failure mode"; failing open turned a
    store hiccup into a free pass through every plan wall."""
    logger.warning("mcp.admission_unavailable", exc_info=True)
    return {
        "error": "The memory service could not check your plan limits just now; "
                 "the write was not made. Try again in a moment.",
        "code": "admission_unavailable",
    }


async def _spend_admission_block() -> Optional[dict]:
    """v108: the door for tools that spend LLM money WITHOUT creating
    memory (memory_synthesize). Same daily allowance, no fact wall."""
    cid = _current_customer_id.get()
    if not cid:
        return None
    from ..control.admission import daily_capacity_block

    try:
        customer = await _get_state()["store"].get_customer_by_id(cid)
        if customer is None:
            return None
        return await daily_capacity_block(_get_state()["store"], customer)
    except Exception:  # noqa: BLE001
        return _admission_unavailable()  # RC-05: fail CLOSED


# ---------------------------------------------------------------------------
# FastMCP server + the memory_* tool surface
# ---------------------------------------------------------------------------

# L2-S5c (LIVE-FOUND 2026-09-25, the 421s): the SDK ships DNS-rebinding
# protection ON with EMPTY allowlists, which 421s any request carrying an
# Origin header — so every Python/SDK client and curl (no Origin) worked
# for months while Claude's fetch (Origin: https://claude.ai) died with
# "Misdirected Request" the moment tokens were finally right. Rebinding
# protection defends localhost servers from hostile web pages; a public
# HTTPS API secured by bearer tokens gains nothing from origin gating
# (the same posture as the app's wildcard CORS). Exported for the pin.
_TRANSPORT_SECURITY = TransportSecuritySettings(
    enable_dns_rebinding_protection=False,
)

mcp = FastMCP(
    name="crystal-cache",
    instructions=(
        "Crystal Cache is the user's persistent memory across "
        "conversations, tools, and machines. USE IT PROACTIVELY, without "
        "being asked. Before answering anything about the user, their "
        "projects, decisions, people, or history, call memory_recall or "
        "memory_search first; never say you don't know and never ask the "
        "user to repeat themselves without searching. When a conversation "
        "produces durable knowledge (a decision and its reasoning, a new "
        "fact, a preference, a status change), store it with memory_store "
        "in the same turn; for substantial documents use memory_ingest on "
        "the document itself rather than storing your summary. Many "
        "records with the same fields (games, prices, stats) go in a "
        "lattice (memory_lattices, memory_lattice_write), not one fact "
        "each. The test: "
        "if the user would be annoyed to repeat it next session, store it "
        "now; if a past session might know it, recall it now. Your "
        "identity is taken from the connection, never from tool arguments."
    ),
    stateless_http=True,
    json_response=True,
    streamable_http_path="/",
    transport_security=_TRANSPORT_SECURITY,
)


def _tool_boundary(fn, name: Optional[str] = None):
    """RC-15 / S15 (2026-10-07): the ONE place a refusal becomes a protocol
    error. Tools return {"error": ..., "code": ...} for every wall and
    refusal; on mcp 1.30 a returned dict reaches the client as
    isError=false, so agents treated refusals as answers. Raising
    ToolError puts isError=true on the wire with the stable code in the
    message (verified in a sandbox: "Error executing tool X: [code] ...").
    The raw function stays reachable as .fn for in-process callers."""
    import functools
    import time as _time

    from mcp.server.fastmcp.exceptions import ToolError

    tool_name = name or getattr(fn, "__name__", "tool")

    @functools.wraps(fn)
    async def wrapped(*args, **kwargs):
        t0 = _time.monotonic()
        try:
            out = await fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — recorded, then re-raised unchanged
            await _record_invocation(
                tool_name, kwargs, out=None, error=f"{type(e).__name__}: {str(e)[:300]}",
                code=getattr(e, "code", None) or type(e).__name__,
                duration_ms=int((_time.monotonic() - t0) * 1000),
            )
            raise
        ms = int((_time.monotonic() - t0) * 1000)
        if isinstance(out, dict) and out.get("error") and out.get("code"):
            await _record_invocation(
                tool_name, kwargs, out=out, error=str(out["error"])[:300], code=str(out["code"]),
                duration_ms=ms,
            )
            raise ToolError(f"[{out['code']}] {out['error']}")
        await _record_invocation(tool_name, kwargs, out=out, error=None, code=None, duration_ms=ms)
        return out

    wrapped.fn = fn  # type: ignore[attr-defined]
    return wrapped


def _mcp_tool(*, name: str, description: str, consumer: bool = False):
    """L4 Q1=A (2026-09-05): registration gate for the full-only memory_*
    surface. CC_MCP_TOOLSET=consumer registers ONLY the four consumer tools
    (remember / recall / status / forget, declared with consumer=True);
    "full" (default) registers everything. Read once at import,
    process-level, exactly like CC_AGENT_DISABLED_TOOLS: an unset
    deployment is unchanged. Every registered tool passes through
    _tool_boundary (RC-15)."""
    from ..config import get_settings

    if not consumer and get_settings().mcp_toolset.strip().lower() == "consumer":
        def _skip(fn):
            return fn
        return _skip

    def _register(fn):
        return mcp.tool(name=name, description=description)(_tool_boundary(fn, name=name))

    return _register


@_mcp_tool(
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


@_mcp_tool(
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


@_mcp_tool(
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


@_mcp_tool(
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


@_mcp_tool(
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
    # v108 (Q11=A): deep synthesis spends LLM money, so it passes the
    # daily allowance door. Shallow (no LLM) stays a pure read.
    if merged.get("depth_mode") == "deep":
        denied = await _spend_admission_block()
        if denied:
            return denied
    return await _dispatch("depth_search", query=query, k=k, hints=merged)


@_mcp_tool(
    name="memory_recall",
    description=(
        "Call this BEFORE answering questions about the user's work, "
        "projects, history, or preferences, and whenever they reference "
        "shared context as if you already know it ('the launch plan', "
        "'what did we decide about X', 'what am I working on') — not only "
        "when asked to check memory. Convenience 'what do we know about "
        "X' lookup returning the top matches grouped by entity; the "
        "simple entry point when you don't want to choose among the more "
        "specific search tools. BY DESIGN results span strong matches "
        "down to loose associations — the way human memory surfaces "
        "'this reminds me of' alongside the answer — and every result "
        "carries its score. Read high scorers as grounded answer "
        "material; read low scorers as associative context that may "
        "spark useful connections but must never be asserted as the "
        "answer. When nothing scores high, the bank holds no direct "
        "answer: say so plainly and treat the loose matches as leads at "
        "most. Rows of the person's lattices that the question names (a "
        "player, a team) come back under lattice_rows."
    ),
)
async def memory_recall(
    query: str,
    k: int = 10,
) -> dict:
    return await _with_lattice_hits(await _dispatch("crystal_recall", query=query, k=k), query)


async def _with_lattice_hits(result: Any, query: str) -> Any:
    """Lattices (2026-10-25): rows a question names by entity ride along
    with recall under 'lattice_rows' (only when there are some)."""
    if not isinstance(result, dict) or "error" in result:
        return result
    try:
        from ..lattices.service import recall_hits
        hits = await recall_hits(_get_state()["store"], _customer_id(), query)
    except Exception:  # noqa: BLE001 — recall never fails because of a lattice lookup
        logger.debug("mcp.lattice_recall_failed", exc_info=True)
        return result
    if hits:
        result = {**result, "lattice_rows": hits}
    return result


@_mcp_tool(
    name="memory_store",
    description=(
        "Call this UNPROMPTED whenever the conversation produces durable "
        "knowledge: a decision and its reasoning, a new fact about the "
        "user or their projects, a preference, a status change. Do not "
        "wait to be asked to remember; if the user would be annoyed to "
        "repeat it next session, store it now. Stores a (key, value) pair "
        "for future recall. pair_type defaults to 'question_answer'; set "
        "answer_value to make a key eligible for cache-hit "
        "short-circuiting on future lookups. The pair is stored under the "
        "deployment's default visibility; pass scope='personal' (only you "
        "and admins can retrieve it) or scope='team' (the whole team can) "
        "to override for this write."
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
    denied = await _write_admission_block()
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


@_mcp_tool(
    name="memory_forget",
    description=(
        "Permanently delete stored knowledge. Provide exactly ONE of: "
        "crystal_id (removes a whole memory cluster and all of its facts) or "
        "fact_id (removes a single fact and rebuilds its cluster from the "
        "survivors). IDs come from memory_search / memory_list / memory_keys "
        "results. This cannot be undone: the fact ledger keeps each fact's "
        "before-text for audit, and the document a memory came from is "
        "scrubbed once nothing derives from it. Owner or workspace admin only."
    ),
)
async def memory_forget(
    crystal_id: Optional[str] = None,
    fact_id: Optional[str] = None,
) -> dict:
    # RC-05 (2026-10-04): forgetting is ALWAYS possible. A full bank or an
    # expired trial pauses writes; it must never stop a customer from
    # removing what they stored, which is also how they get back under
    # the cap. No PLAN wall here; the tenancy rule (viewers cannot
    # mutate, Q4=C) still applies, and the store guards the customer.
    denied = _viewer_write_block()
    if denied:
        return denied
    if bool(crystal_id) == bool(fact_id):
        return {
            "deleted": False,
            "error": "provide exactly one of crystal_id or fact_id",
            "code": "bad_arguments",
        }
    bad = _invalid_id("crystal_id", crystal_id) or _invalid_id("fact_id", fact_id)
    if bad:
        return {"deleted": False, **bad}
    state = _get_state()
    store = state["store"]
    cid = _customer_id()
    from ..infrastructure.permissions import can_delete

    if crystal_id:
        before = await store.get_crystal(crystal_id)
        # Lockdown PR-3 (B4-4, 2026-10-09): deleting is the owner's or an
        # admin's. A foreign id and an unreadable one look the same.
        if before is not None and before.customer_id == cid and not can_delete(
            before, get_current_operator()
        ):
            return {"deleted": False, **_not_yours(crystal_id=crystal_id)}
        # RC-05 / Q54 (PR-4): the upload the crystal came from is scrubbed
        # inside delete_crystal itself, on every delete path.
        deleted = await store.delete_crystal(
            crystal_id,
            cid,
            vector_store=state["vector_store"],
            fact_vector_store=state.get("fact_vector_store"),
        )
        return {"deleted": bool(deleted), "crystal_id": crystal_id}
    fact = await store.get_fact(fact_id)
    if fact is not None:
        owner_crystal = await store.get_crystal(fact.crystal_id)
        if (
            owner_crystal is not None
            and owner_crystal.customer_id == cid
            and not can_delete(owner_crystal, get_current_operator())
        ):
            return {"deleted": False, **_not_yours(fact_id=fact_id)}
    deleted = await store.delete_fact(
        fact_id,
        cid,
        encoder=state["encoder"],
        vector_store=state["vector_store"],
        fact_vector_store=state.get("fact_vector_store"),
    )
    return {"deleted": bool(deleted), "fact_id": fact_id}


@_mcp_tool(
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
    # Q31=A: extraction runs models, so ingest passes the daily door.
    denied = await _write_admission_block(spend=True)
    if denied:
        return denied
    if not text.strip():
        return {"crystals_written": 0, "error": "text is required", "code": "bad_arguments"}
    # Lockdown PR-5 (Q45): the one label sanitiser every lane uses. It
    # keeps the RC-04 / E-B3 rule (2026-10-05): an unlabelled ingest gets
    # a label derived from its content, so distinct texts are distinct
    # sources and the same text ingested twice dedups as unchanged.
    from ..ingestion.file_extract import sanitize_label, valid_upload_crystal_type

    label = sanitize_label(label, text=text)
    if not valid_upload_crystal_type(crystal_type):
        return {
            "crystals_written": 0,
            "error": "crystal_type must match customer:<name> (lowercase letters, digits, _ . -, at most 64)",
            "code": "bad_arguments",
        }

    # Audit item (d) (Q2=A, ratified 2026-08-25): synchronous ingest runs
    # one small-tier extraction call per chunk, so unbounded text was
    # unbounded model spend — metered on the ledger but never refused.
    # Refuse BEFORE any row is written or any model is called.
    from ..config import get_settings
    _settings = get_settings()
    # Lockdown PR-1 (B1-44): the cap is a hard default. 0 used to disable
    # it; now 0 (or any non-positive value) falls back to the document
    # cap every other ingest lane has (500,000), never to "unbounded".
    _max_chars = int(_settings.mcp_ingest_max_chars) or int(_settings.document_max_chars)
    if _max_chars <= 0:
        _max_chars = 500_000
    if len(text) > _max_chars:
        return {
            "crystals_written": 0,
            "error": (
                f"text is {len(text):,} characters — over this deployment's "
                f"{_max_chars:,}-character synchronous-ingest ceiling. Split it "
                "and ingest the parts with memory_ingest, each under the "
                "ceiling; the console's upload (Knowledge) takes whole files."
            ),
            "code": "ingest_too_large",
            "max_chars": _max_chars,
            "received_chars": len(text),
        }

    state = _get_state()
    store = state["store"]
    cid = _customer_id()

    # 1. Create the upload row (pending). Lockdown PR-3 (B2-10,
    #    2026-10-09): the upload is a SOURCE and carries the acting
    #    operator's scope stamps, so what an operator ingests is born at
    #    the deployment default scope (personal) and owned by them —
    #    exactly as a console upload would be — instead of unowned team
    #    memory every member could read. System lane (no operator)
    #    keeps the legacy unowned row.
    _op = get_current_operator()
    _scope = _settings.default_ingest_scope if _op is not None else None
    doc = await store.create_document_upload(
        customer_id=cid, label=label, text=text, crystal_type=crystal_type,
        scope=_scope,
        owner_operator_id=(_op.id if _op is not None else None),
        # 2026-10-21: this call runs the pipeline on the row itself, so the
        # row is born claimed; a 'pending' row here was also claimed and
        # extracted by the crystallization worker in the worker service.
        claimed=True,
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
            # PR-4 class 6: the boundary raises only on a truthy error, so
            # a row that failed without a message still says it failed.
            "error": doc2.error_message or "Extraction failed.",
            "code": "ingest_failed",
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
        # crystal ids so the document knows its crystal set. PR-3 (B2-1):
        # the authoritative set is the server-recorded crystal_ids column.
        await store.update_document_review_edits(
            doc.id, cid, extracted_items=doc2.extracted_items or [],
        )
        await store.set_document_crystal_ids(doc.id, cid, list(result.crystal_ids))
    except Exception as e:  # noqa: BLE001 - report failure to the caller, don't 500
        # S3 + S7 (2026-09-30): neither the persisted document error nor
        # the tool result carries raw exception text any more.
        from ..hygiene import safe_error
        ref, message = safe_error(
            "mcp.memory_ingest.crystallize_failed", e,
            user_message="Crystallization failed.",
            document_id=doc.id, customer_id=cid,
        )
        await store.mark_document_error(doc.id, message)
        return {
            "document_id": doc.id,
            "status": "error",
            "error": message,
            "code": "ingest_failed",
            "ref": ref,
            "crystals_written": 0,
        }

    return {
        "document_id": doc.id,
        "status": "crystallized",
        "crystals_written": result.crystals_written,
        "items_extracted": result.items_extracted,
        "errors": result.errors,
    }


@_mcp_tool(
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
    denied = await _write_admission_block()
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


@_mcp_tool(
    name="memory_stats",
    description=(
        "Return summary statistics for the memory bank: total clusters and "
        "facts, distributions by cluster type / quality tier / source / fact "
        "type, and how many entries are cache-hit eligible. Use to gauge how "
        "much is stored and of what kind."
    ),
)
async def memory_stats() -> dict:
    return await _stats_impl()


async def _stats_impl() -> dict:
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

    # RC-15 / S14 (2026-10-07): the plan's caps and what is left of them,
    # so an agent can plan bulk work and explain a wall before it hits.
    plan: dict = {}
    try:
        from ..control.admission import daily_capacity, resolve_tier

        customer = await store.get_customer_by_id(cid)
        if customer is not None:
            tier_name = getattr(customer, "subscription_tier", None) or "free"
            tier = resolve_tier(tier_name)
            billable = await store.count_billable_facts(cid)
            plan = {
                "tier": tier_name,
                "fact_cap": tier.fact_cap,
                "facts_billable": billable,
                "facts_remaining": (
                    max(0, tier.fact_cap - billable) if tier.fact_cap is not None else None
                ),
            }
            if getattr(customer, "inference_mode", "byok") == "managed":
                cap_state = await daily_capacity(store, customer)
                # Q4=A: no customer-facing dollars; state and percentage only.
                plan["daily_capacity"] = {
                    "state": cap_state.get("state"),
                    "pct_used": cap_state.get("pct"),
                }
    except Exception as e:  # noqa: BLE001  (stats never fails on the plan read)
        logger.warning("mcp.stats.plan_unavailable", error=str(e)[:200])

    return {
        "crystal_count": crystal_count,
        "fact_count": total_facts,
        "quality_distribution": dict(quality_dist),
        "crystal_type_distribution": dict(type_dist),
        "pair_type_distribution": dict(pair_type_dist),
        "source_kind_distribution": dict(source_dist),
        "cache_hit_eligible": cache_hit_eligible,
        "total_query_logs": total_query_logs,
        "plan": plan,
    }


@_mcp_tool(
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
    # Q46 (PR-6b): the page is clamped the way memory_export's is (Q3=A);
    # an id that cannot name a row is refused.
    bad = _invalid_id("crystal_id", crystal_id)
    if bad:
        return bad
    limit = max(1, min(int(limit), MCP_LIST_MAX_LIMIT))
    offset = max(0, min(int(offset), MCP_LIST_MAX_OFFSET))

    from ..infrastructure.acl_read_filter import readable_crystals

    if crystal_id:
        crystal = await store.get_crystal(crystal_id)
        if crystal is None or crystal.customer_id != cid:
            return {"error": "crystal not found", "code": "not_found", "crystal_id": crystal_id}
        # Lockdown PR-3 (B4-4, 2026-10-09): the detail read applies the
        # same can_read retrieval does; a colleague's personal crystal is
        # indistinguishable from an unknown id.
        if not await readable_crystals(store, get_current_operator(), [crystal]):
            return {"error": "crystal not found", "code": "not_found", "crystal_id": crystal_id}
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
    # PR-3: the listing is an enumeration path (summaries are content);
    # the page carries only what the caller may read. `total` stays the
    # workspace count so paging by offset still walks every row.
    crystals = await readable_crystals(store, get_current_operator(), crystals)
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "visible": len(crystals),
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


@_mcp_tool(
    name="memory_export",
    description=(
        "Export the workspace's own facts as portable import records "
        "{key, value, key_is_path, pair_type, source_kind, answer_value, "
        "crystal_type, scope} (the published shape at GET /v1/import/schema), "
        "the inverse of memory_import. Use for backup or moving memory to "
        "another workspace. Carries foreground memory only (system-derived "
        "assumptions and background curation travel in the console's "
        "exact-restore export). PAGINATED: returns up to `limit` records per call "
        "(default 500, the import cap) in a stable order; walk pages by advancing `offset` "
        "until has_more is false. Each page's records import cleanly on "
        "their own."
    ),
)
async def memory_export(limit: int = 500, offset: int = 0) -> dict:
    state = _get_state()
    store = state["store"]
    cid = _customer_id()

    # Audit item (d) (Q3=A, ratified 2026-08-25): the whole-bank single
    # response is gone — fact-grain pagination over a deterministic order
    # (the store method's (created_at, id) sort), with the crystal fields
    # joined via a per-call cache so a page costs one query plus one
    # get_crystal per DISTINCT crystal on the page.
    # RC-07 (2026-10-05): the page cap is MEMORY_IMPORT_MAX_RECORDS so a
    # page exports exactly what one import accepts (1,000 vs 500 broke
    # the round trip on page one).
    from ..ingress.import_schema import record_from_fact

    limit = max(1, min(int(limit), MEMORY_IMPORT_MAX_RECORDS))
    offset = max(0, int(offset))
    # Lockdown PR-2 (Q44=A as modified): the portable export carries the
    # workspace's own foreground memory only (origin direct, ungated,
    # customer:* crystals, active facts). Filtered in the query so pages
    # stay exact. System-derived memory travels in the topology export.
    total, facts = await store.list_facts_for_customer_paginated(
        cid, limit=limit, offset=offset, portable_only=True,
    )
    page_len = len(facts)
    # Lockdown PR-3 (B3-1, 2026-10-09): the export is a read; it carries
    # only facts the acting operator may read (can_read, the way
    # retrieval filters). The page window is unchanged — `next_offset`
    # tells the caller where the next page starts even when this one
    # was narrowed.
    from ..infrastructure.acl_read_filter import readable_facts

    facts = await readable_facts(store, get_current_operator(), facts)
    crystal_cache: dict = {}
    records: list = []
    for f in facts:
        c = crystal_cache.get(f.crystal_id)
        if c is None:
            c = await store.get_crystal(f.crystal_id)
            crystal_cache[f.crystal_id] = c
        if c is None:
            continue
        # The key IS the stored sparse path (key_is_path); memory_import
        # keeps it verbatim instead of paying a call per record to
        # re-derive it. The record is exactly the published shape.
        records.append(record_from_fact(f, c))
    return {
        "record_count": len(records),
        "total_records": total,
        "offset": offset,
        "limit": limit,
        "next_offset": offset + page_len,
        "has_more": offset + page_len < total,
        "export_format": "jsonl",
        "data": records,
    }


@_mcp_tool(
    name="memory_import",
    description=(
        "Import records into the bank: the shape memory_export produces, "
        "published at GET /v1/import/schema (key, value, optional key_is_path, "
        "pair_type, source_kind, answer_value, crystal_type, scope; no other "
        "fields). The whole batch (at most 500) is validated first; one bad "
        "record refuses the batch with errors[] naming each record's index "
        "and field, and nothing is written. Imported facts are the caller's "
        "own memory: direct, ungated, counted against the plan, owned by the "
        "caller at the record's scope. wipe=true erases the existing bank first "
        "and needs a workspace admin. Fact-faithful, not cluster-topology-exact."
    ),
)
async def memory_import(
    records: list,
    wipe: bool = False,
    crystal_type: str = "customer:legacy",
) -> dict:
    # Lockdown PR-2 (Q44=A as modified, 2026-10-08): every record is
    # validated against the published import record (GET
    # /v1/import/schema) BEFORE any write; one bad record refuses the
    # whole batch with every invalid record's index, field and message.
    # Records carry no origin, gate or ownership: an imported fact is
    # direct, ungated, counted, owned by the caller at its scope.
    import re as _re

    from ..config import get_settings
    from ..infrastructure.permissions import mode_for_scope
    from ..ingress.import_schema import CRYSTAL_TYPE_PATTERN, validate_records

    # The tenancy rule (viewers cannot mutate) is checked first and stops
    # both halves; a wipe is admin-only (B3-2: a member could erase every
    # colleague's personal memory).
    viewer = _viewer_write_block()
    if viewer:
        return viewer
    operator = get_current_operator()
    if wipe and getattr(operator, "role", None) != "admin":
        return {
            "error": "only a workspace admin can erase the bank with wipe=true",
            "code": "admin_required",
        }
    if not _re.fullmatch(CRYSTAL_TYPE_PATTERN, crystal_type or ""):
        return {
            "error": "crystal_type must look like customer:<name>",
            "code": "invalid_record",
            "errors": [{"index": None, "field": "crystal_type",
                        "message": "must match " + CRYSTAL_TYPE_PATTERN}],
        }
    # v108 (Q15=A, AUDIT_LLM_SPEND G6): one metered call per record and
    # no ceiling meant a 10,000-record import was $6.50 per call,
    # repeatable. Hard cap per call; callers page.
    if isinstance(records, list) and len(records) > MEMORY_IMPORT_MAX_RECORDS:
        return {
            "error": (
                f"memory_import accepts at most {MEMORY_IMPORT_MAX_RECORDS:,} "
                f"records per call ({len(records):,} given); send them in "
                "batches"
            ),
            "code": "import_too_large",
            "max_records": MEMORY_IMPORT_MAX_RECORDS,
        }
    parsed, record_errors = validate_records(records)
    if record_errors:
        return {
            "error": (
                f"{len(record_errors)} problem(s) in the records; nothing was "
                "imported; see errors[] (index, field, message) and the "
                "record shape at GET /v1/import/schema"
            ),
            "code": "invalid_record",
            "errors": record_errors,
        }
    # One metered call per record: the daily door applies, and the fact
    # cap is judged on bank + batch (B3-7). RC-05: the PLAN door is
    # checked AFTER an optional wipe, because erasing is always allowed;
    # only the import half is walled.
    denied = await _write_admission_block(spend=True, incoming=len(parsed))
    if denied and not wipe:
        return denied
    from ..encoding.sparse_keys import generate_sparse_key_metered

    state = _get_state()
    store = state["store"]
    cid = _customer_id()
    encoder = state["encoder"]
    vector_store = state["vector_store"]
    vector_index = state.get("vector_index")
    fact_vector_store = state.get("fact_vector_store")
    default_scope = get_settings().default_ingest_scope
    owner_operator_id = operator.id if operator is not None else None
    group_team_id = operator.team_id if operator is not None else None

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
        if denied:
            # The wipe happened; the import did not. Say so.
            denied = dict(denied)
            denied["wiped"] = len(existing)
            return denied

    processed = 0
    errors = 0
    seen: set = set()
    for rec in parsed:
        try:
            key = rec.key.strip()
            sparse_key = (
                key if rec.key_is_path else
                await generate_sparse_key_metered(key, customer_id=cid, store=store)
            )
            scope = rec.scope or default_scope
            crystal, _fact = await store.add_pair_for_customer(
                customer_id=cid,
                prompt_text=sparse_key,
                answer_text=rec.value,
                pair_type=rec.pair_type,
                encoder=encoder,
                vector_store=vector_store,
                vector_index=vector_index,
                crystal_type=rec.crystal_type or crystal_type,
                source_kind=rec.source_kind,
                answer_value=rec.answer_value,
                # Q44: an import is the caller's own foreground memory.
                origin="direct",
                recall_gated=False,
                owner_operator_id=owner_operator_id,
                group_team_id=group_team_id,
                mode=mode_for_scope(scope) if operator is not None else 0o640,
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


@_mcp_tool(
    name="memory_conflicts",
    description=(
        "List contradictions the system has detected in its own memory — pairs "
        "of stored facts that can't both be true, with the conflicting claims. "
        "Use to see where the memory disagrees with itself before trusting it."
    ),
)
async def memory_conflicts(status: str = "open", limit: int = 50) -> dict:
    return await _dispatch("knowledge_conflicts", status=status, limit=limit)


@_mcp_tool(
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
@_mcp_tool(
    name="memory_record_gap",
    description=(
        "Record a question the memory could not answer. MCP lookups never "
        "record gaps on their own; record every miss. Use when a search "
        "returned results that still did not answer what was asked. "
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
    denied = await _write_admission_block()
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


# Scheduled runs (launch feature, 2026-10-18; Q78-Q81): a standing
# instruction from the client's chat ("every weekday at 8, brief me").
# Plain writes: the viewer gate only, never the capacity door — the run
# pays it when the cognition worker executes it.
@_mcp_tool(
    name="memory_schedule",
    description=(
        "Schedule a run that happens on its own. question: what the run "
        "should do, in the person's words (it can read the web and this "
        "knowledge bank; it does not act on other services yet). kind is "
        "optional: 'research' (default) or 'brief' (the fixed wording: what "
        "changed in memory, open gaps, what waits for review; no question "
        "needed). cadence: once | daily | weekdays | weekly | hourly. time is HH:MM in "
        "tz (IANA, default UTC); weekday for weekly; every_hours for hourly; "
        "run_at (ISO with zone) for once. Ask the person for the time zone "
        "when it is not obvious. deliver_url (optional): an https Slack or "
        "Discord webhook, or any public URL, the result is posted to when a "
        "run completes; the signing secret comes back once. A run spends "
        "daily AI capacity; with none left that occurrence is skipped."
    ),
)
async def memory_schedule(
    cadence: str,
    kind: str = "",
    time: str = "",
    weekday: str = "",
    every_hours: int = 0,
    run_at: str = "",
    tz: str = "UTC",
    question: str = "",
    title: str = "",
    deliver_url: str = "",
) -> dict:
    denied = _viewer_write_block()
    if denied:
        return denied
    return await _dispatch(
        "schedule_run",
        kind=kind or None, cadence=cadence, time=time or None, weekday=weekday or None,
        every_hours=every_hours or None, run_at=run_at or None, tz=tz or "UTC",
        question=question or None, title=title or None, deliver_url=deliver_url or None,
    )


@_mcp_tool(
    name="memory_schedules",
    description=(
        "List the scheduled runs on this memory with their next run, last "
        "outcome and the latest result's summary and crystal id."
    ),
)
async def memory_schedules() -> dict:
    return await _dispatch("list_schedules")


@_mcp_tool(
    name="memory_unschedule",
    description="Delete a scheduled run by schedule_id (from memory_schedules).",
)
async def memory_unschedule(schedule_id: str) -> dict:
    denied = _viewer_write_block()
    if denied:
        return denied
    return await _dispatch("unschedule_run", schedule_id=schedule_id)


# ---------------------------------------------------------------------------
# Consumer tool surface (L4 Q1-Q4=A, 2026-09-05): the four-tool chat-host
# shape the skill and registry listings point at. Registered in BOTH
# toolsets (plain @mcp.tool); CC_MCP_TOOLSET=consumer hides everything else.
# Thin by design: remember/recall ride the same registry bridge as the
# memory_* tools; forget composes the LEDGERED RETIRE exactly as the tenant
# console does (endpoints/admin.py retire route) — never the hard delete.
# ---------------------------------------------------------------------------

_RECALL_MODES = ("quick", "deep", "conflicts", "gaps")


@_mcp_tool(
    consumer=True,
    name="remember",
    description=(
        "Save something worth keeping to the user's memory bank: a decision, "
        "fact, preference, or outcome, as plain text (optionally with a short "
        "title). The fact is at most 800 characters; for anything longer "
        "(a note, a page, a document) use memory_ingest, which chunks and "
        "extracts it. Write real knowledge, not meta-notes about memory "
        "itself; if it contradicts something stored, store the new truth — "
        "the bank detects conflicts and flags them for the user to settle."
    ),
)
async def remember(fact: str, title: Optional[str] = None) -> dict:
    denied = await _write_admission_block()
    if denied:
        return denied
    key = (title or fact).strip()[:120]
    return await _dispatch(
        "crystal_write",
        key=key,
        value=fact,
        pair_type="question_answer",
        crystal_type="customer:legacy",
        source_kind="model_reasoning",
    )


@_mcp_tool(
    consumer=True,
    name="recall",
    description=(
        "Look something up in the user's memory bank. mode='quick' (default) "
        "returns the top matching memories; 'deep' also searches raw ingested "
        "text for verbatim passages; 'conflicts' lists open contradictions "
        "the memory has noticed in itself; 'gaps' lists questions it knows "
        "it cannot answer yet. Each result carries a `tier` (whitelist, "
        "neutral, quarantine, blacklist): read 'whitelist' as strongest and "
        "'quarantine' as unconfirmed, never as equal facts. "
        "Matches deliberately range from strong down to loose associations, "
        "the way human memory offers 'this reminds me of' alongside the "
        "answer: read each result's score, treat high scorers as answer "
        "material and low scorers as associative sparks, and never assert "
        "a loose match as the answer. "
        "If nothing comes back, say so and consider recording a gap — do not "
        "guess."
    ),
)
async def recall(query: str = "", mode: str = "quick", k: int = 10) -> dict:
    m = (mode or "quick").strip().lower()
    if m not in _RECALL_MODES:
        return {"error": f"mode must be one of {list(_RECALL_MODES)}", "code": "bad_arguments"}
    if m == "quick":
        return await _with_lattice_hits(await _dispatch("crystal_recall", query=query, k=k), query)
    if m == "deep":
        facts = await _dispatch("knowledge_search", query=query, k=k, hints=None)
        passages = await _dispatch("content_search", query=query, k=k, hints=None)
        return {"facts": facts, "passages": passages}
    if m == "conflicts":
        return await _dispatch("knowledge_conflicts", status="open", limit=k)
    return await _dispatch("knowledge_gaps", status="open", limit=k)


@_mcp_tool(
    consumer=True,
    name="status",
    description=(
        "How the user's memory bank is doing: how much is stored, of what "
        "kind and quality tier, and how much is cache-hit eligible, plus "
        "scheduled_results: what the person's scheduled runs produced in "
        "the last 24 hours (title, summary, crystal_id), so a morning brief "
        "or overnight research is the first thing you can mention. steward: "
        "whether Crystal runs itself (the Steward), what its last session "
        "did, and, when it is paused at the day's capacity, the refill ask "
        "to pass on to the person. Use before bulk work or when the user "
        "asks what you remember about them."
    ),
)
async def status() -> dict:
    out = await _stats_impl()
    # The Steward (Q100 A, 2026-10-23): the refill ask reaches chat here.
    try:
        from ..endpoints.steward import steward_status
        st = await steward_status(_get_state()["store"], _customer_id())
        steward: dict = {"enabled": st["enabled"], "sessions_today": st["today"]["sessions"],
                         "actions_today": st["today"]["actions"]}
        if st.get("latest"):
            steward["last_brief"] = (st["latest"].get("brief") or "")[:1200]
        if st.get("paused"):
            steward["paused"] = st["paused"]
            steward["ask"] = (
                "Crystal's Steward is paused: today's AI capacity is used. Tell the person they can "
                "upgrade the plan, raise the cap, or wait for midnight UTC; it resumes on its own."
            )
        out["steward"] = steward
    except Exception as e:  # noqa: BLE001 — status never fails on this read
        logger.warning("mcp.status.steward_unavailable", error=str(e)[:200])
    # Scheduled runs (2026-10-18, Q90=A): the results land here so every
    # MCP client sees overnight work on its first call, no webhook needed.
    try:
        from datetime import datetime, timedelta, timezone
        store = _get_state()["store"]
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        rows = await store.list_recent_scheduled_results(_customer_id(), since=since, limit=10)
        out["scheduled_results"] = [
            {
                "schedule_id": r.id, "title": r.title, "kind": r.kind,
                "ran_at": r.last_run_at.isoformat() if r.last_run_at else None,
                "crystal_id": r.last_crystal_id, "summary": r.last_result_summary,
            }
            for r in rows
        ]
    except Exception as e:  # noqa: BLE001  (status never fails on this read)
        logger.warning("mcp.status.scheduled_results_unavailable", error=str(e)[:200])
        out["scheduled_results"] = []
    return out


@_mcp_tool(
    consumer=True,
    name="forget",
    description=(
        "Retire a memory. Provide the crystal_id of the memory cluster (ids "
        "appear in recall results). The cluster and its facts are removed "
        "from recall and from the bank; the facts' full text is kept in the "
        "bank's append-only fact ledger for audit, but there is no restore. "
        "Only forget when the user clearly asks; when a fact is merely "
        "outdated, prefer remember-ing the new truth and letting curation "
        "settle it."
    ),
)
async def forget(crystal_id: str) -> dict:
    # RC-05 (2026-10-04): forgetting is always possible; no PLAN wall. The
    # tenancy rule (viewers cannot mutate, Q4=C) still applies.
    denied = _viewer_write_block()
    if denied:
        return denied
    bad = _invalid_id("crystal_id", crystal_id)
    if bad:
        return {"retired": False, **bad}
    state = _get_state()
    store = state["store"]
    cid = _customer_id()
    crystal = await store.get_crystal(crystal_id)
    if crystal is None or crystal.customer_id != cid:
        return {"retired": False, "error": "crystal not found",
                "code": "not_found", "crystal_id": crystal_id}
    from ..infrastructure.permissions import can_delete

    if not can_delete(crystal, get_current_operator()):
        # Lockdown PR-3 (B4-4): owner or admin only.
        return {"retired": False, **_not_yours(crystal_id=crystal_id)}
    # Mirror the tenant console's retire (endpoints/admin.py): one ledger
    # row per fact with the FULL before-text, then removal through the
    # standard machinery (vector/index parity). A failure mid-way leaves
    # live facts plus honest ledger rows — never lost knowledge.
    facts = await store.list_facts_for_crystal(crystal_id)
    ledgered = []
    for f in facts:
        row = await store.append_fact_ledger(
            cid, crystal_id, f.id,
            op="retire", actor="mcp_consumer",
            before_prompt=f.prompt_text,
            before_text=f.claim_text,
        )
        ledgered.append(row["id"] if isinstance(row, dict) else getattr(row, "id", None))
    # RC-05 / Q54 (PR-4): the upload is scrubbed inside delete_crystal.
    deleted = await store.delete_crystal(
        crystal_id,
        cid,
        vector_store=state["vector_store"],
        fact_vector_store=state.get("fact_vector_store"),
    )
    return {
        "retired": bool(deleted),
        "crystal_id": crystal_id,
        "facts_ledgered": len(ledgered),
        "note": "retired from recall; full text preserved in the fact ledger",
    }




# ---------------------------------------------------------------------------
# The controls (2026-10-23). Anthony: "I want ALL THE CONTROLS. Give me the
# mcp route." Every curation act the console offers, as tools an AI client
# (or a person through one) can use: the same implementations the in-app
# Steward runs (cognition/controls.py), the same ledger with who / why /
# the inverse, the same 24-hour undo on Activity › Steward. Reads: the
# assumptions queue and the action ledger. Walls: the viewer role is
# refused everywhere; acts that add facts pay the fact cap; acts that
# queue a model run pay the daily capacity door. Not in the consumer
# toolset (CC_MCP_TOOLSET=consumer keeps the four).
# ---------------------------------------------------------------------------

def _actor() -> str:
    op = get_current_operator()
    oid = getattr(op, "id", None) if op is not None else None
    return f"chat:{oid}" if oid else "chat"


def _controls_state() -> dict:
    s = _get_state()
    return {"encoder": s.get("encoder"), "vector_store": s.get("vector_store"),
            "fact_vector_store": s.get("fact_vector_store"), "vector_index": s.get("vector_index")}


@_mcp_tool(
    name="memory_assumptions",
    description=(
        "Assumptions and consequences the bank drew on its own, as a reviewer reads them: "
        "the statement, what it was built from (assumptions) or derived from and how the "
        "facts judged it (consequences), whether anything contradicts it. state: waiting "
        "(default) | approved | invalidated | any. kind: any (default) | assumption | "
        "consequence. Read this before approving or rejecting."
    ),
)
async def memory_assumptions(state: str = "waiting", kind: str = "any", limit: int = 30) -> dict:
    from ..cognition.controls import list_assumptions_view
    items = await list_assumptions_view(_get_state()["store"], _customer_id(), state=state, kind=kind, limit=limit)
    return {"assumptions": items, "count": len(items)}


@_mcp_tool(
    name="memory_actions",
    description=(
        "The curation ledger: every act taken on this bank through the controls or by the "
        "in-app Steward, newest first, with who did it, why, and whether it can still be "
        "undone (24 hours). Read it before acting so you never repeat what a person undid."
    ),
)
async def memory_actions(limit: int = 50) -> dict:
    rows = await _get_state()["store"].list_steward_actions(_customer_id(), limit=max(1, min(int(limit or 50), 200)))
    return {"actions": rows, "count": len(rows)}


@_mcp_tool(
    name="memory_decide_conflict",
    description=(
        "Settle an open conflict (from memory_conflicts). resolution: superseded (one "
        "claim is outdated; pass loser 'a' or 'b', that fact leaves recall, reversibly), "
        "qualified (both true under different conditions), dismissed (never a real "
        "contradiction), blacklisted (the losing claim is wrong and must never return; "
        "pass loser). Needs a reason a person will read. Undoable for 24 hours."
    ),
)
async def memory_decide_conflict(conflict_id: str, resolution: str, reason: str, loser: str = "") -> dict:
    from ..cognition.controls import decide_conflict
    denied = _viewer_write_block()
    if denied:
        return denied
    bad = _invalid_id("conflict_id", conflict_id)
    if bad:
        return bad
    return await decide_conflict(_get_state()["store"], _customer_id(), conflict_id=conflict_id, resolution=resolution,
                                 reason=reason, loser=(loser or None), actor=_actor())


@_mcp_tool(
    name="memory_reopen_conflict",
    description="Reopen a settled conflict: it goes back to open and a retired claim returns to recall. Needs a reason.",
)
async def memory_reopen_conflict(conflict_id: str, reason: str) -> dict:
    from ..cognition.controls import reopen_conflict
    denied = _viewer_write_block()
    if denied:
        return denied
    bad = _invalid_id("conflict_id", conflict_id)
    if bad:
        return bad
    return await reopen_conflict(_get_state()["store"], _customer_id(), conflict_id=conflict_id, reason=reason, actor=_actor())


@_mcp_tool(
    name="memory_approve_assumption",
    description=(
        "Approve a waiting assumption or consequence (from memory_assumptions): it enters "
        "recall and takes its parent's standing. Refused when a stated fact contradicts it "
        "or a conflict is open on it, unless force is true. Needs a reason."
    ),
)
async def memory_approve_assumption(crystal_id: str, reason: str, force: bool = False) -> dict:
    from ..cognition.controls import approve_assumption
    denied = _viewer_write_block()
    if denied:
        return denied
    bad = _invalid_id("crystal_id", crystal_id)
    if bad:
        return bad
    return await approve_assumption(_get_state()["store"], _customer_id(), crystal_id=crystal_id, reason=reason,
                                    actor=_actor(), force=bool(force))


@_mcp_tool(
    name="memory_reject_assumption",
    description="Invalidate a waiting assumption or consequence the facts show to be wrong. It leaves recall; undoable for 24 hours. Needs a reason.",
)
async def memory_reject_assumption(crystal_id: str, reason: str) -> dict:
    from ..cognition.controls import reject_assumption
    denied = _viewer_write_block()
    if denied:
        return denied
    bad = _invalid_id("crystal_id", crystal_id)
    if bad:
        return bad
    return await reject_assumption(_get_state()["store"], _customer_id(), crystal_id=crystal_id, reason=reason, actor=_actor())


@_mcp_tool(
    name="memory_verify_assumption",
    description="Queue background research to confirm or refute an assumption. Spends the plan's daily AI capacity. Needs a reason.",
)
async def memory_verify_assumption(crystal_id: str, reason: str) -> dict:
    from ..cognition.controls import verify_assumption
    denied = await _write_admission_block(spend=True)
    if denied:
        return denied
    bad = _invalid_id("crystal_id", crystal_id)
    if bad:
        return bad
    return await verify_assumption(_get_state()["store"], _customer_id(), crystal_id=crystal_id, reason=reason, actor=_actor())


@_mcp_tool(
    name="memory_research_gap",
    description="Queue background research for an open gap (from memory_gaps). Spends the plan's daily AI capacity. Needs a reason.",
)
async def memory_research_gap(gap_id: str, reason: str) -> dict:
    from ..cognition.controls import research_gap
    denied = await _write_admission_block(spend=True)
    if denied:
        return denied
    bad = _invalid_id("gap_id", gap_id)
    if bad:
        return bad
    return await research_gap(_get_state()["store"], _customer_id(), gap_id=gap_id, reason=reason, actor=_actor())


@_mcp_tool(
    name="memory_dismiss_gap",
    description="Close an open gap that is not worth researching. Reversible with memory_reopen_gap or undo. Needs a reason.",
)
async def memory_dismiss_gap(gap_id: str, reason: str) -> dict:
    from ..cognition.controls import dismiss_gap
    denied = _viewer_write_block()
    if denied:
        return denied
    bad = _invalid_id("gap_id", gap_id)
    if bad:
        return bad
    return await dismiss_gap(_get_state()["store"], _customer_id(), gap_id=gap_id, reason=reason, actor=_actor())


@_mcp_tool(
    name="memory_reopen_gap",
    description="Reopen a dismissed gap. Needs a reason.",
)
async def memory_reopen_gap(gap_id: str, reason: str) -> dict:
    from ..cognition.controls import reopen_gap
    denied = _viewer_write_block()
    if denied:
        return denied
    bad = _invalid_id("gap_id", gap_id)
    if bad:
        return bad
    return await reopen_gap(_get_state()["store"], _customer_id(), gap_id=gap_id, reason=reason, actor=_actor())


@_mcp_tool(
    name="memory_answer_gap",
    description=(
        "Answer an open gap yourself: the text is stored as a stated fact under the gap's "
        "key (or the key you pass, Source|Locator|Subject) and the gap is filled with it. "
        "Pays the fact cap. Undoable for 24 hours. Needs a reason."
    ),
)
async def memory_answer_gap(gap_id: str, text: str, reason: str, key: str = "") -> dict:
    from ..cognition.controls import answer_gap
    denied = await _write_admission_block(incoming=1)
    if denied:
        return denied
    bad = _invalid_id("gap_id", gap_id)
    if bad:
        return bad
    return await answer_gap(_get_state()["store"], _controls_state(), _customer_id(), gap_id=gap_id, text=text,
                            key=(key or None), reason=reason, actor=_actor())


@_mcp_tool(
    name="memory_edit_fact",
    description=(
        "Rewrite a fact's text in place (and its key, optionally). The old text survives in "
        "the fact ledger and the edit is undoable for 24 hours. crystal_id and fact_id come "
        "from memory_list / memory_search results. Needs a reason."
    ),
)
async def memory_edit_fact(crystal_id: str, fact_id: str, text: str, reason: str, key: str = "") -> dict:
    from ..cognition.controls import edit_fact
    denied = _viewer_write_block()
    if denied:
        return denied
    for name, value in (("crystal_id", crystal_id), ("fact_id", fact_id)):
        bad = _invalid_id(name, value)
        if bad:
            return bad
    return await edit_fact(_get_state()["store"], _controls_state(), _customer_id(), crystal_id=crystal_id, fact_id=fact_id,
                           text=text, key=(key or None), reason=reason, actor=_actor())


@_mcp_tool(
    name="memory_retire_fact",
    description=(
        "Delete ONE fact from a crystal. Permanent: the fact ledger keeps its text for audit, "
        "nothing restores it (memory_forget retires a whole crystal). Needs a reason."
    ),
)
async def memory_retire_fact(crystal_id: str, fact_id: str, reason: str) -> dict:
    from ..cognition.controls import retire_fact
    denied = _viewer_write_block()
    if denied:
        return denied
    for name, value in (("crystal_id", crystal_id), ("fact_id", fact_id)):
        bad = _invalid_id(name, value)
        if bad:
            return bad
    state = _get_state()
    crystal = await state["store"].get_crystal(crystal_id)
    if crystal is None or crystal.customer_id != _customer_id():
        return {"ok": False, **_not_yours(crystal_id=crystal_id)}
    from ..infrastructure.permissions import can_delete
    if not can_delete(crystal, get_current_operator()):
        return {"ok": False, **_not_yours(crystal_id=crystal_id)}
    return await retire_fact(state["store"], _controls_state(), _customer_id(), crystal_id=crystal_id, fact_id=fact_id,
                             reason=reason, actor=_actor())


@_mcp_tool(
    name="memory_set_tier",
    description=(
        "Move a crystal's quality tier: whitelist (trusted), neutral, quarantine (unvetted), "
        "blacklist (wrong; leaves recall). A tier is a signal readers weigh, not a gate. "
        "Undoable for 24 hours. Needs a reason."
    ),
)
async def memory_set_tier(crystal_id: str, tier: str, reason: str) -> dict:
    from ..cognition.controls import set_tier
    denied = _viewer_write_block()
    if denied:
        return denied
    bad = _invalid_id("crystal_id", crystal_id)
    if bad:
        return bad
    return await set_tier(_get_state()["store"], _customer_id(), crystal_id=crystal_id, tier=tier, reason=reason, actor=_actor())


@_mcp_tool(
    name="memory_reprobe_fact",
    description="Let the implication scan probe a fact again on the next idle cycle (its consequences are re-derived and re-checked). Harmless. Needs a reason.",
)
async def memory_reprobe_fact(fact_id: str, reason: str) -> dict:
    from ..cognition.controls import reprobe_fact
    denied = _viewer_write_block()
    if denied:
        return denied
    bad = _invalid_id("fact_id", fact_id)
    if bad:
        return bad
    return await reprobe_fact(_get_state()["store"], _customer_id(), fact_id=fact_id, reason=reason, actor=_actor())


@_mcp_tool(
    name="memory_undo",
    description="Undo one act from memory_actions within its 24-hour window: the bank goes back to how it was.",
)
async def memory_undo(action_id: str) -> dict:
    from ..cognition.controls import undo_action
    denied = _viewer_write_block()
    if denied:
        return denied
    bad = _invalid_id("action_id", action_id)
    if bad:
        return bad
    return await undo_action(_get_state()["store"], _customer_id(), action_id, state=_controls_state(), actor=_actor())


@_mcp_tool(
    name="memory_steward",
    description=(
        "The in-app Steward: read its state (enabled, today's sessions and actions, the last "
        "brief, whether it is paused at the day's capacity), or switch it on or off by passing "
        "enabled. When on, Crystal runs one unattended curation session per idle cycle with "
        "these same controls."
    ),
)
async def memory_steward(enabled: Optional[bool] = None) -> dict:
    from ..endpoints.steward import steward_status
    store = _get_state()["store"]
    cid = _customer_id()
    if enabled is not None:
        denied = _viewer_write_block()
        if denied:
            return denied
        await store.set_customer_steward_enabled(cid, bool(enabled))
        try:
            await store.record_curation_event(cid, event_type="steward_enabled" if enabled else "steward_disabled", subject_id=None,
                                              label="Crystal now runs itself" if enabled else "Crystal no longer runs itself",
                                              payload={"by": _actor()})
        except Exception:  # noqa: BLE001
            pass
    st = await steward_status(store, cid)
    return {"enabled": st["enabled"], "platform_enabled": st["platform_enabled"], "today": st["today"],
            "paused": st["paused"], "latest": st["latest"]}


# ---------------------------------------------------------------------------
# Automation rules (Q101, 2026-10-23): the ordered allow/deny list. "Think
# of a networking allow and deny rules." The same list Settings > Curation
# shows; rules run in the background from the next idle cycle.
# ---------------------------------------------------------------------------

def _rule_view(rule, order: int) -> dict:
    from ..system_rules.curation import rule_text
    return {
        "id": rule.id, "order": order, "enabled": bool(rule.enabled),
        "decision": rule.action.get("decision", "allow"), "text": rule_text(rule), "name": rule.name,
        "selector": dict(rule.selector), "conditions": dict(rule.conditions), "action": dict(rule.action),
        "fire_count": int(rule.fire_count or 0),
        "last_fired_at": rule.last_fired_at.isoformat() if rule.last_fired_at else None,
    }


async def _rules_view(store, cid: str) -> dict:
    from ..system_rules.curation import DEFAULT_LINE
    from ..system_rules.store import convert_legacy_rules, list_rules
    await convert_legacy_rules(store, cid)
    rules = await list_rules(store, cid)
    return {"rules": [_rule_view(r, i) for i, r in enumerate(rules, start=1)], "count": len(rules),
            "default": DEFAULT_LINE}


@_mcp_tool(
    name="memory_rules",
    description=(
        "The automation rules, in order. They run top to bottom over everything waiting "
        "for a decision (gated memory, assumptions, consequences); the first rule that "
        "matches decides. allow approves it, deny holds it for a person, and when nothing "
        "matches a person decides. Each rule is one sentence."
    ),
)
async def memory_rules() -> dict:
    return await _rules_view(_get_state()["store"], _customer_id())


@_mcp_tool(
    name="memory_rule_add",
    description=(
        "Add an automation rule at the bottom of the list. decision: allow (approve it, or "
        "let a fact into recall) | deny (hold it for a person). kind: fact | assumption | "
        "consequence | any. Narrow what it applies to with key_prefix (any key starts with "
        "it), source (document | chat | research | reasoning) or tier (quarantine | neutral "
        "| whitelist). Conditions, all of which must hold: min_agreeing_facts (stated facts "
        "that agree), no_contradictions (default true for allow), min_citations, "
        "parent_trusted, min_confidence (0 to 1), waited_days, scan_passed. trust=true lets "
        "an allowed fact in at the neutral tier. Nothing here is undone by memory_undo; "
        "remove the rule instead."
    ),
)
async def memory_rule_add(
    decision: str,
    kind: str = "any",
    key_prefix: str = "",
    source: str = "",
    tier: str = "",
    min_agreeing_facts: int = 0,
    no_contradictions: Optional[bool] = None,
    min_citations: int = 0,
    parent_trusted: bool = False,
    min_confidence: float = 0.0,
    waited_days: int = 0,
    scan_passed: bool = False,
    trust: bool = False,
    name: str = "",
) -> dict:
    from ..system_rules import RuleValidationError, SystemRule, validate_rule
    from ..system_rules.curation import RULE_TYPE, rule_text
    from ..system_rules.store import create_rule, list_rules

    denied = _viewer_write_block()
    if denied:
        return denied
    selector: dict = {"kind": (kind or "any").strip().lower()}
    if (key_prefix or "").strip():
        selector["key_prefix"] = key_prefix.strip()
    if (source or "").strip():
        selector["source"] = source.strip().lower()
    if (tier or "").strip():
        selector["tier"] = tier.strip().lower()
    conditions: dict = {}
    if min_agreeing_facts:
        conditions["min_agreeing_facts"] = int(min_agreeing_facts)
    if no_contradictions is None:
        no_contradictions = (decision or "").strip().lower() == "allow"
    if no_contradictions:
        conditions["no_contradictions"] = True
    if min_citations:
        conditions["min_citations"] = int(min_citations)
    if parent_trusted:
        conditions["parent_trusted"] = True
    if min_confidence:
        conditions["min_confidence"] = float(min_confidence)
    if waited_days:
        conditions["waited_days"] = int(waited_days)
    if scan_passed:
        conditions["scan_passed"] = True
    action: dict = {"decision": (decision or "").strip().lower()}
    if trust:
        action["tier"] = "neutral"
    store = _get_state()["store"]
    cid = _customer_id()
    if len(await list_rules(store, cid)) >= 50:
        return {"error": "a bank holds at most 50 rules", "code": "rule_cap"}
    draft = SystemRule(id="", customer_id=cid, rule_type=RULE_TYPE, name="", selector=selector,
                       conditions=conditions, action=action)
    try:
        validate_rule(draft)
        label = (name or "").strip()[:120] or rule_text(draft)[:120]
        rule = await create_rule(store, cid, RULE_TYPE, label, selector=selector,
                                 conditions=conditions, action=action)
    except RuleValidationError as e:
        return {"error": str(e), "code": "rule_invalid"}  # detail-ok: the strict validator's message names the bad key for the caller; no internals
    view = await _rules_view(store, cid)
    added = next((r for r in view["rules"] if r["id"] == rule.id), None)
    return {"ok": True, "rule": added, "rules": view["rules"], "default": view["default"]}


@_mcp_tool(
    name="memory_rule_remove",
    description="Delete one automation rule (from memory_rules). The lines below it move up.",
)
async def memory_rule_remove(rule_id: str) -> dict:
    from ..system_rules.store import delete_rule
    denied = _viewer_write_block()
    if denied:
        return denied
    bad = _invalid_id("rule_id", rule_id)
    if bad:
        return bad
    store = _get_state()["store"]
    cid = _customer_id()
    if not await delete_rule(store, cid, rule_id):
        return {"error": "no such rule", "code": "not_found"}
    view = await _rules_view(store, cid)
    return {"ok": True, "deleted": rule_id, "rules": view["rules"], "default": view["default"]}


@_mcp_tool(
    name="memory_rule_move",
    description=(
        "Move one automation rule one line up or down (direction: up | down). Order is what "
        "the list means: the first matching rule decides, so a deny above an allow wins."
    ),
)
async def memory_rule_move(rule_id: str, direction: str) -> dict:
    from ..system_rules.store import move_rule
    denied = _viewer_write_block()
    if denied:
        return denied
    bad = _invalid_id("rule_id", rule_id)
    if bad:
        return bad
    direction = (direction or "").strip().lower()
    if direction not in ("up", "down"):
        return {"error": "direction must be up or down", "code": "invalid_argument"}
    store = _get_state()["store"]
    cid = _customer_id()
    rules = await move_rule(store, cid, rule_id, direction)
    if rules is None:
        return {"error": "no such rule", "code": "not_found"}
    view = await _rules_view(store, cid)
    return {"ok": True, "rules": view["rules"], "default": view["default"]}


@_mcp_tool(
    name="memory_rule_toggle",
    description="Switch one automation rule on or off without deleting it. Off, the line is skipped.",
)
async def memory_rule_toggle(rule_id: str, enabled: bool) -> dict:
    from ..system_rules import RuleValidationError
    from ..system_rules.store import update_rule
    denied = _viewer_write_block()
    if denied:
        return denied
    bad = _invalid_id("rule_id", rule_id)
    if bad:
        return bad
    store = _get_state()["store"]
    cid = _customer_id()
    try:
        rule = await update_rule(store, cid, rule_id, enabled=bool(enabled))
    except RuleValidationError as e:
        return {"error": str(e), "code": "rule_invalid"}  # detail-ok: the strict validator's message names the bad key for the caller; no internals
    if rule is None:
        return {"error": "no such rule", "code": "not_found"}
    view = await _rules_view(store, cid)
    return {"ok": True, "rule": next((r for r in view["rules"] if r["id"] == rule.id), None)}


# ---------------------------------------------------------------------------
# Connected apps (2026-10-24): a person's Drive, Calendar, Notion, Slack and
# (private beta) Gmail, reached through Nango. Connecting needs the
# person's own sign-in, so it happens in the console; from chat a client
# can see the apps and set what Crystal may do with each one.
# ---------------------------------------------------------------------------

@_mcp_tool(
    name="memory_connections",
    description=(
        "The person's connected apps and what Crystal may do with each: watch (read it on a "
        "schedule; goals see what it shows, kept 7 days), remember (turn items into memory, only "
        "those the app's rules allow), act (goals may send, post or create through it). Also "
        "lists the apps they could still connect, and which are invite-only."
    ),
)
async def memory_connections() -> dict:
    from ..connectors.service import catalog_view
    view = await catalog_view(_get_state()["store"], _customer_id())
    return {"configured": view["configured"],
            "apps": [{"integration": a["integration"], "name": a["name"], "available": a["available"],
                      "beta": a["beta"], "connections": a["connections"]} for a in view["apps"]]}


@_mcp_tool(
    name="memory_connection_set",
    description=(
        "Set what Crystal may do with one connected app (from memory_connections). watch / remember / "
        "act: true or false (omit to keep). rules: the ordered list deciding what is remembered, each "
        "{decision: allow | deny, field: from | subject | text | folder | label | channel, match: "
        "contains | starts_with | equals, value}; the first rule that matches decides and nothing "
        "matching means not remembered. cadence_minutes: how often to check (5 to 1440). paused: stop "
        "or resume watching."
    ),
)
async def memory_connection_set(
    connection_id: str,
    watch: Optional[bool] = None,
    remember: Optional[bool] = None,
    act: Optional[bool] = None,
    rules: Optional[list] = None,
    cadence_minutes: Optional[int] = None,
    paused: Optional[bool] = None,
) -> dict:
    from ..connectors.service import ConnectionRefused, set_uses
    denied = _viewer_write_block()
    if denied:
        return denied
    bad = _invalid_id("connection_id", connection_id)
    if bad:
        return bad
    uses = {k: v for k, v in (("watch", watch), ("remember", remember), ("act", act)) if v is not None} or None
    try:
        out = await set_uses(_get_state()["store"], _customer_id(), connection_id, uses=uses, rules=rules,
                             cadence_minutes=cadence_minutes, paused=paused)
    except ConnectionRefused as e:
        return {"error": e.message, "code": e.code}  # detail-ok: our own refusal text, no internals
    return {"ok": True, "connection": out}


# ---------------------------------------------------------------------------
# Goals (Q109 A, 2026-10-24): standing directives a frontier model plans
# and a large model works, every check-in. Same service as the console
# (goals/service.py); sending a draft from here is the person's approval.
# ---------------------------------------------------------------------------

def _goal_view(g: dict) -> dict:
    plan = g.get("plan") or {}
    return {"goal_id": g["id"], "title": g["title"], "status": g["status"], "directive": g["directive"],
            "summary": plan.get("summary"), "success_measure": plan.get("success_measure"),
            "check_every_minutes": plan.get("check_every_minutes"),
            "steps": [{"n": s.get("n"), "title": s.get("title"), "done": s.get("done")} for s in plan.get("steps", [])],
            "planned": bool(g.get("planned_at")), "may_send": g["may_send"], "may_post": g["may_post"],
            "missing": [{k: m.get(k) for k in ("kind", "need", "why", "suggestion", "blocking")} for m in plan.get("missing", [])],
            "held_for_setup": g["status"] == "needs_setup",
            "waiting_for_answers": g["status"] == "deciding",
            "decisions": [{"index": i, "question": d.get("question"), "why": d.get("why"),
                           "options": [o.get("label") for o in d.get("options", [])],
                           "recommended": d.get("recommended"), "answer": d.get("answer")}
                          for i, d in enumerate(plan.get("decisions", []))],
            "lattices": [{"name": x.get("name"), "lattice_id": x.get("lattice_id")} for x in plan.get("lattices") or []],
            "last_check_in": g.get("last_run_at"), "next_check_in": g.get("next_run_at"),
            "last_summary": (g.get("last_summary") or "")[:1200]}


@_mcp_tool(
    name="memory_goals",
    description=(
        "The person's goals: standing directives Crystal plans (a strong model turns the words into "
        "steps, a success measure and a check-in cadence) and works at every check-in, using what their "
        "connected apps show. Each goal with its plan, permissions and last check-in summary."
    ),
)
async def memory_goals() -> dict:
    goals = await _get_state()["store"].list_goals(_customer_id())
    return {"goals": [_goal_view(g) for g in goals], "count": len(goals)}


@_mcp_tool(
    name="memory_goal_add",
    description=(
        "Add a goal in the person's words (directive), e.g. 'Reply to every email within 3 hours; draft "
        "replies for me to approve.' It is planned on the next pass and then checked in on by its own "
        "cadence. may_send: it may send email on its own; may_post: it may post, create events or add "
        "to pages on its own. Both default false (drafts wait for the person). Only set them when the "
        "person said so."
    ),
)
async def memory_goal_add(title: str, directive: str, may_send: bool = False, may_post: bool = False) -> dict:
    from ..goals.service import GoalRefused, create
    denied = _viewer_write_block()
    if denied:
        return denied
    try:
        g = await create(_get_state()["store"], _customer_id(), title=title, directive=directive,
                         may_send=may_send, may_post=may_post)
    except GoalRefused as e:
        return {"error": e.message, "code": e.code}  # detail-ok: our own refusal text, no internals
    return {"ok": True, "goal": _goal_view(g)}


@_mcp_tool(
    name="memory_goal_update",
    description=(
        "Change one goal (from memory_goals). title, directive (a new directive gets a new plan), "
        "status (active | paused | done | archived), may_send, may_post: omit to keep. replan: true asks "
        "for a fresh plan, keeping finished steps (and re-checks what it is missing). run_now: true checks in "
        "on the next pass. run_anyway: true lets a goal held for setup (held_for_setup) run with what it has; "
        "only when the person says so. answers: for a goal waiting_for_answers, the person's choices, "
        "[{index, choice (option index) | other (their own words)}] for every open decision; then the plan is "
        "written on them. Never answer for the person."
    ),
)
async def memory_goal_update(
    goal_id: str,
    title: Optional[str] = None,
    directive: Optional[str] = None,
    status: Optional[str] = None,
    may_send: Optional[bool] = None,
    may_post: Optional[bool] = None,
    replan: bool = False,
    run_now: bool = False,
    run_anyway: bool = False,
    answers: Optional[list] = None,
) -> dict:
    from ..goals import service as goals_service
    denied = _viewer_write_block()
    if denied:
        return denied
    bad = _invalid_id("goal_id", goal_id)
    if bad:
        return bad
    store, cid = _get_state()["store"], _customer_id()
    try:
        g = await goals_service.update(store, cid, goal_id, title=title, directive=directive, status=status,
                                       may_send=may_send, may_post=may_post)
        if replan:
            g = await goals_service.replan(store, cid, goal_id)
        if answers is not None:
            g = await goals_service.decide(store, cid, goal_id, answers)
        if run_anyway:
            g = await goals_service.run_anyway(store, cid, goal_id)
        if run_now:
            g = await goals_service.run_now(store, cid, goal_id)
    except goals_service.GoalRefused as e:
        return {"error": e.message, "code": e.code}  # detail-ok: our own refusal text, no internals
    return {"ok": True, "goal": _goal_view(g)}


@_mcp_tool(
    name="memory_goal_items",
    description=(
        "What goals produced: action items for the person, drafts waiting for approval (with the exact "
        "text and where it would go), sent messages, and notes about what a goal needs. Filter by "
        "goal_id and status (open | done | dismissed | sent | failed)."
    ),
)
async def memory_goal_items(goal_id: Optional[str] = None, status: Optional[str] = "open", limit: int = 30) -> dict:
    if goal_id:
        bad = _invalid_id("goal_id", goal_id)
        if bad:
            return bad
    items = await _get_state()["store"].list_goal_items(_customer_id(), goal_id, status=status or None,
                                                        limit=max(1, min(int(limit or 30), 100)))
    return {"items": [{"item_id": i["id"], "goal_id": i["goal_id"], "kind": i["kind"], "status": i["status"],
                       "title": i["title"], "body": i["body"][:4000], "due_at": i["due_at"],
                       "target": {k: v for k, v in (i.get("target") or {}).items() if k != "connection_id"}}
                      for i in items], "count": len(items)}


@_mcp_tool(
    name="memory_goal_item",
    description=(
        "Act on one goal item (from memory_goal_items). action: send (a draft; this IS the person's "
        "approval, so only when they asked; title / body edit it first), done, dismiss, reopen. Sending "
        "goes out through their connected app and is recorded with the exact text."
    ),
)
async def memory_goal_item(item_id: str, action: str, title: Optional[str] = None, body: Optional[str] = None) -> dict:
    from ..goals.service import GoalRefused, item_action
    denied = _viewer_write_block()
    if denied:
        return denied
    bad = _invalid_id("item_id", item_id)
    if bad:
        return bad
    try:
        out = await item_action(_get_state()["store"], _customer_id(), item_id, action, actor="chat",
                                title=title, body=body)
    except GoalRefused as e:
        return {"error": e.message, "code": e.code}  # detail-ok: our own refusal text, no internals
    item = out.get("item") or {}
    return {"ok": True, "item_id": item_id, "status": item.get("status"), "ref": out.get("ref")}

# ---------------------------------------------------------------------------
# Lattices (2026-10-25; Q115 A, Q116 A, Q117 A): Crystal storage's tables.
# Same service as the console and goals (lattices/service.py). Writes are
# on the curation ledger, undoable for 24 hours; rows pay the lattice row
# cap, never the fact cap. Snapshot cells already holding a value change
# only on a correction the person asked for.
# ---------------------------------------------------------------------------

async def _lattice_write_block() -> tuple[Optional[dict], Any]:
    """Viewer and trial-expiry walls for lattice writes (the row cap is in
    the service). Returns (denial, customer)."""
    denied = _viewer_write_block()
    if denied:
        return denied, None
    cid = _current_customer_id.get()
    if not cid:
        return None, None
    from ..ingress.auth import trial_expired
    try:
        customer = await _get_state()["store"].get_customer_by_id(cid)
    except Exception:  # noqa: BLE001
        return _admission_unavailable(), None
    if customer is not None and trial_expired(customer):
        return {"error": ("trial expired: memory writes are paused; your lattices are safe and readable; "
                          "upgrade in the console to resume writing"), "code": "trial_expired"}, None
    return None, customer


def _lattice_refused(e) -> dict:
    out = {"error": e.message, "code": e.code}  # detail-ok: our own refusal text, no internals
    if e.detail:
        out["detail"] = e.detail
    return out


@_mcp_tool(
    name="memory_lattices",
    description=(
        "The person's lattices: named tables in Crystal storage for many records with the same fields "
        "(games, prices, stats, contacts). Each with its columns (type; kind snapshot = fixed once true, "
        "live = changes and keeps history; entity = browsable), the key columns that identify a row, and "
        "its row count. Use a lattice, not memory_store, for anything that is rows."
    ),
)
async def memory_lattices() -> dict:
    from ..lattices import service as lat
    view = await lat.list_view(_get_state()["store"], _customer_id())
    return {"lattices": view, "count": len(view)}


@_mcp_tool(
    name="memory_lattice_rows",
    description=(
        "Read rows of one lattice (id or exact name). filters: {column: value} for equality, or "
        "[{column, op: eq | contains | gte | lte, value}]; search: text in any text column; sort: a column, "
        "descending true for highest first; limit up to 200, offset to page. history: true adds each "
        "cell's as-of date, source and earlier values. Rows list 'stale' live cells past their refresh."
    ),
)
async def memory_lattice_rows(lattice: str, filters: Optional[Any] = None, search: Optional[str] = None,
                              sort: Optional[str] = None, descending: bool = False, limit: int = 50,
                              offset: int = 0, history: bool = False) -> dict:
    from ..lattices import service as lat
    try:
        return await lat.query(_get_state()["store"], _customer_id(), lattice, filters=filters, search=search,
                               sort=sort, descending=descending, limit=limit, offset=offset, history=history)
    except lat.LatticeRefused as e:
        return _lattice_refused(e)


@_mcp_tool(
    name="memory_lattice_entities",
    description=(
        "Browse one lattice by entity: each entity column's values (a team, a player) with how many rows "
        "hold each. column narrows to one column; search narrows the values."
    ),
)
async def memory_lattice_entities(lattice: str, column: Optional[str] = None, search: Optional[str] = None,
                                  limit: int = 30) -> dict:
    from ..lattices import service as lat
    try:
        return await lat.entities(_get_state()["store"], _customer_id(), lattice, column, search=search, limit=limit)
    except lat.LatticeRefused as e:
        return _lattice_refused(e)


@_mcp_tool(
    name="memory_lattice_aggregate",
    description=(
        "Analyze one lattice: group rows by up to three columns and compute count plus sum, avg, min or max "
        "of number columns, e.g. group_by ['team'], metrics [{fn: 'avg', column: 'points'}] for points per "
        "game by team. filters and search as in memory_lattice_rows. sort: 'count' or a metric label such "
        "as 'avg_points' (highest first unless descending is false). No group_by gives totals over every "
        "matching row. Use this, not row-by-row reading, for any average, total, record or ranking."
    ),
)
async def memory_lattice_aggregate(lattice: str, group_by: Optional[list] = None, metrics: Optional[list] = None,
                                   filters: Optional[Any] = None, search: Optional[str] = None,
                                   sort: Optional[str] = None, descending: bool = True, limit: int = 50) -> dict:
    from ..lattices import service as lat
    try:
        return await lat.aggregate(_get_state()["store"], _customer_id(), lattice, group_by=group_by, metrics=metrics,
                                   filters=filters, search=search, sort=sort, descending=descending, limit=limit)
    except lat.LatticeRefused as e:
        return _lattice_refused(e)


@_mcp_tool(
    name="memory_lattice_create",
    description=(
        "Create a lattice. columns: [{key (lowercase_with_underscores), label, type: text | number | date "
        "(YYYY-MM-DD) | bool | url, kind: snapshot (default; true once: a date, a final score) | live "
        "(changes: a season average, a price), entity: true for columns to browse by (text only), "
        "refresh_minutes: how fresh a live column must stay}]. key_columns: the snapshot columns that "
        "identify a row, e.g. ['date', 'home', 'away']. Check memory_lattices first; never make a second "
        "lattice for the same records."
    ),
)
async def memory_lattice_create(name: str, columns: list, key_columns: list, description: str = "") -> dict:
    from ..lattices import service as lat
    denied, _ = await _lattice_write_block()
    if denied:
        return denied
    try:
        out = await lat.create(_get_state()["store"], _customer_id(), name=name, columns=columns,
                               key_columns=key_columns, description=description, actor=_actor())
    except lat.LatticeRefused as e:
        return _lattice_refused(e)
    return {"ok": True, "lattice": lat.describe(out)}


@_mcp_tool(
    name="memory_lattice_write",
    description=(
        "Add or update rows of one lattice, matched by its key columns (up to 500 rows per call). Empty "
        "cells fill; live cells take the new value and keep the old one in history; snapshot cells that "
        "already hold a value are kept and reported under 'locked'. correct: true overwrites snapshot "
        "cells too, only when the person said the stored value is wrong. source: where the values came "
        "from. reason: why, for the ledger. Undoable for 24 hours (memory_undo)."
    ),
)
async def memory_lattice_write(lattice: str, rows: list, source: str, reason: str, correct: bool = False) -> dict:
    from ..cognition import controls
    from ..lattices import service as lat
    denied, customer = await _lattice_write_block()
    if denied:
        return denied
    bad = controls.reason_check(reason)
    if bad:
        return bad
    store, cid = _get_state()["store"], _customer_id()
    try:
        out = await lat.write_rows(store, cid, lattice, rows, source=source, actor=_actor(), customer=customer,
                                   correct=bool(correct))
    except lat.LatticeRefused as e:
        return _lattice_refused(e)
    action_id = await lat.ledger(store, cid, out, actor=_actor(), reason=reason)
    return {**lat.public(out), "action_id": action_id}


@_mcp_tool(
    name="memory_lattice_import",
    description=(
        "Move a table from a public web page into a lattice, done on the server (up to 2,000 rows per call). "
        "table: its index or id on the page. columns: {lattice column: table header, or {from: header, link: "
        "true for the cell's link, pattern: regex whose first group is kept}}. constants: a value for every row "
        "(a season, a source). Dates in common formats become YYYY-MM-DD. Matched by the lattice's key columns, "
        "so importing again updates. reason: why, for the ledger. next_offset continues a longer table."
    ),
)
async def memory_lattice_import(lattice: str, url: str, table: Any, columns: dict, reason: str,
                                constants: Optional[dict] = None, offset: int = 0) -> dict:
    from ..cognition import controls
    from ..lattices import service as lat
    denied, customer = await _lattice_write_block()
    if denied:
        return denied
    bad = controls.reason_check(reason)
    if bad:
        return bad
    try:
        return await lat.import_table(_get_state()["store"], _customer_id(), lattice, url=url, table=table,
                                      columns=columns, constants=constants, offset=offset, actor=_actor(),
                                      customer=customer, reason=reason)
    except lat.LatticeRefused as e:
        return _lattice_refused(e)


@_mcp_tool(
    name="memory_lattice_change",
    description=(
        "Change one lattice's shape: name, description, add_columns (same form as create), column_changes "
        "{column: {label, kind, entity, refresh_minutes}} (a column's key and type never change; key columns "
        "stay snapshot)."
    ),
)
async def memory_lattice_change(lattice: str, name: Optional[str] = None, description: Optional[str] = None,
                                add_columns: Optional[list] = None, column_changes: Optional[dict] = None) -> dict:
    from ..lattices import service as lat
    denied, _ = await _lattice_write_block()
    if denied:
        return denied
    try:
        out = await lat.update(_get_state()["store"], _customer_id(), lattice, name=name, description=description,
                               add_columns=add_columns, column_changes=column_changes)
    except lat.LatticeRefused as e:
        return _lattice_refused(e)
    return {"ok": True, "lattice": lat.describe(out)}


@_mcp_tool(
    name="memory_lattice_delete",
    description=(
        "Delete rows of a lattice (row_ids from memory_lattice_rows), or the whole lattice with every row "
        "(whole: true). Only when the person asked. There is no undo."
    ),
)
async def memory_lattice_delete(lattice: str, row_ids: Optional[list] = None, whole: bool = False) -> dict:
    from ..lattices import service as lat
    denied = _viewer_write_block()
    if denied:
        return denied
    store, cid = _get_state()["store"], _customer_id()
    try:
        if whole:
            return {"ok": True, **(await lat.delete(store, cid, lattice))}
        return await lat.delete_rows(store, cid, lattice, row_ids)
    except lat.LatticeRefused as e:
        return _lattice_refused(e)

# ---------------------------------------------------------------------------
# Tool annotations (2026-09-27, Anthropic Connectors Directory requirement:
# every tool carries a title and the applicable readOnlyHint /
# destructiveHint). One table beats 21 decorator edits: names are checked
# against the registered tools at import, so a new tool without a row here
# fails at startup, and tests/test_mcp_tool_annotations.py pins the whole
# contract. Classification: retrieval and reporting tools are read-only;
# writers say so explicitly; the two forget tools are the destructive pair
# (ledgered delete: the fact ledger keeps the before-text for audit, there
# is no restore). openWorldHint False everywhere: this server talks only
# to its own bank. memory_synthesize deliberately carries the explicit
# False/False pair rather than a read-only claim its implementation has
# not been audited for.
# ---------------------------------------------------------------------------

def _ro(title: str) -> ToolAnnotations:
    return ToolAnnotations(
        title=title, readOnlyHint=True, destructiveHint=False,
        idempotentHint=True, openWorldHint=False,
    )


def _wr(title: str) -> ToolAnnotations:
    return ToolAnnotations(
        title=title, readOnlyHint=False, destructiveHint=False,
        openWorldHint=False,
    )


_TOOL_ANNOTATIONS: dict[str, ToolAnnotations] = {
    "memory_search": _ro("Search memory"),
    "memory_search_documents": _ro("Search ingested documents"),
    "memory_outline": _ro("Outline a subject"),
    "memory_keys": _ro("Browse memory keys"),
    "memory_synthesize": _wr("Synthesize across memories"),
    "memory_recall": _ro("Recall from memory"),
    "memory_store": _wr("Store a memory"),
    "memory_forget": ToolAnnotations(
        title="Retire a memory", readOnlyHint=False, destructiveHint=True,
        idempotentHint=True, openWorldHint=False,
    ),
    "memory_ingest": _wr("Ingest a document"),
    "memory_learn": _wr("Learn from an outcome"),
    "memory_stats": _ro("Memory bank statistics"),
    "memory_list": _ro("List memories"),
    "memory_export": _ro("Export memories"),
    "memory_import": _wr("Import memories"),
    "memory_conflicts": _ro("List knowledge conflicts"),
    "memory_gaps": _ro("List knowledge gaps"),
    "memory_record_gap": _wr("Record a knowledge gap"),
    "memory_schedule": _wr("Schedule a run"),
    "memory_schedules": _ro("List scheduled runs"),
    "memory_unschedule": ToolAnnotations(
        title="Delete a scheduled run", readOnlyHint=False, destructiveHint=True,
        idempotentHint=True, openWorldHint=False,
    ),
    # The controls (2026-10-23).
    "memory_assumptions": _ro("Review assumptions"),
    "memory_actions": _ro("The curation ledger"),
    "memory_decide_conflict": _wr("Settle a conflict"),
    "memory_reopen_conflict": _wr("Reopen a conflict"),
    "memory_approve_assumption": _wr("Approve an assumption"),
    "memory_reject_assumption": _wr("Reject an assumption"),
    "memory_verify_assumption": _wr("Verify an assumption"),
    "memory_research_gap": _wr("Research a gap"),
    "memory_dismiss_gap": _wr("Dismiss a gap"),
    "memory_reopen_gap": _wr("Reopen a gap"),
    "memory_answer_gap": _wr("Answer a gap"),
    "memory_edit_fact": _wr("Edit a fact"),
    "memory_retire_fact": ToolAnnotations(
        title="Retire a fact", readOnlyHint=False, destructiveHint=True,
        idempotentHint=True, openWorldHint=False,
    ),
    "memory_set_tier": _wr("Set a crystal's tier"),
    "memory_reprobe_fact": _wr("Re-probe a fact"),
    "memory_undo": _wr("Undo an act"),
    "memory_steward": _wr("The Steward switch"),
    # Automation rules (Q101, 2026-10-23).
    "memory_rules": _ro("The automation rules"),
    "memory_rule_add": _wr("Add an automation rule"),
    "memory_rule_remove": ToolAnnotations(
        title="Delete an automation rule", readOnlyHint=False, destructiveHint=True,
        idempotentHint=True, openWorldHint=False,
    ),
    "memory_rule_move": _wr("Reorder an automation rule"),
    "memory_rule_toggle": _wr("Switch an automation rule"),
    # Connected apps (2026-10-24).
    "memory_connections": _ro("Connected apps"),
    "memory_connection_set": _wr("Set what Crystal does with an app"),
    "memory_goals": _ro("Goals"),
    "memory_goal_add": _wr("Add a goal"),
    "memory_goal_update": _wr("Change a goal"),
    "memory_goal_items": _ro("Goal items"),
    "memory_goal_item": ToolAnnotations(
        title="Act on a goal item", readOnlyHint=False, destructiveHint=False, openWorldHint=True,
    ),
    # Lattices (2026-10-25).
    "memory_lattices": _ro("Lattices"),
    "memory_lattice_rows": _ro("Read lattice rows"),
    "memory_lattice_entities": _ro("Browse a lattice by entity"),
    "memory_lattice_aggregate": _ro("Analyze a lattice"),
    "memory_lattice_create": _wr("Create a lattice"),
    "memory_lattice_write": _wr("Write lattice rows"),
    "memory_lattice_import": ToolAnnotations(
        title="Import a web table into a lattice", readOnlyHint=False, destructiveHint=False, openWorldHint=True,
    ),
    "memory_lattice_change": _wr("Change a lattice"),
    "memory_lattice_delete": ToolAnnotations(
        title="Delete lattice rows", readOnlyHint=False, destructiveHint=True,
        idempotentHint=True, openWorldHint=False,
    ),
    "remember": _wr("Remember this"),
    "recall": _ro("Recall"),
    "status": _ro("Memory status"),
    "forget": ToolAnnotations(
        title="Forget a memory", readOnlyHint=False, destructiveHint=True,
        idempotentHint=True, openWorldHint=False,
    ),
}


def _apply_tool_annotations() -> None:
    """Stamp annotations onto the registered tools. A REGISTERED tool
    without a table row fails LOUD at import (the directory contract:
    nothing unannotated ever serves). Table rows for tools the
    CC_MCP_TOOLSET knob excluded from this registration are fine — the
    consumer toolset registers four of the twenty-one."""
    tools = mcp._tool_manager._tools  # pinned SDK (mcp==1.30.0, exact)
    missing = set(tools) - set(_TOOL_ANNOTATIONS)
    if missing:
        raise RuntimeError(
            f"tool annotation drift: unannotated={sorted(missing)}"
        )
    for tool_name, tool in tools.items():
        tool.annotations = _TOOL_ANNOTATIONS[tool_name]


_apply_tool_annotations()


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
