"""FastAPI application entry point — v2 port (May 2026).

This module is intentionally minimal. The v1 monolith at
`crystal-cache-v1/src/crystal_cache/app.py` was ~4900 lines: lifespan,
three background workers inline, all route handlers, all helpers,
error handlers. v2 splits that into focused modules:

  - workers/         → the three background workers (Phase 6 Wave A)
  - endpoints/       → route handlers grouped by concern (this wave)
  - cognition/       → multi-agent cognition environment (Phase 6 Wave C)
  - app.py           → app construction, lifespan, error handlers (this file)

The agent reframe (D-A1, D-A2 in PROJECT_LEDGER.md) shows up here in
two ways:

  1. /v1/chat/completions is the proxy adapter (endpoints/chat_proxy.py)
     in pass-through mode. v1 behavior preserved verbatim per D-A2.
  2. /v1/agent/messages is the new flagship endpoint (endpoints/agent.py).
     Stub for now; full implementation lands in Phase 7.5.

Both endpoints coexist; existing v1 customers keep working unchanged.

Phase 6 of the v2 port. Phase 6.5 added the inspector-tier endpoints,
the OpenAI-compat surface, and the frontend SPA mount. Phase 6 Wave C
ported the cognition package and mounted its admin router here.
Phase 7 Wave 7F (this update) wired ShadowEvaluator and Mem0 into
the lifespan now that their modules have ported.
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .decomposer import (
    DslConfigStore,
    HostedLLMDecomposer,
    TracingDecomposer,
    build_trace_writer_from_settings,
)
from .decomposer.base import Decomposer, DecomposerError
from .dsl.schema.loader import set_schema_loader
from .execution.shadow_evaluator import ShadowEvaluator
from .infrastructure.decoder_loader import (
    DecoderLoader,
    is_decoder_enabled,
    set_decoder_loader,
)
from .infrastructure.metadata_store import set_metadata_store
from .ingress.auth import platform_admin_error
from .workers.idle import note_request
from .ingress.errors import CrystalCacheError, build_error_envelope
from .llm import get_llm_client
from .retrieval.mem0_session import init_mem0
from .runtime import build_core_runtime
# WS C — in-process MCP memory server (mounted at /mcp) and the tool-state
# injector it shares with the agent tool registry.
from .agent.mcp_server import build_mcp_asgi_app, mcp as mcp_server
from .agent.tools.retrievers import set_tool_state

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Process startup + shutdown.

    The structure mirrors v1's lifespan; Phase 7 Wave 7F (2026-05-25)
    wired the previously-None ShadowEvaluator and Mem0 placeholders to
    real instances now that `execution/shadow_evaluator.py` (Wave 7C)
    and `retrieval/mem0_session.py` (Wave 7F) are ported.

    1. Workers are imported from `workers/` and invoked with explicit
       dependencies, not defined inline.
    2. ShadowEvaluator is constructed unconditionally; the evaluator
       itself is cheap, sampling decides which requests trigger a real
       shadow call.
    3. Mem0 is gated on CC_MEM0_ENABLED — init_mem0 returns None when
       disabled or when initialization fails. Endpoints check for None
       and skip Mem0 calls.

    This lets v2 boot today on the substrate that's already ported.
    """
    logger.info("crystal_cache.startup", environment=settings.environment)

    # --- Production safety gate (WS D / D.1 + D.4): fail closed on missing
    # secrets. In production the platform-admin surface MUST be locked and
    # stored key hashes MUST be peppered; we refuse to boot rather than come
    # up with an open admin surface or brute-forceable hashes. Dev/self-host
    # are exempt (the gate stays open / the SHA-256 fallback applies).
    if settings.is_production:
        _missing = [
            name
            for name, val in (
                ("CC_ADMIN_API_KEY", settings.admin_api_key),
                ("CC_API_KEY_PEPPER", settings.api_key_pepper),
            )
            if not (val or "").strip()
        ]
        if _missing:
            raise RuntimeError(
                "Refusing to start in production without: "
                + ", ".join(_missing)
                + ". Set them, or run with CC_ENVIRONMENT != production."
            )

    # --- Core runtime (shared with the worker process — see runtime.py) ---
    # Builds the store (+ singleton), encoder, both vector stores, the
    # schema loader (+ singleton), and seeds crystal_types. The standalone
    # worker entrypoint builds the identical bundle so the two can't drift.
    core = await build_core_runtime()
    store = core.store
    app.state.metadata_store = store

    # --- Decoder loader (April 2026 item 6) ---
    # Gated on CC_ENABLE_DECODER. Same logic as v1.
    if is_decoder_enabled():
        try:
            loader = DecoderLoader()
            set_decoder_loader(loader)
            app.state.decoder_loader = loader
            logger.info(
                "decoder_loader.ready",
                device=loader.device,
                text_v1=loader.text_v1 is not None,
                bind_v1=loader.bind_v1 is not None,
            )
        except ImportError as e:
            logger.warning(
                "decoder_loader.disabled",
                reason="missing dependency",
                error=str(e),
            )
            app.state.decoder_loader = None
        except Exception as e:
            logger.error(
                "decoder_loader.init_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            app.state.decoder_loader = None
    else:
        logger.info(
            "decoder_loader.skipped",
            reason="CC_ENABLE_DECODER not set",
        )
        app.state.decoder_loader = None

    # --- Encoder + vector stores + schema loader (from core runtime) ---
    app.state.prompt_encoder = core.encoder
    app.state.vector_store = core.vector_store
    app.state.fact_vector_store = core.fact_vector_store
    app.state.vector_index = core.vector_index
    app.state.schema_loader = core.schema_loader

    # --- Agent / MCP tool state (agent/tools/retrievers.set_tool_state) ---
    # The in-process MCP server (agent/mcp_server.py) builds no Agent, so it
    # needs the process singletons injected once here for its bridged tools
    # to run. The Agent class also sets this per-construction; setting it at
    # startup makes both paths work. Internal LLM work (cognition_run, depth
    # synthesis, the scans) routes through the provider-neutral seam
    # (llm/client.py) and no longer needs a client in tool state.
    set_tool_state({
        "store": store,
        "vector_store": core.vector_store,
        "fact_vector_store": core.fact_vector_store,
        "vector_index": core.vector_index,
        "encoder": core.encoder,
    })

    # --- DSL config store + decomposer ---
    app.state.dsl_config_store = DslConfigStore(metadata_store=store)
    app.state.decomposer = _build_decomposer()

    # --- Shadow evaluator (Wave 7F: wired now that execution/ ported) ---
    # The evaluator itself is cheap; the shadow CALLS it makes are not.
    # Sampling gates whether any particular request triggers one.
    app.state.shadow_evaluator = ShadowEvaluator()

    # --- Mem0 (Wave 7F: wired now that mem0_session.py ported) ---
    # Gated on CC_MEM0_ENABLED env var. init_mem0 returns None when
    # the gate is off OR initialization fails (no API key, mem0 package
    # missing, qdrant unable to open). Non-fatal — endpoints that
    # consume Mem0 (chat_proxy's post-response add_conversation_turn)
    # check for None first.
    try:
        app.state.mem0 = init_mem0(anthropic_api_key=settings.anthropic_api_key)
    except Exception as e:
        # init_mem0 already catches everything internally and returns
        # None on failure; this outer try is belt-and-suspenders so
        # an unexpected mem0 import-time error can't break boot.
        logger.warning("mem0.startup_failed", error=str(e))
        app.state.mem0 = None

    # --- Background workers ---
    # In-process workers are the single-process deployment. When
    # CC_RUN_WORKERS=false (the docker-compose split) the API runs WITHOUT
    # them and a dedicated worker process (`python -m crystal_cache.workers`)
    # runs the same loops against the shared database. Spawned tasks are
    # collected so shutdown drains whichever set was started; the shared
    # shutdown event is always present on app.state.
    shutdown_event = asyncio.Event()
    app.state.worker_shutdown_event = shutdown_event

    worker_tasks: list[tuple[asyncio.Task, str]] = []
    if settings.run_workers:
        from .workers import (
            run_crystallization_worker,
            run_drive_sync_worker,
            run_cognition_worker,
            run_metacognition_worker,
        )

        worker_tasks.append((
            asyncio.create_task(run_crystallization_worker(
                store=store,
                encoder=app.state.prompt_encoder,
                vector_store=app.state.vector_store,
                shutdown_event=shutdown_event,
            )),
            "crystallization",
        ))
        worker_tasks.append((
            asyncio.create_task(run_drive_sync_worker(
                store=store,
                shutdown_event=shutdown_event,
            )),
            "drive_sync",
        ))
        worker_tasks.append((
            asyncio.create_task(run_cognition_worker(
                store=store,
                fact_vector_store=app.state.fact_vector_store,
                encoder=app.state.prompt_encoder,
                shutdown_event=shutdown_event,
            )),
            "cognition",
        ))

        # Phase 10C metacognition worker (P0.87). Gated on
        # settings.enable_metacognition_worker (default True). The shadow-
        # critic pass routes through the provider-neutral seam and no-ops
        # when no provider is configured (synthesis pass still runs).
        if settings.enable_metacognition_worker:
            worker_tasks.append((
                asyncio.create_task(run_metacognition_worker(
                    store=store,
                    shutdown_event=shutdown_event,
                )),
                "metacognition",
            ))
            logger.info(
                "metacog_worker.wired",
                provider_ready=get_llm_client().is_ready(),
            )
        else:
            logger.info(
                "metacog_worker.disabled",
                reason="settings.enable_metacognition_worker=False",
            )

        logger.info("workers.started", count=len(worker_tasks))
    else:
        logger.info(
            "workers.disabled",
            reason="CC_RUN_WORKERS=false (API-only process)",
        )

    try:
        # The MCP server is mounted as a sub-app; its streamable-HTTP session
        # manager must be running for /mcp requests to work, and a mounted
        # sub-app's own lifespan does not fire on its own — so we enter it here.
        async with mcp_server.session_manager.run():
            yield
    finally:
        # Signal shutdown; drain whichever workers were started.
        shutdown_event.set()
        for task, name in worker_tasks:
            try:
                await asyncio.wait_for(task, timeout=10)
            except asyncio.TimeoutError:
                logger.warning("worker.shutdown_timeout", worker=name)
                task.cancel()

        logger.info("crystal_cache.shutdown")

        # Close the decomposer's HTTP client if it owns one.
        decomp = getattr(app.state, "decomposer", None)
        if decomp is not None and hasattr(decomp, "aclose"):
            try:
                await decomp.aclose()
            except Exception as e:
                logger.warning("decomposer.close_failed", error=str(e))

        await store.dispose()

        # Clear process-wide singletons so test re-imports don't point
        # at disposed engines.
        set_metadata_store(None)  # type: ignore[arg-type]
        set_schema_loader(None)
        set_decoder_loader(None)


def _build_decomposer() -> Optional[Decomposer]:
    """Construct the process's decomposer from settings (verbatim from v1).

    Returns None when no Groq key is configured; this is the expected
    state for local dev and is NOT an error.
    """
    if not settings.groq_api_key:
        logger.info(
            "decomposer.disabled",
            reason="no GROQ_API_KEY/CC_GROQ_API_KEY configured",
            anthropic_key_loaded=bool(settings.anthropic_api_key),
        )
        return None
    try:
        inner: Decomposer = HostedLLMDecomposer()
    except DecomposerError as e:
        logger.warning("decomposer.init_failed", error=str(e))
        return None

    writer = build_trace_writer_from_settings()
    if writer is not None:
        inner = TracingDecomposer(
            inner,
            writer,
            tenant_id_fn=lambda ctx: ctx.get("tenant_id", "unknown"),
        )
        logger.info(
            "decomposer.tracing_enabled",
            path=str(settings.decomposer_trace_path),
        )
    return inner


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Crystal Cache",
    description=(
        "Semantic preprocessing cache for enterprise LLM stacks. "
        "v2 architecture per docs/AGENT_ARCHITECTURE.md."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# WS C — mount the in-process MCP memory server at /mcp. This also creates
# mcp_server.session_manager, which the lifespan above enters. Authentication
# (customer API key -> customer_id) is handled by the middleware wrapping the
# sub-app; see agent/mcp_server.py.
app.mount("/mcp", build_mcp_asgi_app())


# ---------------------------------------------------------------------------
# Rate limiting (C3, 2026-07-03): sliding-window guard on auth-adjacent and
# expensive routes, keyed by bearer-token hash (else client IP). In-process
# by design — see ingress/rate_limit.py for the single-instance rationale
# and the Redis follow-on for multi-instance hosted. Registered before the
# other middlewares so a limited burst is rejected as early as possible.
if settings.enable_rate_limiting:
    from .ingress.rate_limit import build_rate_limit_middleware

    app.middleware("http")(build_rate_limit_middleware(
        auth_per_minute=settings.rate_limit_auth_per_minute,
        expensive_per_minute=settings.rate_limit_expensive_per_minute,
    ))


# ---------------------------------------------------------------------------
# Idle-activity stamp (workers/idle.py): substantive traffic only — /v1/*
# paths — so the cognition worker's opportunistic idle work defers while the
# API is actively serving, without an open admin dashboard's polling ever
# counting as load. Registered BEFORE the admin guard so it stamps even
# requests the guard would reject (a rejected burst is still load).
@app.middleware("http")
async def idle_activity_stamp(request: Request, call_next):
    if request.url.path.startswith("/v1/"):
        note_request()
    return await call_next(request)


# ---------------------------------------------------------------------------
# WS D / D.1 + D.2 — platform-admin guard.
# ---------------------------------------------------------------------------
# Thin glue over ingress.auth.platform_admin_error, which holds the whole
# policy (which surfaces are protected, whether the gate is on, and the
# constant-time key check) in one auditable, unit-testable place. Covers every
# /admin/api/* route (admin, cognition, metacognition routers) plus customer
# minting (POST /v1/customers); the /admin SPA shell stays public. Open in
# dev/self-host until a key is set; always enforced in production (see the
# lifespan boot guard). Middleware runs outside FastAPI's Depends system, so it
# reads the header off the raw request. 401 shape matches the other auth
# failures ({"detail": ...}).
@app.middleware("http")
async def platform_admin_guard(request: Request, call_next):
    err = platform_admin_error(
        request.method,
        request.url.path,
        request.headers.get("authorization")
        or request.headers.get("Authorization"),
    )
    if err is not None:
        return JSONResponse(
            status_code=err[0],
            content={"detail": err[1]},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Error handlers — verbatim from v1 (Phase 1.5.5)
# ---------------------------------------------------------------------------
#
# Three handlers translate raised exceptions into the OpenAI-compatible
# error envelope. See v1 app.py for the full design rationale; verbatim
# port here.

@app.exception_handler(CrystalCacheError)
async def crystal_cache_error_handler(
    request: Request,
    exc: CrystalCacheError,
) -> JSONResponse:
    headers: dict[str, str] = {}
    if exc.http_status == 429:
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is not None:
            headers["Retry-After"] = str(retry_after)
    if exc.http_status == 401:
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse(
        status_code=exc.http_status,
        content=build_error_envelope(
            exc.message,
            error_type=exc.error_type,
            param=exc.param,
            code=exc.code,
        ),
        headers=headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Translate Pydantic validation errors into the OpenAI envelope.

    Multi-error requests collapse to the first error's message and
    param. The SDK's BadRequestError doesn't expose multi-error
    surface and customer code overwhelmingly handles them one-at-a-
    time anyway.
    """
    errors = exc.errors()
    first = errors[0] if errors else {}
    loc = first.get("loc", ())
    if loc and loc[0] in ("body", "query", "path", "header"):
        loc = loc[1:]
    param: Optional[str] = (
        ".".join(str(p) for p in loc) if loc else None
    )
    msg = first.get("msg", "Invalid request")
    return JSONResponse(
        status_code=400,
        content=build_error_envelope(
            f"Invalid request: {msg}",
            error_type="invalid_request_error",
            param=param,
        ),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    """Wrap bare HTTPException raises into the OpenAI envelope."""
    type_map = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        409: "conflict_error",
        429: "rate_limit_error",
    }
    error_type = type_map.get(exc.status_code, "api_error")
    headers = dict(exc.headers or {})

    # Pass-through for callers that already produced an envelope.
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail,
            headers=headers,
        )

    message = (
        exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=build_error_envelope(message, error_type=error_type),
        headers=headers,
    )


# ---------------------------------------------------------------------------
# Helper: structured 501 (used by stub endpoints in this Phase 6)
# ---------------------------------------------------------------------------

def not_implemented(feature: str, doc_ref: str) -> JSONResponse:
    """Return a structured 501 response.

    Used by endpoints that exist in v1 but aren't ported yet, AND by
    the agent endpoint stub (Phase 7.5 fills it in).
    """
    return JSONResponse(
        status_code=501,
        content={
            "error": {
                "type": "not_implemented",
                "feature": feature,
                "message": f"'{feature}' is scaffolded but not yet implemented.",
                "doc_ref": doc_ref,
            }
        },
    )


# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------
#
# All route handlers live in the endpoints/ package. Each module
# exports a `router: APIRouter` that we include here. Order doesn't
# matter for behavior but is kept stable for readability.
#
# Important: the cognition router lives in `cognition/api.py` (Phase 6
# Wave C) and uses prefix `/admin/api/cognition`. It MUST be registered
# before the SPA fallback below — otherwise the catch-all
# `/admin/{full_path:path}` route would shadow `/admin/api/cognition/*`
# and the inspector would 503 instead of hitting the JSON endpoints.

from .endpoints import (
    admin,
    agent,
    chat_proxy,
    compliance,
    control,
    cost,
    customers,
    diagnostics,
    documents,
    drive,
    dsl_configs,
    feedback,
    groups,
    health,
    marketplace,
    openai_compat,
    operators,
    promotion,
    sdk,
    sessions,
    stubs,
)
from .cognition import api as cognition_api
from .metacognition import api as metacog_api

app.include_router(health.router)
app.include_router(customers.router)
app.include_router(operators.router)
app.include_router(promotion.router)  # Foundation F3 promotion API
app.include_router(chat_proxy.router)
app.include_router(agent.router)
app.include_router(feedback.router)
app.include_router(sdk.router)
app.include_router(groups.router)  # P3 groups — named sub-teams as grant targets
app.include_router(sessions.router)  # Foundation F4 session registry API
app.include_router(control.router)  # Growth G2 control-plane command channel
app.include_router(cost.router)  # Growth G3 cost accounting + budgets
app.include_router(marketplace.router)  # Growth G4 marketplace shard ledger
app.include_router(documents.router)
app.include_router(drive.router)
app.include_router(compliance.router)
app.include_router(admin.router)

# Phase 6.5 additions: restore endpoints v1 had that Wave B dropped.
app.include_router(openai_compat.router)   # P1.1: /v1/models, /v1/completions
app.include_router(diagnostics.router)     # P1.3: /api/crystals, /api/edits
app.include_router(dsl_configs.router)     # P1.4: /api/dsl_configs/*
app.include_router(stubs.router)           # P1.2: 501 stubs for inspector

# Phase 6 Wave C: cognition admin endpoints (/admin/api/cognition/*).
# MUST be registered before the SPA fallback below.
app.include_router(cognition_api.router)

# Phase 10.5: metacognition admin endpoints (/admin/api/metacognition/*).
# Substrate review surface (D-MCR-13 V1, MCR §9, §11 Q7). MUST be
# registered before the SPA fallback below.
app.include_router(metacog_api.router)


# ---------------------------------------------------------------------------
# Frontend SPA mount — /admin/* (Phase 6.5 P1.5)
# ---------------------------------------------------------------------------
#
# v1 mounts the inspector's bundled frontend at /admin via StaticFiles
# plus an SPA fallback. The frontend repo lives separately; if it's
# been built (frontend/dist/ exists), we serve the SPA. If not, we
# return a friendly 503 with build instructions so an inspector visit
# doesn't 404 silently.
#
# We add the route registrations LAST so they don't shadow any earlier
# /admin/api/* routes (admin.router and cognition_api.router were both
# registered above).

_FRONTEND_DIST = (
    Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
)

if _FRONTEND_DIST.exists() and (_FRONTEND_DIST / "index.html").exists():
    # Built bundle present — serve static assets + SPA fallback for
    # client-side routing.
    app.mount(
        "/admin/assets",
        StaticFiles(directory=str(_FRONTEND_DIST / "assets")),
        name="admin-assets",
    )

    @app.get("/admin")
    @app.get("/admin/{full_path:path}")
    async def serve_admin_spa(full_path: str = "") -> FileResponse:
        """SPA fallback — return index.html for any path under /admin
        that doesn't match an API route, an asset file, or one of the
        /admin/api/* routes registered earlier."""
        return FileResponse(str(_FRONTEND_DIST / "index.html"))

    logger.info("frontend.mounted", path=str(_FRONTEND_DIST))
else:
    # No built bundle. Serve a friendly 503 from /admin so a clueless
    # visit doesn't 404. Matches v1 verbatim.
    @app.get("/admin")
    @app.get("/admin/{full_path:path}")
    async def admin_not_built(full_path: str = "") -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Inspector UI not built",
                "message": (
                    "The inspector frontend bundle is not present at "
                    f"{_FRONTEND_DIST}. Build it from the frontend "
                    "repo (npm run build) and copy the dist/ output "
                    "into this path, then restart the server."
                ),
                "expected_path": str(_FRONTEND_DIST),
            },
        )

    logger.info(
        "frontend.not_built",
        path=str(_FRONTEND_DIST),
        note="serving 503 fallback at /admin",
    )
