"""Endpoint smoke test — Workstream 0.1.

Goal: prove every HTTP route resolves to REAL v2 functionality — catch a
handler calling a store method that doesn't exist (a request-time 500 that
app-boot can't catch), and pin which routes are intentional 501 stubs. This
is the "the endpoints were built before v2 — do they still point at real
functionality?" check.

Design: mount every router (minus Drive, which needs Google OAuth + external
calls) on a test app wired to the conftest in-memory store + stub encoder,
then:
  - sweep every GET route (path/query params substituted) — the read surface
    is where "built before v2" staleness would hide;
  - hit a few LLM-FREE POSTs (store / documents-create / feedback) to confirm
    the write path resolves end-to-end.

Only an UNEXPECTED 5xx fails the test. 2xx / 3xx / 4xx and known-501 stubs are
recorded and acceptable — a dummy id (404), a missing field (422), or an
auth-surface mismatch (401/403) is not a wiring bug; a 500 is. LLM-driven
routes (chat/agent, crystallize, learn-fail, conflicts/scan) are intentionally
NOT exercised here — they need a live model and are covered by their own
tests.

Follows the SS2 pattern in test_phase11_5_smoke.py: httpx.AsyncClient +
ASGITransport, routers mounted on a bespoke app, get_metadata_store overridden
to the fixture store, app.state populated manually so no lifespan / workers.
"""
from __future__ import annotations

import re

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


# Routes that legitimately return 501 (scaffolded, not yet built) or 410
# (deliberately deprecated). Keyed by (METHOD, route-template-path) because
# GET /v1/crystals/{crystal_id} is REAL while DELETE on the same path is a
# stub. Any 501 seen outside this set is treated as an unexpected finding.
KNOWN_NOT_IMPLEMENTED: set[tuple[str, str]] = {
    ("GET", "/api/dashboard/overview"),
    ("GET", "/api/dashboard/crystals"),
    ("GET", "/api/verify/queue"),
    ("POST", "/api/verify/approve/{task_id}"),
    ("POST", "/api/verify/reject/{task_id}"),
    ("POST", "/api/documents"),
    ("GET", "/api/documents"),
    ("GET", "/api/settings"),
    ("POST", "/api/settings"),
    ("GET", "/api/crystals/{crystal_id}/history"),
    ("POST", "/v1/completions"),
}

# Only sweep our own surface; skip FastAPI's framework routes (docs/openapi).
_SURFACE_PREFIXES = ("/v1", "/api", "/admin")
_SURFACE_EXACT = {"/health", "/health/deep"}

_PARAM_RE = re.compile(r"\{(\w+)(?::[^}]+)?\}")


def _build_app(store, encoder, vector_store, fact_vector_store) -> FastAPI:
    """Mount every router (minus Drive) and wire app.state by hand.

    Mirrors app.py's router registration + the lifespan's app.state setup,
    but without spawning workers or loading the real (gtr) encoder.
    """
    from crystal_cache.decomposer import DslConfigStore
    from crystal_cache.dsl.schema.loader import SchemaLoader, set_schema_loader
    from crystal_cache.execution.shadow_evaluator import ShadowEvaluator
    from crystal_cache.infrastructure.metadata_store import (
        get_metadata_store,
        set_metadata_store,
    )

    from crystal_cache.endpoints import (
        admin, agent, chat_proxy, compliance, control, cost, customers,
        diagnostics, documents, dsl_configs, feedback, health, marketplace,
        openai_compat, operators, promotion, sdk, sessions, stubs,
    )
    from crystal_cache.cognition import api as cognition_api
    from crystal_cache.metacognition import api as metacog_api

    app = FastAPI()
    for r in (
        health, customers, operators, promotion, chat_proxy, agent, feedback,
        sdk, sessions, control, cost, marketplace, documents, compliance,
        admin, openai_compat, diagnostics, dsl_configs, stubs,
        cognition_api, metacog_api,
    ):
        app.include_router(r.router)

    # Dependency + global store both point at the in-memory fixture store.
    async def _get_test_store():
        return store

    app.dependency_overrides[get_metadata_store] = _get_test_store
    set_metadata_store(store)

    # app.state the handlers read (encoder / vector stores / dsl / shadow /
    # schema loader). decomposer + mem0 are legitimately None on this path.
    app.state.metadata_store = store
    app.state.prompt_encoder = encoder
    app.state.vector_store = vector_store
    app.state.fact_vector_store = fact_vector_store
    app.state.dsl_config_store = DslConfigStore(metadata_store=store)
    app.state.decomposer = None
    app.state.shadow_evaluator = ShadowEvaluator()
    app.state.mem0 = None
    schema_loader = SchemaLoader(metadata_store=store)
    set_schema_loader(schema_loader)
    app.state.schema_loader = schema_loader

    return app


def _categorize(method: str, template_path: str, status: int) -> str:
    """Map a response to one of: ok | redirect | known_stub | deprecated |
    client_error (acceptable) | UNEXPECTED_5XX | UNEXPECTED_STUB."""
    if 200 <= status < 300:
        return "ok"
    if 300 <= status < 400:
        return "redirect"
    if status == 501:
        return (
            "known_stub"
            if (method, template_path) in KNOWN_NOT_IMPLEMENTED
            else "UNEXPECTED_STUB"
        )
    if status == 410:
        return "deprecated"
    if 400 <= status < 500:
        return "client_error"  # dummy id / missing field / auth surface — fine
    return "UNEXPECTED_5XX"


@pytest.mark.asyncio
async def test_all_endpoints_resolve(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store
):
    """Sweep the full HTTP surface; fail only on an unexpected 5xx or an
    unexpected 501. Prints a per-route table for coverage visibility."""
    app = _build_app(
        store, semantic_encoder_stub, vector_store, fact_vector_store
    )

    # Team-root Bearer (admitted for require_customer / resolve_principal /
    # require_role) + one operator for operator-scoped reads.
    headers = {"Authorization": f"Bearer {customer.api_key}"}
    operator, _raw = await store.create_operator(
        team_id=customer.id, display_name="smoke-op",
    )

    # Path-param substitutions. Real ids where a handler should succeed; dummy
    # ids elsewhere (a 404 is acceptable and still exercises the handler body).
    path_subs = {
        "customer_id": customer.id,
        "operator_id": operator.id,
        "crystal_id": "smoke_dummy_crystal",
        "document_id": "smoke_dummy_doc",
        "session_id": "smoke_dummy_session",
        "item_id": "smoke_dummy_item",
        "conflict_id": "smoke_dummy_conflict",
        "task_id": "smoke_dummy_task",
        "env_id": "smoke_dummy_env",
        "fact_id": "smoke_dummy_fact",
        "name": "smoke_dummy_name",
    }
    # Query superset — required customer_id / operator_id satisfied; FastAPI
    # ignores params a route doesn't declare.
    query = {
        "customer_id": customer.id,
        "operator_id": operator.id,
        "limit": 5,
        "offset": 0,
    }

    def _fill(template: str) -> str:
        return _PARAM_RE.sub(
            lambda m: str(path_subs.get(m.group(1), "smoke_dummy")), template
        )

    results: list[tuple[str, str, int, str]] = []

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://smoke") as client:
        # --- GET sweep over every route on our surface ---
        # Enumerate via the OpenAPI schema (version-stable; gives method+path
        # directly, sidestepping route-object .methods quirks, and auto-excludes
        # framework routes like /openapi.json).
        paths = app.openapi().get("paths", {})
        get_paths = sorted(
            p for p, ops in paths.items()
            if "get" in ops
            and (p in _SURFACE_EXACT or p.startswith(_SURFACE_PREFIXES))
        )
        print(f"[smoke] openapi paths={len(paths)} GET-on-surface={len(get_paths)}")
        for template in get_paths:
            resp = await client.get(_fill(template), headers=headers, params=query)
            results.append(
                ("GET", template, resp.status_code,
                 _categorize("GET", template, resp.status_code))
            )

        # --- Curated LLM-free POSTs (confirm the write path resolves) ---
        post_cases = [
            ("/v1/store", {"key": "smoke test key", "value": "smoke test value"}),
            ("/v1/documents", {"text": "Smoke test document body.", "label": "smoke"}),
            ("/v1/feedback", {"sequence_id": "smoke_seq", "turn_index": 0, "signal": "up"}),
        ]
        for path, body in post_cases:
            resp = await client.post(path, headers=headers, json=body)
            results.append(
                ("POST", path, resp.status_code,
                 _categorize("POST", path, resp.status_code))
            )

    # --- Report + assert ---
    failures = [r for r in results if r[3] in ("UNEXPECTED_5XX", "UNEXPECTED_STUB")]

    summary: dict[str, int] = {}
    for _m, _p, _s, cat in results:
        summary[cat] = summary.get(cat, 0) + 1

    print("\n=== endpoint smoke ===")
    print(f"routes hit: {len(results)}  ->  {summary}")
    for m, p, s, cat in sorted(results, key=lambda r: (r[0], r[1])):
        flag = "  <-- FAIL" if cat in ("UNEXPECTED_5XX", "UNEXPECTED_STUB") else ""
        print(f"  {s} {cat:<14} {m:<6} {p}{flag}")

    assert not failures, (
        "Endpoints not pointing at real functionality (unexpected 5xx / 501):\n"
        + "\n".join(f"  {s} {m} {p}  [{cat}]" for m, p, s, cat in failures)
    )
