"""Security batch tests (2026-07-03, pre-launch): F1 OAuth CSRF,
C3 rate limiting, B4 cross-tenant isolation.

F1 — the Drive OAuth state is now an opaque single-use server-stored
nonce with a TTL: the callback rejects unknown, replayed, and stale
states, and never trusts a customer id embedded in the state string.

C3 — sliding-window limiter on auth-adjacent + expensive routes,
keyed per bearer token (hashed) else client IP; 429 with Retry-After.

B4 — cross-tenant isolation at the HTTP surface: customer A cannot
read customer B's crystals through the SDK endpoints. This is the CI
tripwire the security review asked for: any future endpoint change
that drops customer scoping fails here.

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from crystal_cache.ingress.rate_limit import (
    SlidingWindowLimiter,
    build_rate_limit_middleware,
)
from crystal_cache.models.crystal import Crystal


# --- F1: OAuth state store ops ----------------------------------------------

async def test_oauth_state_roundtrip_is_single_use(store, customer):
    await store.create_oauth_state("state-abc", customer.id)
    assert await store.consume_oauth_state("state-abc") == customer.id
    # Second redemption fails: single-use.
    assert await store.consume_oauth_state("state-abc") is None


async def test_oauth_state_unknown_is_rejected(store, customer):
    assert await store.consume_oauth_state("never-issued") is None


async def test_oauth_state_expiry(store, customer):
    await store.create_oauth_state("state-old", customer.id)
    # A zero-second TTL makes any age stale.
    assert (
        await store.consume_oauth_state("state-old", max_age_seconds=0)
        is None
    )
    # And the failed redemption consumed it — no second chance.
    assert (
        await store.consume_oauth_state("state-old", max_age_seconds=9999)
        is None
    )


# --- C3: limiter core ---------------------------------------------------------

def test_sliding_window_allows_then_blocks():
    lim = SlidingWindowLimiter(3, window_seconds=60)
    assert all(lim.allow("k", now=100.0 + i) for i in range(3))
    assert lim.allow("k", now=103.0) is False           # 4th inside window
    assert lim.allow("other", now=103.0) is True        # keys independent
    assert lim.allow("k", now=100.5 + 60.0) is True     # window slid


def test_zero_limit_means_unlimited():
    lim = SlidingWindowLimiter(0)
    assert all(lim.allow("k") for _ in range(50))


# --- C3: middleware behavior --------------------------------------------------

def _tiny_app() -> FastAPI:
    app = FastAPI()
    app.middleware("http")(build_rate_limit_middleware(
        auth_per_minute=2, expensive_per_minute=2,
    ))

    @app.post("/v1/customers")
    async def create():  # auth class
        return {"ok": True}

    @app.get("/v1/crystals")
    async def unlimited():  # neither class
        return {"ok": True}

    return app


async def test_auth_route_limited_and_general_route_not():
    app = _tiny_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        r1 = await client.post("/v1/customers")
        r2 = await client.post("/v1/customers")
        r3 = await client.post("/v1/customers")
        assert (r1.status_code, r2.status_code) == (200, 200)
        assert r3.status_code == 429
        assert r3.headers.get("retry-after") == "30"
        # Unlimited class: hammer freely.
        for _ in range(10):
            assert (await client.get("/v1/crystals")).status_code == 200


async def test_limits_are_keyed_per_bearer_token():
    app = _tiny_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        a = {"Authorization": "Bearer key-a"}
        b = {"Authorization": "Bearer key-b"}
        for _ in range(2):
            assert (await client.post("/v1/customers", headers=a)).status_code == 200
        assert (await client.post("/v1/customers", headers=a)).status_code == 429
        # A different customer's key is not affected by A's exhaustion.
        assert (await client.post("/v1/customers", headers=b)).status_code == 200


# --- B4: cross-tenant isolation at the HTTP surface ---------------------------

async def _two_customers(store):
    other = await store.create_customer(
        provider="anthropic", model_id="claude-sonnet-4-5-20250929",
        api_key_ref="test-ref-b4",
    )
    return other


async def _mk_crystal(store, customer_id, cid):
    await store.upsert_crystal(Crystal(
        id=cid, customer_id=customer_id, summary_vector=[0.1],
        crystal_type="customer:legacy", summary_text="tenant data",
    ))


async def test_cross_tenant_crystal_access_denied(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    """Customer A must not read customer B's crystal via the SDK surface:
    the list excludes it and the direct GET is a 404 (not 403 — existence
    is not disclosed either)."""
    # Import the shared app builder. Two forms because sys.path differs by
    # platform/runner: in-container runs have the repo root on the path
    # (tests.*), while Windows pytest puts the tests dir itself on the
    # path (no __init__.py → bare module name).
    try:
        from tests.test_endpoint_smoke import _build_app
    except ModuleNotFoundError:
        from test_endpoint_smoke import _build_app

    other = await _two_customers(store)
    await _mk_crystal(store, customer.id, "mine-b4")
    await _mk_crystal(store, other.id, "theirs-b4")

    app = _build_app(
        store, semantic_encoder_stub, vector_store, fact_vector_store,
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        headers = {"Authorization": f"Bearer {customer.api_key}"}

        listing = await client.get("/v1/crystals", headers=headers)
        assert listing.status_code == 200
        ids = {c["id"] for c in listing.json().get("crystals", [])}
        assert "mine-b4" in ids
        assert "theirs-b4" not in ids  # B's crystal invisible to A

        direct = await client.get("/v1/crystals/theirs-b4", headers=headers)
        assert direct.status_code == 404  # not found, not forbidden
