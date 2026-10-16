"""Lockdown PR-7 pins (security hygiene, 2026-10-13).

  Q55   the docs and the schema are off when hosted
  B4-6  one scope, `memory`; unknown scopes refused; the card states the grant
  B4-7  OAuth secrets hashed at rest; the raw value never in `data`;
        the migration helper converts plaintext rows and is idempotent
  B4-9  client_name bounded, client table capped, the auth routes rated
  B5-5  anonymous buckets keyed by the proxy-appended client address;
        buckets evicted; /health shows how the limiter sees the caller
  B5-6  every identity-minting and money-spending route has a class
  B6-2  stdlib and uvicorn log records are redacted at creation
  B6-7  an unhandled exception and a router 404 are the envelope
  B6-9  HSTS on the console origin; B6-12 CSP has no unsafe-inline script
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from mcp.server.auth.provider import RegistrationError
from mcp.shared.auth import OAuthClientInformationFull
from sqlalchemy import select
from starlette.requests import Request

import crystal_cache
from crystal_cache import app as app_mod
from crystal_cache.control import oauth_provider as prov_mod
from crystal_cache.control.oauth_provider import VALID_SCOPES, CrystalOAuthProvider
from crystal_cache.infrastructure import metadata_store_oauth_ext as oauth_ext
from crystal_cache.infrastructure.credentials import hash_api_key
from crystal_cache.infrastructure.schema import OAuthRecordRow
from crystal_cache.ingress import rate_limit as rl

REPO = Path(crystal_cache.__file__).resolve().parents[3]


def _client_info(cid="cl_pr7", name="Claude"):
    return OAuthClientInformationFull(
        client_id=cid,
        redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
        client_name=name,
        token_endpoint_auth_method="none",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    )


def _req(path="/", headers=None, client=("10.0.0.7", 1234)) -> Request:
    hdrs = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "method": "GET", "path": path, "headers": hdrs,
                    "client": client, "query_string": b""})


# ---------------------------------------------------------------------------
# Q55
# ---------------------------------------------------------------------------

def test_docs_and_schema_are_off_when_hosted():
    assert app_mod.docs_urls(True) == {"docs_url": None, "redoc_url": None, "openapi_url": None}
    assert app_mod.docs_urls(False) == {
        "docs_url": "/docs", "redoc_url": "/redoc", "openapi_url": "/openapi.json",
    }
    # The real app reads the hosted flag (the tests run as development).
    from crystal_cache.config import settings

    assert app_mod.app.openapi_url == app_mod.docs_urls(settings.is_production)["openapi_url"]
    assert app_mod.app.openapi()["info"]["title"] == "Crystal Cache"  # generation still works


# ---------------------------------------------------------------------------
# B4-7: hashed at rest
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_oauth_secrets_are_hashed_at_rest_and_the_raw_value_is_not_in_data(store):
    provider = CrystalOAuthProvider(store)
    await provider.register_client(_client_info())
    client = await provider.get_client("cl_pr7")
    code = await provider.create_authorization_code(
        client_id="cl_pr7", operator_id="op_1",
        redirect_uri="https://claude.ai/api/mcp/auth_callback",
        code_challenge="chal", scopes=["memory"], redirect_uri_provided_explicitly=True,
    )
    assert code.startswith("ccc_")
    loaded = await provider.load_authorization_code(client, code)
    assert loaded is not None and loaded.code == code
    tokens = await provider.exchange_authorization_code(client, loaded)
    assert tokens.access_token.startswith("cco_") and tokens.refresh_token.startswith("ccr_")

    async with store.session() as s:
        rows = (await s.execute(select(OAuthRecordRow))).scalars().all()
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r.kind, []).append(r)
    assert set(by_kind) == {"client", "access", "refresh"}  # the code died on exchange
    for kind in ("access", "refresh"):
        (row,) = by_kind[kind]
        assert not row.key.startswith(("cco_", "ccr_", "ccc_")), row.key
        assert len(row.key) == 64
        assert "token" not in row.data and tokens.access_token not in row.data \
            and tokens.refresh_token not in row.data
    assert by_kind["access"][0].key == hash_api_key(tokens.access_token)
    assert by_kind["refresh"][0].key == hash_api_key(tokens.refresh_token)
    # The client row is public metadata and keeps its id.
    assert by_kind["client"][0].key == "cl_pr7"

    # Loading by the presented raw value still works, with the value put back.
    at = await provider.load_access_token(tokens.access_token)
    assert at is not None and at.token == tokens.access_token and at.subject == "op_1"
    rt = await provider.load_refresh_token(client, tokens.refresh_token)
    assert rt is not None and rt.token == tokens.refresh_token
    # Revoking by raw value finds the hashed row.
    await provider.revoke_token(at)
    assert await provider.load_access_token(tokens.access_token) is None
    assert oauth_ext.oauth_storage_key("client", "cl_x") == "cl_x"
    assert oauth_ext.oauth_storage_key("access", "cco_x") == hash_api_key("cco_x")


@pytest.mark.asyncio
async def test_migration_helper_rehashes_plaintext_rows_once(store):
    """The rows written before PR-7: raw key, secret inside data. The
    helper (what migration c9f1a3b5d7e9 runs) converts them so the
    presented token still resolves, and a second run changes nothing."""
    raw_access = "cco_" + "a" * 43
    raw_code = "ccc_" + "c" * 43
    async with store.session() as s:
        s.add(OAuthRecordRow(
            kind="access", key=raw_access, client_id="cl_old", subject="op_old",
            data='{"token": "%s", "client_id": "cl_old", "scopes": ["memory"], '
                 '"expires_at": 9999999999, "resource": null, "subject": "op_old"}' % raw_access,
        ))
        s.add(OAuthRecordRow(
            kind="code", key=raw_code, client_id="cl_old", subject="op_old",
            data='{"code": "%s", "scopes": [], "expires_at": 9999999999, "client_id": "cl_old", '
                 '"code_challenge": "x", "redirect_uri": "https://claude.ai/cb", '
                 '"redirect_uri_provided_explicitly": true, "resource": null, "subject": "op_old"}' % raw_code,
        ))
        s.add(OAuthRecordRow(kind="client", key="cl_old", client_id="cl_old",
                             data='{"client_id": "cl_old", "redirect_uris": ["https://claude.ai/cb"]}'))

    async with store._engine.begin() as conn:
        converted = await conn.run_sync(oauth_ext.rehash_plaintext_oauth_secrets)
    assert converted == 2
    async with store._engine.begin() as conn:
        assert await conn.run_sync(oauth_ext.rehash_plaintext_oauth_secrets) == 0

    async with store.session() as s:
        rows = {(r.kind, r.key): r for r in (await s.execute(select(OAuthRecordRow))).scalars().all()}
    assert ("access", hash_api_key(raw_access)) in rows
    assert ("code", hash_api_key(raw_code)) in rows
    assert ("client", "cl_old") in rows
    assert raw_access not in rows[("access", hash_api_key(raw_access))].data
    assert raw_code not in rows[("code", hash_api_key(raw_code))].data
    # The pre-PR-7 token still resolves through the provider.
    provider = CrystalOAuthProvider(store)
    at = await provider.load_access_token(raw_access)
    assert at is not None and at.subject == "op_old" and at.token == raw_access


# ---------------------------------------------------------------------------
# B4-6 / B4-9: scope, name, cap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_registration_bounds_the_name_and_the_table(store, monkeypatch):
    provider = CrystalOAuthProvider(store)
    with pytest.raises(RegistrationError) as exc:
        await provider.register_client(_client_info(name="n" * 65))
    assert exc.value.error == "invalid_client_metadata" and "64" in exc.value.error_description
    with pytest.raises(RegistrationError):
        await provider.register_client(_client_info(name="bad\nname"))
    await provider.register_client(_client_info(cid="cl_a", name="n" * 64))
    monkeypatch.setattr(prov_mod, "OAUTH_MAX_CLIENTS", 1)
    with pytest.raises(RegistrationError) as exc:
        await provider.register_client(_client_info(cid="cl_b"))
    assert "too many" in exc.value.error_description
    assert await store.oauth_count("client") == 1


def test_the_one_scope_is_wired_into_registration_and_the_card():
    assert VALID_SCOPES == ("memory",)
    src = (REPO / "memory/src/crystal_cache/app.py").read_text(encoding="utf-8")
    assert "valid_scopes=list(VALID_SCOPES)" in src and "default_scopes=list(VALID_SCOPES)" in src
    tsx = (REPO / "frontend/src/pages/OAuthConsent.tsx").read_text(encoding="utf-8")
    assert "Full access to this workspace's memory" in tsx


@pytest.mark.asyncio
async def test_approve_refuses_unknown_scopes_and_the_card_states_the_grant(store, monkeypatch):
    from fastapi import HTTPException

    from crystal_cache.config import Settings
    from crystal_cache.endpoints import me as me_mod
    from crystal_cache.endpoints import oauth as oauth_mod
    from crystal_cache.endpoints.oauth import ApproveRequest
    from crystal_cache.ingress import auth as auth_mod

    s = lambda: Settings(firebase_project_id="test-proj", oauth_enabled=True)  # noqa: E731
    for mod in (me_mod, auth_mod, oauth_mod):
        monkeypatch.setattr(mod, "get_settings", s)
    monkeypatch.setattr(auth_mod, "_verify_firebase_jwt", lambda tok, proj: {"sub": "uid_pr7", "email": "pr7@test.dev"})

    class _Req:
        def __init__(self, token):
            self.headers = {"authorization": f"Bearer {token}"}

    await me_mod.signup(_Req("eyJx.eyJy.sig"), store)  # the user's workspace
    provider = CrystalOAuthProvider(store)
    await provider.register_client(_client_info())
    card = await oauth_mod.oauth_client_info("cl_pr7", store)
    assert card["scopes"] == ["memory"] and "Full access" in card["grant"]

    body = dict(client_id="cl_pr7", redirect_uri="https://claude.ai/api/mcp/auth_callback",
                state="s", code_challenge="c", explicit_redirect=True)
    with pytest.raises(HTTPException) as exc:
        await oauth_mod.oauth_approve(ApproveRequest(scopes="memory admin:all", **body), _Req("eyJx.eyJy.sig"), store)
    assert exc.value.status_code == 400 and "admin:all" in exc.value.detail
    # Empty means the one scope there is; the minted code carries it.
    out = await oauth_mod.oauth_approve(ApproveRequest(scopes="", **body), _Req("eyJx.eyJy.sig"), store)
    code = out["redirect_to"].split("code=")[1].split("&")[0]
    loaded = await provider.load_authorization_code(await provider.get_client("cl_pr7"), code)
    assert loaded is not None and loaded.scopes == ["memory"]


# ---------------------------------------------------------------------------
# B5-5 / B5-6: the limiter
# ---------------------------------------------------------------------------

def test_client_address_takes_the_proxy_appended_hop_only_when_told_to():
    h = {"X-Forwarded-For": "6.6.6.6, 203.0.113.9, 10.1.1.1"}
    assert rl.client_address(_req(headers=h), hops=0) == "10.0.0.7"      # no proxy: header ignored
    assert rl.client_address(_req(headers=h), hops=1) == "10.1.1.1"      # the last appended entry
    assert rl.client_address(_req(headers=h), hops=2) == "203.0.113.9"   # behind one more proxy
    assert rl.client_address(_req(headers=h), hops=5) == "10.0.0.7"      # fewer entries than hops
    assert rl.client_address(_req(headers={"X-Forwarded-For": "  "}), hops=1) == "10.0.0.7"
    assert rl.client_address(_req(), hops=1) == "10.0.0.7"
    # The default comes from settings (0 unless the deployment says).
    from crystal_cache.config import Settings

    assert Settings().rate_limit_trusted_proxy_hops == 0
    # uvicorn's own forwarded-header middleware with "*" takes the LEFTMOST
    # entry (the one a client can forge); that is why it is not used.
    from uvicorn.middleware.proxy_headers import _TrustedHosts

    assert _TrustedHosts("*").get_trusted_client_address("6.6.6.6, 10.1.1.1")[0] == "6.6.6.6"


def test_limiter_evicts_stale_buckets_and_keeps_counting():
    lim = rl.SlidingWindowLimiter(2, window_seconds=10.0)
    assert lim.allow("a", now=0.0) and lim.allow("a", now=1.0) and not lim.allow("a", now=2.0)
    assert lim.allow("a", now=11.0)  # the window moved on
    for i in range(50):
        lim.allow(f"random-{i}", now=20.0)
    assert lim.bucket_count() == 51
    lim._sweep(cutoff=100.0)
    assert lim.bucket_count() == 0
    # The sweep runs by itself past the threshold.
    import crystal_cache.ingress.rate_limit as mod

    old = mod.SWEEP_THRESHOLD
    mod.SWEEP_THRESHOLD = 10
    try:
        for i in range(12):
            lim.allow(f"k{i}", now=200.0)
        assert lim.bucket_count() <= 13
        assert lim.allow("fresh", now=300.0)  # everything from t=200 is stale by 300
        assert lim.bucket_count() == 1
    finally:
        mod.SWEEP_THRESHOLD = old


def _tiny_app(**limits) -> FastAPI:
    app = FastAPI()
    app.middleware("http")(rl.build_rate_limit_middleware(**limits))
    paths = [
        "/register", "/token", "/authorize", "/v1/auth/signup", "/v1/oauth/approve",
        "/v1/billing/checkout", "/admin/api/customers/c1/chat", "/admin/api/customers/c1/agent",
        "/admin/api/customers/c1/learn", "/admin/api/assumptions/a1/respond",
        "/admin/api/assumptions/a1/verify", "/admin/api/conflicts/scan",
        "/admin/api/cognition/tasks/t1/requeue", "/v1/gaps/g1/research",
        "/unclassified", "/admin/api/customers/c1/spend",
    ]

    async def _ok():
        return {"ok": True}

    for p in paths:
        app.add_api_route(p, _ok, methods=["GET"])
    return app


@pytest.mark.asyncio
async def test_every_identity_and_spend_route_has_a_class():
    app = _tiny_app(auth_per_minute=1, expensive_per_minute=1, mcp_per_minute=100)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        for path in ("/register", "/token", "/authorize", "/v1/auth/signup",
                     "/v1/oauth/approve", "/v1/billing/checkout"):
            h = {"Authorization": f"Bearer tok_{path}"}
            assert (await c.get(path, headers=h)).status_code == 200, path
            assert (await c.get(path, headers=h)).status_code == 429, path  # auth class
        for path in ("/admin/api/customers/c1/chat", "/admin/api/customers/c1/agent",
                     "/admin/api/customers/c1/learn", "/admin/api/assumptions/a1/respond",
                     "/admin/api/assumptions/a1/verify", "/admin/api/conflicts/scan",
                     "/admin/api/cognition/tasks/t1/requeue", "/v1/gaps/g1/research"):
            h = {"Authorization": f"Bearer tok_{path}"}
            assert (await c.get(path, headers=h)).status_code == 200, path
            assert (await c.get(path, headers=h)).status_code == 429, path  # expensive class
        # Reads stay unlimited, including an admin read that is not in the regex set.
        for path in ("/unclassified", "/admin/api/customers/c1/spend"):
            for _ in range(5):
                assert (await c.get(path, headers={"Authorization": "Bearer tok_r"})).status_code == 200


@pytest.mark.asyncio
async def test_anonymous_buckets_follow_the_client_address(monkeypatch):
    monkeypatch.setattr(rl, "_trusted_proxy_hops", lambda: 1)
    app = _tiny_app(auth_per_minute=1, expensive_per_minute=100, mcp_per_minute=100)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # Two callers behind the same peer, told apart by the appended hop.
        assert (await c.get("/register", headers={"X-Forwarded-For": "1.1.1.1"})).status_code == 200
        assert (await c.get("/register", headers={"X-Forwarded-For": "1.1.1.1"})).status_code == 429
        assert (await c.get("/register", headers={"X-Forwarded-For": "2.2.2.2"})).status_code == 200
        # A forged leading entry does not buy a fresh bucket: the appended hop is what counts.
        assert (await c.get("/register", headers={"X-Forwarded-For": "9.9.9.9, 2.2.2.2"})).status_code == 429


@pytest.mark.asyncio
async def test_health_shows_how_the_limiter_sees_the_caller(monkeypatch):
    from crystal_cache.endpoints import health as health_mod

    class _Stub:
        def is_ready(self):
            return False

    monkeypatch.setattr(health_mod, "get_llm_client", lambda: _Stub())
    monkeypatch.setattr(rl, "_trusted_proxy_hops", lambda: 1)
    body = await health_mod.health(_req(headers={"X-Forwarded-For": "1.1.1.1, 2.2.2.2"}))
    assert body["client"]["address"] == "2.2.2.2"
    assert body["client"]["forwarded_entries"] == 2
    assert "trusted_proxy_hops" in body["client"]


# ---------------------------------------------------------------------------
# B6-2: stdlib and uvicorn records are redacted at creation
# ---------------------------------------------------------------------------

def test_stdlib_and_uvicorn_log_records_are_redacted(caplog):
    assert getattr(logging, "_crystal_redacting_factory", False) is True
    key = "sk_live_LEAK1234567890"
    with caplog.at_level(logging.INFO):
        logging.getLogger("crystal_cache.maintenance.consolidation_service").error(
            "upstream rejected header %s", f"Bearer {key}",
        )
        try:
            raise RuntimeError(f"InvalidHeader: {key} and cco_{'x' * 40}")
        except RuntimeError:
            logging.getLogger("uvicorn.error").exception("Exception in ASGI application")
    text = caplog.text
    assert key not in text and "cco_" + "x" * 40 not in text
    assert "[redacted]" in text
    for rec in caplog.records:
        assert key not in rec.getMessage()
        if rec.exc_text:
            assert key not in rec.exc_text and "RuntimeError" in rec.exc_text
    # Formatting through a plain Formatter (uvicorn's path) stays redacted.
    out = logging.Formatter("%(message)s").format(caplog.records[-1])
    assert key not in out and "Traceback" in out


def test_uvicorn_access_records_keep_their_five_args_and_still_redact(caplog):
    """2026-10-16 (Cloud Monitoring Traceback alert, ~1 per request since
    PR-7): uvicorn's AccessFormatter unpacks record.args into five
    fields; the factory folded them into msg, so every access line
    raised inside the handler. The access record keeps its tuple (with
    its strings redacted) and formats cleanly through uvicorn's own
    formatter; the folding path is unchanged for everyone else."""
    from uvicorn.logging import AccessFormatter

    key = "sk_live_LEAK1234567890"
    with caplog.at_level(logging.INFO):
        logging.getLogger("uvicorn.access").info(
            '%s - "%s %s HTTP/%s" %d', "10.0.0.1:1234", "GET", f"/v1/recall?key={key}", "1.1", 200,
        )
    rec = caplog.records[-1]
    assert isinstance(rec.args, tuple) and len(rec.args) == 5
    out = AccessFormatter('%(client_addr)s - "%(request_line)s" %(status_code)s', use_colors=False).format(rec)
    assert out.startswith('10.0.0.1:1234 - "GET /v1/recall?key=') and "200 OK" in out
    assert key not in out and "[redacted]" in out


# ---------------------------------------------------------------------------
# B6-7: the envelope on every error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unhandled_exception_and_router_404_are_the_envelope():
    from starlette.exceptions import HTTPException as StarletteHTTPException

    assert Exception in app_mod.app.exception_handlers
    assert StarletteHTTPException in app_mod.app.exception_handlers

    tiny = FastAPI()
    tiny.add_exception_handler(Exception, app_mod.unhandled_exception_handler)
    tiny.add_exception_handler(StarletteHTTPException, app_mod.http_exception_handler)

    @tiny.get("/boom")
    async def _boom():
        raise RuntimeError("header 'Bearer sk_live_LEAK' rejected")

    async with AsyncClient(transport=ASGITransport(app=tiny, raise_app_exceptions=False),
                           base_url="http://t") as c:
        r = await c.get("/boom")
        assert r.status_code == 500
        err = r.json()["error"]
        assert err["type"] == "api_error" and err["code"] == "internal_error"
        assert "sk_live_LEAK" not in r.text and "RuntimeError" not in r.text
        assert "reference" in err["message"]
        r = await c.get("/no-such-route")
        assert r.status_code == 404
        assert r.json()["error"]["type"] == "not_found_error"
        r = await c.post("/boom")
        assert r.status_code == 405 and r.json()["error"]["type"] == "api_error"


# ---------------------------------------------------------------------------
# B6-9 / B6-12: the console origin
# ---------------------------------------------------------------------------

def test_console_nginx_has_hsts_and_no_inline_scripts():
    conf = (REPO / "deploy/inspector/nginx.conf.template").read_text(encoding="utf-8")
    assert 'Strict-Transport-Security "max-age=31536000' in conf
    csp = [line for line in conf.splitlines() if "Content-Security-Policy" in line][0]
    script_src = csp.split("script-src")[1].split(";")[0]
    assert "'unsafe-inline'" not in script_src
    assert "frame-ancestors 'none'" in csp
