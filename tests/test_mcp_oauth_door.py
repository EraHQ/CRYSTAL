"""L2-S5c pins (2026-09-24): the MCP door's dual auth.

The whole point of the arc, pinned: an OAuth access token minted by our
own authorization server opens the same door static keys always have,
acting as the operator it was issued to, while static cc_ keys keep
working untouched. Plus the discovery breadcrumb: when the AS is
armed, denied requests advertise the RFC 9728 resource-metadata URL
that Claude's connector dialog follows to find sign-in.
"""
from typing import Optional

import pytest

from mcp.shared.auth import OAuthClientInformationFull

from crystal_cache.agent import mcp_server as door
from crystal_cache.agent.mcp_server import _CustomerKeyAuthMiddleware
from crystal_cache.config import Settings
from crystal_cache.control.oauth_provider import CrystalOAuthProvider
from crystal_cache.endpoints import me as me_mod
from crystal_cache.ingress import auth as auth_mod


class _Sent:
    def __init__(self):
        self.status: Optional[int] = None
        self.headers: dict[bytes, bytes] = {}

    async def __call__(self, message):
        if message["type"] == "http.response.start":
            self.status = message["status"]
            self.headers = dict(message.get("headers", []))


class _InnerApp:
    def __init__(self):
        self.called_with: Optional[str] = None

    async def __call__(self, scope, receive, send):
        self.called_with = door._current_customer_id.get(None)


def _scope(token: Optional[str]):
    headers = []
    if token:
        headers.append((b"authorization", f"Bearer {token}".encode()))
    return {"type": "http", "headers": headers}


class _StubRequest:
    def __init__(self, token, body=None):
        self.headers = {"authorization": f"Bearer {token}"}
        self._body = body

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


async def _mint_access_token(store, uid, email):
    """Signup -> registered client -> code for that seat -> token pair."""
    r = await me_mod.signup(_StubRequest("eyJx.eyJy.sig", {}), store)
    provider = CrystalOAuthProvider(store)
    await provider.register_client(OAuthClientInformationFull(
        client_id="cl_door",
        redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
        client_name="Claude",
        token_endpoint_auth_method="none",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    ))
    code = await provider.create_authorization_code(
        client_id="cl_door",
        operator_id=r["operator_id"],
        redirect_uri="https://claude.ai/api/mcp/auth_callback",
        code_challenge="chal",
        scopes=["memory"],
        redirect_uri_provided_explicitly=True,
    )
    client = await provider.get_client("cl_door")
    loaded = await provider.load_authorization_code(client, code)
    tokens = await provider.exchange_authorization_code(client, loaded)
    return r, tokens


def _wire(monkeypatch, store, uid, email, oauth_enabled=True):
    s = lambda: Settings(
        firebase_project_id="test-proj", oauth_enabled=oauth_enabled
    )
    monkeypatch.setattr(me_mod, "get_settings", s)
    monkeypatch.setattr(auth_mod, "get_settings", s)
    monkeypatch.setattr(
        auth_mod, "_verify_firebase_jwt",
        lambda tok, proj: {"sub": uid, "email": email},
    )
    monkeypatch.setattr(door, "get_metadata_store", lambda: store)
    import crystal_cache.config as config_mod
    monkeypatch.setattr(config_mod, "get_settings", s)


@pytest.mark.asyncio
async def test_oauth_bearer_opens_the_door_as_the_right_tenant(
    store, monkeypatch
):
    _wire(monkeypatch, store, "uid_door_a", "doora@test.dev")
    r, tokens = await _mint_access_token(store, "uid_door_a", "doora@test.dev")

    inner = _InnerApp()
    sent = _Sent()
    await _CustomerKeyAuthMiddleware(inner)(
        _scope(tokens.access_token), None, sent
    )
    # The inner app ran, inside the token subject's own tenant.
    assert inner.called_with == r["customer_id"]
    assert sent.status is None  # no auth error was emitted


@pytest.mark.asyncio
async def test_bad_or_revoked_cco_token_401s_and_never_falls_through(
    store, monkeypatch
):
    _wire(monkeypatch, store, "uid_door_b", "doorb@test.dev")
    _, tokens = await _mint_access_token(store, "uid_door_b", "doorb@test.dev")

    provider = CrystalOAuthProvider(store)
    at = await provider.load_access_token(tokens.access_token)
    await provider.revoke_token(at)

    inner = _InnerApp()
    sent = _Sent()
    await _CustomerKeyAuthMiddleware(inner)(
        _scope(tokens.access_token), None, sent
    )
    assert inner.called_with is None
    assert sent.status == 401
    # The armed-AS challenge carries the RFC 9728 breadcrumb.
    www = sent.headers.get(b"www-authenticate", b"").decode()
    assert "resource_metadata=" in www
    assert "/.well-known/oauth-protected-resource/mcp" in www


@pytest.mark.asyncio
async def test_static_keys_still_work_with_oauth_armed(store, monkeypatch):
    _wire(monkeypatch, store, "uid_door_c", "doorc@test.dev")
    r = await me_mod.signup(_StubRequest("eyJx.eyJy.sig", {}), store)

    inner = _InnerApp()
    sent = _Sent()
    await _CustomerKeyAuthMiddleware(inner)(
        _scope(r["api_key"]), None, sent
    )
    assert inner.called_with == r["customer_id"]
    assert sent.status is None


@pytest.mark.asyncio
async def test_unarmed_deployments_challenge_plain_bearer(store, monkeypatch):
    _wire(monkeypatch, store, "uid_door_d", "doord@test.dev",
          oauth_enabled=False)
    inner = _InnerApp()
    sent = _Sent()
    await _CustomerKeyAuthMiddleware(inner)(_scope("cc_nonsense"), None, sent)
    assert sent.status == 401
    assert sent.headers.get(b"www-authenticate") == b"Bearer"


def test_cors_posture_for_browser_mcp_clients():
    """LIVE-FOUND 2026-09-25: with no CORS, Claude's browser-side
    connector checks see only opaque responses and die on the first
    probe — the whole OAuth surface is invisible however correct it is.
    Wildcard origins without credentials is the remote-MCP posture
    (bearer auth, never cookies); WWW-Authenticate must be exposed or
    the RFC 9728 breadcrumb is unreadable."""
    from starlette.middleware.cors import CORSMiddleware

    from crystal_cache.app import app

    m = [mw for mw in app.user_middleware if mw.cls is CORSMiddleware]
    assert m, "CORS middleware missing from the app"
    kw = m[0].kwargs
    assert kw["allow_origins"] == ["*"]
    assert kw["allow_credentials"] is False
    assert "WWW-Authenticate" in kw["expose_headers"]


@pytest.mark.asyncio
async def test_exact_mcp_path_never_redirects():
    """LIVE-FOUND 2026-09-25: POST /mcp answered 307 -> http://.../mcp/
    (Starlette trailing-slash redirect from the mount, scheme downgraded
    behind the proxy). Anthropic's prober refuses downgrade redirects,
    so the connector check died in one round trip. The shim rewrites the
    stripped exact path to "/" so no redirect can exist."""
    from crystal_cache.app import _ExactMountPathShim

    seen = {}

    class _Inner:
        async def __call__(self, scope, receive, send):
            seen["path"] = scope["path"]

    shim = _ExactMountPathShim(_Inner())
    await shim({"type": "http", "path": ""}, None, None)
    assert seen["path"] == "/"
    await shim({"type": "http", "path": "/"}, None, None)
    assert seen["path"] == "/"
