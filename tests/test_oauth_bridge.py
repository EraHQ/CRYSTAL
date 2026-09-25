"""L2-S5b pins (2026-09-24): the consent bridge.

The full loop is what matters: a registered client, a signed-in user,
an approval, and a code the PROVIDER then redeems into tokens whose
subject is the user's own seat. Plus the security invariant that makes
the consent page unphishable: unregistered redirect targets are
refused, whatever the query string claims.
"""
from typing import Optional

import pytest
from fastapi import HTTPException

from mcp.shared.auth import OAuthClientInformationFull

from crystal_cache.config import Settings
from crystal_cache.control.oauth_provider import CrystalOAuthProvider
from crystal_cache.endpoints import me as me_mod
from crystal_cache.endpoints import oauth as oauth_mod
from crystal_cache.endpoints.oauth import ApproveRequest
from crystal_cache.ingress import auth as auth_mod


class _StubRequest:
    def __init__(self, token, body: Optional[dict] = None):
        self.headers = {"authorization": f"Bearer {token}"}
        self._body = body

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


def _wire(monkeypatch, uid, email):
    s = lambda: Settings(firebase_project_id="test-proj", oauth_enabled=True)
    monkeypatch.setattr(me_mod, "get_settings", s)
    monkeypatch.setattr(auth_mod, "get_settings", s)
    monkeypatch.setattr(oauth_mod, "get_settings", s)
    monkeypatch.setattr(
        auth_mod, "_verify_firebase_jwt",
        lambda tok, proj: {"sub": uid, "email": email},
    )


async def _register(store, cid="cl_claude"):
    provider = CrystalOAuthProvider(store)
    await provider.register_client(OAuthClientInformationFull(
        client_id=cid,
        redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
        client_name="Claude",
        token_endpoint_auth_method="none",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    ))
    return provider


def _approve_body(**over):
    d = dict(
        client_id="cl_claude",
        redirect_uri="https://claude.ai/api/mcp/auth_callback",
        state="xyz",
        code_challenge="chal",
        scopes="memory",
        explicit_redirect=True,
    )
    d.update(over)
    return ApproveRequest(**d)


@pytest.mark.asyncio
async def test_approve_mints_a_code_the_provider_redeems(store, monkeypatch):
    _wire(monkeypatch, "uid_ob_x", "obx@test.dev")
    r = await me_mod.signup(_StubRequest("eyJx.eyJy.sig", {}), store)
    provider = await _register(store)

    out = await oauth_mod.oauth_approve(
        _approve_body(), _StubRequest("eyJx.eyJy.sig"), store
    )
    assert out["redirect_to"].startswith(
        "https://claude.ai/api/mcp/auth_callback?code=ccc_"
    )
    assert "state=xyz" in out["redirect_to"]

    code = out["redirect_to"].split("code=")[1].split("&")[0]
    client = await provider.get_client("cl_claude")
    loaded = await provider.load_authorization_code(client, code)
    assert loaded is not None
    tokens = await provider.exchange_authorization_code(client, loaded)
    at = await provider.load_access_token(tokens.access_token)
    # The subject is the user's own seat.
    assert at is not None and at.subject == r["operator_id"]


@pytest.mark.asyncio
async def test_unregistered_redirect_uri_is_refused(store, monkeypatch):
    _wire(monkeypatch, "uid_ob_y", "oby@test.dev")
    await me_mod.signup(_StubRequest("eyJx.eyJy.sig", {}), store)
    await _register(store)
    with pytest.raises(HTTPException) as e:
        await oauth_mod.oauth_approve(
            _approve_body(redirect_uri="https://evil.example/cb"),
            _StubRequest("eyJx.eyJy.sig"), store,
        )
    assert e.value.status_code == 400


@pytest.mark.asyncio
async def test_unknown_client_and_missing_session(store, monkeypatch):
    _wire(monkeypatch, "uid_ob_z", "obz@test.dev")
    await me_mod.signup(_StubRequest("eyJx.eyJy.sig", {}), store)
    with pytest.raises(HTTPException) as e:
        await oauth_mod.oauth_approve(
            _approve_body(client_id="cl_nope"),
            _StubRequest("eyJx.eyJy.sig"), store,
        )
    assert e.value.status_code == 404

    await _register(store)

    class _NoAuth:
        headers: dict = {}

    with pytest.raises(HTTPException) as e:
        await oauth_mod.oauth_approve(_approve_body(), _NoAuth(), store)
    assert e.value.status_code == 401


@pytest.mark.asyncio
async def test_client_info_card_payload(store, monkeypatch):
    _wire(monkeypatch, "uid_ob_w", "obw@test.dev")
    await _register(store)
    out = await oauth_mod.oauth_client_info("cl_claude", store)
    assert out["client_name"] == "Claude"
    assert out["redirect_hosts"] == ["claude.ai"]
    with pytest.raises(HTTPException):
        await oauth_mod.oauth_client_info("cl_nope", store)
