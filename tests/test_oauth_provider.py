"""L2-S5a pins (2026-09-24): the OAuth provider against real storage.

What must not regress: dynamic registration roundtrips (Q1=A), the
consent redirect carries every parameter the S5b bridge needs (Q4=A),
codes are single-use and client-bound, token pairs mint with the
ratified lifetimes (Q3=A), access tokens die at expiry and on
revocation, and refresh rotation kills the old token on first use.
Static cc_ keys appear nowhere here — the doors stay separate.
"""
import time

import pytest

from mcp.server.auth.provider import AuthorizationParams
from mcp.shared.auth import OAuthClientInformationFull

from crystal_cache.control.oauth_provider import (
    ACCESS_TTL_SECONDS,
    CrystalOAuthProvider,
)


def _client(cid="cl_test_1"):
    return OAuthClientInformationFull(
        client_id=cid,
        redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
        client_name="Claude",
        token_endpoint_auth_method="none",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    )


@pytest.fixture
def provider(store):
    return CrystalOAuthProvider(store)


@pytest.mark.asyncio
async def test_dynamic_registration_roundtrip(provider):
    info = _client()
    await provider.register_client(info)
    back = await provider.get_client("cl_test_1")
    assert back is not None
    assert back.client_id == "cl_test_1"
    assert str(back.redirect_uris[0]) == "https://claude.ai/api/mcp/auth_callback"
    assert await provider.get_client("cl_nope") is None


@pytest.mark.asyncio
async def test_authorize_redirects_to_consent_with_full_context(provider):
    info = _client()
    await provider.register_client(info)
    url = await provider.authorize(info, AuthorizationParams(
        state="st4te",
        scopes=["memory"],
        code_challenge="chal123",
        redirect_uri="https://claude.ai/api/mcp/auth_callback",
        redirect_uri_provided_explicitly=True,
        resource=None,
    ))
    assert url.startswith("https://inspector.erahq.ai/admin/oauth/consent?")
    for frag in ("client_id=cl_test_1", "state=st4te", "code_challenge=chal123",
                 "scopes=memory", "explicit_redirect=1"):
        assert frag in url


async def _mint_code(provider, operator_id="op_1"):
    return await provider.create_authorization_code(
        client_id="cl_test_1",
        operator_id=operator_id,
        redirect_uri="https://claude.ai/api/mcp/auth_callback",
        code_challenge="chal123",
        scopes=["memory"],
        redirect_uri_provided_explicitly=True,
    )


@pytest.mark.asyncio
async def test_code_exchange_mints_pair_and_is_single_use(provider):
    info = _client()
    await provider.register_client(info)
    code = await _mint_code(provider)
    loaded = await provider.load_authorization_code(info, code)
    assert loaded is not None and loaded.subject == "op_1"

    tokens = await provider.exchange_authorization_code(info, loaded)
    assert tokens.access_token.startswith("cco_")
    assert tokens.refresh_token.startswith("ccr_")
    assert tokens.expires_in == ACCESS_TTL_SECONDS
    # Single use: the code is gone.
    assert await provider.load_authorization_code(info, code) is None

    at = await provider.load_access_token(tokens.access_token)
    assert at is not None and at.subject == "op_1"


@pytest.mark.asyncio
async def test_code_is_client_bound(provider):
    info = _client()
    other = _client("cl_other")
    await provider.register_client(info)
    await provider.register_client(other)
    code = await _mint_code(provider)
    assert await provider.load_authorization_code(other, code) is None


@pytest.mark.asyncio
async def test_access_token_dies_at_expiry_and_on_revocation(provider, monkeypatch):
    info = _client()
    await provider.register_client(info)
    code = await _mint_code(provider)
    loaded = await provider.load_authorization_code(info, code)
    tokens = await provider.exchange_authorization_code(info, loaded)

    # Expiry: jump the clock past the TTL.
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + ACCESS_TTL_SECONDS + 5)
    assert await provider.load_access_token(tokens.access_token) is None
    monkeypatch.setattr(time, "time", real)

    # Revocation.
    at = await provider.load_access_token(tokens.access_token)
    assert at is not None
    await provider.revoke_token(at)
    assert await provider.load_access_token(tokens.access_token) is None


@pytest.mark.asyncio
async def test_refresh_rotation_kills_the_old_token(provider):
    info = _client()
    await provider.register_client(info)
    code = await _mint_code(provider)
    loaded = await provider.load_authorization_code(info, code)
    first = await provider.exchange_authorization_code(info, loaded)

    rt = await provider.load_refresh_token(info, first.refresh_token)
    assert rt is not None
    second = await provider.exchange_refresh_token(info, rt, ["memory"])
    assert second.access_token != first.access_token
    assert second.refresh_token != first.refresh_token
    # The old refresh token is dead the moment it was used.
    assert await provider.load_refresh_token(info, first.refresh_token) is None
    # The new pair works and carries the same subject.
    at = await provider.load_access_token(second.access_token)
    assert at is not None and at.subject == "op_1"
