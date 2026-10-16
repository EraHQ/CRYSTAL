"""PR 12b (2026-10-16): OAuth discovery follows the host the client used.

When a product domain is mapped, the issuer moves there while the
connector directory and the MCP registry still carry the raw Cloud Run
URL (no domain is mapped as of 2026-10-16; the example host stands in). For
an allow-listed host the discovery documents and the 401 breadcrumb
name THAT host; any other Host header gets the configured issuer.
"""
from __future__ import annotations

import pytest

from crystal_cache.config import Settings
from crystal_cache.ingress.oauth_hosts import (
    allowed_issuer_hosts, is_discovery_path, issuer_for, rewrite_discovery,
)

ALT = "crystal-api-118881845105.us-east5.run.app"


def _settings(**over):
    kwargs = dict(firebase_project_id="test-proj", oauth_enabled=True,
                  oauth_issuer_url="https://api.example.test", oauth_alt_issuer_hosts=ALT)
    kwargs.update(over)
    return Settings(**kwargs)


def test_issuer_follows_allow_listed_hosts_only():
    s = _settings()
    assert allowed_issuer_hosts(s) == {"api.example.test", ALT}
    assert issuer_for("api.example.test", s) == "https://api.example.test"
    assert issuer_for(ALT, s) == f"https://{ALT}"
    assert issuer_for("API.EXAMPLE.TEST:443", s) == "https://api.example.test"
    assert issuer_for("evil.example", s) == "https://api.example.test"
    assert issuer_for(None, s) == "https://api.example.test"
    assert issuer_for("localhost:8000", _settings(oauth_alt_issuer_hosts=f"{ALT}, localhost")) == "https://localhost:8000"
    assert is_discovery_path("/.well-known/oauth-protected-resource/mcp")
    assert is_discovery_path("/.well-known/oauth-authorization-server")
    assert not is_discovery_path("/.well-known/other")


def test_discovery_documents_are_rewritten_for_the_alt_host():
    s = _settings()
    doc = '{"resource": "https://api.example.test/mcp", "authorization_servers": ["https://api.example.test"]}'
    assert rewrite_discovery(doc, ALT, s) == f'{{"resource": "https://{ALT}/mcp", "authorization_servers": ["https://{ALT}"]}}'
    assert rewrite_discovery(doc, "api.example.test", s) == doc
    assert rewrite_discovery(doc, "evil.example", s) == doc


@pytest.mark.asyncio
async def test_the_401_breadcrumb_names_the_host_the_client_used(store, monkeypatch):
    import crystal_cache.config as config_mod
    from crystal_cache.agent import mcp_server as door
    from crystal_cache.agent.mcp_server import _CustomerKeyAuthMiddleware

    s = _settings()
    monkeypatch.setattr(config_mod, "get_settings", lambda: s)
    monkeypatch.setattr(door, "get_metadata_store", lambda: store)

    class _Sent:
        status = None
        headers: dict = {}

        async def __call__(self, message):
            if message["type"] == "http.response.start":
                self.status = message["status"]
                self.headers = dict(message.get("headers", []))

    async def inner(scope, receive, send):  # never reached without a token
        raise AssertionError("inner app ran")

    for host, expected in ((ALT, f"https://{ALT}"), ("api.example.test", "https://api.example.test"),
                           ("evil.example", "https://api.example.test")):
        sent = _Sent()
        await _CustomerKeyAuthMiddleware(inner)({"type": "http", "headers": [(b"host", host.encode())]}, None, sent)
        assert sent.status == 401
        www = sent.headers.get(b"www-authenticate", b"").decode()
        assert www == f'Bearer resource_metadata="{expected}/.well-known/oauth-protected-resource/mcp"', (host, www)
