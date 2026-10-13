"""L6 pin (2026-09-05): /health carries the self-curation readiness signal.

A keyless deployment must SAY it isn't curating — the audit's
highest-value line. Both branches pinned so the field can neither
disappear nor invent a third state.
"""
import pytest

from crystal_cache.endpoints import health as health_mod


class _Stub:
    def __init__(self, ready: bool):
        self._ready = ready

    def is_ready(self) -> bool:
        return self._ready


def _req():
    """A bare request for the liveness probe (B5-5: /health reads the
    caller's address and X-Forwarded-For)."""
    from starlette.requests import Request

    return Request({"type": "http", "method": "GET", "path": "/health", "headers": [],
                    "client": ("127.0.0.1", 1234), "query_string": b""})


@pytest.mark.asyncio
async def test_health_reports_active_when_provider_ready(monkeypatch):
    monkeypatch.setattr(health_mod, "get_llm_client", lambda: _Stub(True))
    body = await health_mod.health(_req())
    assert body["self_curation"] == "active"


@pytest.mark.asyncio
async def test_health_reports_idle_when_keyless(monkeypatch):
    monkeypatch.setattr(health_mod, "get_llm_client", lambda: _Stub(False))
    body = await health_mod.health(_req())
    assert body["self_curation"] == "idle_no_model_key"
    # The rest of the liveness payload is unchanged.
    assert body["status"] == "ok"
