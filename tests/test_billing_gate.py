"""Launch gate pin (2026-09-27): billing_live on /v1/me.

Free-first launch: the console must never offer sandbox checkout to
real users. billing_live is derived from the configured Stripe key's
prefix, so the gate flips itself the day live keys land and reads
false on self-host (no key at all).
"""
from typing import Optional

import pytest

from crystal_cache.config import Settings
from crystal_cache.endpoints import me as me_mod
from crystal_cache.ingress import auth as auth_mod


class _StubRequest:
    def __init__(self, token, body: Optional[dict] = None):
        self.headers = {"authorization": f"Bearer {token}"}
        self._body = body

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


def _wire(monkeypatch, uid, email, stripe_key):
    s = lambda: Settings(
        firebase_project_id="test-proj", stripe_secret_key=stripe_key
    )
    monkeypatch.setattr(me_mod, "get_settings", s)
    monkeypatch.setattr(auth_mod, "get_settings", s)
    monkeypatch.setattr(
        auth_mod, "_verify_firebase_jwt",
        lambda tok, proj: {"sub": uid, "email": email},
    )


@pytest.mark.asyncio
async def test_sandbox_and_absent_keys_read_billing_live_false(
    store, monkeypatch
):
    _wire(monkeypatch, "uid_bg_a", "bga@test.dev", "sk_test_abc123")
    await me_mod.signup(_StubRequest("eyJx.eyJy.sig", {}), store)
    out = await me_mod.get_me(_StubRequest("eyJx.eyJy.sig"), store)
    assert out["billing_live"] is False

    _wire(monkeypatch, "uid_bg_a", "bga@test.dev", "")
    out = await me_mod.get_me(_StubRequest("eyJx.eyJy.sig"), store)
    assert out["billing_live"] is False


@pytest.mark.asyncio
async def test_live_key_reads_billing_live_true(store, monkeypatch):
    _wire(monkeypatch, "uid_bg_b", "bgb@test.dev", "sk_live_abc123")
    await me_mod.signup(_StubRequest("eyJx.eyJy.sig", {}), store)
    out = await me_mod.get_me(_StubRequest("eyJx.eyJy.sig"), store)
    assert out["billing_live"] is True
