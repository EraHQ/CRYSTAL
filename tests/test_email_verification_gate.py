"""T1b pins (S4.6, 2026-09-24): the email-verification gate on signup.

Creation-only by design: password-provider accounts must carry
email_verified before a tenant is born; OAuth providers pass by
construction; existing identities resolve regardless (a legacy
unverified account is never locked out of its own console).
"""
import pytest
from fastapi import HTTPException

from crystal_cache.config import Settings
from crystal_cache.endpoints import me as me_mod
from crystal_cache.ingress import auth as auth_mod


class _StubRequest:
    def __init__(self, token):
        self.headers = {"authorization": f"Bearer {token}"}

    async def json(self):
        raise ValueError("no body")


def _wire(monkeypatch, claims):
    monkeypatch.setattr(
        me_mod, "get_settings",
        lambda: Settings(firebase_project_id="test-proj"),
    )
    monkeypatch.setattr(
        auth_mod, "_verify_firebase_jwt", lambda tok, proj: claims,
    )


@pytest.mark.asyncio
async def test_unverified_password_signup_is_refused(store, monkeypatch):
    _wire(monkeypatch, {
        "sub": "uid_ev_1", "email": "ev1@test.dev",
        "email_verified": False,
        "firebase": {"sign_in_provider": "password"},
    })
    with pytest.raises(HTTPException) as e:
        await me_mod.signup(_StubRequest("eyJx.eyJy.sig"), store)
    assert e.value.status_code == 403
    assert "Verify your email" in e.value.detail
    # No tenant was born.
    assert await store.get_user_by_id("uid_ev_1") is None


@pytest.mark.asyncio
async def test_verified_password_signup_creates(store, monkeypatch):
    _wire(monkeypatch, {
        "sub": "uid_ev_2", "email": "ev2@test.dev",
        "email_verified": True,
        "firebase": {"sign_in_provider": "password"},
    })
    r = await me_mod.signup(_StubRequest("eyJx.eyJy.sig"), store)
    assert r["created"] is True and r["api_key"]


@pytest.mark.asyncio
async def test_oauth_provider_passes_without_claim(store, monkeypatch):
    _wire(monkeypatch, {
        "sub": "uid_ev_3", "email": "ev3@test.dev",
        "firebase": {"sign_in_provider": "google.com"},
    })
    r = await me_mod.signup(_StubRequest("eyJx.eyJy.sig"), store)
    assert r["created"] is True


@pytest.mark.asyncio
async def test_existing_identity_resolves_even_unverified(store, monkeypatch):
    # Born verified…
    _wire(monkeypatch, {
        "sub": "uid_ev_4", "email": "ev4@test.dev",
        "email_verified": True,
        "firebase": {"sign_in_provider": "password"},
    })
    r1 = await me_mod.signup(_StubRequest("eyJx.eyJy.sig"), store)
    assert r1["created"] is True
    # …later arrives with an unverified-looking token: identity still
    # resolves (creation-only gate).
    _wire(monkeypatch, {
        "sub": "uid_ev_4", "email": "ev4@test.dev",
        "email_verified": False,
        "firebase": {"sign_in_provider": "password"},
    })
    r2 = await me_mod.signup(_StubRequest("eyJx.eyJy.sig"), store)
    assert r2["created"] is False
    assert r2["user_id"] == r1["user_id"]
