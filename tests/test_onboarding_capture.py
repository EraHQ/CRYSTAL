"""T2a pins (2026-09-25): onboarding capture + the first-contact signal.

The connect screen's whole promise rests on stamp_mcp_seen and the
status route agreeing, so both sides of that handshake are pinned, plus
the seat rename landing on the SAME default-admin operator (never a
second one) and the throttle actually throttling.
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


def _wire(monkeypatch, uid, email):
    monkeypatch.setattr(
        me_mod, "get_settings",
        lambda: Settings(firebase_project_id="test-proj"),
    )
    monkeypatch.setattr(
        auth_mod, "get_settings",
        lambda: Settings(firebase_project_id="test-proj"),
    )
    monkeypatch.setattr(
        auth_mod, "_verify_firebase_jwt",
        lambda tok, proj: {"sub": uid, "email": email},
    )


@pytest.mark.asyncio
async def test_signup_captures_name_and_tools(store, monkeypatch):
    _wire(monkeypatch, "uid_ob_1", "ob1@test.dev")
    r = await me_mod.signup(_StubRequest("eyJx.eyJy.sig", {
        "operator_name": "Jordan Reyes",
        "tools": ["claude_desktop", "cursor"],
    }), store)
    assert r["created"] is True
    # The rename landed on the SAME default admin, no second operator.
    ops = await store.list_operators_for_team(r["customer_id"])
    assert len(ops) == 1
    assert ops[0].id == r["operator_id"]
    assert ops[0].display_name == "Jordan Reyes"
    u = await store.get_user_by_id("uid_ob_1")
    assert u.ai_tools == "claude_desktop,cursor"


@pytest.mark.asyncio
async def test_first_contact_stamp_and_status(store, monkeypatch):
    _wire(monkeypatch, "uid_ob_2", "ob2@test.dev")
    r = await me_mod.signup(_StubRequest("eyJx.eyJy.sig", {
        "tools": ["claude_code"],
    }), store)

    out = await me_mod.onboarding_status(_StubRequest("eyJx.eyJy.sig"), store)
    assert out["connected"] is False
    assert out["ai_tools"] == ["claude_code"]

    await store.stamp_mcp_seen(r["customer_id"])
    out = await me_mod.onboarding_status(_StubRequest("eyJx.eyJy.sig"), store)
    assert out["connected"] is True
    assert out["last_seen_at"] is not None


@pytest.mark.asyncio
async def test_stamp_is_throttled(store, customer):
    await store.stamp_mcp_seen(customer.id)
    c1 = await store.get_customer_by_id(customer.id)
    first = c1.last_mcp_seen_at
    assert first is not None
    # Within the throttle window: no rewrite.
    await store.stamp_mcp_seen(customer.id)
    c2 = await store.get_customer_by_id(customer.id)
    assert c2.last_mcp_seen_at == first
