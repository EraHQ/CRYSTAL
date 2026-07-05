"""Keyless admin chat + learn proxy (inspector playground) — cleanup after
the hashed-key change made admin_key fetch impossible.

The happy paths run the full chat / learn pipelines (upstream call +
app.state), so they're exercised against the live server, not here. These
cover the cheap, deterministic guard: an unknown customer id resolves to a
404 before any pipeline work runs.

asyncio_mode=auto (pyproject) — async tests need no marker.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from crystal_cache.endpoints.admin import (
    admin_customer_chat,
    admin_customer_learn,
)
from crystal_cache.ingress.schema import ChatCompletionRequest, LearnRequest


class _FakeRequest:
    """Stand-in for fastapi.Request; never dereferenced on the 404 path —
    the customer lookup raises before run_chat_completion / run_learn touch
    request.app.state."""
    pass


async def test_admin_chat_unknown_customer_404(store):
    body = ChatCompletionRequest(
        model="gpt-4", messages=[{"role": "user", "content": "hi"}],
    )
    with pytest.raises(HTTPException) as exc:
        await admin_customer_chat(
            customer_id="cus_ghost", body=body, request=_FakeRequest(), store=store,
        )
    assert exc.value.status_code == 404


async def test_admin_learn_unknown_customer_404(store):
    body = LearnRequest(prompt="p", response="r", outcome="fail")
    with pytest.raises(HTTPException) as exc:
        await admin_customer_learn(
            customer_id="cus_ghost", body=body, request=_FakeRequest(), store=store,
        )
    assert exc.value.status_code == 404
