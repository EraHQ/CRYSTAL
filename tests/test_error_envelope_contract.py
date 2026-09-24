"""The error-envelope wire contract (live-found 2026-09-24).

Why this file exists: the console shipped a 402/403 humane-message
reader against FastAPI's DEFAULT error shape ({"detail": ...}) while
this app wraps every HTTPException in the OpenAI envelope
({"error": {"message": ...}}). Nothing caught it because the route pins
call functions directly, which bypasses the app-level exception
handlers where the envelope is applied. These pins exercise the actual
translation layer, so the wire shape the frontend (and every SDK
consumer) depends on can never drift silently again. The frontend's
mirror of this contract lives in frontend/src/lib/errorMessage.test.ts.
"""
import json

import pytest
from fastapi import HTTPException

from crystal_cache.app import http_exception_handler


class _Req:
    """The handler only logs from the request; a stub suffices."""
    url = "http://test/v1/auth/signup"
    method = "POST"


async def _envelope_for(exc: HTTPException) -> tuple[int, dict]:
    resp = await http_exception_handler(_Req(), exc)
    return resp.status_code, json.loads(resp.body)


@pytest.mark.asyncio
async def test_http_exceptions_speak_the_openai_envelope():
    status, body = await _envelope_for(
        HTTPException(status_code=403, detail="Verify your email to finish")
    )
    assert status == 403
    # The envelope, exactly: error.message carries the humane string,
    # all four inner keys present, and NO top-level "detail" — the
    # FastAPI default shape is never emitted by this app.
    assert body["error"]["message"] == "Verify your email to finish"
    assert body["error"]["type"] == "permission_error"
    assert set(body["error"].keys()) == {"message", "type", "param", "code"}
    assert "detail" not in body


@pytest.mark.asyncio
async def test_402_walls_keep_their_humane_message_on_the_wire():
    detail = (
        "Memory is full (550 crystals; your plan holds 500). Everything "
        "stored stays fully recallable and exportable. Upgrade in the "
        "console to keep remembering."
    )
    status, body = await _envelope_for(
        HTTPException(status_code=402, detail=detail)
    )
    assert status == 402
    assert body["error"]["message"] == detail
    # 402 has no type_map entry — the fallback must stay stable too.
    assert body["error"]["type"] == "api_error"


@pytest.mark.asyncio
async def test_prebuilt_envelopes_pass_through_unwrapped():
    inner = {"error": {"message": "already enveloped", "type": "api_error",
                       "param": None, "code": None}}
    status, body = await _envelope_for(
        HTTPException(status_code=403, detail=inner)
    )
    assert status == 403
    assert body == inner  # never double-wrapped
