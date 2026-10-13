"""Request body size guard (Lockdown PR-6, Q46=A, 2026-10-09).

One ASGI middleware, outside every router: a body larger than the
route's limit is refused with 413 before any handler parses it.

  - `Content-Length` is checked first; a declared size past the limit is
    answered at once and the body is never read.
  - A chunked body (no Content-Length, or a lying one) is counted as the
    handler reads it; the byte that crosses the limit raises
    `BodyTooLarge` out of `receive()`, which the app's exception handler
    turns into the same 413 envelope.

Limits come from `ingress/limits.py`: BODY_MAX_BYTES (10 MiB) for every
route, with `/v1/documents/upload` and `/v1/import/topology` at 32 MiB
(the upload has its own per-file cap from PR-1, the restore its own
compressed cap). The envelope is the OpenAI shape every other refusal
uses, code `body_too_large`.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from fastapi import HTTPException

from .limits import body_limit_for


def too_large_envelope(limit: int) -> dict[str, Any]:
    return {
        "error": {
            "message": (
                f"Request body is larger than {limit // 2**20} MiB. "
                "Send less in one request."
            ),
            "type": "invalid_request_error",
            "param": None,
            "code": "body_too_large",
        }
    }


class BodyTooLarge(HTTPException):
    """Raised out of `receive()` when a streamed body crosses the limit.

    An HTTPException on purpose: FastAPI reads a JSON body inside a
    try/except that turns ANY other exception into a 400 "error parsing
    the body" and re-raises only HTTPException (fastapi/routing.py,
    get_request_handler). As an HTTPException it reaches the app's
    handlers intact: the dedicated BodyTooLarge handler renders 413, and
    even the generic HTTPException handler would (the detail IS the
    envelope, which that handler passes through)."""

    def __init__(self, limit: int, path: str = ""):
        super().__init__(status_code=413, detail=too_large_envelope(limit))
        self.limit = limit
        self.path = path


async def _send_413(send: Callable, limit: int) -> None:
    body = json.dumps(too_large_envelope(limit)).encode("utf-8")
    await send({
        "type": "http.response.start",
        "status": 413,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
    })
    await send({"type": "http.response.body", "body": body})


class BodySizeGuard:
    """Pure ASGI middleware; see the module docstring."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "") or ""
        limit = body_limit_for(path)

        for name, value in scope.get("headers", []) or []:
            if name == b"content-length":
                try:
                    declared = int(value.decode("latin-1").strip())
                except ValueError:
                    declared = None
                if declared is not None and declared > limit:
                    await _send_413(send, limit)
                    return
                break

        seen = 0

        async def counting_receive() -> dict:
            nonlocal seen
            message = await receive()
            if message.get("type") == "http.request":
                seen += len(message.get("body", b"") or b"")
                if seen > limit:
                    raise BodyTooLarge(limit, path)
            return message

        await self.app(scope, counting_receive, send)
