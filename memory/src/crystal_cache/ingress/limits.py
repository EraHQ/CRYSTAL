"""Shared field limits (Lockdown PR-6, Q46=A, 2026-10-09).

One place for the bounded scalar types every request model and query
parameter uses, so a limit is a name, not a number copied around:

  ID64    an identifier (crystal, fact, customer, operator, session, ...):
          1..64 chars of [A-Za-z0-9_.:-]
  CTYPE   a crystal type: `<namespace>:<name>`, lowercase, 3..128 chars
  LIMIT   a page size: 1..200
  OFFSET  a page offset: 0..100,000
  TOPK    a retrieval k: 1..50
  SCOPE   "personal" | "team"
  DAYS    a look-back window in days: 1..3650
  SEQ64   an opaque sequence/cursor token: 1..64 chars
  ISO_TS  an ISO-8601 timestamp as text: 4..40 chars (parsed by the route)
  TOKEN32 a short enum-like filter value (a status, a scope): 1..32 id chars
  SEQ_INT a non-negative sequence number: 0..2**62
  BUDGET  a non-negative work budget (calls, pairs): 0..100,000
  DRIVE_ID a Google Drive file/folder id: 1..128 chars of [A-Za-z0-9_-]

Plus helpers for the two checks pydantic cannot express as a constraint:
`serialized_size_ok` (a dict or list bounded by its JSON size) and
`public_http_url` (a customer-supplied upstream URL that may not point
at this deployment's own network: loopback, link-local, private ranges
or the cloud metadata host).

The body-size guard (`BODY_MAX_BYTES`, with the two per-route
exceptions) lives here too so the middleware in app.py and its pin read
the same numbers.
"""
from __future__ import annotations

import ipaddress
import json
import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Optional
from urllib.parse import urlsplit

from fastapi import HTTPException
from pydantic import Field, StringConstraints

from .errors import build_error_envelope

ID_PATTERN = r"^[A-Za-z0-9_.:-]+$"
CTYPE_PATTERN = r"^[a-z0-9_]+:[a-z0-9_.-]+$"

ID64 = Annotated[str, StringConstraints(min_length=1, max_length=64, pattern=ID_PATTERN)]
CTYPE = Annotated[str, StringConstraints(min_length=3, max_length=128, pattern=CTYPE_PATTERN)]
LIMIT = Annotated[int, Field(ge=1, le=200)]
OFFSET = Annotated[int, Field(ge=0, le=100_000)]
TOPK = Annotated[int, Field(ge=1, le=50)]
SCOPE = Literal["personal", "team"]
DAYS = Annotated[int, Field(ge=1, le=3650)]
SEQ64 = Annotated[str, StringConstraints(min_length=1, max_length=64)]
ISO_TS = Annotated[str, StringConstraints(min_length=4, max_length=40)]
TOKEN32 = Annotated[str, StringConstraints(min_length=1, max_length=32, pattern=ID_PATTERN)]
SEQ_INT = Annotated[int, Field(ge=0, le=2**62)]
BUDGET = Annotated[int, Field(ge=0, le=100_000)]
DRIVE_ID = Annotated[str, StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")]

# Body-size guard (Q46): every route but the two below is refused past
# BODY_MAX_BYTES before the body is parsed; the file upload has its own
# per-file cap (PR-1, CC_UPLOAD_MAX_BYTES) and the topology restore its
# own compressed cap. Content-Length is checked first; a chunked body is
# counted as it streams and cut at the same line.
BODY_MAX_BYTES = 10 * 2**20
BODY_MAX_BYTES_BY_PATH: dict[str, int] = {
    "/v1/documents/upload": 32 * 2**20,
    "/v1/import/topology": 32 * 2**20,
}

# Serialized-size caps for the free-form OpenAI pass-through fields.
TOOL_MAX_BYTES = 16 * 1024
TOOL_CHOICE_MAX_BYTES = 1024
RESPONSE_FORMAT_MAX_BYTES = 32 * 1024


def serialized_size(value: Any) -> int:
    """The JSON size of a value, compact separators, as the wire has it."""
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def body_limit_for(path: str) -> int:
    p = path.rstrip("/") or path
    return BODY_MAX_BYTES_BY_PATH.get(p, BODY_MAX_BYTES)


_PRIVATE_HOSTS = frozenset({
    "localhost", "localhost.localdomain", "metadata.google.internal",
    "metadata", "169.254.169.254",
})


def public_http_url(url: str) -> str:
    """Validate a customer-supplied upstream base URL (Q46,
    CreateCustomerRequest.base_url): http(s) only, a host present, and
    the host not a name or address inside this deployment's own network
    (loopback, link-local, private ranges, the cloud metadata host).
    Returns the URL unchanged; raises ValueError with a public message."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ValueError("base_url must start with http:// or https://")
    host = (parts.hostname or "").strip().lower().rstrip(".")
    if not host:
        raise ValueError("base_url must name a host")
    if host in _PRIVATE_HOSTS or host.endswith(".internal") or host.endswith(".local"):
        raise ValueError("base_url must be a public address")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return url  # a public hostname; resolution happens upstream
    if (
        ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
        or ip.is_reserved or ip.is_unspecified
    ):
        raise ValueError("base_url must be a public address")
    return url


# ---------------------------------------------------------------------------
# Bounds for the raw-JSON routes (the ones that read `await request.json()`
# and validate by hand). A `Bound` names the limit for one field; the
# handler calls `require_bounds(body, {...})` before its own checks, so an
# over-limit value is refused before any of it is touched. Only values
# that are present are checked, by their actual type; a missing field or
# a wrong-typed one is left to the handler's existing checks, so nothing a
# handler accepted before is refused here except by size or by choice.
# The refusal is the app's validation shape (400, invalid_request_error)
# with code `field_too_long` or `invalid_value` and the field as `param`.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Bound:
    """Limits for one JSON field, applied by the value's actual type:
    str: `max_len` chars, `pattern`, `choices`; list: `max_len` items then
    `each` per item; dict: `max_len` keys, `key_max` chars per key, `fields`
    per named key, then `each` per remaining value; int/float: `max_int`."""
    max_len: Optional[int] = None
    each: Optional["Bound"] = None
    choices: Optional[tuple[str, ...]] = None
    pattern: Optional[str] = None
    max_int: Optional[int] = None
    key_max: Optional[int] = None
    fields: Optional[dict[str, "Bound"]] = None


def _refuse(param: str, message: str, code: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=build_error_envelope(
            f"Invalid request: {message}",
            error_type="invalid_request_error",
            param=param,
            code=code,
        ),
    )


def _check(path: str, value: Any, b: Bound) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, str):
        if b.max_len is not None and len(value) > b.max_len:
            raise _refuse(path, f"{path} must be at most {b.max_len} characters", "field_too_long")
        if b.choices is not None and value not in b.choices:
            raise _refuse(path, f"{path} must be one of {', '.join(b.choices)}", "invalid_value")
        if b.pattern is not None and not re.fullmatch(b.pattern, value):
            raise _refuse(path, f"{path} is not in the accepted form", "invalid_value")
        return
    if isinstance(value, (int, float)):
        if b.max_int is not None and abs(value) > b.max_int:
            raise _refuse(path, f"{path} must be at most {b.max_int}", "field_too_long")
        return
    if isinstance(value, (list, tuple)):
        if b.max_len is not None and len(value) > b.max_len:
            raise _refuse(path, f"{path} must have at most {b.max_len} items", "field_too_long")
        if b.each is not None:
            for i, item in enumerate(value):
                _check(f"{path}[{i}]", item, b.each)
        return
    if isinstance(value, dict):
        if b.max_len is not None and len(value) > b.max_len:
            raise _refuse(path, f"{path} must have at most {b.max_len} keys", "field_too_long")
        for k, v in value.items():
            if b.key_max is not None and isinstance(k, str) and len(k) > b.key_max:
                raise _refuse(path, f"{path} keys must be at most {b.key_max} characters", "field_too_long")
            sub = (b.fields or {}).get(k) if isinstance(k, str) else None
            if sub is None:
                sub = b.each
            if sub is not None:
                _check(f"{path}.{k}", v, sub)
        return


def require_bounds(body: Any, spec: dict[str, Bound]) -> None:
    """Refuse a raw-JSON body whose present fields exceed `spec`. A body
    that is not an object is left to the handler."""
    if not isinstance(body, dict):
        return
    for name, bound in spec.items():
        if name in body:
            _check(name, body[name], bound)
