"""Lockdown PR-6a pins (Q46=A field limits, 2026-10-09).

Three lanes, all on the real routes:

  1. Over-limit values on the wire. One over-limit value per field,
     sent through the HTTP stack with the app's own validation and
     HTTPException handlers installed (the OpenAI envelope: a Pydantic
     refusal is 400 `invalid_request_error` with the field as `param`;
     a raw-JSON route's bound is the same shape with code
     `field_too_long` / `invalid_value`).
  2. The body-size guard: a declared Content-Length past the limit is
     413 before the handler runs; a chunked body that crosses the limit
     is cut as the handler reads it and is 413 too; the two 32 MiB
     routes keep their larger limit.
  3. Structure: every request model reachable from the app's routes has
     extra="forbid" except the two OpenAI-compat models; every str/int
     path, query and form parameter on every route carries a bound
     (an Annotated max_length / le, a Literal, or a bounded Query/Form);
     both walks assert a floor so an empty walk cannot pass.
"""
from __future__ import annotations

import inspect
import types
import typing
from typing import Annotated, Any, Literal, Optional, Union, get_args, get_origin

import annotated_types
import pytest
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo
from pydantic.types import StringConstraints

from crystal_cache.infrastructure.schema import DocumentUploadRow
from crystal_cache.ingress import limits
from crystal_cache.ingress.body_guard import BodySizeGuard, BodyTooLarge
from crystal_cache.ingress.limits import Bound, require_bounds
from crystal_cache.ingress.schema import ChatCompletionRequest, ChatMessage


def _app(store, encoder, vector_store, fact_vector_store):
    """The smoke-test app (every router, in-memory store) with the real
    app's guard and handlers on it, so the wire shape under test is the
    deployed one."""
    try:
        from tests.test_endpoint_smoke import _build_app
    except ModuleNotFoundError:
        from test_endpoint_smoke import _build_app
    from crystal_cache import app as app_mod

    app = _build_app(store, encoder, vector_store, fact_vector_store)
    app.add_middleware(BodySizeGuard)
    app.add_exception_handler(RequestValidationError, app_mod.validation_error_handler)
    app.add_exception_handler(HTTPException, app_mod.http_exception_handler)
    app.add_exception_handler(BodyTooLarge, app_mod.body_too_large_handler)
    return app


def _client(app):
    from httpx import ASGITransport, AsyncClient

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


def _err(r) -> dict:
    body = r.json()
    assert isinstance(body.get("error"), dict), body
    return body["error"]


# ---------------------------------------------------------------------------
# 1. Over-limit values on the wire
# ---------------------------------------------------------------------------

LONG_ID = "x" * 65
_MSG = {"role": "user", "content": "hi"}

# (method, path, kwargs for the request, expected param, expected code or None)
# `code` None means the Pydantic lane (param carries the field); a code
# means a raw-JSON route's require_bounds refusal.
WIRE_CASES = [
    # --- ingress/schema.py request models ---
    ("POST", "/v1/retrieve", dict(json={"query": "q", "crystal_type": "Bad Type"}), "crystal_type", None),
    ("POST", "/v1/retrieve", dict(json={"query": "q", "k": 999}), "k", None),
    ("POST", "/v1/store", dict(json={"key": "k", "value": "v", "scope": "public"}), "scope", None),
    ("POST", "/v1/store", dict(json={"key": "k", "value": "v", "surprise": 1}), "surprise", None),
    ("POST", "/v1/learn", dict(json={"prompt": "p", "response": "r", "outcome": "pass", "crystal_type": "nocolon"}), "crystal_type", None),
    # --- endpoint-local models ---
    ("POST", "/v1/sessions/heartbeat", dict(json={"session_id": "s1", "status": "dancing"}), "status", None),
    ("POST", "/v1/sessions/heartbeat", dict(json={"session_id": LONG_ID}), "session_id", None),
    ("POST", "/v1/sessions/heartbeat", dict(json={"session_id": "s1", "pid": 2**40}), "pid", None),
    ("POST", "/v1/agent/messages", dict(json={"messages": [_MSG] * 501}), "messages", None),
    ("POST", "/v1/agent/messages", dict(json={"messages": [_MSG], "max_tokens": 10**9}), "max_tokens", None),
    ("POST", "/v1/agent/messages", dict(json={"messages": [_MSG], "thinking": {"type": "enabled", "budget_tokens": 1024, "extra": 1}}), "thinking.extra", None),
    ("POST", "/v1/crystals/c1/scope", dict(json={"scope": "public"}), "scope", None),
    ("POST", "/v1/crystals/c1/grants", dict(json={"principal_type": "world", "principal_id": "x"}), "principal_type", None),
    # --- query and path parameters ---
    ("GET", "/v1/crystals", dict(params={"limit": 201}), "limit", None),
    ("GET", "/v1/crystals", dict(params={"offset": 100_001}), "offset", None),
    ("GET", f"/v1/crystals/{LONG_ID}", {}, "crystal_id", None),
    ("GET", "/v1/documents", dict(params={"status": "s" * 33}), "status", None),
    ("GET", "/v1/documents", dict(params={"limit": 0}), "limit", None),
    # --- raw-JSON routes (require_bounds) ---
    ("PUT", "/v1/documents/doc_x/review", dict(json={"content_chunks": [{}] * 5001}), "content_chunks", "field_too_long"),
    ("PUT", "/v1/documents/doc_x/review", dict(json={"extracted_items": [{}] * 2001}), "extracted_items", "field_too_long"),
    ("PUT", "/v1/documents/doc_x/review", dict(json={"confirmed_type": "not a kind!"}), "confirmed_type", "invalid_value"),
    ("PUT", "/api/dsl_configs/cfg", dict(json={"source_text": "x" * 100_001}), "source_text", "field_too_long"),
    ("PUT", "/api/dsl_configs/no spaces", dict(json={"source_text": "x"}), "name", None),
    ("POST", "/admin/api/source-schemas/preview", dict(json={"mapping": {"roles": {"a": "skip"}}, "sample": [{}] * 201}), "sample", "field_too_long"),
    ("POST", "/admin/api/source-schemas/preview", dict(json={"mapping": {"roles": {"a": "skip"}}, "label": "l" * 257}), "label", "field_too_long"),
    ("POST", "/admin/api/source-schemas/preview", dict(json={"mapping": {"roles": {"a" * 257: "skip"}}}), "mapping.roles", "field_too_long"),
]


@pytest.mark.asyncio
async def test_over_limit_values_are_refused_on_the_wire(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    headers = {"Authorization": f"Bearer {customer.api_key}"}
    async with _client(app) as c:
        for method, path, kw, param, code in WIRE_CASES:
            r = await c.request(method, path, headers=headers, **kw)
            assert r.status_code == 400, f"{method} {path} {param}: {r.status_code} {r.text[:300]}"
            err = _err(r)
            assert err["type"] == "invalid_request_error", (path, err)
            assert err["param"] == param or (err["param"] or "").startswith(param), (path, err)
            if code is not None:
                assert err["code"] == code, (path, err)

        # Self-or-admin raw routes: the bound is checked before the body
        # is used (the caller is the customer itself).
        cid = customer.id
        for method, path, body, param in (
            ("PUT", f"/v1/customers/{cid}/budgets/auto_research", {"cap_micro_usd": 10**13}, "cap_micro_usd"),
            ("PATCH", f"/v1/customers/{cid}/model", {"model_id": "m" * 129}, "model_id"),
            ("PATCH", f"/v1/customers/{cid}/upstream_key", {"api_key_ref": "k" * 513}, "api_key_ref"),
            ("PATCH", f"/v1/customers/{cid}/inference_mode", {"inference_mode": "m" * 33}, "inference_mode"),
            ("PUT", f"/v1/customers/{cid}/budgets/Not-A-Function", {"cap_micro_usd": 1}, "function"),
        ):
            r = await c.request(method, path, headers=headers, json=body)
            assert r.status_code == 400, f"{method} {path}: {r.status_code} {r.text[:300]}"
            assert (_err(r)["param"] or "").startswith(param), (path, r.text[:300])

        # Operator seed (admin surface; open in the test app): name and facts.
        for body, param in (
            ({"name": "n" * 257}, "name"),
            ({"name": "ok", "facts": ["f"] * 51}, "facts"),
            ({"name": "ok", "facts": ["f" * 2001]}, "facts[0]"),
        ):
            r = await c.post(f"/admin/api/customers/{cid}/operators/seed", json=body)
            assert r.status_code == 400, (body, r.status_code, r.text[:300])
            assert _err(r)["code"] == "field_too_long" and _err(r)["param"] == param, r.text[:300]

        # Approve: the lists are bounded after the row is found (a real
        # review row), before the selection is merged.
        async with store.session() as s:
            s.add(DocumentUploadRow(
                id="up_q46", customer_id=cid, label="q46", text="t", status="review",
                source_uri="upload://up_q46", content_chunks=[{"index": 0, "text": "t"}],
                extracted_items=[],
            ))
        r = await c.post("/v1/documents/up_q46/approve", headers=headers,
                         json={"items": [{"key": "k", "value": "v"}] * 2001})
        assert r.status_code == 400 and _err(r)["param"] == "items", r.text[:300]


@pytest.mark.asyncio
async def test_bounds_leave_in_range_bodies_alone_and_the_two_compat_models_stay_open(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    """The guard is a ceiling, not a new shape: a body that was valid
    before PR-6 still is, and the two OpenAI-compat models still take
    the extra keys clients send."""
    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    headers = {"Authorization": f"Bearer {customer.api_key}"}
    async with _client(app) as c:
        r = await c.post("/v1/store", headers=headers,
                         json={"key": "pr6 key", "value": "pr6 value", "crystal_type": "customer:legacy"})
        assert r.status_code == 200, r.text[:300]
        r = await c.get("/v1/crystals", headers=headers, params={"limit": 200, "offset": 0})
        assert r.status_code == 200, r.text[:300]
    assert ChatCompletionRequest.model_config.get("extra") == "allow"
    assert ChatMessage.model_config.get("extra") == "allow"
    body = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], vendor_x=1)
    assert body.model_extra == {"vendor_x": 1}
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}] * 501)


def test_require_bounds_checks_by_actual_type_and_names_the_field():
    """The raw-JSON helper: present values are bounded by their type; a
    missing or wrong-typed field is left to the handler; the refusal is
    the app's 400 envelope naming the field."""
    spec = {
        "text": Bound(max_len=5),
        "n": Bound(max_int=10),
        "tags": Bound(max_len=2, each=Bound(max_len=3)),
        "kind": Bound(choices=("a", "b")),
        "m": Bound(max_len=2, key_max=2, fields={"k": Bound(max_len=1)}),
    }
    require_bounds({"text": "12345", "n": -10, "tags": ["abc", "d"], "kind": "a", "m": {"k": "x", "zz": 1}}, spec)
    # A wrong-typed value is bounded by ITS type, so a value that fits that
    # type's bound passes here and the handler's own type check decides.
    require_bounds({"text": None, "n": "not an int", "tags": 7, "kind": 3, "m": ["a", "b"]}, spec)
    require_bounds("not an object", spec)
    for body, param, code in (
        ({"text": "123456"}, "text", "field_too_long"),
        ({"n": 11}, "n", "field_too_long"),
        ({"n": -11}, "n", "field_too_long"),
        ({"tags": ["a", "b", "c"]}, "tags", "field_too_long"),
        ({"tags": ["abcd"]}, "tags[0]", "field_too_long"),
        ({"kind": "c"}, "kind", "invalid_value"),
        ({"m": {"a": 1, "b": 2, "c": 3}}, "m", "field_too_long"),
        ({"m": {"abc": 1}}, "m", "field_too_long"),
        ({"m": {"k": "xy"}}, "m.k", "field_too_long"),
    ):
        with pytest.raises(HTTPException) as exc:
            require_bounds(body, spec)
        assert exc.value.status_code == 400
        env = exc.value.detail["error"]
        assert env["param"] == param and env["code"] == code and env["type"] == "invalid_request_error", env


# ---------------------------------------------------------------------------
# 2. The body-size guard
# ---------------------------------------------------------------------------

async def _drive(app, path: str, chunks: list[bytes], *, content_length: Optional[int]):
    """Run one request through BodySizeGuard as the ASGI server would."""
    sent: list[dict] = []
    headers = [(b"content-type", b"application/json")]
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    scope = {"type": "http", "method": "POST", "path": path, "headers": headers}
    pending = [{"type": "http.request", "body": ch, "more_body": i < len(chunks) - 1}
               for i, ch in enumerate(chunks)] or [{"type": "http.request", "body": b"", "more_body": False}]

    async def receive():
        return pending.pop(0)

    async def send(message):
        sent.append(message)

    await BodySizeGuard(app)(scope, receive, send)
    return sent


@pytest.mark.asyncio
async def test_body_guard_refuses_a_declared_oversize_body_before_the_app_runs():
    ran = []

    async def inner(scope, receive, send):
        ran.append(scope["path"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    limit = limits.BODY_MAX_BYTES
    sent = await _drive(inner, "/v1/store", [b"{}"], content_length=limit + 1)
    assert sent[0]["status"] == 413 and not ran
    assert b'"code": "body_too_large"' in sent[1]["body"]
    # At the limit exactly: through.
    sent = await _drive(inner, "/v1/store", [b"{}"], content_length=limit)
    assert sent[0]["status"] == 200 and ran == ["/v1/store"]
    # The two large routes keep their own limit, trailing slash or not.
    for path in ("/v1/documents/upload", "/v1/import/topology", "/v1/import/topology/"):
        assert limits.body_limit_for(path) == 32 * 2**20
        ran.clear()
        sent = await _drive(inner, path, [b"{}"], content_length=20 * 2**20)
        assert sent[0]["status"] == 200 and ran, path
        sent = await _drive(inner, path, [b"{}"], content_length=32 * 2**20 + 1)
        assert sent[0]["status"] == 413, path
    assert limits.body_limit_for("/v1/documents/upload/extra") == limits.BODY_MAX_BYTES
    # Non-HTTP scopes pass straight through.
    seen = []

    async def ws(scope, receive, send):
        seen.append(scope["type"])

    await BodySizeGuard(ws)({"type": "websocket", "path": "/x"}, None, None)
    assert seen == ["websocket"]


@pytest.mark.asyncio
async def test_body_guard_cuts_a_chunked_body_at_the_limit():
    """No Content-Length (or a lying one): the byte that crosses the
    limit raises BodyTooLarge out of receive(), inside the app."""
    limit = limits.BODY_MAX_BYTES
    read: list[int] = []

    async def inner(scope, receive, send):
        while True:
            m = await receive()
            read.append(len(m.get("body", b"")))
            if not m.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    half = limit // 2
    with pytest.raises(BodyTooLarge) as exc:
        await _drive(inner, "/v1/store", [b"x" * half, b"x" * half, b"x"], content_length=None)
    assert exc.value.limit == limit and exc.value.path == "/v1/store"
    assert read == [half, half]  # the third chunk never reached the app
    # A lying Content-Length below the limit does not help.
    read.clear()
    with pytest.raises(BodyTooLarge):
        await _drive(inner, "/v1/store", [b"x" * (limit + 1)], content_length=10)
    # Exactly the limit, chunked: through.
    read.clear()
    sent = await _drive(inner, "/v1/store", [b"x" * half, b"x" * half], content_length=None)
    assert sent[0]["status"] == 200 and sum(read) == limit


@pytest.mark.asyncio
async def test_chunked_oversize_body_is_413_through_the_real_handler(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    """End to end: a streamed body past the limit on a real JSON route
    is the 413 envelope from the app's BodyTooLarge handler, and the
    handler never ran."""
    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    headers = {"Authorization": f"Bearer {customer.api_key}", "Content-Type": "application/json"}
    limit = limits.BODY_MAX_BYTES

    async def gen():
        yield b'{"key": "k", "value": "'
        yield b"v" * limit
        yield b'"}'

    async with _client(app) as c:
        r = await c.post("/v1/store", headers=headers, content=gen())
        assert r.status_code == 413, r.text[:300]
        err = _err(r)
        assert err["code"] == "body_too_large" and err["type"] == "invalid_request_error"


def test_the_real_app_mounts_the_guard_and_the_handler():
    from crystal_cache.app import app

    classes = [mw.cls for mw in app.user_middleware]
    assert BodySizeGuard in classes
    assert BodyTooLarge in app.exception_handlers
    assert RequestValidationError in app.exception_handlers


# ---------------------------------------------------------------------------
# 3. Structure: every request model forbids extras; every scalar
#    parameter is bounded
# ---------------------------------------------------------------------------

OPEN_MODELS = {ChatMessage, ChatCompletionRequest}
MIN_REQUEST_MODELS = 25
MIN_SCALAR_PARAMS = 100


def _models_in(annotation: Any, out: set) -> None:
    """Every BaseModel subclass reachable from a type annotation."""
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        if annotation in out:
            return
        out.add(annotation)
        if annotation in OPEN_MODELS:
            return  # their nested shapes are the compat surface
        for f in annotation.model_fields.values():
            _models_in(f.annotation, out)
        return
    for a in get_args(annotation):
        _models_in(a, out)


def _walk(routes):
    """FastAPI keeps an included router in app.routes as an _IncludedRouter
    whose `original_router.routes` are the flat APIRoute objects."""
    for route in routes:
        orig = getattr(route, "original_router", None)
        if orig is not None:
            yield from _walk(orig.routes)
        else:
            yield route


def _routes():
    from crystal_cache.app import app

    rs = [r for r in _walk(app.routes) if isinstance(r, APIRoute)]
    assert len(rs) >= 100, len(rs)
    return rs


def test_every_request_model_forbids_extra_fields():
    found: set = set()
    for route in _routes():
        for field in route.dependant.body_params:
            _models_in(field.field_info.annotation, found)
            for m in getattr(field.field_info, "metadata", []):
                _models_in(m, found)
    found -= OPEN_MODELS
    assert len(found) >= MIN_REQUEST_MODELS, sorted(m.__name__ for m in found)
    loose = sorted(m.__name__ for m in found if m.model_config.get("extra") != "forbid")
    assert not loose, loose


def _strip_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        return args[0] if len(args) == 1 else annotation
    return annotation


def _has_bound(metadata: list, base: Any) -> bool:
    for m in metadata:
        if isinstance(m, FieldInfo):
            if _has_bound(m.metadata, base):
                return True
            continue
        if base is str:
            if isinstance(m, StringConstraints) and m.max_length is not None:
                return True
            if isinstance(m, annotated_types.MaxLen):
                return True
        elif base in (int, float):
            if isinstance(m, (annotated_types.Le, annotated_types.Lt)):
                return True
    return False


def _bounded(annotation: Any, default: Any) -> Optional[bool]:
    """True/False for a scalar parameter; None for anything that is not
    a scalar str/int/float (models, Request, Depends, bool, files)."""
    annotation = _strip_optional(annotation)
    origin = get_origin(annotation)
    if origin is Literal:
        return True
    if origin is Annotated:
        base, *meta = get_args(annotation)
        base = _strip_optional(base)
        if get_origin(base) is Literal:
            return True
        if base not in (str, int, float):
            return None
        if _has_bound(list(meta), base):
            return True
        return isinstance(default, FieldInfo) and _has_bound(default.metadata, base)
    if annotation in (str, int, float):
        return isinstance(default, FieldInfo) and _has_bound(default.metadata, base=annotation)
    return None


def test_every_scalar_route_parameter_is_bounded():
    unbounded: list[str] = []
    unresolved: list[str] = []
    checked = 0
    for route in _routes():
        try:
            hints = typing.get_type_hints(route.endpoint, include_extras=True)
        except Exception as e:  # noqa: BLE001 — a hint that cannot resolve is a finding, not a skip
            unresolved.append(f"{route.path} {route.endpoint.__name__}: {type(e).__name__}: {e}")
            continue
        sig = inspect.signature(route.endpoint)
        for name, p in sig.parameters.items():
            if name not in hints:
                continue
            verdict = _bounded(hints[name], p.default)
            if verdict is None:
                continue
            checked += 1
            if not verdict:
                unbounded.append(f"{route.path} {route.endpoint.__name__}.{name}")
    assert not unresolved, "\n".join(unresolved)
    assert checked >= MIN_SCALAR_PARAMS, checked
    assert not unbounded, "\n".join(unbounded)


def test_shared_limit_types_are_what_the_report_says():
    """The names the report cites, pinned to their numbers."""
    from pydantic import TypeAdapter

    def ok(t, v):
        TypeAdapter(t).validate_python(v)

    def bad(t, v):
        with pytest.raises(ValidationError):
            TypeAdapter(t).validate_python(v)

    ok(limits.ID64, "a" * 64); bad(limits.ID64, "a" * 65); bad(limits.ID64, ""); bad(limits.ID64, "a b")
    ok(limits.CTYPE, "customer:legacy"); bad(limits.CTYPE, "Customer:Legacy"); bad(limits.CTYPE, "x:" + "y" * 127)
    ok(limits.LIMIT, 200); bad(limits.LIMIT, 201); bad(limits.LIMIT, 0)
    ok(limits.OFFSET, 100_000); bad(limits.OFFSET, 100_001); bad(limits.OFFSET, -1)
    ok(limits.TOPK, 50); bad(limits.TOPK, 51)
    ok(limits.DAYS, 3650); bad(limits.DAYS, 3651); bad(limits.DAYS, 0)
    ok(limits.TOKEN32, "needs_operator"); bad(limits.TOKEN32, "x" * 33)
    ok(limits.SCOPE, "team"); bad(limits.SCOPE, "public")
    assert limits.BODY_MAX_BYTES == 10 * 2**20
    assert limits.public_http_url("https://api.example.com/v1") == "https://api.example.com/v1"
    for bad_url in ("http://localhost:8000", "http://169.254.169.254/", "https://10.0.0.5/",
                    "http://metadata.google.internal/", "ftp://example.com/", "https://[::1]/",
                    "https://svc.internal/"):
        with pytest.raises(ValueError):
            limits.public_http_url(bad_url)
