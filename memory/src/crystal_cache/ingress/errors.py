"""OpenAI-compatible error envelope and exception types.

Phase 1.5.5 (May 2026). The OpenAI Python SDK parses error responses
by reading `body["error"]` and dispatching on `type`:

  - "invalid_request_error" → openai.BadRequestError
  - "authentication_error"  → openai.AuthenticationError
  - "permission_error"      → openai.PermissionDeniedError
  - "not_found_error"       → openai.NotFoundError
  - "conflict_error"        → openai.ConflictError
  - "rate_limit_error"      → openai.RateLimitError
  - "api_error"             → openai.APIError (catchall)

Error envelope shape (all keys required, even when null):

    {
      "error": {
        "message": str,        # human-readable, surfaced to caller
        "type": str,           # SDK dispatch key
        "param": str | null,   # which field caused it (validation errors)
        "code": str | null     # machine-readable subtype (e.g. "upstream_error")
      }
    }

Why this exists. Pre-1.5.5 the gateway raised bare HTTPException with
inconsistent payload shapes — bare strings on auth/404, nested dicts on
upstream 502, FastAPI's default `{"detail": [...]}` on validation
errors. The OpenAI SDK's error parser fails on all three, leaving
customer code to either pattern-match on substrings or just retry
blindly. The envelope below is the contract that lets `try: ... except
openai.AuthenticationError` work end-to-end.

Greenfield posture: this is a breaking change to the response shape
(no users yet, per Finding 17). Tests asserting on the old `detail`
field need to flip to `error.message`.
"""
from __future__ import annotations

from typing import Any, Optional


# ---------------------------------------------------------------------------
# Exception types
# ---------------------------------------------------------------------------

class CrystalCacheError(Exception):
    """Base for all in-process exceptions that surface as HTTP errors.

    Carries the shape data the FastAPI handler needs to build the
    OpenAI envelope. Subclasses pin `http_status` and `error_type`;
    `param` and `code` are per-instance.

    Subclassing is preferred over raising the base directly — that way
    the type system (and reading the code) makes the error category
    explicit at the raise site.
    """

    http_status: int = 500
    error_type: str = "api_error"

    def __init__(
        self,
        message: str,
        *,
        param: Optional[str] = None,
        code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.param = param
        self.code = code


class InvalidRequestError(CrystalCacheError):
    """400-class: malformed input, missing required field, bad value."""

    http_status = 400
    error_type = "invalid_request_error"


class AuthenticationError(CrystalCacheError):
    """401-class: missing or invalid api_key."""

    http_status = 401
    error_type = "authentication_error"


class NotFoundError(CrystalCacheError):
    """404-class: resource doesn't exist (or caller can't see it)."""

    http_status = 404
    error_type = "not_found_error"


class UpstreamError(CrystalCacheError):
    """The configured upstream LLM returned an error or transport-failed.

    Carries the upstream status code in `code` (formatted as
    `upstream_<status>`) so SDK consumers can programmatically
    distinguish "downstream issue" from "our system's fault" without
    parsing the message. Per Phase 1.5.5 lean answer, the upstream
    response body is NOT included — leaks provider details and was
    only ever useful for our own debugging (which we have logs for).
    """

    http_status = 502
    error_type = "api_error"

    def __init__(
        self,
        message: str,
        *,
        upstream_status: Optional[int] = None,
    ) -> None:
        if upstream_status is not None:
            code = f"upstream_{upstream_status}"
        else:
            code = "upstream_error"
        super().__init__(message, code=code)
        self.upstream_status = upstream_status


class RoutingError(CrystalCacheError):
    """Retrieval or routing failure that surfaces as 500.

    Placeholder for future retrieval failures that should bubble out of
    the gateway. Today retrieval errors degrade to passthrough rather
    than raising; this type is here so when we add stricter modes
    (e.g. "fail closed if vector store unreachable") the shape is
    pre-defined.
    """

    http_status = 500
    error_type = "api_error"


# ---------------------------------------------------------------------------
# Envelope builder
# ---------------------------------------------------------------------------

def build_error_envelope(
    message: str,
    *,
    error_type: str,
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> dict[str, Any]:
    """Construct the OpenAI-compatible error response body.

    All four inner keys are always present (null when unused) so SDK
    consumers can rely on the shape rather than checking each field's
    existence.
    """
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }
