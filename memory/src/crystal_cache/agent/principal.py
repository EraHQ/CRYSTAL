"""Request-scoped acting operator — the ACL principal for the tool surface.

Phase 1.4 (Q1=A, ratified 2026-08-24): the agent's tool surface cannot
carry a per-request principal through `_tool_state` (a module global set
at Agent construction — concurrent runs would clobber each other), so
the acting operator rides a ContextVar instead — the same pattern the
MCP surface already uses for customer identity
(`agent/mcp_server.py::_current_customer_id`).

WHO SETS IT (the doors — identity comes from the API key, never from
tool arguments or request bodies, P0.23):
  - `POST /v1/agent/messages`, after `resolve_principal` and BEFORE the
    detached pipeline task is created (contextvars snapshot at task
    creation, so the run inherits the principal even though it outlives
    the request).
  - The MCP auth middleware, alongside the customer contextvar.

WHO DOESN'T (the system lane): cognition, the scans, the workers, the
CLI's local runtime, and the keyless admin inspector wrapper set
nothing, so `get_current_operator()` returns None there — which the
vector backends treat as "unfiltered", the ratified
filter-never-replaces contract (Q2=A; infrastructure/permissions.py).
The fail-closed backend default is the intended NEXT hardening arc,
recorded in docs/SESSION_HANDOFF.md.
"""
from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # typing only — no runtime model import (permissions.py precedent)
    from ..models import Operator

current_operator: "contextvars.ContextVar[Optional[Operator]]" = (
    contextvars.ContextVar("cc_current_operator", default=None)
)


def get_current_operator() -> "Optional[Operator]":
    """The acting operator for this request context, or None (system lane)."""
    return current_operator.get()


def set_current_operator(operator: "Optional[Operator]") -> "contextvars.Token":
    """Pin the acting operator; returns the Token for reset_current_operator."""
    return current_operator.set(operator)


def reset_current_operator(token: "contextvars.Token") -> None:
    """Restore the previous value (door/middleware finally-hygiene)."""
    current_operator.reset(token)
