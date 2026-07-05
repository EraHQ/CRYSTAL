"""Ingress layer — §7 of BUILD_PROPOSAL.md.

API gateway (OpenAI-compatible) + request logger. Customer apps point here
instead of directly at their upstream LLM. Every request enters the system
through this layer and every response leaves through it.

NOTE FOR v2 PHASE 2 PORT:
This temporary __init__.py omits APIGateway and RequestLogger exports
because:
  - `gateway.py` and `logger.py` are dead stubs in v1 (tracked as
    Cleanup-CU-4 and Cleanup-CU-5; the actual API surface lives in
    app.py, ported in Phase 6).
  - require_customer depends on MetadataStore which lands in Phase 3.
    It's exported below so Phase 3+ can pick it up immediately.

Phase 6 will revisit this file once the API surface is fully ported.
"""
from .auth import require_customer
from .schema import (
    ChatCompletionRequest,
    ChatMessage,
    CreateCustomerRequest,
    CreateCustomerResponse,
    GetCustomerResponse,
)

__all__ = [
    "require_customer",
    "ChatCompletionRequest",
    "ChatMessage",
    "CreateCustomerRequest",
    "CreateCustomerResponse",
    "GetCustomerResponse",
]
