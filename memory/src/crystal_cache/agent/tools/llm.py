"""LLM tool — `llm_invoke`.

Per §4.1: wraps the customer's model routing configuration as a tool.
This is what the old proxy did, exposed as a tool. The agent calls
this when it wants to produce a freeform completion — typically to
compose its final answer to the user from retrieved context.

CONTEXT (D-A10 + §6.5.3):
- llm_invoke is agent-only. Cognition has its own LLM primitives
  (analyze, synthesize, format) that are structured differently:
  they take prior step outputs as input, follow the role's
  information barrier, and use a fixed cognition system prompt.
  Letting cognition workers also call llm_invoke would create two
  paths to the same model with different barriers and prompts.

CUSTOMER MODEL ROUTING:
- The customer's `model_routing_config` (Pydantic model on Customer)
  carries provider + model_id + api_key_ref. Phase 5's
  `infrastructure/upstream_client.py` (ported in Wave 7C) translates
  these into provider-specific clients. We use `get_upstream_client`
  to obtain the right client per the customer record.
- The agent passes through `messages`, `temperature`, `max_tokens`,
  `model` (overrides the customer default), and `extra` (anything
  the agent wants to forward — tools, response_format, etc.).
"""
from __future__ import annotations

from typing import Any, Optional

import structlog

from ..tool_registry import register_tool
from .retrievers import _get_state

logger = structlog.get_logger(__name__)


@register_tool(
    name="llm_invoke",
    description=(
        "Send a prompt to the customer's configured upstream LLM "
        "and return the completion. Use this to compose your final "
        "answer once you have the context you need from retrievers "
        "and memory tools. The customer's model_routing_config "
        "determines which provider / model is invoked. Returns the "
        "completion text and usage metadata."
    ),
    contexts={"agent"},
    parameters_schema={
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "description": (
                    "OpenAI-compatible message list "
                    "[{role: 'system'|'user'|'assistant', content: str}, ...]."
                ),
                "maxItems": 50,
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "enum": ["system", "user", "assistant"]},
                        "content": {"type": "string", "maxLength": 100000},
                    },
                    "required": ["role", "content"],
                    "additionalProperties": False,
                },
            },
            "model": {
                "type": "string",
                "maxLength": 128,
                "description": (
                    "Optional model id override. When omitted, the "
                    "customer's model_routing_config.model_id is used. On a "
                    "managed plan only the plan's models are accepted."
                ),
            },
            "temperature": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Sampling temperature, 0.0-1.0.",
            },
            "max_tokens": {
                "type": "integer",
                "minimum": 1,
                "maximum": 128000,
                "description": (
                    "Maximum tokens to generate. On a managed plan the plan's "
                    "output ceiling applies, and is the default when omitted."
                ),
            },
        },
        "required": ["messages"],
        "additionalProperties": False,
    },
    returns_description=(
        "{'assistant_text': str, 'prompt_tokens': int | None, "
        "'completion_tokens': int | None, 'model': str, "
        "'finish_reason': str | None}"
    ),
)
async def llm_invoke(
    customer_id: str,
    messages: list[dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> dict[str, Any]:
    from ...execution.upstream_client import get_upstream_client

    state = _get_state()
    store = state["store"]

    # Fetch the customer record to get its model_routing_config.
    # The agent loop passes customer_id; we resolve the full
    # Customer record here. This is a single DB hit per llm_invoke
    # call — acceptable on the cost path because llm_invoke is the
    # heaviest tool already (LLM call dominates).
    customer = await store.get_customer_by_id(customer_id)
    if customer is None:
        return {
            "error": f"customer {customer_id!r} not found",
            "assistant_text": "",
        }

    # Lockdown PR-2 (Q52=A, AUDIT_LAUNCH_VERIFY B1-3): the tool used to
    # forward `model` and `max_tokens` untouched, so on a managed plan
    # (the PLATFORM's key) a prompt-injected agent could pick any model
    # with any output cap. The same two guards the agent turn applies
    # (endpoints/agent.py) apply here, and the arguments are bounded.
    from ...control.admission import (
        PlanWallError,
        clamp_max_tokens,
        enforce_managed_model,
        resolve_tier,
    )

    if not isinstance(messages, list) or not messages or len(messages) > 50:
        return {
            "error": "messages must be a list of 1 to 50 messages",
            "code": "invalid_argument", "assistant_text": "",
        }
    for m in messages:
        if (not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant")
                or not isinstance(m.get("content"), str) or len(m["content"]) > 100_000):
            return {
                "error": "each message needs a role (system, user or assistant) and a string content of at most 100,000 characters",
                "code": "invalid_argument", "assistant_text": "",
            }
    if temperature is not None and not (0.0 <= float(temperature) <= 1.0):
        return {"error": "temperature must be between 0 and 1",
                "code": "invalid_argument", "assistant_text": ""}
    if max_tokens is not None and (int(max_tokens) < 1 or int(max_tokens) > 128_000):
        return {"error": "max_tokens must be between 1 and 128,000",
                "code": "invalid_argument", "assistant_text": ""}
    if model is not None and (not isinstance(model, str) or len(model) > 128):
        return {"error": "model must be a model id of at most 128 characters",
                "code": "invalid_argument", "assistant_text": ""}

    effective_model = model or customer.model_routing_config.model_id
    try:
        enforce_managed_model(customer, effective_model)
    except PlanWallError as wall:
        return {"error": wall.message, "code": wall.code, "assistant_text": ""}
    managed = getattr(customer, "inference_mode", "byok") == "managed"
    if managed:
        ceiling = resolve_tier(getattr(customer, "subscription_tier", None)).max_output_tokens
        max_tokens = clamp_max_tokens(customer, int(max_tokens) if max_tokens else ceiling)

    client = await get_upstream_client(customer, store)

    try:
        response = await client.complete(
            messages=messages,
            model=effective_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        # S3 + B1 (2026-09-30): this is the site where a malformed BYOK
        # key used to come back inside the SDK's error text, straight
        # into the agent's tool result.
        from ...hygiene import safe_error
        ref, message = safe_error(
            "llm_invoke.failed", e,
            user_message="The upstream model call failed.",
            customer_id=customer_id, model=effective_model,
        )
        return {
            "error": message,
            "ref": ref,
            "assistant_text": "",
        }

    # Gate B (2026-07-16): the agent's upstream spend stamps the ledger
    # like the proxy lane — billing='managed' only when Crystal's own key
    # paid for the call (BYOK upstream spend is the customer's).
    from ...cost.emit import record_model_call

    await record_model_call(
        customer_id=customer_id,
        model=effective_model,
        origin="agent_llm_invoke",
        input_tokens=response.prompt_tokens,
        output_tokens=response.completion_tokens,
        billing=(
            "managed"
            if getattr(customer, "inference_mode", "byok") == "managed"
            else "byok"  # v108: explicit, so the spend gates can exclude it
        ),
        store=store,
    )

    # UpstreamResponse carries assistant_text, prompt_tokens,
    # completion_tokens, openai_format. Surface the high-signal fields
    # as a flat dict so adapters can re-emit them per protocol.
    return {
        "assistant_text": response.assistant_text,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "model": effective_model,
        "finish_reason": (
            response.openai_format.get("choices", [{}])[0].get("finish_reason")
            if response.openai_format
            else None
        ),
    }
