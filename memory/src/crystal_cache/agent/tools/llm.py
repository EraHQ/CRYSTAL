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
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["role", "content"],
                },
            },
            "model": {
                "type": "string",
                "description": (
                    "Optional model id override. When omitted, the "
                    "customer's model_routing_config.model_id is used."
                ),
            },
            "temperature": {
                "type": "number",
                "description": "Sampling temperature, 0.0-2.0.",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Maximum tokens to generate.",
            },
        },
        "required": ["messages"],
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

    client = await get_upstream_client(customer, store)
    effective_model = model or customer.model_routing_config.model_id

    try:
        response = await client.complete(
            messages=messages,
            model=effective_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.error(
            "llm_invoke.failed",
            customer_id=customer_id,
            model=effective_model,
            error=str(e),
            error_type=type(e).__name__,
        )
        return {
            "error": f"upstream call failed: {e}",
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
            else None
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
