"""Upstream LLM client — provider-agnostic interface for calling OpenAI,
Anthropic, and self-hosted OpenAI-compatible endpoints.

Inbound wire format is OpenAI-compatible (industry de-facto standard).
Outbound varies by provider, with translation happening inside each client.

Design:
  UpstreamClient is the abstract shape. Concrete classes:
    - OpenAIClient      — pass-through
    - AnthropicClient   — translates OpenAI → Anthropic Messages API
    - SelfHostedClient  — pass-through to a configurable OpenAI-compatible base_url
                          (vLLM, Ollama, llama.cpp, etc. serving Qwen or any other model)

  get_upstream_client(customer) returns the right one based on the
  customer's ModelRoutingConfig.

This module has NO knowledge of crystals, retrieval, or injection. It is
a pure LLM proxy. Crystal-aware injection happens upstream of this layer
in the request pipeline (future work — stub endpoints today).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import httpx

from ..models import Customer


# -----------------------------------------------------------------------------
# Unified response shape
# -----------------------------------------------------------------------------

@dataclass
class UpstreamResponse:
    """What all upstream clients return, regardless of provider."""
    # OpenAI-compatible response body. The ingress endpoint returns this
    # directly to the caller, so it must match OpenAI's /v1/chat/completions
    # schema (id, object, created, model, choices, usage).
    openai_format: dict[str, Any]
    # Metadata our system tracks independent of the wire format
    latency_ms: int
    provider: str
    model_id: str
    # Raw text of the assistant message, for convenience/logging
    assistant_text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class StreamChunk:
    """One chunk of a streaming response, normalized to OpenAI shape.

    Phase 1.5.1 (May 2026). Each upstream client's `stream()` method
    yields these. The chunks correspond to OpenAI Chat Completions
    streaming events (objects of type "chat.completion.chunk"). For
    Anthropic, AnthropicClient.stream() translates Anthropic's
    event-stream format into this shape on the fly.

    Three things land in any given chunk:

      delta_text       — incremental assistant text. "" on chunks that
                         only carry usage / role / finish_reason.
      finish_reason    — set on the LAST text chunk: "stop", "length",
                         "tool_calls". None elsewhere.
      prompt_tokens /  — set ONLY on the usage-bearing chunk (typically
      completion_tokens  the very last chunk before [DONE]). Both
                         providers emit usage at the end of the
                         stream, not per-chunk. Treat None as "not
                         yet known."

    The `raw_chunk` field carries the OpenAI-shaped dict that the
    SSE generator will JSON-encode and emit downstream. We keep it
    on the dataclass so the generator doesn't have to reconstruct
    it; the upstream client already had to build it to extract the
    delta text and finish_reason.

    Why a dataclass instead of just yielding dicts: type safety on
    the consumer side (the SSE generator in app.py needs to accumulate
    `delta_text` for QueryLog and read `finish_reason` to decide when
    to emit [DONE]) and a single owner for the OpenAI-shape contract.
    The on-the-wire shape is the dict; the dataclass is the
    process-internal representation.
    """
    delta_text: str
    finish_reason: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    raw_chunk: dict[str, Any]


# -----------------------------------------------------------------------------
# Client interface
# -----------------------------------------------------------------------------

class UpstreamClient:
    """Abstract shape. Not an ABC because we never call this directly;
    the routing function returns a concrete instance."""

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> UpstreamResponse:
        raise NotImplementedError

    def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Open a streaming completion.

        Returns an async iterator yielding `StreamChunk`s in the
        order they arrive from upstream. The implementation is an
        async generator (`async def` containing `yield`), so the
        consumer iterates with `async for chunk in client.stream(...)`.

        Lifecycle: each implementation manages its own httpx connection
        inside the generator. When the consumer finishes (either by
        exhausting the iterator or by GC'ing it / breaking out of the
        loop), the generator's `aclose()` runs the cleanup blocks. No
        leak unless the consumer never iterates at all (which would
        also mean the upstream call never started).

        Phase 1.5.1 (May 2026).
        """
        raise NotImplementedError


# -----------------------------------------------------------------------------
# OpenAI — pass-through
# -----------------------------------------------------------------------------

class OpenAIClient(UpstreamClient):
    """Calls api.openai.com/v1/chat/completions. Uses the customer's key."""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1") -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> UpstreamResponse:
        body: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        # Pass any additional known OpenAI params straight through.
        # Tool calling fields (tools / tool_choice / parallel_tool_calls)
        # are forwarded verbatim — OpenAI's wire format is the source of
        # truth for our gateway, no translation needed for OpenAI-routed
        # customers. Phase 1.5.2.
        for k in (
            "top_p", "frequency_penalty", "presence_penalty", "stop", "n",
            "tools", "tool_choice", "parallel_tool_calls",
            "response_format",
        ):
            if k in kwargs and kwargs[k] is not None:
                body[k] = kwargs[k]

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        start = time.monotonic()
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{self._base_url}/chat/completions",
                json=body,
                headers=headers,
            )
            r.raise_for_status()
            payload = r.json()
        latency_ms = int((time.monotonic() - start) * 1000)

        choices = payload.get("choices", [])
        # Coerce None content to "" for assistant_text. OpenAI sets
        # message.content=None when only tool_calls are emitted; we
        # store empty-string for logging to keep assistant_text typed
        # as str. The full payload (including tool_calls) flows back
        # to the customer via openai_format unchanged. Phase 1.5.2.
        assistant_text = (
            (choices[0].get("message", {}).get("content") or "")
            if choices else ""
        )
        usage = payload.get("usage", {}) or {}

        return UpstreamResponse(
            openai_format=payload,
            latency_ms=latency_ms,
            provider="openai",
            model_id=model,
            assistant_text=assistant_text,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion via OpenAI's SSE format.

        OpenAI emits one `data: {...}` line per chunk plus one
        terminal `data: [DONE]`. We yield a StreamChunk per data
        line (DONE is consumed but not yielded). Usage arrives on
        the final chunk before [DONE] when stream_options.include_usage
        is set; we set it unconditionally so QueryLog gets real token
        counts.

        Per-chunk shape (OpenAI):
          {
            "id": "chatcmpl-...",
            "object": "chat.completion.chunk",
            "created": ...,
            "model": "...",
            "choices": [
              {"index": 0,
               "delta": {"role": "assistant", "content": "..."},
               "finish_reason": null | "stop" | "length"}
            ],
            "usage": null OR {"prompt_tokens": ..., ...}
          }

        We pass the raw chunk dict back via StreamChunk.raw_chunk so
        the SSE generator in app.py can JSON-encode it directly
        without reconstructing the shape. Saves a round-trip and
        keeps the OpenAI-format contract in one place.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            # Ask OpenAI to emit usage in the final chunk. Without this,
            # the stream finishes with usage=None and we'd have to leave
            # token counts unset on the QueryLog row. With it, the last
            # data event before [DONE] carries usage.
            "stream_options": {"include_usage": True},
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        # Same forwarding allowlist as complete(), including tool calling
        # fields. Phase 1.5.2.
        for k in (
            "top_p", "frequency_penalty", "presence_penalty", "stop", "n",
            "tools", "tool_choice", "parallel_tool_calls",
            "response_format",
        ):
            if k in kwargs and kwargs[k] is not None:
                body[k] = kwargs[k]

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        # The double `async with` pattern below is the canonical httpx
        # streaming shape. AsyncClient owns the pool; .stream(...)
        # opens a single response. Both close on exit. Both run when
        # the generator is GC'd or aclose()'d, so consumer disconnect
        # cleans up cleanly.
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json=body,
                headers=headers,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    # OpenAI uses `data: <payload>` lines separated by
                    # blank lines. aiter_lines strips trailing newlines
                    # but leaves leading whitespace; sanity-strip first.
                    line = line.strip()
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        # Comments (lines starting with ":") and any
                        # other framing get dropped. We only emit
                        # data events.
                        continue
                    payload_str = line[len("data:"):].strip()
                    if payload_str == "[DONE]":
                        # Stream over. The for-loop will terminate;
                        # the with-blocks unwind; the generator
                        # returns.
                        return
                    try:
                        chunk = json.loads(payload_str)
                    except json.JSONDecodeError:
                        # Malformed chunk — skip rather than crash the
                        # whole stream. Production OpenAI doesn't emit
                        # these; bug-bait if a self-hosted server does.
                        continue

                    # Extract delta_text + finish_reason from choices[0].
                    # OpenAI sometimes emits chunks with empty choices
                    # (the usage-only final chunk); handle gracefully.
                    choices = chunk.get("choices", []) or []
                    delta_text = ""
                    finish_reason: Optional[str] = None
                    if choices:
                        first = choices[0] or {}
                        delta = first.get("delta", {}) or {}
                        # delta.content can be None on the first chunk
                        # (which carries role=assistant only) or on
                        # any chunk that's just signaling a tool_call /
                        # finish. Treat None as empty.
                        delta_text = delta.get("content") or ""
                        finish_reason = first.get("finish_reason")

                    usage = chunk.get("usage")
                    prompt_tokens = (
                        int(usage["prompt_tokens"])
                        if usage and "prompt_tokens" in usage
                        else None
                    )
                    completion_tokens = (
                        int(usage["completion_tokens"])
                        if usage and "completion_tokens" in usage
                        else None
                    )

                    yield StreamChunk(
                        delta_text=delta_text,
                        finish_reason=finish_reason,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        raw_chunk=chunk,
                    )


# -----------------------------------------------------------------------------
# Anthropic — translate inbound OpenAI shape to Messages API, outbound back.
# -----------------------------------------------------------------------------

# OpenAI stop_reason → Anthropic finish_reason. Shared across complete()
# and stream() so the streaming and non-streaming paths can't drift.
_ANTHROPIC_STOP_TO_OPENAI_FINISH = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}


def _translate_messages_to_anthropic(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Translate OpenAI message list to Anthropic shape.

    Returns (system_text, anthropic_messages). System text is the
    concatenation of all system messages joined with double-newlines
    (Anthropic puts these on a top-level `system` param, not in the
    message array). anthropic_messages is the role-alternating array
    of user/assistant turns Anthropic expects.

    Translation rules (Phase 1.5.2):

      role:"system"     → collected into top-level system text.
      role:"user"       → user message, content passes through.
      role:"assistant"  → assistant message. If the OpenAI message has
                          `tool_calls`, build content blocks: optional
                          text block (if `content` is non-empty) plus
                          one `tool_use` block per tool_call entry.
                          Otherwise pass content through.
      role:"tool"       → AGGREGATED into ONE user message. Consecutive
                          tool messages collapse into a single user
                          message whose content is a list of
                          `tool_result` blocks (one per consumed
                          tool message). See Q1 in the scope doc.

    The aggregator walks the message list with an index variable so it
    can consume runs of tool messages in one pass.
    """
    system_parts: list[str] = []
    anthropic_messages: list[dict[str, Any]] = []

    i = 0
    while i < len(messages):
        m = messages[i]
        role = m.get("role")
        content = m.get("content", "")

        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                # OpenAI sometimes allows structured content; flatten text parts.
                system_parts.append(
                    "".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                )
            i += 1
            continue

        if role == "tool":
            # AGGREGATE: consume this and all consecutive tool messages
            # into one Anthropic user message with tool_result blocks.
            tool_result_blocks: list[dict[str, Any]] = []
            while i < len(messages) and messages[i].get("role") == "tool":
                tm = messages[i]
                tool_call_id = tm.get("tool_call_id", "") or ""
                tm_content = tm.get("content", "")
                # Anthropic's tool_result.content accepts a string or a
                # list of content blocks. We pass strings through as-is
                # and lists through as-is; OpenAI customers most often
                # send strings.
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": tm_content,
                    }
                )
                i += 1
            anthropic_messages.append(
                {"role": "user", "content": tool_result_blocks}
            )
            continue

        if role == "assistant":
            tool_calls = m.get("tool_calls")
            if tool_calls:
                # Assistant turn carries tool_calls → build content
                # blocks: optional text block + one tool_use block per
                # call. Anthropic's tool_use shape:
                #   {"type": "tool_use",
                #    "id": <call_id>,
                #    "name": <function_name>,
                #    "input": <parsed_arguments_object>}
                blocks: list[dict[str, Any]] = []
                if isinstance(content, str) and content:
                    blocks.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    # Already-structured content; pass text parts through.
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            blocks.append(
                                {"type": "text", "text": part.get("text", "")}
                            )
                for tc in tool_calls:
                    fn = tc.get("function", {}) or {}
                    raw_args = fn.get("arguments", "") or ""
                    # OpenAI's `function.arguments` is a JSON-encoded
                    # string; Anthropic's `input` is a parsed object.
                    if isinstance(raw_args, str):
                        try:
                            parsed_args = json.loads(raw_args) if raw_args else {}
                        except json.JSONDecodeError:
                            # Malformed args from a misbehaving caller.
                            # Forward as a string under a single key so
                            # Anthropic still gets something parseable;
                            # better than a 400 from the upstream.
                            parsed_args = {"_raw": raw_args}
                    else:
                        parsed_args = raw_args
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": parsed_args,
                        }
                    )
                anthropic_messages.append({"role": "assistant", "content": blocks})
            else:
                anthropic_messages.append({"role": "assistant", "content": content})
            i += 1
            continue

        if role == "user":
            anthropic_messages.append({"role": "user", "content": content})
            i += 1
            continue

        # Unknown role — skip silently. tool roles are handled above;
        # anything else (function, future roles) just gets dropped.
        i += 1

    return ("\n\n".join(system_parts), anthropic_messages)


def _translate_tools_to_anthropic(
    tools: Optional[list[dict[str, Any]]],
) -> Optional[list[dict[str, Any]]]:
    """Translate OpenAI tools list to Anthropic tools list.

    OpenAI:
      [{"type": "function",
        "function": {"name": ..., "description": ..., "parameters": {...}}}]

    Anthropic:
      [{"name": ..., "description": ..., "input_schema": {...}}]

    Only `type:"function"` tools are translated; OpenAI's other tool
    types (e.g. `type:"file_search"` on Assistants API) don't apply
    to chat completions and are skipped.
    """
    if not tools:
        return None
    out: list[dict[str, Any]] = []
    for t in tools:
        if t.get("type") != "function":
            continue
        fn = t.get("function", {}) or {}
        translated: dict[str, Any] = {"name": fn.get("name", "")}
        if "description" in fn:
            translated["description"] = fn["description"]
        # OpenAI's `parameters` → Anthropic's `input_schema`.
        if "parameters" in fn:
            translated["input_schema"] = fn["parameters"]
        out.append(translated)
    return out or None


def _translate_tool_choice_to_anthropic(
    tool_choice: Optional[Any],
) -> Optional[dict[str, Any]]:
    """Translate OpenAI tool_choice to Anthropic tool_choice.

    OpenAI → Anthropic mapping:
      "none"      → {"type": "none"}    (Anthropic 2024-10+ supports this)
      "auto"      → {"type": "auto"}
      "required"  → {"type": "any"}     (semantics: must call SOME tool)
      {"type":"function", "function":{"name":<n>}}
                  → {"type":"tool", "name":<n>}

    Returns None when tool_choice is None or unrecognized; the
    Anthropic API defaults to auto in that case, which matches OpenAI's
    default behavior.
    """
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice == "none":
            return {"type": "none"}
        if tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "required":
            return {"type": "any"}
        return None
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            fn = tool_choice.get("function", {}) or {}
            name = fn.get("name")
            if name:
                return {"type": "tool", "name": name}
    return None


class AnthropicClient(UpstreamClient):
    """Calls api.anthropic.com/v1/messages, translating to/from OpenAI shape."""

    ANTHROPIC_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com/v1",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> UpstreamResponse:
        # Translate inbound OpenAI shape to Anthropic shape via the
        # shared helper. Phase 1.5.2 uses this for both complete() and
        # stream() so the two paths can't drift on tool-message handling.
        system_text, anthropic_messages = _translate_messages_to_anthropic(messages)

        body: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            # Anthropic requires max_tokens; default to a safe number.
            "max_tokens": max_tokens if max_tokens is not None else 4096,
        }
        if system_text:
            body["system"] = system_text
        if temperature is not None:
            body["temperature"] = temperature
        if "top_p" in kwargs and kwargs["top_p"] is not None:
            body["top_p"] = kwargs["top_p"]
        if "stop" in kwargs and kwargs["stop"] is not None:
            # OpenAI "stop" → Anthropic "stop_sequences"
            stop = kwargs["stop"]
            body["stop_sequences"] = stop if isinstance(stop, list) else [stop]

        # Phase 1.5.2: tool calling. Translate inbound OpenAI tools and
        # tool_choice to their Anthropic equivalents. parallel_tool_calls
        # has no Anthropic equivalent (Anthropic always allows parallel)
        # and is silently dropped on this provider.
        anthropic_tools = _translate_tools_to_anthropic(kwargs.get("tools"))
        if anthropic_tools:
            body["tools"] = anthropic_tools
        anthropic_tool_choice = _translate_tool_choice_to_anthropic(
            kwargs.get("tool_choice")
        )
        if anthropic_tool_choice is not None:
            body["tool_choice"] = anthropic_tool_choice

        # Phase 1.5.4: JSON mode. Anthropic doesn't have a direct
        # `response_format` parameter. For {"type": "json_object"},
        # we append a system-prompt hint asking the model to respond
        # in JSON. For {"type": "json_schema", ...}, we also append
        # the schema. This is best-effort — Anthropic's compliance
        # with JSON-mode directives is model-dependent.
        response_format = kwargs.get("response_format")

        # Extended thinking (Anthropic-native).
        # {"type": "enabled", "budget_tokens": 10000}
        thinking = kwargs.get("thinking")
        if isinstance(thinking, dict):
            body["thinking"] = thinking
            # Extended thinking requires temperature=1 on Anthropic
            body.pop("temperature", None)

        if isinstance(response_format, dict):
            fmt_type = response_format.get("type", "")
            if fmt_type == "json_object":
                json_hint = "\n\nYou must respond with valid JSON only. No other text."
                if system_text:
                    body["system"] = system_text + json_hint
                else:
                    body["system"] = json_hint.strip()
            elif fmt_type == "json_schema":
                schema = response_format.get("json_schema", {})
                schema_str = json.dumps(schema, indent=2)
                json_hint = (
                    f"\n\nYou must respond with valid JSON matching this schema:\n"
                    f"{schema_str}\nNo other text."
                )
                if system_text:
                    body["system"] = system_text + json_hint
                else:
                    body["system"] = json_hint.strip()

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": self.ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        start = time.monotonic()
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{self._base_url}/messages",
                json=body,
                headers=headers,
            )
            r.raise_for_status()
            anthropic_payload = r.json()
        latency_ms = int((time.monotonic() - start) * 1000)

        # Translate Anthropic response → OpenAI shape.
        # Anthropic returns:
        #   { id, type: "message", role: "assistant",
        #     content: [ {type:"text", text:"..."},
        #                {type:"tool_use", id, name, input}, ... ],
        #     model, stop_reason, usage: {input_tokens, output_tokens} }
        #
        # Phase 1.5.2: scan content blocks. Text blocks accumulate into
        # `assistant_text` (existing behavior). Tool_use blocks build an
        # OpenAI-shaped `tool_calls` array on the assistant message.
        # Both can co-exist: Anthropic frequently emits a thinking-style
        # text block followed by one or more tool_use blocks.
        text_parts: list[str] = []
        tool_calls_out: list[dict[str, Any]] = []
        for block in anthropic_payload.get("content", []) or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", "") or "")
            elif btype == "tool_use":
                # Anthropic `input` is a parsed object; OpenAI's
                # `function.arguments` is a JSON-encoded string. Round-
                # trip through json.dumps so SDK consumers can json.loads
                # it the way they expect.
                input_obj = block.get("input", {}) or {}
                try:
                    args_str = json.dumps(input_obj)
                except (TypeError, ValueError):
                    args_str = "{}"
                tool_calls_out.append(
                    {
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": args_str,
                        },
                    }
                )
            # Other block types (e.g. future thinking blocks) are
            # silently dropped — they don't have OpenAI equivalents.
        assistant_text = "".join(text_parts)
        usage = anthropic_payload.get("usage", {}) or {}
        prompt_tokens = int(usage.get("input_tokens", 0))
        completion_tokens = int(usage.get("output_tokens", 0))

        finish_reason = _ANTHROPIC_STOP_TO_OPENAI_FINISH.get(
            anthropic_payload.get("stop_reason") or "", "stop"
        )

        # Build the OpenAI-shape assistant message. `content` is None
        # (not empty string) when only tool_calls were emitted — matches
        # OpenAI's wire format. `tool_calls` is omitted when empty so
        # SDK consumers don't see a stray empty-array field.
        assistant_message: dict[str, Any] = {"role": "assistant"}
        if tool_calls_out:
            assistant_message["content"] = assistant_text or None
            assistant_message["tool_calls"] = tool_calls_out
        else:
            assistant_message["content"] = assistant_text

        openai_format = {
            "id": anthropic_payload.get("id", ""),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": anthropic_payload.get("model", model),
            "choices": [
                {
                    "index": 0,
                    "message": assistant_message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

        return UpstreamResponse(
            openai_format=openai_format,
            latency_ms=latency_ms,
            provider="anthropic",
            model_id=model,
            assistant_text=assistant_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion via Anthropic's event-stream, translating
        each event to OpenAI chunk shape on the fly.

        Anthropic's stream format is `event: <name>\\ndata: <json>\\n\\n`
        with named event types:

          message_start         — carries message id, model, and
                                  usage.input_tokens.
          content_block_start   — marks the beginning of a content
                                  block (text, tool_use, etc.). For
                                  tool_use blocks we open a tool-call
                                  state record and emit the OPENING
                                  OpenAI tool_calls chunk (with id +
                                  name + empty arguments).
          content_block_delta   — carries delta.text for text blocks,
                                  or delta.partial_json for tool_use
                                  blocks (input_json_delta). Each text
                                  delta becomes one OpenAI content
                                  chunk; each input_json_delta becomes
                                  one OpenAI tool_calls.arguments
                                  chunk.
          content_block_stop    — end of one content block. Nothing
                                  emitted; the message_delta signals
                                  finish.
          message_delta         — carries stop_reason and the FINAL
                                  usage.output_tokens.
          message_stop          — end of message. We exit; the SSE
                                  generator in app.py emits [DONE].
          ping                  — keepalive. Ignored.
          error                 — anthropic-side error. We let httpx
                                  surface the HTTP error; mid-stream
                                  errors raise via raise_for_status
                                  on the underlying response.

        Tool-call streaming protocol (Phase 1.5.2, Q2 = stream-through):

          - When a tool_use content block starts (content_block_start
            with content_block.type == "tool_use"), we capture
            (anthropic_index → openai_tool_index) so subsequent
            input_json_delta events for that block can be tagged with
            the right OpenAI tool_calls index. The FIRST chunk for
            that index carries id + type:"function" + function.name
            + function.arguments="" per OpenAI's wire format.
          - Each input_json_delta emits a chunk where ONLY
            function.arguments is set to the partial-json delta. id
            and name are not repeated (matches OpenAI streaming
            convention; SDK consumers concatenate on the client side).
        """
        # Translate inbound OpenAI shape via the shared helper. Same
        # path as complete() so tool messages, tool_calls on assistant
        # messages, and tool_results all translate identically.
        system_text, anthropic_messages = _translate_messages_to_anthropic(messages)

        body: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens if max_tokens is not None else 4096,
            "stream": True,
        }
        if system_text:
            body["system"] = system_text
        if temperature is not None:
            body["temperature"] = temperature
        if "top_p" in kwargs and kwargs["top_p"] is not None:
            body["top_p"] = kwargs["top_p"]
        if "stop" in kwargs and kwargs["stop"] is not None:
            stop = kwargs["stop"]
            body["stop_sequences"] = stop if isinstance(stop, list) else [stop]

        # Phase 1.5.2: tool calling fields. parallel_tool_calls dropped
        # for Anthropic (no equivalent).
        anthropic_tools = _translate_tools_to_anthropic(kwargs.get("tools"))
        if anthropic_tools:
            body["tools"] = anthropic_tools
        anthropic_tool_choice = _translate_tool_choice_to_anthropic(
            kwargs.get("tool_choice")
        )
        if anthropic_tool_choice is not None:
            body["tool_choice"] = anthropic_tool_choice

        # Phase 1.5.4: JSON mode (same logic as complete()).
        response_format = kwargs.get("response_format")
        if isinstance(response_format, dict):
            fmt_type = response_format.get("type", "")
            if fmt_type == "json_object":
                json_hint = "\n\nYou must respond with valid JSON only. No other text."
                if system_text:
                    body["system"] = system_text + json_hint
                else:
                    body["system"] = json_hint.strip()
            elif fmt_type == "json_schema":
                schema = response_format.get("json_schema", {})
                schema_str = json.dumps(schema, indent=2)
                json_hint = (
                    f"\n\nYou must respond with valid JSON matching this schema:\n"
                    f"{schema_str}\nNo other text."
                )
                if system_text:
                    body["system"] = system_text + json_hint
                else:
                    body["system"] = json_hint.strip()

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": self.ANTHROPIC_VERSION,
            "content-type": "application/json",
            "accept": "text/event-stream",
        }

        # Stream-state we accumulate across events to fill OpenAI-shaped
        # chunks. We need the message id from message_start to put on
        # every emitted chunk (OpenAI keeps id consistent across the
        # stream); we also need input_tokens to emit alongside
        # output_tokens on the final usage chunk.
        stream_id: str = ""
        prompt_tokens: int = 0

        # Phase 1.5.2 tool-call state.
        # Anthropic's content blocks have their own indices (the `index`
        # field on content_block_* events) which are NOT the same as
        # OpenAI's tool_calls array index. Anthropic indices count text
        # AND tool_use blocks together (text=0, tool_use=1, tool_use=2...);
        # OpenAI tool_calls indices only count tool_use blocks (first
        # tool_use=0, second tool_use=1, ...). We map between them.
        anthropic_idx_to_openai_tool_idx: dict[int, int] = {}
        # Counter for the next OpenAI tool_calls index to assign.
        next_openai_tool_idx: int = 0

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/messages",
                json=body,
                headers=headers,
            ) as response:
                response.raise_for_status()

                # Anthropic emits `event: <name>` then `data: <json>`
                # then a blank line. aiter_lines yields each line
                # individually, so we accumulate per-event state across
                # consecutive lines.
                current_event: Optional[str] = None
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        # Blank line = event boundary. Reset.
                        current_event = None
                        continue
                    if line.startswith("event:"):
                        current_event = line[len("event:"):].strip()
                        continue
                    if not line.startswith("data:"):
                        continue
                    payload_str = line[len("data:"):].strip()
                    try:
                        event_data = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue

                    # Per-event handling. Each branch either yields a
                    # StreamChunk or updates accumulator state.
                    event_type = current_event or event_data.get("type")

                    if event_type == "message_start":
                        # Capture stream id and prompt_tokens. Don't
                        # yield — there's no delta yet, just a header.
                        msg = event_data.get("message", {}) or {}
                        stream_id = msg.get("id", "") or stream_id
                        usage = msg.get("usage", {}) or {}
                        prompt_tokens = int(usage.get("input_tokens", 0))
                        continue

                    if event_type == "content_block_start":
                        # If this is a tool_use block, record its index
                        # mapping AND emit the OPENING tool_calls chunk
                        # (carries id + name + empty arguments).
                        block = event_data.get("content_block", {}) or {}
                        if block.get("type") != "tool_use":
                            # Text block start — nothing to emit; the
                            # text deltas come via content_block_delta.
                            continue
                        anthropic_index = event_data.get("index")
                        if anthropic_index is None:
                            continue
                        openai_tool_idx = next_openai_tool_idx
                        next_openai_tool_idx += 1
                        anthropic_idx_to_openai_tool_idx[anthropic_index] = (
                            openai_tool_idx
                        )
                        # Opening chunk: OpenAI sends id + type + name
                        # + arguments="" on the first delta for a given
                        # tool_calls index. Subsequent chunks for the
                        # same index only carry function.arguments
                        # increments.
                        chunk_dict = {
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": openai_tool_idx,
                                                "id": block.get("id", ""),
                                                "type": "function",
                                                "function": {
                                                    "name": block.get("name", ""),
                                                    "arguments": "",
                                                },
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield StreamChunk(
                            delta_text="",
                            finish_reason=None,
                            prompt_tokens=None,
                            completion_tokens=None,
                            raw_chunk=chunk_dict,
                        )
                        continue

                    if event_type == "content_block_delta":
                        # Two flavors: text_delta (assistant text) and
                        # input_json_delta (tool-call argument JSON).
                        delta = event_data.get("delta", {}) or {}
                        delta_type = delta.get("type")

                        if delta_type == "text_delta":
                            text = delta.get("text", "") or ""
                            chunk_dict = {
                                "id": stream_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": text},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield StreamChunk(
                                delta_text=text,
                                finish_reason=None,
                                prompt_tokens=None,
                                completion_tokens=None,
                                raw_chunk=chunk_dict,
                            )
                            continue

                        if delta_type == "input_json_delta":
                            # Phase 1.5.2: tool-call argument streaming.
                            # Look up the OpenAI tool_calls index for
                            # this anthropic block; emit a chunk with
                            # ONLY function.arguments set to the partial
                            # json delta. id/name are not repeated
                            # (already sent on the opening chunk).
                            anthropic_index = event_data.get("index")
                            openai_tool_idx = anthropic_idx_to_openai_tool_idx.get(
                                anthropic_index
                            )
                            if openai_tool_idx is None:
                                # Delta arrived for a block we never
                                # saw a content_block_start for. Skip
                                # rather than emit a bogus chunk.
                                continue
                            partial_json = delta.get("partial_json", "") or ""
                            chunk_dict = {
                                "id": stream_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "tool_calls": [
                                                {
                                                    "index": openai_tool_idx,
                                                    "function": {
                                                        "arguments": partial_json,
                                                    },
                                                }
                                            ]
                                        },
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield StreamChunk(
                                delta_text="",
                                finish_reason=None,
                                prompt_tokens=None,
                                completion_tokens=None,
                                raw_chunk=chunk_dict,
                            )
                            continue

                        # Other delta types (future block-typed deltas)
                        # are ignored.
                        continue

                    if event_type == "content_block_stop":
                        # Block ended. We don't emit anything — the
                        # message_delta event carries finish_reason.
                        continue

                    if event_type == "message_delta":
                        # Final-ish event carrying stop_reason +
                        # output_tokens. Emit the terminal chunk with
                        # finish_reason set and usage populated.
                        delta = event_data.get("delta", {}) or {}
                        usage = event_data.get("usage", {}) or {}
                        completion_tokens = int(usage.get("output_tokens", 0))
                        finish_reason = _ANTHROPIC_STOP_TO_OPENAI_FINISH.get(
                            delta.get("stop_reason") or "", "stop"
                        )
                        chunk_dict = {
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": finish_reason,
                                }
                            ],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": prompt_tokens + completion_tokens,
                            },
                        }
                        yield StreamChunk(
                            delta_text="",
                            finish_reason=finish_reason,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            raw_chunk=chunk_dict,
                        )
                        continue

                    if event_type == "message_stop":
                        # End of stream. SSE generator emits [DONE]
                        # downstream; we just exit.
                        return

                    # Unrecognized events (ping, error) are ignored.
                    # HTTP-level errors surface via raise_for_status
                    # before we get here; mid-stream Anthropic-side
                    # error events are out of scope for v0.


# -----------------------------------------------------------------------------
# Self-hosted — points at any OpenAI-compatible endpoint (vLLM, Ollama, etc.)
# -----------------------------------------------------------------------------

class SelfHostedClient(OpenAIClient):
    """Same wire format as OpenAI; different base URL. Used for vLLM serving
    Qwen, Ollama, llama.cpp server, or any other OpenAI-compatible LLM host.

    The customer's ModelRoutingConfig supplies the base_url. api_key is
    optional (some self-hosted setups don't require auth); if blank, we
    still send an Authorization header but with a placeholder — most
    servers ignore it.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
    ) -> None:
        super().__init__(api_key=api_key or "sk-local", base_url=base_url)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> UpstreamResponse:
        resp = await super().complete(messages, model, temperature, max_tokens, **kwargs)
        # Override provider tag to distinguish from public OpenAI for telemetry
        return UpstreamResponse(
            openai_format=resp.openai_format,
            latency_ms=resp.latency_ms,
            provider="self_hosted",
            model_id=resp.model_id,
            assistant_text=resp.assistant_text,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
        )


# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------

async def get_upstream_client(customer: Customer, store) -> UpstreamClient:
    """Return the right client for a customer's routing config.

    P4 (2026-07-10): Key B is enc:v2 — encrypted under the TENANT's DEK
    with AAD binding — and is decrypted here, at the single point of
    use, via the tenant surface (hence async + store). Anything
    non-empty that isn't enc:v2 is refused: the cutover nulled the one
    orphaned v1 row; no legacy-decrypt path exists by design. Empty
    refs pass through (self_hosted endpoints may not need a key).
    """
    from ..infrastructure.token_crypto import is_v2_encrypted

    # Managed inference (E4, Accounts Phase B 2026-07-06): the PLATFORM's
    # provider credentials serve this tenant — Key B is ignored entirely
    # (a managed customer may not have one). The model stays the
    # customer's model_id; only the credential source changes. The proxy
    # door has already enforced the monthly spend cap before we get here.
    if getattr(customer, "inference_mode", "byok") == "managed":
        return _get_managed_client()

    cfg = customer.model_routing_config
    provider = cfg.provider.lower()
    raw_ref = cfg.api_key_ref or ""
    if not raw_ref:
        api_key = raw_ref
    elif is_v2_encrypted(raw_ref):
        api_key = await store.decrypt_tenant_secret(
            customer.id, "key_b", raw_ref
        )
    else:
        raise RuntimeError(
            f"customer {customer.id} has a stored upstream key that is "
            "not enc:v2 — refusing to use it. Re-enter the key in "
            "Settings to store it under the tenant envelope."
        )

    if provider == "openai":
        return OpenAIClient(api_key=api_key)
    if provider == "anthropic":
        return AnthropicClient(api_key=api_key)
    if provider == "self_hosted":
        if not cfg.base_url:
            raise ValueError(
                "self_hosted provider requires ModelRoutingConfig.base_url to be set"
            )
        return SelfHostedClient(api_key=api_key, base_url=cfg.base_url)

    raise ValueError(f"Unknown provider: {cfg.provider!r}")


def _get_managed_client() -> UpstreamClient:
    """The platform-keyed client for managed-inference customers.

    Provider from CC_MANAGED_INFERENCE_PROVIDER (anthropic at launch);
    key from the platform's own settings — never from the customer row.
    Fails LOUD when the platform key is missing: a managed customer on a
    box with no provider credential is an operator configuration error,
    not something to paper over.
    """
    from ..config import get_settings

    settings = get_settings()
    provider = (
        getattr(settings, "managed_inference_provider", "anthropic") or "anthropic"
    ).lower()
    if provider == "anthropic":
        key = (getattr(settings, "anthropic_api_key", "") or "").strip()
        if not key:
            raise RuntimeError(
                "managed inference requires CC_ANTHROPIC_API_KEY (the "
                "platform key) — none is configured"
            )
        return AnthropicClient(api_key=key)
    if provider == "openai":
        key = (getattr(settings, "llm_api_key", "") or "").strip()
        if not key:
            raise RuntimeError(
                "managed inference with provider=openai requires "
                "CC_LLM_API_KEY (the platform key) — none is configured"
            )
        return OpenAIClient(api_key=key)
    raise RuntimeError(
        f"Unknown managed inference provider: {provider!r} "
        "(CC_MANAGED_INFERENCE_PROVIDER)"
    )
