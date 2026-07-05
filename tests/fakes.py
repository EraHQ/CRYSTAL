"""Hand-rolled fakes for Phase 8 smoke tests.

Per P0.31: no unittest.mock.MagicMock. Each fake is a small concrete
class that mirrors the surface the agent / cognition code actually
uses. Failure modes are debuggable because the fake's source is
readable.

FakeAnthropic mirrors `anthropic.Anthropic`:
  client.messages.create(model, max_tokens, system, messages, tools)
    -> response with .content, .stop_reason, .usage

The fake supports scripting: tests `script_response(...)` ahead of
each expected model call. Calls beyond the script length raise
AssertionError so tests catch unexpected loop iterations.

The fake DEEP-COPIES the messages list at record time so tests can
inspect per-iteration message-list snapshots. The agent loop mutates
its working messages list in place across iterations; without the
copy, every recorded call would point at the same (final) list state.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Content block fakes
# ---------------------------------------------------------------------------

@dataclass
class FakeTextBlock:
    """Fake equivalent of anthropic.types.TextBlock."""
    text: str
    type: str = "text"


@dataclass
class FakeToolUseBlock:
    """Fake equivalent of anthropic.types.ToolUseBlock."""
    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)
    type: str = "tool_use"


@dataclass
class FakeUsage:
    """Fake equivalent of anthropic.types.Usage.

    Includes the prompt-caching fields (default 0) so cache-aware tests can
    script them; they mirror the real Usage object the agent reads under C1.
    """
    input_tokens: int = 100
    output_tokens: int = 50
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class FakeResponse:
    """Fake equivalent of anthropic.types.Message.

    Carries content (list of FakeTextBlock | FakeToolUseBlock),
    stop_reason, and usage. The Agent class reads these via
    `response.content`, `response.stop_reason`, `response.usage`.
    """
    content: list[Any]
    stop_reason: str = "end_turn"
    usage: FakeUsage = field(default_factory=FakeUsage)
    model: str = "claude-sonnet-4-5-20250929"
    id: str = "msg_fake_001"
    role: str = "assistant"


# ---------------------------------------------------------------------------
# FakeAnthropic — the top-level client fake
# ---------------------------------------------------------------------------

class _FakeMessages:
    """The `.messages` sub-client. Exposes `.create(...)`."""

    def __init__(self, parent: "FakeAnthropic") -> None:
        self._parent = parent

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: Optional[str] = None,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        **_extra: Any,  # tolerate temperature/top_p/etc. real callers pass
    ) -> FakeResponse:
        """Synchronous call matching the Anthropic SDK signature.

        Records the call args on the parent for assertion, then pops
        the next scripted response. Raises AssertionError if no more
        scripted responses are available — catches infinite loops.

        IMPORTANT: messages is DEEP-COPIED before recording. The
        agent loop mutates its working list across iterations; without
        the copy, all recorded calls would share a reference to the
        same (eventually final) list state. Tools is also deep-copied
        for consistency, though it doesn't mutate in practice.
        """
        self._parent.calls.append({
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools) if tools is not None else None,
        })
        if not self._parent._scripted:
            raise AssertionError(
                f"FakeAnthropic.messages.create called "
                f"{len(self._parent.calls)} time(s), but no more "
                f"scripted responses remain. The test either "
                f"under-scripted the expected loop iterations or the "
                f"agent looped more than expected. Last call args: "
                f"model={model!r}, message_count={len(messages)}, "
                f"tools={len(tools) if tools else 0}."
            )
        return self._parent._scripted.pop(0)


class NotReadyLLM:
    """A seam stand-in whose is_ready() is False.

    Inject via set_llm_client to force the no-provider path deterministically,
    regardless of any real API key in the developer's environment.
    """

    def is_ready(self) -> bool:
        return False


class FakeAnthropic:
    """Hand-rolled fake of the anthropic.Anthropic client.

    Per P0.31, no unittest.mock. The fake exposes only the surface
    `agent/agent.py::Agent._call_model` actually uses: the
    `.messages.create(...)` method returning an object with
    `.content`, `.stop_reason`, `.usage`.

    Usage:
        anth = FakeAnthropic()
        anth.script_text("Hello!")
        # ... call Agent.run(...) ...
        assert len(anth.calls) == 1

        anth.script_tool_use("knowledge_search", {"query": "test"}, "tu_001")
        anth.script_text("Based on the lookup: ...")
        # ... agent loop runs twice ...
    """

    def __init__(self) -> None:
        self.messages = _FakeMessages(self)
        self.calls: list[dict[str, Any]] = []
        self._scripted: list[FakeResponse] = []

    # -----------------------------------------------------------------
    # Scripting helpers
    # -----------------------------------------------------------------

    def script_text(
        self,
        text: str,
        *,
        stop_reason: str = "end_turn",
        input_tokens: int = 100,
        output_tokens: int = 50,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> None:
        """Script the next model call to return a text-only response."""
        self._scripted.append(FakeResponse(
            content=[FakeTextBlock(text=text)],
            stop_reason=stop_reason,
            usage=FakeUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
            ),
        ))

    def script_tool_use(
        self,
        name: str,
        input_dict: dict[str, Any],
        tool_use_id: str = "tu_fake_001",
        *,
        preamble_text: str = "",
        input_tokens: int = 150,
        output_tokens: int = 80,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> None:
        """Script the next model call to emit a tool_use block.

        Optionally prefixed by a short text preamble (matches how
        Claude often emits "I'll look that up:" before the tool call).
        stop_reason is fixed to "tool_use" because that's what
        Anthropic emits when there's a pending tool call.
        """
        content: list[Any] = []
        if preamble_text:
            content.append(FakeTextBlock(text=preamble_text))
        content.append(FakeToolUseBlock(
            id=tool_use_id,
            name=name,
            input=input_dict,
        ))
        self._scripted.append(FakeResponse(
            content=content,
            stop_reason="tool_use",
            usage=FakeUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
            ),
        ))

    def script_multi_tool_use(
        self,
        calls: list[tuple[str, dict[str, Any], str]],
        *,
        preamble_text: str = "",
        input_tokens: int = 150,
        output_tokens: int = 80,
    ) -> None:
        """Script a single model response that emits multiple tool_use blocks.

        Mirrors Anthropic's parallel tool calling — one response, N
        tool_use blocks. The Agent loop dispatches all of them and
        appends one user message with N tool_result blocks.

        Args:
            calls: list of (tool_name, input_dict, tool_use_id) tuples.
        """
        content: list[Any] = []
        if preamble_text:
            content.append(FakeTextBlock(text=preamble_text))
        for name, input_dict, tool_use_id in calls:
            content.append(FakeToolUseBlock(
                id=tool_use_id,
                name=name,
                input=input_dict,
            ))
        self._scripted.append(FakeResponse(
            content=content,
            stop_reason="tool_use",
            usage=FakeUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        ))

    # -----------------------------------------------------------------
    # Assertion helpers
    # -----------------------------------------------------------------

    def complete(
        self,
        *,
        system: Optional[str] = None,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float = 0.0,
        tier: str = "small",
        model: Optional[str] = None,
        json_schema: Optional[dict[str, Any]] = None,
    ) -> str:
        """Mirror LLMClient.complete for migrated text-call sites.

        Records the call (so assert_call_count / last_call span both
        .messages.create and .complete) and returns the scripted text.
        """
        self.calls.append({
            "model": model or tier,
            "max_tokens": max_tokens,
            "system": system,
            "messages": copy.deepcopy(messages),
            "tools": None,
            "json_schema": json_schema,
        })
        if not self._scripted:
            raise AssertionError(
                f"FakeAnthropic.complete called {len(self.calls)} time(s), "
                f"but no more scripted responses remain."
            )
        resp = self._scripted.pop(0)
        parts = [
            getattr(b, "text", "")
            for b in (resp.content or [])
            if getattr(b, "type", None) == "text"
        ]
        return "".join(parts).strip()

    def complete_detailed(
        self,
        *,
        system: Optional[str] = None,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float = 0.0,
        tier: str = "small",
        model: Optional[str] = None,
        json_schema: Optional[dict[str, Any]] = None,
    ):
        """Mirror LLMClient.complete_detailed: record via complete() and wrap
        the scripted text in an LLMResult (the fake reports no token usage).
        """
        from crystal_cache.llm.client import LLMResult

        text = self.complete(
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tier=tier,
            model=model,
            json_schema=json_schema,
        )
        return LLMResult(text=text, model=model or tier)

    def is_ready(self) -> bool:
        """Seam-readiness for use as an injected LLM client (set_llm_client).

        Always ready in tests so an injected FakeAnthropic drives the
        self-critique path rather than skipping it.
        """
        return True

    @property
    def provider(self) -> str:
        """Mirror LLMClient.provider. The fake plays the anthropic path,
        so the Agent applies its cache-mark decoration — matching what the
        caching tests assert."""
        return "anthropic"

    def complete_messages(
        self,
        *,
        system: Any = None,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        max_tokens: int,
        model: Optional[str] = None,
        tier: str = "large",
    ) -> Any:
        """Mirror LLMClient.complete_messages (anthropic path).

        Delegates to .messages.create so recording, the scripted-response
        queue, and the ran-out-of-scripts loop guard all behave identically
        to the pre-seam fake.
        """
        return self.messages.create(
            model=model or tier,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools,
        )

    def assert_called_once(self) -> dict[str, Any]:
        """Assert exactly one call and return its args."""
        assert len(self.calls) == 1, (
            f"expected exactly 1 model call, got {len(self.calls)}"
        )
        return self.calls[0]

    def assert_call_count(self, n: int) -> None:
        """Assert exactly n model calls were made."""
        assert len(self.calls) == n, (
            f"expected {n} model calls, got {len(self.calls)}: "
            f"{[c['messages'][-1] for c in self.calls]}"
        )

    def last_call(self) -> dict[str, Any]:
        """Return the args of the most recent call."""
        assert self.calls, "no calls recorded"
        return self.calls[-1]
