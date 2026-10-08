"""Q42=B (2026-10-08): Scale's 32,768 output cap is real for API callers.

The agent already streams the model call when the request streams. A
NON-streaming request above the SDK's non-streaming ceiling (21,333
output tokens, the SDK's own rule) used to be refused before it started,
so a Scale customer could not use the cap they pay for from the API.
Now such a call is served through the seam's stream_messages with the
deltas discarded; the caller still gets one complete response.

Built on a bare Agent instance with exactly the attributes _call_model
reads (verified by listing self.* in that method); the fake LLM exposes
the real seam's two methods with its real keyword surface.
"""
import pytest

from crystal_cache.agent.agent import Agent
from crystal_cache.llm.client import NON_STREAMING_MAX_TOKENS


class _Seam:
    def __init__(self):
        self.calls = []

    def complete_messages(self, *, system, messages, tools=None, max_tokens, model=None, **kw):
        self.calls.append(("complete", max_tokens))
        return {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"}

    def stream_messages(self, *, system, messages, tools=None, max_tokens, model=None,
                        tier="large", temperature=None, thinking=None, on_text=None):
        self.calls.append(("stream", max_tokens, on_text is not None))
        if on_text:
            on_text("do")
            on_text("ne")
        return {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"}


def _bare_agent(seam, max_tokens):
    a = Agent.__new__(Agent)
    a.llm = seam
    a.max_tokens = max_tokens
    a.model = "claude-sonnet-5"
    a.temperature = None
    a.thinking = None
    a.stream_tokens = False   # a non-streaming request
    a.emit = None
    a.system_tail = None
    return a


@pytest.mark.asyncio
async def test_non_streaming_request_above_the_ceiling_streams_internally():
    seam = _Seam()
    agent = _bare_agent(seam, 32_768)
    out = await agent._call_model(system="s", messages=[{"role": "user", "content": "q"}], tools=[])
    assert out["stop_reason"] == "end_turn"
    assert seam.calls == [("stream", 32_768, True)]


@pytest.mark.asyncio
async def test_non_streaming_request_within_the_ceiling_is_unchanged():
    seam = _Seam()
    agent = _bare_agent(seam, 16_384)
    await agent._call_model(system="s", messages=[{"role": "user", "content": "q"}], tools=[])
    assert seam.calls == [("complete", 16_384)]


def test_the_ceiling_is_the_sdk_rule():
    # 60*60*max_tokens/128_000 seconds > 600 s  <=>  max_tokens > 21,333
    assert NON_STREAMING_MAX_TOKENS == 21_333
    assert 60 * 60 * NON_STREAMING_MAX_TOKENS / 128_000 < 600
    assert 60 * 60 * (NON_STREAMING_MAX_TOKENS + 1) / 128_000 > 600
