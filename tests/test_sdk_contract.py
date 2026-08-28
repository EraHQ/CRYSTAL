"""LLM seam <-> installed Anthropic SDK contract (2026-08-28, L0-Q1=B).

Why this file exists: accounts-v84 shipped with anthropic 1.2.0 resolved
unpinned. SDK v1 (2026-08-20) REMOVED `temperature` / `top_p` / `top_k`
from `messages.create()` / `messages.stream()` — a Python TypeError, not a
400 — and every lane that pins 0.0 by default (extraction, cognition,
self-critique) raised on first use. The suite was green the whole time:
every fake accepted `**kwargs`, so no test ever compared what the seam
SENDS with what the SDK TAKES. The LongMemEval compose smoke found it.

These tests close that gap structurally. Each one drives a real seam path
through a capturing fake, then BINDS the captured kwargs against the
signature of the SDK that is actually installed:

    inspect.signature(anthropic.Anthropic(...).messages.create).bind(**sent)

A kwarg the installed SDK does not accept is a TypeError here — the same
error prod would raise — with no network call (client construction is
offline). The suite must run against the SDK prod runs; the version pin
below keeps the local venv honest about that.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import anthropic
import pytest

from crystal_cache.llm.client import (
    _ANTHROPIC_NO_SAMPLING,
    _SUPPORTED_SDK_MAJOR,
    LLMClient,
    _anthropic_sampling_kwargs,
    check_installed_sdk,
    get_llm_client,
    reset_llm_client,
)

_ROOT = Path(__file__).parent.parent
_PYPROJECT = _ROOT / "pyproject.toml"

# One offline client; signatures are all we read from it.
_REAL = anthropic.Anthropic(api_key="contract-test-no-network")
_CREATE_SIG = inspect.signature(_REAL.messages.create)
_STREAM_SIG = inspect.signature(_REAL.messages.stream)


def _bind(sig: inspect.Signature, sent: dict) -> None:
    """Raise TypeError if `sent` is not a valid call of the real method."""
    sig.bind(**sent)


# ---------------------------------------------------------------------------
# Capturing fakes (deliberately tolerant — the binding is the assertion)
# ---------------------------------------------------------------------------

class _Capture:
    def __init__(self, resp):
        self._resp = resp
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._resp


class _CaptureStream:
    class _Ctx:
        def __init__(self, final):
            self.text_stream = iter(["x"])
            self._final = final

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            return self._final

    def __init__(self, final):
        self._final = final
        self.last_kwargs = None

    def stream(self, **kwargs):
        self.last_kwargs = kwargs
        return self._Ctx(self._final)


def _resp():
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=1, output_tokens=1,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        ),
    )


def _client(*, stream: bool = False) -> LLMClient:
    c = LLMClient(
        provider="anthropic", api_key="k", base_url=None,
        model_small="claude-haiku-4-5-20251001",
        model_large="claude-sonnet-4-6",
        model_frontier="claude-opus-4-8",
    )
    c._anthropic_client = SimpleNamespace(
        messages=_CaptureStream(_resp()) if stream else _Capture(_resp())
    )
    return c


_MSGS = [{"role": "user", "content": "q"}]
_TOOLS = [{"name": "t", "description": "d",
           "input_schema": {"type": "object", "properties": {}}}]


# ---------------------------------------------------------------------------
# 1. The installed SDK is the major the seam is written for, and the
#    package metadata says so (the mcp<2 rule applied to anthropic).
# ---------------------------------------------------------------------------

def test_installed_sdk_is_major_1():
    major = int(anthropic.__version__.split(".")[0])
    assert major == 1 == _SUPPORTED_SDK_MAJOR, (
        f"anthropic {anthropic.__version__} installed; the seam speaks 1.x "
        "(extra_body sampling). Run: pip install --upgrade 'anthropic>=1,<2'"
    )
    assert check_installed_sdk() == anthropic.__version__


# ---------------------------------------------------------------------------
# 1b. The boot-time assertion (L0-Q1b=C): a wrong major fails LOUD, at
#     boot (get_llm_client) and at first SDK use (_get_anthropic), instead
#     of the per-call TypeErrors v84 shipped with. Both directions.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["0.111.0", "2.0.0", "garbage"])
def test_check_installed_sdk_rejects_other_majors(monkeypatch, bad):
    monkeypatch.setattr(anthropic, "__version__", bad)
    with pytest.raises(RuntimeError, match="anthropic SDK"):
        check_installed_sdk()


def test_first_sdk_use_fails_loud_on_wrong_major(monkeypatch):
    monkeypatch.setattr(anthropic, "__version__", "0.111.0")
    c = LLMClient(provider="anthropic", api_key="k", base_url=None,
                  model_small=None, model_large=None, model_frontier=None)
    with pytest.raises(RuntimeError, match="anthropic SDK"):
        c.complete(system="s", messages=_MSGS, max_tokens=8, tier="small")


def test_boot_singleton_fails_loud_on_wrong_major(monkeypatch):
    """get_llm_client() is what app startup touches first."""
    reset_llm_client()
    try:
        monkeypatch.setattr(anthropic, "__version__", "0.111.0")
        with pytest.raises(RuntimeError, match="anthropic SDK"):
            get_llm_client()
    finally:
        reset_llm_client()


def test_pyproject_caps_boot_critical_sdks():
    text = _PYPROJECT.read_text(encoding="utf-8")
    specs = re.findall(r'"(anthropic(?:\[vertex\])?[^"]*)"', text)
    assert specs, "no anthropic requirement found in pyproject.toml"
    for spec in specs:
        assert "<2" in spec, f"unbounded major on a boot-critical SDK: {spec}"
        assert ">=1" in spec, f"seam speaks 1.x but spec allows 0.x: {spec}"
    mcp = re.findall(r'"(mcp[^"]*)"', text)
    assert mcp and all("<2" in s for s in mcp), mcp


# ---------------------------------------------------------------------------
# 2. Every seam path binds against the REAL signatures.
# ---------------------------------------------------------------------------

def test_agent_lane_default_binds():
    c = _client()
    c.complete_messages(system="s", messages=_MSGS, tools=_TOOLS,
                        max_tokens=64, model="claude-haiku-4-5-20251001")
    sent = c._anthropic_client.messages.last_kwargs
    _bind(_CREATE_SIG, sent)
    assert "temperature" not in sent and "extra_body" not in sent


def test_agent_lane_temperature_on_sampling_model_binds():
    """The harness path: temperature 0 on Haiku 4.5 — the exact call that
    raised in prod under 1.2.0."""
    c = _client()
    c.complete_messages(system="s", messages=_MSGS, max_tokens=64,
                        model="claude-haiku-4-5-20251001", temperature=0.0)
    sent = c._anthropic_client.messages.last_kwargs
    _bind(_CREATE_SIG, sent)
    assert sent["extra_body"] == {"temperature": 0.0}


def test_agent_lane_temperature_on_adaptive_model_binds():
    c = _client()
    c.complete_messages(system="s", messages=_MSGS, max_tokens=64,
                        model="claude-fable-5", temperature=0.0)
    sent = c._anthropic_client.messages.last_kwargs
    _bind(_CREATE_SIG, sent)
    assert "extra_body" not in sent


def test_agent_lane_thinking_binds():
    c = _client()
    c.complete_messages(system="s", messages=_MSGS, max_tokens=4096,
                        model="claude-sonnet-4-6",
                        thinking={"type": "enabled", "budget_tokens": 1024})
    sent = c._anthropic_client.messages.last_kwargs
    _bind(_CREATE_SIG, sent)
    assert sent["thinking"]["budget_tokens"] == 1024


def test_stream_lane_binds_default_and_temperature():
    c = _client(stream=True)
    c.stream_messages(system="s", messages=_MSGS, tools=_TOOLS,
                      max_tokens=64, model="claude-sonnet-4-6")
    _bind(_STREAM_SIG, c._anthropic_client.messages.last_kwargs)

    c = _client(stream=True)
    c.stream_messages(system="s", messages=_MSGS, max_tokens=64,
                      model="claude-sonnet-4-6", temperature=0.0)
    sent = c._anthropic_client.messages.last_kwargs
    _bind(_STREAM_SIG, sent)
    assert sent["extra_body"] == {"temperature": 0.0}


def test_single_shot_lane_default_binds():
    """complete() defaults to temperature=0.0 on the small tier — the
    extraction / cognition / critique lane that broke in prod."""
    c = _client()
    c.complete(system="s", messages=_MSGS, max_tokens=64, tier="small")
    sent = c._anthropic_client.messages.last_kwargs
    _bind(_CREATE_SIG, sent)
    assert sent["model"] == "claude-haiku-4-5-20251001"
    assert sent["extra_body"] == {"temperature": 0.0}
    assert "temperature" not in sent


def test_single_shot_lane_json_schema_binds():
    c = _client()
    c.complete_detailed(system="s", messages=_MSGS, max_tokens=64,
                        tier="large", json_schema={"type": "object"})
    sent = c._anthropic_client.messages.last_kwargs
    _bind(_CREATE_SIG, sent)
    assert sent["output_config"]["format"]["type"] == "json_schema"


def test_single_shot_lane_adaptive_model_binds():
    c = _client()
    c.complete(system="s", messages=_MSGS, max_tokens=64, tier="frontier")
    sent = c._anthropic_client.messages.last_kwargs
    _bind(_CREATE_SIG, sent)
    assert sent["model"] in _ANTHROPIC_NO_SAMPLING
    assert "extra_body" not in sent


# ---------------------------------------------------------------------------
# 3. The helper itself, and the negative control that proves the binding
#    is a real check (a literal `temperature` kwarg must fail to bind).
# ---------------------------------------------------------------------------

def test_sampling_helper_table():
    assert _anthropic_sampling_kwargs("claude-haiku-4-5-20251001", None) == {}
    assert _anthropic_sampling_kwargs("claude-haiku-4-5-20251001", 0.0) == {
        "extra_body": {"temperature": 0.0}}
    assert _anthropic_sampling_kwargs("claude-fable-5", 0.0) == {}


def test_negative_control_temperature_kwarg_does_not_bind():
    with pytest.raises(TypeError):
        _bind(_CREATE_SIG, {"model": "m", "max_tokens": 1, "messages": _MSGS,
                            "temperature": 0.0})
