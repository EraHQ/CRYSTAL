"""C6 — per-conversation model selection (cost + parity, 2026-06-17).

One model-selection change with three parts, all covered here:

  1. Store: agent_conversations.model get/set
     (ConversationExtensionsMixin.get_conversation_model /
     set_conversation_model) — focused (never clobbers transcript/meta),
     last-writer-wins, tenant-scoped, thin-row insert.
  2. Endpoint precedence: resolve_conversation_model — client-sent model wins
     AND is persisted; a no-model request reuses the saved one; None when
     neither applies (the Agent then fills the house default); a None
     sequence_id can't have a sticky model; fail-safe on store errors.
  3. House default: Agent honors CC_AGENT_MODEL (settings.agent_model) when no
     explicit model is passed, and an explicit model still wins over it.
  4. Depth synthesis runs through the provider-neutral seam: DepthRouter's
     _slm_synthesize pre-digests via complete_detailed at tier small (the
     sub-step model knob is gone; the small tier governs which model runs).

R14 note: these assertions are verified by `pytest`; they describe expected
behavior and have not yet been run at authoring time.
"""
from __future__ import annotations

from crystal_cache import config
from crystal_cache.agent import Agent
from crystal_cache.agent.agent import DEFAULT_MODEL
from crystal_cache.endpoints.agent import resolve_conversation_model
from crystal_cache.llm import reset_llm_client, set_llm_client
from crystal_cache.retrieval.v3_depth import DepthRouter


# ---------------------------------------------------------------------------
# 1. Store — get_conversation_model / set_conversation_model
# ---------------------------------------------------------------------------

async def test_set_then_get_model(store, customer):
    await store.set_conversation_model(
        customer.id, conversation_key="k1", model="claude-opus-4-7",
    )
    assert await store.get_conversation_model(
        customer.id, conversation_key="k1"
    ) == "claude-opus-4-7"
    # A thin row was created for the scope (web-chat common case): empty
    # transcript, general mode.
    row = await store.get_conversation(customer.id, conversation_key="k1")
    assert row is not None
    assert row["mode"] == "general"
    assert row["transcript"] == []


async def test_get_missing_model_returns_none(store, customer):
    assert await store.get_conversation_model(
        customer.id, conversation_key="nope"
    ) is None


async def test_set_model_overwrites(store, customer):
    await store.set_conversation_model(
        customer.id, conversation_key="k2", model="first",
    )
    await store.set_conversation_model(
        customer.id, conversation_key="k2", model="second",
    )
    assert await store.get_conversation_model(
        customer.id, conversation_key="k2"
    ) == "second"
    # Still one row for the scope (last-writer-wins, not a new row).
    listed = await store.list_conversations(customer.id)
    assert len([c for c in listed if c["conversation_key"] == "k2"]) == 1


async def test_set_model_preserves_transcript(store, customer):
    """Setting only the model on an EXISTING conversation must not touch its
    transcript/meta — the reason model is a typed column, not a `meta` key."""
    transcript = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    await store.upsert_conversation(
        customer.id, conversation_key="k3",
        transcript=transcript, turn_count=1,
        last_summary="said hello", meta={"last_files": ["a.py"]},
    )
    await store.set_conversation_model(
        customer.id, conversation_key="k3", model="claude-opus-4-7",
    )
    # Model set...
    assert await store.get_conversation_model(
        customer.id, conversation_key="k3"
    ) == "claude-opus-4-7"
    # ...and the conversation's own state is untouched.
    row = await store.get_conversation(customer.id, conversation_key="k3")
    assert row["transcript"] == transcript
    assert row["turn_count"] == 1
    assert row["last_summary"] == "said hello"
    assert row["meta"] == {"last_files": ["a.py"]}


async def test_model_is_tenant_scoped(store, customer):
    other = await store.create_customer(
        provider="anthropic", model_id="claude-sonnet-4-5-20250929",
        api_key_ref="sk-other-model",
    )
    await store.set_conversation_model(
        customer.id, conversation_key="shared", model="mine",
    )
    await store.set_conversation_model(
        other.id, conversation_key="shared", model="theirs",
    )
    assert await store.get_conversation_model(
        customer.id, conversation_key="shared"
    ) == "mine"
    assert await store.get_conversation_model(
        other.id, conversation_key="shared"
    ) == "theirs"


# ---------------------------------------------------------------------------
# 2. Endpoint precedence — resolve_conversation_model
# ---------------------------------------------------------------------------

async def test_resolve_client_model_wins_and_persists(store, customer):
    out = await resolve_conversation_model(
        store=store, customer_id=customer.id,
        sequence_id="thread-1", requested_model="claude-opus-4-7",
    )
    assert out == "claude-opus-4-7"
    # The client's choice stuck for the conversation.
    assert await store.get_conversation_model(
        customer.id, conversation_key="thread-1"
    ) == "claude-opus-4-7"


async def test_resolve_falls_back_to_saved(store, customer):
    await store.set_conversation_model(
        customer.id, conversation_key="thread-2",
        model="claude-haiku-4-5-20251001",
    )
    out = await resolve_conversation_model(
        store=store, customer_id=customer.id,
        sequence_id="thread-2", requested_model=None,
    )
    assert out == "claude-haiku-4-5-20251001"


async def test_resolve_client_overwrites_saved(store, customer):
    await store.set_conversation_model(
        customer.id, conversation_key="t3", model="old-model",
    )
    out = await resolve_conversation_model(
        store=store, customer_id=customer.id,
        sequence_id="t3", requested_model="new-model",
    )
    assert out == "new-model"
    assert await store.get_conversation_model(
        customer.id, conversation_key="t3"
    ) == "new-model"


async def test_resolve_none_when_no_client_and_no_saved(store, customer):
    out = await resolve_conversation_model(
        store=store, customer_id=customer.id,
        sequence_id="t4", requested_model=None,
    )
    assert out is None


async def test_resolve_no_sequence_id_returns_requested_unchanged(store, customer):
    # No conversation scope: nothing to save against, return the request as-is
    # (the Agent fills the house default when it's None).
    out = await resolve_conversation_model(
        store=store, customer_id=customer.id,
        sequence_id=None, requested_model="claude-x",
    )
    assert out == "claude-x"
    out2 = await resolve_conversation_model(
        store=store, customer_id=customer.id,
        sequence_id=None, requested_model=None,
    )
    assert out2 is None
    # Nothing was persisted (no scope to persist under).
    assert await store.list_conversations(customer.id) == []


async def test_resolve_failsafe_on_store_error():
    """A store whose set/get raise must never break resolution."""
    class _BoomStore:
        async def set_conversation_model(self, *a, **k):
            raise RuntimeError("db down")

        async def get_conversation_model(self, *a, **k):
            raise RuntimeError("db down")

    boom = _BoomStore()
    # client model + set raises -> still returns the requested model.
    out = await resolve_conversation_model(
        store=boom, customer_id="cus_x", sequence_id="t", requested_model="m",
    )
    assert out == "m"
    # no client model + get raises -> None (falls through to the house default).
    out2 = await resolve_conversation_model(
        store=boom, customer_id="cus_x", sequence_id="t", requested_model=None,
    )
    assert out2 is None


# ---------------------------------------------------------------------------
# 3. House default — Agent honors CC_AGENT_MODEL
# ---------------------------------------------------------------------------

async def test_agent_uses_builtin_default_when_model_unset(
    customer, tool_state, fake_anthropic, monkeypatch,
):
    monkeypatch.setattr(config.settings, "agent_model", "")
    agent = Agent(
        customer=customer, llm=fake_anthropic, tool_state=tool_state,
    )
    assert agent.model == DEFAULT_MODEL


async def test_agent_uses_house_default_when_set(
    customer, tool_state, fake_anthropic, monkeypatch,
):
    monkeypatch.setattr(config.settings, "agent_model", "claude-opus-4-7")
    agent = Agent(
        customer=customer, llm=fake_anthropic, tool_state=tool_state,
    )
    assert agent.model == "claude-opus-4-7"


async def test_explicit_model_beats_house_default(
    customer, tool_state, fake_anthropic, monkeypatch,
):
    monkeypatch.setattr(config.settings, "agent_model", "claude-opus-4-7")
    agent = Agent(
        customer=customer, llm=fake_anthropic, tool_state=tool_state,
        model="claude-sonnet-4-5-20250929",
    )
    assert agent.model == "claude-sonnet-4-5-20250929"


# ---------------------------------------------------------------------------
# 4. Depth synthesis runs through the provider-neutral seam
# ---------------------------------------------------------------------------


async def test_depth_synthesis_runs_through_seam(store, vector_index, fake_anthropic):
    """_slm_synthesize pre-digests raw facts via the seam (tier=small) and
    returns its text. The sub-step model knob is gone; the small tier governs
    which model runs."""
    fake_anthropic.script_text("synthesized summary")
    set_llm_client(fake_anthropic)
    try:
        router = DepthRouter(vector_index=vector_index, metadata_store=store)
        out = await router._slm_synthesize(
            "some raw context", "how does X relate to Y?"
        )
    finally:
        reset_llm_client()
    assert out == "synthesized summary"
