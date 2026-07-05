"""CRYS session continuity — agent_conversations store surface
(ConversationExtensionsMixin, P5).

Per-scope conversation persistence so context survives exit/relaunch. Covers
the P5 A1 contract: upsert + get round-trip (transcript preserved), upsert
overwrites the same scope (one rolling conversation per project), get-missing,
delete, tenant scoping, and list.
"""
from __future__ import annotations

_KEY = "C:/Users/example/projects/aethermoor"

_TRANSCRIPT = [
    {"role": "user", "content": "Append the remaining lines to the game file"},
    {"role": "assistant", "content": "Done — appended to game_combat.js"},
]


async def test_upsert_and_get(store, customer):
    saved = await store.upsert_conversation(
        customer.id,
        conversation_key=_KEY,
        transcript=_TRANSCRIPT,
        turn_count=1,
        last_summary="appended lines to game_combat.js",
        mode="coding",
        meta={"last_files": ["game_combat.js"]},
    )
    assert saved["id"].startswith("conv_")
    assert saved["mode"] == "coding"

    got = await store.get_conversation(customer.id, conversation_key=_KEY)
    assert got is not None
    assert got["transcript"] == _TRANSCRIPT  # list-of-dicts preserved
    assert got["turn_count"] == 1
    assert got["last_summary"] == "appended lines to game_combat.js"
    assert got["meta"] == {"last_files": ["game_combat.js"]}


async def test_upsert_overwrites_same_scope(store, customer):
    first = await store.upsert_conversation(
        customer.id, conversation_key=_KEY,
        transcript=[{"role": "user", "content": "one"}], turn_count=1,
    )
    second = await store.upsert_conversation(
        customer.id, conversation_key=_KEY,
        transcript=[
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
        ],
        turn_count=2, last_summary="turn two",
    )
    # Same row reused (one rolling conversation per scope), latest content wins.
    assert first["id"] == second["id"]
    got = await store.get_conversation(customer.id, conversation_key=_KEY)
    assert got["turn_count"] == 2
    assert len(got["transcript"]) == 2
    assert got["last_summary"] == "turn two"

    listed = await store.list_conversations(customer.id)
    assert len(listed) == 1


async def test_get_missing_returns_none(store, customer):
    assert await store.get_conversation(
        customer.id, conversation_key="nope"
    ) is None


async def test_delete_conversation(store, customer):
    await store.upsert_conversation(
        customer.id, conversation_key=_KEY, transcript=_TRANSCRIPT, turn_count=1,
    )
    assert await store.delete_conversation(customer.id, conversation_key=_KEY) is True
    assert await store.get_conversation(customer.id, conversation_key=_KEY) is None
    # Deleting again is a no-op False (the /reset path is safe to repeat).
    assert await store.delete_conversation(customer.id, conversation_key=_KEY) is False


async def test_conversations_are_tenant_scoped(store, customer):
    other = await store.create_customer(
        provider="anthropic", model_id="claude-sonnet-4-5-20250929",
        api_key_ref="sk-test-other",
    )
    await store.upsert_conversation(
        customer.id, conversation_key=_KEY,
        transcript=[{"role": "user", "content": "mine"}], turn_count=1,
    )
    # Same conversation_key under a different customer is a distinct row.
    await store.upsert_conversation(
        other.id, conversation_key=_KEY,
        transcript=[{"role": "user", "content": "theirs"}], turn_count=1,
    )
    mine = await store.get_conversation(customer.id, conversation_key=_KEY)
    theirs = await store.get_conversation(other.id, conversation_key=_KEY)
    assert mine["transcript"][0]["content"] == "mine"
    assert theirs["transcript"][0]["content"] == "theirs"
    assert mine["id"] != theirs["id"]


async def test_list_conversations_count_and_limit(store, customer):
    for i in range(3):
        await store.upsert_conversation(
            customer.id, conversation_key=f"/proj/{i}",
            transcript=[{"role": "user", "content": f"c{i}"}], turn_count=1,
        )
    assert len(await store.list_conversations(customer.id)) == 3
    assert len(await store.list_conversations(customer.id, limit=2)) == 2
