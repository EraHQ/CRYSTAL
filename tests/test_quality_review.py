"""S11 (2026-07-09): response-quality critique stream."""
import pytest

from crystal_cache.metacognition.quality_review import (
    group_quality_observations,
    list_quality_observations,
)


async def _seed(store, customer, *, role="shadow", obs=None, summary="s"):
    return await store.create_critique(
        customer_id=customer.id,
        critic_role=role,
        critic_model="haiku",
        summary_text=summary,
        observations=obs or [],
    )


async def test_quality_stream_excludes_substrate(store, customer):
    await _seed(store, customer, obs=[
        {"type": "assumption_identified", "text": "assumed X"},
        {"type": "substrate_complaint", "text": "retrieval was bad"},
        {"type": "reasoning_skip", "text": "skipped step"},
    ])
    views = await list_quality_observations(
        store, customer_id=customer.id)
    types = [v.observation_type for v in views]
    assert "assumption_identified" in types
    assert "reasoning_skip" in types
    assert "substrate_complaint" not in types
    # composed context present
    assert views[0].critic_role == "shadow"
    assert views[0].summary_text == "s"


async def test_quality_stream_role_filter_and_limit(store, customer):
    await _seed(store, customer, role="shadow", obs=[
        {"type": "source_contradiction", "text": "a"}])
    await _seed(store, customer, role="agent_self", obs=[
        {"type": "gap_papered_over", "text": "b"}])
    only_shadow = await list_quality_observations(
        store, customer_id=customer.id, critic_role="shadow")
    assert {v.critic_role for v in only_shadow} == {"shadow"}

    limited = await list_quality_observations(
        store, customer_id=customer.id, limit=1)
    assert len(limited) == 1


async def test_quality_grouping_loudest_first(store, customer):
    await _seed(store, customer, obs=[
        {"type": "reasoning_skip", "text": "1"},
        {"type": "reasoning_skip", "text": "2"},
        {"type": "assumption_identified", "text": "3"},
    ])
    groups = await group_quality_observations(
        store, customer_id=customer.id)
    assert groups[0].observation_type == "reasoning_skip"
    assert groups[0].count == 2
    assert groups[0].latest[0].detail.get("text") in ("1", "2")


async def test_malformed_observation_skipped_not_fatal(store, customer):
    await _seed(store, customer, obs=[
        {"no_type_key": True},
        {"type": "tool_output_questionable", "text": "ok"},
    ])
    views = await list_quality_observations(
        store, customer_id=customer.id)
    assert [v.observation_type for v in views] == [
        "tool_output_questionable"]
