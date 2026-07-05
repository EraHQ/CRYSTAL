"""Bundle test for finalize_agent_turn — the shared post-turn layer.

`finalize_agent_turn` is the single function both CRYS surfaces (the agent
endpoint via run_agent_messages, and the coding agent via cli.py/background.py)
call after `Agent.run`, so a turn's universal signals can't drift between
lenses. Its three constituents are covered elsewhere — the cost row in
test_agent_cost, citations/credit/coverage-gap in test_agent_citations, the MCR
trace + self-critique in test_phase9a_mcr_emitter. These lock the BUNDLE: one
call records the cost-ledger row, grounds + records citations, and emits the
MCR trace; returns the documented {cost, cost_micro_usd, mcr} shape; threads
`origin` through to the cost ledger (so coding spend is attributable); and
stays fail-safe when a constituent step errors.

The CRYS wiring itself (that cli.py / background.py call this at the
right point) is not unit-tested here — the crystal_code package is outside the
`tests/` suite — and is verified end-to-end by the showcase driving the real
front door.

R14 note: verified by pytest; describes expected behavior, not yet run at
authoring time.
"""
from __future__ import annotations

from crystal_cache import config
from crystal_cache.agent.turn_finalize import finalize_agent_turn

# Patch grounding at its SOURCE module (ground_agent_citations imports it
# locally at call time, so the patch resolves through finalize). Same target
# test_agent_citations uses.
_GROUNDING_PATH = (
    "crystal_cache.retrieval.citation_grounding.ground_sources_against_answer"
)

# A clean, empty self-critique the scripted Haiku call returns — exercises the
# MCR path without depending on parsing nuance (covered in test_phase9a).
_CLEAN_CRITIQUE = '{"observations": [], "action_items": [], "summary_text": "clean"}'

_LONG_ANSWER = (
    "This is a substantive multi-sentence answer that comfortably exceeds the "
    "uncited-gap minimum so the bundle exercises the full grounding path."
)


def _result_surfacing(crystal_id: str = "cry_A", *, run_id: str = "run_1") -> dict:
    """An Agent.run()-shaped result that surfaced one crystal and has tokens."""
    return {
        "id": run_id,
        "model": "claude-sonnet-4-5-20250929",  # priced in DEFAULT_PRICE_TABLE
        "final_text": _LONG_ANSWER,
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "tool_calls": [
            {
                "tool_name": "knowledge_search",
                "output": {
                    "matched_crystal_ids": [crystal_id],
                    "matched_fact_ids": ["f1"],
                },
            },
        ],
    }


def _patch_grounded(monkeypatch, *, grounded: bool = True) -> None:
    async def _fake(encoder, answer_text, sources, *, threshold=0.25):
        return [
            {
                "source": s,
                "claim_span": "",
                "grounding_score": 0.9 if grounded else 0.05,
                "grounded": grounded,
            }
            for s, _ in sources
        ]
    monkeypatch.setattr(_GROUNDING_PATH, _fake)


def _enable(monkeypatch) -> None:
    monkeypatch.setattr(config.settings, "enable_cost_accounting", True)
    monkeypatch.setattr(config.settings, "enable_citations", True)
    monkeypatch.setattr(config.settings, "enable_marketplace_metering", False)


async def test_bundle_records_cost_citations_and_mcr(
    store, customer, fake_anthropic, monkeypatch
):
    _enable(monkeypatch)
    _patch_grounded(monkeypatch, grounded=True)
    fake_anthropic.script_text(_CLEAN_CRITIQUE)  # the MCR self-critique call

    finalized = await finalize_agent_turn(
        store=store,
        encoder=object(),  # grounding is patched; the encoder is unused
        customer=customer,
        anthropic_client=fake_anthropic,
        result=_result_surfacing(),
        user_query="how do I X?",
        sequence_id="sess_1",
        origin="coding",
    )

    # Documented return shape.
    assert set(finalized.keys()) == {"cost", "cost_micro_usd", "mcr"}

    # 1. Cost-ledger row recorded (and the figure is surfaced for reuse).
    assert finalized["cost"] is not None
    assert finalized["cost_micro_usd"] > 0
    totals = await store.cost_totals_for_team(customer.id)
    assert totals["call_count"] == 1
    assert totals["input_tokens"] == 1000

    # 2. Citation row recorded for the surfaced crystal.
    rows = await store.list_citations_for_crystal(
        customer.id, "cry_A", grounded_only=False,
    )
    assert len(rows) == 1 and rows[0]["grounded"] is True

    # 3. MCR trace persisted.
    assert finalized["mcr"]["trace_id"] is not None


async def test_origin_threads_to_cost_ledger(
    store, customer, fake_anthropic, monkeypatch
):
    _enable(monkeypatch)
    _patch_grounded(monkeypatch, grounded=True)
    fake_anthropic.script_text(_CLEAN_CRITIQUE)

    finalized = await finalize_agent_turn(
        store=store, encoder=object(), customer=customer,
        anthropic_client=fake_anthropic, result=_result_surfacing(),
        user_query="q", sequence_id="sess_1", origin="coding-bg",
    )
    # The coding surfaces' origin lands on the row, making their spend
    # attributable (the HTTP endpoint passes the default "agent").
    assert finalized["cost"]["origin"] == "coding-bg"


async def test_failsafe_when_a_step_errors(
    store, customer, fake_anthropic, monkeypatch
):
    # The cost step blows up; the bundle must not raise, and the later steps
    # (citations, MCR) must still run — each step is independently fail-safe.
    _enable(monkeypatch)
    _patch_grounded(monkeypatch, grounded=True)
    fake_anthropic.script_text(_CLEAN_CRITIQUE)

    async def _boom(*a, **k):
        raise RuntimeError("ledger down")
    monkeypatch.setattr(store, "record_llm_call", _boom)

    finalized = await finalize_agent_turn(
        store=store, encoder=object(), customer=customer,
        anthropic_client=fake_anthropic, result=_result_surfacing(),
        user_query="q", sequence_id="sess_1", origin="coding",
    )

    # Cost failed → None (and the convenience figure is None too), but the turn
    # did not crash and the MCR trace still emitted.
    assert finalized["cost"] is None
    assert finalized["cost_micro_usd"] is None
    assert finalized["mcr"]["trace_id"] is not None
    # Citations still landed despite the cost failure.
    rows = await store.list_citations_for_crystal(
        customer.id, "cry_A", grounded_only=False,
    )
    assert len(rows) == 1
