"""System rules (2026-07-03, steps 3-4) — typed rules + promotion evaluator.

Covers:
  - STRICT validation: unknown keys / wrong types / bad tiers reject;
  - CRUD round-trip;
  - the promotion evaluator: a rule clears the recall gate on matching
    gated crystals when conditions hold, and does NOT when they don't;
  - the safe default: no rule => nothing promoted;
  - safety: rules never promote non-matching origin, never touch blacklist,
    audit fields update.

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

import pytest

from crystal_cache.models.crystal import Crystal
from crystal_cache.system_rules import (
    RuleValidationError,
    SystemRule,
    validate_rule,
)
from crystal_cache.system_rules import store as rules_store


# --- strict validation ------------------------------------------------------

def _rule(**over):
    base = dict(
        id="r", customer_id="c", rule_type="promotion", name="t",
        selector={"origin": "background_worker"},
        conditions={"outbound_scan_passed": True},
        action={"clear_recall_gate": True},
    )
    base.update(over)
    return SystemRule(**base)


def test_valid_promotion_rule_passes():
    validate_rule(_rule())


def test_unknown_condition_key_rejected():
    with pytest.raises(RuleValidationError, match="unknown key"):
        validate_rule(_rule(conditions={"conditon_typo": True}))


def test_unknown_selector_key_rejected():
    with pytest.raises(RuleValidationError, match="unknown key"):
        validate_rule(_rule(selector={"origin": "x", "orign2": "y"}))


def test_unknown_action_key_rejected():
    with pytest.raises(RuleValidationError, match="unknown key"):
        validate_rule(_rule(action={"clear_recall_gate": True, "nope": 1}))


def test_missing_origin_rejected():
    with pytest.raises(RuleValidationError, match="origin"):
        validate_rule(_rule(selector={}))


def test_clear_recall_gate_must_be_true():
    with pytest.raises(RuleValidationError, match="clear_recall_gate"):
        validate_rule(_rule(action={"clear_recall_gate": False}))


def test_set_tier_cannot_be_blacklist():
    with pytest.raises(RuleValidationError, match="set_tier"):
        validate_rule(_rule(action={"clear_recall_gate": True,
                                    "set_tier": "blacklist"}))


def test_bool_not_accepted_where_int_required():
    with pytest.raises(RuleValidationError, match="int"):
        validate_rule(_rule(conditions={"min_grounded_citations": True}))


def test_unknown_rule_type_rejected():
    with pytest.raises(RuleValidationError, match="unknown rule_type"):
        validate_rule(_rule(rule_type="does_not_exist"))


# --- CRUD -------------------------------------------------------------------

async def test_create_and_list_rule(store, customer):
    r = await rules_store.create_rule(
        store, customer.id, "promotion", "auto-promote research",
        selector={"origin": "background_worker"},
        conditions={"outbound_scan_passed": True},
        action={"clear_recall_gate": True},
    )
    assert r.id.startswith("rule_")
    listed = await rules_store.list_rules(store, customer.id)
    assert len(listed) == 1
    assert listed[0].name == "auto-promote research"


async def test_create_invalid_rule_raises_before_write(store, customer):
    with pytest.raises(RuleValidationError):
        await rules_store.create_rule(
            store, customer.id, "promotion", "bad",
            selector={"origin": "background_worker"},
            conditions={"typo": True},
            action={"clear_recall_gate": True},
        )
    # nothing persisted
    assert await rules_store.list_rules(store, customer.id) == []


async def test_delete_rule(store, customer):
    r = await rules_store.create_rule(
        store, customer.id, "promotion", "x",
        selector={"origin": "background_worker"},
        conditions={}, action={"clear_recall_gate": True},
    )
    assert await rules_store.delete_rule(store, customer.id, r.id) is True
    assert await rules_store.list_rules(store, customer.id) == []


# --- promotion evaluator ----------------------------------------------------

async def _gated(store, customer_id, cid, *, origin="background_worker",
                 tags=None, tier="quarantine"):
    c = Crystal(
        id=cid, customer_id=customer_id, summary_vector=[0.1],
        crystal_type="customer:legacy", recall_gated=True, origin=origin,
        quality_tier=tier, diagnostic_tags=tags or [],
    )
    await store.upsert_crystal(c)
    return c


async def test_no_rule_promotes_nothing(store, customer):
    await _gated(store, customer.id, "g1")
    result = await rules_store.run_promotion_rules(store, customer.id)
    assert result == {"promoted": 0, "rules_fired": 0}
    # still gated
    assert (await store.get_crystal("g1")).recall_gated is True


async def test_rule_clears_gate_when_conditions_hold(store, customer):
    # gated crystal WITH the scan-passed tag
    await _gated(store, customer.id, "g2", tags=["outbound_scan_passed"])
    await rules_store.create_rule(
        store, customer.id, "promotion", "auto",
        selector={"origin": "background_worker"},
        conditions={"outbound_scan_passed": True},
        action={"clear_recall_gate": True},
    )
    result = await rules_store.run_promotion_rules(store, customer.id)
    assert result["promoted"] == 1
    assert result["rules_fired"] == 1
    # now usable
    got = await store.get_crystal("g2")
    assert got.recall_gated is False
    assert got.quality_tier == "quarantine"  # default set_tier


async def test_rule_does_not_fire_when_conditions_fail(store, customer):
    # gated crystal WITHOUT the scan-passed tag
    await _gated(store, customer.id, "g3", tags=[])
    await rules_store.create_rule(
        store, customer.id, "promotion", "auto",
        selector={"origin": "background_worker"},
        conditions={"outbound_scan_passed": True},
        action={"clear_recall_gate": True},
    )
    result = await rules_store.run_promotion_rules(store, customer.id)
    assert result["promoted"] == 0
    assert (await store.get_crystal("g3")).recall_gated is True


async def test_rule_only_matches_its_origin(store, customer):
    # a gated crystal of a DIFFERENT origin must not be promoted
    await _gated(store, customer.id, "g4", origin="direct",
                 tags=["outbound_scan_passed"])
    await rules_store.create_rule(
        store, customer.id, "promotion", "auto",
        selector={"origin": "background_worker"},
        conditions={"outbound_scan_passed": True},
        action={"clear_recall_gate": True},
    )
    result = await rules_store.run_promotion_rules(store, customer.id)
    assert result["promoted"] == 0
    assert (await store.get_crystal("g4")).recall_gated is True


async def test_set_tier_neutral_override(store, customer):
    await _gated(store, customer.id, "g5", tags=["outbound_scan_passed"])
    await rules_store.create_rule(
        store, customer.id, "promotion", "trust-scan",
        selector={"origin": "background_worker"},
        conditions={"outbound_scan_passed": True},
        action={"clear_recall_gate": True, "set_tier": "neutral"},
    )
    await rules_store.run_promotion_rules(store, customer.id)
    got = await store.get_crystal("g5")
    assert got.recall_gated is False
    assert got.quality_tier == "neutral"


async def test_fire_audit_updates(store, customer):
    await _gated(store, customer.id, "g6", tags=["outbound_scan_passed"])
    r = await rules_store.create_rule(
        store, customer.id, "promotion", "auto",
        selector={"origin": "background_worker"},
        conditions={"outbound_scan_passed": True},
        action={"clear_recall_gate": True},
    )
    await rules_store.run_promotion_rules(store, customer.id)
    listed = await rules_store.list_rules(store, customer.id)
    fired = next(x for x in listed if x.id == r.id)
    assert fired.fire_count == 1
    assert fired.last_fired_at is not None


# --- the idle-cycle worker pass (steps 3-4 wiring) ---------------------------

async def test_worker_pass_clears_gates_across_customers(store, customer):
    """_run_system_promotion_rules is the convergence-family wiring: one
    idle pass evaluates every customer's rules. Proves the evaluator is
    actually CALLED by the worker path (not dead code) and that a
    second customer without rules is untouched (fail-safe + no-op)."""
    from crystal_cache.workers.cognition import _run_system_promotion_rules

    # Customer A: a gated crystal + a rule whose conditions hold.
    await _gated(store, customer.id, "wp1", tags=["outbound_scan_passed"])
    await rules_store.create_rule(
        store, customer.id, "promotion", "auto",
        selector={"origin": "background_worker"},
        conditions={"outbound_scan_passed": True},
        action={"clear_recall_gate": True},
    )

    # Customer B: a gated crystal, NO rules — must stay gated.
    other = await store.create_customer(
        provider="anthropic", model_id="claude-sonnet-4-5-20250929",
        api_key_ref="test-ref-other",
    )
    await _gated(store, other.id, "wp2", tags=["outbound_scan_passed"])

    await _run_system_promotion_rules(store=store)

    assert (await store.get_crystal("wp1")).recall_gated is False  # promoted
    assert (await store.get_crystal("wp2")).recall_gated is True   # untouched
