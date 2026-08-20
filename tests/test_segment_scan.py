"""C3 — contradiction participation via the segment layer (ratified
2026-08-11: Q1=A any-shared-segment grouping, rarity-ordered,
fraction-excluded; Q2=A pure-D5 verdicts + conflict_found witness;
Q3=A gap domain = widest segment).

Covers the segment helpers, the contradiction scan's segment-set
enumeration (the headline retroactivity case: an `Assumptions|X` fact
finally pairing against a stated fact carrying X at any position), the
conflict write + its curation-feed witness, gap discovery's
domain-from-leftmost, and the funnel's segment adoption. The renderer-
level helpers additionally ran 11 standalone container checks at
authoring time (/tmp/work3 rig, 2026-08-11).
"""
from __future__ import annotations


from crystal_cache.infrastructure.schema import CrystalRow, FactRow
from crystal_cache.scan.segments import (
    exclusion_threshold,
    group_by_shared_segment,
    has_segment,
    most_specific_shared_segment,
    segments_of,
    widest_segment_of,
)


_FAKE_SEQ = iter(range(10_000))


class _FakeFact:
    def __init__(self, key, claim="claim", crystal="cr"):
        self.id = f"fake_{next(_FAKE_SEQ)}"
        self.prompt_text = key
        self.claim_text = claim
        self.crystal_id = crystal


# ---------------------------------------------------------------------------
# Segment helpers (pure)
# ---------------------------------------------------------------------------

def test_segment_primitives():
    assert segments_of("Film|Corporate Mistletoe|Script") == (
        "Film", "Corporate Mistletoe", "Script",
    )
    assert segments_of("") == () and segments_of(None) == ()
    assert widest_segment_of("Film|X|Y") == "Film"
    assert widest_segment_of(None) is None
    assert has_segment("A|B|C", "b") is True
    assert has_segment("A|B|C", "d") is False
    assert has_segment("A|B|C", "") is False


def test_most_specific_shared_segment():
    assert most_specific_shared_segment(
        "A|B|Scene 5", "Z|Scene 5|B",
    ) == "Scene 5"
    assert most_specific_shared_segment("A|B", "C|D") is None
    assert most_specific_shared_segment("", "A|B") is None


def test_grouping_rarity_order_and_exclusion():
    facts = [
        _FakeFact(f"Common|item {i}", crystal=f"c{i}") for i in range(20)
    ]
    facts += [
        _FakeFact("Common|Rare Pair|alpha", crystal="ca"),
        _FakeFact("Zeta|Rare Pair|beta", crystal="cb"),
    ]
    groups = group_by_shared_segment(facts, max_group_fraction=0.25)
    assert "Common" not in groups          # namespace-scale: excluded
    assert "Rare Pair" in groups
    assert list(groups)[0] == "Rare Pair"  # rarest first
    assert exclusion_threshold(8, 0.25) == 4  # small-bank floor


def test_grouping_is_case_insensitive_first_seen_casing():
    facts = [
        _FakeFact("Ops|deploy RISK windows", crystal="c1"),
        _FakeFact("Assumptions|Deploy Risk Windows", crystal="c2"),
    ]
    groups = group_by_shared_segment(facts, max_group_fraction=0.95)
    merged = [k for k, v in groups.items() if len(v) == 2]
    assert merged == ["deploy RISK windows"]  # first-seen casing


# ---------------------------------------------------------------------------
# Contradiction scan — enumeration + write + witness
# ---------------------------------------------------------------------------

def test_enumeration_pairs_assumption_with_stated_fact():
    """The headline C3 case: a 2-part namespace key meets a deep
    unified key at their shared segment — cross-crystal, any position.
    The retired positional parse was structurally blind to this pair."""
    from crystal_cache.scan.contradiction import _enumerate_candidate_pairs

    facts = [
        _FakeFact(
            "Assumptions|Deploy risk windows",
            claim="Friday deploys carry elevated risk",
            crystal="cr_asm",
        ),
        _FakeFact(
            "Ops|Deploy risk windows|Policy|Section 3",
            claim="Friday deploys were banned in March",
            crystal="cr_doc",
        ),
        _FakeFact(
            "Film|Corporate Mistletoe|Script|Scene 5",
            claim="Unrelated scene content",
            crystal="cr_film",
        ),
    ]
    pairs = _enumerate_candidate_pairs(facts, max_pairs=50)
    crystal_pairs = {frozenset((a.crystal_id, b.crystal_id)) for a, b in pairs}
    assert frozenset(("cr_asm", "cr_doc")) in crystal_pairs


class _ContradictsClient:
    """Legacy-shape test client: every judged pair CONTRADICTS."""

    def complete(self, **kwargs) -> str:
        return "CONTRADICTS"


async def _seed_fact(store, customer_id, crystal_id, key, claim):
    async with store.session() as s:
        s.add(CrystalRow(
            id=crystal_id, customer_id=customer_id,
            crystal_type="customer:legacy", summary_vector=[],
        ))
        s.add(FactRow(
            id=f"f_{crystal_id}",
            crystal_id=crystal_id,
            pair_type="question_answer",
            prompt_text=key,
            claim_text=claim,
            vector=[],
        ))


async def test_scan_writes_conflict_and_witness(store, customer):
    """End-to-end: assumption-keyed fact vs stated fact -> CONTRADICTS
    -> open conflict (segment-derived subject) + conflict_found in the
    curation feed (Q2=A)."""
    from crystal_cache.scan.contradiction import scan_for_contradictions

    await _seed_fact(
        store, customer.id, "cr_asm",
        "Assumptions|Deploy risk windows",
        "Friday deploys carry elevated risk",
    )
    await _seed_fact(
        store, customer.id, "cr_doc",
        "Ops|Deploy risk windows|Policy|Section 3",
        "Friday deploys were banned in March",
    )

    result = await scan_for_contradictions(
        store=store,
        slm_client=_ContradictsClient(),
        customer_id=customer.id,
    )
    assert result.conflicts_found >= 1

    conflicts = await store.list_knowledge_conflicts(
        customer.id, status="open", limit=10,
    )
    assert any(c.subject == "Deploy risk windows" for c in conflicts)

    events = await store.list_curation_events(customer.id)
    found = [e for e in events if e["event_type"] == "conflict_found"]
    assert len(found) >= 1
    assert found[0]["payload"]["detector"] == "contradiction_scan"
    assert "Deploy risk windows" in found[0]["label"]


async def test_rescan_is_noop_and_witnesses_once(store, customer):
    """D4 idempotence survives the segment rewrite: the second scan
    skips the recorded pair and emits no second witness."""
    from crystal_cache.scan.contradiction import scan_for_contradictions

    await _seed_fact(
        store, customer.id, "cr_a", "Assumptions|Rate policy",
        "The rate is $120/hr",
    )
    await _seed_fact(
        store, customer.id, "cr_b", "Contract|Acme|Rate policy|Finance",
        "The rate is $95/hr",
    )
    for _ in range(2):
        await scan_for_contradictions(
            store=store,
            slm_client=_ContradictsClient(),
            customer_id=customer.id,
        )

    events = await store.list_curation_events(customer.id)
    assert len([
        e for e in events if e["event_type"] == "conflict_found"
    ]) == 1


# ---------------------------------------------------------------------------
# Gap discovery — grouping + domain from the widest segment (Q3=A)
# ---------------------------------------------------------------------------

def test_gap_discovery_grouping_uses_segments():
    from crystal_cache.scan.gap_discovery import _group_by_subject

    facts = [
        _FakeFact("Assumptions|Vendor pricing", crystal="c1"),
        _FakeFact("Docs|Vendor pricing|Quote 7", crystal="c2"),
    ]
    grouped = _group_by_subject(facts, min_facts=2)
    assert "Vendor pricing" in grouped
    assert len(grouped["Vendor pricing"]) == 2


class _GapClient:
    """Every subject yields a discovered gap."""

    def complete(self, **kwargs) -> str:
        return "What the current vendor pricing tiers are"


async def test_gap_discovery_domain_is_widest_segment(store, customer):
    from crystal_cache.scan.gap_discovery import discover_gaps

    await _seed_fact(
        store, customer.id, "cr_1",
        "Finance|Vendor pricing|Quote 7", "Quote 7 says $10/unit",
    )
    await _seed_fact(
        store, customer.id, "cr_2",
        "Assumptions|Vendor pricing", "Pricing likely tiered",
    )

    result = await discover_gaps(
        store=store,
        slm_client=_GapClient(),
        customer_id=customer.id,
        min_facts_per_subject=2,
    )
    assert result.gaps_found >= 1

    gaps = await store.list_knowledge_gaps(
        customer.id, status="open", limit=10,
    )
    gap = next(g for g in gaps if g.source == "gap_discovery")
    # Q3=A: the representative fact's LEFTMOST segment, read back.
    assert gap.domain in ("Finance", "Assumptions")


# ---------------------------------------------------------------------------
# Funnel — segment adoption (gap_subject any-position; key_adjacent rare)
# ---------------------------------------------------------------------------

async def test_funnel_gap_subject_matches_any_position(store, customer):
    from crystal_cache.scan.pairing_funnel import FunnelState, run_pairing_funnel

    await _seed_fact(
        store, customer.id, "cr_x",
        "Docs|Deploy risk windows|Section 1", "stated a",
    )
    await _seed_fact(
        store, customer.id, "cr_y",
        "Policy|Ops|Deploy risk windows", "stated b",
    )
    await store.create_knowledge_gap(
        customer.id,
        domain=None,
        subject="Deploy risk windows",
        missing="Which deploy windows are risky",
    )

    result = await run_pairing_funnel(
        store=store,
        customer_id=customer.id,
        state=FunnelState(),
    )
    assert result.gap_subject_edges >= 1
