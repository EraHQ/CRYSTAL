"""Memory-blend Increment 4 — identity-query routing (Tier 2, D-MB6).

Covers:
  - `is_identity_query` (pure detection + symbol extraction)
  - `try_identity_injection` (key-scan routing, via a fake store so no DB
    seeding is needed — the helper only touches list_facts_by_key_prefix)

See docs/MEMORY_BLEND_PLAN.md. Additive routing: any miss falls through to
recall, so these assert both the hits AND that non-identity / miss cases
return None without disturbing the recall path.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from crystal_cache.retrieval.navigation_dispatch import (
    is_identity_query,
    try_identity_injection,
)


# ---------------------------------------------------------------------------
# is_identity_query — pure
# ---------------------------------------------------------------------------

def test_detects_where_is_defined():
    assert is_identity_query("where is generate_sparse_key defined") == "generate_sparse_key"


def test_detects_what_file_is_in():
    assert is_identity_query("what file is CrystalReader in") == "CrystalReader"


def test_detects_which_module_is_in():
    assert is_identity_query("which module is generate_sparse_key in") == "generate_sparse_key"


def test_detects_location_of_with_article():
    assert is_identity_query("location of the drive watcher") == "drive watcher"


def test_detects_where_can_i_find():
    assert is_identity_query("where can I find parse_key") == "parse_key"


def test_ignores_resemblance_queries():
    assert is_identity_query("how does crystallization work") is None
    assert is_identity_query("what is a sparse key") is None
    assert is_identity_query("tell me about the cognition loop") is None
    assert is_identity_query("") is None


# ---------------------------------------------------------------------------
# try_identity_injection — routing via a fake store
# ---------------------------------------------------------------------------

class _FakeStore:
    def __init__(self, facts, *, fail_if_called=False):
        self._facts = facts
        self._fail = fail_if_called
        self.called = False

    async def list_facts_by_key_prefix(
        self, customer_id, *, key_prefix, subject_contains=None, limit=None
    ):
        self.called = True
        if self._fail:
            raise AssertionError("scan should not run for a non-identity query")
        return list(self._facts)


def _fact(prompt_text, claim_text, crystal_id):
    return SimpleNamespace(
        prompt_text=prompt_text,
        claim_text=claim_text,
        answer_value=None,
        crystal_id=crystal_id,
    )


@pytest.mark.asyncio
async def test_identity_hit_injects_provenance():
    facts = [
        _fact(
            "Code|sparse_keys.py|generate_sparse_key",
            "def generate_sparse_key(source, locator): ...",
            "crys_gsk",
        ),
    ]
    store = _FakeStore(facts)
    msgs = [{"role": "user", "content": "where is generate_sparse_key defined"}]

    outcome = await try_identity_injection(
        query_text="where is generate_sparse_key defined",
        customer_id="cus_x",
        store=store,
        messages=msgs,
    )

    assert outcome is not None
    assert outcome.match_type == "high"
    assert outcome.injection_method == "text"
    assert outcome.matched_crystal_ids == ["crys_gsk"]
    # Provenance breadcrumb injected as a NEW system message at the front.
    assert outcome.messages[0]["role"] == "system"
    assert "Code > sparse_keys.py > generate_sparse_key" in outcome.messages[0]["content"]
    # Original user turn preserved at the end.
    assert outcome.messages[-1]["content"] == "where is generate_sparse_key defined"


@pytest.mark.asyncio
async def test_identity_prefers_specific_end_match_over_mid_path():
    """A key naming the symbol at its SPECIFIC (right) end — the identity
    entry point — should beat one that only names it mid-path."""
    facts = [
        _fact(
            "Codebase|sparse_keys|generate_sparse_key|fallback behavior",
            "knowledge about the fn",
            "crys_mid",
        ),
        _fact(
            "Code|sparse_keys.py|generate_sparse_key",
            "fn body",
            "crys_fn",
        ),
    ]
    store = _FakeStore(facts)
    outcome = await try_identity_injection(
        query_text="where is generate_sparse_key defined",
        customer_id="cus_x",
        store=store,
        messages=[{"role": "user", "content": "q"}],
    )
    assert outcome is not None
    assert outcome.matched_crystal_ids == ["crys_fn"]


@pytest.mark.asyncio
async def test_identity_miss_falls_through_to_recall():
    """Scan returns facts that don't literally name the symbol → None."""
    facts = [_fact("Code|other_module.py|something_else", "body", "crys_other")]
    store = _FakeStore(facts)
    outcome = await try_identity_injection(
        query_text="where is generate_sparse_key defined",
        customer_id="cus_x",
        store=store,
        messages=[{"role": "user", "content": "q"}],
    )
    assert outcome is None


@pytest.mark.asyncio
async def test_unstructured_key_is_not_a_location_answer():
    """A non-structured prompt_text (no Source|Locator) can't form a location."""
    facts = [_fact("generate_sparse_key returns a key", "blah", "crys_loose")]
    store = _FakeStore(facts)
    outcome = await try_identity_injection(
        query_text="where is generate_sparse_key defined",
        customer_id="cus_x",
        store=store,
        messages=[{"role": "user", "content": "q"}],
    )
    assert outcome is None


@pytest.mark.asyncio
async def test_non_identity_query_does_not_scan():
    store = _FakeStore([], fail_if_called=True)
    outcome = await try_identity_injection(
        query_text="how does crystallization work",
        customer_id="cus_x",
        store=store,
        messages=[{"role": "user", "content": "q"}],
    )
    assert outcome is None
    assert store.called is False
