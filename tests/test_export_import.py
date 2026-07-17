"""Tests for /v1/export + /v1/import (B.4).

Exercises the bank export/import round-trip through the real SDK handlers
(sdk_store -> sdk_export -> sdk_import), using the conftest fixtures (fresh
in-memory store + customer per test, semantic encoder stub, vector stores).

generate_sparse_key is patched to a deterministic, network-free identity so
tests neither hit the Haiku key generator nor depend on its output. Because an
exported `key` is the already-sparse stored prompt_text, identity sparsification
makes the export->import round-trip byte-exact here — the same property the
LLM-free fallback path (first-8-words of an already-short key) gives in prod.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from crystal_cache.endpoints.sdk import sdk_export, sdk_import, sdk_store
from crystal_cache.ingress.schema import ImportRequest, StoreRequest


def _req(encoder: Any, vector_store: Any, fact_vector_store: Any = None) -> Any:
    """A minimal Request stand-in carrying the app.state the handlers read."""
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                prompt_encoder=encoder,
                vector_store=vector_store,
                fact_vector_store=fact_vector_store,
            )
        )
    )


def _expected_key(key: str, value: str) -> str:
    """The stored sparse key for a raw (unflagged) store/import under the
    identity sparsifier patched in below: /v1/store and unflagged /v1/import
    derive the key from `key + value`, and _fake collapses whitespace.
    """
    return " ".join(f"{key}: {value}".split())


@pytest.fixture(autouse=True)
def _identity_sparse_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch BOTH sparse-key entry points to a deterministic, network-free
    identity.

    The handlers import them locally at call time, so patching the module
    attributes is enough. Collapsing whitespace mirrors the real sanitizer
    closely enough for round-trip assertions without an LLM call.

    Gate B (2026-07-16): the endpoints moved to generate_sparse_key_metered;
    patching only the sync name let the metered path reach the REAL client on
    machines with ANTHROPIC_API_KEY exported (real Haiku calls inside pytest).
    Both names are pinned so the suite never depends on shell env.
    """
    def _fake(text: str, **_kwargs: Any) -> str:
        return " ".join(str(text).split())

    async def _fake_metered(text: str, **_kwargs: Any) -> str:
        return " ".join(str(text).split())

    monkeypatch.setattr(
        "crystal_cache.encoding.sparse_keys.generate_sparse_key", _fake
    )
    monkeypatch.setattr(
        "crystal_cache.encoding.sparse_keys.generate_sparse_key_metered",
        _fake_metered,
    )


async def _seed(store, customer, encoder, vector_store, facts) -> None:
    """Store (key, value) pairs through the real sdk_store handler."""
    req = _req(encoder, vector_store)
    for key, value in facts:
        await sdk_store(
            body=StoreRequest(key=key, value=value),
            request=req,
            principal=(customer, None),
            store=store,
        )


async def test_export_import_round_trip(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store
):
    facts = [
        ("primary database", "PostgreSQL 16 in production."),
        ("deploy process", "GitHub Actions on merge to main."),
    ]
    await _seed(store, customer, semantic_encoder_stub, vector_store, facts)

    # Export reflects what was stored.
    exp = await sdk_export(customer=customer, store=store)
    assert exp.record_count == 2
    assert exp.export_format == "jsonl"
    assert {r["key"] for r in exp.data} == {_expected_key(*f) for f in facts}
    assert {r["value"] for r in exp.data} == {f[1] for f in facts}
    # Export stamps key_is_path so a round-trip import preserves the key.
    assert all(r["key_is_path"] is True for r in exp.data)
    # Crystal metadata rides on each record (stored under the default type).
    assert all(r["crystal_type"] == "customer:legacy" for r in exp.data)
    assert all("pair_type" in r and "source_kind" in r for r in exp.data)

    # Wipe + re-import the exported payload; the bank reconstitutes.
    req = _req(semantic_encoder_stub, vector_store, fact_vector_store)
    imp = await sdk_import(
        body=ImportRequest(records=exp.data, wipe=True),
        request=req,
        customer=customer,
        store=store,
    )
    assert imp.records_processed == 2
    assert imp.errors == 0
    assert imp.crystals_written >= 1

    exp2 = await sdk_export(customer=customer, store=store)
    assert exp2.record_count == 2
    assert {r["key"] for r in exp2.data} == {_expected_key(*f) for f in facts}
    assert {r["value"] for r in exp2.data} == {f[1] for f in facts}


async def test_import_wipe_clears_existing(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store
):
    await _seed(
        store, customer, semantic_encoder_stub, vector_store,
        [("old key", "old value")],
    )
    req = _req(semantic_encoder_stub, vector_store, fact_vector_store)
    imp = await sdk_import(
        body=ImportRequest(
            records=[{"key": "new key", "value": "new value"}], wipe=True
        ),
        request=req,
        customer=customer,
        store=store,
    )
    assert imp.records_processed == 1

    exp = await sdk_export(customer=customer, store=store)
    assert exp.record_count == 1
    assert exp.data[0]["key"] == _expected_key("new key", "new value")
    assert exp.data[0]["value"] == "new value"


async def test_import_without_wipe_appends(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store
):
    await _seed(
        store, customer, semantic_encoder_stub, vector_store, [("first", "one")]
    )
    req = _req(semantic_encoder_stub, vector_store, fact_vector_store)
    await sdk_import(
        body=ImportRequest(
            records=[{"key": "second", "value": "two"}], wipe=False
        ),
        request=req,
        customer=customer,
        store=store,
    )
    exp = await sdk_export(customer=customer, store=store)
    assert exp.record_count == 2
    assert {r["key"] for r in exp.data} == {_expected_key("first", "one"), _expected_key("second", "two")}


async def test_export_empty_bank(store, customer):
    exp = await sdk_export(customer=customer, store=store)
    assert exp.record_count == 0
    assert exp.data == []
    assert exp.export_format == "jsonl"


async def test_import_bad_records_counted_not_fatal(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store
):
    req = _req(semantic_encoder_stub, vector_store, fact_vector_store)
    imp = await sdk_import(
        body=ImportRequest(
            records=[
                {"key": "good", "value": "ok"},
                {"key": "", "value": "missing key"},   # empty key -> error
                {"key": "novalue", "value": ""},        # empty value -> error
            ],
            wipe=False,
        ),
        request=req,
        customer=customer,
        store=store,
    )
    assert imp.records_processed == 1
    assert imp.errors == 2

    # The one good record landed.
    exp = await sdk_export(customer=customer, store=store)
    assert exp.record_count == 1
    assert exp.data[0]["key"] == _expected_key("good", "ok")


async def test_flagged_record_preserved_verbatim_unflagged_derived(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store
):
    """key_is_path=True stores the key verbatim (path-stable restore); an
    unflagged record derives a path from key + value, like /v1/store. This is
    the branch that keeps export->import idempotent."""
    req = _req(semantic_encoder_stub, vector_store, fact_vector_store)
    imp = await sdk_import(
        body=ImportRequest(
            records=[
                {
                    "key": "Infrastructure|Database|Production",
                    "value": "PostgreSQL 16.",
                    "key_is_path": True,
                },
                {"key": "deploy target", "value": "GitHub Actions on merge."},
            ],
            wipe=True,
        ),
        request=req,
        customer=customer,
        store=store,
    )
    assert imp.records_processed == 2
    assert imp.errors == 0

    keys = {
        r["key"]
        for r in (await sdk_export(customer=customer, store=store)).data
    }
    # Flagged record: the already-finished path is preserved verbatim.
    assert "Infrastructure|Database|Production" in keys
    # Unflagged record: a path derived from key + value (identity sparsifier).
    assert _expected_key("deploy target", "GitHub Actions on merge.") in keys
