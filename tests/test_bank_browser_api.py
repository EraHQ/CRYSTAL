"""Console Refine PR 4 (docs/CONSOLE_UX_AUDIT.md 5.1, Part 6): the bank
list route searches and filters on the server, so the console can find
a crystal that sits past the first page, and carries the overview and
recall counts the page renders.

Driven over the real app (test_endpoint_smoke._build_app) with 250
crystals seeded straight into the store: the acceptance line in the spec
is "with 250 crystals, search finds one that sits past the first 200".
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from crystal_cache.infrastructure.schema import CrystalRow, FactRow


def _app(store, encoder, vector_store, fact_vector_store):
    try:
        from tests.test_endpoint_smoke import _build_app
    except ModuleNotFoundError:
        from test_endpoint_smoke import _build_app
    return _build_app(store, encoder, vector_store, fact_vector_store)


def _client(app):
    from httpx import ASGITransport, AsyncClient

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


TIERS = ["whitelist", "neutral", "quarantine", "blacklist"]


async def _seed(store, cid: str, n: int = 250) -> dict[str, str]:
    """n crystals, newest last-created first in the listing; the needle
    is the sixth oldest, so it sits at position ~245 of 250."""
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ids: dict[str, str] = {}
    async with store.session() as s:
        for i in range(n):
            cid_i = f"cr_{i:04d}"
            when = base + timedelta(seconds=i)
            summary = "the needle zebra crystal" if i == 5 else f"crystal number {i}"
            s.add(CrystalRow(
                id=cid_i, customer_id=cid, summary_vector=[], summary_text=summary,
                quality_tier=TIERS[i % 4], crystal_type="customer:legacy",
                source_path=("docs/guide_50%_done.md" if i == 7 else None),
                created_at=when, last_activity=when,
            ))
            s.add(FactRow(
                id=f"f_{i:04d}", crystal_id=cid_i, pair_type="question_answer",
                prompt_text=f"Topic|k{i}", claim_text=("the unicorn claim" if i == 9 else f"claim {i}"),
                vector=[], hit_count=i % 7, created_at=when,
            ))
            ids[str(i)] = cid_i
    return ids


@pytest.mark.asyncio
async def test_search_finds_a_crystal_past_the_first_page(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    ids = await _seed(store, customer.id)
    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    base = f"/admin/api/customers/{customer.id}/crystals"
    async with _client(app) as c:
        # The first page without a term: 200 newest; the needle is not in it.
        r = await c.get(base, params={"limit": 200})
        assert r.status_code == 200, r.text[:300]
        body = r.json()
        assert body["total"] == 250 and len(body["crystals"]) == 200
        assert ids["5"] not in {x["id"] for x in body["crystals"]}
        # By summary text.
        r = await c.get(base, params={"q": "zebra"})
        assert r.status_code == 200
        assert [x["id"] for x in r.json()["crystals"]] == [ids["5"]]
        assert r.json()["total"] == 1
        # By a fact's claim and by a fact's key.
        r = await c.get(base, params={"q": "UNICORN"})
        assert [x["id"] for x in r.json()["crystals"]] == [ids["9"]]
        r = await c.get(base, params={"q": "Topic|k12"})
        found = {x["id"] for x in r.json()["crystals"]}
        assert ids["12"] in found and ids["5"] not in found
        # By source path, with the LIKE metacharacters in the term taken
        # literally: "50%" matches the one path that contains it, not 250 rows.
        r = await c.get(base, params={"q": "50%"})
        assert [x["id"] for x in r.json()["crystals"]] == [ids["7"]]
        r = await c.get(base, params={"q": "_"})
        assert r.json()["total"] == 1 and r.json()["crystals"][0]["id"] == ids["7"]
        # A term that matches nothing is an honest empty page.
        r = await c.get(base, params={"q": "nothing-here-xyz"})
        assert r.json()["total"] == 0 and r.json()["crystals"] == []
        # Bounded: a 201-character term is refused like every other scalar
        # (400 behind app.py's validation handler; the bare test app says 422).
        r = await c.get(base, params={"q": "z" * 201})
        assert r.status_code in (400, 422)


@pytest.mark.asyncio
async def test_tier_filter_overview_and_recall_hits(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    ids = await _seed(store, customer.id)
    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    base = f"/admin/api/customers/{customer.id}/crystals"
    async with _client(app) as c:
        r = await c.get(base, params={"tier": "quarantine", "limit": 200})
        assert r.status_code == 200
        body = r.json()
        # i % 4 == 2 -> 62 or 63 of 250
        assert body["total"] == len([i for i in range(250) if i % 4 == 2])
        assert all(x["quality_tier"] == "quarantine" for x in body["crystals"])
        # A term AND a tier compose.
        r = await c.get(base, params={"tier": "neutral", "q": "zebra"})
        assert r.json()["total"] == 1 and r.json()["crystals"][0]["id"] == ids["5"]  # 5 % 4 == 1 -> neutral
        r = await c.get(base, params={"tier": "blacklist", "q": "zebra"})
        assert r.json()["total"] == 0
        # An unknown tier is refused, not ignored.
        r = await c.get(base, params={"tier": "golden"})
        assert r.status_code in (400, 422)
        # The overview rides every page and counts the whole bank, not the filter.
        r = await c.get(base, params={"tier": "whitelist", "limit": 5})
        bank = r.json()["bank"]
        assert bank["crystals"] == 250 and bank["facts"] == 250
        assert bank["tier_counts"] == {t: len([i for i in range(250) if TIERS[i % 4] == t]) for t in TIERS}
        assert bank["updated_at"].startswith("2026-01-01T00:04:09")
        # Recall hits per crystal come from the facts' hit_count.
        r = await c.get(base, params={"limit": 10})
        for x in r.json()["crystals"]:
            i = int(x["id"].split("_")[1])
            assert x["recall_hits"] == i % 7, x
        # The detail route carries the per-fact count and the identity
        # fields the drawer needs when opened straight from /bank/{id}.
        r = await c.get(f"/admin/api/crystals/{ids['13']}")
        assert r.status_code == 200
        body = r.json()
        assert body["facts"][0]["hit_count"] == 13 % 7
        assert body["facts"][0]["source_doc_id"] is None
        assert body["crystal"]["recall_hits"] == 13 % 7
        assert body["crystal"]["headline_key"] == "Topic|k13"
        assert body["crystal"]["source_path"] is None
        r = await c.get(f"/admin/api/crystals/{ids['7']}")
        assert r.json()["crystal"]["source_path"] == "docs/guide_50%_done.md"
