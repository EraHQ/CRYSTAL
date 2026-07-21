"""Qdrant-on-Cloud-Run auth (ratified 2026-07-20, option B): audience
derivation + provider caching contract. The live ID-token mint rides
the deploy smoke (metadata server exists only on GCP)."""
from crystal_cache.infrastructure.vector_index import (
    _audience_from_url,
    _make_gcp_id_token_provider,
)


def test_audience_strips_port_and_path():
    assert _audience_from_url(
        "https://crystal-qdrant-118881845105.us-east5.run.app:443"
    ) == "https://crystal-qdrant-118881845105.us-east5.run.app"
    assert _audience_from_url(
        "https://x.run.app/some/path"
    ) == "https://x.run.app"


def test_provider_caches_token(monkeypatch):
    calls = {"n": 0}

    def fake_fetch(request, audience):
        calls["n"] += 1
        return f"tok-{audience}-{calls['n']}"

    import google.oauth2.id_token as idt
    monkeypatch.setattr(idt, "fetch_id_token", fake_fetch)

    provider = _make_gcp_id_token_provider("https://aud.run.app")
    t1 = provider()
    t2 = provider()
    assert t1 == t2                      # cached inside the window
    assert calls["n"] == 1


def test_hydration_upserts_are_batched():
    """413 fix (2026-07-20): a real bank's one-shot hydration upsert
    exceeds Qdrant's 32MB request cap; batches must be bounded."""
    import asyncio

    from crystal_cache.infrastructure.qdrant_vector_index import (
        QdrantVectorIndex,
    )

    calls = []

    class _FakeClient:
        async def upsert(self, collection, points=None):
            calls.append((collection, len(points)))

    idx = QdrantVectorIndex.__new__(QdrantVectorIndex)
    idx._client = _FakeClient()

    points = list(range(1234))
    asyncio.run(idx._upsert_batched("crys_facts", points, 500))
    assert [n for _, n in calls] == [500, 500, 234]
    assert all(c == "crys_facts" for c, _ in calls)
