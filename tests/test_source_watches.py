"""Gate M slice 1 (2026-07-18): source_watches — the general watch
registration. One table for every scheme (M-Q1=A); these pin the CRUD
contract and the due-cycle semantics the sync worker builds on."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


@pytest.mark.asyncio
async def test_watch_lifecycle(store, customer):
    w = await store.create_source_watch(
        customer.id, scheme="git",
        source_name="crystal-cache-v2",
        config={"repo": "https://github.com/EraHQ/crystal-cache-v2",
                "branch": "master"},
        cadence_minutes=15,
    )
    assert w.id.startswith("watch_")
    assert w.review_mode == "auto"          # M-Q3 default

    got = await store.get_source_watch(w.id, customer.id)
    assert got is not None
    assert got.scheme == "git"
    assert got.config["branch"] == "master"

    listed = await store.list_source_watches(customer.id)
    assert [x.id for x in listed] == [w.id]

    assert await store.set_source_watch_status(w.id, customer.id, "paused")
    assert (await store.get_source_watch(w.id, customer.id)).status == "paused"

    assert await store.delete_source_watch(w.id, customer.id)
    assert await store.get_source_watch(w.id, customer.id) is None


@pytest.mark.asyncio
async def test_tenancy_guard(store, customer):
    w = await store.create_source_watch(
        customer.id, scheme="git", source_name="x", config={},
    )
    assert await store.get_source_watch(w.id, "cus_other") is None
    assert not await store.delete_source_watch(w.id, "cus_other")


@pytest.mark.asyncio
async def test_due_cycle_semantics(store, customer):
    now = datetime.now(timezone.utc)
    w = await store.create_source_watch(
        customer.id, scheme="git", source_name="due-test",
        config={}, cadence_minutes=15,
    )
    # Never checked -> due immediately (first sync, M-Q4).
    due = await store.list_source_watches_due(now)
    assert w.id in {x.id for x in due}

    # Freshly checked -> not due.
    await store.update_source_watch_state(
        w.id, customer.id, last_state={"head": "abc123"}, checked_at=now,
    )
    due = await store.list_source_watches_due(now + timedelta(minutes=5))
    assert w.id not in {x.id for x in due}

    # Cadence elapsed -> due again, state intact.
    due = await store.list_source_watches_due(now + timedelta(minutes=16))
    assert w.id in {x.id for x in due}
    got = await store.get_source_watch(w.id, customer.id)
    assert got.last_state == {"head": "abc123"}
    assert got.last_error is None

    # Paused watches never come due.
    await store.set_source_watch_status(w.id, customer.id, "paused")
    due = await store.list_source_watches_due(now + timedelta(hours=1))
    assert w.id not in {x.id for x in due}


# --- Slice 2: the C6 envelope + handler registry ---------------------------

def test_registry_dispatch_and_idempotent_registration():
    from crystal_cache.ingestion.source_handlers import (
        ChangeSet,
        get_handler,
        register_handler,
        registered_schemes,
    )

    class FakeHandler:
        scheme = "faketest"
        async def check(self, watch, token):
            return ChangeSet(new_state={"v": 1}, changed=["a.py"])
        async def fetch(self, watch, path, token):
            raise NotImplementedError

    register_handler(FakeHandler())
    assert "faketest" in registered_schemes()
    assert get_handler("faketest").scheme == "faketest"
    assert get_handler("nope") is None
    # Re-registration replaces, not duplicates.
    register_handler(FakeHandler())
    assert registered_schemes().count("faketest") == 1


def test_changeset_empty_semantics():
    from crystal_cache.ingestion.source_handlers import ChangeSet
    assert ChangeSet(new_state={}).empty
    assert not ChangeSet(new_state={}, changed=["x"]).empty
    assert not ChangeSet(new_state={}, removed=["y"]).empty


@pytest.mark.asyncio
async def test_watch_token_roundtrip(store, customer):
    """M-Q5: per-watch token encrypts under the tenant DEK and resolves
    back; absent token resolves None (scheme fallback is handler
    policy); a foreign tenant's decrypt attempt fails closed."""
    from crystal_cache.ingestion.source_handlers import (
        encrypt_watch_token,
        resolve_watch_token,
    )
    enc = await encrypt_watch_token(store, customer.id, "ghp_secret123")
    assert enc.startswith("enc:v2")
    w = await store.create_source_watch(
        customer.id, scheme="git", source_name="tok-test",
        config={}, encrypted_token=enc,
    )
    assert await resolve_watch_token(store, w) == "ghp_secret123"

    w2 = await store.create_source_watch(
        customer.id, scheme="git", source_name="tokenless", config={},
    )
    assert await resolve_watch_token(store, w2) is None

    class _Foreign:
        customer_id = "cus_intruder"
        encrypted_token = enc
    with pytest.raises(ValueError):
        await resolve_watch_token(store, _Foreign())


# --- Slice 3: the git handler (faked API — no network in the suite) --------

class _FakeGit:
    """Canned GitHub API. Keyed by URL substring."""
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def __call__(self, url, token):
        self.calls.append(url)
        for key, value in self.responses.items():
            if key in url:
                return value
        raise AssertionError(f"unexpected URL: {url}")


class _W:
    def __init__(self, config, last_state=None, source_name="myrepo"):
        self.config = config
        self.last_state = last_state
        self.source_name = source_name
        self.customer_id = "cus_x"
        self.encrypted_token = None


def _handler(responses):
    from crystal_cache.ingestion.git_handler import GitSourceHandler
    fake = _FakeGit(responses)
    return GitSourceHandler(http_get=fake), fake


@pytest.mark.asyncio
async def test_git_unchanged_head_is_one_cheap_call():
    h, fake = _handler({
        "/branches/master": {"commit": {"sha": "aaa"}},
    })
    w = _W({"repo": "EraHQ/crystal-cache-v2"}, last_state={"head": "aaa"})
    assert await h.check(w, None) is None
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_git_first_sync_walks_scoped_tree():
    h, _ = _handler({
        "/branches/master": {"commit": {"sha": "bbb"}},
        "/git/trees/bbb": {"tree": [
            {"type": "blob", "path": "cost/emit.py"},
            {"type": "blob", "path": "cost/pricing.py"},
            {"type": "blob", "path": "node_modules/x.js"},
            {"type": "blob", "path": ".github/ci.yml"},
            {"type": "blob", "path": "logo.png"},
            {"type": "tree", "path": "cost"},
        ]},
    })
    w = _W({"repo": "https://github.com/EraHQ/crystal-cache-v2"})
    cs = await h.check(w, None)
    assert cs.new_state == {"head": "bbb"}
    assert cs.changed == ["cost/emit.py", "cost/pricing.py"]
    assert cs.removed == []


@pytest.mark.asyncio
async def test_git_compare_maps_statuses_and_renames():
    h, _ = _handler({
        "/branches/master": {"commit": {"sha": "ccc"}},
        "/compare/aaa...ccc": {"files": [
            {"status": "added", "filename": "new.py"},
            {"status": "modified", "filename": "cost/emit.py"},
            {"status": "removed", "filename": "old.py"},
            {"status": "renamed", "filename": "renamed_to.py",
             "previous_filename": "renamed_from.py"},
            {"status": "modified", "filename": "assets/logo.png"},
        ]},
    })
    w = _W({"repo": "EraHQ/x"}, last_state={"head": "aaa"})
    cs = await h.check(w, None)
    assert cs.changed == ["new.py", "cost/emit.py", "renamed_to.py"]
    assert cs.removed == ["old.py", "renamed_from.py"]


@pytest.mark.asyncio
async def test_git_include_exclude_globs():
    h, _ = _handler({
        "/branches/main": {"commit": {"sha": "ddd"}},
        "/git/trees/ddd": {"tree": [
            {"type": "blob", "path": "src/a.py"},
            {"type": "blob", "path": "docs/b.md"},
            {"type": "blob", "path": "src/vendor/c.py"},
        ]},
    })
    w = _W({"repo": "o/n", "branch": "main",
            "include": ["src/**"], "exclude": ["src/vendor/**"]})
    cs = await h.check(w, None)
    assert cs.changed == ["src/a.py"]


@pytest.mark.asyncio
async def test_git_fetch_builds_d6_identity():
    import base64 as b64
    h, _ = _handler({
        "/contents/cost/emit.py": {
            "content": b64.b64encode(b"import json\n").decode(),
        },
    })
    w = _W({"repo": "EraHQ/x"}, last_state={"head": "eee"},
           source_name="crystal-cache-v2")
    env = await h.fetch(w, "cost/emit.py", None)
    assert env.payload_bytes == b"import json\n"
    assert env.source_uri == "repo://crystal-cache-v2/cost/emit.py"
    assert env.label == "crystal-cache-v2/cost/emit.py"
    assert env.mime_type.startswith("text/")


def test_git_repo_slug_shapes():
    from crystal_cache.ingestion.git_handler import _repo_slug
    assert _repo_slug({"repo": "EraHQ/crystal-cache-v2"}) == "EraHQ/crystal-cache-v2"
    assert _repo_slug({"repo": "https://github.com/EraHQ/crystal-cache-v2"}) == "EraHQ/crystal-cache-v2"
    assert _repo_slug({"repo": "https://github.com/EraHQ/crystal-cache-v2.git"}) == "EraHQ/crystal-cache-v2"
    with pytest.raises(ValueError):
        _repo_slug({"repo": "just-a-name"})
