"""Gate D4a (2026-07-17): crystal operations — the auth allowlist.

The live-smoke 401 that motivated these: the endpoints existed but the
tenant write allowlist ("a new write is platform-only until someone
consciously adds it here") had no entry — the gate worked, the build
was incomplete. These pin the conscious add.
"""
from crystal_cache.ingress.auth import _tenant_writable


def test_tier_patch_is_tenant_writable():
    assert _tenant_writable("PATCH", "/admin/api/crystals/crys_abc123/tier")
    assert _tenant_writable("PATCH", "/admin/api/crystals/crys_abc123/tier/")


def test_crystal_delete_is_tenant_writable():
    assert _tenant_writable("DELETE", "/admin/api/crystals/crys_abc123")
    assert _tenant_writable("DELETE", "/admin/api/crystals/crys_abc123/")


def test_delete_scope_is_crystal_root_only():
    # Fact paths, tier paths, collections: never deletable by tenants.
    assert not _tenant_writable(
        "DELETE", "/admin/api/crystals/crys_a/facts/fact_b")
    assert not _tenant_writable("DELETE", "/admin/api/crystals/crys_a/tier")
    assert not _tenant_writable("DELETE", "/admin/api/crystals")
    assert not _tenant_writable("DELETE", "/admin/api/customers/cus_a")


def test_write_methods_are_the_designed_set():
    # The middleware admits write-shaped methods (POST/PUT/PATCH share
    # one allowlist by design — the router enforces the exact verb, a
    # POST to /tier gets 405 there). PUT joined the set 2026-07-23 for
    # Gate G3's mapping replacement. Read-shaped methods never pass.
    assert _tenant_writable("PUT", "/admin/api/crystals/crys_a/tier")
    assert not _tenant_writable("GET", "/admin/api/crystals/crys_a/tier")
    assert not _tenant_writable("HEAD", "/admin/api/crystals/crys_a/tier")


# --- Gate M slice 5: watch routes — the conscious add, proactive ------------

def test_watch_routes_are_tenant_writable():
    assert _tenant_writable("POST", "/admin/api/watches")
    assert _tenant_writable("PATCH", "/admin/api/watches/watch_abc")
    assert _tenant_writable("DELETE", "/admin/api/watches/watch_abc")


def test_watch_delete_scope_is_single_watch():
    assert not _tenant_writable("DELETE", "/admin/api/watches")
    assert not _tenant_writable("DELETE", "/admin/api/watches/w/extra")
