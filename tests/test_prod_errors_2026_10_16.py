"""The two real errors in three days of production logs (2026-10-16),
each pinned at its source.

1. GET /admin/.env, /admin/.git/HEAD and friends (scanner probes) answered
   503 "Inspector UI not built" from the hosted API image, whose console
   is a separate service. A 5xx on a junk path is wrong, Cloud Run logs
   it as ERROR, and the body named the bundle path. Now: nothing is
   mounted without a bundle, so the router's 404 envelope answers.
2. crystal-cognition: DBAPIError "connection was closed in the middle of
   operation" inside gap_discovery, on the SELECT of a tenant's ENTIRE
   fact table (vectors included). The idle-pass scans never passed the
   limit that list_recent_facts_for_customer was designed around. Now
   every idle-pass read is bounded by CC_SCAN_FACT_FETCH_LIMIT (2000,
   the pairing funnel's number).
"""
from __future__ import annotations

import inspect

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_admin_junk_paths_never_answer_503():
    from crystal_cache import app as app_mod

    # No route under /admin returns a 503 by design: the only /admin
    # catch-all allowed is the self-host SPA fallback (FileResponse).
    for route in app_mod.app.routes:
        if getattr(route, "path", "") in ("/admin", "/admin/{full_path:path}"):
            assert route.endpoint.__name__ == "serve_admin_spa", route.endpoint.__name__
    async with AsyncClient(transport=ASGITransport(app=app_mod.app, raise_app_exceptions=False),
                           base_url="http://t") as c:
        for path in ("/admin/.env", "/admin/.git/HEAD", "/admin/config/.env.prod"):
            r = await c.get(path)
            # 404 (hosted image: no bundle) or 200 (self-host SPA fallback
            # on a checkout that built the console); never a 5xx.
            assert r.status_code in (200, 404), (path, r.status_code, r.text[:200])
            if r.status_code == 404:
                assert r.json()["error"]["type"] == "not_found_error"
            assert "frontend/dist" not in r.text and "dist" not in r.headers.get("x-expected-path", "")


def test_every_idle_pass_facts_read_is_bounded():
    from crystal_cache import config
    from crystal_cache.scan import assumptions
    from crystal_cache.workers import cognition

    assert config.Settings().scan_fact_fetch_limit == 2000
    loop = inspect.getsource(cognition.run_cognition_worker)
    assert loop.count("fact_fetch_limit=settings.scan_fact_fetch_limit") == 3
    for helper in (cognition._run_contradiction_scan, cognition._run_dedup_scan, cognition._run_gap_discovery):
        src = inspect.getsource(helper)
        assert "fact_fetch_limit: Optional[int] = None" in src and "fact_fetch_limit=fact_fetch_limit" in src, helper.__name__
    assert "limit=get_settings().scan_fact_fetch_limit" in inspect.getsource(assumptions._gap_seeded_pairs)
    # The store read itself honors the bound (newest first, then the cap).
    from crystal_cache.infrastructure import metadata_store_cognition_ext as ext

    src = inspect.getsource(ext.CognitionExtensionsMixin.list_recent_facts_for_customer)
    assert "stmt = stmt.limit(limit)" in src
