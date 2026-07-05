"""Phase 11.5 smoke tests — CLI + HTTP entry points.

Per P0.103 (PRD-7): two smoke tests that catch silent regressions
in the substrate review surface's entry points.

These are SMOKE tests, not coverage tests. They verify the entry
points exist and produce structurally valid output. Real behavior
is covered by `tests/test_phase10_5_substrate_review.py`
(library-function tests).

Tests:
  SS1 — Invoking `cli.substrate_review.main(["--help"])` exits
       cleanly (SystemExit(0)) with help text containing 'substrate'.
       Catches regressions in argparse wiring.
       Direct-call approach (not subprocess) avoids depending on
       venv install state — argparse wiring is the actual regression
       surface; the `python -m` entry point is rock-solid via the
       `if __name__ == "__main__"` guard.

  SS2 — httpx.AsyncClient + ASGITransport against a minimal FastAPI
       app with the metacog router mounted + the store dependency
       overridden to point at the test fixture's in-memory store.
       Returns 200 with the expected JSON shape on a nonexistent
       customer query.
       Avoids TestClient + full-app lifespan (which would spawn
       workers); the smoke test's concern is the endpoint contract,
       not the full app boot.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# SS1 — CLI smoke (direct main() call)
# ---------------------------------------------------------------------------

def test_ss1_cli_help_exits_clean(capsys):
    """PRD-7 / P0.103 — `cli.substrate_review.main(["--help"])` exits
    via SystemExit(0) and prints help text containing 'substrate'.

    argparse's `--help` always raises SystemExit; the exit code is
    0 on success. Direct invocation is the right regression surface:
    if someone breaks the argparse setup or renames the description,
    this test catches it. The `python -m` path is a guarantee of
    the language, not application code.
    """
    from crystal_cache.cli.substrate_review import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0, (
        f"--help should exit 0; got {exc_info.value.code}"
    )

    captured = capsys.readouterr()
    assert "substrate" in captured.out.lower(), (
        f"Help output missing 'substrate' keyword; got: {captured.out!r}"
    )


# ---------------------------------------------------------------------------
# SS2 — HTTP smoke (mini-app + httpx.AsyncClient)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ss2_http_endpoint_returns_empty_on_empty_db(store):
    """PRD-7 / P0.103 — HTTP endpoint returns 200 with the expected
    JSON shape on a nonexistent customer query.

    Builds a minimal FastAPI app with ONLY the metacog router
    mounted. Overrides the `get_metadata_store` dependency to return
    the test fixture's store (in-memory SQLite, empty). This
    sidesteps the full app.py lifespan (which would attempt to
    construct encoders, spawn workers, etc.) — for a smoke test,
    that's the wrong shape.

    Uses httpx.AsyncClient + ASGITransport for native async testing;
    avoids the sync TestClient + nested-event-loop friction.

    Catches:
      - Route registration regressions (someone removes
        @router.get(...) or changes the path).
      - Endpoint parameter handling (the limit / since /
        customer_id query params).
      - JSON response shape regressions (key names change, response
        becomes a list instead of an object, etc.).
    """
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from crystal_cache.infrastructure.metadata_store import (
        get_metadata_store,
    )
    from crystal_cache.metacognition import api as metacog_api

    # Build a minimal app — just the router + dependency override.
    mini_app = FastAPI()
    mini_app.include_router(metacog_api.router)

    async def _get_test_store():
        return store

    mini_app.dependency_overrides[get_metadata_store] = _get_test_store

    # Hit the endpoint via httpx.AsyncClient + ASGITransport
    # (native-async path; no nested event loop).
    async with AsyncClient(
        transport=ASGITransport(app=mini_app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/admin/api/metacognition/substrate-observations",
            params={"customer_id": "nonexistent_smoke_test_cust"},
        )

    assert response.status_code == 200, (
        f"HTTP endpoint returned {response.status_code}; "
        f"body: {response.text!r}"
    )

    body = response.json()
    assert "total" in body, f"Response missing 'total' key: {body!r}"
    assert "observations" in body, (
        f"Response missing 'observations' key: {body!r}"
    )
    assert body["total"] == 0, (
        f"Expected total=0 for nonexistent customer; got {body['total']}"
    )
    assert body["observations"] == [], (
        f"Expected empty observations list; got {body['observations']!r}"
    )
