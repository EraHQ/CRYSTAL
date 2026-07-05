"""HTTP route handlers — Phase 6 of the v2 port.

Each module exports `router: APIRouter` registered in app.py.

Submodules (Phase 6 Wave B + Phase 6.5):
  - health, customers, chat_proxy, agent, feedback, sdk,
    documents, drive, compliance, admin (Wave B)
  - openai_compat, stubs, diagnostics, dsl_configs (Phase 6.5)

Per the agent reframe in PROJECT_LEDGER.md, chat_proxy.py and agent.py
are peers: customers can hit either depending on which deployment mode
they're using.

The Phase 6.5 modules restore v1 endpoints Wave B initially dropped:
  - openai_compat: /v1/models, /v1/completions (OpenAI SDK surface)
  - stubs: 10 inspector-tier 501 stubs from v1
  - diagnostics: /api/crystals, /api/crystals/{id}/diagnostic, /api/edits
  - dsl_configs: /api/dsl_configs CRUD with synchronous compile validation
"""
from fastapi.responses import JSONResponse


def not_implemented(feature: str, doc_ref: str) -> JSONResponse:
    """Return a structured 501 response.

    Used by endpoints that exist in v1 but aren't ported yet, AND by
    the agent endpoint stub (Phase 7.5 fills it in).

    Defined here in endpoints/__init__ rather than in app.py to avoid
    a circular import: endpoint modules (e.g. agent.py) need this
    helper but app.py imports from endpoints/, so endpoint modules
    cannot import from app.py.
    """
    return JSONResponse(
        status_code=501,
        content={
            "error": {
                "type": "not_implemented",
                "feature": feature,
                "message": f"'{feature}' is scaffolded but not yet implemented.",
                "doc_ref": doc_ref,
            }
        },
    )
