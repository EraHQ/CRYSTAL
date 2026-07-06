"""Cost-emit helper — the one place the model-call sites turn a response into
an `llm_calls` row (Growth G3 / WS D D.3).

`record_model_call` is the shared emitter the proxy, agent loop, cognition
roles, and depth synthesis all funnel through. It exists so those four sites
don't each re-implement the same flag-check + price-table + fail-safe
boilerplate around `MetadataStore.record_llm_call` (the SQL sink lives there;
this is the thin call-path glue).

Contract:
  - Flag-gated on `enable_cost_accounting` — OFF by default, so when cost
    accounting is disabled this returns immediately and touches nothing (not
    even the store singleton), keeping it inert in tests and dev.
  - FULLY fail-safe — any error (including a missing store singleton or a bad
    price config) is swallowed with a `cost.record_failed` log. Cost telemetry
    must never break a model-call path.
  - Token source — pass an Anthropic `usage` object (preferred; cache tokens
    are read off it via the `cache_creation_input_tokens` /
    `cache_read_input_tokens` names) OR explicit counts. The proxy's upstream
    usage is OpenAI-shaped (no cache fields) so it passes input/output only;
    the agent path passes the real `usage` and so meters cache tokens too.
  - Attribution — `origin` tags the source (interactive / agent / cognition /
    depth / …); `session_id` buckets per-agent rollups; `operator_id` when the
    path has one.
  - Store — defaults to the process `MetadataStore` singleton (resolved lazily
    AFTER the flag check); tests inject a fake.
"""
from __future__ import annotations

from typing import Any, Optional

import structlog

from ..config import get_settings
from ..infrastructure.metadata_store import get_metadata_store
from .pricing import price_table_from_settings

logger = structlog.get_logger(__name__)


def _usage_tokens(usage: Any) -> tuple[int, int, int, int]:
    """Extract (input, output, cache_creation, cache_read) token counts from an
    Anthropic usage object, tolerant of missing fields. Anthropic names the
    cache fields `cache_creation_input_tokens` / `cache_read_input_tokens`."""
    if usage is None:
        return (0, 0, 0, 0)
    return (
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
        int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        int(getattr(usage, "cache_read_input_tokens", 0) or 0),
    )


async def record_model_call(
    *,
    customer_id: str,
    model: str,
    origin: str,
    usage: Any = None,
    session_id: Optional[str] = None,
    parent_session_id: Optional[str] = None,
    operator_id: Optional[str] = None,
    store: Any = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    cache_creation_tokens: Optional[int] = None,
    cache_read_tokens: Optional[int] = None,
    billing: Optional[str] = None,
) -> None:
    """Emit one `llm_calls` cost row for a model invocation (G3 / D.3).

    Pass either an Anthropic ``usage`` object (preferred) or explicit token
    counts. No-op when ``enable_cost_accounting`` is off. Never raises — a
    failure is logged as ``cost.record_failed`` and swallowed.
    """
    settings = get_settings()
    if not getattr(settings, "enable_cost_accounting", False):
        return
    try:
        if usage is not None:
            in_t, out_t, cc_t, cr_t = _usage_tokens(usage)
        else:
            in_t = int(input_tokens or 0)
            out_t = int(output_tokens or 0)
            cc_t = int(cache_creation_tokens or 0)
            cr_t = int(cache_read_tokens or 0)
        st = store if store is not None else get_metadata_store()
        await st.record_llm_call(
            customer_id,
            model=model,
            input_tokens=in_t,
            output_tokens=out_t,
            cache_creation_tokens=cc_t,
            cache_read_tokens=cr_t,
            session_id=session_id,
            parent_session_id=parent_session_id,
            operator_id=operator_id,
            origin=origin,
            billing=billing,
            price_table=price_table_from_settings(
                getattr(settings, "llm_price_table_overrides", None)
            ),
        )
    except Exception as e:  # noqa: BLE001 — cost telemetry never breaks the call path
        logger.warning(
            "cost.record_failed", origin=origin, model=model, error=str(e)
        )
