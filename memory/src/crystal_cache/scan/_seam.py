"""One metered small-tier seam call, shared by the discriminator scans
(G3 metering at the seam boundary, 2026-07-02).

The three convergence scans (contradiction, dedup, gap discovery) each make
bounded fast-model calls in idle cycles — and they default ON since the
flag-stance pass, so their spend must be visible in the cost ledger the
same as every other call path. This helper is the one place the scans turn
a seam call into (text + one llm_calls row via cost/emit.record_model_call,
origin-tagged per scan).

Compatibility: when the injected client exposes only `complete` (legacy
test fakes), the call runs UNMETERED but otherwise identical — behavior
for existing tests is unchanged. Real seam clients expose
`complete_detailed` and are metered. Exceptions propagate so each scan's
existing fail-safe (ERROR / no-row) keeps working.
"""
from __future__ import annotations

import asyncio
import functools
from typing import Any, Optional

from ..cost.emit import record_model_call


async def metered_call(
    client: Any,
    *,
    customer_id: str,
    origin: str,
    system: str,
    user: str,
    max_tokens: int,
    tier: str = "small",
    store: Any = None,
) -> Optional[str]:
    """Run one metered completion at an explicit tier; emit its cost row.

    Tier-general seam (2026-07-03): the outbound review scan needs the
    HIGH tier (ratified: background-worker output is reviewed by a
    high-tier model), while the discriminator scans stay small. Same
    metering + legacy-client compatibility as metered_small_call, which
    now delegates here.
    """
    kwargs = dict(
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=0.0,
        tier=tier,
    )
    detailed = getattr(client, "complete_detailed", None)
    if detailed is None:
        # Legacy/test clients exposing only complete(): unmetered but
        # functionally identical.
        return await asyncio.to_thread(functools.partial(client.complete, **kwargs))

    result = await asyncio.to_thread(functools.partial(detailed, **kwargs))
    await record_model_call(
        customer_id=customer_id,
        model=result.model,
        origin=origin,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cache_creation_tokens=result.cache_creation_tokens,
        cache_read_tokens=result.cache_read_tokens,
        store=store,
    )
    return result.text


async def metered_small_call(
    client: Any,
    *,
    customer_id: str,
    origin: str,
    system: str,
    user: str,
    max_tokens: int,
    store: Any = None,
) -> Optional[str]:
    """Run one small-tier completion off the event loop; emit its cost row.

    Returns the completion text (None-safe: whatever the client returned).
    Delegates to metered_call with tier='small' (the discriminator scans'
    tier); kept as the named entrypoint so scan call sites read plainly.
    """
    return await metered_call(
        client,
        customer_id=customer_id,
        origin=origin,
        system=system,
        user=user,
        max_tokens=max_tokens,
        tier="small",
        store=store,
    )
