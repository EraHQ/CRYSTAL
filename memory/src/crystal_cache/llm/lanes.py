"""Lane classes and the tenant context the shared LLM client reads.

Q74=A / Q87 (2026-10-17): a tenant picks two models in Settings, one
for ANSWERS (chat, every connected tool, scheduled briefs; the agent
turn already carries its model explicitly) and one for BACKGROUND
CURATION (research and composition, scheduled research, extraction,
and the curation judgments: contradiction, dedup, gap discovery,
assumptions, conflict explanation, outbound review). INFRA lanes
(sparse keys, the shadow and self critics, structural scans,
metacognition) stay on the platform's own tier map and never read the
tenant's choice.

The mechanism is two context variables, set once where a request or a
claimed task becomes tenant work, and read by LLMClient._resolve_model:
inside `tenant_lane(customer, "background")` a tier resolves to the
tenant's background model (its family's light model for the small
tier); everywhere else resolution is unchanged. Nothing is threaded
through the 53 singleton call sites.

Two families resolve here: "claude" (the Anthropic singleton) and, since
M3 (2026-10-19), "nemotron" (NVIDIA Nemotron on Nebius Token Factory, a
second OpenAI-compatible client). The provider switch lives in
client.get_llm_client(): inside a background lane whose pick is a
Nemotron id it returns the Nemotron client when the lane is enabled
(the key is set); each client then only honours picks of its own family,
so a Nemotron pick on the Anthropic singleton falls through to the
platform default instead of sending a Nemotron id to Anthropic.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Literal, Optional

Lane = Literal["answers", "background", "infra"]
LANES: tuple[str, ...] = ("answers", "background", "infra")

# The light model each family runs its small-tier (quick, repetitive)
# steps on when the tenant's background pick is a bigger sibling. The
# Nemotron entry is configured (the ids are verified against Token
# Factory's /v1/models, not hard-coded), so read it through family_small().
FAMILY_SMALL: dict[str, str] = {
    "claude": "claude-haiku-5-5",
}


def family_small(family: Optional[str]) -> Optional[str]:
    if family == "nemotron":
        from ..config import settings

        return settings.nemotron_model_small
    return FAMILY_SMALL.get(family or "")


def family_of(model_id: Optional[str]) -> Optional[str]:
    """'claude' for an Anthropic id, 'nemotron' for an NVIDIA id on Token
    Factory (nvidia/...), None for anything this module does not route."""
    if not model_id:
        return None
    if model_id.startswith("claude-"):
        return "claude"
    if model_id.lower().startswith("nvidia/"):
        return "nemotron"
    return None


def nemotron_enabled() -> bool:
    """The lane exists only while the platform holds a Token Factory key."""
    from ..config import settings

    return bool((settings.nebius_api_key or "").strip())


def nemotron_models() -> tuple[str, ...]:
    """The configured Nemotron ids (small, large, frontier), deduplicated,
    in that order. Empty when the lane is off."""
    if not nemotron_enabled():
        return ()
    from ..config import settings

    out: list[str] = []
    for m in (settings.nemotron_model_small, settings.nemotron_model_large, settings.nemotron_model_frontier):
        if m and m not in out:
            out.append(m)
    return tuple(out)


@dataclass(frozen=True)
class TenantModels:
    customer_id: str
    answers_model: Optional[str]
    background_model: Optional[str]


def tenant_models_from(customer) -> TenantModels:
    cfg = getattr(customer, "model_routing_config", None)
    return TenantModels(
        customer_id=getattr(customer, "id", ""),
        answers_model=getattr(cfg, "model_id", None),
        background_model=getattr(cfg, "background_model_id", None),
    )


_tenant: ContextVar[Optional[TenantModels]] = ContextVar("cc_tenant_models", default=None)
_lane: ContextVar[str] = ContextVar("cc_lane", default="infra")


def current_tenant() -> Optional[TenantModels]:
    return _tenant.get()


def current_lane() -> str:
    return _lane.get()


@contextmanager
def tenant_lane(customer, lane: Lane) -> Iterator[None]:
    """Bind a tenant and a lane for the calls made inside the block; both
    are restored on exit, so a worker loop over many tenants never leaks
    one tenant's choice into the next."""
    if lane not in LANES:
        raise ValueError(f"unknown lane {lane!r}; use one of {LANES}")
    t_tok = _tenant.set(tenant_models_from(customer) if customer is not None else None)
    l_tok = _lane.set(lane)
    try:
        yield
    finally:
        _lane.reset(l_tok)
        _tenant.reset(t_tok)


def lane_pick() -> Optional[str]:
    """The bound tenant's background pick when a background lane is
    active and the pick belongs to a family this module routes; else None."""
    if _lane.get() != "background":
        return None
    t = _tenant.get()
    if t is None or not t.background_model:
        return None
    return t.background_model if family_of(t.background_model) else None


def lane_model_for_tier(tier: str) -> Optional[str]:
    """The model the bound tenant's lane asks for at `tier`, or None when
    resolution should fall through to the platform map: no tenant bound,
    an infra or answers lane, no background pick, or a family this
    module does not route."""
    pick = lane_pick()
    if pick is None:
        return None
    if tier == "small":
        return family_small(family_of(pick)) or pick
    return pick
