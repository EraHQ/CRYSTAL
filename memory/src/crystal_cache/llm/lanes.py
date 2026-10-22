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
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Mapping, Optional

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
    """'claude' for an Anthropic id; 'nemotron' (the family id of the
    Token Factory lane) for an NVIDIA id (nvidia/...) or ANY id Token
    Factory's catalog lists (hackathon item 1, 2026-10-22: a tenant may
    pick any text model the platform's key can serve, metered at its
    listed price); None for anything this module does not route."""
    if not model_id:
        return None
    if model_id.startswith("claude-"):
        return "claude"
    if model_id.lower().startswith("nvidia/"):
        return "nemotron"
    if nemotron_enabled():
        from .catalog import catalog_ids

        if model_id.lower() in catalog_ids():
            return "nemotron"
    return None


def nemotron_enabled() -> bool:
    """The lane exists only while the platform holds a Token Factory key."""
    from ..config import settings

    return bool((settings.nebius_api_key or "").strip())


def nemotron_models() -> tuple[str, ...]:
    """The configured Nemotron ids (small, large, frontier), deduplicated,
    in that order, then every other TEXT model Token Factory's catalog
    lists (hackathon item 1). Empty when the lane is off."""
    if not nemotron_enabled():
        return ()
    from ..config import settings

    out: list[str] = []
    for m in (settings.nemotron_model_small, settings.nemotron_model_large, settings.nemotron_model_frontier):
        if m and m not in out:
            out.append(m)
    from .catalog import text_models

    seen = {m.lower() for m in out}
    for cm in text_models():
        if cm.id.lower() not in seen:
            out.append(cm.id)
            seen.add(cm.id.lower())
    return tuple(out)


def family_frontier(family: Optional[str], pick: Optional[str]) -> Optional[str]:
    """Hackathon item 3: the model a family's FRONTIER tier runs on under
    a background pick. Nemotron picks (nvidia/...) lift to the configured
    frontier (Ultra); any other Token Factory model keeps the pick (the
    tenant chose that vendor on purpose); Claude keeps the pick."""
    if family == "nemotron" and pick and pick.lower().startswith("nvidia/"):
        from ..config import settings

        return settings.nemotron_model_frontier or pick
    return pick


@dataclass(frozen=True)
class TenantModels:
    customer_id: str
    answers_model: Optional[str]
    background_model: Optional[str]
    # Hackathon item 1: {model id (lower): {"json_schema": bool, "tools": bool, ...}}
    # from the probe run when the tenant picked the model (llm/catalog.probe_model).
    caps: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def tenant_models_from(customer) -> TenantModels:
    cfg = getattr(customer, "model_routing_config", None)
    raw_caps = getattr(cfg, "model_caps", None) or {}
    return TenantModels(
        customer_id=getattr(customer, "id", ""),
        answers_model=getattr(cfg, "model_id", None),
        background_model=getattr(cfg, "background_model_id", None),
        caps={str(k).lower(): dict(v) for k, v in raw_caps.items() if isinstance(v, dict)},
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
    module does not route. Small lifts down to the family's light model,
    frontier lifts up to the family's frontier (item 3), large is the pick."""
    pick = lane_pick()
    if pick is None:
        return None
    fam = family_of(pick)
    if tier == "small":
        return family_small(fam) or pick
    if tier == "frontier":
        return family_frontier(fam, pick) or pick
    return pick


def lane_caps(model_id: Optional[str]) -> Optional[Mapping[str, Any]]:
    """The bound tenant's probe record for `model_id`, or None when there is
    none (no tenant bound, or the model was never probed for this tenant)."""
    t = _tenant.get()
    if t is None or not model_id:
        return None
    return t.caps.get(model_id.lower())
