"""Customer entity — §6 of BUILD_PROPOSAL.md."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


InjectionPreference = Literal["text", "hidden_state", "none"]


class ModelRoutingConfig(BaseModel):
    """Per-customer upstream model routing."""
    provider: str  # "openai" | "anthropic" | "self_hosted" | ...
    model_id: str
    api_key_ref: str  # Key B — stored AES-256-GCM encrypted (enc:v1 composite; token_crypto)
    # Only used for self_hosted providers (vLLM, Ollama, etc.)
    base_url: Optional[str] = None
    # Customer-controlled means we can do hidden-state injection.
    customer_controlled: bool = False


class RetrievalThresholds(BaseModel):
    """Per-customer tuning of the match classifier.

    Two-tier (legacy) thresholds drive the high/medium/low classifier:
      high      — minimum top-1 cosine to call this a strong match
      medium    — minimum top-1 cosine to inject context
      below medium      — "low", pass through unchanged

    Four-way (April 2026) routing decision adds two more knobs that drive
    the perfect/spread/low/no-match classifier defined in CLAUDE.md's
    routing decision table:
      perfect_margin   — minimum (top1 − top2) gap for a 'perfect' route.
                          Above this we trust top-1 alone.
      spread_margin    — minimum (top1 − top2) gap for a 'spread' route.
                          Below 'perfect' but above this means top-2 are
                          plausibly both relevant — invoke bind synthesis.
                          Below this is 'low confidence' — do not synthesize.
      noise_floor      — cosine value at or below which we treat top-1 as
                          random. If top-1 sits below this, decision is
                          'no match', regardless of margins.

    Defaults are calibrated for the SEMANTIC encoder (gtr-t5-base +
    P-projection). Hash-encoder banks need different numbers — their
    cosines run higher because token-overlap inflates the dot product.
    See CLAUDE.md §"When to use which decoder" for derivation of these
    defaults from the v2 spike's measured distribution.

    These defaults give:
      cos ≥ 0.45                       → "high"
      0.20 ≤ cos < 0.45                → "medium"
      cos < 0.20                       → "low"

      top1 < noise_floor (0.05)        → NoMatch
      top1 − top2 ≥ perfect_margin     → Perfect (top-1 owns it)
      top1 − top2 ≥ spread_margin      → Spread (synthesize via bind-v1)
      otherwise                        → LowConfidence
    """
    high: float = Field(default=0.45, ge=0.0, le=1.0)
    medium: float = Field(default=0.20, ge=0.0, le=1.0)

    # Four-way routing thresholds. Optional in the sense that the legacy
    # high/medium classifier still works; these are consulted by the new
    # classify_routing() path.
    perfect_margin: float = Field(default=0.20, ge=0.0, le=1.0)
    spread_margin: float = Field(default=0.05, ge=0.0, le=1.0)
    noise_floor: float = Field(default=0.05, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_ordering(self) -> "RetrievalThresholds":
        if self.high < self.medium:
            raise ValueError(
                f"retrieval_thresholds.high ({self.high}) must be >= medium "
                f"({self.medium})"
            )
        if self.perfect_margin < self.spread_margin:
            raise ValueError(
                f"retrieval_thresholds.perfect_margin ({self.perfect_margin}) "
                f"must be >= spread_margin ({self.spread_margin})"
            )
        return self


class Customer(BaseModel):
    id: str
    # Raw Crystal Cache API key (Key A). Present ONLY on the object
    # returned at creation (shown once); None on every subsequent load —
    # the DB stores only a hash (`api_key_hash` on CustomerRow), never the
    # raw key. No plaintext at rest (2026-06-13).
    api_key: Optional[str] = None
    model_routing_config: ModelRoutingConfig
    injection_preference: InjectionPreference = "text"
    # Fraction of medium-match requests that also run a baseline (no-injection)
    # pass for telemetry. 0.0 disables shadow eval; 1.0 doubles all
    # medium-match compute.
    shadow_sample_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    # Thresholds for the match classifier. Defaults in RetrievalThresholds
    # docstring.
    retrieval_thresholds: RetrievalThresholds = Field(
        default_factory=RetrievalThresholds
    )
    # Phase 1.5.3: multi-turn routing window. Controls how many recent
    # user turns the encoder considers when building the query vector
    # for crystal routing. None = use system default (3). An explicit
    # int overrides (e.g. 1 = single-turn, 5 = wider context).
    # Routing-only — doesn't affect storage or upstream forwarding.
    routing_context_window: Optional[int] = None
    # Phase 12 (CU-27): per-customer override for the daily shadow-
    # critique cost cap. None = use the global default
    # (settings.shadow_max_per_customer_per_day). An explicit int caps
    # this customer's shadow critiques per rolling 24h window. Used by
    # the metacognition worker's shadow pass to bound per-customer
    # R&D spend. Mirrors routing_context_window's nullable-override.
    shadow_max_per_day: Optional[int] = None
    retention_policy: Optional[str] = None
    billing_config: Optional[str] = None
    # General crystal subscriptions. Which general crystal types this
    # customer receives during retrieval. Empty = no general crystals.
    general_crystal_types: list[str] = Field(
        default_factory=lambda: ["general:legacy"]
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
