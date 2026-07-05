"""Per-model price table + cost arithmetic (Growth G3). PURE — no DB, no I/O.

**Money is INTEGER micro-USD (1e-6 USD) everywhere.** Prices are stored as
*micro-USD per million tokens* (i.e. a $3.00 / Mtok rate is 3_000_000), so a
cost is `tokens * micro_per_mtok // 1_000_000` — integer arithmetic with no
float drift.

RATES VERIFIED 2026-07-02 against platform.claude.com/docs pricing +
claude.com/pricing (Anthropic rows) — see the table comments. UPDATE
PRACTICE: re-verify rates whenever a new model is adopted anywhere in the
config (tiers, agent model, upstream defaults); operators can correct rates
without a release via CC_LLM_PRICE_TABLE_OVERRIDES
(`settings.llm_price_table_overrides` → `price_table_from_settings`). All
surfaced figures remain ESTIMATES — vendors add multipliers (regional
routing, batch, fast modes) this table does not model.

Lookup is exact-match first, then LONGEST-PREFIX match, so dated model
strings (claude-haiku-4-5-20251001) hit their family row and vendors'
version suffixes don't silently zero a cost.

The recording mixin (infrastructure/metadata_store_cost_ext.py) calls
`compute_cost_micro_usd` and persists the integer it returns. An unknown model
returns 0 (and the mixin logs it) rather than raising — observability must
never break the call path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# 1 USD = 1_000_000 micro-USD. Prices are micro-USD per 1e6 tokens.
MICRO_PER_USD = 1_000_000
_TOKENS_PER_MTOK = 1_000_000


@dataclass(frozen=True)
class ModelPrice:
    """A model's per-million-token rates, in micro-USD. All integers."""
    input_micro_per_mtok: int
    output_micro_per_mtok: int
    cache_creation_micro_per_mtok: int = 0
    cache_read_micro_per_mtok: int = 0


def _usd_per_mtok(usd: float) -> int:
    """Convert a human USD/Mtok rate to integer micro-USD/Mtok (used only to
    author the placeholder table below; stored money is never a float)."""
    return int(round(usd * MICRO_PER_USD))


# Verified rates (USD/Mtok → micro), 2026-07-02, per Anthropic's pricing
# docs. Cache multipliers per the same docs: 5m cache write = 1.25× input,
# cache hit = 0.1× input. Keys are FAMILY PREFIXES (longest-prefix lookup
# matches dated variants). Non-Anthropic rows are upstream-proxy estimate
# rates for models customers route through the proxy (stable published
# rates; override via CC_LLM_PRICE_TABLE_OVERRIDES if they move).
DEFAULT_PRICE_TABLE: dict[str, ModelPrice] = {
    # --- Anthropic, current generation (verified 2026-07-02) ---
    "claude-opus-4-8": ModelPrice(
        _usd_per_mtok(5.0), _usd_per_mtok(25.0),
        _usd_per_mtok(6.25), _usd_per_mtok(0.5),
    ),
    "claude-opus-4-7": ModelPrice(
        _usd_per_mtok(5.0), _usd_per_mtok(25.0),
        _usd_per_mtok(6.25), _usd_per_mtok(0.5),
    ),
    # Sonnet 5 STANDARD rate ($3/$15, effective 2026-09-01). Intro pricing
    # through 2026-08-31 is $2/$10 — encoded conservatively at standard;
    # set the intro rate via overrides if precision matters this summer.
    "claude-sonnet-5": ModelPrice(
        _usd_per_mtok(3.0), _usd_per_mtok(15.0),
        _usd_per_mtok(3.75), _usd_per_mtok(0.3),
    ),
    "claude-sonnet-4-6": ModelPrice(
        _usd_per_mtok(3.0), _usd_per_mtok(15.0),
        _usd_per_mtok(3.75), _usd_per_mtok(0.3),
    ),
    "claude-haiku-4-5": ModelPrice(
        _usd_per_mtok(1.0), _usd_per_mtok(5.0),
        _usd_per_mtok(1.25), _usd_per_mtok(0.1),
    ),
    "claude-fable-5": ModelPrice(
        _usd_per_mtok(10.0), _usd_per_mtok(50.0),
        _usd_per_mtok(12.5), _usd_per_mtok(1.0),
    ),
    "claude-mythos-5": ModelPrice(
        _usd_per_mtok(10.0), _usd_per_mtok(50.0),
        _usd_per_mtok(12.5), _usd_per_mtok(1.0),
    ),
    # --- Anthropic, previous generation still routable ---
    "claude-sonnet-4-5": ModelPrice(
        _usd_per_mtok(3.0), _usd_per_mtok(15.0),
        _usd_per_mtok(3.75), _usd_per_mtok(0.3),
    ),
    # --- Upstream-proxy estimate rates (non-Anthropic) ---
    "gpt-4o-mini": ModelPrice(_usd_per_mtok(0.15), _usd_per_mtok(0.60)),
    "gpt-4o": ModelPrice(_usd_per_mtok(2.50), _usd_per_mtok(10.0)),
    "gpt-4.1-mini": ModelPrice(_usd_per_mtok(0.40), _usd_per_mtok(1.60)),
    "gpt-4.1-nano": ModelPrice(_usd_per_mtok(0.10), _usd_per_mtok(0.40)),
    "gpt-4.1": ModelPrice(_usd_per_mtok(2.0), _usd_per_mtok(8.0)),
    "gemini-2.0-flash": ModelPrice(_usd_per_mtok(0.075), _usd_per_mtok(0.30)),
    "gemini-2.5-flash": ModelPrice(_usd_per_mtok(0.15), _usd_per_mtok(0.60)),
    "gemini-2.5-pro": ModelPrice(_usd_per_mtok(1.25), _usd_per_mtok(10.0)),
}


def resolve_model_price(
    model: str,
    price_table: Optional[dict[str, ModelPrice]] = None,
) -> Optional[ModelPrice]:
    """Exact-match first, then longest-prefix match (case-insensitive).

    Dated variants (claude-haiku-4-5-20251001) hit their family row;
    unknown models return None so callers keep their own semantics
    (compute → 0, estimate → None).
    """
    table = price_table if price_table is not None else DEFAULT_PRICE_TABLE
    if model in table:
        return table[model]
    m = model.lower()
    best_key = None
    for key in table:
        if m.startswith(key.lower()) or key.lower() in m:
            if best_key is None or len(key) > len(best_key):
                best_key = key
    return table[best_key] if best_key else None


def compute_cost_micro_usd(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    price_table: Optional[dict[str, ModelPrice]] = None,
) -> int:
    """Cost of one model call in INTEGER micro-USD.

    Unknown model → 0 (caller logs). Integer arithmetic throughout:
    `tokens * micro_per_mtok // 1_000_000`. Negative token counts are clamped
    to 0 (a malformed usage block can't produce a negative charge).
    """
    table = price_table if price_table is not None else DEFAULT_PRICE_TABLE
    price = resolve_model_price(model, table)
    if price is None:
        return 0

    def _clamp(n: int) -> int:
        return n if n > 0 else 0

    total = (
        _clamp(input_tokens) * price.input_micro_per_mtok
        + _clamp(output_tokens) * price.output_micro_per_mtok
        + _clamp(cache_creation_tokens) * price.cache_creation_micro_per_mtok
        + _clamp(cache_read_tokens) * price.cache_read_micro_per_mtok
    )
    return int(total // _TOKENS_PER_MTOK)


def estimate_cost_usd(
    model: str,
    *,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    price_table: Optional[dict[str, ModelPrice]] = None,
) -> Optional[float]:
    """Float-USD ESTIMATE for log/display surfaces (the chat proxy's
    per-turn line). None when tokens are missing or the model is unknown —
    display surfaces show nothing rather than a fabricated zero. Ledger
    surfaces use compute_cost_micro_usd; this is presentation only."""
    if prompt_tokens is None or completion_tokens is None:
        return None
    price = resolve_model_price(model, price_table)
    if price is None:
        return None
    micro = compute_cost_micro_usd(
        model,
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        price_table=price_table,
    )
    return round(micro / MICRO_PER_USD, 6)


def price_table_from_settings(
    raw: Optional[dict[str, Any]],
) -> dict[str, ModelPrice]:
    """Build a price table from a config override, merged over the defaults.

    `raw` is a plain dict model → {input, output, cache_creation?, cache_read?}
    where each value is micro-USD per Mtok (integer; the same unit the table
    stores). Missing models fall back to DEFAULT_PRICE_TABLE; a malformed
    entry is skipped (defaults win) rather than raising — a bad price config
    must not break cost recording.
    """
    table = dict(DEFAULT_PRICE_TABLE)
    if not raw:
        return table
    for model, spec in raw.items():
        try:
            table[model] = ModelPrice(
                input_micro_per_mtok=int(spec["input"]),
                output_micro_per_mtok=int(spec["output"]),
                cache_creation_micro_per_mtok=int(spec.get("cache_creation", 0)),
                cache_read_micro_per_mtok=int(spec.get("cache_read", 0)),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return table
