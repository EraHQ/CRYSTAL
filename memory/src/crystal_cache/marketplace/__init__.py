"""Marketplace (Growth G4) — the marketplace package.

Pure crediting policy (crediting.py) decoupled from the DB-bound shard ledger
in infrastructure/metadata_store_shard_ext.py, so eligibility + weighting are
unit-testable without a store. Shard units are integers; the reward-pool
apportionment that turns raw usefulness weights into final shares is D7 —
deferred — so v1 credits a fixed integer per grounded citation and preserves
the fractional weight in the ledger for when the pool lands.
"""
from .crediting import (
    is_marketplace_crystal,
    is_self_traffic,
    shards_from_weight,
    split_weight,
)

__all__ = [
    "is_marketplace_crystal",
    "is_self_traffic",
    "shards_from_weight",
    "split_weight",
]
