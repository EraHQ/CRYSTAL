"""Cost accounting (Growth G3) — the cost package.

Pure cost arithmetic (pricing.py) decoupled from the DB-bound recording in
infrastructure/metadata_store_cost_ext.py, so the money math is unit-testable
without a store. Money is INTEGER micro-USD throughout — never a float.
"""
from .pricing import (
    DEFAULT_PRICE_TABLE,
    ModelPrice,
    compute_cost_micro_usd,
    price_table_from_settings,
    resolve_model_price,
    estimate_cost_usd,
)

__all__ = [
    "DEFAULT_PRICE_TABLE",
    "ModelPrice",
    "compute_cost_micro_usd",
    "price_table_from_settings",
    "resolve_model_price",
    "estimate_cost_usd",
]
