"""The generic SystemRule model + the validation error type.

The rule is stored with JSON selector/conditions/action; this dataclass is
the in-memory shape. Per-rule-type validators (registry.py + the type
modules) enforce the real schema of those JSON blobs. STRICT validation:
unknown keys are rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


class RuleValidationError(ValueError):
    """A rule failed strict validation (unknown key, wrong type, missing
    required field, unknown rule_type). Raised at create/update time so a
    misconfigured rule never reaches the evaluator."""


@dataclass
class SystemRule:
    id: str
    customer_id: str
    rule_type: str
    name: str
    selector: dict[str, Any] = field(default_factory=dict)
    conditions: dict[str, Any] = field(default_factory=dict)
    action: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 100
    last_fired_at: Optional[datetime] = None
    fire_count: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


def require_keys(
    blob: dict[str, Any], allowed: set[str], *, where: str,
) -> None:
    """STRICT key check: every key in `blob` must be in `allowed`, else
    RuleValidationError. This is the core of strict validation — a typo'd
    key ('conditon') fails loud instead of silently never matching."""
    unknown = set(blob) - allowed
    if unknown:
        raise RuleValidationError(
            f"{where}: unknown key(s) {sorted(unknown)}; "
            f"allowed keys are {sorted(allowed)}"
        )


def require_type(
    value: Any, expected: type, *, key: str, where: str,
) -> None:
    """Type check one field. bool is checked before int (bool is a subclass
    of int in Python, so an int check would wrongly accept True)."""
    if expected is int and isinstance(value, bool):
        raise RuleValidationError(
            f"{where}: key {key!r} must be int, got bool"
        )
    if not isinstance(value, expected):
        raise RuleValidationError(
            f"{where}: key {key!r} must be {expected.__name__}, "
            f"got {type(value).__name__}"
        )
