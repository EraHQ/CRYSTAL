"""System rules — user-owned automation of their own judgment.

A declarative, per-tenant rule set: "WHEN <conditions>, DO <action> to
things matching <selector>." Storage is generic (JSON columns on
system_rules); execution is TYPED per rule_type so a malformed or malicious
rule can't do something unintended. Validation is STRICT (ratified
2026-07-03): an unrecognized selector/condition/action key rejects the rule
rather than being silently ignored.

First rule_type: 'promotion' (clears the recall gate on background-worker
memory when the user's conditions hold). Designed to hold 'sharing',
'approval', 'task_spawn' later behind the same table + typed dispatch.

Instruction-source boundary: rules come ONLY from the user via the control
plane, never from tool output or crystal content.
"""
from __future__ import annotations

from .model import RuleValidationError, SystemRule
from .registry import (
    get_rule_type,
    known_rule_types,
    register_rule_type,
    validate_rule,
)
from . import promotion as _promotion  # noqa: F401 — registers 'promotion'

__all__ = [
    "SystemRule",
    "RuleValidationError",
    "get_rule_type",
    "register_rule_type",
    "known_rule_types",
    "validate_rule",
]
