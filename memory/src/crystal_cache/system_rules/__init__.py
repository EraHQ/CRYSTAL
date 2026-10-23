"""System rules — user-owned automation of their own judgment.

A declarative, per-tenant rule set: "WHEN <conditions>, DO <action> to
things matching <selector>." Storage is generic (JSON columns on
system_rules); execution is TYPED per rule_type so a malformed or malicious
rule can't do something unintended. Validation is STRICT (ratified
2026-07-03): an unrecognized selector/condition/action key rejects the rule
rather than being silently ignored.

First rule_type: 'promotion' (clears the recall gate on background-worker
memory when the user's conditions hold). Second (2026-10-23, Q99 C):
'approval' (approves an assumption once enough facts agree with it and
none contradict). Designed to hold 'sharing', 'task_spawn' later behind
the same table + typed dispatch.

Instruction-source boundary: rules come ONLY from the user via the control
plane, never from tool output or crystal content.
"""
from __future__ import annotations

from .model import RuleValidationError, SystemRule
from .registry import (
    get_rule_type,
    register_rule_type,
    validate_rule,
)
from . import promotion as _promotion  # noqa: F401 — registers 'promotion'
from . import approval as _approval  # noqa: F401 — registers 'approval' (Q99 C, 2026-10-23)
from . import curation as _curation  # noqa: F401 — registers 'curation' (Q101, 2026-10-23): the ordered allow/deny list both fold into

__all__ = [
    "SystemRule",
    "RuleValidationError",
    "get_rule_type",
    "register_rule_type",
    "validate_rule",
]
