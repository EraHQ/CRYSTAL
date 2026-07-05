"""Typed rule-type registry.

Each rule_type registers a RuleType with a validate() (strict, raises
RuleValidationError) and an apply() (executes the rule against one
candidate, returns True if it acted). The generic storage layer stays
schema-free; this is where type safety lives — the executor is the safety
boundary for judgment automation.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional, Protocol

from .model import RuleValidationError, SystemRule


class RuleType(Protocol):
    name: str

    def validate(self, rule: SystemRule) -> None:
        """Raise RuleValidationError if the rule's selector/conditions/action
        are not exactly what this type accepts. STRICT: unknown keys reject."""
        ...


_REGISTRY: dict[str, "RuleTypeImpl"] = {}


class RuleTypeImpl:
    """Concrete rule-type: a validator plus an async applier."""

    def __init__(
        self,
        name: str,
        validate: Callable[[SystemRule], None],
        apply: Callable[..., Awaitable[bool]],
    ) -> None:
        self.name = name
        self._validate = validate
        self._apply = apply

    def validate(self, rule: SystemRule) -> None:
        self._validate(rule)

    async def apply(self, rule: SystemRule, *, store, candidate) -> bool:
        return await self._apply(rule, store=store, candidate=candidate)


def register_rule_type(
    name: str,
    validate: Callable[[SystemRule], None],
    apply: Callable[..., Awaitable[bool]],
) -> None:
    _REGISTRY[name] = RuleTypeImpl(name, validate, apply)


def get_rule_type(name: str) -> "RuleTypeImpl":
    impl = _REGISTRY.get(name)
    if impl is None:
        raise RuleValidationError(
            f"unknown rule_type {name!r}; known types: "
            f"{sorted(_REGISTRY)}"
        )
    return impl


def known_rule_types() -> list[str]:
    return sorted(_REGISTRY)


def validate_rule(rule: SystemRule) -> None:
    """Validate a rule against its type. Raises RuleValidationError on any
    problem (unknown type, unknown key, wrong shape)."""
    get_rule_type(rule.rule_type).validate(rule)
