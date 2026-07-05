"""Schema DSL field type registry.

Type tags that appear in `prompt_field` and `answer_field` declarations.
This is a CLOSED registry: customers cannot define new type tags via DSL.
Adding a new tag requires a code change here AND in the runtime
validator that enforces field-shape at write time.

Each tag carries:
  - `description`: human-readable explanation, surfaced in the inspector.
  - `allowed_attrs`: set of attribute names that may legally appear on
    a field of this type. Empty set means "no attributes allowed."
  - `attr_value_kinds`: per-attribute, the literal kind (`string`,
    `number`, `bool`) the attribute's value must be. Used by the
    validator to type-check attribute values.

DESIGN NOTE: the registry is intentionally small at v0. We add tags
when we author a real crystal_type that needs one. Speculative additions
just create surface area no one's tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet


# Literal kinds an attribute value can be required to take. Matches the
# parser's literal types (StringLit, NumberLit, BoolLit).
AttrValueKind = str  # 'string' | 'number' | 'bool'


@dataclass(frozen=True)
class FieldTypeSpec:
    tag: str
    description: str
    allowed_attrs: FrozenSet[str] = field(default_factory=frozenset)
    attr_value_kinds: dict[str, AttrValueKind] = field(default_factory=dict)


# v0 registry. Coding-vertical-leaning per the product goal.
FIELD_TYPE_REGISTRY: dict[str, FieldTypeSpec] = {
    "text": FieldTypeSpec(
        tag="text",
        description=(
            "Free-form natural-language text. The default for human-readable "
            "prompts and answers."
        ),
        allowed_attrs=frozenset(),
        attr_value_kinds={},
    ),
    "code": FieldTypeSpec(
        tag="code",
        description=(
            "Source code. Optional `language` attribute pins the language "
            "for syntactic checks at write time; optional `max_lines` caps "
            "the bound size."
        ),
        allowed_attrs=frozenset({"language", "max_lines"}),
        attr_value_kinds={"language": "string", "max_lines": "number"},
    ),
    "date": FieldTypeSpec(
        tag="date",
        description=(
            "A calendar date. Defaults to ISO-8601 (YYYY-MM-DD); other "
            "formats can be specified via the `format` attribute."
        ),
        allowed_attrs=frozenset({"format"}),
        attr_value_kinds={"format": "string"},
    ),
    "iso8601": FieldTypeSpec(
        tag="iso8601",
        description=(
            "Strict ISO-8601 datetime. Equivalent to `date format=\"iso8601\"` "
            "but reads more naturally as a tag for full timestamps."
        ),
        allowed_attrs=frozenset(),
        attr_value_kinds={},
    ),
    "traceback": FieldTypeSpec(
        tag="traceback",
        description=(
            "A runtime stack trace. Optional `language` pins the trace "
            "format (e.g. python, javascript) for parsing at write time."
        ),
        allowed_attrs=frozenset({"language"}),
        attr_value_kinds={"language": "string"},
    ),
    "error_message": FieldTypeSpec(
        tag="error_message",
        description=(
            "A short error string (e.g. an exception message, a build "
            "error line). Stored as text; tagged separately so the "
            "router can recognize 'this is the error part of a "
            "(error, fix) pair'."
        ),
        allowed_attrs=frozenset(),
        attr_value_kinds={},
    ),
    "enum": FieldTypeSpec(
        tag="enum",
        description=(
            "A value from a fixed set. The `values` attribute holds a "
            "comma-separated list of legal values. Validated at write "
            "time; the `pair_type` rejects writes whose value isn't in "
            "the set."
        ),
        allowed_attrs=frozenset({"values"}),
        attr_value_kinds={"values": "string"},
    ),
}


def is_known_tag(tag: str) -> bool:
    return tag in FIELD_TYPE_REGISTRY


def get_spec(tag: str) -> FieldTypeSpec:
    """Return the spec for a tag. Raises KeyError if unknown."""
    return FIELD_TYPE_REGISTRY[tag]


def all_tags() -> list[str]:
    """All known type tags, sorted, for error messages."""
    return sorted(FIELD_TYPE_REGISTRY.keys())
