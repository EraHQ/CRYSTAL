"""The structured-output seam (2026-10-02): every json_schema handed to the
Anthropic client passes through _strict_schema, so no caller can 400 on
shape. Three deploys of the respond feature failed on shape rules
(additionalProperties, then maxItems) that this seam now applies for
everyone. Pure-function pins; the SDK's transform_schema is used when the
installed SDK has it, and the local pass is pinned regardless.
"""
from crystal_cache.llm.client import _strict_schema


LOOSE = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1, "maxLength": 80},
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                    "tags": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
                },
                "required": ["name", "score", "tags"],
            },
        },
    },
    "required": ["items"],
}

_BOUNDS = {"minItems", "maxItems", "uniqueItems", "minLength", "maxLength",
           "minimum", "maximum", "pattern", "exclusiveMinimum", "exclusiveMaximum",
           "multipleOf"}


def _walk(node, path=""):
    if isinstance(node, dict):
        yield path or "<root>", node
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")


def test_strict_schema_closes_every_object_and_drops_every_bound():
    out = _strict_schema(LOOSE)
    for path, node in _walk(out):
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False, path
        for k, v in node.items():
            if k == "minItems" and v in (0, 1):
                continue  # the API supports minItems 0 and 1; the SDK keeps them
            assert k not in _BOUNDS, f"{path}: {k}={v} survived"
    # Shape and required lists survive.
    inner = out["properties"]["items"]["items"]
    assert set(inner["properties"]) == {"name", "score", "tags"}
    assert set(inner["required"]) == {"name", "score", "tags"}
    assert out["required"] == ["items"]


def test_strict_schema_is_idempotent_and_leaves_a_strict_schema_alone():
    strict = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
    }
    once = _strict_schema(strict)
    assert once["additionalProperties"] is False
    assert once["properties"] == {"a": {"type": "string"}}
    assert _strict_schema(once) == once


def test_the_respond_schema_survives_the_seam_unchanged_in_shape():
    from crystal_cache.endpoints.admin import _RESPOND_SCHEMA

    out = _strict_schema(_RESPOND_SCHEMA)
    assert out["additionalProperties"] is False
    item = out["properties"]["assumptions"]["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == {"statement", "subject", "confidence", "rationale", "replaces_parent"}


def test_seam_drops_temperature_for_models_that_reject_it():
    """The third live respond call 400'd with "temperature is deprecated
    for this model" on Sonnet 5. The seam, not each caller, owns this."""
    from crystal_cache.llm.client import _anthropic_sampling_kwargs

    for model in ("claude-sonnet-5", "claude-sonnet-5-5", "claude-opus-5",
                  "claude-opus-4-8", "claude-fable-5-1"):
        assert _anthropic_sampling_kwargs(model, 0.2) == {}, model
        assert _anthropic_sampling_kwargs(model, 0.0) == {}, model
    assert _anthropic_sampling_kwargs("claude-haiku-4-5", 0.0) == {"extra_body": {"temperature": 0.0}}
    assert _anthropic_sampling_kwargs("claude-haiku-4-5", None) == {}
