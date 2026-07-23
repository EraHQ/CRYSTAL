"""JSON schema fingerprinting — Gate G (C5, Q9-A ratified 2026-07-16).

The identity of a JSON *shape*: sha256 over sorted key-paths + JSON
types, values ignored, arrays collapsed. Two payloads with the same
structure hash identically regardless of content, record order, or
array length — so one approved mapping serves every future arrival of
that shape ("one human judgment per shape of data, ever").

Rules, pinned by test:
- Arrays collapse to a single "[]" path segment (index-invariant).
- JSON types only: object, array, string, number, boolean, null —
  ints and floats are both "number" (JSON has one number type).
- Array-rooted payloads (and JSONL record streams) fingerprint the
  UNION of paths across a bounded sample of records
  (_SAMPLE_RECORDS), so heterogeneous optional fields still land in
  one schema.
- Key order never matters (paths are sorted before hashing).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# Records sampled from an array/JSONL root when fingerprinting. Bounded
# so a million-record export fingerprints in microseconds; unioned so
# optional fields present in only some records still shape the schema.
_SAMPLE_RECORDS = 20


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):  # BEFORE int — bool subclasses int
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def collect_key_paths(value: Any, prefix: str = "") -> set[str]:
    """Every `path:type` string in a JSON value, arrays collapsed."""
    paths: set[str] = set()
    kind = _json_type(value)
    if kind == "object":
        if prefix:
            paths.add(f"{prefix}:object")
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            paths |= collect_key_paths(child, child_prefix)
    elif kind == "array":
        if prefix:
            paths.add(f"{prefix}:array")
        for item in value[:_SAMPLE_RECORDS]:
            paths |= collect_key_paths(item, f"{prefix}[]" if prefix else "[]")
    else:
        paths.add(f"{prefix or '$'}:{kind}")
    return paths


def schema_key_paths(payload: Any) -> set[str]:
    """The path set that IS the schema. Array-rooted payloads union
    across a sample of records; object-rooted use the whole value."""
    if isinstance(payload, list):
        paths: set[str] = set()
        for record in payload[:_SAMPLE_RECORDS]:
            paths |= collect_key_paths(record, "[]")
        return paths or {"[]:empty"}
    return collect_key_paths(payload)


def schema_hash(payload: Any) -> str:
    """sha256 hex over the sorted path set (the C5 fingerprint)."""
    paths = sorted(schema_key_paths(payload))
    joined = "\n".join(paths)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def parse_json_source(text: str) -> tuple[Any, str]:
    """Parse .json or .jsonl/.ndjson text into (payload, shape).

    shape: 'array' (array root OR a JSONL record stream, normalized to
    a list) | 'object' (single object root). Raises ValueError on
    unparseable input — the caller routes that to the normal
    ingestion-error path.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty JSON source")
    try:
        payload = json.loads(stripped)
        if isinstance(payload, list):
            return payload, "array"
        return payload, "object"
    except json.JSONDecodeError:
        pass
    # JSONL: one JSON value per non-empty line.
    records = []
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))  # raises -> caller handles
    if not records:
        raise ValueError("no JSON records found")
    return records, "array"
