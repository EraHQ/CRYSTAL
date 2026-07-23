"""Mapping inference + mechanical application — Gate G slice 2.

The judgment-once-mechanism-forever core (C5, Q9-A): `propose_mapping`
makes the ONE inference call a new JSON shape ever gets, producing the
role-per-path mapping spec (format v1, ratified 2026-07-23);
`apply_mapping` executes an approved spec per record with zero LLM —
one fact per record, auditable, deterministic.

Mapping format v1:
    {
      "version": 1,
      "roles": {"[].name": "key", "[].role": "value",
                "[].id": "locator", "[].created_at": "timestamp",
                "[].internal_uuid": "skip"},
      "subject": "[].name",
      "domain": "Team Directory"
    }

Roles: key (joins into the fact key), value (rendered `field: value`
into the fact value), locator (sparse-key Locator; falls back to the
record index), timestamp (appended to the value), skip (explicitly
ignored — visible in the review preview). Unknown/unmapped paths are
treated as skip. "Segment feeds" (multi-fact-per-record) are a
version-2 concern; the version field exists so that lands without
churn.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_VALID_ROLES = {"key", "value", "locator", "timestamp", "skip"}

MAPPING_MAX_TOKENS = 1200
MAPPING_TEMPERATURE = 0.0

MAPPING_SYSTEM = """You design a field mapping for ingesting JSON records into a knowledge base.

You receive the record shape (its key-paths with JSON types) and a few sample records. Assign each key-path exactly one role:

- "key": identifies the record (names, titles, identifiers a human would use)
- "value": informative content worth stating as knowledge
- "locator": a stable reference/id for citing the record
- "timestamp": when the record happened/was created
- "skip": internal ids, hashes, booleans of no standalone meaning, noise

Also choose:
- "subject": the ONE key-path whose value best names what each record is about
- "domain": a short human phrase (2-4 words) naming what this data collection is about

Respond with ONLY a JSON object, no prose, no markdown fences:
{"version": 1, "roles": {"<path>": "<role>", ...}, "subject": "<path>", "domain": "<phrase>"}

Every key-path you were given must appear in roles."""


def _leaf(path: str) -> str:
    """Human field name from a path: '[].user.email' -> 'email'."""
    return path.replace("[]", "").strip(".").rsplit(".", 1)[-1] or path


def _get_path(record: Any, path: Optional[str]) -> Optional[Any]:
    """Resolve a mapping path against ONE record. Leading '[].' (the
    array-root marker from schema_hash) is stripped — the caller
    iterates records. Nested lists render as comma-joined scalars."""
    if not path:
        return None
    clean = path
    if clean.startswith("[]"):
        clean = clean[2:]
    clean = clean.strip(".")
    value: Any = record
    if clean:
        for part in clean.split("."):
            # Inner array markers ('tags[]') collapse: resolve the list
            # itself under the bare name.
            part = part.replace("[]", "")
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
            if value is None:
                return None
    if isinstance(value, list):
        scalars = [str(v) for v in value if not isinstance(v, (dict, list))]
        return ", ".join(scalars) if scalars else None
    if isinstance(value, (dict,)):
        return None
    return value


def validate_mapping(candidate: Any) -> Optional[dict[str, Any]]:
    """Coerce a model response into a valid v1 mapping, or None.

    Lenient by design: unknown role strings coerce to 'skip' (visible
    in the preview, fixable by edit); a missing/invalid roles table is
    a failed proposal."""
    if not isinstance(candidate, dict):
        return None
    roles = candidate.get("roles")
    if not isinstance(roles, dict) or not roles:
        return None
    clean_roles: dict[str, str] = {}
    for path, role in roles.items():
        if not isinstance(path, str):
            continue
        role_str = role if isinstance(role, str) else "skip"
        clean_roles[path] = role_str if role_str in _VALID_ROLES else "skip"
    if not clean_roles:
        return None
    subject = candidate.get("subject")
    domain = candidate.get("domain")
    return {
        "version": 1,
        "roles": clean_roles,
        "subject": subject if isinstance(subject, str) else None,
        "domain": domain.strip() if isinstance(domain, str) and domain.strip()
        else "General",
    }


def _parse_json_object(text: str) -> Optional[dict]:
    """Model output -> dict; tolerates markdown fences."""
    body = text.strip()
    if body.startswith("```"):
        body = body.strip("`")
        if body.startswith("json"):
            body = body[4:]
        body = body.strip()
    try:
        parsed = json.loads(body)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


async def propose_mapping(
    client: Any,
    *,
    key_paths: list[str],
    sample_records: list[Any],
    customer_id: Optional[str] = None,
    store: Any = None,
) -> Optional[dict[str, Any]]:
    """THE one inference call per new shape (C5). Small tier,
    temperature 0, ledger-stamped under origin='schema_inference'.
    Fail-safe: any failure returns None and the proposal ships with an
    empty mapping for the reviewer to fill."""
    user = (
        "Record shape (key-path: type):\n"
        + "\n".join(key_paths)
        + "\n\nSample records:\n"
        + json.dumps(sample_records[:3], ensure_ascii=False, indent=2)[:6000]
    )
    kwargs = dict(
        system=MAPPING_SYSTEM,
        messages=[{"role": "user", "content": user}],
        max_tokens=MAPPING_MAX_TOKENS,
        temperature=MAPPING_TEMPERATURE,
        tier="small",
    )
    try:
        detailed = getattr(client, "complete_detailed", None)
        if detailed is not None:
            result = detailed(**kwargs)
            text = (result.text or "").strip()
            if customer_id and store is not None:
                from ..cost.emit import record_model_call

                await record_model_call(
                    customer_id=customer_id,
                    origin="schema_inference",
                    model=result.model,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    cache_creation_tokens=result.cache_creation_tokens,
                    cache_read_tokens=result.cache_read_tokens,
                    store=store,
                )
        else:
            text = (client.complete(**kwargs) or "").strip()
    except Exception as e:  # noqa: BLE001 — best-effort by design
        logger.warning(
            "schema_mapping.propose_failed",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        return None
    return validate_mapping(_parse_json_object(text))


def apply_mapping(
    payload: Any, mapping: dict[str, Any], *, label: str = "",
) -> list[dict[str, Any]]:
    """Execute an approved mapping mechanically: one fact per record,
    zero LLM. Output matches crystallization's extracted_items shape
    ({key, sparse_key, value, type, citation}); item_type 'fact' maps
    to the question_answer pair type downstream.

    Records that yield no value content are skipped; a mapping with no
    value roles yields [] — both visible in the review preview, both
    fixable by editing the mapping."""
    records = payload if isinstance(payload, list) else [payload]
    roles: dict[str, str] = mapping.get("roles") or {}
    key_paths = [p for p, r in roles.items() if r == "key"]
    value_paths = [p for p, r in roles.items() if r == "value"]
    locator_paths = [p for p, r in roles.items() if r == "locator"]
    ts_paths = [p for p, r in roles.items() if r == "timestamp"]
    subject_path = mapping.get("subject")
    domain = mapping.get("domain") or "General"
    source = (label.rsplit("/", 1)[-1].rsplit(".", 1)[0]) or "json"

    items: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        key_parts = []
        for p in key_paths:
            v = _get_path(record, p)
            if v not in (None, ""):
                key_parts.append(str(v))
        key = " ".join(key_parts) or f"{source} record {i + 1}"

        values = []
        for p in value_paths:
            v = _get_path(record, p)
            if v not in (None, ""):
                values.append(f"{_leaf(p)}: {v}")
        for p in ts_paths:
            v = _get_path(record, p)
            if v not in (None, ""):
                values.append(f"{_leaf(p)}: {v}")
        if not values:
            continue

        locator = None
        for p in locator_paths:
            v = _get_path(record, p)
            if v not in (None, ""):
                locator = str(v)
                break
        locator = locator or f"record {i + 1}"

        subject = _get_path(record, subject_path)
        subject_str = str(subject) if subject not in (None, "") else key

        items.append({
            "key": key,
            "sparse_key": f"{source} | {locator} | {subject_str} | {domain}",
            "value": "; ".join(values),
            "type": "fact",
            "citation": locator,
        })
    return items
