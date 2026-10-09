"""The public import record (Q44=A as modified, ratified 2026-10-08).

One schema, one validator, two importers: `POST /v1/import` and the MCP
`memory_import` tool both validate every record of a batch against
`ImportRecord` BEFORE any write, and refuse the whole batch with
per-record errors on the first bad one. `GET /v1/import/schema` serves
the same shape as JSON Schema (2020-12) so a third party can build to it,
and `docs/IMPORT_FORMAT.md` documents it. Both exporters (`GET /v1/export`
and MCP `memory_export`) emit records that validate against it; a pin
holds that.

What the record deliberately does NOT carry (AUDIT_LAUNCH_VERIFY B3-6,
B1-21): `origin`, `recall_gated`, `owner_operator_id`, `group_team_id`,
`mode`. Everything a customer imports is their own foreground memory:
`origin="direct"`, ungated, counted against the fact cap, owned by the
importing operator with the scope they chose. Letting an importer set
`origin` would let them import past the cap for free (only direct facts
count), and letting them set the gate or ownership would let a member
write into a colleague's personal memory. The exact-restore (topology)
format keeps those fields and is admin-only.

Export side: the portable format carries the customer's own facts, so the
exporters emit only facts of `origin="direct"`, ungated, `customer:*`
crystals that are still active. System-derived memory (assumptions,
reflections, background curation) travels only in the topology export,
where it comes back AS IT WAS instead of being reborn as plain facts.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationError,
)
from typing_extensions import Annotated

from ..models.crystal import SourceKind

SCHEMA_ID = "https://crystalcache.ai/schemas/import/v1"
MAX_RECORDS = 500
KEY_MAX_CHARS = 5_000
VALUE_MAX_CHARS = 50_000
ANSWER_MAX_CHARS = 50_000
PAIR_TYPE_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
CRYSTAL_TYPE_PATTERN = r"^customer:[A-Za-z0-9_.-]{1,64}$"

# The closed provenance set the bank stores (models/crystal.py). The
# import accepts the whole set so an export round-trips byte for byte;
# the manual write surfaces (/v1/store, memory_store) keep their
# narrower lists.
SOURCE_KINDS: tuple[str, ...] = tuple(get_args(SourceKind))

PairType = Annotated[str, StringConstraints(pattern=PAIR_TYPE_PATTERN, max_length=64)]
CrystalType = Annotated[str, StringConstraints(pattern=CRYSTAL_TYPE_PATTERN, max_length=72)]


class ImportRecord(BaseModel):
    """One fact to import. Unknown fields are refused."""

    model_config = ConfigDict(extra="forbid", title="Crystal Cache import record v1")

    key: str = Field(
        min_length=1, max_length=KEY_MAX_CHARS,
        description="What a question should match. A plain phrase, or a finished sparse path when key_is_path is true.",
    )
    value: str = Field(
        min_length=1, max_length=VALUE_MAX_CHARS,
        description="The knowledge itself.",
    )
    key_is_path: bool = Field(
        default=False,
        description="True when `key` is already a Crystal Cache sparse path (as exported); it is stored verbatim instead of derived.",
    )
    pair_type: PairType = Field(
        default="question_answer",
        description="The fact's shape, lowercase with underscores; question_answer unless you know otherwise.",
    )
    source_kind: SourceKind = Field(
        default="model_reasoning",
        description="Where the knowledge came from.",
    )
    answer_value: Optional[str] = Field(
        default=None, max_length=ANSWER_MAX_CHARS,
        description="If set, an exact-match question returns this value directly.",
    )
    crystal_type: CrystalType = Field(
        default="customer:legacy",
        description="A customer:<name> bucket for the fact.",
    )
    scope: Optional[Literal["personal", "team"]] = Field(
        default=None,
        description="personal (only the importing operator and admins can recall it) or team (the whole workspace). Omitted = the workspace default.",
    )


class ImportBatch(BaseModel):
    """The body of POST /v1/import and the arguments of memory_import."""

    model_config = ConfigDict(extra="forbid", title="Crystal Cache import batch v1")

    records: list[ImportRecord] = Field(
        min_length=1, max_length=MAX_RECORDS,
        description=f"1 to {MAX_RECORDS} records; send more as further batches.",
    )
    crystal_type: CrystalType = Field(
        default="customer:legacy",
        description="Default bucket for records that do not name one.",
    )
    wipe: bool = Field(
        default=False,
        description="Erase the workspace's existing memory first. Workspace admins only.",
    )


_RECORDS = TypeAdapter(list[ImportRecord])


def validate_records(records: Any) -> tuple[list[ImportRecord], list[dict[str, Any]]]:
    """Parse a batch. Returns (parsed, errors); `errors` is the complete
    per-record list `[{index, field, message}]` and is empty only when
    every record is valid. Nothing is written by this function."""
    if not isinstance(records, list):
        return [], [{"index": None, "field": "records", "message": "records must be a list"}]
    if not records:
        return [], [{"index": None, "field": "records", "message": "records is empty"}]
    if len(records) > MAX_RECORDS:
        return [], [{
            "index": None, "field": "records",
            "message": f"at most {MAX_RECORDS:,} records per batch ({len(records):,} given); send the rest as further batches",
        }]
    try:
        return _RECORDS.validate_python(records), []
    except ValidationError as e:
        errors: list[dict[str, Any]] = []
        for err in e.errors():
            loc = err.get("loc") or ()
            index = loc[0] if loc and isinstance(loc[0], int) else None
            field = ".".join(str(p) for p in loc[1:]) if len(loc) > 1 else (
                str(loc[0]) if loc and not isinstance(loc[0], int) else ""
            )
            errors.append({"index": index, "field": field or None, "message": err.get("msg", "invalid")})
        return [], errors


def json_schema() -> dict[str, Any]:
    """The published JSON Schema for the batch (the record rides in
    `$defs`). Served by GET /v1/import/schema and printed in
    docs/IMPORT_FORMAT.md."""
    schema = ImportBatch.model_json_schema()
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        **schema,
    }
    schema.setdefault("description", (
        "A batch of facts for a Crystal Cache workspace. Validated whole "
        "before any write; a bad record refuses the batch with its index "
        "and field."
    ))
    return schema


def is_portable(crystal: Any) -> bool:
    """Whether a crystal's facts belong in the portable (fact-level)
    export: the customer's own foreground memory, in a customer:* bucket,
    not held behind the recall gate."""
    if crystal is None:
        return False
    if (getattr(crystal, "origin", None) or "direct") != "direct":
        return False
    if bool(getattr(crystal, "recall_gated", False)):
        return False
    ctype = getattr(crystal, "crystal_type", "") or ""
    return ctype.startswith("customer:")


def scope_of(crystal: Any) -> str:
    """The record scope for a crystal: personal when its mode denies the
    group read bit (0o600), team otherwise (0o640 and legacy unowned)."""
    mode = getattr(crystal, "mode", None)
    if mode is None:
        return "team"
    return "personal" if (int(mode) & 0o040) == 0 else "team"


def record_from_fact(fact: Any, crystal: Any) -> dict[str, Any]:
    """The export shape: exactly the ImportRecord fields, nothing else."""
    return {
        "key": fact.prompt_text,
        "value": fact.claim_text,
        "key_is_path": True,
        "pair_type": fact.pair_type,
        "source_kind": getattr(fact, "source_kind", None) or crystal.source_kind,
        "answer_value": crystal.answer_value,
        "crystal_type": crystal.crystal_type,
        "scope": scope_of(crystal),
    }
