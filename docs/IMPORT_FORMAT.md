# Crystal Cache import format (v1)

This is the file format Crystal Cache imports. Anything that can write
JSON can produce it: a script over a spreadsheet, another memory tool's
export, a hand-written list of facts. Crystal Cache's own export
(`GET /v1/export`, or the `memory_export` MCP tool) produces exactly this
shape, so it doubles as the reference example.

The machine-readable contract is served live at `GET /v1/import/schema`
(JSON Schema 2020-12, no authentication). The copy below is the same
document; the live one is authoritative.

## The batch

```json
{
  "records": [
    {"key": "What is our refund window?", "value": "Thirty days from delivery, no questions asked."},
    {"key": "Who approves vendor contracts over 10k?", "value": "The COO, with Finance copied.", "crystal_type": "customer:policy", "scope": "team"}
  ],
  "crystal_type": "customer:legacy",
  "wipe": false
}
```

- `records`: 1 to 500 per batch. Send more as further batches.
- `crystal_type`: the default bucket for records that do not name one.
- `wipe`: erase the workspace's existing memory first. Workspace admins
  only. Everything else is additive.

## The record

| Field | Required | Type | Meaning |
|---|---|---|---|
| `key` | yes | string, 1 to 5,000 chars | What a question should match: the question itself, a title, a subject line. When `key_is_path` is true it is a Crystal Cache sparse path (as exported) and is stored verbatim. |
| `value` | yes | string, 1 to 50,000 chars | The knowledge. |
| `key_is_path` | no | boolean, default `false` | Leave false unless the record came from a Crystal Cache export. |
| `pair_type` | no | `^[a-z][a-z0-9_]{0,63}$`, default `question_answer` | The fact's shape. `question_answer` unless you know otherwise. |
| `source_kind` | no | one of `model_reasoning`, `failed_reasoning`, `web_search_result`, `code_execution_result`, `document_chunk`, `document_extraction`, `operator_stated`, `agent_inferred`; default `model_reasoning` | Where the knowledge came from. `operator_stated` is the right value for facts a person wrote down. |
| `answer_value` | no | string or null, up to 50,000 chars | If set, an exact-match question returns this value directly. |
| `crystal_type` | no | `^customer:[A-Za-z0-9_.-]{1,64}$`, default `customer:legacy` | A `customer:<name>` bucket. |
| `scope` | no | `personal` or `team`; omitted = the workspace default | `personal`: only the importing operator and workspace admins can recall it. `team`: everyone in the workspace. |

No other fields are accepted. A record with an unknown field, an
over-long value, or a value outside its enum refuses the whole batch.

### What a record never carries

Origin, recall gating, ownership ids and permission modes are not part of
the format. Every imported fact is the importing workspace's own
foreground memory: it is born `direct`, visible to recall, counted
against the plan's crystal-fact capacity, and owned by the operator who
imported it at the scope the record chose. The exact-restore format the
console's Export button produces (`crystal-cache-<workspace>-<date>.json`)
carries all of that for a byte-exact restore of the same workspace; it is
a backup format, not an interchange format.

## Validation and errors

The whole batch is validated before anything is written. If any record
is invalid, nothing is imported and the response lists every problem:

```json
{
  "error": {
    "message": "2 problem(s) in the records; nothing was imported. See errors[] for each record's index and field, and GET /v1/import/schema for the record shape.",
    "type": "invalid_request_error",
    "param": "records",
    "code": "invalid_record"
  },
  "errors": [
    {"index": 3, "field": "value", "message": "String should have at least 1 character"},
    {"index": 7, "field": "origin", "message": "Extra inputs are not permitted"}
  ]
}
```

HTTP returns this with status 422. The MCP tool returns the same
`error` / `code` / `errors` fields in its result. `index` is the record's
position in `records` (0-based); `field` is the offending field.

Other refusals: `admin_required` (wipe by a non-admin), `memory_full`
(the batch would take the bank past the plan's capacity; the check is
bank plus batch), `daily_capacity` (the plan's daily AI allowance is used
up; importing records without `key_is_path` runs one small model call
per record to derive its path).

## Importing

HTTP, with a workspace key or an operator key:

```
POST /v1/import
Authorization: Bearer <key>
Content-Type: application/json

<the batch>
```

Response: `{"records_processed": 2, "crystals_written": 2, "errors": 0}`.

MCP: call `memory_import` with `records` (the same list), and optionally
`wipe` and `crystal_type`.

## Exporting

`GET /v1/export` returns `{"record_count", "export_format": "jsonl",
"data": [records]}` where every record validates against this schema.
`memory_export` returns the same records 500 at a time (`limit`,
`offset`, `has_more`). The export carries foreground memory only; see
"What a record never carries".

## The schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://crystalcache.ai/schemas/import/v1",
  "$defs": {
    "ImportRecord": {
      "additionalProperties": false,
      "description": "One fact to import. Unknown fields are refused.",
      "properties": {
        "key": {
          "description": "What a question should match. A plain phrase, or a finished sparse path when key_is_path is true.",
          "maxLength": 5000,
          "minLength": 1,
          "title": "Key",
          "type": "string"
        },
        "value": {
          "description": "The knowledge itself.",
          "maxLength": 50000,
          "minLength": 1,
          "title": "Value",
          "type": "string"
        },
        "key_is_path": {
          "default": false,
          "description": "True when `key` is already a Crystal Cache sparse path (as exported); it is stored verbatim instead of derived.",
          "title": "Key Is Path",
          "type": "boolean"
        },
        "pair_type": {
          "default": "question_answer",
          "description": "The fact's shape, lowercase with underscores; question_answer unless you know otherwise.",
          "maxLength": 64,
          "pattern": "^[a-z][a-z0-9_]{0,63}$",
          "title": "Pair Type",
          "type": "string"
        },
        "source_kind": {
          "default": "model_reasoning",
          "description": "Where the knowledge came from.",
          "enum": [
            "model_reasoning",
            "failed_reasoning",
            "web_search_result",
            "code_execution_result",
            "document_chunk",
            "document_extraction",
            "operator_stated",
            "agent_inferred"
          ],
          "title": "Source Kind",
          "type": "string"
        },
        "answer_value": {
          "anyOf": [
            {
              "maxLength": 50000,
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "If set, an exact-match question returns this value directly.",
          "title": "Answer Value"
        },
        "crystal_type": {
          "default": "customer:legacy",
          "description": "A customer:<name> bucket for the fact.",
          "maxLength": 72,
          "pattern": "^customer:[A-Za-z0-9_.-]{1,64}$",
          "title": "Crystal Type",
          "type": "string"
        },
        "scope": {
          "anyOf": [
            {
              "enum": [
                "personal",
                "team"
              ],
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "personal (only the importing operator and admins can recall it) or team (the whole workspace). Omitted = the workspace default.",
          "title": "Scope"
        }
      },
      "required": [
        "key",
        "value"
      ],
      "title": "Crystal Cache import record v1",
      "type": "object"
    }
  },
  "additionalProperties": false,
  "description": "The body of POST /v1/import and the arguments of memory_import.",
  "properties": {
    "records": {
      "description": "1 to 500 records; send more as further batches.",
      "items": {
        "$ref": "#/$defs/ImportRecord"
      },
      "maxItems": 500,
      "minItems": 1,
      "title": "Records",
      "type": "array"
    },
    "crystal_type": {
      "default": "customer:legacy",
      "description": "Default bucket for records that do not name one.",
      "maxLength": 72,
      "pattern": "^customer:[A-Za-z0-9_.-]{1,64}$",
      "title": "Crystal Type",
      "type": "string"
    },
    "wipe": {
      "default": false,
      "description": "Erase the workspace's existing memory first. Workspace admins only.",
      "title": "Wipe",
      "type": "boolean"
    }
  },
  "required": [
    "records"
  ],
  "title": "Crystal Cache import batch v1",
  "type": "object"
}
```
