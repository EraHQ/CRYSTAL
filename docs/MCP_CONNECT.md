# Connecting to Crystal Cache over MCP

Crystal Cache exposes its memory bank as a standard [MCP](https://modelcontextprotocol.io) server, so any MCP-speaking client — Claude Desktop, Claude Code, an agent framework, your own tooling — can search, store, and manage memory directly.

## Endpoint

```
https://<your-deployment>/mcp
```

Hosted accounts use `https://api.erahq.ai/mcp`. Self-hosted deployments serve it from the same container as the API (mounted at `/mcp`); no extra process or configuration is needed.

The server speaks streamable HTTP (plain request/response JSON) — no SSE session required.

## Authentication

Every request carries your API key as a Bearer token:

```
Authorization: Bearer <your-key>
```

Two kinds of key work, with different visibility:

- **Operator key** (recommended) — you act as yourself. Searches return what you own plus what's shared with you; things you store are attributed to you.
- **Team key** — you act as the team's administrator and see everything in the team.

Operators with the **viewer** role can use all the read tools but are refused by every tool that writes or deletes.

## Client configuration example

For clients that take a JSON server entry (Claude Desktop and most others):

```json
{
  "mcpServers": {
    "crystal-cache": {
      "type": "http",
      "url": "https://api.erahq.ai/mcp",
      "headers": {
        "Authorization": "Bearer <your-key>"
      }
    }
  }
}
```

## Tools

**Finding things**

- `memory_search` — semantic search over stored knowledge; returns the top matching facts with their keys and values.
- `memory_search_documents` — search the verbatim text of ingested documents.
- `memory_outline` — structural overview of what the bank knows about a subject (counts, listings, gaps), no semantic ranking.
- `memory_keys` — enumerate items by hierarchical key prefix or subject.
- `memory_synthesize` — cross-item synthesis ("how does X relate to Y"); heavier than search.
- `memory_recall` — simple "what do we know about X" entry point, grouped by entity.

**Storing things**

- `memory_store` — store one (key, value) fact. Pass `scope="personal"` (only you and admins can retrieve it) or `scope="team"` (whole team) to override your deployment's default visibility for that write.
- `memory_ingest` — chunk, extract, and store a document's text in one call. Inputs over the deployment's character ceiling (default 200,000) are refused; send very large documents through the async upload endpoint (`POST /v1/documents`) instead.
- `memory_learn` — teach from an outcome: cache a successful answer for fast recall, or record a correction after a failure.

**Managing the bank**

- `memory_list` — browse stored clusters, or inspect one in full detail.
- `memory_stats` — bank-level statistics (counts and distributions).
- `memory_forget` — permanently delete a cluster or a single fact.
- `memory_export` — export fact-level records for backup or migration. Paginated: up to `limit` records per call (default 1000); advance `offset` until `has_more` is false.
- `memory_import` — import records in the export format; `wipe=true` replaces the bank first.

**Self-curation**

- `memory_conflicts` — contradictions the system has detected in its own memory.
- `memory_gaps` — things the memory was asked about but doesn't know.
- `memory_record_gap` — record a question the memory couldn't answer, so it can be researched or taught later.

## Limits

- **Requests:** 240 per minute per key by default (deployment-configurable). Over-limit requests get HTTP 429 with a `Retry-After` header.
- **Ingest:** `memory_ingest` accepts up to 200,000 characters per call by default; larger documents go through `POST /v1/documents`.
- **Export:** `memory_export` returns at most 1,000 records per page.

## Notes

- Your identity always comes from your API key — tool arguments never carry it.
- `memory_forget` and `memory_import` with `wipe=true` are permanent; there is no undo.
- Stored knowledge is visible to future searches immediately; ingested documents become searchable when the call returns.
