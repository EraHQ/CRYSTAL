# CRYSTAL

**Self-curating memory for AI agents.**

Today's memory layer is static. Vector stores and RAG pipelines are
storage: what goes in sits there, right or wrong, stale or current,
duplicated or contradicted. Retrieval quality decays as the pile grows.
RAG gets worse with time. Crystal gets better.

CRYSTAL is memory with cognition and metacognition built in. It does
not just store what your agents learn. It thinks about what it knows:
finds the gaps, researches and fills them, checks its own reasoning,
resolves contradictions, and promotes or demotes knowledge as evidence
moves. Storage is a feature. Curation is the product.

**CRYS**, the bundled agent, is the first agent built on this memory.
Any agent can be next: connect through the agent API, or point an
existing OpenAI-shaped client at the drop-in proxy and gain persistent,
self-improving memory with zero code changes.

```
Your agents ──► CRYSTAL memory ──► any upstream LLM (Anthropic, OpenAI, Vertex)
                    │
                    ▼
         self-curating memory
    (crystals: clustered, keyed,
     cited, tiered, cost-accounted)
```

## Cognition: memory that fills its own gaps

When retrieval misses, CRYSTAL records a knowledge gap. A background
cognition engine picks gaps up and runs a structured research workflow:
an orchestrator writes a goal contract and plan, workers execute
research steps against the knowledge bank and the web, and an
independent validator judges the deliverable against the goal before
anything is committed. Rejected work is retried with the validator's
reasoning fed back; approved knowledge enters through the same review
gate as everything else. Misses become gaps. Gaps become research.
Research becomes knowledge that surfaces next time. The memory
compounds where your agents actually work.

## Metacognition: memory that checks its own thinking

Every agent turn carries a self-critique, and a shadow critic reviews
it independently. A metacognitive layer compares the two, calibrates
how much each critic deserves to be trusted from its track record, and
applies a promotion policy to what the agent claims to have learned. In
idle time, convergence scans sweep the bank for contradictions,
duplicates, undiscovered gaps, and tier promotions. Facts carry
epistemic tiers (verified, neutral, quarantined) that move with
evidence, and retrieval shows the tier as a signal the agent can reason
about rather than a filter that hides candidates. Nothing enters
durable memory as a side effect: stated knowledge, inferred knowledge,
and machine-ingested knowledge are provenance-tracked and gated.

## What ships around that core

- **Agent API and drop-in proxy.** First-class agent endpoint with
  tools for memory, retrieval, research, and curation; plus an
  OpenAI-compatible `/v1/chat/completions` for existing clients,
  unchanged.
- **Crystals.** Knowledge clusters with vector embeddings and
  structured sparse keys (Source | Locator | Subject | Domain),
  citations back to the exact stored facts, and per-source identity for
  clean supersede and retire.
- **Ingestion that learns your shapes.** Documents, spreadsheets, chat
  exports, code, watched folders, Drive, and git repos. New JSON shapes
  get one human-reviewed mapping, then every future record of that
  shape ingests mechanically. One judgment per shape of data, ever.
- **Cost accounting built in.** Every model call lands in a ledger per
  customer, per session, per origin, with daily budget stop-losses on
  all background work.
- **Bring your own model.** Anthropic, OpenAI-compatible endpoints, or
  Claude on Google Vertex. Any provider, any model string. One config
  switch, no code changes.
- **Your accommodation, not your migration.** CRYSTAL adapts to your
  data as it is. No schema normalization demanded, no rigid ontology
  imposed.

## Repository layout

```
memory/   the memory server: agent API, proxy, retrieval, self-curation, ingestion
CRYS/     the agent built on it: terminal agent, disposable environments
```

## Self-hosting ships the complete product

The self-host image is the whole thing: all self-curation, cognition,
metacognition, idle-time convergence, citations, cost visibility, and
the operator console. Ungated and inspectable. The hosted platform adds
managed operations, not withheld features.

## Quick start

Requirements: Docker (or Podman with compose), ~2 GB RAM.

```bash
git clone https://github.com/EraHQ/CRYSTAL.git
cd CRYSTAL
cp .env.example .env
echo "CC_TOKEN_ENCRYPTION_KEY=$(openssl rand -hex 32)" >> .env
docker compose up -d
```

This brings up Postgres, the API on **http://localhost:8000**,
background workers, and a bundled zero-key web search provider. No
accounts and no provider API keys are needed to boot. The one secret
above is generated locally and never leaves your machine (it encrypts
stored credentials at rest).

The **Inspector**, CRYSTAL's web console for browsing crystals,
cognition runs, cost, and customers, runs as its own small service:
build it with
`docker build -f deploy/inspector/Dockerfile -t crystal-inspector .`
and point `API_UPSTREAM` at your API, or run it in dev with
`npm run dev` inside `frontend/`.

Create a customer pointing at your upstream LLM. Any provider, any
model string (the `model_id` below is just an example; use whatever
model your provider offers):

```bash
curl -X POST localhost:8000/v1/customers \
  -H 'content-type: application/json' \
  -d '{"provider":"anthropic","model_id":"claude-sonnet-4-5-20250929","api_key_ref":"<your upstream key>"}'
```

Then send chat completions to `/v1/chat/completions` with the returned
`api_key` as your Bearer token. That's it. Memory accrues from there.

### Smallest footprint (single container, SQLite)

```bash
docker build -t crystal .
docker run -p 8000:8000 -v crys-data:/data \
  -e CC_DATABASE_URL=sqlite+aiosqlite:////data/crystal_cache.db \
  -e CC_TOKEN_ENCRYPTION_KEY=$(openssl rand -hex 32) \
  crystal
```

(Persist that key somewhere safe if you keep the volume. It decrypts
the credentials stored in it.)

## Configuration at a glance

Everything is a `CC_*` environment variable. The defaults are safe:
features that spend model calls are **off** until you turn them on;
features that are free are **on**.

| | |
|---|---|
| `CC_DATABASE_URL` | SQLite file by default; compose wires Postgres |
| `CC_LLM_PROVIDER` | `anthropic` \| `openai` \| `vertex` |
| `CC_LLM_MODEL_SMALL/LARGE/FRONTIER` | the model tier mapping |
| `CC_ENABLE_CONVERGENCE_SCAN` | opt-in idle-time contradiction/dedup/gap discovery |
| `CC_DAILY_LLM_BUDGET_USD` | daily stop-loss across all background work |
| `CC_ENABLE_RATE_LIMITING` | on by default, generous limits |

The full configuration reference, deployment guide, upgrade and backup
procedures, and architecture documentation live on the website.

## Security posture

Upstream keys and OAuth tokens are encrypted at rest. The web fetcher
refuses non-public addresses and pins connections against DNS
rebinding. OAuth flows use single-use server-side state. Auth-adjacent
and expensive routes are rate limited. Cross-tenant isolation is
enforced in the store and proven at the HTTP surface by tests in this
repository.

## Development

```bash
pip install -e . --break-system-packages   # or use a venv
pytest -q
```

The test suite is the contract: 1,300+ tests covering the agent, proxy,
memory, cognition, ingestion, workers, and security surfaces.

## License

See [LICENSE](LICENSE).
