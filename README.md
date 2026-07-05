# CRYSTAL

**Where knowledge takes shape.**

CRYSTAL is a self-curating AI memory system. Its core is a **memory
server** — a retrieval proxy that sits between your application and any
upstream LLM, remembers what matters, and makes every future request
smarter than the last — across any workload: support, research,
operations, coding, creative work. **CRYS**, the bundled agent, is the
first application built on that memory.

Point your existing OpenAI-shaped client at the memory server instead of
the provider, and your application gains a persistent, self-improving
knowledge base with zero application changes.

```
Your app ──► CRYSTAL memory ──► any upstream LLM (Anthropic, OpenAI, Vertex)
                  │
                  ▼
        self-curating memory
   (crystals: clustered, keyed,
    cited, cost-accounted facts)
```

## Repository layout

```
memory/   the memory server — proxy, retrieval, self-curation, ingestion
CRYS/     the agent built on it — terminal agent, disposable environments
```

## What the memory server does

- **Drop-in proxy.** OpenAI-compatible `/v1/chat/completions`. Your
  existing SDKs and tools work unchanged; CRYSTAL injects relevant
  knowledge into each request and extracts new knowledge from each
  response.
- **Self-curating memory.** Knowledge is organized into *crystals* —
  clusters of related facts with vector embeddings and structured sparse
  keys. The system identifies gaps in its own knowledge, researches and
  fills them, and validates what it learns before committing it. Misses
  become gaps; gaps get filled; filled knowledge surfaces next time. The
  memory compounds.
- **Grounded citations.** Answers can carry citations back to the exact
  stored knowledge they drew from.
- **Epistemic tiers.** Facts carry a quality tier that moves with
  evidence, so retrieval can distinguish "verified" from "provisional."
- **Cost accounting built in.** Every model call lands in a ledger —
  per customer, per session, per origin. You always know what your
  memory layer spends.
- **Document ingestion.** Upload documents (or connect Google Drive) and
  they are chunked, typed, reviewed, and crystallized into retrievable
  knowledge.
- **Bring your own model.** Anthropic, OpenAI-compatible endpoints, or
  Claude on Google Vertex — any provider, any model string. One config
  switch, no code changes.
- **Your accommodation, not your migration.** CRYSTAL adapts to your
  data as it is. No schema normalization demanded, no rigid ontology
  imposed.

## Self-hosting ships the complete product

The self-host image is the whole thing — all self-curation,
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

This brings up Postgres, the API on **http://localhost:8000** (admin UI
at `/admin`), background workers, and a bundled zero-key web search
provider. No accounts and no provider API keys are needed to boot — the
one secret above is generated locally and never leaves your machine (it
encrypts stored credentials at rest).

Create a customer pointing at your upstream LLM — any provider, any
model string (the `model_id` below is just an example; use whatever
model your provider offers):

```bash
curl -X POST localhost:8000/v1/customers \
  -H 'content-type: application/json' \
  -d '{"provider":"anthropic","model_id":"claude-sonnet-4-5-20250929","api_key_ref":"<your upstream key>"}'
```

Then send chat completions to `/v1/chat/completions` with the returned
`api_key` as your Bearer token. That's it — memory accrues from there.

### Smallest footprint (single container, SQLite)

```bash
docker build -t crystal .
docker run -p 8000:8000 -v crys-data:/data \
  -e CC_DATABASE_URL=sqlite+aiosqlite:////data/crystal_cache.db \
  -e CC_TOKEN_ENCRYPTION_KEY=$(openssl rand -hex 32) \
  crystal
```

(Persist that key somewhere safe if you keep the volume — it decrypts
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
| `CC_ENABLE_RATE_LIMITING` | on by default, generous limits |

The full configuration reference, deployment guide, upgrade and backup
procedures, and architecture documentation live on the website.

## Security posture

Upstream keys and OAuth tokens are encrypted at rest. The web fetcher
refuses non-public addresses and pins connections against DNS rebinding.
OAuth flows use single-use server-side state. Auth-adjacent and
expensive routes are rate limited. Cross-tenant isolation is enforced in
the store and proven at the HTTP surface by tests in this repository.

## Development

```bash
pip install -e . --break-system-packages   # or use a venv
pytest -q
```

The test suite is the contract: 900+ tests covering the proxy, memory,
ingestion, workers, and security surfaces.

## License

See [LICENSE](LICENSE).
