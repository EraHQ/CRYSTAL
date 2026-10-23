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
  keys. The system identifies gaps in its own knowledge, researches them
  on request (or automatically when
  `CC_AUTO_RESEARCH_DEFAULT_MONTHLY_CAP_MICRO_USD` is set), and
  validates what it learns before committing it. Misses
  become gaps; gaps are researched on request (or automatically within a
  monthly allowance you set); filled knowledge surfaces next time. The
  memory compounds.
- **Contradictions, including the ones wording hides.** Facts that
  disagree are surfaced for a decision, never averaged away. Pairs that
  look alike meet a judge that knows each claim's scope (its key path,
  source and date); and every new fact is probed for what it implies,
  those consequences are searched against the bank, and a fact that
  disagrees with a consequence is surfaced too, with the bridge written
  out ("A implies no liquid contains oxygen; B says water does").
- **Consequences are knowledge too.** What a fact implies is kept, not
  thrown away: each consequence that passes a checker is stored beside
  its parent fact with a typed link to every fact it was judged against
  (agrees, unrelated, contradicts). Knowing that two things do not
  contradict is knowledge in itself. A consequence becomes recallable
  once a stated fact agrees with it, marked "derived from" its parent in
  every answer, and takes its parent's standing when a person approves
  it, or when an approval rule you wrote does ("approve once N facts
  agree and nothing contradicts"). No rule means a person approves.
- **Let Crystal run itself.** Switch on the Steward and one standing
  session per idle cycle works the bank with the curation toolset:
  settles conflicts the facts decide, approves what the bank agrees
  with, researches gaps it can answer, leaves the rest for you, and
  writes a brief. Every action is on the Activity page with a 24-hour
  undo; it runs until the day's AI capacity is used, then asks for a
  refill (console and chat). Off by default.
- **Grounded citations.** Answers can carry citations back to the exact
  stored knowledge they drew from.
- **Epistemic tiers.** Facts carry a quality tier that moves with
  evidence (whitelist, neutral, quarantine, blacklist), so a reader can
  tell evidence-backed knowledge from unvetted knowledge.
- **Cost accounting built in.** Every model call lands in a ledger —
  per customer, per session, per origin. You always know what your
  memory layer spends.
- **Document ingestion.** Upload documents and
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

This brings up Postgres, the API on **http://localhost:8000**, background
workers, and a bundled zero-key web search provider. No accounts and no
provider API keys are needed to boot — the one secret above is generated
locally and never leaves your machine (it encrypts stored credentials at
rest).

> **One honest caveat:** a keyless boot serves memory but does not curate
> it. Self-curation — contradiction and duplicate detection, gap discovery
> and gap filling, multi-segment key generation — needs an internal model
> key and stays idle until one is set (`CC_ANTHROPIC_API_KEY=...` in your
> `.env`, then restart). The server tells you: `/health` reports
> `"self_curation": "idle_no_model_key"` and logs one warning at startup
> until the key is in place.

The **Inspector** — CRYSTAL's web console for browsing crystals, cost,
and customers — runs as its own small service: build it with
`docker build -f deploy/inspector/Dockerfile -t crystal-inspector .` and
point `API_UPSTREAM` at your API, or run it in dev with `npm run dev`
inside `frontend/`.

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
background scans are **on**, capped by `CC_CONVERGENCE_MAX_CALLS_PER_DAY`
(500); features that spend frontier-tier model calls on their own are
**off** until you turn them on; features that are free are **on**.

| | |
|---|---|
| `CC_DATABASE_URL` | SQLite file by default; compose wires Postgres |
| `CC_LLM_PROVIDER` | `anthropic` \| `openai` \| `vertex` |
| `CC_LLM_MODEL_SMALL/LARGE/FRONTIER` | the model tier mapping |
| `CC_ENABLE_CONVERGENCE_SCAN` | on by default: idle-time contradiction/dedup/gap discovery, capped per day |
| `CC_ENABLE_RATE_LIMITING` | on by default, generous limits |

The full configuration reference, deployment guide, upgrade and backup
procedures, and architecture documentation live on the website.

## NVIDIA models and Nebius Token Factory

CRYSTAL's model seam speaks two protocols: Anthropic Messages and any
OpenAI-compatible chat endpoint. Nebius Token Factory is OpenAI-compatible,
so every model call in CRYSTAL (the agent turn, background cognition,
document extraction, the curation scans, assumption responses) can run on
NVIDIA Nemotron through Token Factory with configuration, not a new SDK.
The hosted demo tenant runs on this lane.

| CRYSTAL tier | Model on Token Factory | Used for |
|---|---|---|
| small | `nvidia/Nemotron-3_5-Lightning` | quick, repetitive steps: classifying one fact, naming a subject, planning |
| large | `nvidia/nemotron-3-super-120b-a12b` | answers, composition, extraction, the curation judgments |
| frontier | `nvidia/Nemotron-3-Ultra-550b-a55b` | the cognition validator (reads the whole deliverable and decides whether it ships) and outbound review |

How the lane works:

- **Any model in the catalog.** CRYSTAL reads `GET /v1/models?verbose=true`
  with the platform key and lets a tenant pick any text model Token
  Factory lists, Nemotron or not. The listing's context length and
  pricing feed the picker and the spend ledger directly, so a pick is
  metered at its real price and the same daily capacity applies
  (`memory/src/crystal_cache/llm/catalog.py`). The listing's price unit
  is calibrated against one hand-verified anchor before any row is used.
- **Probed, not assumed.** Token Factory does not say per model whether
  `response_format: json_schema` or tool calling is honoured, so CRYSTAL
  asks: two tiny calls when a model is picked. A model that will not make
  a tool call is refused as the answers model (code `model_no_tools`); a
  model without `json_schema` still works for background curation (the
  seam sends `json_object` with the schema in the prompt).
- **Reasoning only where it pays.** Nemotron reasons by default; CRYSTAL
  sends `reasoning_effort: none` on structured-output and small-tier
  calls and leaves the model default for free-text composition.
- **Background on flex.** Curation and cognition calls carry
  `service_tier: flex` (no rate-limit credits consumed, lower priority);
  answers never do. `CC_NEBIUS_BACKGROUND_SERVICE_TIER=` turns it off.
- **Memory from what you see.** A photo of a whiteboard, a screenshot or
  a chart uploads like any document (png, jpg, webp, also from a watched
  source). On the Nemotron lane `nvidia/Nemotron-3.5-Super-VL` reads it:
  the text verbatim, what the image shows, and the facts worth keeping,
  which then go through the same extraction and review as a PDF, citing
  the image as their source. On the Anthropic lane the large tier reads
  images natively. With no model that reads images the upload is refused
  (code `image_needs_vision`), never stored as an empty document.
- **Your own Token Factory key.** In Settings, save a key with provider
  "Nebius Token Factory": CRYSTAL checks it against `/v1/models` before
  storing it (encrypted under the tenant's own key), then both answers
  and background curation run on it. Those calls are ledgered as yours
  and never count toward the hosted daily capacity. The same path takes
  an Anthropic, OpenAI or self-hosted OpenAI-compatible key.

To run a tenant, or a whole self-hosted instance, on this lane: set
`CC_NEBIUS_API_KEY` (and `CC_NEBIUS_BASE_URL` if not the default), pick
Nemotron ids with `CC_NEMOTRON_MODEL_SMALL/LARGE/FRONTIER` if you want
different ones, restart the services, and choose the models in Settings.
Clearing the key turns the lane off; every pick falls back to the default.

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
