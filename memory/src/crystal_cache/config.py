"""Global settings for the Crystal Cache application.

Settings are loaded from environment variables via pydantic-settings. A
`.env` file in the project root is respected.

Customer-scoped settings (thresholds, injection preference, shadow rate)
live on the Customer model, NOT here. This config is for process-wide
state: DB connection, ports, upstream model defaults, feature flags.

Usage:
    from crystal_cache.config import settings
    print(settings.database_url)

For tests, you can construct Settings directly with overrides:
    from crystal_cache.config import Settings
    test_settings = Settings(database_url="sqlite+aiosqlite:///:memory:")
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


Environment = Literal["development", "staging", "production", "test"]


class Settings(BaseSettings):
    """Process-wide configuration.

    Environment variables use the prefix `CC_` (e.g. `CC_DATABASE_URL`).
    """

    model_config = SettingsConfigDict(
        env_prefix="CC_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Deployment
    environment: Environment = "development"
    log_level: str = "INFO"

    # API server
    host: str = "127.0.0.1"
    port: int = 8000

    # Database — default to SQLite for local dev, Postgres in prod
    database_url: str = "sqlite+aiosqlite:///./crystal_cache.db"
    database_echo: bool = False

    # Vector store — separate from metadata store so we can pick independently
    # For MVP, re-uses database_url (pgvector extension in the same Postgres).
    # If set, overrides and uses a separate vector DB.
    vector_store_url: str | None = None

    # Vector index backend (Step 2 — fact lane on Qdrant). Default "memory"
    # keeps the in-memory FactVectorStore/VectorStore (today's behavior). Set
    # "qdrant" to put the FACT lane on Qdrant (routing/summary still delegate
    # to the in-memory stores this slice). The client targets qdrant_url (a
    # running server, e.g. http://localhost:6333) when set; otherwise
    # qdrant_location (an embedded path, or ":memory:") — defaulting to an
    # ephemeral in-memory Qdrant, which is fine for dev but NOT production (set
    # qdrant_url there). qdrant_collection names the single payload-partitioned
    # facts collection.
    #   CC_VECTOR_BACKEND / CC_QDRANT_URL / CC_QDRANT_LOCATION / CC_QDRANT_COLLECTION
    vector_backend: Literal["memory", "qdrant", "sqlite_vec"] = "memory"
    qdrant_url: str | None = None
    qdrant_location: str | None = None
    qdrant_collection: str = "crys_facts"
    # Routing lane (10k) on Qdrant (Step 2b): a SECOND collection, binary-
    # quantized with a float rescore (Step-0 proven recall@1 = 1.000 at 10k).
    # qdrant_routing_oversampling is the rescore pool multiplier (Qdrant pulls
    # k*oversampling candidates by 1-bit Hamming, then rescores on the full
    # float vector) — the latency/recall dial, default 2.0 (the benchmarked
    # value). Only used when vector_backend == "qdrant".
    #   CC_QDRANT_ROUTING_COLLECTION / CC_QDRANT_ROUTING_OVERSAMPLING
    qdrant_routing_collection: str = "crys_routing"
    qdrant_routing_oversampling: float = 2.0

    # Embedding / HDC dimensions
    d_model: int = 1024  # model hidden size (Qwen3-0.6B = 1024)
    d_hdc: int = 10_000  # HDC dimensionality

    # Default similarity thresholds (customers can override per-account).
    #
    # CALIBRATED FOR SEMANTIC ENCODER (gtr-t5-base + P-projection).
    # Hash-encoder banks ran with ~0.90 / ~0.50; semantic banks live in
    # a tighter range. From spike v2 (April 2026): on-topic FAQ matches
    # land at cosine ~0.40-0.55, off-topic at ~0.05-0.15. The old
    # hash-era 0.90 / 0.50 thresholds will NEVER trigger 'high' in this
    # regime, so we use 0.45 / 0.20 by default.
    #
    # If running the legacy hash encoder, override these via env to
    # CC_DEFAULT_TAU_HIGH / CC_DEFAULT_TAU_LOW — or better, override per
    # customer through retrieval_thresholds.
    default_tau_high: float = 0.45
    default_tau_low: float = 0.20
    default_tau_conf: float = -0.3  # lp threshold for confidence gate (separate signal)
    default_shadow_sample_rate: float = 0.05

    # Background workers
    worker_poll_interval_seconds: int = 5

    # Worker-process split (A.5 / WS E). When True (default) the API
    # lifespan spawns the background workers in-process — the single-
    # container / single-process deployment. Set False to run the API
    # WITHOUT workers: the docker-compose split runs a dedicated worker
    # container (`python -m crystal_cache.workers`) against the same
    # database while the API process serves requests only. Either way the
    # database tables are the queue, so the two processes stay decoupled.
    #
    #   CC_RUN_WORKERS
    run_workers: bool = True

    # Feature flags — gate the research paths per BUILD_PROPOSAL.md §9.
    # (The hidden-state / confidence-gate flags were removed in the
    # launch-prep purge 2026-07-02 with their stub modules — the parked
    # research line is documented in docs/RESEARCH_DIRECTIONS.md.)
    enable_diagnostic_engine: bool = True  # Path M runs against rich-sweep-v1

    # Decomposer (concept-path bridge) — see src/crystal_cache/decomposer/
    # The decomposer turns free text into structured DSL-shaped payloads.
    # We default to Groq-hosted Llama 3.1 8B Instant, but any
    # OpenAI-compatible endpoint works (llama.cpp server, Together,
    # Fireworks, etc.) by overriding `decomposer_base_url`.
    groq_api_key: str | None = Field(
        default=None,
        # Accept either CC_GROQ_API_KEY (prefixed, like other settings)
        # or the bare GROQ_API_KEY (conventional for Groq). Either works.
        validation_alias=AliasChoices("CC_GROQ_API_KEY", "GROQ_API_KEY"),
    )  # required for HostedLLMDecomposer
    decomposer_base_url: str = "https://api.groq.com/openai/v1"
    decomposer_model: str = "llama-3.1-8b-instant"
    decomposer_timeout_seconds: float = 10.0
    decomposer_max_retries: int = 2
    # JSONL file where successful decomposer calls are logged as
    # training data for the eventual distilled classifier. None disables.
    decomposer_trace_path: str | None = None

    # Anthropic API key for the failure-reflection path in the
    # production crystallizer (Phase 0.2 of the bind-storage rebuild).
    # When a customer's pipeline records a wrong answer (thumbs-down
    # feedback row), the crystallizer fires one Haiku call asking
    # "what one-sentence imperative rule would have prevented this?"
    # The rule is stored as a `failed_reasoning` crystal that future
    # similar queries can consult.
    #
    # Cost: ~$0.001 per failure. Latency: not on the request path —
    # crystallization runs async via the diagnostic loop.
    #
    # If unset, the crystallizer falls back to storing the raw
    # truncated failure trace instead of a reflection. That's the
    # behavior described in CLAUDE.md item 12 of Phase 1 cleanup as
    # the v0 fallback when reflection isn't wired.
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "CC_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"
        ),
    )
    # Model used by the reflection client. Defaults to Haiku 4.5 because
    # reflections are short, mechanical, and benefit from speed/cost
    # over reasoning depth. Per CLAUDE.md Rule 13, Haiku 4.5's exact
    # API string is `claude-haiku-4-5-20251001` (date snapshot).
    reflection_model: str = "claude-haiku-4-5-20251001"

    # --- Provider-neutral LLM seam (the swappable reasoning backend) -----
    # Crystal's reasoning calls route through crystal_cache.llm.get_llm_client,
    # which picks a provider here and resolves models by TIER (small / large /
    # frontier) so no provider-specific model string lives at the call site.
    # Anthropic is the default; any OpenAI-compatible chat-completions endpoint
    # (OpenAI, Groq, Together, Fireworks, a local llama.cpp / Ollama server) is
    # a config change, not a code change.
    #   llm_provider  - "anthropic" (default) | "openai" (any OpenAI-compatible
    #                   endpoint, selected together with llm_base_url).
    #   llm_api_key   - the provider key. For Anthropic this is OPTIONAL and
    #                   falls back to anthropic_api_key (so CC_ANTHROPIC_API_KEY
    #                   / ANTHROPIC_API_KEY keep working); for openai it is the
    #                   endpoint key (may be blank for a keyless local server).
    #   llm_base_url  - required for the openai provider (e.g.
    #                   https://api.openai.com/v1 , http://localhost:11434/v1).
    #   llm_model_small / _large / _frontier - per-tier model id. Anthropic has
    #                   built-in defaults (Haiku / Sonnet / Opus); a non-
    #                   Anthropic provider MUST set the tiers it uses.
    # Env: CC_LLM_PROVIDER / CC_LLM_API_KEY / CC_LLM_BASE_URL /
    #      CC_LLM_MODEL_SMALL / CC_LLM_MODEL_LARGE / CC_LLM_MODEL_FRONTIER
    llm_provider: Literal["anthropic", "vertex", "openai"] = "anthropic"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    # Vertex provider (2026-07-03, GCP consolidation): Claude-on-Vertex via
    # the Anthropic SDK's AnthropicVertex — the same Messages API billed to
    # GCP and authenticated with Application Default Credentials (no API
    # key). Requires the SDK vertex extra (pip install anthropic[vertex] or
    # the project's [vertex] optional group), CC_VERTEX_PROJECT, and
    # CC_VERTEX_REGION (a Claude-serving region, e.g. us-east5). Per-tier
    # models MUST be set via CC_LLM_MODEL_SMALL/_LARGE/_FRONTIER using
    # Vertex model ids (e.g. claude-sonnet-4-5@20250929) — there are no
    # universal defaults on this provider. Gemini as the small tier is a
    # documented follow-on (needs OAuth token-refresh plumbing on the
    # openai-compatible endpoint; LAUNCH_CHECKLIST §B).
    vertex_project: str | None = None
    vertex_region: str | None = None
    llm_model_small: str | None = None
    llm_model_large: str | None = None
    llm_model_frontier: str | None = None

    # MCR shadow critic (Phase 9.5, May 2026).
    #
    # The shadow critic is the second MCR critic (MCR_FRAMEWORK.md §5.2,
    # D-MCR-10). It runs SAMPLED (not always-on) and reviews a persisted
    # reasoning trace + its agent_self critique, producing a
    # Critique(critic_role="shadow"). Per §5.2 the shadow should run on a
    # FRONTIER model — better than the agent's self-critique tier (Haiku)
    # — so its second opinion adds value over the self-critique.
    #
    # shadow_critic_model defaults to the current flagship Opus. Operators
    # can override with a date-snapshotted id via CC_SHADOW_CRITIC_MODEL.
    # If the configured model is unavailable the shadow critic logs a
    # warning and the LLM call fails gracefully (empty critique with a
    # failure summary per the mcr_emitter NEVER-raises discipline).
    #
    # shadow_critic_sample_rate is the Bernoulli rate for CLEAN traces
    # (self-critique flagged nothing). Traces whose self-critique flagged
    # >=1 observation are ALWAYS shadowed (the agent already self-flagged
    # — worth a second opinion). §11 Q3 initial proposal: 1-5% random +
    # always-on for self-flagged uncertainty. Default 0.05 (5%).
    #
    # Cost budgeting (§11 Q10) is a SOFT target in Phase 9.5 — the rate is
    # exposed but no hard per-customer cap is enforced. The hard cap lands
    # with Phase 10's scheduler (which tracks per-window spend).
    shadow_critic_model: str = "claude-opus-4-8"
    shadow_critic_sample_rate: float = 0.05

    # MCR shadow critic cost cap (Phase 10B, P0.80, MCR §11 Q10).
    #
    # Per-customer hard cap on shadow critique calls per rolling
    # 24-hour window. The Phase 10B metacognition worker counts
    # critiques with critic_role='shadow' created in the last 24h
    # per customer and skips new shadow calls when at/over this cap.
    #
    # Default 100/day at the typical Opus price (≈$0.05/call worst-
    # case) caps spend at ≈$5/customer/day. Per-customer overrides
    # may land in Phase 11+ via a Customer column (mirrors the
    # existing per-customer shadow_sample_rate override).
    shadow_max_per_customer_per_day: int = 100

    # Phase 10C metacognition worker lifespan switch (P0.87).
    #
    # When True (default), the lifespan spawns
    # `run_metacognition_worker` as a background task at startup.
    # When False, the worker module remains importable but is not
    # auto-started — operators can invoke it manually for testing
    # or special-case backfill.
    #
    # Default ON matches the Phase 9C pattern of "ship the code
    # with a settings flag, default to ON in the same phase." The
    # off-switch lets operators disable without code changes if
    # cost or LLM-availability concerns arise.
    enable_metacognition_worker: bool = True

    # Benchmark / single-pass mode: when True, the chat proxy does NOT
    # inject the crystal push/pull tools into upstream requests. Used by
    # the LongMemEval harness to honor the benchmark's single-pass
    # contract (no pull_research / pull_expand second chances). Env:
    # CC_DISABLE_CRYSTAL_TOOLS. Default off — normal operation injects.
    disable_crystal_tools: bool = False

    # Growth G1 (citations). When True, the chat proxy tags injected
    # knowledge with citation handles ([[cc:N]]) + a cite instruction so the
    # model can attribute its sources, then (post-response, non-streaming)
    # parses the citations, grounds each against the cited crystal, renders a
    # provenance footer, and records the result for the G4 metering rail.
    # LAUNCH DEFAULT ON (flag-stance pass 2026-07-02): the trust story +
    # the tier-promotion signal. Env: CC_ENABLE_CITATIONS.
    enable_citations: bool = True

    # Growth G2 (control plane). The server-side command channel
    # (endpoints/control.py) is always available; these gate the AGENT-side
    # poll/verify behavior (built but pending live wiring, like the F4 runtime).
    #   control_plane_enabled       — master switch for the agent's poll loop.
    #   control_decision_timeout_seconds — no-decision timeout → DENY (never
    #                                 approve). The agent enforces it while
    #                                 blocking at an approval gate.
    #   control_require_signature   — when True the agent rejects a decision
    #                                 whose Ed25519 signature doesn't verify
    #                                 against the operator's pinned key
    #                                 (fail-closed). Dev default False so the
    #                                 state machine is exercisable without a
    #                                 provisioned signing key; production sets
    #                                 True. WebAuthn is the recommended prod
    #                                 anchor (control/signing.py).
    # Env: CC_CONTROL_PLANE_ENABLED / CC_CONTROL_DECISION_TIMEOUT_SECONDS /
    #      CC_CONTROL_REQUIRE_SIGNATURE.
    control_plane_enabled: bool = False
    control_decision_timeout_seconds: int = 300
    control_require_signature: bool = False

    # Growth G3 (cost accounting + budgets). When enabled, the chat proxy
    # emits one llm_calls cost row per upstream call (record_llm_call). OFF by
    # default — additive + fail-safe; flip on after live validation.
    #   enable_cost_accounting      — gate the proxy cost emitter.
    #   llm_price_table_overrides   — optional per-model price overrides merged
    #                                 over cost/pricing.py defaults. Shape:
    #                                 {model: {input, output, cache_creation?,
    #                                 cache_read?}} in micro-USD per Mtok
    #                                 (integer; money is never a float). None =
    #                                 use the built-in dev placeholder rates.
    #   per_session_budget_micro_usd / daily_team_budget_micro_usd — caps in
    #                                 micro-USD (0 = no cap). Read by the budget
    #                                 check; the breach → G2 auto-pause wiring
    #                                 is deferred (needs the agent-side channel).
    # Env: CC_ENABLE_COST_ACCOUNTING / CC_LLM_PRICE_TABLE_OVERRIDES /
    #      CC_PER_SESSION_BUDGET_MICRO_USD / CC_DAILY_TEAM_BUDGET_MICRO_USD.
    # LAUNCH DEFAULT ON (flag-stance pass 2026-07-02): pure visibility,
    # verified rates, no model calls.
    enable_cost_accounting: bool = True
    llm_price_table_overrides: dict | None = None
    per_session_budget_micro_usd: int = 0
    daily_team_budget_micro_usd: int = 0

    # Growth G4 (marketplace metering). When enabled, a grounded citation of a
    # general/marketplace crystal mints a shard credit (record_citation_credit)
    # in the proxy citation block — grounding-gated, self-traffic excluded,
    # idempotent. OFF by default.
    #   enable_marketplace_metering — gate the citation→credit hook.
    #   marketplace_convertibility_enabled — shards offsetting subscription.
    #                                 EXPLICITLY OFF at launch (the design gates
    #                                 convertibility behind metering that has
    #                                 survived adversarial traffic); the spend
    #                                 substrate exists but is not wired to
    #                                 billing.
    # Env: CC_ENABLE_MARKETPLACE_METERING / CC_MARKETPLACE_CONVERTIBILITY_ENABLED.
    enable_marketplace_metering: bool = False
    marketplace_convertibility_enabled: bool = False

    # Never-Idle Convergence (contradiction scan). The convergence half of the
    # accommodation thesis (docs/NEVER_IDLE_CONVERGENCE.md): the cognition
    # worker's idle Phase 3 scans a customer's own facts for contradictions and
    # surfaces knowledge_conflicts (surfacing-only — no destructive writes).
    #
    # enable_convergence_scan gates the AUTONOMOUS path (the worker's idle
    # pass) only. It is OFF by default — like enable_citations / cost /
    # marketplace, a new always-on LLM-spend feature ships off and flips on
    # after live validation. The on-demand paths (`python -m crystal_code
    # --audit` and POST /admin/api/conflicts/scan) run regardless of this flag
    # (explicit operator action), respecting the budget below.
    #
    # Budget gate (D7): the worker spends at most convergence_max_calls_per_cycle
    # discriminator calls per idle cycle, and at most convergence_max_calls_per_day
    # per UTC day (a process-local ceiling, reset at the day boundary — the
    # gap_backoff precedent). convergence_max_pairs_per_scan is the D2 candidate
    # cap; convergence_customers_per_cycle is how many customers the idle pass
    # scans per cycle (round-robin across cycles for fairness).
    #   CC_ENABLE_CONVERGENCE_SCAN / CC_CONVERGENCE_MAX_PAIRS_PER_SCAN /
    #   CC_CONVERGENCE_MAX_CALLS_PER_CYCLE / CC_CONVERGENCE_MAX_CALLS_PER_DAY /
    #   CC_CONVERGENCE_CUSTOMERS_PER_CYCLE
    # LAUNCH DEFAULT ON (flag-stance pass 2026-07-02): the never-idle
    # self-curation story, bounded by the shared daily call ceiling.
    enable_convergence_scan: bool = True
    convergence_max_pairs_per_scan: int = 200
    convergence_max_calls_per_cycle: int = 50
    convergence_max_calls_per_day: int = 500
    convergence_customers_per_cycle: int = 3

    # Load-aware idle gate (workers/idle.py, 2026-07-02 — BACKLOG §3
    # remainder). The worker's opportunistic idle work (gap fill, the
    # convergence scans, tier promotion) waits until the API has been quiet
    # for this many seconds (substantive /v1/* traffic only). 0 disables the
    # gate. In the split-process compose shape the stamp is per-process, so
    # the gate is inert there (documented in the module) — the cross-process
    # signal is the noted follow-up.  CC_IDLE_QUIET_SECONDS
    idle_quiet_seconds: int = 30

    # Dedup + gap-discovery generators (P5 fast-follow). Each is independently
    # gated and OFF by default (same posture as enable_convergence_scan); they
    # run as sibling idle Phase-3 passes after the contradiction scan and SHARE
    # the convergence budget knobs above. The per-UTC-day discriminator-call
    # ceiling (convergence_max_calls_per_day) is ONE shared total across all
    # three scans, so enabling more generators bounds — never multiplies — the
    # daily cost. The dedup scan reuses convergence_max_pairs_per_scan /
    # _max_calls_per_cycle (it is pairwise like the contradiction scan);
    # gap_discovery_max_subjects_per_cycle is the gap-discovery pass's per-cycle
    # budget (it spends one model call per subject, not per pair).
    #   CC_ENABLE_DEDUP_SCAN / CC_ENABLE_GAP_DISCOVERY /
    #   CC_GAP_DISCOVERY_MAX_SUBJECTS_PER_CYCLE
    # LAUNCH DEFAULTS ON (flag-stance pass 2026-07-02): same shared ceiling.
    enable_dedup_scan: bool = True
    enable_gap_discovery: bool = True
    gap_discovery_max_subjects_per_cycle: int = 20

    # Tier promotion (launch-prep sweep, 2026-07-02) — quality tiers that
    # MOVE. No model calls: promotes on grounded citations + age + zero open
    # conflicts (quarantine→neutral→whitelist, one rung per pass) and
    # demotes whitelist→neutral on an open conflict. Runs in the cognition
    # worker's idle Phase 3 alongside the convergence scans; blacklist is
    # human-set and never touched. Retrieval does not consume the tier yet
    # (BACKLOG §13 follow-on).
    #   CC_ENABLE_TIER_PROMOTION / CC_TIER_PROMOTION_MIN_CITATIONS /
    #   CC_TIER_PROMOTION_MIN_AGE_DAYS
    # LAUNCH DEFAULT ON (flag-stance pass 2026-07-02): zero model calls;
    # upward movement needs enable_citations ON (grounded citations).
    enable_tier_promotion: bool = True
    tier_promotion_min_citations: int = 3
    tier_promotion_min_age_days: int = 7

    # System rules (2026-07-03): the user-owned judgment-automation layer
    # (system_rules/). The idle pass evaluates each customer's enabled
    # 'promotion' rules against their recall-gated crystals, clearing gates
    # where the user's conditions hold. Runs in the cognition worker's idle
    # family next to tier promotion. SAFE BY CONSTRUCTION: absent any rule
    # nothing is promoted (human approval stays the default); rules only
    # loosen toward usable, never touch blacklist, and every fire is
    # audited. Zero model calls, so no budget from the shared ceiling.
    #   CC_ENABLE_SYSTEM_RULES
    # LAUNCH DEFAULT ON (matches the flag-stance pass): a no-op until a
    # user writes a rule.
    enable_system_rules: bool = True

    # Outbound review (2026-07-03, RATIFIED): background-worker output is
    # reviewed by a HIGH-TIER model and/or a human before it can be relied
    # on — behind an explicit option + flag. This gates the model half
    # (scan/outbound_review.py): an idle pass that walks recall-gated
    # background-worker crystals with no verdict and stamps
    # outbound_scan_passed / outbound_scan_failed. A PASS requires the
    # model (or a human) — the deterministic injection screen can only
    # FAIL; regex finding nothing is not a review. OFF BY DEFAULT because
    # it spends frontier-tier model calls (the explicit-opt-in posture of
    # the convergence scans); the promotion path stays human-only until
    # the operator turns it on.
    #   CC_ENABLE_OUTBOUND_SCAN / CC_OUTBOUND_SCAN_MAX_CRYSTALS_PER_CYCLE
    enable_outbound_scan: bool = False
    outbound_scan_max_crystals_per_cycle: int = 20

    # Rate limiting (C3, 2026-07-03): in-process sliding-window guard on
    # auth-adjacent (customer creation, Drive OAuth) and expensive
    # (completions, documents, retrieval) routes. Keyed per bearer token
    # (hashed) else per client IP. Defaults are generous — normal use never
    # trips them; they exist to bound abuse of a public endpoint. Set a
    # limit to 0 to make that class unlimited.
    #   CC_ENABLE_RATE_LIMITING / CC_RATE_LIMIT_AUTH_PER_MINUTE /
    #   CC_RATE_LIMIT_EXPENSIVE_PER_MINUTE
    enable_rate_limiting: bool = True
    rate_limit_auth_per_minute: int = 20
    rate_limit_expensive_per_minute: int = 120

    # Admission (Phase 3 G6, 2026-07-03): the tier used when a hosted
    # tenant has no subscription_tier set (and the fallback for unknown
    # tier names). Self-host never consults tiers.
    #   CC_DEFAULT_SUBSCRIPTION_TIER
    default_subscription_tier: str = "free"

    # Decay (ratified 2026-07-02: 30 days): a whitelist crystal
    # with no grounded citation inside the window drifts back to neutral —
    # trust must stay earned. Staleness never demotes below neutral and
    # never deletes.  CC_TIER_PROMOTION_DECAY_DAYS
    tier_promotion_decay_days: int = 30

    # Ingest scope (P2, ratified 2026-07-02): the deployment default for
    # authored knowledge — 'personal' (owner-only; private by
    # default) or 'team' (group-readable, the pre-P2 behavior). Per-request
    # override via the /v1/store scope field; scope-on-sources (drive,
    # documents) extends this per source. P1's identity chain makes
    # personal always well-defined: team keys act as the Default Admin,
    # who reads everything in its team regardless (admin = root), so
    # team-key workflows are unaffected; child-operator readers see only
    # what they own or what is shared. CC_DEFAULT_INGEST_SCOPE
    default_ingest_scope: str = "personal"

    # Topic seeding (BACKLOG §3 remainder, 2026-07-02) — research seeds
    # WITHOUT model calls: thin crystals (few facts) and an operator topic
    # list each write knowledge_gaps rows the Phase-2 fill sweep already
    # consumes, so all resulting model spend stays inside the fill sweep's
    # existing budget. research_topics is comma-separated; empty = the
    # topic half is inert. LAUNCH DEFAULT ON: the thin half is store-signal
    # only, flood-guarded by the open-gap cap.
    #   CC_ENABLE_TOPIC_SEEDING / CC_RESEARCH_TOPICS /
    #   CC_THIN_CRYSTAL_MAX_FACTS / CC_TOPIC_SEED_MAX_PER_CYCLE /
    #   CC_TOPIC_SEED_OPEN_GAP_CAP
    enable_topic_seeding: bool = True
    research_topics: str = ""
    thin_crystal_max_facts: int = 2
    topic_seed_max_per_cycle: int = 3
    topic_seed_open_gap_cap: int = 20

    # Web search (launch-prep sweep, 2026-07-02) — the provider seam behind
    # the web_search tool and cognition's research steps (search/web.py).
    # "" = unconfigured: the tool returns an EXPLICIT error result instead
    # of an empty-success lie, so the model can replan.
    #   searxng — self-hosted meta-search (CC_WEB_SEARCH_URL, no key): the
    #             self-host / air-gap story. Snippets only.
    #   tavily  — hosted, returns extracted page CONTENT per result
    #             (CC_WEB_SEARCH_API_KEY): crystallization-grade evidence.
    # DIRECTION: paid services are a bridge, not the destination — the end
    # state is zero paid external services (SearXNG + our own fetch and
    # extraction layer, BACKLOG §13); the only bill is our own servers.
    #   CC_WEB_SEARCH_PROVIDER / CC_WEB_SEARCH_URL /
    #   CC_WEB_SEARCH_API_KEY / CC_WEB_SEARCH_MAX_RESULTS
    web_search_provider: str = ""
    web_search_url: str | None = None
    web_search_api_key: str | None = None
    web_search_max_results: int = 5
    # Level 2 (2026-07-02): our own guarded fetch + extraction upgrades
    # snippet-only results to content-grade — the zero-paid-services end
    # state (SearXNG + this replaces tavily). Max pages fetched per search;
    # 0 disables. SSRF-guarded (search/fetch.py).  CC_WEB_SEARCH_FETCH_PAGES
    web_search_fetch_pages: int = 3

    # Text encoder — picks the implementation used by the retrieval hot
    # path (HashTextEncoder or SemanticTextEncoder).
    #
    # DEFAULT IS "semantic" as of April 2026. The hash encoder is kept
    # for back-compat with banks built before the semantic switch — if
    # CC_TEXT_ENCODER=hash, the legacy bag-of-tokens encoder loads. Don't
    # mix banks: a customer whose crystals were imported with one
    # encoder must be queried with the same one or scores collapse.
    #
    # The semantic encoder uses gtr-t5-base by default (see semantic_model)
    # and lifts vectors into d_hdc-space via the same P-projection that
    # KnowledgeCrystal uses for HDC math. This is what the April 2026
    # decoder fine-tunes (text-v1, bind-v1) were trained against.
    text_encoder: Literal["hash", "semantic"] = "semantic"
    # Model id for SemanticTextEncoder. Default is gtr-t5-base — if
    # changed, decoder fine-tunes will need to be retrained on the new
    # model's embeddings or decoding will be nonsense.
    semantic_model: str | None = None

    # Synthesis cleanup-alpha (Findings 14 + 15, April 2026).
    #
    # As of Finding 15 the synthesis primitive is BUNDLE
    # (`proj_a + proj_b`) instead of bind (`proj_a * proj_b`). Bundle
    # vectors stay ~96% on-manifold; the alpha cleanup that was
    # critical for bind (manifold fraction 0.15) does essentially
    # nothing for bundle. Default is now 0.0 (no cleanup).
    #
    # The cleanup wiring is preserved end-to-end (config setting,
    # synthesize_joint_statement kwargs, _build_manifold_basis,
    # _apply_cleanup) for two reasons:
    #   1. Experimentation — if we re-activate the bind path or build
    #      new operators, the alpha knob is already there.
    #   2. Larger banks — with 100+ FAQs the manifold rank grows;
    #      the bundle vector may have less of its mass on-manifold,
    #      and alpha may matter again.
    #
    # Set CC_SYNTHESIS_CLEANUP_ALPHA=0.75 to reproduce the bind+cleanup
    # behavior we shipped in Finding 14 before Finding 15 swapped the
    # primitive (useful for A/B testing or debugging regressions).
    synthesis_cleanup_alpha: float = 0.0

    # Phase 1.3 write-side routing (April 2026).
    #
    # `add_pair_for_customer` routes a new (prompt, answer) pair into
    # the customer's existing crystal bank: if the prompt's HDC vector
    # has cosine >= bond_threshold against the top-1 crystal AND that
    # crystal isn't at hard-ceiling capacity, bind into top-1; else
    # spawn a fresh crystal.
    #
    # Default 0.65 per resolved decision 6 of the build spec
    # (BIND_STORAGE_REBUILD.md). Intentionally LOWER than the
    # read-side perfect_margin / tau_high (0.45) because:
    #   - Read-side thresholds are calibrated for "is this query
    #     answered by this crystal". Conservative; want few false
    #     positives.
    #   - Write-side bonding wants pairs to coalesce somewhat
    #     aggressively into existing crystals so banks stay
    #     organized. False bonds at 0.65 produce
    #     somewhat-related-but-mixed crystals; the eviction loop
    #     (Phase 7.5) is designed to surface those for split.
    #
    # Per resolved decision 6's TODO: pin empirically against the
    # FAQ bank during Phase 1 stress tests, then re-tune at Phase 6.3
    # (bank-scale validation). The 0.65 default is a working guess,
    # not a measured value.
    bond_threshold: float = 0.65

    # Phase 2 read-side cleanup threshold (April 2026).
    #
    # `recall_from_crystal` runs unbind + reverse-project + cleanup
    # against the crystal's per-pair codebook (Fact.vector entries).
    # The cleanup step finds the nearest-neighbor Fact by cosine in
    # native space (768-dim). When the best cosine is below this
    # threshold, recall returns None and the pipeline falls through
    # to LOW_CONFIDENCE: the routing was right (PERFECT decision)
    # but cleanup couldn't pin down a specific pair, which is the
    # right signal that we should skip injection rather than emit
    # noise.
    #
    # Default 0.3 is a bootstrap fallback. The proper number is
    # PER-CRYSTAL: the recoverability margin (lowest-stored-pair
    # cosine vs highest-crosstalk-pair cosine) varies by how many
    # pairs are bundled and how semantically similar the answers
    # are. Phase 6.3 ("Bank-scale validation") will introduce a
    # calibrator that walks each crystal's facts, replays the
    # unbind per pair, and pins per-crystal thresholds; until then,
    # the global 0.3 covers most crystals reasonably well.
    #
    # Why 0.3 and not 0.65 (bond_threshold) or 0.5 (a midpoint):
    # the recovered native vector after unbind+reverse-project is
    # noisier than a freshly-encoded query vector (bind preserves
    # one stored pair's signal but spreads N-1 pairs as crosstalk).
    # Real per-pair recoverability cosines from the 30-pair recall
    # validation (test_30_pair_crystal_recovers_each_answer_via
    # _unbind_cleanup) sit in the 0.4-0.7 band; crosstalk noise
    # sits below 0.2. 0.3 is comfortably above the noise floor and
    # below the typical recovery, so it accepts genuine matches
    # and rejects pure noise. Tune up per-customer if false
    # positives become an issue; tune down if false negatives do.
    cleanup_threshold: float = 0.3

    # Mode 9 fix (May 2026): recovery ratio floor.
    #
    # ||a_projected|| / ||summary_vec|| measures how much energy the
    # unbind recovered relative to the crystal's bundle size. On-domain
    # queries produce ratio ~1.3-1.6; off-domain ~0.9-1.2 (calibrated
    # on BCB bank, diagnose_mode9_magnitudes.py). Default 1.25 rejects
    # clear off-domain queries while accepting borderline ones.
    #
    # Set CC_RECOVERY_RATIO_FLOOR=0.0 to disable the gate entirely
    # (equivalent to pre-May-2026 behavior).
    recovery_ratio_floor: float = 1.25

    # Hybrid rank — code/prose retrieval calibration (June 2026, CRYS
    # live test). The agent-surface ContentRouter ranks content_chunk
    # facts by raw gtr-t5-base cosine, which systematically scores
    # English prose above verbatim code for conceptual queries (the
    # same-language bias), so code crystals get buried under prose /
    # ledger crystals. When ON, ContentRouter.search widens its
    # candidate pool to hybrid_rank_pool_size, buckets candidates into
    # code (sparse key starts "Code|") vs prose, z-normalizes cosine
    # WITHIN each bucket, and re-ranks by the calibrated score —
    # cancelling the cross-modal skew. The returned tuples keep their
    # ORIGINAL cosine (only the ordering changes), so downstream
    # thresholds and telemetry are unaffected. OFF by default — additive
    # and reversible; flip on after the retrieval eval shows a recall@k
    # / MRR gain on code-targeted queries. This is stage 1a; the
    # identifier/lexical-fusion half (1b) layers on later behind the
    # same flag.
    #
    #   CC_ENABLE_HYBRID_RANK / CC_HYBRID_RANK_POOL_SIZE
    enable_hybrid_rank: bool = False
    hybrid_rank_pool_size: int = 50

    # Code descriptions — index code by what it DOES, not its source
    # (June 2026, CRYS live test). Code chunks store the fact's search
    # vector as encode_native(verbatim source), so a conceptual query is
    # matched NL-against-raw-code and only hits when the query's words
    # collide with identifiers in the body — the symbol you can't name is
    # unfindable, and there is no natural-language text anywhere to match
    # against. When ON, ingest generates a functional natural-language
    # description per symbol (and a file-level summary) with one model
    # call per file, and that description — not the code — becomes the
    # fact's search vector (via add_pair_*'s embed_text seam), while
    # claim_text still returns the verbatim body. OFF by default: code
    # ingest stays LLM-free and byte-identical until flipped, so the
    # generation cost is a runtime choice, not a property of the build.
    #
    #   CC_ENABLE_CODE_DESCRIPTIONS
    enable_code_descriptions: bool = False

    # Agent loop ceiling (June 2026, CRYS live test).
    #
    # Max tool-call iterations per agent turn. The original 12 was hit
    # mid-debug-loop in the first real-world CRYS session (scrape → fix
    # → test cycles burn 2-3 iterations each); 24 gives honest work room
    # while still bounding runaways. Explicit Agent(max_iterations=...)
    # wins over this; this wins over the module default.
    #
    #   CC_AGENT_MAX_ITERATIONS
    agent_max_iterations: int = 24

    # Agent output budget (June 2026, CRYS live test).
    #
    # Max output tokens per model call. 4096 was too small for file
    # writes — a large write_file truncated mid-tool-call, the loop
    # dispatched the partial call, and the agent spiralled (2026-06-13
    # MMORPG session). 8192 is Sonnet 4.5's standard output ceiling;
    # raise it for deployments that routinely emit very large files.
    # Explicit Agent(max_tokens=...) wins over this; this wins over the
    # module default.
    #
    #   CC_AGENT_MAX_TOKENS
    agent_max_tokens: int = 8192

    # Agent retrieval pre-flight (C2 — cost + parity, 2026-06-16). When ON, a
    # fresh / no-context (opening) agent turn runs a retrieval pre-flight via
    # retrieve_and_inject BEFORE the tool loop: a cache hit (PERFECT routing +
    # source_kind=model_reasoning + answer_value) short-circuits the whole
    # loop with the cached answer; a miss injects the retrieved context into
    # the system prompt as warm-start so CRYS usually skips its first
    # knowledge_search. On follow-ups (context present) the pre-flight is
    # skipped and the model drives retrieval via tools. OFF by default — flip
    # after a live smoke check (same cautious rollout as cost accounting).
    #
    #   CC_AGENT_RETRIEVAL_PREFLIGHT
    # LAUNCH DEFAULT ON (flag-stance pass 2026-07-02): cache-hit
    # short-circuit + warm start; fail-safe by construction.
    agent_retrieval_preflight: bool = True

    # Agent-loop compaction (C3 — cost + parity, 2026-06-17). When ON, the
    # agent compacts its own growing tool-loop trajectory: once the working
    # message list crosses CC_COMPACT_THRESHOLD user turns it summarizes the
    # oldest turns (rule-based) into a block folded into the system prompt and
    # keeps the recent turns verbatim, so each model call stays bounded instead
    # of re-sending a monotonically growing context. The threshold self-gates
    # the cadence (a sawtooth — re-fires every few iterations in the long
    # regime). Reuses CC_COMPACT_THRESHOLD / CC_KEEP_RECENT_TURNS (compaction.py).
    # OFF by default — flip after a live smoke (same cautious rollout as the
    # pre-flight / cost accounting).
    #
    #   CC_AGENT_COMPACTION
    # LAUNCH DEFAULT ON (flag-stance pass 2026-07-02): long-session
    # robustness; returned trajectories stay complete either way.
    agent_compaction: bool = True

    # Agent tool-output cap (C4 — cost + parity, 2026-06-17). Max characters of
    # a single tool_result's content the model re-reads each iteration; the
    # output is truncated head+tail around a marker before it enters the
    # working trajectory (the full untrimmed output is still recorded in
    # tool_calls_log). Complements C3: compaction bounds the re-send of OLD
    # turns across iterations, this bounds a single OVERSIZED output even in the
    # recent window. 0 disables (the codebase's "no cap" idiom); ~12000 (≈3k
    # tokens) is a sane on-value. OFF by default — flip after a live smoke.
    #
    #   CC_AGENT_TOOL_OUTPUT_MAX_CHARS
    agent_tool_output_max_chars: int = 0

    # Agent controlling-model house default (C6 — model selection,
    # 2026-06-17). The model the CRYS agent loop uses when a request doesn't
    # name one AND the conversation has no saved model. "" = provider default
    # (agent.py: the built-in DEFAULT_MODEL under anthropic; the seam's large
    # tier under any other provider — the loop is provider-routed via
    # complete_messages since the provider-swap arc, 2026-07-02). This is the
    # process-wide FALLBACK at the bottom of the model precedence: client-sent
    # model (persisted per conversation) → the conversation's saved model →
    # THIS house default → provider default. Under CC_LLM_PROVIDER=openai this
    # holds a provider-native model name.
    #
    #   CC_AGENT_MODEL
    agent_model: str = ""

    # Agent citation grounding threshold (P3 / CC-D13 — agent citations,
    # 2026-06-17). The minimum answer-level cosine for a SURFACED crystal to
    # count as grounded in the agent path (ground_agent_citations ->
    # ground_sources_against_answer). DELIBERATELY separate from the proxy's
    # claim-span CITATION_GROUNDING_THRESHOLD (0.25): the agent grounds a whole
    # answer against a whole source, and gtr-t5-base's native-space cosine has
    # a high floor (~0.5 for unrelated text — anisotropy), so 0.25 grounds
    # everything. The smoke (scripts/smoke_p3_citations.py) measured a clean
    # gap — irrelevant <=~0.52, topical >=~0.76 — and 0.60 sits in it: it
    # rejects the floor (restoring BOTH G4 credit discrimination AND the G1c
    # uncited-answer gap) while still grounding genuine paraphrases. Thinly
    # calibrated (few scenarios) — re-run the smoke to recalibrate per
    # encoder / deployment.
    #
    # UPDATE (2026-06-18, showcase Act 1): 0.60 still grounded ~every crystal on
    # a single-DOMAIN bank — gtr-t5-base puts same-domain-different-topic cosines
    # at ~0.70 (the bonder's inter-FAQ floor; learning/bonder.py T_low), ABOVE
    # 0.60, so on a one-domain bank everything grounds. Raised to 0.75 = the
    # bonder's T_low, just under the ~0.76 topical floor the smoke measured: it
    # tracks within-domain discrimination (drops same-domain-different-topic,
    # keeps genuine supporters). Confirmed on Helios — 4/20 grounded, all genuine
    # supporters. Residual: a theme-overlapping passage can still edge over the
    # bar (whole-answer cosine ranks theme, not claim support); claim-level
    # grounding is the eventual fix (docs/SHOWCASE_NOTES.md N5).
    #
    #   CC_AGENT_CITATION_GROUNDING_THRESHOLD
    agent_citation_grounding_threshold: float = 0.75

    # API-key pepper (Foundation F1, 2026-06-13). Server-side secret
    # mixed into the HMAC-SHA256 hash of stored API keys (operators now;
    # customers as they migrate off plaintext). Defense-in-depth: a
    # stolen DB can't be brute-forced for keys without it. Empty = plain
    # SHA-256 fallback (dev convenience, still no-plaintext). SET THIS in
    # any real deployment.
    #
    #   CC_API_KEY_PEPPER
    api_key_pepper: str = ""

    # Platform-admin key (WS D / D.1, 2026-06). The single deployment-wide
    # superuser credential that gates the cross-customer operator surface
    # (/admin/api/*) and, in production, customer minting (POST
    # /v1/customers). Orthogonal to team/operator keys — it is NOT under any
    # team; it is the platform root. Compared constant-time against the
    # presented Bearer (it's a process-side env secret, so no DB pepper/hash
    # applies — that's for stored keys). Empty => the platform-admin gate is
    # OPEN (dev / self-host convenience); set it to lock the surface, and it
    # turns the gate ON even outside production. Production REFUSES to boot
    # when empty (see the lifespan boot guard in app.py). Per-admin key
    # issuance / rotation is a later (E.5) concern; this is the bootstrap root.
    #
    #   CC_ADMIN_API_KEY
    admin_api_key: str = ""

    # B2 hardening (2026-07-03): the platform-admin gate is fail-closed on
    # any non-loopback bind. Set CC_ADMIN_GATE_DISABLE=1 to force the gate
    # OFF even on a networked bind with no admin key — a conscious,
    # strongly-discouraged escape hatch for keyless networked dev. Never
    # set this in a real deployment; configure CC_ADMIN_API_KEY instead.
    admin_gate_disable: bool = False

    # Hosted identity (Accounts Phase A, 2026-07-06). Setting the Firebase /
    # Identity Platform project id ACTIVATES the JWT principal path — unset
    # (self-host default) means JWT bearers are simply never valid, so
    # self-hosters carry zero new burden (ratified D4: presence-as-switch,
    # no separate flag to drift).
    #
    #   CC_FIREBASE_PROJECT_ID
    firebase_project_id: str = ""

    # Comma-separated emails granted platform_admin at FIRST login (admin
    # bootstrap; ratified plan). Case-insensitive. Only consulted when the
    # JWT resolves to no existing user row.
    #
    #   CC_PLATFORM_ADMIN_EMAILS
    platform_admin_emails: str = ""

    # Managed inference (E4, Accounts Phase B 2026-07-06). The provider
    # whose PLATFORM credentials serve customers with inference_mode=
    # 'managed'. anthropic = settings.anthropic_api_key (the key the box
    # already holds for curation) — Anthropic-direct at launch (Vertex
    # quota pending). On self-host, 'managed' means "the box's key": the
    # operator IS the platform. Missing key at call time fails LOUD.
    #
    #   CC_MANAGED_INFERENCE_PROVIDER
    managed_inference_provider: str = "anthropic"

    # Google Drive OAuth (for Drive connector)
    google_client_id: str | None = None
    google_client_secret: str | None = None
    token_encryption_key: str | None = None

    # E3 key rotation (2026-07-03): comma-separated list of RETIRED 64-hex
    # token-encryption keys, kept for DECRYPTION only during a rotation
    # transition. To rotate: move the current CC_TOKEN_ENCRYPTION_KEY here,
    # set a fresh primary, boot, run the rotation walk to re-encrypt all
    # stored secrets under the new primary, then drop the old key from this
    # list. Empty in steady state.
    token_encryption_keys_retired: str = ""

    # Source connector — read-only code/source access for cognition
    # (see infrastructure/source_connector.py). OFF by default: when the
    # backend is unset, the source_lookup tool reports "unavailable" and
    # cognition cannot fabricate file/code claims.
    #
    #   CC_SOURCE_BACKEND        "" (off) | "local_fs" | "github"
    #   CC_SOURCE_FS_ROOT        absolute root dir for local_fs (required
    #                            for that backend; never defaults to / or cwd)
    #   CC_SOURCE_GITHUB_OWNER    repo owner (github backend)
    #   CC_SOURCE_GITHUB_REPO     repo name  (github backend)
    #   CC_SOURCE_GITHUB_REF      branch/tag/sha; empty = default branch
    #   CC_SOURCE_GITHUB_TOKEN    PAT for private repos / higher rate
    #                            limits (also honors bare GITHUB_TOKEN)
    source_backend: Literal["", "local_fs", "github"] = ""
    source_fs_root: str | None = None
    source_github_owner: str | None = None
    source_github_repo: str | None = None
    source_github_ref: str = ""
    source_github_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "CC_SOURCE_GITHUB_TOKEN", "GITHUB_TOKEN"
        ),
    )

    @property
    def is_test(self) -> bool:
        return self.environment == "test"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Lazy singleton. Override in tests via dependency injection or direct construction."""
    return Settings()


# Convenience: `from crystal_cache.config import settings`
settings = get_settings()
