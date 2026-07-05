"""Build a ready-to-use Crystal Cache agent for the terminal.

The web app builds the agent's dependencies in its startup lifespan. A
terminal app can't use that, so we build the same bundle here from the
crystal_cache library's PUBLIC building blocks. This file imports the
library and nothing from the web app (no app.py, no endpoints) — that's
what keeps the coding agent a clean, separate segment.

What gets built:
  - a metadata store backed by local SQLite, with tables created on
    first run
  - the text encoder and the two vector stores
  - the LLM client (the agent's controlling model)
  - an in-memory customer record (the Agent accepts a customer object
    directly, so no database row is needed to start)

Credentials (name, provider, model, API key) are resolved separately by
config_store and passed in — see config_store.py for the precedence and
the agent's own config file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from crystal_cache.agent import Agent
from crystal_cache.encoding import build_text_encoder
from crystal_cache.infrastructure import MetadataStore, VectorStore
from crystal_cache.infrastructure.fact_vector_store import FactVectorStore
from crystal_cache.infrastructure.vector_index import InMemoryVectorIndex
from crystal_cache.infrastructure.metadata_store import set_metadata_store
from crystal_cache.config import get_settings
from crystal_cache.models import CrystalType, Customer, ModelRoutingConfig

from .config_store import Credentials


# A stable local customer id. The agent accepts a Customer object
# directly, so no database row is required — but a fixed id keeps any
# crystals this agent writes grouped together across runs.
LOCAL_CUSTOMER_ID = "coding-agent-local"


def _build_local_customer(provider: str, model: str) -> Customer:
    """Construct an in-memory customer for the local coding agent.

    The agent's main model call uses the LLM client + model passed to
    Agent(...), not this routing config. But a Customer object is
    required, and its routing config is what the `llm_invoke` tool uses,
    so we mirror the resolved provider/model for consistency.
    """
    return Customer(
        id=LOCAL_CUSTOMER_ID,
        api_key="local-coding-agent",
        model_routing_config=ModelRoutingConfig(
            provider=provider,
            model_id=model,
            api_key_ref="config:crystal-code",
        ),
    )


def _resolve_db_url(db_arg: Optional[str]) -> Optional[str]:
    """Turn a --db argument into a SQLAlchemy async URL, or None.

    Accepts either a full URL (anything containing '://' — a Postgres
    connection string or an explicit sqlite+aiosqlite URL) or a plain
    file path, which is treated as a local SQLite database.
    """
    if not db_arg:
        return None
    if "://" in db_arg:
        return db_arg
    abs_path = Path(db_arg).expanduser().resolve()
    return f"sqlite+aiosqlite:///{abs_path}"


def _make_store(db_url: Optional[str]) -> MetadataStore:
    """Build the metadata store, optionally pointed at a specific DB.

    With no db_url, uses the default (a local SQLite file in the launch
    folder). With a db_url, overrides database_url via a settings copy
    so the agent reads a real, populated store.
    """
    if db_url is None:
        return MetadataStore()
    settings = get_settings().model_copy(update={"database_url": db_url})
    return MetadataStore(settings_override=settings)


def schema_mismatch_message(check: dict) -> str:
    """The fresh-start message for a failed schema-compatibility check.

    Local default stores have no migration story (backlog): the honest
    fix is a fresh start, stated plainly with the file and the columns
    named — instead of the cryptic `no such column` this used to
    surface as, mid-feature, with no explanation.
    """
    cols = ", ".join(check["mismatches"][:6])
    more = len(check["mismatches"]) - 6
    if more > 0:
        cols += f" (+{more} more)"
    return (
        f"this knowledge store predates a schema change — it is missing "
        f"columns the current code needs ({cols}).\n"
        f"  store: {check['database']}\n"
        "  Local stores don't migrate yet. The fix is a fresh start: "
        "delete that store file and re-run /ingest (re-syncs are cheap — "
        "unchanged files skip by content hash).\n"
        "  If this is a shared/dev database instead, run "
        "`alembic upgrade head` against it rather than deleting it."
    )


async def ensure_schema_compatible(store: MetadataStore) -> None:
    """Raise RuntimeError with the fresh-start message on mismatch.
    Called right after store.init() by build_agent; the CLI surfaces
    the message via its normal could-not-start path."""
    check = await store.check_schema_compatibility()
    if check["mismatches"]:
        raise RuntimeError(schema_mismatch_message(check))


async def _seed_legacy_crystal_types(store: MetadataStore) -> None:
    """Seed the two legacy crystal-type rows the library expects.

    Mirrors the web app's startup seed, using only public methods.
    Idempotent — checks for the row first and upserts if missing.
    """
    if await store.get_crystal_type("customer:legacy") is None:
        await store.upsert_crystal_type(CrystalType(
            id="customer:legacy",
            display_name="Customer (legacy catch-all)",
            scope="customer",
        ))
        await store.upsert_crystal_type(CrystalType(
            id="general:legacy",
            display_name="General (legacy catch-all)",
            scope="general",
        ))


DEFAULT_FAST_MODEL = "claude-haiku-4-5-20251001"


def build_llm_client(creds: Credentials, models: Optional[dict] = None):
    """Per-user provider-neutral LLM client from CRYS credentials.

    The recorded provider selects the transport (anthropic SDK or an
    OpenAI-compatible endpoint at creds.base_url); the key is the user's
    own. Tier mapping when a models dict is given: fast -> small,
    main -> large, so seam-tier call sites (the audit scans) resolve to
    the session's configured models. Explicit model= arguments at call
    sites always win over tiers.
    """
    from crystal_cache.llm import LLMClient

    m = models or {}
    return LLMClient(
        provider=(creds.provider or "anthropic"),
        api_key=creds.api_key,
        base_url=getattr(creds, "base_url", None),
        model_small=m.get("fast"),
        model_large=m.get("main") or creds.model,
        model_frontier=None,
    )


def resolve_models(creds: Credentials, project_models: dict) -> dict:
    """F6: resolve the model routing for a session.

    main — the agent's controlling model: `models.main` from
    .crystal-code.json when set, otherwise the credentials' model.
    fast — the cheap model for delegated work (F7 subagents):
    `models.fast` when set, otherwise a Haiku default. Token economics
    as configuration: the big model thinks, the cheap model fetches.
    """
    main = project_models.get("main")
    fast = project_models.get("fast")
    return {
        "main": main if isinstance(main, str) and main.strip() else creds.model,
        "main_source": "project config" if isinstance(main, str) and main.strip() else "credentials",
        "fast": fast if isinstance(fast, str) and fast.strip() else DEFAULT_FAST_MODEL,
        "fast_source": "project config" if isinstance(fast, str) and fast.strip() else "default",
    }


async def build_agent(
    creds: Credentials,
    *,
    db: Optional[str] = None,
    customer_id: Optional[str] = None,
    on_status: Optional[Callable[[str], None]] = None,
    intercept: Optional[Any] = None,
    after_tool: Optional[Any] = None,
    model: Optional[str] = None,
) -> Agent:
    """Build and return a ready Crystal Cache agent.

    Args:
        creds: resolved credentials (provider, model, API key) from
            config_store. The caller resolves these and runs first-time
            setup if needed, so by the time we're here a key exists.
        db: optional database to read/write — a file path (local SQLite)
            or a full URL (e.g. Postgres). None uses the default local
            store in the launch folder.
        customer_id: optional existing customer to load. When given, the
            agent works with that customer's FULL memory (retrieval is
            scoped by customer id). None uses an in-memory local customer.
        on_status: optional callback for progress messages (the encoder
            load is slow on first run, so the caller can tell the user
            what's happening).
    """
    def status(msg: str) -> None:
        if on_status is not None:
            on_status(msg)

    # --- LLM client (the agent's controlling model) ---
    # Built from the recorded provider through the provider-neutral seam:
    # anthropic or any OpenAI-compatible endpoint, on the user's own key.
    llm = build_llm_client(creds)

    # --- Metadata store + tables ---
    status("Setting up the database")
    store = _make_store(_resolve_db_url(db))
    await store.init()  # create tables if missing (no-op on a populated DB)
    await ensure_schema_compatible(store)  # refuse early, with the fix named
    set_metadata_store(store)  # some library paths read the process-wide store
    await _seed_legacy_crystal_types(store)

    # --- Encoder + vector stores ---
    status("Loading the language model (first run can take a minute)")
    encoder = build_text_encoder()
    vector_store = VectorStore(store=store)
    fact_vector_store = FactVectorStore(store=store)
    vector_index = InMemoryVectorIndex(
        fact_store=fact_vector_store,
        vector_store=vector_store,
        metadata_store=store,
    )

    # --- Customer ---
    # With a customer id, load that real customer so the agent works with
    # their FULL memory — retrieval is scoped by customer id, so this is
    # what makes the agent "know the company." Without one, fall back to
    # an in-memory local customer.
    customer: Optional[Customer] = None
    if customer_id:
        customer = await store.get_customer_by_id(customer_id)
        if customer is None:
            status(f"No customer {customer_id!r} in that store; using a local one")
        else:
            status(f"Loaded customer {customer_id!r}")
    if customer is None:
        customer = _build_local_customer(creds.provider, creds.model)

    status("Starting the agent")
    return Agent(
        customer=customer,
        llm=llm,
        intercept=intercept,
        after_tool=after_tool,
        tool_state={
            "store": store,
            "vector_store": vector_store,
            "fact_vector_store": fact_vector_store,
            "vector_index": vector_index,
            "encoder": encoder,
            # Step 1: decomposer not wired (needs a Groq key). Safe —
            # only the `decompose` tool uses it, and the agent recovers
            # gracefully if that one tool is unavailable.
            "decomposer": None,
        },
        model=model or creds.model,
    )


async def resolve_customer_by_key(db: Optional[str], api_key: str) -> dict:
    """Resolve a customer from a pasted API key (Key A) for /login.

    Read-oriented: builds a store against the given DB, looks the
    customer up by api_key, counts their crystals for the confirmation
    line, and disposes the store. Returns
    {customer_id, name, crystals, error} — customer_id is None when the
    key doesn't match (error explains what went wrong, e.g. a missing
    file or a v1-schema store).
    """
    out: dict = {"customer_id": None, "crystals": -1, "error": None}
    store = _make_store(_resolve_db_url(db))
    try:
        try:
            customer = await store.get_customer_by_api_key(api_key)
        except Exception as e:  # noqa: BLE001 — surface the real reason
            out["error"] = f"{type(e).__name__}: {e}"
            return out
        if customer is None:
            out["error"] = "no customer with that API key in that store"
            return out
        out["customer_id"] = customer.id
        try:
            out["crystals"] = await store.count_crystals_for_customer(customer.id)
        except Exception:
            pass  # cosmetic count only
        return out
    finally:
        await store.dispose()


def inspect_store_tables(db: Optional[str]) -> Optional[list[tuple[str, int]]]:
    """Read-only raw peek at a local SQLite store: [(table, row_count)] or None.

    Uses the stdlib sqlite3 driver in read-only mode, so it makes NO
    assumptions about the v2 ORM schema and never modifies the file. This
    is the diagnostic that reveals a v1-vs-v2 mismatch: it shows the
    tables that actually exist, even when the v2 ORM can't read them.
    Returns None if db isn't a local SQLite file we can open.
    """
    import os
    import sqlite3
    if not db or "://" in db:
        return None
    path = os.path.abspath(os.path.expanduser(db))
    if not os.path.exists(path):
        return None
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except Exception:
        return None
    try:
        cur = con.cursor()
        tables = [r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        out: list[tuple[str, int]] = []
        for t in tables:
            try:
                n = cur.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            except Exception:
                n = -1
            out.append((t, n))
        return out
    finally:
        con.close()


async def diagnose_store(db: Optional[str]) -> dict:
    """Inspect a store for the --list-customers helper.

    Returns {customers, v2_error, tables}. Tries the v2 ORM read of
    customers (capturing any error verbatim instead of swallowing it),
    then adds a raw read-only table listing so a v1-vs-v2 schema mismatch
    is visible rather than hidden behind an empty result. READ-ONLY: it
    never creates tables or modifies the database.
    """
    out: dict = {"customers": [], "v2_error": None, "tables": None}
    store = _make_store(_resolve_db_url(db))
    try:
        try:
            customers = await store.list_customers()
            for c in customers:
                try:
                    n = await store.count_crystals_for_customer(c.id)
                except Exception:
                    n = -1
                out["customers"].append((c.id, n))
        except Exception as e:  # noqa: BLE001 — surface the real reason
            out["v2_error"] = f"{type(e).__name__}: {e}"
    finally:
        await store.dispose()
    out["tables"] = inspect_store_tables(db)
    return out


async def run_audit(
    db: Optional[str],
    customer_id: Optional[str],
    *,
    max_pairs: int = 200,
    max_calls: int = 50,
) -> int:
    """`--audit`: run the contradiction scan once over a customer's bank and
    print the conflicts it surfaces, then exit (no REPL, no agent build).

    The on-demand convergence path (docs/NEVER_IDLE_CONVERGENCE.md): runs
    regardless of CC_ENABLE_CONVERGENCE_SCAN (that flag gates only the
    autonomous worker pass; --audit is an explicit operator action), bounded
    by max_calls / max_pairs. Surfacing-only — it writes knowledge_conflicts
    and nothing destructive. Scans the --customer bank, or the local agent's
    own (coding-agent-local) when none is given.
    """
    from crystal_cache.scan import scan_for_contradictions
    from .config_store import resolve_credentials

    creds = resolve_credentials()
    if creds is None:
        print(
            "\n  No credentials found. Launch `crys` once to set up, or set "
            "ANTHROPIC_API_KEY, then re-run --audit.\n"
        )
        return 1
    llm = build_llm_client(creds, resolve_models(creds, {}))

    store = _make_store(_resolve_db_url(db))
    try:
        await store.init()
        try:
            await ensure_schema_compatible(store)
        except RuntimeError as e:
            print(f"\n  {e}\n")
            return 1

        cid = customer_id or LOCAL_CUSTOMER_ID
        customer = await store.get_customer_by_id(cid)
        # A local default store may hold crystals under the local id with no
        # customer row; scan by id regardless of whether a row exists.
        target_id = customer.id if customer is not None else cid

        print(
            f"\n  Auditing bank for customer {target_id} "
            f"(up to {max_calls} checks)…\n"
        )
        result = await scan_for_contradictions(
            store=store,
            slm_client=llm,
            customer_id=target_id,
            max_candidate_pairs=max_pairs,
            max_discriminator_calls=max_calls,
        )
        print(f"  facts scanned        : {result.facts_scanned}")
        print(f"  candidate pairs      : {result.candidate_pairs}")
        print(f"  pairs evaluated      : {result.pairs_evaluated}")
        print(f"  already-known skips  : {result.skipped_existing}")
        print(f"  conflicts surfaced   : {result.conflicts_found}")
        if result.budget_exhausted:
            print("  (budget reached — re-run to evaluate more pairs)")
        if result.conflicts_found:
            conflicts = await store.list_knowledge_conflicts(
                target_id, status="open", limit=result.conflicts_found,
            )
            print("\n  Open conflicts (newest first):")
            for c in conflicts:
                subj = c.subject or "(no subject)"
                print(
                    f"   • [{subj}] {(c.claim_a or '')[:70]}  ⟷  "
                    f"{(c.claim_b or '')[:70]}"
                )
        print()
        return 0
    finally:
        await store.dispose()
