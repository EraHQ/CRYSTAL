"""conftest for crystal-cache v2 tests.

Phase 8 (2026-05-26): smoke-test infrastructure for the agent package
+ cognition §6.5.5 refactor. Per the locked Phase 8 decisions
(P0.29–P0.32):

- In-memory SQLite per test (P0.30). Each fixture constructs a fresh
  MetadataStore against `sqlite+aiosqlite:///:memory:` and seeds the
  two legacy crystal_type rows via the test-only helper.
- Hand-rolled Anthropic fake (P0.31). Lives in `tests/fakes.py` and is
  imported via the `fake_anthropic` fixture below.
- No real LLM calls. Tests that exercise the agent loop or cognition
  use the fake; tests that exercise tool dispatch directly mock at
  the tool boundary instead.

Path setup: tests/ is added to sys.path so `from fakes import ...`
works without making tests/ a proper package (which would otherwise
require an __init__.py and adjust test discovery semantics). src/ is
also added so `from crystal_cache.X import ...` works without an
editable install.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Launch-prep security pass (2026-07-02): Key B is encrypted UNCONDITIONALLY
# at the store writers, so every test that creates a customer needs an
# encryption key. Set BEFORE any crystal_cache import so the settings
# singleton reads it at construction.
os.environ.setdefault("CC_TOKEN_ENCRYPTION_KEY", "ab" * 32)

ROOT = Path(__file__).parent.parent
SRC_DIR = ROOT / "src"
TESTS_DIR = Path(__file__).parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

# Imports below this point need the sys.path insertions to have taken
# effect, so they live AFTER them.

from typing import Any, AsyncIterator

import pytest
import pytest_asyncio

from crystal_cache.config import Settings
from crystal_cache.encoding import build_text_encoder
from crystal_cache.infrastructure import MetadataStore, VectorStore
from crystal_cache.infrastructure.fact_vector_store import FactVectorStore
from crystal_cache.infrastructure.vector_index import InMemoryVectorIndex


# ---------------------------------------------------------------------------
# Async store fixture
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def store() -> AsyncIterator[MetadataStore]:
    """Yield a fresh in-memory MetadataStore per test.

    The store is initialized (tables created) and seeded with the two
    legacy crystal_type rows that production migration 0012 inserts.
    On test teardown the engine is disposed so SQLAlchemy's connection
    pool releases the in-memory DB.

    Each test gets its own database — no shared state between tests.
    """
    settings = Settings(database_url="sqlite+aiosqlite:///:memory:")
    s = MetadataStore(settings_override=settings)
    await s.init()
    await s._seed_legacy_crystal_types_for_tests()
    try:
        yield s
    finally:
        await s.dispose()


# ---------------------------------------------------------------------------
# Encoder + vector stores
# ---------------------------------------------------------------------------

@pytest.fixture
def encoder() -> Any:
    """Build the text encoder per CC_TEXT_ENCODER (hash by default).

    Phase 8 uses the hash encoder for speed — semantic would pull in
    sentence-transformers as a hard dep. The agent's content_search /
    knowledge_search / depth_search tools fail-fast on the hash
    encoder via the `BindCapableEncoder` protocol's runtime
    `hasattr(encoder, "encode_native")` check (see add_pair_to_crystal
    contract). Phase 8 smoke tests do not exercise the write path
    against the hash encoder; tests that need a write use a tiny
    semantic stub instead. See `semantic_encoder_stub` below.
    """
    return build_text_encoder()


@pytest.fixture
def semantic_encoder_stub() -> Any:
    """Minimal BindCapableEncoder stub.

    Returns an object with `.encode(text)` producing a 10k-dim
    float32 vector and `.encode_native(text)` producing a 768-dim
    float32 vector, plus `.fingerprint()` returning a stable string.
    Deterministic from text bytes so repeated calls yield identical
    outputs.

    Why not use the real SemanticTextEncoder: it loads gtr-t5-base
    (~440 MB), which is a Phase 8 dependency we'd rather not require.
    The stub satisfies the BindCapableEncoder protocol that
    add_pair_to_crystal's hasattr check requires.
    """
    import hashlib
    import numpy as np

    class _SemanticStub:
        def encode(self, text: str) -> np.ndarray:
            seed = int.from_bytes(
                hashlib.sha256(text.encode("utf-8")).digest()[:8], "big",
            )
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(10_000, dtype=np.float32)
            v /= np.linalg.norm(v) + 1e-12
            return v

        def encode_native(self, text: str) -> np.ndarray:
            seed = int.from_bytes(
                hashlib.sha256(text.encode("utf-8")).digest()[8:16], "big",
            )
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(768, dtype=np.float32)
            v /= np.linalg.norm(v) + 1e-12
            return v

        def fingerprint(self) -> str:
            return "semantic-stub-v1"

    return _SemanticStub()


@pytest.fixture
def vector_store(store: MetadataStore) -> VectorStore:
    return VectorStore(store=store)


@pytest.fixture
def fact_vector_store(store: MetadataStore) -> FactVectorStore:
    return FactVectorStore(store=store)


@pytest.fixture
def vector_index(
    store: MetadataStore,
    vector_store: VectorStore,
    fact_vector_store: FactVectorStore,
) -> InMemoryVectorIndex:
    return InMemoryVectorIndex(
        fact_store=fact_vector_store,
        vector_store=vector_store,
        metadata_store=store,
    )


# ---------------------------------------------------------------------------
# Anthropic fake
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_anthropic():
    """Hand-rolled Anthropic client fake (P0.31).

    Implemented in tests/fakes.py to keep the fixture module focused
    on wiring. Returns a fresh FakeAnthropic instance per test so
    scripted responses don't bleed between tests.

    The TESTS_DIR sys.path insertion above lets us import directly
    from `fakes` (no `tests.` prefix needed). This avoids requiring
    tests/__init__.py which would change pytest's test-discovery
    semantics.
    """
    from fakes import FakeAnthropic
    return FakeAnthropic()


# ---------------------------------------------------------------------------
# Customer fixture
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def customer(store: MetadataStore):
    """Create a test customer and return the Pydantic Customer record.

    Uses the production `create_customer` path so the row has a real
    auto-generated id, api_key, and model_routing_config. Tests that
    need a customer fixture can depend on this; tests that test
    customer creation itself should NOT depend on this.
    """
    return await store.create_customer(
        provider="anthropic",
        model_id="claude-sonnet-4-5-20250929",
        api_key_ref="sk-test-upstream-key",
    )


# ---------------------------------------------------------------------------
# Tool state injection helper
# ---------------------------------------------------------------------------

@pytest.fixture
def tool_state(
    store: MetadataStore,
    vector_store: VectorStore,
    fact_vector_store: FactVectorStore,
    vector_index: InMemoryVectorIndex,
    semantic_encoder_stub: Any,
) -> dict[str, Any]:
    """Build the tool-state dict the Agent injects.

    Tests that bypass the Agent class (e.g. invoking a tool directly
    against the registry) need to call `set_tool_state(tool_state)`
    before dispatch. Tests that go through Agent.run get the
    injection for free.
    """
    return {
        "store": store,
        "vector_store": vector_store,
        "fact_vector_store": fact_vector_store,
        "vector_index": vector_index,
        "encoder": semantic_encoder_stub,
        "decomposer": None,
    }


# ---------------------------------------------------------------------------
# NOTE on registry isolation
# ---------------------------------------------------------------------------
#
# Phase 8 fix (2026-05-26): the original conftest had an autouse
# fixture that called `reset_registry()` after every test. That
# fixture was harmful — `reset_registry()` clears the singleton but
# Python's import cache keeps the tool modules loaded, so the next
# `import_all_tools()` call was a no-op against an empty registry.
# Tests that asserted "registry populates" silently saw an empty
# registry and either failed the assertion or, worse, fell back to
# v1 helpers without the test noticing (AN-13).
#
# The fix landed in `agent/tool_registry.py::reset_registry`: it now
# also drops the tool modules from sys.modules so the next
# `import_all_tools()` actually fires the @register_tool decorators.
#
# With that fix in place, the autouse fixture became redundant: the
# registry is a process-wide singleton; once populated, it stays
# correct across tests. The only test that wants a CLEAR registry
# (Test C2: `test_cognition_falls_back_to_v1_helper_when_registry_empty`)
# calls `reset_registry()` explicitly inline. After that test runs
# `reset_registry()`, the next test that needs the registry calls
# `import_all_tools()` (directly or via Agent.__init__) and gets the
# decorators to fire again thanks to the sys.modules pop.
