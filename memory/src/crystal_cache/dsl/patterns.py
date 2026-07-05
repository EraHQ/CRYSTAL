"""HDC pattern library — stable runtime primitives.

VECTOR SPACE: CONCEPT SPACE (v0.2)
-----------------------------------
The DSL operates in its own vector space, separate from PromptEncoder's
text space. Concept vectors are dense ±1 hypervectors derived
deterministically from (tenant_id, concept_name) via SHA-256. This is
the only representation that makes HDC bind/unbind/bundle math work
correctly — sparse sum-of-hash vectors collapse to near-zero under
elementwise multiply.

WHY TWO SPACES
--------------
PromptEncoder (in crystal_cache/encoding/) produces sparse, unit-norm
float32 vectors from text. Those work great for cosine-similarity
retrieval of Fact vectors against text queries. They do NOT work for
HDC role-filler composition, because elementwise-multiplying two sparse
vectors mostly gives you zero.

The DSL needs a different vector space for its job (structured
composition, bind/unbind, reasoning traces). So we build one. The two
spaces don't bridge automatically. The intended bridge is an upstream
decomposer (small LLM in structured-output mode) that emits DSL
expressions from free text; the DSL compiles those expressions into
concept-space vectors, and *those* are what gets compared.

MULTI-TENANT ISOLATION
----------------------
Concept vectors derive from SHA-256(ENCODER_SEED + tenant_id + ":" +
concept_name + ":" + dim_index). Two tenants get orthogonal vectors for
the same concept name. For cross-tenant concepts use
tenant_id=SHARED_TENANT.

NAMES AND CASE
--------------
Concept names are lowercased on lookup, matching PromptEncoder's
tokenization convention. `vocab.FOCUS` and `vocab.focus` return the
same vector.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Optional

import numpy as np

from crystal_cache.config import settings


# Same seed byte-string used by PromptEncoder — keeps the project
# coherent; changing it invalidates every derived vector.
ENCODER_SEED = b"crystal_cache.v1"

# Sentinel tenant id for concepts visible across tenants.
SHARED_TENANT = "_shared_"


# ------------------------------------------------------------
# Primitives
# ------------------------------------------------------------


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise multiply. Self-inverse for ±1 bipolar vectors."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return (a * b).astype(np.int8, copy=False)


# Deterministic tie-break mask, generated once per dimension size.
# Must be UNCORRELATED with any concept vector to avoid systematic bias
# in the bundle output. A constant (+1 everywhere) correlates with every
# hypervector that has a +1 majority at those positions, which leaks into
# unbind results and makes field-names alias to values. A deterministic
# pseudo-random mask avoids this while preserving reproducibility.
_TIE_BREAK_CACHE: dict[int, np.ndarray] = {}


def _tie_break_mask(d: int) -> np.ndarray:
    if d not in _TIE_BREAK_CACHE:
        # Seeded with a different constant than concept vectors so the
        # mask is uncorrelated with any Vocabulary-derived hypervector.
        h = hashlib.sha256()
        h.update(b"crystal_cache.bundle_tie_break.v1")
        buf = bytearray()
        counter = 0
        bytes_needed = (d + 7) // 8
        while len(buf) < bytes_needed:
            hh = hashlib.sha256()
            hh.update(b"crystal_cache.bundle_tie_break.v1")
            hh.update(counter.to_bytes(4, "little"))
            buf.extend(hh.digest())
            counter += 1
        arr = np.frombuffer(bytes(buf[:bytes_needed]), dtype=np.uint8)
        bits = np.unpackbits(arr)[:d]
        _TIE_BREAK_CACHE[d] = np.where(bits == 1, 1, -1).astype(np.int8)
    return _TIE_BREAK_CACHE[d]


def bundle(vectors: Iterable[np.ndarray]) -> np.ndarray:
    """Majority-vote bundle. Ties broken via a fixed pseudo-random mask.

    The tie-break mask is deterministic (SHA-256-derived) so same inputs
    always produce same outputs, but uncorrelated with any concept
    vector — crucial for preventing field-names from aliasing to values
    after unbind on even-numbered bundles.
    """
    vecs = list(vectors)
    if not vecs:
        return np.zeros(settings.d_hdc, dtype=np.int8)
    stacked = np.stack([v.astype(np.int32, copy=False) for v in vecs])
    summed = stacked.sum(axis=0)
    result = np.sign(summed).astype(np.int8)
    ties = result == 0
    if ties.any():
        mask = _tie_break_mask(result.shape[0])
        result[ties] = mask[ties]
    return result


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized dot product for bipolar vectors. Returns value in [-1, 1]."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    d = a.shape[0]
    return float(np.dot(a.astype(np.int32), b.astype(np.int32))) / d


def shift(hv: np.ndarray, n: int = 1) -> np.ndarray:
    """Circular shift — position encoding for sequences."""
    return np.roll(hv, n)


# ------------------------------------------------------------
# Deterministic dense vector derivation
# ------------------------------------------------------------


def _derive_hv(tenant_id: str, concept_name: str, d: int) -> np.ndarray:
    """Derive a dense ±1 hypervector from (tenant, name).

    Uses SHA-256 as a PRNG. We draw d bits by hashing
      ENCODER_SEED || tenant || ":" || name || ":" || counter
    and unpacking each byte's bits into ±1 values.

    This is load-bearing:
      - Deterministic: same (tenant, name) always gives the same vector
      - Process-independent: no seeded RNG state to synchronize
      - Tenant-isolated: different tenants get uncorrelated vectors
      - Dense: every dimension carries signal, which is what makes
        HDC bind/unbind actually work
    """
    if not tenant_id:
        raise ValueError("tenant_id must be non-empty")
    if not concept_name:
        raise ValueError("concept_name must be non-empty")

    prefix = ENCODER_SEED + tenant_id.encode("utf-8") + b":" + concept_name.encode("utf-8") + b":"

    # Each SHA-256 call gives 32 bytes = 256 bits. We need d bits.
    bytes_needed = (d + 7) // 8
    buf = bytearray()
    counter = 0
    while len(buf) < bytes_needed:
        h = hashlib.sha256()
        h.update(prefix)
        h.update(counter.to_bytes(4, "little"))
        buf.extend(h.digest())
        counter += 1

    # Unpack bits: 1 -> +1, 0 -> -1
    arr = np.frombuffer(bytes(buf[:bytes_needed]), dtype=np.uint8)
    bits = np.unpackbits(arr)[:d]
    return np.where(bits == 1, 1, -1).astype(np.int8)


# ------------------------------------------------------------
# Vocabulary
# ------------------------------------------------------------


class Vocabulary:
    """Named-concept codebook producing dense bipolar hypervectors.

    Each concept name maps to a stable hypervector determined by
    (tenant_id, lowercased_name). No RNG state, no seed parameter —
    the same pair always produces the same vector, bit-for-bit,
    across processes and machines.

    Usage:
        vocab = Vocabulary(tenant_id="acme_corp")
        vocab.get("focus")       # dense ±1 vector
        vocab.FOCUS              # same vector (case-insensitive)
        vocab.known()            # list of concept names seen
        vocab.nearest(hv, 3)     # closest concepts to a given vector
    """

    def __init__(self, tenant_id: str = SHARED_TENANT, d: Optional[int] = None) -> None:
        if not tenant_id:
            raise ValueError("tenant_id must be non-empty; use SHARED_TENANT for cross-tenant")
        self.tenant_id = tenant_id
        self.d = d or settings.d_hdc
        # Cache computed vectors — SHA-256 is cheap but not free, and
        # concept names get looked up many times per query.
        self._concepts: dict[str, np.ndarray] = {}

    def _normalize(self, name: str) -> str:
        return name.lower()

    def get(self, name: str) -> np.ndarray:
        key = self._normalize(name)
        if key not in self._concepts:
            self._concepts[key] = _derive_hv(self.tenant_id, key, self.d)
        return self._concepts[key]

    def __getattr__(self, name: str) -> np.ndarray:
        if name.startswith("_") or name in {"tenant_id", "d"}:
            raise AttributeError(name)
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        return self._normalize(name) in self._concepts

    def known(self) -> list[str]:
        return sorted(self._concepts.keys())

    def nearest(self, hv: np.ndarray, top_k: int = 1) -> list[tuple[str, float]]:
        if not self._concepts:
            return []
        scores = [(name, similarity(hv, vec)) for name, vec in self._concepts.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def to_state(self) -> dict:
        """Serializable state. Vectors are re-derived deterministically on load."""
        return {
            "tenant_id": self.tenant_id,
            "d": self.d,
            "concept_names": sorted(self._concepts.keys()),
        }

    @classmethod
    def from_state(cls, state: dict) -> "Vocabulary":
        vocab = cls(tenant_id=state["tenant_id"], d=state["d"])
        for name in state["concept_names"]:
            vocab.get(name)
        return vocab


# ------------------------------------------------------------
# Cleanup memory
# ------------------------------------------------------------


class Cleanup:
    """Snap a noisy hypervector to its nearest known concept."""

    def __init__(self, vocab: Vocabulary, threshold: float = 0.15) -> None:
        self.vocab = vocab
        self.threshold = threshold

    def snap(self, hv: np.ndarray) -> tuple[Optional[str], np.ndarray, float]:
        if not self.vocab._concepts:
            return None, hv, 0.0
        name, sim = self.vocab.nearest(hv, top_k=1)[0]
        if sim < self.threshold:
            return None, hv, sim
        return name, self.vocab.get(name), sim

    def top_k(self, hv: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        return self.vocab.nearest(hv, top_k=k)


# ------------------------------------------------------------
# Step codebook for chains
# ------------------------------------------------------------


class StepCodebook:
    """Snap noisy step-records back to known completed steps in a chain."""

    def __init__(self, threshold: float = 0.15) -> None:
        self.threshold = threshold
        self._steps: dict[str, np.ndarray] = {}

    def register(self, name: str, step_hv: np.ndarray) -> None:
        self._steps[name] = step_hv

    def snap(self, hv: np.ndarray) -> tuple[Optional[str], np.ndarray, float]:
        if not self._steps:
            return None, hv, 0.0
        scored = [(name, similarity(hv, step)) for name, step in self._steps.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        name, sim = scored[0]
        if sim < self.threshold:
            return None, hv, sim
        return name, self._steps[name], sim

    def known(self) -> list[str]:
        return sorted(self._steps.keys())


# ------------------------------------------------------------
# Patterns
# ------------------------------------------------------------


def make_record(fields: dict[str, np.ndarray], vocab: Vocabulary) -> np.ndarray:
    if not fields:
        return np.zeros(vocab.d, dtype=np.int8)
    bindings = [bind(vocab.get(field_name), value) for field_name, value in fields.items()]
    return bundle(bindings)


def read_field(
    record: np.ndarray, field_name: str, vocab: Vocabulary, cleanup: Cleanup
) -> tuple[Optional[str], float]:
    recovered = bind(record, vocab.get(field_name))
    name, _, sim = cleanup.snap(recovered)
    return name, sim


def make_lookup(table: dict[str, np.ndarray], vocab: Vocabulary) -> np.ndarray:
    if not table:
        return np.zeros(vocab.d, dtype=np.int8)
    bindings = [bind(vocab.get(key_name), value) for key_name, value in table.items()]
    return bundle(bindings)


def lookup(
    table_hv: np.ndarray, key_name: str, vocab: Vocabulary, cleanup: Cleanup
) -> tuple[Optional[str], float]:
    recovered = bind(table_hv, vocab.get(key_name))
    name, _, sim = cleanup.snap(recovered)
    return name, sim


def branch(
    condition: np.ndarray,
    if_true: np.ndarray,
    if_false: np.ndarray,
    vocab: Vocabulary,
) -> np.ndarray:
    return bundle(
        [
            bind(vocab.get("true"), if_true),
            bind(vocab.get("false"), if_false),
            bind(vocab.get("condition"), condition),
        ]
    )


def resolve_branch(
    branch_hv: np.ndarray,
    condition_value: bool,
    vocab: Vocabulary,
    cleanup: Cleanup,
) -> tuple[Optional[str], float]:
    key = vocab.get("true") if condition_value else vocab.get("false")
    recovered = bind(branch_hv, key)
    name, _, sim = cleanup.snap(recovered)
    return name, sim


def make_sequence(items: list[np.ndarray]) -> np.ndarray:
    if not items:
        return np.zeros(settings.d_hdc, dtype=np.int8)
    return bundle([shift(item, i) for i, item in enumerate(items)])


def get_at_position(
    seq_hv: np.ndarray, position: int, cleanup: Cleanup
) -> tuple[Optional[str], float]:
    unshifted = shift(seq_hv, -position)
    name, _, sim = cleanup.snap(unshifted)
    return name, sim


def chain_step(
    prev_step: Optional[np.ndarray],
    step_fields: dict[str, np.ndarray],
    vocab: Vocabulary,
    codebook: Optional[StepCodebook] = None,
    step_name: Optional[str] = None,
) -> np.ndarray:
    fields = dict(step_fields)
    if prev_step is not None:
        fields["prev"] = prev_step
    step_hv = make_record(fields, vocab)
    if codebook is not None and step_name is not None:
        codebook.register(step_name, step_hv)
    return step_hv


# ------------------------------------------------------------
# Random hypervector (for tests/experiments, NOT for production concepts)
# ------------------------------------------------------------


def random_hv(
    d: Optional[int] = None, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Return a random bipolar hypervector.

    For stable cross-process vectors, use Vocabulary.get(name) instead.
    """
    d = d or settings.d_hdc
    rng = rng or np.random.default_rng()
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=d)
