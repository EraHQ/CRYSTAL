#!/usr/bin/env python3
"""Routing-lane quantization gate — Step 0 of docs/VECTOR_STORE_RESEARCH.md §6.

THE QUESTION THIS ANSWERS
-------------------------
The read-performance plan (Qdrant, both lanes) puts the 10k `routing_vector`
lane behind binary quantization + float rescore: a 1-bit coarse search, then
rescore the top candidates on the full float vector. That is fast and
memory-light AND — for over-parameterized embeddings — usually preserves
recall. But CRYS's routing vector is an additive `Σ encode(prompt_i)`
accumulator, NOT a generic sentence embedding, so "binary holds recall" is a
HYPOTHESIS until measured. This harness measures it.

WHAT IT MEASURES (the gate)
---------------------------
Per query, two routings are compared:
  • EXACT FLOAT  — `normalize(routing_vector) · query`, argmax. EXACTLY what
    VectorStore.search does in production (rows L2-normalized at load, scored
    by dot = cosine). The ground truth.
  • BINARY+RESCORE — sign-quantize the routing rows and the query to 1 bit,
    Hamming-coarse to a candidate pool (k·oversample), float-rescore the pool,
    take top-k. The proposed Qdrant fast path, in NumPy.
Reported: top-1 routing AGREEMENT (binary picks the same crystal as exact —
THE gate metric), recall@{1,5,10} of exact's top-1 inside binary's top-k, and
coarse-stage retention (did Hamming keep exact's winner before rescore).

TWO MODES
---------
  • REAL BANK (default): loads a customer's crystals + facts from a DB and
    builds REAL queries via `normalize(FactRow.vector @ P)` = `encode(prompt)`.
    Faithful, but only as large as the bank — at a few dozen crystals the
    coarse filter barely engages (pool ≈ n), so it confirms correctness, not
    scale.
  • --synthetic-scale N: generates N realistic CRYS-shaped routing vectors
    (sums of P-projected unit-native prompts, clustered into topics with
    near-siblings) and runs a SEPARABILITY SWEEP — it varies how tightly
    siblings cluster (the centroid `gap`) so exact-float home@1 ranges from
    easy (~0.95) down through the hard regime (~0.1, near-ties among siblings),
    reporting the gate at pool 20 and pool 200 (pool << n) for each level. The
    row near home@1≈0.57 matches the real bank; rows below it stress the
    small-margin regime. --anisotropy adds an embedding-cone bias (the one
    real-world stressor isotropic Gaussians miss). THIS is the scale test.

WHAT IT DOES *NOT* MEASURE
--------------------------
Latency. Timings are NumPy wall-clock, INDICATIVE ONLY — not Qdrant/HNSW p95.
This isolates the *algorithmic recall ceiling*: if binary+rescore can't match
exact float HERE, no engine will; if it can, the real-latency Qdrant
benchmark is the follow-on.

NO MODEL, NO QDRANT
-------------------
The semantic encoder is `encode(text) = normalize( normalize(native) @ P )`,
P a fixed (native, d_hdc) bipolar ±1 matrix from a fixed seed
(encoding/semantic.py PROJECTION_SEED). `FactRow.vector` is the unit-norm
native vector, so `normalize(fact.vector @ P)` reproduces `encode(prompt)`
exactly. P is rebuilt from the same seed here; only numpy is needed.

USAGE (repo root; .venv-clean is enough — no gtr-t5-base load):
    # real bank
    python scripts/bench_routing_quantization.py --db crystal_cache_bench.db \\
        --customer cus_1dcea31a7ef04205
    # scale stress test (no DB needed) — separability sweep
    python scripts/bench_routing_quantization.py --synthetic-scale 10000
    python scripts/bench_routing_quantization.py --synthetic-scale 20000 \\
        --cluster-size 8 --anisotropy 1.0
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import numpy as np

from crystal_cache.config import get_settings
from crystal_cache.encoding.semantic import DEFAULT_MODEL_NAME, PROJECTION_SEED
from crystal_cache.infrastructure import MetadataStore
from crystal_cache.infrastructure.metadata_store import set_metadata_store

K_VALUES = (1, 5, 10)
K_MAX = max(K_VALUES)
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)

# Default native dim for the synthetic generator (gtr-t5-base = 768).
SYNTH_NATIVE_DIM = 768


# --- stack construction: mirrors scripts/eval_hybrid_rank.py ---------------

def _resolve_db_url(db_arg: str) -> str:
    if "://" in db_arg:
        return db_arg
    return f"sqlite+aiosqlite:///{Path(db_arg).expanduser().resolve()}"


def _make_store(db_url: str) -> MetadataStore:
    settings = get_settings().model_copy(update={"database_url": db_url})
    return MetadataStore(settings_override=settings)


# --- geometry --------------------------------------------------------------

def _l2(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (mat / norms).astype(np.float32)


def _build_projection(native_dim: int, d_hdc: int) -> np.ndarray:
    """Rebuild encoding/semantic.SemanticTextEncoder's fixed P, model-free."""
    rng = np.random.RandomState(PROJECTION_SEED)
    return rng.choice([-1.0, 1.0], size=(native_dim, d_hdc)).astype(np.float32)


def _pack(rows: np.ndarray) -> np.ndarray:
    """Sign-quantize to 1 bit (>0 -> 1) and pack to bytes. (n, d) -> (n, d/8)."""
    return np.packbits(rows > 0.0, axis=1)


def _hamming(bits_matrix: np.ndarray, q_bits: np.ndarray) -> np.ndarray:
    return _POPCOUNT[np.bitwise_xor(bits_matrix, q_bits)].sum(axis=1)


# --- the comparison (shared by both modes) ---------------------------------

def _evaluate(
    r_float: np.ndarray,
    r_bits: np.ndarray,
    q_float: np.ndarray,
    q_bits: np.ndarray,
    home_idx: np.ndarray | None,
    oversamples: list[int],
    collect_examples: int = 0,
) -> dict:
    """Exact-float vs binary-quant+rescore over a query batch.

    Hamming is computed ONCE per query (the ranking is oversample-independent;
    only the pool prefix changes), so all oversamples share one sort.
    """
    n = r_float.shape[0]
    q = q_float.shape[0]
    pools = {m: min(K_MAX * m, n) for m in oversamples}
    max_pool = max(pools.values())

    # Exact float (ground truth), batched over query-chunks to bound memory.
    exact_top1 = np.empty(q, dtype=np.int64)
    CH = 512
    t0 = time.perf_counter()
    for s in range(0, q, CH):
        e = r_float @ q_float[s:s + CH].T          # (n, chunk)
        exact_top1[s:s + CH] = np.argmax(e, axis=0)
    exact_ms = (time.perf_counter() - t0) / max(q, 1) * 1000.0

    agree = {m: 0 for m in oversamples}
    kept = {m: 0 for m in oversamples}
    recall = {m: {k: 0 for k in K_VALUES} for m in oversamples}
    examples: list[tuple[int, int, int]] = []
    smallest = min(oversamples)

    t0 = time.perf_counter()
    for qi in range(q):
        gt = int(exact_top1[qi])
        qf = q_float[qi]
        ham = _hamming(r_bits, q_bits[qi])
        if max_pool < n:
            part = np.argpartition(ham, max_pool - 1)[:max_pool]
            part = part[np.argsort(ham[part])]      # ascending Hamming
        else:
            part = np.argsort(ham)
        for m in oversamples:
            cand = part[:pools[m]]
            if (cand == gt).any():
                kept[m] += 1
            cand_scores = r_float[cand] @ qf
            order = np.argsort(-cand_scores)[:K_MAX]
            topk = cand[order]
            if topk.size and int(topk[0]) == gt:
                agree[m] += 1
            elif (m == smallest and collect_examples
                  and len(examples) < collect_examples):
                examples.append((qi, gt, int(topk[0]) if topk.size else -1))
            for k in K_VALUES:
                if (topk[:k] == gt).any():
                    recall[m][k] += 1
    bin_ms = (time.perf_counter() - t0) / max(q * len(oversamples), 1) * 1000.0

    home1 = (float(np.mean(exact_top1 == home_idx))
             if home_idx is not None else float("nan"))
    results = [{
        "oversample": m,
        "pool": pools[m],
        "agree": agree[m] / q,
        "recall": {k: recall[m][k] / q for k in K_VALUES},
        "coarse_retained": kept[m] / q,
    } for m in oversamples]
    return {
        "results": results, "home1": home1, "exact_top1": exact_top1,
        "exact_ms": exact_ms, "bin_ms": bin_ms, "examples": examples,
    }


# --- synthetic-bank generation ---------------------------------------------

def _synth_centroids(
    anchors: np.ndarray, assign: np.ndarray, gap: float,
    native_dim: int, rng: np.random.RandomState,
) -> np.ndarray:
    """Crystal centroids = normalize(cluster_anchor + gap·unit_noise).

    Small gap -> siblings sit tight around a shared topic anchor (near-ties,
    low home@1); large gap -> crystals drift apart (separable, high home@1).
    Noise is scaled by 1/sqrt(native_dim) so `gap` is the perturbation norm
    relative to the unit anchor, independent of dimension.
    """
    n = assign.shape[0]
    c = anchors[assign] + (gap / np.sqrt(native_dim)) * rng.randn(
        n, native_dim).astype(np.float32)
    return _l2(c)


def _build_routing(
    centroids: np.ndarray, ppc: int, prompt_spread: float,
    p_matrix: np.ndarray, rng: np.random.RandomState,
) -> np.ndarray:
    """N realistic routing vectors: each = normalize(Σ over ppc projected,
    unit-normed native prompts drawn near the crystal's centroid).

    Production pipeline (encode(prompt)=normalize(unit_native @ P),
    routing += encode(prompt)). prompt_spread (scaled by 1/sqrt(native)) is the
    intra-crystal prompt scatter.
    """
    n, native = centroids.shape
    d = p_matrix.shape[1]
    sd = prompt_spread / np.sqrt(native)
    out = np.empty((n, d), dtype=np.float32)
    chunk = 1000
    for s in range(0, n, chunk):
        c = centroids[s:s + chunk]                  # (b, native)
        b = c.shape[0]
        prompts = c[:, None, :] + sd * rng.randn(b, ppc, native).astype(np.float32)
        prompts /= (np.linalg.norm(prompts, axis=2, keepdims=True) + 1e-12)
        proj = prompts.reshape(b * ppc, native) @ p_matrix
        proj /= (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)
        summed = proj.reshape(b, ppc, d).sum(axis=1)
        out[s:s + chunk] = _l2(summed)
    return out


def _build_queries(
    centroids: np.ndarray, picks: np.ndarray, prompt_spread: float,
    p_matrix: np.ndarray, rng: np.random.RandomState,
) -> np.ndarray:
    """One fresh prompt near each picked crystal's centroid -> encode(prompt).

    A NEW near-topic prompt (not a training prompt) — the realistic query:
    does a query semantically near a crystal route to it under binary?
    """
    c = centroids[picks]
    sd = prompt_spread / np.sqrt(c.shape[1])
    p = c + sd * rng.randn(len(picks), c.shape[1]).astype(np.float32)
    p /= (np.linalg.norm(p, axis=1, keepdims=True) + 1e-12)
    proj = p @ p_matrix
    proj /= (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)
    return proj.astype(np.float32)


# --- reporting (real mode) -------------------------------------------------

def _print_table(ev: dict, n: int) -> None:
    print("\n  oversample  pool   top1-agree   recall@1  recall@5  recall@10   "
          "coarse-kept   ~bin ms/q")
    for r in ev["results"]:
        rc = r["recall"]
        print(f"  {r['oversample']:>9}  {r['pool']:>5}   "
              f"{r['agree']:>10.3f}   {rc[1]:>8.3f}  {rc[5]:>8.3f}  "
              f"{rc[10]:>9.3f}   {r['coarse_retained']:>11.3f}   {ev['bin_ms']:>8.3f}")
    print(f"\n  exact-float scan: ~{ev['exact_ms']:.3f} ms/query (NumPy, n={n})")
    print("  NOTE: timings are NumPy wall-clock, INDICATIVE ONLY — not "
          "Qdrant/HNSW p95.")
    print("  This gate measures the RECALL ceiling. top1-agree is the decision "
          "metric:")
    print("    high agreement at small pool (pool << n)  -> binary+rescore "
          "fast path is safe; proceed to the Qdrant latency benchmark.")
    print("    agreement drops as pool shrinks           -> binary pruning "
          "loses winners; fall back to exact-float HNSW (still Qdrant).")


async def _run_real(ns) -> int:
    store = _make_store(_resolve_db_url(ns.db))
    await store.init()
    set_metadata_store(store)
    try:
        crystals = await store.list_crystals_for_customer(ns.customer)
        facts = await store.list_all_facts_for_customer(ns.customer)
    finally:
        await store.dispose()

    usable = [c for c in crystals if c.routing_vector]
    if not usable:
        print(f"No crystals with routing_vector for customer {ns.customer!r}. "
              f"(Bank empty, wrong customer, or routing_vectors not backfilled.)")
        return 1
    d_hdc = len(usable[0].routing_vector)
    ids: list[str] = []
    rows: list[np.ndarray] = []
    for c in usable:
        if len(c.routing_vector) != d_hdc:
            continue
        ids.append(c.id)
        rows.append(np.asarray(c.routing_vector, dtype=np.float32))
    r_float = _l2(np.vstack(rows))
    r_bits = _pack(r_float)
    n = r_float.shape[0]
    idx_of = {cid: i for i, cid in enumerate(ids)}

    native_facts = [
        f for f in facts
        if f.vector and f.crystal_id in idx_of and len(f.vector) < d_hdc
    ]
    if native_facts:
        native_dim = len(native_facts[0].vector)
        query_src = [(f.crystal_id, np.asarray(f.vector, dtype=np.float32))
                     for f in native_facts if len(f.vector) == native_dim]
        query_kind = "fact prompt (FactRow.vector @ P)"
    else:
        ans = [(c.id, np.asarray(c.answer_embedding_native, dtype=np.float32))
               for c in usable
               if c.answer_embedding_native and len(c.answer_embedding_native) < d_hdc]
        if not ans:
            print("No native-dim vectors to build queries from. This bank looks "
                  "hash-encoded; the gate needs a semantic bank.")
            return 1
        native_dim = len(ans[0][1])
        query_src = [(cid, v) for cid, v in ans if len(v) == native_dim]
        query_kind = "answer_embedding_native @ P"

    p_matrix = _build_projection(native_dim, d_hdc)
    rng = np.random.RandomState(ns.seed)
    if len(query_src) > ns.max_queries:
        pick = rng.choice(len(query_src), size=ns.max_queries, replace=False)
        query_src = [query_src[i] for i in pick]

    home_idx: list[int] = []
    q_list: list[np.ndarray] = []
    for cid, v in query_src:
        nv = v / (np.linalg.norm(v) or 1.0)
        proj = nv @ p_matrix
        nrm = float(np.linalg.norm(proj))
        if nrm == 0.0:
            continue
        q_list.append((proj / nrm).astype(np.float32))
        home_idx.append(idx_of[cid])
    if not q_list:
        print("All sampled queries projected to zero — nothing to measure.")
        return 1
    q_float = np.vstack(q_list)
    q_bits = _pack(q_float)
    home = np.asarray(home_idx, dtype=np.int64)
    oversamples = [int(x) for x in ns.oversample.split(",") if x.strip()]
    ev = _evaluate(r_float, r_bits, q_float, q_bits, home, oversamples,
                   collect_examples=(8 if ns.verbose else 0))

    settings_d = int(getattr(get_settings(), "d_hdc", d_hdc))
    print(f"\nRouting quantization gate (REAL bank)  —  customer={ns.customer}")
    print(f"  bank: {ns.db}")
    print(f"  crystals routed: {n}   queries: {q_float.shape[0]}   "
          f"native_dim: {native_dim}   d_hdc: {d_hdc}"
          + ("" if d_hdc == settings_d else f"  (settings.d_hdc={settings_d}!)"))
    print(f"  query source: {query_kind}")
    print(f"  projection: seed={PROJECTION_SEED}, model={DEFAULT_MODEL_NAME} "
          f"(P rebuilt model-free)")
    print(f"  exact-float home@1 (bank routing quality): {ev['home1']:.3f}")
    if max(o for o in oversamples) * K_MAX >= n:
        print(f"  WARNING: max pool ({max(oversamples) * K_MAX}) >= n ({n}) — the "
              f"coarse filter barely engages; this confirms correctness, NOT "
              f"scale. Use --synthetic-scale for the scale test.")
    if not ns.no_self_floor:
        self_n = min(n, 1000)
        sel = (rng.choice(n, size=self_n, replace=False) if n > self_n
               else np.arange(n))
        pool0 = min(K_MAX * oversamples[0], n)
        ok = 0
        for i in sel:
            ham = _hamming(r_bits, r_bits[i])
            cand = (np.argpartition(ham, pool0 - 1)[:pool0] if pool0 < n
                    else np.arange(n))
            ok += int(cand[int(np.argmax(r_float[cand] @ r_float[i]))]) == int(i)
        print(f"  self-routing floor (query = own routing_vector, oversample "
              f"{oversamples[0]}): {ok}/{len(sel)} ({ok / len(sel):.3f})")
    _print_table(ev, n)
    if ev["examples"]:
        print("\n  example disagreements (home -> exact_top1 vs binary_top1):")
        for qi, gt, bi in ev["examples"]:
            print(f"    {ids[home[qi]][:18]:>18}  ->  exact={ids[gt][:18]:<18}  "
                  f"binary={ids[bi][:18] if bi >= 0 else '—'}")
    print()
    return 0


# --- synthetic separability sweep ------------------------------------------

def _run_synthetic(ns) -> int:
    n = ns.synthetic_scale
    d_hdc = int(getattr(get_settings(), "d_hdc", 10000))
    native_dim = SYNTH_NATIVE_DIM
    ppc = ns.prompts_per_crystal
    cs = max(2, ns.cluster_size)
    ps = ns.prompt_spread
    q = min(ns.max_queries, n)
    rng = np.random.RandomState(ns.seed)

    if ns.gap_grid:
        gaps = [float(x) for x in ns.gap_grid.split(",") if x.strip()]
    else:
        gaps = list(np.geomspace(0.005, 0.25, 8))

    est_mb = n * d_hdc * 4 / 1e6
    print(f"\nRouting quantization gate (SYNTHETIC separability sweep)")
    print(f"  crystals: {n}   cluster_size: {cs}   prompts/crystal: {ppc}   "
          f"prompt_spread: {ps}   anisotropy: {ns.anisotropy}")
    print(f"  native_dim: {native_dim}   d_hdc: {d_hdc}   queries: {q}")
    print(f"  routing matrix ~{est_mb:.0f} MB resident, rebuilt per gap "
          f"({len(gaps)} gaps) — expect a few minutes. Lower --synthetic-scale "
          f"if RAM-tight.")
    print(f"  projection: seed={PROJECTION_SEED}, model={DEFAULT_MODEL_NAME} "
          f"(P rebuilt model-free)")

    p_matrix = _build_projection(native_dim, d_hdc)
    nclus = max(2, int(np.ceil(n / cs)))
    # Anisotropy: bias all topic anchors toward one shared cone direction, so
    # the projected routing/query vectors share a component -> sign bits skew.
    anchors_raw = rng.randn(nclus, native_dim).astype(np.float32)
    if ns.anisotropy > 0.0:
        u = _l2(rng.randn(1, native_dim).astype(np.float32))
        anchors_raw = anchors_raw + float(ns.anisotropy) * u
    anchors = _l2(anchors_raw)
    assign = rng.randint(0, nclus, size=n)
    picks = rng.choice(n, size=q, replace=False)

    over = [2, 20]  # pools 20 and 200 (K_MAX=10)
    print(f"\n     gap   home@1   agree@{K_MAX * over[0]}  kept@{K_MAX * over[0]}"
          f"   agree@{K_MAX * over[1]}  kept@{K_MAX * over[1]}   ~bin ms/q")
    for gap in gaps:
        centroids = _synth_centroids(anchors, assign, float(gap), native_dim, rng)
        r_float = _build_routing(centroids, ppc, ps, p_matrix, rng)
        r_bits = _pack(r_float)
        q_float = _build_queries(centroids, picks, ps, p_matrix, rng)
        q_bits = _pack(q_float)
        ev = _evaluate(r_float, r_bits, q_float, q_bits, picks, over)
        r_small, r_big = ev["results"][0], ev["results"][1]
        print(f"  {gap:>7.4f}   {ev['home1']:>6.3f}    {r_small['agree']:>6.3f}   "
              f"{r_small['coarse_retained']:>6.3f}     {r_big['agree']:>6.3f}   "
              f"{r_big['coarse_retained']:>6.3f}    {ev['bin_ms']:>7.3f}",
              flush=True)
        del r_float, r_bits, q_float, q_bits, centroids  # bound peak RAM

    print(f"\n  exact-float home@1 spans the sweep (low gap = near-ties among "
          f"~{cs} siblings; high gap = separable topics).")
    print("  NOTE: timings are NumPy wall-clock, INDICATIVE ONLY — not "
          "Qdrant/HNSW p95. This is a RECALL sweep.")
    print("  How to read it:")
    print("    • find the row nearest home@1≈0.57 — that matches the real bank; "
          "rows below it are HARDER than production.")
    print("    • top1-agree ~1.0 across the sweep (incl. the hard, low-home@1 "
          "rows) -> binary+rescore is cleared at scale; go to the Qdrant "
          "latency benchmark.")
    print("    • agree dips but kept@200 > kept@20 -> a larger rescore pool "
          "recovers it; size Qdrant's oversampling accordingly.")
    print("    • agree dips AND kept@200 is low -> binary loses winners; use "
          "exact-float HNSW on the 10k lane (still Qdrant).")
    print()
    return 0


async def main() -> int:
    ap = argparse.ArgumentParser(
        description="Routing-lane binary-quant + rescore vs exact-float gate.")
    ap.add_argument("--db", help="SQLite path or URL of the bank (real mode).")
    ap.add_argument("--customer", default="CRYS-local",
                    help="Customer id whose bank to route over (real mode).")
    ap.add_argument("--max-queries", type=int, default=2000,
                    help="Cap on queries sampled (real) / generated (synthetic).")
    ap.add_argument("--oversample", default="2,4,8,16",
                    help="Comma list of pool multipliers (pool = 10 * m). Real "
                         "mode only; the synthetic sweep uses pools 20 and 200.")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for sampling/generation (not the projection).")
    ap.add_argument("--no-self-floor", action="store_true",
                    help="Skip the self-routing sanity floor (real mode).")
    ap.add_argument("--verbose", action="store_true",
                    help="Print a few example disagreements (real mode).")
    # synthetic separability sweep
    ap.add_argument("--synthetic-scale", type=int, default=0,
                    help="If >0, run the synthetic separability sweep with this "
                         "many crystals (ignores --db).")
    ap.add_argument("--cluster-size", type=int, default=8,
                    help="Crystals per shared topic anchor (near-siblings that "
                         "compete for routing). Default 8.")
    ap.add_argument("--prompts-per-crystal", type=int, default=4,
                    help="Prompts summed per synthetic routing vector.")
    ap.add_argument("--prompt-spread", type=float, default=0.3,
                    help="Intra-crystal prompt/query scatter (norm relative to "
                         "the unit centroid). Default 0.3.")
    ap.add_argument("--gap-grid", default=None,
                    help="Comma list of centroid gaps to sweep. Default: "
                         "geomspace(0.005, 0.25, 8) (spans home@1 ~0.1->~0.95).")
    ap.add_argument("--anisotropy", type=float, default=0.0,
                    help="Embedding-cone bias toward one shared direction "
                         "(stresses sign-bit balance). 0 = isotropic.")
    ns = ap.parse_args()

    if ns.synthetic_scale and ns.synthetic_scale > 0:
        return _run_synthetic(ns)
    if not ns.db:
        ap.error("provide --db for a real bank, or --synthetic-scale N for the "
                 "scale test.")
    return await _run_real(ns)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
