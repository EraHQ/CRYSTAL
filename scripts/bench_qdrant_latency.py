#!/usr/bin/env python3
"""Qdrant routing-lane latency + recall benchmark — the follow-on to the
binary-quant RECALL gate (scripts/bench_routing_quantization.py).

WHAT THIS ADDS OVER THE GATE
----------------------------
The gate proved, in NumPy, that binary-quant + EXACT rescore reproduces
exact-float routing (top1-agree = 1.000 at scale, even at home@1 ~0.1 and
under anisotropy). But it used an EXHAUSTIVE Hamming coarse stage and NumPy
timings. Real Qdrant adds two things the gate could not measure:
  • HNSW graph approximation ON TOP of binary quant (can miss the winner even
    when exhaustive Hamming wouldn't) -> real recall is <= the gate's ceiling.
  • Real engine latency (HNSW walk over binary codes, not a full popcount
    scan) -> real p95.
So this benchmark measures, against the SAME exhaustive-float ground truth:
  arm A "float_hnsw"  — plain HNSW over float vectors (no quantization).
  arm B "binary+rescore" — HNSW + binary quantization + float rescore
                           (oversampling) — the proposed fast path.
For each: p50/p95/p99 query latency and recall@1 / recall@k vs exact float.

REQUIRES A RUNNING QDRANT (the in-process client stub is NOT representative):
    docker run -d -p 6333:6333 qdrant/qdrant
Then point --qdrant-url at it (default http://localhost:6333).

DATA: reuses the validated synthetic generator from
scripts/bench_routing_quantization.py (clustered, CRYS-shaped routing vectors;
P rebuilt model-free). No model load. One separability level (--gap); the
recall question across separability was already answered by the gate.

USAGE (repo root; needs `pip install qdrant-client`):
    python scripts/bench_qdrant_latency.py --synthetic-scale 10000
    python scripts/bench_qdrant_latency.py --synthetic-scale 20000 \\
        --oversampling 2 --k 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Reuse the validated synthetic generator (clustered routing vectors + P).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_routing_quantization as gen  # noqa: E402

from qdrant_client import QdrantClient, models  # noqa: E402

EXACT_COLL = "crys_routing_float"
BINARY_COLL = "crys_routing_binary"


def _ground_truth_topk(r_float: np.ndarray, q_float: np.ndarray, k: int) -> np.ndarray:
    """Exhaustive exact-float top-k crystal indices per query (the reference)."""
    n = r_float.shape[0]
    q = q_float.shape[0]
    out = np.empty((q, min(k, n)), dtype=np.int64)
    kk = min(k, n)
    for s in range(0, q, 512):
        scores = r_float @ q_float[s:s + 512].T          # (n, chunk)
        for j in range(scores.shape[1]):
            col = scores[:, j]
            top = np.argpartition(-col, kk - 1)[:kk]
            out[s + j] = top[np.argsort(-col[top])]
    return out


def _make_collection(client, name, dim, binary):
    if client.collection_exists(name):
        client.delete_collection(name)
    quant = (models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(always_ram=True))
             if binary else None)
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
        # Force HNSW to build even below the default 20k threshold so both arms
        # actually use the graph (otherwise small banks stay brute-force).
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1000),
        quantization_config=quant,
    )


def _wait_indexed(client, name, n, timeout_s=600):
    """Wait until HNSW has (effectively) finished indexing the collection.

    Qdrant commonly leaves a small tail of vectors in an unoptimized segment,
    so ``indexed_vectors_count`` can plateau a few short of ``n`` indefinitely
    — demanding an exact ``idx >= n`` would burn the whole timeout every run.
    So we return when the count reaches ``n`` OR when it has gone near-complete
    (>= 98% of n) and stopped climbing for a few polls (the optimizer settled).
    A plateau well below 98% is a real problem and is left to time out.
    """
    t0 = time.time()
    last = -1
    stable = 0
    near_floor = int(0.98 * n)
    while time.time() - t0 < timeout_s:
        idx = client.get_collection(name).indexed_vectors_count or 0
        if idx >= n:
            return idx
        if idx == last:
            stable += 1
            if idx >= near_floor and stable >= 3:   # ~6s settled, near-complete
                return idx
        else:
            stable = 0
            last = idx
        time.sleep(2.0)
    return client.get_collection(name).indexed_vectors_count or 0


def _bench_arm(client, name, q_float, k, search_params):
    lat_ms: list[float] = []
    returned: list[list[int]] = []
    for i in range(q_float.shape[0]):
        t0 = time.perf_counter()
        resp = client.query_points(
            collection_name=name,
            query=q_float[i].tolist(),
            limit=k,
            search_params=search_params,
            with_payload=False,
        )
        lat_ms.append((time.perf_counter() - t0) * 1000.0)
        returned.append([int(p.id) for p in resp.points])
    return lat_ms, returned


def _recall(returned, gt_topk, k):
    """recall@1 (gt top-1 is returned #1) and recall@k (gt top-1 in returned k)."""
    q = len(returned)
    r1 = sum(1 for i in range(q) if returned[i] and returned[i][0] == int(gt_topk[i][0]))
    rk = sum(1 for i in range(q) if int(gt_topk[i][0]) in returned[i][:k])
    return r1 / q, rk / q


def _pct(xs, p):
    return float(np.percentile(np.asarray(xs), p))


def main() -> int:
    ap = argparse.ArgumentParser(description="Qdrant routing-lane latency + recall.")
    ap.add_argument("--qdrant-url", default="http://localhost:6333",
                    help="URL of a running Qdrant (default http://localhost:6333).")
    ap.add_argument("--synthetic-scale", type=int, default=10000,
                    help="Number of crystals (routing vectors) to index.")
    ap.add_argument("--queries", type=int, default=2000, help="Queries to time.")
    ap.add_argument("--gap", type=float, default=0.05,
                    help="Centroid gap (separability). 0.05 ~ home@1 ~0.45.")
    ap.add_argument("--cluster-size", type=int, default=8)
    ap.add_argument("--prompts-per-crystal", type=int, default=4)
    ap.add_argument("--prompt-spread", type=float, default=0.3)
    ap.add_argument("--oversampling", type=float, default=2.0,
                    help="Binary-arm rescore oversampling (Qdrant pulls k*os "
                         "candidates by binary, rescores with float).")
    ap.add_argument("--k", type=int, default=10, help="top-k to retrieve.")
    ap.add_argument("--seed", type=int, default=42)
    ns = ap.parse_args()

    n = ns.synthetic_scale
    d_hdc = int(getattr(gen.get_settings(), "d_hdc", 10000))
    native = gen.SYNTH_NATIVE_DIM
    cs = max(2, ns.cluster_size)
    rng = np.random.RandomState(ns.seed)

    print(f"\nQdrant routing-lane latency + recall")
    print(f"  qdrant: {ns.qdrant_url}")
    print(f"  crystals: {n}   d_hdc: {d_hdc}   queries: {ns.queries}   "
          f"gap: {ns.gap}   k: {ns.k}   oversampling: {ns.oversampling}")

    # --- generate the bank + queries (reuse the gate's generator) ----------
    p_matrix = gen._build_projection(native, d_hdc)
    nclus = max(2, int(np.ceil(n / cs)))
    anchors = gen._l2(rng.randn(nclus, native).astype(np.float32))
    assign = rng.randint(0, nclus, size=n)
    centroids = gen._synth_centroids(anchors, assign, ns.gap, native, rng)
    r_float = gen._build_routing(centroids, ns.prompts_per_crystal,
                                 ns.prompt_spread, p_matrix, rng)
    picks = rng.choice(n, size=min(ns.queries, n), replace=False)
    q_float = gen._build_queries(centroids, picks, ns.prompt_spread, p_matrix, rng)
    gt = _ground_truth_topk(r_float, q_float, ns.k)
    home1 = float(np.mean(gt[:, 0] == picks))
    print(f"  exact-float home@1 of this bank: {home1:.3f}  (sanity: matches the "
          f"gate's separability dial)")

    client = QdrantClient(url=ns.qdrant_url, timeout=300)

    # --- build both collections + ingest -----------------------------------
    ids = list(range(n))
    for name, binary in ((EXACT_COLL, False), (BINARY_COLL, True)):
        label = "binary+quant" if binary else "float"
        print(f"\n  [{label}] creating collection + uploading {n} vectors "
              f"(dim {d_hdc})...", flush=True)
        _make_collection(client, name, d_hdc, binary)
        t0 = time.time()
        client.upload_collection(collection_name=name, vectors=r_float,
                                 ids=ids, batch_size=128, parallel=1)
        print(f"  [{label}] uploaded in {time.time() - t0:.1f}s; waiting for HNSW "
              f"index...", flush=True)
        idx = _wait_indexed(client, name, n)
        print(f"  [{label}] indexed_vectors_count = {idx}/{n}", flush=True)

    # --- time both arms vs the same ground truth ----------------------------
    print(f"\n  timing {ns.queries} queries per arm...", flush=True)
    exact_params = models.SearchParams(hnsw_ef=128)
    lat_e, ret_e = _bench_arm(client, EXACT_COLL, q_float, ns.k, exact_params)
    bin_params = models.SearchParams(
        hnsw_ef=128,
        quantization=models.QuantizationSearchParams(
            rescore=True, oversampling=ns.oversampling),
    )
    lat_b, ret_b = _bench_arm(client, BINARY_COLL, q_float, ns.k, bin_params)

    r1_e, rk_e = _recall(ret_e, gt, ns.k)
    r1_b, rk_b = _recall(ret_b, gt, ns.k)

    print(f"\n  arm              p50 ms   p95 ms   p99 ms   recall@1  recall@{ns.k}")
    print(f"  float HNSW       {_pct(lat_e,50):>6.2f}   {_pct(lat_e,95):>6.2f}   "
          f"{_pct(lat_e,99):>6.2f}   {r1_e:>8.3f}  {rk_e:>8.3f}")
    print(f"  binary+rescore   {_pct(lat_b,50):>6.2f}   {_pct(lat_b,95):>6.2f}   "
          f"{_pct(lat_b,99):>6.2f}   {r1_b:>8.3f}  {rk_b:>8.3f}")
    print(f"\n  ground truth = exhaustive exact-float (NumPy). recall@1 = arm's "
          f"#1 equals exact float's #1.")
    print(f"  latency = client->server round trip over {ns.qdrant_url} "
          f"(localhost has ~no network; still includes serialization + engine).")
    print("  read: if binary recall@1 stays ~the float arm's AND p95 is lower/"
          "comparable, the binary fast path wins. If binary recall@1 drops, "
          "raise --oversampling (more rescore candidates) and re-run.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
