#!/usr/bin/env python3
"""Hybrid-rank retrieval eval (stage 1a).

Measures whether code/prose calibration (CC_ENABLE_HYBRID_RANK) lifts the
right CODE crystal into the top-k for conceptual queries, versus the raw
gtr-t5-base cosine baseline — on a real, ingested bank.

DESIGN — one retrieval, two rankings, no env toggling:
Per query we run ONE FactVectorStore.search(..., with_keys=True) over the
content_chunk pool, then score TWO orderings from that single pool using
the REAL production code:

  baseline (off)  — raw cosine order (what ContentRouter returns with
                    CC_ENABLE_HYBRID_RANK off): the pool as search sorts it.
  calibrated (on) — retrieval.v3_routers._calibrate_by_subtype(pool) (the
                    exact function ContentRouter calls with the flag on).

Because both orderings come from the same pool and the same functions, the
delta is apples-to-apples with zero drift. The harness never flips the env
flag and never re-runs retrieval.

The stack (store + encoder + fact store) is built from the crystal_cache
library's public building blocks exactly as the coding agent builds it —
see CRYS/crystal_code/runtime.py::build_agent.

A query counts as a HIT at rank r if the r-th returned content_chunk key
contains any of that query's `expect` substrings (a distinctive symbol
name and/or a src/ file path). See scripts/eval_hybrid_rank_queries.jsonl.

Usage (from the repo root, MAIN venv active — loads gtr-t5-base, ~1 min):
    python scripts/eval_hybrid_rank.py --db "$WORK_REPO/crystal_cache.db"
    python scripts/eval_hybrid_rank.py --db "$WORK_REPO/crystal_cache.db" --verbose
    python scripts/eval_hybrid_rank.py --db /path/to/crystal_cache.db \\
        --customer CRYS-local --queries scripts/eval_hybrid_rank_queries.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from crystal_cache.config import get_settings
from crystal_cache.encoding import build_text_encoder
from crystal_cache.encoding.executor import encode_native_async
from crystal_cache.infrastructure import MetadataStore
from crystal_cache.infrastructure.fact_vector_store import FactVectorStore
from crystal_cache.infrastructure.metadata_store import set_metadata_store
from crystal_cache.retrieval.v3_routers import _calibrate_by_subtype

# Content chunks are the pair_type that holds code (keyed "Code|path|symbol")
# AND prose (ledger/docs/cognition). The bug + the 1a fix both live in how
# this single pool is ranked, so the eval searches exactly it.
CONTENT_PAIR_TYPES = ["content_chunk"]
K_VALUES = (5, 10)


# --- stack construction: mirrors CRYS/crystal_code/runtime.py ------

def _resolve_db_url(db_arg: str | None) -> str | None:
    """A --db file path becomes a SQLite async URL; a full URL passes through."""
    if not db_arg:
        return None
    if "://" in db_arg:
        return db_arg
    return f"sqlite+aiosqlite:///{Path(db_arg).expanduser().resolve()}"


def _make_store(db_url: str | None) -> MetadataStore:
    if db_url is None:
        return MetadataStore()
    settings = get_settings().model_copy(update={"database_url": db_url})
    return MetadataStore(settings_override=settings)


# --- scoring ---------------------------------------------------------------

def _first_hit_rank(ranked_keys: list[str], expected: list[str]) -> int | None:
    """1-based rank of the first key containing any expected substring, else None."""
    for i, key in enumerate(ranked_keys):
        k = key or ""
        if any(sub in k for sub in expected):
            return i + 1
    return None


def _summarize(ranks: list[int | None], n: int) -> dict[str, float]:
    """recall@5, recall@10, and MRR (1/rank over all queries, miss = 0)."""
    out: dict[str, float] = {}
    for k in K_VALUES:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        out[f"recall@{k}"] = hits / n if n else 0.0
    out["mrr"] = (sum(1.0 / r for r in ranks if r is not None) / n) if n else 0.0
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description="Hybrid-rank (1a) retrieval eval.")
    ap.add_argument("--db", required=True,
                    help="SQLite file path or full DB URL of the ingested bank.")
    ap.add_argument("--customer", default="CRYS-local",
                    help="Customer id whose bank to search (default: CRYS-local).")
    ap.add_argument("--queries",
                    default=str(Path(__file__).with_name("eval_hybrid_rank_queries.jsonl")),
                    help="Labeled query set (JSONL).")
    ap.add_argument("--pool", type=int, default=None,
                    help="Candidate pool size (default: settings.hybrid_rank_pool_size).")
    ap.add_argument("--verbose", action="store_true",
                    help="Print a per-query rank table (+ improved, - worsened, . same).")
    ns = ap.parse_args()

    pool_k = ns.pool or getattr(get_settings(), "hybrid_rank_pool_size", 50)

    # Load the labeled set (skip blank lines and # comments).
    items: list[dict] = []
    with open(ns.queries, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(json.loads(line))
    if not items:
        print(f"No queries found in {ns.queries}")
        return 1

    store = _make_store(_resolve_db_url(ns.db))
    await store.init()              # no-op on a populated DB
    set_metadata_store(store)       # some library paths read the process-wide store
    encoder = build_text_encoder()  # gtr-t5-base; slow first load
    fact_store = FactVectorStore(store=store)

    base_ranks: list[int | None] = []
    cal_ranks: list[int | None] = []
    rows: list[tuple] = []          # (query, rb, rc, pool_size, code_in_pool)
    try:
        for it in items:
            query = it["query"]
            expected = it["expect"]
            qvec = await encode_native_async(encoder, query)
            pool = await fact_store.search(
                customer_id=ns.customer,
                query_vector=qvec,
                pair_types=CONTENT_PAIR_TYPES,
                k=pool_k,
                with_keys=True,
            )
            # base = cosine order (search already sorts desc by score);
            # cal = the real calibration reorder, remapped back to keys
            # (it strips the 5th element, so map fact_id -> key from the pool).
            base_keys = [r[4] for r in pool]
            key_by_fid = {r[0]: r[4] for r in pool}
            cal_keys = [key_by_fid[r[0]] for r in _calibrate_by_subtype(pool)]
            code_in_pool = sum(1 for kk in base_keys if (kk or "").startswith("Code|"))

            rb = _first_hit_rank(base_keys, expected)
            rc = _first_hit_rank(cal_keys, expected)
            base_ranks.append(rb)
            cal_ranks.append(rc)
            rows.append((query, rb, rc, len(pool), code_in_pool))
    finally:
        await store.dispose()

    n = len(items)
    base = _summarize(base_ranks, n)
    cal = _summarize(cal_ranks, n)

    def _better(rb: int | None, rc: int | None) -> int:
        # +1 improved (calibrated rank lower/found), -1 worsened, 0 unchanged
        if rc is not None and (rb is None or rc < rb):
            return 1
        if rb is not None and (rc is None or rc > rb):
            return -1
        return 0

    deltas = [_better(rb, rc) for _, rb, rc, _, _ in rows]
    improved = sum(1 for d in deltas if d > 0)
    worsened = sum(1 for d in deltas if d < 0)
    unchanged = sum(1 for d in deltas if d == 0)
    avg_pool = sum(r[3] for r in rows) / n
    avg_code = sum(r[4] for r in rows) / n

    print(f"\nHybrid-rank eval (1a)  —  {n} queries  •  customer={ns.customer}  •  pool={pool_k}")
    print(f"  bank: {ns.db}")
    print(f"  pool composition (avg): {avg_pool:.0f} candidates, {avg_code:.0f} code / "
          f"{avg_pool - avg_code:.0f} prose")

    if ns.verbose:
        print("\n  base  calib     query")
        marks = {1: "+", -1: "-", 0: "."}
        for (q, rb, rc, _, _), d in zip(rows, deltas):
            print(f"  {str(rb if rb is not None else '—'):>4}  "
                  f"{str(rc if rc is not None else '—'):>4}  {marks[d]}  {q[:62]}")

    print("\n  metric        baseline(off)   calibrated(on)    delta")
    for key in ("recall@5", "recall@10", "mrr"):
        b, c = base[key], cal[key]
        print(f"  {key:<12}  {b:>12.3f}   {c:>13.3f}   {c - b:>+7.3f}")
    print(f"\n  per-query rank change:  +{improved} improved   "
          f"-{worsened} worsened   .{unchanged} unchanged")
    if avg_code in (0,) or avg_code == avg_pool:
        print("  NOTE: pools look single-modality (all prose or all code); "
              "calibration is a structural no-op here — check the bank/customer.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
