#!/usr/bin/env python3
"""Code-descriptions A/B eval (Checkpoint 3, CRYS June 2026).

Answers the one question the whole effort rests on: does indexing code by a
generated natural-language description (CC_ENABLE_CODE_DESCRIPTIONS) let
FUNCTIONAL queries — phrased without the symbols' own vocabulary — find code
that raw-code indexing misses?

It runs ONE functional query set against TWO banks of the SAME source:
  baseline   — code-encoded (descriptions OFF): fact.vector = encode_native(code)
  treatment  — description-encoded (descriptions ON): fact.vector = encode_native(desc)
and reports recall@k / MRR for each, side by side, with the delta.

No calibration here — this measures the raw cosine ranking on each bank (the
hybrid-rank calibration was shelved; this is about representation, not
re-ranking). Build the two banks by ingesting the same subtree twice:

  # baseline (code-encoded)
  python -m crys "$REPO" --db /tmp/cc_off.db        # then /ingest the subtree
  # treatment (description-encoded)
  CC_ENABLE_CODE_DESCRIPTIONS=1 python -m crys "$REPO" --db /tmp/cc_on.db   # /ingest same subtree

Then (MAIN venv, gtr-t5-base loads ~1 min):
  python scripts/eval_code_descriptions.py \\
      --baseline-db /tmp/cc_off.db --treatment-db /tmp/cc_on.db --verbose

A query is a HIT if any returned content_chunk key contains any of its `expect`
substrings. See scripts/eval_code_descriptions_queries.jsonl.
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

CONTENT_PAIR_TYPES = ["content_chunk"]
K_VALUES = (5, 10)


def _resolve_db_url(db_arg: str | None) -> str | None:
    if not db_arg:
        return None
    if "://" in db_arg:
        return db_arg
    return f"sqlite+aiosqlite:///{Path(db_arg).expanduser().resolve()}"


def _make_store(db_url: str | None) -> MetadataStore:
    # Each store is bound to its own DB; FactVectorStore.search uses the store
    # handed to its constructor, so the two banks never cross. No global
    # set_metadata_store — nothing in the search path reads it.
    if db_url is None:
        return MetadataStore()
    settings = get_settings().model_copy(update={"database_url": db_url})
    return MetadataStore(settings_override=settings)


def _first_hit_rank(ranked_keys: list[str], expected: list[str]) -> int | None:
    for i, key in enumerate(ranked_keys):
        k = key or ""
        if any(sub in k for sub in expected):
            return i + 1
    return None


def _summarize(ranks: list[int | None], n: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in K_VALUES:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        out[f"recall@{k}"] = hits / n if n else 0.0
    out["mrr"] = (sum(1.0 / r for r in ranks if r is not None) / n) if n else 0.0
    return out


async def _ranks_for_bank(store, encoder, customer, items, pool_k) -> list[int | None]:
    fact_store = FactVectorStore(store=store)
    ranks: list[int | None] = []
    for it in items:
        qvec = await encode_native_async(encoder, it["query"])
        results = await fact_store.search(
            customer_id=customer, query_vector=qvec,
            pair_types=CONTENT_PAIR_TYPES, k=pool_k, with_keys=True,
        )
        ranked_keys = [r[4] for r in results]
        ranks.append(_first_hit_rank(ranked_keys, it["expect"]))
    return ranks


async def main() -> int:
    ap = argparse.ArgumentParser(description="Code-descriptions A/B retrieval eval.")
    ap.add_argument("--baseline-db", required=True,
                    help="Code-encoded bank (descriptions OFF).")
    ap.add_argument("--treatment-db", required=True,
                    help="Description-encoded bank (descriptions ON).")
    ap.add_argument("--baseline-customer", default="CRYS-local")
    ap.add_argument("--treatment-customer", default="CRYS-local")
    ap.add_argument("--queries",
                    default=str(Path(__file__).with_name("eval_code_descriptions_queries.jsonl")))
    ap.add_argument("--pool", type=int, default=None,
                    help="Candidate pool size (default: settings.hybrid_rank_pool_size).")
    ap.add_argument("--verbose", action="store_true", help="Per-query rank table.")
    ns = ap.parse_args()

    pool_k = ns.pool or getattr(get_settings(), "hybrid_rank_pool_size", 50)

    items: list[dict] = []
    with open(ns.queries, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(json.loads(line))
    if not items:
        print(f"No queries in {ns.queries}")
        return 1

    encoder = build_text_encoder()  # shared; gtr-t5-base, slow first load

    base_store = _make_store(_resolve_db_url(ns.baseline_db))
    treat_store = _make_store(_resolve_db_url(ns.treatment_db))
    await base_store.init()
    await treat_store.init()
    try:
        base_ranks = await _ranks_for_bank(
            base_store, encoder, ns.baseline_customer, items, pool_k)
        treat_ranks = await _ranks_for_bank(
            treat_store, encoder, ns.treatment_customer, items, pool_k)
    finally:
        await base_store.dispose()
        await treat_store.dispose()

    n = len(items)
    base = _summarize(base_ranks, n)
    treat = _summarize(treat_ranks, n)

    improved = worsened = unchanged = 0
    for rb, rt in zip(base_ranks, treat_ranks):
        if rt is not None and (rb is None or rt < rb):
            improved += 1
        elif rb is not None and (rt is None or rt > rb):
            worsened += 1
        else:
            unchanged += 1

    print(f"\nCode-descriptions A/B — {n} functional queries, pool={pool_k}")
    print(f"  baseline  (code-encoded):  {ns.baseline_db}  [{ns.baseline_customer}]")
    print(f"  treatment (desc-encoded):  {ns.treatment_db}  [{ns.treatment_customer}]")

    if ns.verbose:
        print("\n  base  treat     query")
        for it, rb, rt in zip(items, base_ranks, treat_ranks):
            mark = "+" if (rt is not None and (rb is None or rt < rb)) else (
                   "-" if (rb is not None and (rt is None or rt > rb)) else ".")
            print(f"  {str(rb if rb is not None else '—'):>4}  "
                  f"{str(rt if rt is not None else '—'):>4}  {mark}  {it['query'][:60]}")

    print("\n  metric        baseline(code)   treatment(desc)    delta")
    for key in ("recall@5", "recall@10", "mrr"):
        b, t = base[key], treat[key]
        print(f"  {key:<12}  {b:>12.3f}   {t:>13.3f}   {t - b:>+7.3f}")
    print(f"\n  per-query rank change:  +{improved} improved   "
          f"-{worsened} worsened   .{unchanged} unchanged\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
