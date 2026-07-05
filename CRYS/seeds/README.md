# Seed banks — provenance and harvest pipeline

Each `.jsonl` file here is one importable general-knowledge bank:
one JSON object per line, `{"key", "claim", "source"?}`. Keys are
wide→specific sparse keys under the `General|` namespace; claims are
pattern-form (imperative rules, not code dumps) — the form factor the
BCB Hard experiments validated (+17.6pp from pattern rules, zero lift
from raw code; see crystal-cache-v1/docs/BCB_BENCHMARK_FINDINGS.md).

Import (replace semantics per type; re-running is a sync):

    cd coding-agent
    python -m crys --seed-general seeds/general_swe_patterns.jsonl
    python -m crys --seed-general seeds/general_python_pep8.jsonl --general-type general:python_pep8
    python -m crys --seed-general seeds/general_python_google.jsonl --general-type general:python_google
    python -m crys --seed-general seeds/general_testing_pytest.jsonl --general-type general:testing_pytest
    python -m crys --seed-general seeds/general_sql_postgres.jsonl --general-type general:sql_postgres

## Files

| File | Type | Provenance |
|---|---|---|
| `general_swe_patterns.jsonl` | `general:swe_patterns` | Curated by Claude from consensus engineering practice (cross-domain: errors, testing, API, security, concurrency, performance, git, design, review, ops). No single citable source — treat as the editorial layer. |
| `general_python_pep8.jsonl` | `general:python_pep8` | Harvested 2026-06-12 from PEP 8 (peps.python.org/pep-0008, public domain), full text fetched and transformed into pattern-form. Per-entry `source` field. |
| `general_python_google.jsonl` | `general:python_google` | Harvested 2026-06-12 from the Google Python Style Guide (google.github.io/styleguide/pyguide.html, CC-BY 3.0), sections 2.1–3.11, paraphrased into pattern-form with per-entry `source` field citing section numbers. |
| `general_testing_pytest.jsonl` | `general:testing_pytest` | Harvested 2026-06-12 from the pytest documentation (docs.pytest.org: Good Integration Practices fetched in full, plus core how-to pages — assertions, fixtures, parametrize, tmp_path, monkeypatch, markers, skip/xfail, flaky tests). Per-entry `source` names the doc page. |
| `general_sql_postgres.jsonl` | `general:sql_postgres` | Harvested 2026-06-12 from the PostgreSQL wiki "Don't Do This" page (wiki.postgresql.org/wiki/Don't_Do_This), fetched in full — every anti-pattern with its documented exceptions folded into the claim. |

## Harvest pipeline (how to grow this)

The harvest is: fetch an authoritative source → extract its directives
→ rewrite each as a one-to-two-sentence imperative pattern (paraphrase,
never copy; cite the source per entry) → key it `General|Domain|Topic|Slug`
→ import under its own `general:` type so each source can be re-synced
independently.

Next sources, in priority order (harvest paused 2026-06-12 — reflection
loop takes precedence; these remain queued):
1. OWASP Cheat Sheet Series (cheatsheetseries.owasp.org) — security directives
2. The Twelve-Factor App (12factor.net) — service/ops directives
3. PEP 20, PEP 257, PEP 484/585 — remaining core-Python conventions

Quality bar per entry: would a senior engineer state this as a rule in
review? If it needs a paragraph of caveats, it's documentation, not a
pattern — link it via /ingest instead.
