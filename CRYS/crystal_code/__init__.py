"""Crystal Cache coding agent — the terminal doorway.

This is a SEGMENTED top-level component. It lives at the repo root in
`CRYS/`, separate from the `crystal_cache` library under `memory/src/`.

Segmentation contract (see docs/CODING_AGENT_BUILD_PLAN.md):
  - This package IMPORTS the crystal_cache library; it never edits or
    reaches into the library's internals.
  - It builds its own runtime (store, encoder, vector stores, Anthropic
    client) from crystal_cache's public building blocks — it does not
    use the web app's startup code.
  - Everything the coding agent needs lives inside `CRYS/`.
    Deleting that folder removes the coding agent and leaves the
    library untouched. That is the test for "properly segmented."
"""
