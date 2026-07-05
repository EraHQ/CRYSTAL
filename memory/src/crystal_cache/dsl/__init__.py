"""Crystal DSL — v0.2, experimental.

A small configuration language for expressing crystal structures,
routing rules, reasoning chains, and verification policies using HDC
primitives as the underlying data model.

ARCHITECTURE
------------
The DSL operates in CONCEPT SPACE — dense bipolar hypervectors derived
deterministically via SHA-256. This is distinct from PromptEncoder's
TEXT SPACE (sparse sum-of-token-hashes used for Fact retrieval).

The two spaces don't mix via vector math. The intended bridge is a
decomposer (small LLM in structured-output mode) that reads free text
and emits a structured payload. `from_decomposer_output()` turns that
payload into a concept-space hypervector that can be compared against
DSL-compiled configs.

STATUS: EXPERIMENTAL. Syntax and semantics subject to change.

Public API:

    from crystal_cache.dsl import run, SHARED_TENANT

    # Compile a DSL source to hypervectors
    env = run(source_text, tenant_id="acme_corp")
    env.configs["user_alice"]                         # compiled hypervector
    env.get_field("user_alice", "mode")               # (name, similarity)

    # Bridge from a decomposer model's output
    from crystal_cache.dsl import from_decomposer_output
    query_hv = from_decomposer_output(
        {"intent": "solve_problem", "topic": "algebra"},
        env.vocab,
    )
    ranked = env.rank_configs(query_hv)               # [(name, sim), ...]
"""
from __future__ import annotations

from crystal_cache.dsl.patterns import (
    SHARED_TENANT,
    Cleanup,
    StepCodebook,
    Vocabulary,
    bind,
    branch,
    bundle,
    chain_step,
    get_at_position,
    lookup,
    make_lookup,
    make_record,
    make_sequence,
    random_hv,
    read_field,
    resolve_branch,
    shift,
    similarity,
)
from crystal_cache.dsl.parser import ParseError, parse
from crystal_cache.dsl.compiler import CompileError, compile_ast
from crystal_cache.dsl.runtime import (
    RuntimeEnv,
    RuntimeError as DSLRuntimeError,
    eval_compiled,
    from_decomposer_output,
    run,
)

__all__ = [
    # Tenancy
    "SHARED_TENANT",
    # Runtime primitives (stable)
    "Vocabulary",
    "Cleanup",
    "StepCodebook",
    "make_record",
    "read_field",
    "make_lookup",
    "lookup",
    "branch",
    "resolve_branch",
    "make_sequence",
    "get_at_position",
    "chain_step",
    "shift",
    "bind",
    "bundle",
    "similarity",
    "random_hv",
    # Parser/compiler (experimental v0)
    "parse",
    "ParseError",
    "compile_ast",
    "CompileError",
    # Runtime entry points
    "run",
    "eval_compiled",
    "RuntimeEnv",
    "DSLRuntimeError",
    # Bridge from decomposer output
    "from_decomposer_output",
]

__experimental__ = True
__dsl_version__ = "0.2.0-experimental"
