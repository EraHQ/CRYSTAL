"""Crystal DSL runtime — evaluate compiled IR to hypervectors.

The runtime takes a CompiledProgram (from compiler.py) and evaluates
it against a Vocabulary. The result is a RuntimeEnv containing:

  - vocab: the populated Vocabulary (tenant-scoped)
  - cleanup: Cleanup memory over that vocab
  - bindings: name -> hypervector, for every assignment
  - configs: name -> hypervector, for every config block
  - codebooks: name -> StepCodebook, one per chain
  - warnings: list of implicit-concept names

VECTOR SPACE
------------
The runtime operates in the DSL's concept space (dense bipolar vectors
derived via SHA-256). This is distinct from PromptEncoder's text space
(sparse sum-of-hashed-tokens). Free-text queries do NOT automatically
cross into concept space; the intended bridge is an upstream decomposer
(a small LLM in structured-output mode) that emits DSL expressions from
text. See `from_decomposer_output` for the entry point.

LITERALS
--------
- String literals are treated as concept names. `"hello"` becomes the
  concept `hello`. This is consistent with how identifiers work — a
  string value in a field is semantically "this concept" just spelled
  with quotes for cases where the identifier wouldn't be a valid Python
  name (e.g. contains spaces or special characters).
- Number literals become `num_{value}` concepts.
- Boolean literals become `true`/`false` concepts.

No auto-tokenization of strings through PromptEncoder. That would mix
vector spaces and break bind/unbind math.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from crystal_cache.dsl.compiler import (
    CompiledExpr,
    CompiledProgram,
    OpBool,
    OpBranch,
    OpChain,
    OpChainStep,
    OpConcept,
    OpLookup,
    OpNumber,
    OpRecord,
    OpRef,
    OpSequence,
    OpString,
    compile_ast,
)
from crystal_cache.dsl.parser import parse
from crystal_cache.dsl.patterns import (
    SHARED_TENANT,
    Cleanup,
    StepCodebook,
    Vocabulary,
    branch,
    chain_step,
    make_lookup,
    make_record,
    make_sequence,
    read_field,
    similarity,
)


class RuntimeError(Exception):
    pass


@dataclass
class RuntimeEnv:
    """Evaluated DSL program."""

    vocab: Vocabulary
    cleanup: Cleanup
    bindings: dict[str, np.ndarray] = field(default_factory=dict)
    configs: dict[str, np.ndarray] = field(default_factory=dict)
    codebooks: dict[str, StepCodebook] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def get_field(
        self, config_name: str, field_name: str
    ) -> tuple[Optional[str], float]:
        if config_name not in self.configs:
            raise RuntimeError(f"no such config: {config_name!r}")
        return read_field(self.configs[config_name], field_name, self.vocab, self.cleanup)

    def similarity(self, name_a: str, name_b: str) -> float:
        a = self._get_hv(name_a)
        b = self._get_hv(name_b)
        return similarity(a, b)

    def rank_configs(self, query_hv: np.ndarray) -> list[tuple[str, float]]:
        """Compare a concept-space query vector against every compiled config.

        The query_hv must already be in concept space — typically built by
        `from_decomposer_output()` from a small-LLM's structured
        representation of a free-text query.
        """
        scored = [(name, similarity(query_hv, hv)) for name, hv in self.configs.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _get_hv(self, name: str) -> np.ndarray:
        if name in self.configs:
            return self.configs[name]
        if name in self.bindings:
            return self.bindings[name]
        if name in self.vocab:
            return self.vocab.get(name)
        raise RuntimeError(f"no hypervector named {name!r}")


def _eval_expr(op: CompiledExpr, env: RuntimeEnv) -> np.ndarray:
    if isinstance(op, OpConcept):
        return env.vocab.get(op.name)

    if isinstance(op, OpRef):
        if op.name not in env.bindings:
            raise RuntimeError(
                f"reference to undefined name {op.name!r} (line {op.source_line})"
            )
        return env.bindings[op.name]

    if isinstance(op, OpString):
        # Strings are concept names, same as identifiers. This keeps
        # the DSL inside concept space — no mixing with PromptEncoder's
        # text space.
        return env.vocab.get(op.value)

    if isinstance(op, OpNumber):
        # Integer-valued floats normalize to int so `42` == `42.0`
        if op.value == int(op.value):
            key = f"num_{int(op.value)}"
        else:
            key = f"num_{op.value}"
        return env.vocab.get(key)

    if isinstance(op, OpBool):
        return env.vocab.get("true" if op.value else "false")

    if isinstance(op, OpRecord):
        fields = {k: _eval_expr(v, env) for k, v in op.fields.items()}
        return make_record(fields, env.vocab)

    if isinstance(op, OpLookup):
        entries = {k: _eval_expr(v, env) for k, v in op.entries.items()}
        return make_lookup(entries, env.vocab)

    if isinstance(op, OpBranch):
        condition_hv = env.vocab.get(op.condition_concept)
        if_true_hv = _eval_expr(op.if_true, env)
        if_false_hv = _eval_expr(op.if_false, env)
        return branch(condition_hv, if_true_hv, if_false_hv, env.vocab)

    if isinstance(op, OpSequence):
        items = [_eval_expr(i, env) for i in op.items]
        return make_sequence(items)

    if isinstance(op, OpChain):
        codebook = StepCodebook()
        prev_hv: Optional[np.ndarray] = None
        last_step_hv: Optional[np.ndarray] = None
        for step_op in op.steps:
            assert isinstance(step_op, OpChainStep)
            step_fields = {k: _eval_expr(v, env) for k, v in step_op.fields.items()}
            step_hv = chain_step(
                prev_hv, step_fields, env.vocab,
                codebook=codebook, step_name=step_op.name,
            )
            prev_hv = step_hv
            last_step_hv = step_hv
        idx = len(env.codebooks)
        env.codebooks[f"_chain_{idx}"] = codebook
        if last_step_hv is None:
            return np.zeros(env.vocab.d, dtype=np.int8)
        return last_step_hv

    raise RuntimeError(f"unknown op type: {type(op).__name__}")


def run(
    source: str,
    *,
    tenant_id: str = SHARED_TENANT,
    vocab: Optional[Vocabulary] = None,
    cleanup: Optional[Cleanup] = None,
) -> RuntimeEnv:
    """Parse, compile, and execute a DSL source string.

    `tenant_id` scopes all concepts to a tenant. If `vocab` is supplied,
    its tenancy is used and `tenant_id` is ignored.
    """
    program = parse(source)
    compiled = compile_ast(program)
    return eval_compiled(compiled, tenant_id=tenant_id, vocab=vocab, cleanup=cleanup)


def eval_compiled(
    compiled: CompiledProgram,
    *,
    tenant_id: str = SHARED_TENANT,
    vocab: Optional[Vocabulary] = None,
    cleanup: Optional[Cleanup] = None,
) -> RuntimeEnv:
    vocab = vocab or Vocabulary(tenant_id=tenant_id)
    cleanup = cleanup or Cleanup(vocab)

    env = RuntimeEnv(vocab=vocab, cleanup=cleanup)
    env.warnings = list(compiled.implicit_concepts)

    for name in compiled.concepts:
        vocab.get(name)

    for assignment in compiled.assignments:
        hv = _eval_expr(assignment.expr, env)
        env.bindings[assignment.name] = hv

    for config in compiled.configs:
        field_hvs = {k: _eval_expr(v, env) for k, v in config.fields.items()}
        env.configs[config.name] = make_record(field_hvs, env.vocab)

    return env


# ------------------------------------------------------------
# Bridge from decomposer output -> concept-space hypervector
# ------------------------------------------------------------


def from_decomposer_output(
    payload: Any,
    vocab: Vocabulary,
) -> np.ndarray:
    """Build a concept-space hypervector from a decomposer's structured output.

    This is the intended bridge between free-text queries and the DSL.
    A small LLM (e.g. Llama 3.2 3B in JSON-schema mode) reads a user's
    message and emits a structured payload describing the intent. That
    payload maps directly onto DSL patterns:

        dict   → record     (field-name -> value bindings)
        list   → sequence   (position-shifted bundle)
        str    → concept    (named hypervector)
        int    → num_{N}    (named hypervector)
        float  → num_{f}    (named hypervector)
        bool   → true/false (named hypervector)
        None   → null       (named hypervector)
        nested dicts/lists → recursively applied

    Example decomposer output for "help me solve an algebra problem":

        {
          "intent": "solve_problem",
          "topic": "algebra",
          "action": "retrieve_past_skill"
        }

    Passing this to `from_decomposer_output` yields a record hypervector
    with three bindings: `intent->solve_problem`, `topic->algebra`,
    `action->retrieve_past_skill`. That vector can be compared via
    `env.rank_configs()` against stored DSL configs to find semantic
    matches.

    The decomposer model itself is upstream of this function — not
    implemented here. This function assumes the model has already run.
    """
    return _encode_payload(payload, vocab)


def _encode_payload(payload: Any, vocab: Vocabulary) -> np.ndarray:
    if isinstance(payload, dict):
        fields = {k: _encode_payload(v, vocab) for k, v in payload.items()}
        return make_record(fields, vocab)
    if isinstance(payload, (list, tuple)):
        items = [_encode_payload(item, vocab) for item in payload]
        return make_sequence(items)
    if isinstance(payload, bool):
        return vocab.get("true" if payload else "false")
    if isinstance(payload, int):
        return vocab.get(f"num_{payload}")
    if isinstance(payload, float):
        if payload == int(payload):
            return vocab.get(f"num_{int(payload)}")
        return vocab.get(f"num_{payload}")
    if isinstance(payload, str):
        return vocab.get(payload)
    if payload is None:
        return vocab.get("null")
    raise RuntimeError(
        f"cannot encode payload of type {type(payload).__name__}: {payload!r}"
    )
