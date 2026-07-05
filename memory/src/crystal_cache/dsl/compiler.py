"""Crystal DSL compiler — AST -> executable IR.

Walks the parsed AST and produces a CompiledProgram: a flat list of
compiled objects (CompiledConfig, CompiledAssignment) with all
expressions resolved into "ops" that the runtime knows how to execute.

The IR is intentionally simple — it's Python objects, not bytecode.
The runtime (runtime.py) evaluates each op by pattern-matching on type.

SEMANTICS

- Every IDENT referenced in an expression resolves to a hypervector at
  runtime. An identifier might be: (a) a concept declared via `concept`,
  (b) a name bound by an earlier `=` assignment, or (c) an unknown
  name, which is treated as an implicit concept declaration. This last
  behavior is convenient but we warn on it at compile time.

- `chain` steps register their completed step hypervectors in a
  StepCodebook under the step's source-level name. PREV references
  inside a step refer to the immediately previous step in the same
  chain.

- `config` blocks are top-level records. Their field values can be any
  expression. The compiled config is stored by name for later lookup.

STATUS: experimental, matches parser.py v0. Syntax and semantics subject
to change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from crystal_cache.dsl.parser import (
    Assignment,
    BoolLit,
    BranchExpr,
    ChainExpr,
    ChainStep,
    ConceptDecl,
    ConfigDecl,
    Expr,
    Ident,
    LookupExpr,
    NumberLit,
    Program,
    RecordExpr,
    SequenceExpr,
    StringLit,
)


# =====================================================================
# Compiled IR
# =====================================================================


@dataclass
class Op:
    """Base class for runtime ops. Subclasses carry their specific fields."""
    source_line: int = 0


@dataclass
class OpConcept(Op):
    """Resolve to the hypervector for a named vocabulary concept."""
    name: str = ""


@dataclass
class OpRef(Op):
    """Resolve to a previously assigned name in the environment."""
    name: str = ""


@dataclass
class OpString(Op):
    value: str = ""


@dataclass
class OpNumber(Op):
    value: float = 0.0


@dataclass
class OpBool(Op):
    value: bool = False


@dataclass
class OpRecord(Op):
    fields: dict[str, "CompiledExpr"] = field(default_factory=dict)


@dataclass
class OpLookup(Op):
    entries: dict[str, "CompiledExpr"] = field(default_factory=dict)


@dataclass
class OpBranch(Op):
    condition_concept: str = ""
    if_true: "CompiledExpr" = None  # type: ignore
    if_false: "CompiledExpr" = None  # type: ignore


@dataclass
class OpSequence(Op):
    items: list["CompiledExpr"] = field(default_factory=list)


@dataclass
class OpChainStep(Op):
    name: str = ""
    fields: dict[str, "CompiledExpr"] = field(default_factory=dict)


@dataclass
class OpChain(Op):
    steps: list[OpChainStep] = field(default_factory=list)


CompiledExpr = Union[
    OpConcept, OpRef, OpString, OpNumber, OpBool,
    OpRecord, OpLookup, OpBranch, OpSequence, OpChain,
]


@dataclass
class CompiledAssignment:
    name: str
    expr: CompiledExpr


@dataclass
class CompiledConfig:
    name: str
    fields: dict[str, CompiledExpr]


@dataclass
class CompiledProgram:
    concepts: list[str] = field(default_factory=list)
    assignments: list[CompiledAssignment] = field(default_factory=list)
    configs: list[CompiledConfig] = field(default_factory=list)
    # Names that were treated as implicit concepts because they weren't
    # declared. Worth surfacing as warnings.
    implicit_concepts: list[str] = field(default_factory=list)


# =====================================================================
# Errors
# =====================================================================


class CompileError(Exception):
    def __init__(self, message: str, line: int = 0):
        self.line = line
        prefix = f"line {line}: {message}" if line else message
        super().__init__(prefix)


# =====================================================================
# Compiler
# =====================================================================


class _Compiler:
    def __init__(self) -> None:
        self.declared_concepts: set[str] = set()
        self.assigned_names: set[str] = set()
        self.implicit_concepts: list[str] = []
        self.config_names: set[str] = set()

    def compile(self, program: Program) -> CompiledProgram:
        out = CompiledProgram()

        # Pass 1: register all explicit concept declarations first so
        # identifier resolution in later passes can distinguish concepts
        # from references. Also register assignment names as they appear.
        for stmt in program.statements:
            if isinstance(stmt, ConceptDecl):
                for n in stmt.names:
                    if n in self.declared_concepts:
                        # Duplicate concept declaration — benign; skip.
                        continue
                    self.declared_concepts.add(n)
                    out.concepts.append(n)

        # Pass 2: compile assignments and configs in source order.
        # Assignments mutate self.assigned_names as they go so later
        # references can resolve correctly.
        for stmt in program.statements:
            if isinstance(stmt, ConceptDecl):
                continue
            if isinstance(stmt, Assignment):
                expr = self._compile_expr(stmt.value)
                out.assignments.append(CompiledAssignment(name=stmt.name, expr=expr))
                self.assigned_names.add(stmt.name)
            elif isinstance(stmt, ConfigDecl):
                if stmt.name in self.config_names:
                    raise CompileError(
                        f"duplicate config '{stmt.name}'", stmt.line
                    )
                self.config_names.add(stmt.name)
                fields = {
                    k: self._compile_expr(v) for k, v in stmt.fields.items()
                }
                out.configs.append(CompiledConfig(name=stmt.name, fields=fields))

        out.implicit_concepts = list(self.implicit_concepts)
        return out

    # --- expressions

    def _compile_expr(self, node: Expr) -> CompiledExpr:
        if isinstance(node, StringLit):
            return OpString(value=node.value, source_line=node.line)
        if isinstance(node, NumberLit):
            return OpNumber(value=node.value, source_line=node.line)
        if isinstance(node, BoolLit):
            return OpBool(value=node.value, source_line=node.line)
        if isinstance(node, Ident):
            return self._resolve_ident(node)
        if isinstance(node, RecordExpr):
            return OpRecord(
                fields={k: self._compile_expr(v) for k, v in node.fields.items()},
                source_line=node.line,
            )
        if isinstance(node, LookupExpr):
            return OpLookup(
                entries={k: self._compile_expr(v) for k, v in node.entries.items()},
                source_line=node.line,
            )
        if isinstance(node, BranchExpr):
            # Condition must be a concept. We treat it as one — declared or implicit.
            cond_name = node.condition.name
            self._ensure_concept(cond_name)
            return OpBranch(
                condition_concept=cond_name,
                if_true=self._compile_expr(node.if_true),
                if_false=self._compile_expr(node.if_false),
                source_line=node.line,
            )
        if isinstance(node, SequenceExpr):
            return OpSequence(
                items=[self._compile_expr(i) for i in node.items],
                source_line=node.line,
            )
        if isinstance(node, ChainExpr):
            steps = [self._compile_chain_step(s) for s in node.steps]
            return OpChain(steps=steps, source_line=node.line)

        raise CompileError(f"unhandled AST node: {type(node).__name__}")

    def _compile_chain_step(self, step: ChainStep) -> OpChainStep:
        if "PREV" in step.fields:
            raise CompileError(
                f"chain step '{step.name}' must not set PREV explicitly; "
                "it is wired automatically from the previous step",
                step.line,
            )
        return OpChainStep(
            name=step.name,
            fields={k: self._compile_expr(v) for k, v in step.fields.items()},
            source_line=step.line,
        )

    def _resolve_ident(self, ident: Ident) -> CompiledExpr:
        name = ident.name
        # Prefer assignment binding — an `x = ...` takes precedence over a
        # concept of the same name within its own scope. This matches how
        # most config languages resolve.
        if name in self.assigned_names:
            return OpRef(name=name, source_line=ident.line)
        if name in self.declared_concepts:
            return OpConcept(name=name, source_line=ident.line)
        # Implicit concept — not declared, not assigned. Treat as a concept
        # and record a warning.
        self._ensure_concept(name)
        return OpConcept(name=name, source_line=ident.line)

    def _ensure_concept(self, name: str) -> None:
        if name in self.declared_concepts or name in self.assigned_names:
            return
        self.declared_concepts.add(name)
        self.implicit_concepts.append(name)


# =====================================================================
# Public API
# =====================================================================


def compile_ast(program: Program) -> CompiledProgram:
    """Compile a parsed DSL program into an executable IR.

    Raises CompileError on semantic errors (duplicate config, chain
    steps setting PREV explicitly, etc.)

    Implicit concepts — identifiers referenced but not declared with
    `concept` — are accepted and recorded in CompiledProgram.implicit_concepts
    so the runtime can surface them as warnings.
    """
    return _Compiler().compile(program)
