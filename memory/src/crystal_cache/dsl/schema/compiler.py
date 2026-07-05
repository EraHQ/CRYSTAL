"""Schema DSL compiler (Phase 4.3).

Walks a validated AST and produces a CompiledCrystalType (and a
CompiledProgram for multi-type sources) that downstream runtime hooks
consume.

CONTRACT WITH validator.py
--------------------------
The compiler ASSUMES the AST has been validated. It does not re-check
known-by-validator invariants (id format, header shapes, ACL legality,
etc.). If the caller hands it un-validated AST that breaks an invariant,
the compiler may crash with KeyError, AssertionError, or produce
nonsense. The intended call sequence is always:

    program = parse(source)
    diagnostics = validate(program)
    if any(d.level == "error" for d in diagnostics):
        # surface errors, don't compile
        ...
    compiled = compile_program(program)

For tests / one-shot tooling, `compile_or_raise(source)` does parse +
validate + compile in one call.

OUTPUT SHAPE
------------
The compiled IR is plain dataclasses (frozen for hashability where it
matters). Consumers index into them by name:

    compiled.pair_types["error_to_fix"].prompt_field.field_name
    compiled.route_hint.boost
    compiled.acl_defaults

Consumers DO NOT walk the AST themselves. The compiler is the single
boundary between "DSL author intent" and "runtime."

OWNER RESOLUTION
----------------
ACL grants of `owner` are NOT resolved at compile time \u2014 we don't know
the owning customer_id until a crystal of this type is being created.
The compiler emits `CompiledAclGrant(principal_kind="owner", ...)` as
a sentinel. The ACL instantiator (Phase 4.7) translates the sentinel
to a concrete (customer, customer_id) row at crystal creation.

This design keeps the compiled object reusable across customers \u2014 one
compiled `customer:medical_records` works for every customer that
authors a crystal of that type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from crystal_cache.dsl.schema.parser import (
    AclGrantDecl,
    BoolLit,
    CrystalTypeDecl,
    FieldDecl,
    NumberLit,
    PairTypeDecl,
    Program,
    RouteWhenDecl,
    StringLit,
)
from crystal_cache.dsl.schema.validator import (
    SchemaValidationError,
    validate_or_raise,
)


# ---------------------------------------------------------------------------
# Compiled IR
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledFieldAttr:
    """One attribute on a field, with its value already coerced to a Python
    primitive (str, float, int, bool) per its kind in the field-type registry.

    The validator already type-checked attribute values. By the time we
    reach the compiler, attr_value carries a real Python value, not a Lit
    AST node.
    """
    name: str
    value: object  # str | float | int | bool


@dataclass(frozen=True)
class CompiledField:
    """One prompt_field or answer_field, fully resolved."""
    role: str           # 'prompt_field' | 'answer_field'
    field_name: str
    type_tag: str
    attrs: tuple[CompiledFieldAttr, ...] = field(default_factory=tuple)

    def attr(self, name: str) -> Optional[CompiledFieldAttr]:
        """Look up an attribute by name; None if not present."""
        for a in self.attrs:
            if a.name == name:
                return a
        return None


@dataclass(frozen=True)
class CompiledPairType:
    """One pair_type with both its fields resolved.

    The validator guarantees exactly one prompt_field and one answer_field
    \u2014 the compiler relies on that and surfaces them as named attributes
    rather than a list, so downstream code doesn't have to scan.
    """
    name: str
    prompt_field: CompiledField
    answer_field: CompiledField


@dataclass(frozen=True)
class CompiledRouteHint:
    """One route_when block, flattened.

    concept_path may be None if the author wrote a `route_when` block
    with only a boost. boost may be None if the block had only a
    concept_path. (At least one is required by the validator's
    'route_when must have at least one clause' rule \u2014 see validator
    note about empty blocks.)
    """
    concept_path: Optional[str]
    boost: Optional[float]


@dataclass(frozen=True)
class CompiledAclGrant:
    """One ACL grant default for crystals of this type.

    principal_kind values:
      - 'world'   : maps to (global, 'world') in crystal_acls.
      - 'owner'   : sentinel; the ACL instantiator resolves to
                    (customer, owning_customer_id) when a crystal is
                    created. principal_id is None on this row.
      - 'literal' : principal_id holds the explicit customer_id string;
                    maps to (customer, principal_id) in crystal_acls.

    grant_kind values:
      - 'read'          : route INTO + consume facts.
      - 'read_codebook' : chain-extend cleanup codebook only.
    """
    principal_kind: str   # 'world' | 'owner' | 'literal'
    principal_id: Optional[str]
    grant_kind: str       # 'read' | 'read_codebook'


@dataclass(frozen=True)
class CompiledCrystalType:
    """The full compiled object for one crystal_type.

    This is what the loader caches and what downstream runtime hooks
    consume. All fields are populated; defaults from the spec apply
    when the DSL omits a header.
    """
    type_id: str
    display_name: str
    scope: str

    # Default values match models.crystal_type.CrystalType. The compiler
    # applies them when the DSL omits the corresponding header so
    # downstream code can rely on them being present.
    capacity_default: int = 50
    autosplit_policy: str = "split"
    routing_threshold: Optional[float] = None
    cleanup_threshold: Optional[float] = None

    # Pair-types keyed by name for O(1) lookup at write time.
    pair_types: dict[str, CompiledPairType] = field(default_factory=dict)

    # At most one route hint per type.
    route_hint: Optional[CompiledRouteHint] = None

    # ACL defaults applied to every new crystal of this type.
    acl_defaults: tuple[CompiledAclGrant, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CompiledProgram:
    """A whole DSL source file, compiled.

    Multi-type sources (the common case for general-tier seeding) compile
    to a CompiledProgram whose `crystal_types` dict maps type_id ->
    CompiledCrystalType. Single-type sources still wrap one entry.
    """
    crystal_types: dict[str, CompiledCrystalType]


# ---------------------------------------------------------------------------
# Compiler errors
# ---------------------------------------------------------------------------


class SchemaCompileError(Exception):
    """Raised on compiler-time invariant violations.

    These should not happen in practice if the AST has been validated.
    If you see one in production, the validator missed a check.
    """


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------


def compile_program(program: Program) -> CompiledProgram:
    """Compile a validated AST into runtime artifacts.

    The caller is responsible for having run `validate` first; this
    function trusts the AST. See module docstring for the contract.

    Returns a CompiledProgram even if the source has only one
    crystal_type \u2014 keeps the API uniform.
    """
    compiled_types: dict[str, CompiledCrystalType] = {}
    for decl in program.crystal_types:
        compiled = _compile_crystal_type(decl)
        if compiled.type_id in compiled_types:
            # Duplicate IDs should have been caught by the validator. If
            # we see one here it's a validator bug; surface it loudly.
            raise SchemaCompileError(
                f"duplicate crystal_type id {compiled.type_id!r} reached the "
                f"compiler; validator should have rejected this"
            )
        compiled_types[compiled.type_id] = compiled
    return CompiledProgram(crystal_types=compiled_types)


def compile_or_raise(source: str) -> CompiledProgram:
    """Convenience: parse + validate + compile a source string in one call.

    Raises SchemaParseError on syntax errors, SchemaValidationError on
    semantic errors, SchemaCompileError on compiler-time invariant
    violations. Tests and one-shot tools use this; the production
    loader uses parse / validate / compile_program separately so it
    can surface diagnostics without raising.
    """
    from crystal_cache.dsl.schema.parser import parse  # local import: avoids cycle

    program = parse(source)
    validate_or_raise(program)
    return compile_program(program)


# ---------------------------------------------------------------------------
# Per-declaration compilers
# ---------------------------------------------------------------------------


def _compile_crystal_type(decl: CrystalTypeDecl) -> CompiledCrystalType:
    # Pull header values into a dict keyed by name. The validator has
    # already type-checked these; we just unwrap the Lit nodes here.
    headers = {h.name: h.value for h in decl.headers}

    # Required-by-validator: scope. Pull it first so we can derive the
    # display_name default from the type_id if needed.
    scope_lit = headers.get("scope")
    if scope_lit is None:
        raise SchemaCompileError(
            f"crystal_type {decl.type_id!r} missing required 'scope' header "
            f"(validator should have caught this)"
        )
    # IdentLit.name carries the scope value.
    scope = scope_lit.name  # type: ignore[attr-defined]

    display_name_lit = headers.get("display_name")
    if display_name_lit is None:
        # Default: derive a human-readable name from the type_id slug
        # part. 'general:python_debugging' -> 'Python Debugging'.
        slug = decl.type_id.split(":", 1)[1] if ":" in decl.type_id else decl.type_id
        display_name = slug.replace("_", " ").title()
    else:
        assert isinstance(display_name_lit, StringLit)
        display_name = display_name_lit.value

    # Numeric headers, with defaults matching the Crystal model.
    capacity_default = 50
    capacity_lit = headers.get("capacity")
    if capacity_lit is not None:
        assert isinstance(capacity_lit, NumberLit)
        capacity_default = int(capacity_lit.value)

    autosplit_policy = "split"
    autosplit_lit = headers.get("autosplit")
    if autosplit_lit is not None:
        autosplit_policy = autosplit_lit.name  # type: ignore[attr-defined]

    routing_threshold: Optional[float] = None
    rt_lit = headers.get("routing_threshold")
    if rt_lit is not None:
        assert isinstance(rt_lit, NumberLit)
        routing_threshold = float(rt_lit.value)

    cleanup_threshold: Optional[float] = None
    ct_lit = headers.get("cleanup_threshold")
    if ct_lit is not None:
        assert isinstance(ct_lit, NumberLit)
        cleanup_threshold = float(ct_lit.value)

    # Pair-types.
    pair_types: dict[str, CompiledPairType] = {}
    for pt in decl.pair_types:
        compiled_pt = _compile_pair_type(pt)
        pair_types[compiled_pt.name] = compiled_pt

    # Route hint.
    route_hint: Optional[CompiledRouteHint] = None
    if decl.route_when is not None:
        route_hint = _compile_route_when(decl.route_when)

    # ACL defaults. If the DSL omits the acl block, fall back to the
    # scope-default grant the resolver assumes anyway:
    #   - general -> grant world read
    #   - customer / document / personal -> grant owner read
    # Materializing these as compiled grants makes downstream behavior
    # explicit instead of relying on the resolver's implicit fallback.
    acl_defaults: tuple[CompiledAclGrant, ...]
    if decl.acl is not None:
        acl_defaults = tuple(_compile_acl_grant(g) for g in decl.acl.grants)
    else:
        acl_defaults = _scope_default_acl(scope)

    return CompiledCrystalType(
        type_id=decl.type_id,
        display_name=display_name,
        scope=scope,
        capacity_default=capacity_default,
        autosplit_policy=autosplit_policy,
        routing_threshold=routing_threshold,
        cleanup_threshold=cleanup_threshold,
        pair_types=pair_types,
        route_hint=route_hint,
        acl_defaults=acl_defaults,
    )


def _compile_pair_type(pt: PairTypeDecl) -> CompiledPairType:
    prompt_decl: Optional[FieldDecl] = None
    answer_decl: Optional[FieldDecl] = None
    for f in pt.fields:
        if f.role == "prompt_field":
            prompt_decl = f
        elif f.role == "answer_field":
            answer_decl = f

    if prompt_decl is None or answer_decl is None:
        # Validator should have caught the missing-field case. If we get
        # here the AST was not validated.
        raise SchemaCompileError(
            f"pair_type {pt.name!r} missing prompt_field or answer_field "
            f"(validator should have caught this)"
        )

    return CompiledPairType(
        name=pt.name,
        prompt_field=_compile_field(prompt_decl),
        answer_field=_compile_field(answer_decl),
    )


def _compile_field(f: FieldDecl) -> CompiledField:
    attrs = tuple(
        CompiledFieldAttr(
            name=a.name,
            value=_unwrap_literal(a.value),
        )
        for a in f.attrs
    )
    return CompiledField(
        role=f.role,
        field_name=f.field_name,
        type_tag=f.type_tag,
        attrs=attrs,
    )


def _unwrap_literal(lit: object) -> object:
    """Pull a Python primitive out of a parser literal node.

    The validator has already type-checked attribute values against
    the field-type registry; here we just unwrap the Lit wrapper.
    Number literals collapse int-valued floats to int so downstream
    code doesn't have to special-case them.
    """
    if isinstance(lit, StringLit):
        return lit.value
    if isinstance(lit, NumberLit):
        v = lit.value
        if v == int(v):
            return int(v)
        return v
    if isinstance(lit, BoolLit):
        return lit.value
    raise SchemaCompileError(
        f"can't unwrap literal of type {type(lit).__name__}"
    )


def _compile_route_when(rw: RouteWhenDecl) -> CompiledRouteHint:
    concept_path: Optional[str] = None
    boost: Optional[float] = None
    for clause in rw.clauses:
        if clause.kind == "concept_path":
            assert isinstance(clause.value, StringLit)
            concept_path = clause.value.value
        elif clause.kind == "boost":
            assert isinstance(clause.value, NumberLit)
            boost = float(clause.value.value)
    return CompiledRouteHint(concept_path=concept_path, boost=boost)


def _compile_acl_grant(g: AclGrantDecl) -> CompiledAclGrant:
    return CompiledAclGrant(
        principal_kind=g.principal_kind,
        principal_id=g.principal_id,
        grant_kind=g.grant_kind,
    )


def _scope_default_acl(scope: str) -> tuple[CompiledAclGrant, ...]:
    """The implicit ACL the resolver assumes when no explicit grants
    are declared. We materialize these as compiled grants so callers
    don't have to know about implicit fallbacks.

    - general:  world read
    - customer: owner read
    - document: owner read   (inherits from owning customer)
    - personal: owner read   (Phase 5+ scope; basic default for now)
    """
    if scope == "general":
        return (
            CompiledAclGrant(
                principal_kind="world",
                principal_id=None,
                grant_kind="read",
            ),
        )
    return (
        CompiledAclGrant(
            principal_kind="owner",
            principal_id=None,
            grant_kind="read",
        ),
    )
