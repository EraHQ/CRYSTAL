"""Schema DSL validator (Phase 4.2).

Walks an AST produced by parser.py and emits Diagnostic records for
every shape, type, or legality violation. Does NOT raise on first
error \u2014 the inspector / admin endpoint surfaces all problems at once
so the author can fix them in one pass rather than play whack-a-mole.

Validation layers, in order of reasoning (not execution \u2014 each layer
runs independently):

  1. Crystal-type identity \u2014 id format, prefix-vs-scope coherence,
     uniqueness across the program.
  2. Headers \u2014 known names, value shapes, value ranges, no
     duplicates, required headers present.
  3. Pair-types \u2014 unique names, exactly one prompt_field and one
     answer_field, field name uniqueness within a pair_type, type
     tag from the closed registry, attributes legal for the type tag.
  4. Route hints \u2014 at most one concept_path / one boost per block,
     value ranges.
  5. ACLs \u2014 grant legality given the crystal's scope. The big rules:
       general:  CAN grant `world read`. Cannot grant
                  literal customer_id with `read` (no cross-tenant
                  scoped routing into world-readable types; would
                  contradict the open scope default).
       customer/document/personal:
                  CANNOT grant `world read` (operator-only authority).
                  Cannot grant literal customer_id with `read` (no
                  cross-tenant routable read; isolation guarantee).
                  CAN grant literal customer_id with `read_codebook`
                  (cross-tenant chain target, the explicit cross-org
                  share path).
       owner:    Always legal as a redundant-but-explicit grant; the
                  resolver falls back to scope defaults when no ACL
                  rows exist, so `grant owner read` is legal sugar for
                  declaring the implicit grant explicitly.

Inputs are AST nodes; outputs are Diagnostic records. Callers can
filter by level: errors block compilation; warnings are surfaced
to the author but don't block.

USAGE

    from crystal_cache.dsl.schema import parse
    from crystal_cache.dsl.schema.validator import validate

    program = parse(source)
    diagnostics = validate(program)
    errors = [d for d in diagnostics if d.level == \"error\"]
    if errors:
        # surface to author / admin endpoint
        ...

    # Or, for tests / one-shot tooling that wants pass-or-raise:
    from crystal_cache.dsl.schema.validator import validate_or_raise
    validate_or_raise(program)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from crystal_cache.dsl.schema.field_types import (
    FIELD_TYPE_REGISTRY,
    all_tags,
    is_known_tag,
)
from crystal_cache.dsl.schema.parser import (
    AclDecl,
    AclGrantDecl,
    BoolLit,
    CrystalTypeDecl,
    FieldDecl,
    HeaderField,
    IdentLit,
    NumberLit,
    PairTypeDecl,
    Program,
    RouteWhenDecl,
    StringLit,
)


# =====================================================================
# Diagnostic record
# =====================================================================


# Legal scopes \u2014 must match models.crystal_type.CrystalScope.
VALID_SCOPES = frozenset({"general", "customer", "document", "personal"})

# Legal autosplit policies \u2014 must match models.crystal_type.AutosplitPolicy.
VALID_AUTOSPLIT = frozenset({"split", "refuse"})

# Type-id format. lower-snake slug after the scope prefix.
TYPE_ID_RE = re.compile(r"^(general|customer|document|personal):[a-z0-9_]+$")

# Capacity caps. Soft warning above 50 per Finding 16 (bind-bundle
# routing degrades at FAQ scale beyond ~50 pairs per crystal).
CAPACITY_MIN = 1
CAPACITY_MAX = 1000
CAPACITY_SOFT_WARN = 50

# Threshold range. Cosines are in [-1, 1] but routing thresholds live in
# [0, 1] by convention.
THRESHOLD_MIN = 0.0
THRESHOLD_MAX = 1.0

# Route boost range. Values >0.5 warned: the spec example uses 0.15 and
# the calibrated range we expect from the FAQ benchmarks is much smaller
# than the cosine differences a clear winner produces. A boost of 0.7
# would override most legitimate top-1 picks; that's almost certainly a
# misuse.
BOOST_MIN = 0.0
BOOST_MAX = 1.0
BOOST_SOFT_WARN = 0.5


@dataclass(frozen=True)
class Diagnostic:
    level: str          # 'error' | 'warning'
    message: str
    line: int = 0
    col: int = 0
    crystal_type_id: Optional[str] = None  # context for inspector grouping


class SchemaValidationError(Exception):
    """Raised by validate_or_raise when any diagnostic has level=='error'.

    Carries the full diagnostics list as `diagnostics` so callers that
    catch and want to surface all errors can still get them.
    """

    def __init__(self, diagnostics: list[Diagnostic]):
        self.diagnostics = diagnostics
        errors = [d for d in diagnostics if d.level == "error"]
        msg = f"{len(errors)} validation error(s):\n" + "\n".join(
            f"  line {d.line}: {d.message}" for d in errors
        )
        super().__init__(msg)


# =====================================================================
# Validator entry points
# =====================================================================


def validate(program: Program) -> list[Diagnostic]:
    """Run all validation passes and return diagnostics in source order.

    Never raises on validation failure \u2014 always returns a list. Caller
    decides whether errors block. Use `validate_or_raise` if you want
    raise-on-error semantics.
    """
    v = _Validator()
    v.run(program)
    # Stable ordering: by (line, col, message). Diagnostics carry source
    # positions; sorting makes test assertions deterministic.
    v.diagnostics.sort(key=lambda d: (d.line, d.col, d.message))
    return v.diagnostics


def validate_or_raise(program: Program) -> None:
    """Validate and raise SchemaValidationError if any diagnostic is an error.

    Warnings do not raise. The exception carries all diagnostics, not just
    the first error \u2014 catch and inspect `.diagnostics` to see them all.
    """
    diagnostics = validate(program)
    if any(d.level == "error" for d in diagnostics):
        raise SchemaValidationError(diagnostics)


# =====================================================================
# Internal validator
# =====================================================================


class _Validator:
    def __init__(self) -> None:
        self.diagnostics: list[Diagnostic] = []

    # --- top-level

    def run(self, program: Program) -> None:
        # Cross-program checks first: duplicate type IDs.
        seen_ids: dict[str, CrystalTypeDecl] = {}
        for decl in program.crystal_types:
            if decl.type_id in seen_ids:
                first = seen_ids[decl.type_id]
                self._error(
                    f"duplicate crystal_type id {decl.type_id!r} "
                    f"(first declared at line {first.line})",
                    decl.line, decl.col, decl.type_id,
                )
            else:
                seen_ids[decl.type_id] = decl

        # Per-type checks.
        for decl in program.crystal_types:
            self._validate_crystal_type(decl)

    # --- crystal_type-level

    def _validate_crystal_type(self, decl: CrystalTypeDecl) -> None:
        ctx_id = decl.type_id

        # 1a. ID format.
        m = TYPE_ID_RE.match(decl.type_id)
        id_scope: Optional[str] = None
        if not m:
            self._error(
                f"crystal_type id {decl.type_id!r} doesn't match the required "
                f"format <scope>:<lower_snake_slug> "
                f"(scope must be one of {sorted(VALID_SCOPES)})",
                decl.line, decl.col, ctx_id,
            )
        else:
            id_scope = m.group(1)

        # 2. Headers.
        headers_by_name = self._validate_headers(decl, ctx_id)
        scope_value = headers_by_name.get("scope")

        # 1b. Prefix-vs-scope coherence.
        if id_scope is not None and scope_value is not None:
            if scope_value != id_scope:
                self._error(
                    f"crystal_type id {decl.type_id!r} has scope prefix "
                    f"{id_scope!r} but the `scope` header says {scope_value!r}; "
                    f"these must agree",
                    decl.line, decl.col, ctx_id,
                )

        # Required headers. `scope` is required because it's load-bearing
        # for ACL legality and storage routing. `display_name` is
        # encouraged but optional (some general types may have a
        # short id that's display-quality on its own).
        if "scope" not in headers_by_name:
            self._error(
                f"crystal_type {decl.type_id!r} is missing required header "
                f"'scope' (must be one of {sorted(VALID_SCOPES)})",
                decl.line, decl.col, ctx_id,
            )

        # 3. Pair-types.
        self._validate_pair_types(decl, ctx_id)

        # 4. Route hints.
        if decl.route_when is not None:
            self._validate_route_when(decl.route_when, ctx_id)

        # 5. ACLs.
        if decl.acl is not None:
            # Resolve the effective scope for ACL checks. Prefer the
            # explicit scope header; fall back to the id prefix; if
            # neither is valid we already errored above so skip ACL
            # checks to avoid noise.
            effective_scope = scope_value or id_scope
            if effective_scope in VALID_SCOPES:
                self._validate_acl(decl.acl, effective_scope, ctx_id)

    # --- headers

    def _validate_headers(
        self, decl: CrystalTypeDecl, ctx_id: str
    ) -> dict[str, str]:
        """Validate every header field. Returns a dict of well-typed-and-known
        header values keyed by header name. Values are stringified to a
        canonical form (`scope` -> the scope ident; `capacity` -> str(int);
        thresholds -> str(float)). Callers that need typed values look up
        again through the AST; this dict is for cross-checks like
        scope-vs-id-prefix.
        """
        seen: dict[str, HeaderField] = {}
        valid: dict[str, str] = {}

        for header in decl.headers:
            if header.name in seen:
                first = seen[header.name]
                self._error(
                    f"duplicate header {header.name!r} in crystal_type "
                    f"{decl.type_id!r} (first declared at line {first.line})",
                    header.line, header.col, ctx_id,
                )
                continue
            seen[header.name] = header

            ok, canonical = self._check_header_value(header, ctx_id)
            if ok:
                valid[header.name] = canonical

        return valid

    def _check_header_value(
        self, header: HeaderField, ctx_id: str
    ) -> tuple[bool, str]:
        """Type-check + range-check one header field. Returns (ok, canonical)."""
        name = header.name
        value = header.value

        if name == "display_name":
            if not isinstance(value, StringLit):
                self._error(
                    f"header 'display_name' must be a string literal",
                    header.line, header.col, ctx_id,
                )
                return False, ""
            if not value.value.strip():
                self._error(
                    f"header 'display_name' must not be empty",
                    header.line, header.col, ctx_id,
                )
                return False, ""
            return True, value.value

        if name == "scope":
            if not isinstance(value, IdentLit):
                self._error(
                    f"header 'scope' must be a bare identifier "
                    f"(one of {sorted(VALID_SCOPES)})",
                    header.line, header.col, ctx_id,
                )
                return False, ""
            if value.name not in VALID_SCOPES:
                self._error(
                    f"header 'scope' value {value.name!r} is not a valid scope; "
                    f"must be one of {sorted(VALID_SCOPES)}",
                    header.line, header.col, ctx_id,
                )
                return False, ""
            return True, value.name

        if name == "capacity":
            if not isinstance(value, NumberLit):
                self._error(
                    f"header 'capacity' must be a number",
                    header.line, header.col, ctx_id,
                )
                return False, ""
            n = value.value
            if n != int(n):
                self._error(
                    f"header 'capacity' must be a whole number, got {n}",
                    header.line, header.col, ctx_id,
                )
                return False, ""
            n_int = int(n)
            if n_int < CAPACITY_MIN or n_int > CAPACITY_MAX:
                self._error(
                    f"header 'capacity' must be in "
                    f"[{CAPACITY_MIN}, {CAPACITY_MAX}], got {n_int}",
                    header.line, header.col, ctx_id,
                )
                return False, ""
            if n_int > CAPACITY_SOFT_WARN:
                self._warn(
                    f"header 'capacity' is {n_int}; routing accuracy degrades "
                    f"at >{CAPACITY_SOFT_WARN} pairs per crystal "
                    f"(Finding 16). Auto-split is recommended above this.",
                    header.line, header.col, ctx_id,
                )
            return True, str(n_int)

        if name == "autosplit":
            if not isinstance(value, IdentLit):
                self._error(
                    f"header 'autosplit' must be a bare identifier "
                    f"(one of {sorted(VALID_AUTOSPLIT)})",
                    header.line, header.col, ctx_id,
                )
                return False, ""
            if value.name not in VALID_AUTOSPLIT:
                self._error(
                    f"header 'autosplit' value {value.name!r} is invalid; "
                    f"must be one of {sorted(VALID_AUTOSPLIT)}",
                    header.line, header.col, ctx_id,
                )
                return False, ""
            return True, value.name

        if name in ("routing_threshold", "cleanup_threshold"):
            if not isinstance(value, NumberLit):
                self._error(
                    f"header {name!r} must be a number in "
                    f"[{THRESHOLD_MIN}, {THRESHOLD_MAX}]",
                    header.line, header.col, ctx_id,
                )
                return False, ""
            n = value.value
            if n < THRESHOLD_MIN or n > THRESHOLD_MAX:
                self._error(
                    f"header {name!r} must be in "
                    f"[{THRESHOLD_MIN}, {THRESHOLD_MAX}], got {n}",
                    header.line, header.col, ctx_id,
                )
                return False, ""
            return True, str(n)

        # Should not reach here \u2014 the parser only accepts known header
        # names. If we're here something's drifted.
        self._error(
            f"unrecognized header name {name!r} (parser-validator drift)",
            header.line, header.col, ctx_id,
        )
        return False, ""

    # --- pair_types

    def _validate_pair_types(
        self, decl: CrystalTypeDecl, ctx_id: str
    ) -> None:
        seen_names: dict[str, PairTypeDecl] = {}

        for pt in decl.pair_types:
            if pt.name in seen_names:
                first = seen_names[pt.name]
                self._error(
                    f"duplicate pair_type {pt.name!r} in crystal_type "
                    f"{decl.type_id!r} "
                    f"(first declared at line {first.line})",
                    pt.line, pt.col, ctx_id,
                )
            else:
                seen_names[pt.name] = pt

            self._validate_pair_type_body(pt, ctx_id)

    def _validate_pair_type_body(
        self, pt: PairTypeDecl, ctx_id: str
    ) -> None:
        """Validate the fields inside a single pair_type."""
        # Exactly one prompt_field and one answer_field.
        prompt_fields = [f for f in pt.fields if f.role == "prompt_field"]
        answer_fields = [f for f in pt.fields if f.role == "answer_field"]

        if len(prompt_fields) == 0:
            self._error(
                f"pair_type {pt.name!r} is missing a prompt_field",
                pt.line, pt.col, ctx_id,
            )
        elif len(prompt_fields) > 1:
            self._error(
                f"pair_type {pt.name!r} has {len(prompt_fields)} prompt_fields; "
                f"exactly one is required",
                prompt_fields[1].line, prompt_fields[1].col, ctx_id,
            )

        if len(answer_fields) == 0:
            self._error(
                f"pair_type {pt.name!r} is missing an answer_field",
                pt.line, pt.col, ctx_id,
            )
        elif len(answer_fields) > 1:
            self._error(
                f"pair_type {pt.name!r} has {len(answer_fields)} answer_fields; "
                f"exactly one is required",
                answer_fields[1].line, answer_fields[1].col, ctx_id,
            )

        # Field-name uniqueness. (e.g. you can't have two
        # `prompt_field "foo"` even of different roles.)
        seen_field_names: dict[str, FieldDecl] = {}
        for f in pt.fields:
            if f.field_name in seen_field_names:
                first = seen_field_names[f.field_name]
                self._error(
                    f"duplicate field name {f.field_name!r} in pair_type "
                    f"{pt.name!r} (first declared at line {first.line})",
                    f.line, f.col, ctx_id,
                )
            else:
                seen_field_names[f.field_name] = f

            self._validate_field(f, pt.name, ctx_id)

    def _validate_field(
        self, f: FieldDecl, pair_type_name: str, ctx_id: str
    ) -> None:
        """Validate one prompt_field/answer_field declaration."""
        if not is_known_tag(f.type_tag):
            self._error(
                f"unknown field type tag {f.type_tag!r} on "
                f"{f.role} {f.field_name!r} in pair_type "
                f"{pair_type_name!r}; valid tags: {all_tags()}",
                f.line, f.col, ctx_id,
            )
            return

        spec = FIELD_TYPE_REGISTRY[f.type_tag]
        seen_attrs: dict[str, int] = {}  # name -> line for first occurrence

        for attr in f.attrs:
            if attr.name in seen_attrs:
                self._error(
                    f"duplicate attribute {attr.name!r} on field "
                    f"{f.field_name!r} in pair_type {pair_type_name!r} "
                    f"(first declared at line {seen_attrs[attr.name]})",
                    attr.line, attr.col, ctx_id,
                )
                continue
            seen_attrs[attr.name] = attr.line

            if attr.name not in spec.allowed_attrs:
                allowed = sorted(spec.allowed_attrs)
                hint = (
                    f"allowed attributes for type {f.type_tag!r}: "
                    f"{allowed}" if allowed
                    else f"type {f.type_tag!r} accepts no attributes"
                )
                self._error(
                    f"unknown attribute {attr.name!r} on field "
                    f"{f.field_name!r}; {hint}",
                    attr.line, attr.col, ctx_id,
                )
                continue

            # Type-check the attribute value.
            expected = spec.attr_value_kinds.get(attr.name)
            if expected == "string" and not isinstance(attr.value, StringLit):
                self._error(
                    f"attribute {attr.name!r} on field {f.field_name!r} "
                    f"must be a string",
                    attr.line, attr.col, ctx_id,
                )
            elif expected == "number" and not isinstance(attr.value, NumberLit):
                self._error(
                    f"attribute {attr.name!r} on field {f.field_name!r} "
                    f"must be a number",
                    attr.line, attr.col, ctx_id,
                )
            elif expected == "bool" and not isinstance(attr.value, BoolLit):
                self._error(
                    f"attribute {attr.name!r} on field {f.field_name!r} "
                    f"must be a boolean",
                    attr.line, attr.col, ctx_id,
                )

    # --- route_when

    def _validate_route_when(
        self, rw: RouteWhenDecl, ctx_id: str
    ) -> None:
        concept_paths = [c for c in rw.clauses if c.kind == "concept_path"]
        boosts = [c for c in rw.clauses if c.kind == "boost"]

        if len(concept_paths) > 1:
            self._error(
                f"route_when block has {len(concept_paths)} concept_path "
                f"clauses; at most one is allowed",
                concept_paths[1].line, concept_paths[1].col, ctx_id,
            )

        if len(boosts) > 1:
            self._error(
                f"route_when block has {len(boosts)} boost clauses; "
                f"at most one is allowed",
                boosts[1].line, boosts[1].col, ctx_id,
            )

        for c in concept_paths:
            assert isinstance(c.value, StringLit)
            if not c.value.value.strip():
                self._error(
                    f"concept_path must not be empty",
                    c.line, c.col, ctx_id,
                )

        for c in boosts:
            assert isinstance(c.value, NumberLit)
            n = c.value.value
            if n < BOOST_MIN or n > BOOST_MAX:
                self._error(
                    f"boost must be in [{BOOST_MIN}, {BOOST_MAX}], got {n}",
                    c.line, c.col, ctx_id,
                )
            elif n > BOOST_SOFT_WARN:
                self._warn(
                    f"boost {n} is unusually high; routing-cosine differences "
                    f"between clear top-1 winners are typically <0.2. Consider "
                    f"a smaller value (the spec example uses 0.15).",
                    c.line, c.col, ctx_id,
                )

    # --- acl

    def _validate_acl(
        self, acl: AclDecl, scope: str, ctx_id: str
    ) -> None:
        # Track exact-duplicate grants \u2014 same principal + grant_kind \u2014
        # and warn (legal but redundant).
        seen_grants: dict[tuple, AclGrantDecl] = {}

        for grant in acl.grants:
            self._validate_acl_grant(grant, scope, ctx_id)

            key = (grant.principal_kind, grant.principal_id, grant.grant_kind)
            if key in seen_grants:
                first = seen_grants[key]
                self._warn(
                    f"redundant grant: same principal and grant kind "
                    f"already granted at line {first.line}",
                    grant.line, grant.col, ctx_id,
                )
            else:
                seen_grants[key] = grant

    def _validate_acl_grant(
        self, grant: AclGrantDecl, scope: str, ctx_id: str
    ) -> None:
        # `owner read` is always legal as an explicit redundant declaration.
        # `owner read_codebook` is legal too \u2014 means the owner can chain-extend
        # from this crystal's codebook. (Unusual but consistent.)
        if grant.principal_kind == "owner":
            return

        # `world` grants only legal for general scope.
        if grant.principal_kind == "world":
            if scope != "general":
                self._error(
                    f"grant 'world {grant.grant_kind}' is only legal on "
                    f"general-scope crystal types; this is scope={scope!r}",
                    grant.line, grant.col, ctx_id,
                )
            return

        # Literal customer_id principal.
        if grant.principal_kind == "literal":
            if grant.grant_kind == "read":
                # Cross-tenant routable read is the isolation-violating
                # case. Always blocked.
                self._error(
                    f"grant {grant.principal_id!r} read is not legal: "
                    f"cross-tenant routable reads break tenant isolation. "
                    f"For cross-tenant chain target use, grant "
                    f"'read_codebook' instead.",
                    grant.line, grant.col, ctx_id,
                )
                return
            # 'read_codebook' on a literal customer_id is the chain-target
            # path. Legal for all scopes.
            return

        # Should not reach \u2014 parser only emits the three principal kinds.
        self._error(
            f"unrecognized principal kind {grant.principal_kind!r} "
            f"(parser-validator drift)",
            grant.line, grant.col, ctx_id,
        )

    # --- helpers

    def _error(
        self, msg: str, line: int, col: int, ctx_id: Optional[str] = None
    ) -> None:
        self.diagnostics.append(
            Diagnostic(
                level="error",
                message=msg,
                line=line,
                col=col,
                crystal_type_id=ctx_id,
            )
        )

    def _warn(
        self, msg: str, line: int, col: int, ctx_id: Optional[str] = None
    ) -> None:
        self.diagnostics.append(
            Diagnostic(
                level="warning",
                message=msg,
                line=line,
                col=col,
                crystal_type_id=ctx_id,
            )
        )
