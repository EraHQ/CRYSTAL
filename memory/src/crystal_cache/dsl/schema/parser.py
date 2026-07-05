"""SchemaDSL parser — v0, Phase 4.

Parses Schema DSL source text into an Abstract Syntax Tree (AST). The
AST is a tree of dataclasses, consumed by validator.py and compiler.py.

The Schema DSL is a SEPARATE LAYER from the existing concept DSL in
`src/crystal_cache/dsl/{parser,compiler,runtime}.py`. The two are not
versions of each other; they declare different things at different
layers:

  - Concept DSL declares concepts and configs that get encoded as
    hypervectors for M1 concept-routing.
  - Schema DSL declares crystal_types, their valid pair-types, their
    routing hints, their ACLs.

A `route_when.concept_path` reference in the Schema DSL points at a
concept the Concept DSL has declared — that's the integration seam.

GRAMMAR (informal, v0)

    program           := crystal_type_decl+

    crystal_type_decl := 'crystal_type' STRING '{' type_body '}'

    type_body         := ( header_field
                         | pair_type_decl
                         | route_when_decl
                         | acl_decl
                         )*

    header_field      := HEADER_KW literal_or_ident ';'?
    HEADER_KW         := 'display_name' | 'scope' | 'capacity'
                       | 'autosplit'    | 'routing_threshold'
                       | 'cleanup_threshold'

    pair_type_decl    := 'pair_type' STRING '{' field_decl+ '}'

    field_decl        := role_kw STRING type_tag attr* ';'?
    role_kw           := 'prompt_field' | 'answer_field'
    type_tag          := IDENT
    attr              := IDENT '=' literal

    route_when_decl   := 'route_when' '{' route_clause+ '}'
    route_clause      := 'concept_path' STRING ';'?
                       | 'boost' NUMBER ';'?

    acl_decl          := 'acl' '{' acl_grant+ '}'
    acl_grant         := 'grant' principal grant_kind ';'?
    principal         := 'world' | 'owner' | STRING
    grant_kind        := 'read' | 'read_codebook'

    literal           := STRING | NUMBER | BOOL
    literal_or_ident  := literal | IDENT

COMMENTS: '#' to end of line.

DESIGN NOTES

  - crystal_type and pair_type IDs are STRING literals (not identifiers)
    because real IDs contain colons: 'general:python_debugging',
    'customer:medical_records'. Bare identifiers can't carry colons
    without complicating tokenization for no benefit.

  - Field type tags ('text', 'code', 'date', etc.) are bare IDENTs
    drawn from a closed registry (see field_types.py). Not
    user-extensible at v0; if a customer needs a new type tag, it
    requires a code change.

  - Field attributes (e.g. `prompt_field "code" code language="python"`)
    let pair-types specialize beyond the bare type tag without
    inflating the type tag count.

  - 'owner' is a principal sugar that resolves at compile-time to the
    crystal's owning customer_id. Lets general-tier and customer-tier
    ACL syntax stay symmetrical.

The grammar is deliberately minimal. We add syntax when real configs
demand it — not speculatively.

STATUS: v0 of the Schema DSL. Syntax may change in v1 as we author
real configs and find ergonomic gaps.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Union


# =====================================================================
# AST nodes
# =====================================================================


@dataclass
class Node:
    line: int = 0
    col: int = 0


@dataclass
class StringLit(Node):
    value: str = ""


@dataclass
class NumberLit(Node):
    value: float = 0.0


@dataclass
class BoolLit(Node):
    value: bool = False


@dataclass
class IdentLit(Node):
    """Bare identifier used as a value (e.g. `scope general` or a type tag)."""
    name: str = ""


# Literals that can appear on the right-hand side of a header field
# or as an attribute value. NumberLit covers ints and floats uniformly.
Literal = Union[StringLit, NumberLit, BoolLit]
LiteralOrIdent = Union[Literal, IdentLit]


@dataclass
class FieldAttr(Node):
    """A `name=value` attribute on a field declaration.

    Examples: `language="python"`, `format="iso8601"`, `max_length=4096`.
    """
    name: str = ""
    value: Literal = None  # type: ignore[assignment]


@dataclass
class FieldDecl(Node):
    """One `prompt_field` or `answer_field` line inside a pair_type."""
    role: str = ""           # 'prompt_field' | 'answer_field'
    field_name: str = ""     # e.g. 'error_message', 'fix'
    type_tag: str = ""       # e.g. 'text', 'code', 'date'
    attrs: list[FieldAttr] = field(default_factory=list)


@dataclass
class PairTypeDecl(Node):
    name: str = ""           # e.g. 'error_to_fix'
    fields: list[FieldDecl] = field(default_factory=list)


@dataclass
class HeaderField(Node):
    """One header-level setting on a crystal_type.

    name in {display_name, scope, capacity, autosplit,
             routing_threshold, cleanup_threshold}.
    """
    name: str = ""
    value: LiteralOrIdent = None  # type: ignore[assignment]


@dataclass
class RouteClause(Node):
    """One clause inside a route_when block.

    kind == 'concept_path' -> value: StringLit
    kind == 'boost'        -> value: NumberLit
    """
    kind: str = ""
    value: Literal = None  # type: ignore[assignment]


@dataclass
class RouteWhenDecl(Node):
    clauses: list[RouteClause] = field(default_factory=list)


@dataclass
class AclGrantDecl(Node):
    """One `grant <principal> <grant_kind>` line inside an acl block.

    principal_kind == 'world'   -> principal_id is None
    principal_kind == 'owner'   -> principal_id is None
    principal_kind == 'literal' -> principal_id is the string customer_id
    """
    principal_kind: str = ""
    principal_id: Optional[str] = None
    grant_kind: str = ""        # 'read' | 'read_codebook'


@dataclass
class AclDecl(Node):
    grants: list[AclGrantDecl] = field(default_factory=list)


@dataclass
class CrystalTypeDecl(Node):
    type_id: str = ""           # e.g. 'general:python_debugging'
    headers: list[HeaderField] = field(default_factory=list)
    pair_types: list[PairTypeDecl] = field(default_factory=list)
    route_when: Optional[RouteWhenDecl] = None
    acl: Optional[AclDecl] = None


@dataclass
class Program(Node):
    crystal_types: list[CrystalTypeDecl] = field(default_factory=list)


# =====================================================================
# Errors
# =====================================================================


class SchemaParseError(Exception):
    """Raised on Schema DSL syntax errors. Carries line/col + a caret."""

    def __init__(self, message: str, line: int, col: int, source_line: str = ""):
        self.line = line
        self.col = col
        self.source_line = source_line
        prefix = f"line {line}, col {col}: {message}"
        if source_line:
            caret = " " * (col - 1) + "^"
            prefix = f"{prefix}\n    {source_line}\n    {caret}"
        super().__init__(prefix)


# =====================================================================
# Tokenizer
# =====================================================================


# Keywords. IDENTs matching these are tokenized as keywords, not IDENT.
# Header keywords (display_name, scope, capacity, autosplit,
# routing_threshold, cleanup_threshold) are NOT in this set — they're
# parsed as IDENTs and matched contextually so we don't have to extend
# the keyword table every time we add a header field. Only structural
# keywords get reserved.
KEYWORDS = {
    "crystal_type",
    "pair_type",
    "prompt_field",
    "answer_field",
    "route_when",
    "concept_path",
    "boost",
    "acl",
    "grant",
    "world",
    "owner",
    "read",
    "read_codebook",
    "true",
    "false",
}


HEADER_FIELD_NAMES = frozenset({
    "display_name",
    "scope",
    "capacity",
    "autosplit",
    "routing_threshold",
    "cleanup_threshold",
})


@dataclass
class Token:
    kind: str
    value: str
    line: int
    col: int


# Token kinds. Listed in priority order; longer/more specific first.
_TOKEN_SPECS = [
    ("COMMENT", r"\#[^\n]*"),
    ("WS",      r"[ \t\r]+"),
    ("NEWLINE", r"\n"),
    ("STRING",  r'"(?:[^"\\]|\\.)*"'),
    ("NUMBER",  r"-?\d+(?:\.\d+)?"),
    ("LBRACE",  r"\{"),
    ("RBRACE",  r"\}"),
    ("EQ",      r"="),
    ("SEMI",    r";"),
    ("IDENT",   r"[A-Za-z_][A-Za-z0-9_]*"),
]

_TOKEN_RE = re.compile("|".join(f"(?P<{k}>{p})" for k, p in _TOKEN_SPECS))


def tokenize(source: str) -> list[Token]:
    """Tokenize Schema DSL source. Raises SchemaParseError on lexical errors."""
    tokens: list[Token] = []
    line = 1
    line_start = 0
    pos = 0
    length = len(source)
    while pos < length:
        m = _TOKEN_RE.match(source, pos)
        if not m:
            col = pos - line_start + 1
            raise SchemaParseError(
                f"unexpected character {source[pos]!r}",
                line,
                col,
                _extract_source_line(source, line_start),
            )
        kind = m.lastgroup
        raw = m.group()
        col = pos - line_start + 1

        if kind in ("WS", "COMMENT"):
            pos = m.end()
            continue
        if kind == "NEWLINE":
            line += 1
            pos = m.end()
            line_start = pos
            continue

        value = raw
        if kind == "STRING":
            # Strip surrounding quotes and decode standard backslash escapes.
            value = bytes(raw[1:-1], "utf-8").decode("unicode_escape")
        elif kind == "IDENT" and raw in KEYWORDS:
            kind = "KW_" + raw.upper()

        tokens.append(Token(kind=kind, value=value, line=line, col=col))
        pos = m.end()

    tokens.append(Token(kind="EOF", value="", line=line, col=pos - line_start + 1))
    return tokens


def _extract_source_line(source: str, line_start: int) -> str:
    end = source.find("\n", line_start)
    if end == -1:
        end = len(source)
    return source[line_start:end]


# =====================================================================
# Parser
# =====================================================================


class _Parser:
    def __init__(self, tokens: list[Token], source: str) -> None:
        self.tokens = tokens
        self.pos = 0
        self.source = source
        self._source_lines = source.split("\n")

    # --- token helpers

    def peek(self, offset: int = 0) -> Token:
        return self.tokens[self.pos + offset]

    def advance(self) -> Token:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def check(self, kind: str) -> bool:
        return self.peek().kind == kind

    def match(self, *kinds: str) -> Optional[Token]:
        if self.peek().kind in kinds:
            return self.advance()
        return None

    def expect(self, kind: str, context: str = "") -> Token:
        tok = self.peek()
        if tok.kind != kind:
            msg = f"expected {kind}, got {tok.kind}"
            if context:
                msg += f" ({context})"
            raise self._error(msg, tok)
        return self.advance()

    def _error(self, msg: str, tok: Token) -> SchemaParseError:
        src_line = ""
        if 1 <= tok.line <= len(self._source_lines):
            src_line = self._source_lines[tok.line - 1]
        return SchemaParseError(msg, tok.line, tok.col, src_line)

    # --- grammar

    def parse_program(self) -> Program:
        prog = Program()
        while not self.check("EOF"):
            prog.crystal_types.append(self._parse_crystal_type_decl())
        return prog

    def _parse_crystal_type_decl(self) -> CrystalTypeDecl:
        kw = self.expect(
            "KW_CRYSTAL_TYPE",
            "expected 'crystal_type' at top level",
        )
        id_tok = self.expect("STRING", "crystal_type id (must be a string literal)")
        self.expect("LBRACE", f"crystal_type {id_tok.value!r} body")

        decl = CrystalTypeDecl(
            type_id=id_tok.value,
            line=kw.line,
            col=kw.col,
        )

        while not self.check("RBRACE") and not self.check("EOF"):
            tok = self.peek()
            if tok.kind == "KW_PAIR_TYPE":
                decl.pair_types.append(self._parse_pair_type_decl())
            elif tok.kind == "KW_ROUTE_WHEN":
                if decl.route_when is not None:
                    raise self._error(
                        f"duplicate route_when block in crystal_type "
                        f"{id_tok.value!r}",
                        tok,
                    )
                decl.route_when = self._parse_route_when_decl()
            elif tok.kind == "KW_ACL":
                if decl.acl is not None:
                    raise self._error(
                        f"duplicate acl block in crystal_type "
                        f"{id_tok.value!r}",
                        tok,
                    )
                decl.acl = self._parse_acl_decl()
            elif tok.kind == "IDENT" and tok.value in HEADER_FIELD_NAMES:
                decl.headers.append(self._parse_header_field())
            elif tok.kind == "IDENT":
                raise self._error(
                    f"unknown header field '{tok.value}' in crystal_type "
                    f"{id_tok.value!r}; "
                    f"valid headers: {sorted(HEADER_FIELD_NAMES)}",
                    tok,
                )
            else:
                raise self._error(
                    f"unexpected {tok.kind} in crystal_type body; "
                    f"expected pair_type / route_when / acl / a header field",
                    tok,
                )

        self.expect(
            "RBRACE",
            f"end of crystal_type {id_tok.value!r} body",
        )
        return decl

    def _parse_header_field(self) -> HeaderField:
        ident = self.expect("IDENT", "header field name")
        value = self._parse_literal_or_ident(
            context=f"value for header field '{ident.value}'",
        )
        self.match("SEMI")
        return HeaderField(
            name=ident.value, value=value, line=ident.line, col=ident.col
        )

    def _parse_pair_type_decl(self) -> PairTypeDecl:
        kw = self.advance()  # 'pair_type'
        name_tok = self.expect("STRING", "pair_type name (must be a string)")
        self.expect("LBRACE", f"pair_type {name_tok.value!r} body")

        decl = PairTypeDecl(name=name_tok.value, line=kw.line, col=kw.col)

        while not self.check("RBRACE") and not self.check("EOF"):
            tok = self.peek()
            if tok.kind in ("KW_PROMPT_FIELD", "KW_ANSWER_FIELD"):
                decl.fields.append(self._parse_field_decl())
            else:
                raise self._error(
                    f"unexpected {tok.kind} in pair_type body; "
                    f"expected prompt_field or answer_field",
                    tok,
                )

        self.expect("RBRACE", f"end of pair_type {name_tok.value!r}")
        return decl

    def _parse_field_decl(self) -> FieldDecl:
        role_tok = self.advance()  # KW_PROMPT_FIELD or KW_ANSWER_FIELD
        role = "prompt_field" if role_tok.kind == "KW_PROMPT_FIELD" else "answer_field"
        name_tok = self.expect(
            "STRING",
            f"{role} name (must be a string literal)",
        )
        type_tok = self.expect(
            "IDENT",
            f"{role} {name_tok.value!r} type tag",
        )

        attrs: list[FieldAttr] = []
        while self.check("IDENT"):
            attrs.append(self._parse_field_attr())

        self.match("SEMI")
        return FieldDecl(
            role=role,
            field_name=name_tok.value,
            type_tag=type_tok.value,
            attrs=attrs,
            line=role_tok.line,
            col=role_tok.col,
        )

    def _parse_field_attr(self) -> FieldAttr:
        name_tok = self.expect("IDENT", "attribute name")
        self.expect("EQ", f"after attribute name '{name_tok.value}'")
        value = self._parse_literal(
            context=f"value of attribute '{name_tok.value}'",
        )
        return FieldAttr(
            name=name_tok.value, value=value, line=name_tok.line, col=name_tok.col
        )

    def _parse_route_when_decl(self) -> RouteWhenDecl:
        kw = self.advance()  # 'route_when'
        self.expect("LBRACE", "route_when body")

        decl = RouteWhenDecl(line=kw.line, col=kw.col)

        while not self.check("RBRACE") and not self.check("EOF"):
            tok = self.peek()
            if tok.kind == "KW_CONCEPT_PATH":
                self.advance()
                value_tok = self.expect("STRING", "concept_path value")
                decl.clauses.append(
                    RouteClause(
                        kind="concept_path",
                        value=StringLit(
                            value=value_tok.value,
                            line=value_tok.line,
                            col=value_tok.col,
                        ),
                        line=tok.line,
                        col=tok.col,
                    )
                )
                self.match("SEMI")
            elif tok.kind == "KW_BOOST":
                self.advance()
                value_tok = self.expect("NUMBER", "boost value")
                decl.clauses.append(
                    RouteClause(
                        kind="boost",
                        value=NumberLit(
                            value=float(value_tok.value),
                            line=value_tok.line,
                            col=value_tok.col,
                        ),
                        line=tok.line,
                        col=tok.col,
                    )
                )
                self.match("SEMI")
            else:
                raise self._error(
                    f"unexpected {tok.kind} in route_when block; "
                    f"expected 'concept_path' or 'boost'",
                    tok,
                )

        self.expect("RBRACE", "end of route_when block")
        return decl

    def _parse_acl_decl(self) -> AclDecl:
        kw = self.advance()  # 'acl'
        self.expect("LBRACE", "acl body")

        decl = AclDecl(line=kw.line, col=kw.col)

        while not self.check("RBRACE") and not self.check("EOF"):
            tok = self.peek()
            if tok.kind != "KW_GRANT":
                raise self._error(
                    f"unexpected {tok.kind} in acl block; expected 'grant'",
                    tok,
                )
            decl.grants.append(self._parse_acl_grant())

        self.expect("RBRACE", "end of acl block")
        return decl

    def _parse_acl_grant(self) -> AclGrantDecl:
        grant_kw = self.advance()  # 'grant'

        # principal: world | owner | STRING (a customer_id)
        ptok = self.peek()
        principal_kind = ""
        principal_id: Optional[str] = None
        if ptok.kind == "KW_WORLD":
            self.advance()
            principal_kind = "world"
        elif ptok.kind == "KW_OWNER":
            self.advance()
            principal_kind = "owner"
        elif ptok.kind == "STRING":
            self.advance()
            principal_kind = "literal"
            principal_id = ptok.value
        else:
            raise self._error(
                f"expected principal ('world', 'owner', or a quoted "
                f"customer_id), got {ptok.kind}",
                ptok,
            )

        # grant_kind: read | read_codebook
        gtok = self.peek()
        if gtok.kind == "KW_READ":
            self.advance()
            grant_kind = "read"
        elif gtok.kind == "KW_READ_CODEBOOK":
            self.advance()
            grant_kind = "read_codebook"
        else:
            raise self._error(
                f"expected grant kind ('read' or 'read_codebook'), "
                f"got {gtok.kind}",
                gtok,
            )

        self.match("SEMI")
        return AclGrantDecl(
            principal_kind=principal_kind,
            principal_id=principal_id,
            grant_kind=grant_kind,
            line=grant_kw.line,
            col=grant_kw.col,
        )

    # --- literal helpers

    def _parse_literal(self, *, context: str = "") -> Literal:
        tok = self.peek()
        if tok.kind == "STRING":
            self.advance()
            return StringLit(value=tok.value, line=tok.line, col=tok.col)
        if tok.kind == "NUMBER":
            self.advance()
            return NumberLit(value=float(tok.value), line=tok.line, col=tok.col)
        if tok.kind in ("KW_TRUE", "KW_FALSE"):
            self.advance()
            return BoolLit(
                value=(tok.kind == "KW_TRUE"),
                line=tok.line,
                col=tok.col,
            )
        ctx = f" ({context})" if context else ""
        raise self._error(
            f"expected literal (STRING, NUMBER, or BOOL), got {tok.kind}{ctx}",
            tok,
        )

    def _parse_literal_or_ident(self, *, context: str = "") -> LiteralOrIdent:
        tok = self.peek()
        if tok.kind == "IDENT":
            self.advance()
            return IdentLit(name=tok.value, line=tok.line, col=tok.col)
        # KW_TRUE / KW_FALSE handled by _parse_literal; everything else too.
        return self._parse_literal(context=context)


# =====================================================================
# Public API
# =====================================================================


def parse(source: str) -> Program:
    """Parse Schema DSL source into an AST.

    Raises SchemaParseError on syntax errors with line / column / caret.
    Successful parse does NOT validate semantic legality (duplicate
    pair_type names, ACL grant principals, etc.) — that's validator.py.
    """
    tokens = tokenize(source)
    parser = _Parser(tokens, source)
    program = parser.parse_program()
    if not parser.check("EOF"):
        tok = parser.peek()
        raise parser._error(f"unexpected {tok.kind} after program", tok)
    return program
