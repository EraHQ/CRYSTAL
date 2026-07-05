"""Crystal DSL parser — v0, experimental.

Parses Crystal DSL source text into an Abstract Syntax Tree (AST). The
AST is a plain-Python tree of dataclasses, consumable by compiler.py.

GRAMMAR (informal; v0 — will change)

    program        := statement*
    statement      := config_decl | concept_decl | assignment

    concept_decl   := 'concept' IDENT ( ',' IDENT )* ';'?
    assignment     := IDENT '=' expr ';'?
    config_decl    := 'config' IDENT '{' field* '}'

    field          := IDENT ':' expr ','?
    expr           := literal | pattern_expr | IDENT
    literal        := STRING | NUMBER | BOOL
    pattern_expr   := record_expr | lookup_expr | branch_expr |
                      sequence_expr | chain_expr

    record_expr    := 'record' '{' field* '}'
    lookup_expr    := 'lookup' '{' lookup_entry* '}'
    lookup_entry   := IDENT '->' expr ','?
    branch_expr    := 'if' IDENT 'then' expr 'else' expr
    sequence_expr  := 'sequence' '[' expr ( ',' expr )* ']'
    chain_expr     := 'chain' '{' chain_step* '}'
    chain_step     := 'step' IDENT '{' field* '}'

COMMENTS: '#' to end of line.

This grammar covers the five patterns in patterns.py plus top-level
concept declarations. It is deliberately minimal. We will add syntax
as real configs demand it — not speculatively.

STATUS: experimental. Syntax WILL change in v1.
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
class Ident(Node):
    name: str = ""


@dataclass
class RecordExpr(Node):
    fields: dict[str, "Expr"] = field(default_factory=dict)


@dataclass
class LookupExpr(Node):
    entries: dict[str, "Expr"] = field(default_factory=dict)


@dataclass
class BranchExpr(Node):
    condition: Ident = field(default_factory=Ident)
    if_true: "Expr" = None  # type: ignore
    if_false: "Expr" = None  # type: ignore


@dataclass
class SequenceExpr(Node):
    items: list["Expr"] = field(default_factory=list)


@dataclass
class ChainStep(Node):
    name: str = ""
    fields: dict[str, "Expr"] = field(default_factory=dict)


@dataclass
class ChainExpr(Node):
    steps: list[ChainStep] = field(default_factory=list)


Expr = Union[
    StringLit, NumberLit, BoolLit, Ident,
    RecordExpr, LookupExpr, BranchExpr, SequenceExpr, ChainExpr
]


@dataclass
class ConceptDecl(Node):
    names: list[str] = field(default_factory=list)


@dataclass
class Assignment(Node):
    name: str = ""
    value: Expr = None  # type: ignore


@dataclass
class ConfigDecl(Node):
    name: str = ""
    fields: dict[str, Expr] = field(default_factory=dict)


Statement = Union[ConceptDecl, Assignment, ConfigDecl]


@dataclass
class Program(Node):
    statements: list[Statement] = field(default_factory=list)


# =====================================================================
# Errors
# =====================================================================


class ParseError(Exception):
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


# Keywords. Identifiers matching these are tokenized as keywords, not IDENT.
KEYWORDS = {
    "config", "concept", "record", "lookup", "branch", "sequence",
    "chain", "step", "if", "then", "else", "true", "false",
}


@dataclass
class Token:
    kind: str
    value: str
    line: int
    col: int


_TOKEN_SPECS = [
    ("COMMENT",  r"\#[^\n]*"),
    ("WS",       r"[ \t\r]+"),
    ("NEWLINE",  r"\n"),
    ("STRING",   r'"(?:[^"\\]|\\.)*"'),
    ("NUMBER",   r"-?\d+(?:\.\d+)?"),
    ("ARROW",    r"->"),
    ("LBRACE",   r"\{"),
    ("RBRACE",   r"\}"),
    ("LBRACK",   r"\["),
    ("RBRACK",   r"\]"),
    ("COMMA",    r","),
    ("COLON",    r":"),
    ("SEMI",     r";"),
    ("EQ",       r"="),
    ("IDENT",    r"[A-Za-z_][A-Za-z0-9_]*"),
]

_TOKEN_RE = re.compile("|".join(f"(?P<{k}>{p})" for k, p in _TOKEN_SPECS))


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    line = 1
    line_start = 0
    pos = 0
    length = len(source)
    while pos < length:
        m = _TOKEN_RE.match(source, pos)
        if not m:
            col = pos - line_start + 1
            raise ParseError(
                f"unexpected character {source[pos]!r}", line, col,
                _extract_source_line(source, line_start),
            )
        kind = m.lastgroup
        raw = m.group()
        col = pos - line_start + 1

        if kind == "WS" or kind == "COMMENT":
            pos = m.end()
            continue
        if kind == "NEWLINE":
            line += 1
            pos = m.end()
            line_start = pos
            continue

        value = raw
        if kind == "STRING":
            # Strip quotes + basic unescape
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

    def _error(self, msg: str, tok: Token) -> ParseError:
        src_line = ""
        if 1 <= tok.line <= len(self._source_lines):
            src_line = self._source_lines[tok.line - 1]
        return ParseError(msg, tok.line, tok.col, src_line)

    # --- grammar

    def parse_program(self) -> Program:
        prog = Program()
        while not self.check("EOF"):
            prog.statements.append(self._parse_statement())
        return prog

    def _parse_statement(self) -> Statement:
        tok = self.peek()
        if tok.kind == "KW_CONCEPT":
            return self._parse_concept_decl()
        if tok.kind == "KW_CONFIG":
            return self._parse_config_decl()
        if tok.kind == "IDENT":
            return self._parse_assignment()
        raise self._error(
            f"unexpected {tok.kind} at top level; "
            f"expected 'concept', 'config', or an assignment",
            tok,
        )

    def _parse_concept_decl(self) -> ConceptDecl:
        kw = self.advance()  # 'concept'
        names: list[str] = []
        first = self.expect("IDENT", "concept name")
        names.append(first.value)
        while self.match("COMMA"):
            nxt = self.expect("IDENT", "concept name after comma")
            names.append(nxt.value)
        self.match("SEMI")
        return ConceptDecl(names=names, line=kw.line, col=kw.col)

    def _parse_assignment(self) -> Assignment:
        ident = self.expect("IDENT")
        self.expect("EQ", f"assignment to '{ident.value}'")
        value = self._parse_expr()
        self.match("SEMI")
        return Assignment(name=ident.value, value=value, line=ident.line, col=ident.col)

    def _parse_config_decl(self) -> ConfigDecl:
        kw = self.advance()  # 'config'
        name_tok = self.expect("IDENT", "config name")
        self.expect("LBRACE", f"config '{name_tok.value}' body")
        fields = self._parse_field_list()
        self.expect("RBRACE", f"end of config '{name_tok.value}'")
        return ConfigDecl(
            name=name_tok.value, fields=fields, line=kw.line, col=kw.col
        )

    def _parse_field_list(self) -> dict[str, Expr]:
        """Parse a sequence of `IDENT: expr [,]` up to the matching RBRACE."""
        fields: dict[str, Expr] = {}
        while not self.check("RBRACE") and not self.check("EOF"):
            key_tok = self.expect("IDENT", "field name")
            if key_tok.value in fields:
                raise self._error(f"duplicate field '{key_tok.value}'", key_tok)
            self.expect("COLON", f"after field name '{key_tok.value}'")
            value = self._parse_expr()
            fields[key_tok.value] = value
            self.match("COMMA")
        return fields

    def _parse_expr(self) -> Expr:
        tok = self.peek()
        if tok.kind == "STRING":
            self.advance()
            return StringLit(value=tok.value, line=tok.line, col=tok.col)
        if tok.kind == "NUMBER":
            self.advance()
            return NumberLit(value=float(tok.value), line=tok.line, col=tok.col)
        if tok.kind in ("KW_TRUE", "KW_FALSE"):
            self.advance()
            return BoolLit(value=(tok.kind == "KW_TRUE"), line=tok.line, col=tok.col)
        if tok.kind == "KW_RECORD":
            return self._parse_record_expr()
        if tok.kind == "KW_LOOKUP":
            return self._parse_lookup_expr()
        if tok.kind == "KW_SEQUENCE":
            return self._parse_sequence_expr()
        if tok.kind == "KW_CHAIN":
            return self._parse_chain_expr()
        if tok.kind == "KW_IF":
            return self._parse_branch_expr()
        if tok.kind == "IDENT":
            self.advance()
            return Ident(name=tok.value, line=tok.line, col=tok.col)
        raise self._error(f"unexpected {tok.kind} in expression", tok)

    def _parse_record_expr(self) -> RecordExpr:
        kw = self.advance()  # 'record'
        self.expect("LBRACE", "record body")
        fields = self._parse_field_list()
        self.expect("RBRACE", "end of record")
        return RecordExpr(fields=fields, line=kw.line, col=kw.col)

    def _parse_lookup_expr(self) -> LookupExpr:
        kw = self.advance()  # 'lookup'
        self.expect("LBRACE", "lookup body")
        entries: dict[str, Expr] = {}
        while not self.check("RBRACE") and not self.check("EOF"):
            key_tok = self.expect("IDENT", "lookup key")
            if key_tok.value in entries:
                raise self._error(f"duplicate lookup key '{key_tok.value}'", key_tok)
            self.expect("ARROW", f"after lookup key '{key_tok.value}'")
            value = self._parse_expr()
            entries[key_tok.value] = value
            self.match("COMMA")
        self.expect("RBRACE", "end of lookup")
        return LookupExpr(entries=entries, line=kw.line, col=kw.col)

    def _parse_branch_expr(self) -> BranchExpr:
        kw = self.advance()  # 'if'
        cond_tok = self.expect("IDENT", "branch condition (must be a concept name)")
        self.expect("KW_THEN", "after branch condition")
        if_true = self._parse_expr()
        self.expect("KW_ELSE", "after 'then' branch")
        if_false = self._parse_expr()
        return BranchExpr(
            condition=Ident(name=cond_tok.value, line=cond_tok.line, col=cond_tok.col),
            if_true=if_true,
            if_false=if_false,
            line=kw.line,
            col=kw.col,
        )

    def _parse_sequence_expr(self) -> SequenceExpr:
        kw = self.advance()  # 'sequence'
        self.expect("LBRACK", "sequence body")
        items: list[Expr] = []
        if not self.check("RBRACK"):
            items.append(self._parse_expr())
            while self.match("COMMA"):
                if self.check("RBRACK"):
                    break  # trailing comma OK
                items.append(self._parse_expr())
        self.expect("RBRACK", "end of sequence")
        return SequenceExpr(items=items, line=kw.line, col=kw.col)

    def _parse_chain_expr(self) -> ChainExpr:
        kw = self.advance()  # 'chain'
        self.expect("LBRACE", "chain body")
        steps: list[ChainStep] = []
        seen_names: set[str] = set()
        while not self.check("RBRACE") and not self.check("EOF"):
            step_kw = self.expect("KW_STEP", "chain step")
            name_tok = self.expect("IDENT", "step name")
            if name_tok.value in seen_names:
                raise self._error(
                    f"duplicate step name '{name_tok.value}' in chain", name_tok
                )
            seen_names.add(name_tok.value)
            self.expect("LBRACE", f"step '{name_tok.value}' body")
            fields = self._parse_field_list()
            self.expect("RBRACE", f"end of step '{name_tok.value}'")
            steps.append(
                ChainStep(
                    name=name_tok.value,
                    fields=fields,
                    line=step_kw.line,
                    col=step_kw.col,
                )
            )
        self.expect("RBRACE", "end of chain")
        return ChainExpr(steps=steps, line=kw.line, col=kw.col)


# =====================================================================
# Public API
# =====================================================================


def parse(source: str) -> Program:
    """Parse DSL source text into an AST.

    Raises ParseError with line/column info on syntax errors.
    """
    tokens = tokenize(source)
    parser = _Parser(tokens, source)
    program = parser.parse_program()
    if not parser.check("EOF"):
        tok = parser.peek()
        raise parser._error(f"unexpected {tok.kind} after program", tok)
    return program
