"""Schema DSL — Phase 4 of the bind-storage rebuild.

The Schema DSL declares crystal_types, their valid pair-types, their
routing hints, and their ACL defaults. It is a separate layer from the
existing concept DSL (`crystal_cache.dsl.{parser,compiler,runtime}`),
which encodes concepts and configs as hypervectors for M1 routing.

Pipeline:

    parse  (parser.py)     : DSL source     -> AST
    validate (validator.py): AST            -> list[Diagnostic]
    compile  (compiler.py) : AST            -> CompiledProgram
    load     (loader.py)   : DB row         -> CompiledCrystalType (cached)

Runtime hooks (in metadata_store / router / pipeline) consume
CompiledCrystalType for pair_type validation, route-hint boosting,
and ACL-default instantiation on new crystal creation.

Field type registry lives in `field_types.py` and is imported by the
validator; adding a new tag is a code change there plus the runtime
validator that enforces field shape at write time.

STATUS: parser (4.1), validator (4.2), compiler (4.3), and loader
(4.4) landed.
"""

from crystal_cache.dsl.schema.compiler import (
    CompiledAclGrant,
    CompiledCrystalType,
    CompiledField,
    CompiledFieldAttr,
    CompiledPairType,
    CompiledProgram,
    CompiledRouteHint,
    SchemaCompileError,
    compile_or_raise,
    compile_program,
)
from crystal_cache.dsl.schema.field_types import (
    FIELD_TYPE_REGISTRY,
    FieldTypeSpec,
    all_tags,
    get_spec,
    is_known_tag,
)
from crystal_cache.dsl.schema.loader import (
    SchemaLoadError,
    SchemaLoader,
    resolve_acl_defaults,
)
from crystal_cache.dsl.schema.parser import (
    AclDecl,
    AclGrantDecl,
    BoolLit,
    CrystalTypeDecl,
    FieldAttr,
    FieldDecl,
    HeaderField,
    IdentLit,
    NumberLit,
    PairTypeDecl,
    Program,
    RouteClause,
    RouteWhenDecl,
    SchemaParseError,
    StringLit,
    parse,
    tokenize,
)
from crystal_cache.dsl.schema.validator import (
    Diagnostic,
    SchemaValidationError,
    validate,
    validate_or_raise,
)

__all__ = [
    # Parser
    "AclDecl",
    "AclGrantDecl",
    "BoolLit",
    "CrystalTypeDecl",
    "FieldAttr",
    "FieldDecl",
    "HeaderField",
    "IdentLit",
    "NumberLit",
    "PairTypeDecl",
    "Program",
    "RouteClause",
    "RouteWhenDecl",
    "SchemaParseError",
    "StringLit",
    "parse",
    "tokenize",
    # Field type registry
    "FIELD_TYPE_REGISTRY",
    "FieldTypeSpec",
    "all_tags",
    "get_spec",
    "is_known_tag",
    # Validator
    "Diagnostic",
    "SchemaValidationError",
    "validate",
    "validate_or_raise",
    # Compiler
    "CompiledAclGrant",
    "CompiledCrystalType",
    "CompiledField",
    "CompiledFieldAttr",
    "CompiledPairType",
    "CompiledProgram",
    "CompiledRouteHint",
    "SchemaCompileError",
    "compile_or_raise",
    "compile_program",
    # Loader
    "SchemaLoadError",
    "SchemaLoader",
    "resolve_acl_defaults",
]
