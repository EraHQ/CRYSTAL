"""POSIX-style read-permission resolution for crystals — Foundation F2.

A crystal is an owned resource (Foundation F2 axiom): it carries an owner
(`owner_operator_id`), a group (`group_team_id` — the owning team), and
POSIX mode bits. Only the READ bits are consumed today; write/execute are
reserved (the execute bit has no crystal semantics). Named grants in
`crystal_acls` are setfacl over the mode bits.

`can_read` decides whether an operator may read a crystal. It is applied as
a FILTER on top of the existing tenancy/subscription candidate set (see
`FactVectorStore.search`), never as a replacement — so when no operator
context is supplied, today's behavior is unchanged. The filter only ever
*removes* candidates the operator may not read; it never adds reach.

Scope tiers map to mode (the locked F2 mapping):
  - operator-private  → 0o600 (owner rw, group ---, other ---)
  - team              → 0o640 (owner rw, group r--, other ---)  [default]
  - general           → world-shared; gated by subscription upstream

This module imports no models at runtime (the type hints are under
TYPE_CHECKING and the logic is duck-typed on the passed objects), so it adds
no import cycle against the store or the vector store.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:  # imports for typing only — no runtime model dependency
    from ..models import Crystal, CrystalAcl, Operator

# POSIX read bits. Only read is consumed for retrieval gating in F2.
_READ_OWNER = 0o400
_READ_GROUP = 0o040
_READ_OTHER = 0o004

# Default mode when a row's mode is somehow unset (defensive; the column is
# NOT NULL with server_default 416 == 0o640).
_DEFAULT_MODE = 0o640

# Scope names → POSIX modes (P2, ratified 2026-07-02). The vocabulary the
# ingest default knob, the per-request scope field, and the share op all
# share — one mapping so a scope name always means the same bits.
SCOPE_MODES = {
    "personal": 0o600,  # owner-only — the ratified deployment default
    "team": 0o640,      # group-readable — the pre-P2 server default
}


def mode_for_scope(scope: str) -> int:
    """POSIX mode for a scope name; unknown scopes raise — never guess an
    access level."""
    try:
        return SCOPE_MODES[scope]
    except KeyError:
        raise ValueError(
            f"unknown scope {scope!r}; expected one of {sorted(SCOPE_MODES)}"
        ) from None


def may_join(
    crystal: "Crystal",
    *,
    owner_operator_id,
    group_team_id,
    mode: int,
) -> bool:
    """KEYSTONE (P2, ratified 2026-07-02): scope is a MERGE BOUNDARY.

    A pair may only join a crystal whose scope identity matches its own
    stamps — so a crystal can never mix personal facts with shareable
    team facts, which is what keeps crystal-grain ACL sound:

      - modes must match exactly (a personal pair never lands in a team
        crystal, and vice versa);
      - groups must match (legacy crystals fall back to customer_id as
        their group, same as can_read);
      - personal (0o600) additionally requires the SAME OWNER — two
        operators' private knowledge never shares a crystal. Team mode
        deliberately does NOT require same owner: contributing a fact to
        a shared crystal doesn't change who owns it (POSIX intuition).

    Pre-scope banks are unaffected: legacy crystals (mode default 0o640,
    group → customer) match team-stamped incoming pairs naturally.
    """
    c_mode = crystal.mode if crystal.mode is not None else _DEFAULT_MODE
    if c_mode != mode:
        return False
    c_group = crystal.group_team_id or crystal.customer_id
    if c_group != group_team_id:
        return False
    if mode == SCOPE_MODES["personal"]:
        return crystal.owner_operator_id == owner_operator_id
    return True


def can_read(
    crystal: "Crystal",
    operator: "Operator",
    acls: Iterable["CrystalAcl"] = (),
    operator_group_ids: Optional[frozenset] = None,
) -> bool:
    """Return True if `operator` may read `crystal`.

    Resolution order (first decisive match wins):

      0. General crystals (no owning tenant, `customer_id is None`) are
         world-shared knowledge. Subscription is their gate, applied
         upstream at the retrieval merge — so they pass here unconditionally.
      1. admin role is root WITHIN its own team — it bypasses the mode bits
         for any crystal grouped to the operator's team.
      2. a named read grant in `crystal_acls`: a 'global' grant is public; a
         'customer' grant to the operator's team lets the team in. Only the
         'read' grant counts — 'read_codebook' is a chain-only grant (it
         lets a crystal borrow another's facts via chaining; it is NOT a
         route-in/consume grant), so it does not open retrieval read here.
      3. POSIX class + read bit: the owner bit if the operator owns it; else
         the group bit if the crystal's group is the operator's team; else
         the other bit.

    `group_team` falls back to `customer_id` for legacy crystals authored
    before the group column existed — so a team's pre-F2 crystals stay
    readable by that team's operators (mode default 0o640 → group r).
    """
    # 0. General / world-shared crystals: subscription is the gate.
    if crystal.customer_id is None:
        return True

    mode = crystal.mode if crystal.mode is not None else _DEFAULT_MODE
    group_team = crystal.group_team_id or crystal.customer_id

    # 1. admin = root within its own team.
    if operator.role == "admin" and group_team == operator.team_id:
        return True

    # 2. named read grants (setfacl over the mode bits). Honors the existing
    #    crystal_acls vocabulary: 'global' is public; 'customer' to the
    #    operator's team grants the team. Only 'read' (not 'read_codebook').
    for acl in acls:
        if acl.grant != "read":
            continue
        if acl.principal_type == "global":
            return True
        if acl.principal_type == "customer" and acl.principal_id == operator.team_id:
            return True
        # P3 (ratified 2026-07-02): named grants to one operator or one
        # group — sub-team / individual sharing without touching the mode
        # bits. Group grants need the caller-supplied membership set;
        # when None (caller didn't thread it) the grant is IGNORED —
        # fail-closed, a missing lookup can never widen access.
        if acl.principal_type == "operator" and acl.principal_id == operator.id:
            return True
        if (
            acl.principal_type == "group"
            and operator_group_ids is not None
            and acl.principal_id in operator_group_ids
        ):
            return True

    # 3. POSIX class + read bit.
    if (
        crystal.owner_operator_id is not None
        and crystal.owner_operator_id == operator.id
    ):
        return bool(mode & _READ_OWNER)
    if group_team == operator.team_id:
        return bool(mode & _READ_GROUP)
    return bool(mode & _READ_OTHER)
