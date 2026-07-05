"""Foundation F2 — POSIX permission resolution + permission-checked retrieval.

Two layers:
  1. can_read truth table (pure): the resolver's full decision matrix over
     owner / group / other mode bits, admin root-bypass, and named ACL
     grants. No DB — constructs Crystal / Operator / CrystalAcl directly.
  2. FactVectorStore.search() filtering (integration): an operator's search
     hides a teammate's owner-private crystal but surfaces team-readable
     ones, and operator=None preserves today's unfiltered behavior.

asyncio_mode=auto (pyproject) — async tests need no marker.
"""
from __future__ import annotations

from crystal_cache.infrastructure.permissions import can_read
from crystal_cache.models import Crystal, CrystalAcl, Operator


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _op(op_id: str, team_id: str, role: str = "operator") -> Operator:
    return Operator(id=op_id, team_id=team_id, display_name=op_id, role=role)


def _crystal(
    *,
    customer_id="team_1",
    owner=None,
    group=None,
    mode=0o640,
) -> Crystal:
    return Crystal(
        id="crys_test",
        customer_id=customer_id,
        summary_vector=[],
        owner_operator_id=owner,
        group_team_id=group,
        mode=mode,
    )


def _acl(principal_type: str, principal_id: str, grant: str = "read") -> CrystalAcl:
    return CrystalAcl(
        crystal_id="crys_test",
        principal_type=principal_type,
        principal_id=principal_id,
        grant=grant,
    )


# ---------------------------------------------------------------------------
# can_read truth table
# ---------------------------------------------------------------------------

def test_general_crystal_is_world_readable():
    # customer_id None = general/world-shared; subscription gates it
    # upstream, so the resolver passes any operator unconditionally.
    crystal = _crystal(customer_id=None, owner="op_x", group="team_x", mode=0o600)
    assert can_read(crystal, _op("op_a", "team_1")) is True


def test_owner_reads_own_private_crystal():
    crystal = _crystal(owner="op_a", group="team_1", mode=0o600)
    assert can_read(crystal, _op("op_a", "team_1")) is True


def test_teammate_cannot_read_owner_private_crystal():
    # mode 0o600: owner rw, group ---, other ---. A teammate is in the
    # group but the group bit is unset → denied.
    crystal = _crystal(owner="op_a", group="team_1", mode=0o600)
    assert can_read(crystal, _op("op_b", "team_1")) is False


def test_teammate_reads_team_crystal():
    # mode 0o640: group r. Default team-collaborative posture.
    crystal = _crystal(owner="op_a", group="team_1", mode=0o640)
    assert can_read(crystal, _op("op_b", "team_1")) is True


def test_owner_class_takes_precedence_over_group():
    # POSIX: the owner is judged by the OWNER bits, not group — even when
    # group is more permissive. mode 0o040 (group r, owner ---) → the owner
    # gets no read.
    crystal = _crystal(owner="op_a", group="team_1", mode=0o040)
    assert can_read(crystal, _op("op_a", "team_1")) is False
    # ...but a (non-owner) teammate reads it via the group bit.
    assert can_read(crystal, _op("op_b", "team_1")) is True


def test_admin_is_root_within_its_team():
    # admin bypasses the mode bits for a crystal grouped to its team.
    crystal = _crystal(owner="op_a", group="team_1", mode=0o600)
    assert can_read(crystal, _op("op_admin", "team_1", role="admin")) is True


def test_admin_does_not_bypass_other_teams():
    # root is team-scoped: an admin of team_2 has no special power over a
    # team_1 crystal; it falls to the other bit (0o600 → none) → denied.
    crystal = _crystal(customer_id="team_1", owner="op_a", group="team_1", mode=0o600)
    assert can_read(crystal, _op("op_admin", "team_2", role="admin")) is False


def test_other_bit_grants_cross_group_read():
    # A different team's operator reads only via the other bit.
    readable = _crystal(customer_id="team_1", group="team_1", mode=0o644)  # other r
    assert can_read(readable, _op("op_b", "team_2")) is True
    hidden = _crystal(customer_id="team_1", group="team_1", mode=0o640)  # other ---
    assert can_read(hidden, _op("op_b", "team_2")) is False


def test_acl_read_grant_to_team_overrides_mode():
    # A named 'read' grant to the operator's team lets it in even when the
    # mode bits would deny (non-owner, group ---).
    crystal = _crystal(owner="op_a", group="team_1", mode=0o600)
    acls = [_acl("customer", "team_1", "read")]
    assert can_read(crystal, _op("op_b", "team_1"), acls) is True


def test_acl_global_grant_is_public():
    crystal = _crystal(owner="op_a", group="team_1", mode=0o600)
    acls = [_acl("global", "world", "read")]
    assert can_read(crystal, _op("op_b", "team_2"), acls) is True


def test_read_codebook_grant_does_not_open_retrieval():
    # 'read_codebook' is a chain-only grant (borrow facts via chaining); it
    # is NOT a route-in/consume grant, so it must not open retrieval read.
    crystal = _crystal(owner="op_a", group="team_1", mode=0o600)
    acls = [_acl("customer", "team_1", "read_codebook")]
    assert can_read(crystal, _op("op_b", "team_1"), acls) is False


def test_legacy_crystal_with_null_group_falls_back_to_tenant():
    # Pre-F2 crystal: owner/group NULL, mode default 0o640. The resolver
    # falls back to customer_id for the group, so the owning team reads it.
    crystal = _crystal(customer_id="team_1", owner=None, group=None, mode=0o640)
    assert can_read(crystal, _op("op_b", "team_1")) is True
    # An outsider gets only the other bit (none) → denied.
    assert can_read(crystal, _op("op_c", "team_2")) is False


# ---------------------------------------------------------------------------
# FactVectorStore.search() permission filtering
# ---------------------------------------------------------------------------

async def _seed_crystal_with_fact(
    store, *, crystal_id, customer_id, owner, group, mode, answer, encoder,
):
    """Create a crystal with explicit POSIX fields, then bind one fact into
    it (add_pair_to_crystal sets the fact vector but leaves owner/group/mode
    untouched)."""
    await store.upsert_crystal(Crystal(
        id=crystal_id,
        customer_id=customer_id,
        summary_vector=[],
        owner_operator_id=owner,
        group_team_id=group,
        mode=mode,
        crystal_type="customer:legacy",
    ))
    await store.add_pair_to_crystal(
        crystal_id=crystal_id,
        prompt_text=f"q for {answer}",
        answer_text=answer,
        encoder=encoder,
    )


def _crystal_ids(results) -> set[str]:
    # search rows are (fact_id, crystal_id, pair_type, score).
    return {row[1] for row in results}


async def test_search_filters_owner_private_for_teammate(
    store, customer, semantic_encoder_stub, fact_vector_store,
):
    team = customer.id
    op_a, _ = await store.create_operator(team_id=team, display_name="A")
    op_b, _ = await store.create_operator(team_id=team, display_name="B")

    # Private to op_a (0o600); team-readable (0o640).
    await _seed_crystal_with_fact(
        store, crystal_id="crys_private", customer_id=team,
        owner=op_a.id, group=team, mode=0o600,
        answer="alpha secret", encoder=semantic_encoder_stub,
    )
    await _seed_crystal_with_fact(
        store, crystal_id="crys_team", customer_id=team,
        owner=op_a.id, group=team, mode=0o640,
        answer="beta shared", encoder=semantic_encoder_stub,
    )

    # Query vector matches the private crystal's fact (same text → ~1.0
    # cosine under the deterministic stub).
    query_vec = semantic_encoder_stub.encode_native("alpha secret")

    # No operator → today's behavior: both crystals present, unfiltered.
    unfiltered = await fact_vector_store.search(team, query_vec, k=5)
    assert "crys_private" in _crystal_ids(unfiltered)
    assert "crys_team" in _crystal_ids(unfiltered)

    # op_b (teammate, not owner) → private crystal filtered out, team one kept.
    as_b = await fact_vector_store.search(team, query_vec, k=5, operator=op_b)
    ids_b = _crystal_ids(as_b)
    assert "crys_private" not in ids_b
    assert "crys_team" in ids_b

    # op_a (owner) → sees its own private crystal.
    as_a = await fact_vector_store.search(team, query_vec, k=5, operator=op_a)
    assert "crys_private" in _crystal_ids(as_a)


async def test_search_admin_sees_all_team_crystals(
    store, customer, semantic_encoder_stub, fact_vector_store,
):
    team = customer.id
    op_a, _ = await store.create_operator(team_id=team, display_name="A")
    admin, _ = await store.create_operator(
        team_id=team, display_name="Root", role="admin",
    )
    await _seed_crystal_with_fact(
        store, crystal_id="crys_private", customer_id=team,
        owner=op_a.id, group=team, mode=0o600,
        answer="alpha secret", encoder=semantic_encoder_stub,
    )
    query_vec = semantic_encoder_stub.encode_native("alpha secret")

    # Admin is root within the team → reads op_a's private crystal.
    as_admin = await fact_vector_store.search(team, query_vec, k=5, operator=admin)
    assert "crys_private" in _crystal_ids(as_admin)
