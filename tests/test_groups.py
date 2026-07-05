"""P3 groups (ratified 2026-07-02).

Named sub-teams as grant targets: group CRUD with customer guards, the
membership set feeding can_read, group + operator grants opening reads
without touching POSIX mode, fail-closed group grants when the caller
doesn't supply memberships, and the endpoint authorization.

R14 note: these assertions are verified by `pytest`; they describe
expected behavior and have not yet been run at authoring time.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from crystal_cache.endpoints.groups import (
    CreateGroupRequest,
    GroupMemberRequest,
    add_group_member,
    create_group,
    list_groups,
)
from crystal_cache.endpoints.sdk import (
    sdk_add_crystal_grant,
    sdk_remove_crystal_grant,
)
from crystal_cache.infrastructure.permissions import can_read
from crystal_cache.infrastructure.schema import CrystalRow
from crystal_cache.models.crystal_type import CrystalAcl


async def _seed_crystal(store, customer_id, cid, *, owner, mode=0o600):
    async with store.session() as s:
        s.add(CrystalRow(
            id=cid, customer_id=customer_id,
            crystal_type="customer:legacy", summary_vector=[],
            owner_operator_id=owner, group_team_id=customer_id, mode=mode,
        ))


async def test_group_crud_and_membership(store, customer):
    op_a, _ = await store.create_operator(customer.id, display_name="A")
    op_b, _ = await store.create_operator(customer.id, display_name="B")

    group = await store.create_group(customer.id, "backend")
    assert await store.add_group_member(group["id"], op_a.id, customer.id)
    assert await store.add_group_member(group["id"], op_a.id, customer.id)  # idempotent
    assert await store.add_group_member(group["id"], op_b.id, customer.id)

    listed = await store.list_groups_for_customer(customer.id)
    assert listed[0]["name"] == "backend"
    assert set(listed[0]["member_ids"]) == {op_a.id, op_b.id}

    assert await store.list_group_ids_for_operator(op_a.id) == frozenset({group["id"]})

    assert await store.remove_group_member(group["id"], op_b.id, customer.id)
    assert await store.list_group_ids_for_operator(op_b.id) == frozenset()

    # Customer guards: foreign team can't touch the group; foreign
    # operator can't be added.
    assert not await store.add_group_member(group["id"], op_a.id, "cus_other")
    other = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="k",
    )
    op_f, _ = await store.create_operator(other.id, display_name="F")
    assert not await store.add_group_member(group["id"], op_f.id, customer.id)


async def test_group_grant_opens_read_fail_closed_without_memberships(
    store, customer,
):
    """A group grant lets members read a personal crystal — but ONLY when
    the caller threads the membership set. None → grant ignored."""
    owner, _ = await store.create_operator(customer.id, display_name="Own")
    member, _ = await store.create_operator(customer.id, display_name="Mem")
    outsider, _ = await store.create_operator(customer.id, display_name="Out")
    await _seed_crystal(store, customer.id, "c_g", owner=owner.id)

    group = await store.create_group(customer.id, "sub")
    await store.add_group_member(group["id"], member.id, customer.id)
    await store.add_acl(CrystalAcl(
        crystal_id="c_g", principal_type="group",
        principal_id=group["id"], grant="read",
    ))

    crystal = await store.get_crystal("c_g")
    acls = await store.list_acls_for_crystal("c_g")
    member_groups = await store.list_group_ids_for_operator(member.id)
    outsider_groups = await store.list_group_ids_for_operator(outsider.id)

    assert can_read(crystal, member, acls, member_groups) is True
    assert can_read(crystal, outsider, acls, outsider_groups) is False
    # Fail-closed: no membership set supplied → the group grant is ignored.
    assert can_read(crystal, member, acls) is False


async def test_operator_grant_opens_read(store, customer):
    owner, _ = await store.create_operator(customer.id, display_name="Own")
    friend, _ = await store.create_operator(customer.id, display_name="Fr")
    await _seed_crystal(store, customer.id, "c_o", owner=owner.id)

    await store.add_acl(CrystalAcl(
        crystal_id="c_o", principal_type="operator",
        principal_id=friend.id, grant="read",
    ))

    crystal = await store.get_crystal("c_o")
    acls = await store.list_acls_for_crystal("c_o")
    assert can_read(crystal, friend, acls) is True  # no membership set needed


async def test_group_endpoints_admin_gate_and_flow(store, customer):
    admin = await store.ensure_default_admin(customer.id)
    op_a, _ = await store.create_operator(customer.id, display_name="A")

    resp = await create_group(
        CreateGroupRequest(name="frontend", member_ids=[op_a.id, "op_bogus"]),
        (customer, admin), store,
    )
    assert resp.status_code == 200
    import json
    payload = json.loads(resp.body)
    assert payload["skipped_member_ids"] == ["op_bogus"]

    # Non-admin blocked from managing.
    with pytest.raises(HTTPException) as exc:
        await create_group(
            CreateGroupRequest(name="x"), (customer, op_a), store,
        )
    assert exc.value.status_code == 403
    with pytest.raises(HTTPException) as exc:
        await add_group_member(
            payload["id"], GroupMemberRequest(operator_id=op_a.id),
            (customer, op_a), store,
        )
    assert exc.value.status_code == 403

    # Duplicate name → 409.
    with pytest.raises(HTTPException) as exc:
        await create_group(
            CreateGroupRequest(name="frontend"), (customer, admin), store,
        )
    assert exc.value.status_code == 409

    # Listing open to any principal.
    resp = await list_groups((customer, op_a), store)
    assert json.loads(resp.body)["total"] == 1


async def test_grant_endpoint_authz_and_revoke(store, customer):
    owner, _ = await store.create_operator(customer.id, display_name="Own")
    other, _ = await store.create_operator(customer.id, display_name="Oth")
    friend, _ = await store.create_operator(customer.id, display_name="Fr")
    await _seed_crystal(store, customer.id, "c_e", owner=owner.id)

    # Owner grants to an individual operator.
    resp = await sdk_add_crystal_grant(
        "c_e", {"principal_type": "operator", "principal_id": friend.id},
        (customer, owner), store,
    )
    assert resp.status_code == 200
    crystal = await store.get_crystal("c_e")
    acls = await store.list_acls_for_crystal("c_e")
    assert can_read(crystal, friend, acls) is True

    # Non-owner/non-admin can't grant; unknown principal 404; bad type 422.
    with pytest.raises(HTTPException) as exc:
        await sdk_add_crystal_grant(
            "c_e", {"principal_type": "operator", "principal_id": other.id},
            (customer, other), store,
        )
    assert exc.value.status_code == 403
    with pytest.raises(HTTPException) as exc:
        await sdk_add_crystal_grant(
            "c_e", {"principal_type": "group", "principal_id": "grp_missing"},
            (customer, owner), store,
        )
    assert exc.value.status_code == 404
    with pytest.raises(HTTPException) as exc:
        await sdk_add_crystal_grant(
            "c_e", {"principal_type": "world", "principal_id": "x"},
            (customer, owner), store,
        )
    assert exc.value.status_code == 422

    # Revoke closes the read again.
    resp = await sdk_remove_crystal_grant(
        "c_e", {"principal_type": "operator", "principal_id": friend.id},
        (customer, owner), store,
    )
    assert resp.status_code == 200
    acls = await store.list_acls_for_crystal("c_e")
    assert can_read(crystal, friend, acls) is False
