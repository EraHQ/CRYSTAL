"""Promotion engine — Foundation F3 (detect → curate → merge up).

The one operation that moves a crystal up a tier, built first for the
operator→team rung. Multiple operators independently hold near-identical
private crystals; the engine DETECTS the redundancy, an admin CURATES
(decides), and the crystals MERGE UP into one team crystal carrying every
contributor's provenance.

Under the POSIX access model (FOUNDATION_AND_GROWTH.md axiom 4) promotion
is literally chgrp/chmod-up a tier: the surviving crystal is re-grouped to
the team (group_team_id = team, owner cleared) and its mode opened to
group-read (0o640). The superseded originals are deleted — the REPLACE
semantics already used by the versioned-crystal write path, with the team
crystal as the single survivor.

This module is auth-agnostic: the admin (= root) gate lives at the HTTP
boundary (require_role("admin") on the F3 endpoints). "Curation" here means
"an authorized caller chose these source ids to merge."

Reuse, not reinvention: detect reuses the routing-vector geometry the bank
already computes; merge reuses the store's crystal primitives (get_crystal /
upsert_crystal / delete_crystal). The only F3-specific persistence is the
contributor-provenance pair on the store (record_/list_promotion_
contributions — metadata_store_promotion_ext.py).

F3 v1 scope notes (deliberate, revisable):
  - Detect compares `routing_vector` cosine (the crystal's routing address);
    two crystals encoding the same knowledge have near-identical addresses.
    Crystals without a routing_vector are skipped.
  - A cluster is a promotion candidate only if it spans >= 2 DISTINCT
    operators (the point is independent operators holding the same thing).
  - Merge does NOT re-parent the non-survivors' facts onto the survivor.
    The cluster is near-duplicate by construction, so the survivor already
    represents it; re-parenting facts onto a crystal whose HDC
    summary_vector wasn't re-accumulated would make crystal-level recall
    inconsistent. A true fact-union (with vector re-accumulation) is a
    future refinement; knowledge loss is bounded by the similarity gate.
  - Share policy v1: equal split across source crystals (basis points
    summing to 10000), captured per source. G4 aggregates by operator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import structlog

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore
    from ..infrastructure.vector_store import VectorStore
    from ..infrastructure.fact_vector_store import FactVectorStore
    from ..models import Crystal

logger = structlog.get_logger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.95
TOTAL_SHARE_BASIS_POINTS = 10_000


@dataclass
class PromotionCandidate:
    """A cluster of near-duplicate operator-private crystals proposed for
    promotion to the team tier."""
    crystal_ids: list[str]
    operator_ids: list[str]  # distinct contributors, sorted
    mean_similarity: float
    # crystal_id -> summary_text (best-effort preview for the curate UI).
    previews: dict[str, Optional[str]] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.crystal_ids)


@dataclass
class MergeResult:
    """Outcome of a merge: the surviving team crystal + what it cost."""
    merged_crystal_id: str
    superseded_crystal_ids: list[str]
    # [{source_crystal_id, contributor_operator_id, share_basis_points}, ...]
    contributions: list[dict]


class PromotionError(Exception):
    """Raised when a merge request is invalid (unknown / cross-team /
    non-operator-owned sources, or fewer than two distinct operators)."""


def _cosine(a: Optional[list[float]], b: Optional[list[float]]) -> float:
    """Cosine similarity between two raw (non-unit-norm) vectors. Returns
    0.0 if either is empty, None, dimension-mismatched, or zero-norm."""
    if not a or not b or len(a) != len(b):
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class PromotionService:
    """Detect promotion candidates and merge them up a tier (F3)."""

    def __init__(self, store: "MetadataStore"):
        self._store = store

    # ---- DETECT -----------------------------------------------------

    async def detect_candidates(
        self,
        team_id: str,
        *,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> list[PromotionCandidate]:
        """Find clusters of near-duplicate operator-private crystals within
        a team that span >= 2 distinct operators.

        Scans the team's operator-owned crystals (owner_operator_id set,
        routing_vector present), single-linkage clusters them by
        routing-vector cosine >= threshold, and returns each multi-operator
        cluster as a candidate. Admin-governable (D3): root may read private
        content, so a team admin can act on these.
        """
        crystals = await self._store.list_crystals_for_customer(team_id)
        owned = [
            c for c in crystals
            if c.owner_operator_id is not None and c.routing_vector
        ]
        n = len(owned)
        if n < 2:
            return []

        # Union-find over pairwise edges >= threshold (single linkage).
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)

        edge_sims: dict[tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                sim = _cosine(owned[i].routing_vector, owned[j].routing_vector)
                if sim >= threshold:
                    union(i, j)
                    edge_sims[(i, j)] = sim

        comps: dict[int, list[int]] = {}
        for idx in range(n):
            comps.setdefault(find(idx), []).append(idx)

        candidates: list[PromotionCandidate] = []
        for members in comps.values():
            if len(members) < 2:
                continue
            member_set = set(members)
            operator_ids = sorted({
                owned[m].owner_operator_id for m in members
                if owned[m].owner_operator_id is not None
            })
            if len(operator_ids) < 2:
                continue  # within-operator duplicate, not a promotion
            sims = [
                s for (i, j), s in edge_sims.items()
                if i in member_set and j in member_set
            ]
            mean_sim = float(sum(sims) / len(sims)) if sims else 0.0
            candidates.append(PromotionCandidate(
                crystal_ids=sorted(owned[m].id for m in members),
                operator_ids=operator_ids,
                mean_similarity=mean_sim,
                previews={owned[m].id: owned[m].summary_text for m in members},
            ))

        logger.info(
            "promotion.detect",
            team_id=team_id,
            candidate_clusters=len(candidates),
            scanned=n,
            threshold=threshold,
        )
        return candidates

    # ---- MERGE ------------------------------------------------------

    @staticmethod
    def _equal_shares(count: int) -> list[int]:
        """Split TOTAL_SHARE_BASIS_POINTS into `count` integer shares summing
        EXACTLY to the total (remainder distributed to the first few)."""
        if count <= 0:
            return []
        base = TOTAL_SHARE_BASIS_POINTS // count
        rem = TOTAL_SHARE_BASIS_POINTS - base * count
        return [base + (1 if i < rem else 0) for i in range(count)]

    @staticmethod
    def _pick_survivor(sources: list["Crystal"]) -> "Crystal":
        """Richest by fact_count; tiebreak oldest created_at, then id."""
        return sorted(
            sources,
            key=lambda c: (-(c.fact_count or 0), c.created_at, c.id),
        )[0]

    async def merge(
        self,
        team_id: str,
        source_crystal_ids: list[str],
        *,
        vector_store: Optional["VectorStore"] = None,
        fact_vector_store: Optional["FactVectorStore"] = None,
    ) -> MergeResult:
        """Promote a set of operator-private crystals into one team crystal.

        Validates the sources (exist, belong to team_id, operator-owned, >= 2
        distinct operators), picks the survivor, chgrp/chmod's it to the
        team, supersedes the rest, and records contributor provenance +
        reserved shares. Raises PromotionError on an invalid request.
        """
        sources: list["Crystal"] = []
        seen: set[str] = set()
        for cid in source_crystal_ids:
            if cid in seen:
                continue
            seen.add(cid)
            c = await self._store.get_crystal(cid)
            if c is None:
                raise PromotionError(f"source crystal {cid!r} not found")
            if c.customer_id != team_id:
                raise PromotionError(
                    f"source crystal {cid!r} does not belong to team "
                    f"{team_id!r}"
                )
            if c.owner_operator_id is None:
                raise PromotionError(
                    f"source crystal {cid!r} is not operator-owned; only "
                    f"operator-private crystals promote to the team tier"
                )
            sources.append(c)

        if len(sources) < 2:
            raise PromotionError(
                "promotion needs at least two distinct source crystals"
            )
        if len({c.owner_operator_id for c in sources}) < 2:
            raise PromotionError(
                "promotion needs sources from at least two distinct "
                "operators (a single operator's duplicates are dedup, not "
                "promotion)"
            )

        survivor = self._pick_survivor(sources)
        superseded = [c for c in sources if c.id != survivor.id]

        # Reserved credit shares — equal split across source crystals,
        # captured per source (incl. the survivor's own original).
        shares = self._equal_shares(len(sources))
        contributions = [
            {
                "source_crystal_id": c.id,
                "contributor_operator_id": c.owner_operator_id,
                "share_basis_points": shares[i],
            }
            for i, c in enumerate(sources)
        ]

        # chgrp/chmod the survivor up to the team tier. Tenancy
        # (customer_id) is unchanged; ownership moves from the operator to
        # the team (owner cleared), group set to the team, mode opened to
        # group-read.
        survivor.owner_operator_id = None
        survivor.group_team_id = team_id
        survivor.mode = 0o640
        await self._store.upsert_crystal(survivor)

        # Record provenance BEFORE deleting the non-survivors (their ids
        # live on in the rows as historical references).
        await self._store.record_promotion_contributions(
            survivor.id, contributions
        )

        # Supersede the rest (delete crystal + its facts), tenancy-scoped.
        for c in superseded:
            await self._store.delete_crystal(
                c.id,
                customer_id=team_id,
                vector_store=vector_store,
                fact_vector_store=fact_vector_store,
            )

        # The survivor's group/owner/mode changed; refresh routing caches so
        # the team-tier view is consistent (delete_crystal already
        # invalidated for the superseded ones).
        if vector_store is not None:
            vector_store.invalidate(team_id)
        if fact_vector_store is not None:
            fact_vector_store.invalidate(team_id)

        logger.info(
            "promotion.merged",
            team_id=team_id,
            merged_crystal_id=survivor.id,
            superseded=len(superseded),
            contributors=len(contributions),
        )
        return MergeResult(
            merged_crystal_id=survivor.id,
            superseded_crystal_ids=[c.id for c in superseded],
            contributions=contributions,
        )
