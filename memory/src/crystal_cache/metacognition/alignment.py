"""Item-alignment classifier — Phase 10A v1 algorithm per P0.73.

Per `docs/MCR_FRAMEWORK.md` §4.4 and §6: the metacognitive layer
classifies how each pair of action items (across critics for the same
trace) relates. The four classes:

  same_action          — substantively the same proposal
  similar_action       — same action_type with similar content
  divergent_action     — different action_type, OR same type with
                         unrelated content focus
  contradictory_action — same action_type proposing opposed changes
                         (Phase 10A detects this for edit_proposal only)

Phase 10A v1 is a DUMB POLICY per MCR §5.3 ("Initial implementation
can be a dumb policy"). The algorithm:

  1. action_type mismatch → divergent_action.
  2. Same action_type AND canonical-content-key match:
     → check for contradictions (edit_proposal contradiction rule)
     → same_action if compatible, contradictory_action if opposed.
  3. Same action_type AND key is similar-but-not-identical →
     similar_action.
  4. Same action_type AND completely different content focus →
     divergent_action.

Canonical content keys per action_type (P0.73):
  research_task         → topic
  verification_task     → crystal_id
  evidence_gathering    → topic
  gap_declaration       → want
  edit_proposal         → crystal_id
  substrate_observation → subsystem
  escalation            → issue

Phase 10B or 11.5 may refine with semantic similarity (embedding
compare) or per-action-type contradiction rules. The pure-function
shape makes the algorithm testable without DB fixtures and swappable
for a smarter v2.
"""
from __future__ import annotations

from typing import Any, Optional

from ..models.action_item import ActionItem
from ..models.item_alignment import AlignmentClass


# Map from action_type to the field within `content` that uniquely
# identifies the action's focus. Used by `_canonical_key` below to
# produce a normalized comparison string. Verified against MCR §4.3
# + Phase 9A's _SELF_CRITIQUE_SYSTEM_PROMPT vocabulary.
_CANONICAL_KEY_FIELD: dict[str, str] = {
    "research_task": "topic",
    "verification_task": "crystal_id",
    "evidence_gathering": "topic",
    "gap_declaration": "want",
    "edit_proposal": "crystal_id",
    "substrate_observation": "subsystem",
    "escalation": "issue",
}


def _canonical_key(item: ActionItem) -> str:
    """Extract the canonical content key for an action item, normalized.

    Returns the field value per `_CANONICAL_KEY_FIELD`, lowercased
    and whitespace-stripped. Empty string when the field is missing
    or non-string (the classifier treats empty as "no key" → cannot
    match by key).
    """
    field = _CANONICAL_KEY_FIELD.get(item.action_type)
    if field is None:
        return ""
    raw = item.content.get(field) if item.content else None
    if not isinstance(raw, str):
        return ""
    return raw.strip().lower()


def _are_keys_similar(key_a: str, key_b: str) -> bool:
    """Decide whether two canonical keys are 'similar but not identical'.

    Phase 10A v1 heuristic: keys share at least one non-trivial
    whitespace-separated token (length >= 4 chars, to skip
    stopwords/articles). Examples:
      "fiscal year deadline" vs "deadline for fiscal year" → similar
      "deadline" vs "schedule"                              → not similar

    Phase 10B or 11.5 may swap this for semantic similarity. Per
    P0.73, the rule is intentionally narrow.
    """
    if not key_a or not key_b:
        return False
    if key_a == key_b:
        return False  # caller checks `==` separately for same_action
    tokens_a = {t for t in key_a.split() if len(t) >= 4}
    tokens_b = {t for t in key_b.split() if len(t) >= 4}
    return bool(tokens_a & tokens_b)


def _is_edit_proposal_contradiction(
    item_a: ActionItem, item_b: ActionItem
) -> bool:
    """Detect the one contradiction pattern Phase 10A handles (P0.73).

    Two edit_proposal items with the SAME `crystal_id` but DIFFERENT
    `proposed_change` strings → contradictory. The classifier punts
    other potential contradictions to similar/same per the v1 scope.
    """
    if item_a.action_type != "edit_proposal":
        return False
    if item_b.action_type != "edit_proposal":
        return False
    crystal_a = item_a.content.get("crystal_id") if item_a.content else None
    crystal_b = item_b.content.get("crystal_id") if item_b.content else None
    if not crystal_a or crystal_a != crystal_b:
        return False
    change_a = item_a.content.get("proposed_change") if item_a.content else None
    change_b = item_b.content.get("proposed_change") if item_b.content else None
    # Both must be non-empty strings AND different to count as a
    # contradiction. If either is missing, fall back to non-
    # contradictory (the classifier treats missing data as
    # "compatible" rather than "opposed").
    if not isinstance(change_a, str) or not isinstance(change_b, str):
        return False
    if not change_a.strip() or not change_b.strip():
        return False
    return change_a.strip().lower() != change_b.strip().lower()


def classify_pair(item_a: ActionItem, item_b: ActionItem) -> AlignmentClass:
    """Classify the relationship between two action items.

    Per P0.73 v1 dumb policy. The function is PURE — no DB access, no
    side effects. Output is deterministic in (item_a, item_b).

    Order-symmetry: classify_pair(A, B) == classify_pair(B, A). The
    rules don't depend on argument order; tests verify symmetry.

    Args:
        item_a: one action item.
        item_b: another action item from a DIFFERENT critic (caller
            ensures this — comparing two items from the same critic
            would mean comparing within a critique, which §4.4
            doesn't define).

    Returns:
        One of the four AlignmentClass values.
    """
    # Rule 1: action_type mismatch → divergent.
    if item_a.action_type != item_b.action_type:
        return "divergent_action"

    # Rules 2-3-4: same action_type. Compare canonical keys.
    key_a = _canonical_key(item_a)
    key_b = _canonical_key(item_b)

    # Same canonical key — possible same_action or contradiction.
    if key_a and key_b and key_a == key_b:
        if _is_edit_proposal_contradiction(item_a, item_b):
            return "contradictory_action"
        return "same_action"

    # Different canonical keys but the same action_type.
    if _are_keys_similar(key_a, key_b):
        return "similar_action"

    # Same action_type but no canonical-key overlap — different
    # content focus.
    return "divergent_action"
