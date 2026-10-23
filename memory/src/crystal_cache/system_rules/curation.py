"""The 'curation' rule type: an ordered allow/deny list over everything
waiting for a decision (Q101, 2026-10-23).

Anthony: "there is no way to set rules. It should be easy like criteria
met then approve fact. Or I could exclude on x facts from automatically
whitelisting. Think of a networking allow and deny rules."

One rule type now covers what 'promotion' (gated memory) and 'approval'
(assumptions and consequences) did separately. Rules run top to bottom
over every candidate; the first rule that MATCHES decides and nothing
below it is consulted. Allow approves the candidate (an assumption or
consequence is approved; a fact is let into recall). Deny holds it for a
person. The implicit last line of every list is "otherwise a person
decides": with no rule, or no matching rule, nothing is approved by the
system.

A rule matches when its selector fits the candidate AND every condition
it names holds. A condition that cannot be evaluated for a candidate (a
fact has no agreement edges, an assumption has no scan verdict) does not
hold, so the rule falls through to the next line.

Shape (STRICT; unknown keys reject):

  selector (what the rule applies to):
    kind         "fact" | "assumption" | "consequence" | "any"   [required]
    key_prefix   str    any of its keys starts with this (case-insensitive)
    source       "document" | "chat" | "research" | "reasoning"
    tier         "quarantine" | "neutral" | "whitelist"  (its current tier)

  conditions (ALL must hold; each optional):
    min_agreeing_facts   int >= 1   stated facts with a consistent_with edge
    no_contradictions    bool       no contradicted_by edge and no open conflict
    min_citations        int >= 1   grounded citations
    parent_trusted       bool       its parent can be recalled and is neutral or whitelist
    min_confidence       float 0..1 the assumption's confidence
    waited_days          int >= 0   age in days
    scan_passed          bool       the outbound review scan passed

  action:
    decision     "allow" | "deny"                                 [required]
    tier         "keep" | "neutral"  (allow only; facts only: an approved
                 assumption takes its parent's standing)

Safety: a rule only ever approves or holds. It never invalidates, never
blacklists, never whitelists (whitelist is earned through use or set by a
person), and a blacklisted candidate is never considered. Every allow is
audited on the rule and on the curation feed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from .model import (
    RuleValidationError,
    SystemRule,
    require_keys,
    require_type,
)
from .registry import register_rule_type

logger = structlog.get_logger(__name__)

RULE_TYPE = "curation"
KINDS = ("fact", "assumption", "consequence", "any")
SOURCES = ("document", "chat", "research", "reasoning")
TIERS = ("quarantine", "neutral", "whitelist")
DECISIONS = ("allow", "deny")
ALLOW_TIERS = ("keep", "neutral")
DEFAULT_LINE = "Otherwise, a person decides."

_SELECTOR_KEYS = {"kind", "key_prefix", "source", "tier"}
_CONDITION_KEYS = {
    "min_agreeing_facts", "no_contradictions", "min_citations", "parent_trusted",
    "min_confidence", "waited_days", "scan_passed",
}
_ACTION_KEYS = {"decision", "tier"}
_SCAN_TAG = "outbound_scan_passed"
_DOCUMENT_SOURCE_KINDS = {"document_chunk", "document_extraction"}
_REASONING_ORIGINS = {"assumptions", "implication"}


# ---------------------------------------------------------------------------
# Validation (strict)
# ---------------------------------------------------------------------------

def _one_of(value: Any, allowed: tuple, *, key: str, where: str) -> None:
    require_type(value, str, key=key, where=where)
    if value not in allowed:
        raise RuleValidationError(f"{where}: {key} must be one of {list(allowed)}; got {value!r}")


def validate(rule: SystemRule) -> None:
    sel, cond, act = rule.selector, rule.conditions, rule.action

    require_keys(sel, _SELECTOR_KEYS, where="curation.selector")
    if "kind" not in sel:
        raise RuleValidationError("curation.selector: 'kind' is required (fact | assumption | consequence | any)")
    _one_of(sel["kind"], KINDS, key="kind", where="curation.selector")
    if "key_prefix" in sel:
        require_type(sel["key_prefix"], str, key="key_prefix", where="curation.selector")
        if not sel["key_prefix"].strip() or len(sel["key_prefix"]) > 120:
            raise RuleValidationError("curation.selector: key_prefix must be 1 to 120 characters")
    if "source" in sel:
        _one_of(sel["source"], SOURCES, key="source", where="curation.selector")
    if "tier" in sel:
        _one_of(sel["tier"], TIERS, key="tier", where="curation.selector")

    require_keys(cond, _CONDITION_KEYS, where="curation.conditions")
    for key, floor in (("min_agreeing_facts", 1), ("min_citations", 1), ("waited_days", 0)):
        if key in cond:
            require_type(cond[key], int, key=key, where="curation.conditions")
            if cond[key] < floor:
                raise RuleValidationError(f"curation.conditions: {key} must be >= {floor}")
    for key in ("no_contradictions", "parent_trusted", "scan_passed"):
        if key in cond:
            require_type(cond[key], bool, key=key, where="curation.conditions")
    if "min_confidence" in cond:
        v = cond["min_confidence"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 1.0):
            raise RuleValidationError("curation.conditions: min_confidence must be a number from 0 to 1")

    require_keys(act, _ACTION_KEYS, where="curation.action")
    if "decision" not in act:
        raise RuleValidationError("curation.action: 'decision' is required (allow | deny)")
    _one_of(act["decision"], DECISIONS, key="decision", where="curation.action")
    if "tier" in act:
        _one_of(act["tier"], ALLOW_TIERS, key="tier", where="curation.action")
        if act["decision"] != "allow":
            raise RuleValidationError("curation.action: tier only applies to an allow rule")


# ---------------------------------------------------------------------------
# Candidates: everything waiting for a decision, in one shape
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """One thing waiting for a decision: a recall-gated fact crystal, or an
    assumption / consequence nobody has approved or invalidated."""
    id: str
    customer_id: str
    kind: str                      # fact | assumption | consequence
    source: str                    # document | chat | research | reasoning
    quality_tier: str
    created_at: Optional[datetime]
    diagnostic_tags: list[str] = field(default_factory=list)
    parent_crystal_id: Optional[str] = None
    label: str = ""
    keys: list[str] = field(default_factory=list)
    _memo: dict[str, Any] = field(default_factory=dict, repr=False)


def source_of(*, origin: str, source_kind: str, source_document_id: Optional[str]) -> str:
    if origin == "background_worker":
        return "research"
    if origin in _REASONING_ORIGINS:
        return "reasoning"
    if source_kind in _DOCUMENT_SOURCE_KINDS or source_document_id:
        return "document"
    return "chat"


def _parse_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


async def list_candidates(store, customer_id: str) -> list[Candidate]:
    """Gated fact crystals (never blacklisted ones) plus the assumptions
    and consequences still waiting. Assumption crystals come only from the
    assumption read, so a gated consequence is never counted twice."""
    from ..infrastructure.metadata_store_assumption_ext import (
        ASSUMPTION_CRYSTAL_TYPE, CONSEQUENCE_ORIGIN,
    )

    out: list[Candidate] = []
    for c in await store.list_recall_gated_crystals(customer_id):
        if c.crystal_type == ASSUMPTION_CRYSTAL_TYPE or c.quality_tier == "blacklist":
            continue
        out.append(Candidate(
            id=c.id, customer_id=customer_id, kind="fact",
            source=source_of(origin=c.origin, source_kind=c.source_kind,
                             source_document_id=c.source_document_id),
            quality_tier=c.quality_tier, created_at=_parse_dt(c.created_at),
            diagnostic_tags=list(c.diagnostic_tags or []), parent_crystal_id=c.parent_crystal_id,
            label=(c.summary_text or "").strip(),
        ))
    for r in await store.list_unapproved_assumptions(customer_id):
        kind = r.get("kind") or "assumption"
        out.append(Candidate(
            id=r["id"], customer_id=customer_id, kind=kind,
            source="reasoning", quality_tier=r.get("quality_tier") or "quarantine",
            created_at=_parse_dt(r.get("created_at")),
            diagnostic_tags=list(r.get("diagnostic_tags") or []), parent_crystal_id=r.get("parent_crystal_id"),
            label=(r.get("statement") or "").strip(),
        ))
    # origin is implied for assumption kinds; keep the constant referenced so
    # a rename of the consequence origin fails here, not silently.
    assert CONSEQUENCE_ORIGIN in _REASONING_ORIGINS
    return out


async def _attach_keys(store, candidates: list[Candidate]) -> None:
    keys = await store.fact_keys_for_crystals([c.id for c in candidates])
    for c in candidates:
        c.keys = list(keys.get(c.id, []))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def selects(rule: SystemRule, candidate: Candidate) -> bool:
    sel = rule.selector
    if sel["kind"] != "any" and candidate.kind != sel["kind"]:
        return False
    prefix = sel.get("key_prefix")
    if prefix is not None:
        p = prefix.strip().lower()
        if not any((k or "").lower().startswith(p) for k in candidate.keys):
            return False
    if "source" in sel and candidate.source != sel["source"]:
        return False
    if "tier" in sel and candidate.quality_tier != sel["tier"]:
        return False
    return True


async def _edges(store, candidate: Candidate) -> dict:
    if "edges" not in candidate._memo:
        if candidate.kind == "fact":
            candidate._memo["edges"] = {}
        else:
            candidate._memo["edges"] = await store.count_consequence_edges(candidate.id)
    return candidate._memo["edges"]


async def _open_conflicts(store, candidate: Candidate) -> int:
    if "conflicts" not in candidate._memo:
        candidate._memo["conflicts"] = await store.count_open_contradictions_for_crystal(
            candidate.customer_id, candidate.id,
        )
    return candidate._memo["conflicts"]


async def _citations(store, candidate: Candidate) -> int:
    if "citations" not in candidate._memo:
        candidate._memo["citations"] = await store.count_grounded_citations_for_crystal(
            candidate.customer_id, candidate.id,
        )
    return candidate._memo["citations"]


async def _parent_trusted(store, candidate: Candidate) -> bool:
    if "parent" not in candidate._memo:
        trusted = False
        if candidate.parent_crystal_id:
            parent = await store.get_crystal(candidate.parent_crystal_id)
            trusted = bool(
                parent is not None and parent.customer_id == candidate.customer_id
                and not parent.recall_gated and parent.quality_tier in ("neutral", "whitelist")
            )
        candidate._memo["parent"] = trusted
    return candidate._memo["parent"]


def _confidence(candidate: Candidate) -> Optional[float]:
    from ..infrastructure.metadata_store_assumption_ext import parse_assumption_tags
    if candidate.kind == "fact":
        return None
    return parse_assumption_tags(candidate.diagnostic_tags).get("confidence")


def _age_days(candidate: Candidate, now: datetime) -> Optional[int]:
    if candidate.created_at is None:
        return None
    return max(0, (now - candidate.created_at).days)


async def conditions_hold(rule: SystemRule, *, store, candidate: Candidate,
                          now: Optional[datetime] = None) -> bool:
    cond = rule.conditions
    now = now or datetime.now(timezone.utc)
    if "min_agreeing_facts" in cond:
        if (await _edges(store, candidate)).get("consistent_with", 0) < int(cond["min_agreeing_facts"]):
            return False
    if cond.get("no_contradictions"):
        if (await _edges(store, candidate)).get("contradicted_by", 0) > 0:
            return False
        if await _open_conflicts(store, candidate) > 0:
            return False
    if "min_citations" in cond:
        if await _citations(store, candidate) < int(cond["min_citations"]):
            return False
    if cond.get("parent_trusted"):
        if not await _parent_trusted(store, candidate):
            return False
    if "min_confidence" in cond:
        conf = _confidence(candidate)
        if conf is None or conf < float(cond["min_confidence"]):
            return False
    if "waited_days" in cond:
        age = _age_days(candidate, now)
        if age is None or age < int(cond["waited_days"]):
            return False
    if cond.get("scan_passed"):
        if _SCAN_TAG not in candidate.diagnostic_tags:
            return False
    return True


async def matches(rule: SystemRule, *, store, candidate: Candidate,
                  now: Optional[datetime] = None) -> bool:
    if not selects(rule, candidate):
        return False
    return await conditions_hold(rule, store=store, candidate=candidate, now=now)


async def decide(rules: list[SystemRule], candidate: Candidate, *, store,
                 now: Optional[datetime] = None) -> tuple[Optional[SystemRule], str]:
    """Walk the ordered list; the first matching rule decides. Returns
    (rule, "allow" | "deny") or (None, "person")."""
    for rule in rules:
        if rule.rule_type != RULE_TYPE or not rule.enabled:
            continue
        if await matches(rule, store=store, candidate=candidate, now=now):
            return rule, rule.action["decision"]
    return None, "person"


async def allow(rule: SystemRule, *, store, candidate: Candidate) -> bool:
    """Carry out an allow: approve an assumption or consequence (it takes
    its parent's standing), or let a fact into recall (optionally trusting
    it to neutral). Returns True when the bank changed."""
    cid = candidate.customer_id
    if candidate.kind in ("assumption", "consequence"):
        try:
            out = await store.approve_assumption_standing(cid, candidate.id, approved_by=f"rule:{rule.id}")
        except ValueError:
            return False
        event_type, verb = "assumption_approved", "Approved"
        payload = {"rule_id": rule.id, "rule": rule.name, "kind": candidate.kind,
                   "quality_tier": out.get("quality_tier")}
    else:
        if not await store.set_crystal_recall_gate(candidate.id, cid, gated=False):
            return False
        if rule.action.get("tier") == "neutral" and candidate.quality_tier == "quarantine":
            await store.set_crystal_quality_tier(candidate.id, cid, "neutral")
        event_type, verb = "memory_allowed", "Let into recall"
        payload = {"rule_id": rule.id, "rule": rule.name, "kind": "fact",
                   "trusted": rule.action.get("tier") == "neutral"}
    label = f"{verb} by rule {rule.name}"
    if candidate.label:
        label = f"{label} - {candidate.label}"
    try:
        await store.record_curation_event(cid, event_type=event_type, subject_id=candidate.id,
                                          label=label[:256], payload=payload)
    except Exception:  # noqa: BLE001 - the witness never breaks the act
        logger.debug("curation_event.emit_failed", exc_info=True)
    return True


async def apply(rule: SystemRule, *, store, candidate) -> bool:
    """registry contract: apply ONE rule to one candidate. Used by tests
    and by callers that already decided this rule is first to match; the
    list walk is run_curation_rules in store.py."""
    if not isinstance(candidate, Candidate):
        return False
    if not await matches(rule, store=store, candidate=candidate):
        return False
    if rule.action["decision"] != "allow":
        return False
    return await allow(rule, store=store, candidate=candidate)


# ---------------------------------------------------------------------------
# Legacy rule types fold into this one
# ---------------------------------------------------------------------------

def from_legacy(rule: SystemRule) -> Optional[SystemRule]:
    """The curation shape of a 'promotion' or 'approval' rule, or None for
    any other type. Same id, name, order, switch and audit."""
    sel, cond, act = rule.selector, rule.conditions, rule.action
    if rule.rule_type == "promotion":
        selector: dict[str, Any] = {"kind": "fact"}
        if sel.get("origin") == "background_worker":
            selector["source"] = "research"
        conditions: dict[str, Any] = {}
        if cond.get("outbound_scan_passed") is True:
            conditions["scan_passed"] = True
        if cond.get("no_open_conflicts") is True:
            conditions["no_contradictions"] = True
        if isinstance(cond.get("min_grounded_citations"), int) and cond["min_grounded_citations"] > 0:
            conditions["min_citations"] = cond["min_grounded_citations"]
        action: dict[str, Any] = {"decision": "allow"}
        if act.get("set_tier") == "neutral":
            action["tier"] = "neutral"
    elif rule.rule_type == "approval":
        selector = {"kind": sel.get("kind") or "consequence"}
        conditions = {"min_agreeing_facts": int(cond.get("min_consistent_edges") or 1)}
        if cond.get("no_contradicted_edges", True):
            conditions["no_contradictions"] = True
        action = {"decision": "allow"}
    else:
        return None
    return SystemRule(
        id=rule.id, customer_id=rule.customer_id, rule_type=RULE_TYPE, name=rule.name,
        selector=selector, conditions=conditions, action=action, enabled=rule.enabled,
        priority=rule.priority, last_fired_at=rule.last_fired_at, fire_count=rule.fire_count,
        created_at=rule.created_at, updated_at=rule.updated_at,
    )


# ---------------------------------------------------------------------------
# The rule in words (the console's ruleText says the same sentence)
# ---------------------------------------------------------------------------

_SUBJECT = {"fact": "a fact", "assumption": "an assumption", "consequence": "a consequence", "any": "anything"}
_SOURCE_WORDS = {"document": "from a document", "chat": "from chat", "research": "from research",
                 "reasoning": "found by reasoning"}


def _plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


def rule_text(rule: SystemRule) -> str:
    if rule.rule_type != RULE_TYPE:
        legacy = from_legacy(rule)
        return rule_text(legacy) if legacy else f"{rule.rule_type} rule."
    sel, cond, act = rule.selector, rule.conditions, rule.action
    kind = sel.get("kind", "any")
    verb = "Hold" if act.get("decision") == "deny" else ("Let in" if kind == "fact" else "Approve")
    parts = [verb, _SUBJECT.get(kind, "anything")]
    if sel.get("key_prefix"):
        parts.append(f'whose key starts with "{sel["key_prefix"]}"')
    if sel.get("source") in _SOURCE_WORDS:
        parts.append(_SOURCE_WORDS[sel["source"]])
    if sel.get("tier"):
        parts.append(f"in the {sel['tier']} tier")
    if act.get("decision") == "deny":
        parts.append("for a person")
    when: list[str] = []
    if "min_agreeing_facts" in cond:
        n = int(cond["min_agreeing_facts"])
        when.append(f"{n} stated {_plural(n, 'fact agrees', 'facts agree')} with it")
    if cond.get("no_contradictions"):
        when.append("nothing contradicts it")
    if "min_citations" in cond:
        n = int(cond["min_citations"])
        when.append(f"it has been cited at least {n} {_plural(n, 'time', 'times')}")
    if cond.get("parent_trusted"):
        when.append("its parent is trusted")
    if "min_confidence" in cond:
        when.append(f"its confidence is at least {round(float(cond['min_confidence']) * 100)}%")
    if "waited_days" in cond:
        n = int(cond["waited_days"])
        when.append(f"it has waited at least {n} {_plural(n, 'day', 'days')}")
    if cond.get("scan_passed"):
        when.append("the review scan passed")
    text = " ".join(parts)
    if when:
        text += " when " + (when[0] if len(when) == 1 else ", ".join(when[:-1]) + " and " + when[-1])
    if act.get("decision") == "allow" and act.get("tier") == "neutral":
        text += ", and trust it"
    return text + "."


register_rule_type(RULE_TYPE, validate, apply)
