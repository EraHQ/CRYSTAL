"""Substrate review CLI — Phase 10.5 (D-MCR-13 V1).

Operator-facing CLI for reviewing deferred substrate_observation
action items. Per MCR §11 Q7: "CLI initially, then admin
dashboard." This is the CLI.

Usage:
  python -m crystal_cache.cli.substrate_review [--customer-id ID]
                                               [--limit N]
                                               [--since ISO]

Examples:
  # Cross-tenant view of the most recent 50 observations
  python -m crystal_cache.cli.substrate_review

  # One customer's observations
  python -m crystal_cache.cli.substrate_review --customer-id cust_123

  # Recent only (after a specific timestamp)
  python -m crystal_cache.cli.substrate_review --since 2026-05-20T00:00:00

The CLI is a thin print wrapper around
`metacognition.substrate_review.list_substrate_observations`. The
library function is the authoritative composition logic; the CLI
+ HTTP endpoint both consume it.

Per MCR Principle 9 and D-MCR-15: this surface is READ-ONLY.
Observations get displayed for human review. The CLI does NOT
provide mechanisms to act on observations or modify the harness.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from typing import Optional

from ..infrastructure import MetadataStore
from ..metacognition.substrate_review import (
    SubstrateObservationView,
    list_substrate_observations,
)


def _format_observation(view: SubstrateObservationView, index: int) -> str:
    """Format one observation view as a human-readable text block.

    Block shape:
      ===== Observation N =====
      ID:          <action_item_id>
      Created:     <iso timestamp>
      Customer:    <customer_id>
      Critic:      <role> (<model>)
      Confidence:  <critic_confidence>
      Action type: substrate_observation (always)
      Status:      deferred (always)

      [Critique context]
      Critique ID:  <id>
      Trace ID:     <trace_id or "(none)">
      Sequence:     <sequence_id>:<turn_index> or "(none)"
      Summary:      <critique.summary_text>

      [Observation content]
      <action_item.content as JSON-ish>

      [Trace context]
      Trace ID:     <trace_id or "(critique not found)">
      Sequence:     <sequence_id>:<turn_index>
      Created:      <iso timestamp>
    """
    lines: list[str] = []
    item = view.action_item
    critique = view.critique
    trace = view.trace_summary

    lines.append(f"===== Observation {index} =====")
    lines.append(f"ID:          {item.id}")
    lines.append(f"Created:     {item.created_at.isoformat()}")
    lines.append(f"Customer:    {item.customer_id}")
    if critique is not None:
        lines.append(
            f"Critic:      {critique.critic_role} ({critique.critic_model})"
        )
    else:
        lines.append("Critic:      (critique not found)")
    if item.critic_confidence is not None:
        lines.append(f"Confidence:  {item.critic_confidence:.2f}")
    lines.append("Action type: substrate_observation")
    lines.append("Status:      deferred")

    lines.append("")
    lines.append("[Critique context]")
    if critique is not None:
        lines.append(f"Critique ID: {critique.id}")
        lines.append(
            f"Trace ID:    {critique.trace_id or '(no hard trace_id; soft-joined)'}"
        )
        seq = critique.sequence_id or "(none)"
        turn = critique.turn_index if critique.turn_index is not None else "?"
        lines.append(f"Sequence:    {seq}:{turn}")
        if critique.summary_text:
            lines.append(f"Summary:     {critique.summary_text}")
    else:
        lines.append("(critique row missing — orphaned action item)")

    lines.append("")
    lines.append("[Observation content]")
    # ActionItem.content is a JSON-ish dict; render its keys briefly.
    if item.content:
        for k, v in item.content.items():
            lines.append(f"  {k}: {v}")
    else:
        lines.append("  (empty content)")

    lines.append("")
    lines.append("[Trace context]")
    if trace is not None:
        lines.append(f"Trace ID:   {trace.trace_id}")
        seq = trace.sequence_id or "(none)"
        turn = trace.turn_index if trace.turn_index is not None else "?"
        lines.append(f"Sequence:   {seq}:{turn}")
        lines.append(f"Created:    {trace.created_at.isoformat()}")
    else:
        lines.append("(trace not resolved — critique may have no trace_id)")

    return "\n".join(lines)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Build the argparse parser and parse CLI args."""
    parser = argparse.ArgumentParser(
        prog="crystal_cache.cli.substrate_review",
        description=(
            "Review deferred substrate_observation action items for "
            "human inspection (D-MCR-13 V1)."
        ),
    )
    parser.add_argument(
        "--customer-id",
        type=str,
        default=None,
        help=(
            "Scope to one customer's substrate observations. "
            "Omit for cross-tenant (system-wide) view."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max observations to display (default: 50).",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help=(
            "ISO 8601 datetime; only show observations with "
            "created_at >= this value."
        ),
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    """Execute the substrate-review CLI. Returns process exit code.

    Constructs a MetadataStore directly (no FastAPI lifespan
    needed); the store wraps an aiosqlite/asyncpg engine via
    settings. Closes the store cleanly on exit.
    """
    parsed_since: Optional[datetime] = None
    if args.since is not None:
        try:
            parsed_since = datetime.fromisoformat(args.since)
        except ValueError:
            print(
                f"Error: --since={args.since!r} is not a valid ISO "
                "8601 datetime.",
                file=sys.stderr,
            )
            return 2

    store = MetadataStore()
    try:
        views = await list_substrate_observations(
            store=store,
            customer_id=args.customer_id,
            since=parsed_since,
            limit=args.limit,
        )
    finally:
        await store.dispose()

    # Header.
    scope = (
        f"customer={args.customer_id}"
        if args.customer_id is not None
        else "cross-tenant"
    )
    print(
        f"Substrate review — {scope}, {len(views)} observation(s) "
        f"(limit={args.limit})"
    )
    print()

    if not views:
        print("No deferred substrate observations found.")
        return 0

    for i, view in enumerate(views, start=1):
        print(_format_observation(view, i))
        print()

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Used by `python -m crystal_cache.cli.substrate_review`."""
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
