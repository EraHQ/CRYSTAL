#!/usr/bin/env python
"""Live P3 proof + calibration harness - citation grounding (Growth G1/G4, CC-D11/CC-D13).

P3's core is the grounding cosine (no model): ground_sources_against_answer
re-encodes each surfaced crystal's text and the answer in the REAL semantic
encoder's native space and credits the ones that clear the threshold. The unit
suite proves the orchestration (record / credit / gap) with a controlled
encoder; this script proves the part the unit tests can't - that the REAL
encoder discriminates relevant surfaced crystals from irrelevant ones - and
serves as the calibration harness for the answer-level threshold.

It runs at the ACTUAL agent threshold (CC_AGENT_CITATION_GROUNDING_THRESHOLD,
default 0.60 per CC-D13), so the GROUNDED/rejected marks match what the agent
does. gtr-t5-base's cosine floor for unrelated text is ~0.5 (anisotropy), which
is why the proxy's claim-span 0.25 is wrong here and a separate, higher
answer-level threshold is used.

No server, no model, no seeding. First run loads gtr-t5-base (~440MB).

RUN
  python scripts/smoke_p3_citations.py
  CC_AGENT_CITATION_GROUNDING_THRESHOLD=0.65 python scripts/smoke_p3_citations.py   # to retune
"""
from __future__ import annotations

import asyncio
import os
import sys

# The encoder factory reads CC_TEXT_ENCODER; force semantic (grounding needs
# the 768-dim native space) BEFORE importing anything that builds Settings.
os.environ["CC_TEXT_ENCODER"] = "semantic"

from crystal_cache.config import get_settings
from crystal_cache.retrieval.citations import CitationSource
from crystal_cache.retrieval.citation_grounding import ground_sources_against_answer


# Each scenario: an answer the agent produced, plus candidate surfaced crystals
# (label, source_text) spanning clearly-relevant -> borderline -> irrelevant.
_SCENARIOS = [
    {
        "answer": (
            "The Aurora Protocol was ratified in 2019 to standardize "
            "cross-border data exchange between member states."
        ),
        "candidates": [
            ("relevant",
             "Aurora Protocol: ratified 2019; establishes standards for "
             "cross-border data exchange."),
            ("borderline",
             "The Aurora Protocol went through three draft revisions before "
             "its approval."),
            ("irrelevant",
             "Emperor penguins huddle together in dense groups to survive the "
             "Antarctic winter."),
        ],
    },
    {
        "answer": (
            "To rotate the access token, call POST /v1/auth/refresh with the "
            "refresh token; the old access token is revoked immediately."
        ),
        "candidates": [
            ("relevant",
             "Token rotation: POST /v1/auth/refresh with the refresh token "
             "revokes the prior access token."),
            ("irrelevant",
             "The marketing team's Q3 offsite is scheduled for the second week "
             "of September."),
        ],
    },
]


async def _run() -> None:
    print(f"Loading the semantic encoder (CC_TEXT_ENCODER={os.environ['CC_TEXT_ENCODER']}) ...")
    try:
        from crystal_cache.encoding import build_text_encoder
        encoder = build_text_encoder()
    except Exception as e:  # noqa: BLE001
        print(f"ERROR building the semantic encoder: {e}", file=sys.stderr)
        print("(Needs sentence-transformers + the gtr-t5-base model available, "
              "same as the server's CC_TEXT_ENCODER=semantic.)", file=sys.stderr)
        raise SystemExit(2)

    threshold = get_settings().agent_citation_grounding_threshold
    print(f"Agent grounding threshold (CC_AGENT_CITATION_GROUNDING_THRESHOLD): {threshold}\n")

    for i, sc in enumerate(_SCENARIOS, start=1):
        answer = sc["answer"]
        sources = [
            (CitationSource(handle=str(j), crystal_id=f"cry_{i}_{j}"), text)
            for j, (_label, text) in enumerate(sc["candidates"], start=1)
        ]
        results = await ground_sources_against_answer(
            encoder, answer, sources, threshold=threshold,
        )
        print(f"[Scenario {i}] answer: {answer}")
        for (label, _text), r in zip(sc["candidates"], results):
            mark = "GROUNDED " if r["grounded"] else "rejected "
            print(f"    {mark} score={r['grounding_score']:.3f}  ({label})")
        print()

    print(f"EXPECT at {threshold}: 'relevant' + 'borderline' GROUNDED, 'irrelevant' rejected.")
    print("A 'borderline' (topically related but not claim-supporting) grounding is")
    print("Option B's known tradeoff (credits topical, not claim-specific). If the")
    print("split looks wrong, retune CC_AGENT_CITATION_GROUNDING_THRESHOLD and re-run.")


if __name__ == "__main__":
    asyncio.run(_run())
