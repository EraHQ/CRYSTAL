"""The Mind tab (2026-10-21): the whole memory as one scene.

Anthony: "establish a structure for the consciousness we have built for
the whole system: knowledge, assumptions, duplicates, contradictions.
Basically if you could look at a human brain and see each segment,
thought etc."

`layout.py` turns the bank's fact vectors into regions and positions
(pure numpy, deterministic); `endpoints/mind.py` serves everything the
scene draws in one read.
"""
from .layout import MindLayout, layout_facts

__all__ = ["MindLayout", "layout_facts"]
