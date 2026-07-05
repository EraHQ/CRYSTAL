"""Command-line tools for Crystal Cache v2.

This package houses operator-facing CLI utilities that supplement
the HTTP surface. The first inhabitant is the substrate review
CLI (Phase 10.5, MCR §11 Q7's "CLI initially") — humans can run
it to see what the agent has flagged about its own surrounding
system.

Future CLI tools may land here as operational needs surface. The
package boundary is for human-invoked tools; scheduled processes
live in `workers/`, HTTP handlers live in `endpoints/`.

Usage:
  python -m crystal_cache.cli.substrate_review [--customer-id ID]
                                               [--limit N]
                                               [--since ISO]
"""
