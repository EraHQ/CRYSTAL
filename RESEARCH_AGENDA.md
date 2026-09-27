# Crystal Open Research Agenda

Crystal is a self-curating memory system, and every claim in that
sentence is measurable. This is the standing list of open questions
about how well it actually works, each with a method and a metric.
Anyone can claim one: open an issue titled `research: <item>`, run the
experiment against a self-hosted instance, and publish what you find.
Negative results are first-class results here.

The first three are ordered deliberately: item 5 is a trust-integrity
question that makes every other quality measurement suspect until
answered, item 4 takes minutes and probes the surface customers touch
first, and item 1 already has a reproducible failing example.

## 5. Gated-fact leakage audit (priority 1)

Crystals born from autonomous inference are gated out of recall until
approved. Question: do gated facts leak through any retrieval tool, and
can one climb to the trusted tier without human approval? Method: seed
gated crystals, count surfacing per retrieval tool, trace promotion
paths. Metric: leak count per tool; unauthorized promotions found.

## 4. Tier signal on the MCP surface (priority 2)

Quality tiers are supposed to reach the model as an epistemic signal.
Question: do tiers reach MCP recall at all, and when they do, does the
model hedge and cite differently? Method: confirm the field on the
wire, then A/B responses with and without tier notes. Metric: presence
on the wire; hedging and citation behavior deltas.

## 1. Entity dominance in recall ranking (priority 3)

A term every crystal shares, like the account owner's name, may drown
out topical terms. A reproducible failure exists: on a 262-crystal
bank, "what is Anthony working on right now" ranked an identity
crystal above the crystals that list projects. Method: build an eval
set from queries of this shape. Metric: MRR@5.

## 2. Fact granularity at write time

Bundled multi-claim facts versus one-claim atomic facts, same
knowledge, two banks. Metric: recall@k and precision@k.

## 3. Temporal and current-state recall

Store dated status facts, write superseding updates, ask "right now"
questions. Metric: accuracy on LongMemEval-style temporal and update
questions.

## 6. Grounding threshold calibration

Hand-label about 200 claim/source pairs and sweep the cosine threshold
(currently 0.25) for the best precision/recall point.

## 7. Trust continuity across re-ingestion

Re-ingesting an unchanged document currently resets tier age and
orphans citation history. Measure the loss, then prototype
fingerprint-based cluster identity that carries history across
replacement.

## 8. Conflict detection sensitivity

Inject contradictions of graded subtlety. Metric: detection rate,
false positives, time to surface.

## 9. Score-transparent retrieval (H1)

The flagship hypothesis: a gateway model that sees the normalized
retrieval score distribution and can expand candidates on demand beats
blind top-k injection on multi-hop and ambiguous-entity queries at an
equal token budget.

---

Findings, positive or negative, get linked here with credit. The
product's own defects found through this agenda get fixed in the open;
the provenance mislabeling fixed in September 2026 was found exactly
this way, by using the product cold and reading what it wrote.
