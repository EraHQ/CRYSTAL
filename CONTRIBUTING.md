# Contributing to Crystal

Crystal is built in the open by a solo founder who wants company. The
private repo carries the hosted platform (billing, multi-tenant ops);
everything in this repo is the complete product, self-hostable and
inspectable. Help of every size is welcome, from a typo fix to claiming
a research question.

## Where help is wanted

**Research questions.** Crystal maintains an open research agenda
(retrieval ranking, trust-tier signaling, conflict detection, and
more). Each item is a real, measurable question about how well this
memory system works, with a metric attached. Claiming one means
running the experiment and publishing what you find, good or bad. Open
an issue titled `research: <item>` to claim one.

**Good first issues.** Issues labeled `good-first-issue` are small,
scoped, and come with file pointers. They are real defects and real
polish, not busywork.

**Live findings.** Crystal's own culture is the stranger test: use the
product cold and report where it fails you. A reproducible "this recall
should have returned X and returned Y" report is among the most
valuable contributions possible for a memory product.

## Ground rules

- Read the source before proposing a design. Evidence over theory.
- Every change ships with tests. The suite is the contract: `pytest -q`
  for the backend, `npm test` then `npm run build` for the frontend.
- Small PRs land fast. Large changes start as an issue describing the
  approach, so design gets discussed before code exists.
- Honesty about weaknesses is a feature here. If your change reveals
  something broken, say so loudly; it will be received as a gift.

## Running it

Crystal self-hosts with Docker. The backend is Python/FastAPI with
PostgreSQL and Qdrant; the Inspector console is React/TypeScript/Vite.
See the README for setup. The MCP server means any MCP client (Claude,
Cursor, and others) can connect to your local instance and use it as
memory.

## The shape of the project

One person ratifies design decisions today, which is why the project
moves fast. Contributions do not require ceremony: open an issue,
discuss the approach in a few sentences, build it with tests, and it
gets reviewed quickly. As the contributor base grows, governance grows
with it, deliberately and slowly.
