# Crystal Roadmap

The public timeline. Updated as things ship, honestly, including what
broke on the way. Crystal is built in the open by one founder who
intends for that to change.

## Shipped

- Self-curating memory core: crystals, facts, quality tiers, recall
  gating, conflict detection, knowledge-gap tracking, autonomous
  research behind human review
- MCP server: any MCP client (Claude, Cursor, Windsurf, ChatGPT, and
  others) connects to the same memory
- OAuth sign-in for Claude's "Add custom connector": paste the URL,
  sign in, connected (September 2026, with six systemic defects found
  and fixed live on the way)
- Self-serve signup, onboarding wizard, free tier, and billing on the
  hosted platform
- Document ingestion with honest provenance (`document_extraction`
  versus verbatim chunks versus the model's own reasoning)

## Now

- Launch: directory listings, live billing, launch assets, Product
  Hunt
- Natural memory use: tuning so connected models recall and store
  without being prompted (server instructions shipped; skill
  published in `skills/crystal/`)
- The open research agenda (see RESEARCH_AGENDA.md) and first
  community issues

## Next

- Issues cleanup and a considered refactor pass
- Local crystals: a local-only lane for sensitive data
- Third-party data connections through the same curation
- Team seats and invites
- Token revocation UI and account management polish
- The cognition architecture research line (score-transparent
  retrieval and beyond)

## How this project makes decisions

Design decisions get ratified explicitly and recorded. Defects found
live get fixed and pinned with tests so they cannot return. Weak spots
are published, not hidden; the research agenda is a list of ways this
product might be failing, maintained by the people building it.
