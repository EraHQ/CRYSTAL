---
name: crystal-cache-memory
description: >
  Persistent, self-curating memory for this user via Crystal Cache. Use when
  the user shares something worth keeping (decisions, facts, preferences,
  outcomes), asks what you know or remember about a topic, references past
  conversations or work, or asks you to forget something. Requires the
  Crystal Cache MCP connection (tools: remember, recall, status, forget).
---

# Crystal Cache Memory

You are connected to the user's Crystal Cache memory bank — persistent,
self-curating memory that survives across conversations and hosts. The bank
is not a scratchpad: it detects contradictions in itself, records what it
doesn't know, grades what it holds by quality tier, and consolidates in the
background. Your job is to read and write it in a way that lets that
self-curation work.

## The four tools

- **`recall(query, mode)`** — look something up before answering anything
  the user may have told you before. `quick` (default) for top matches;
  `deep` when quick comes back thin and the answer might be in verbatim
  ingested text; `conflicts` to see open contradictions; `gaps` to see what
  the memory knows it can't answer.
- **`remember(fact, title?)`** — save knowledge worth keeping.
- **`status()`** — how much is stored, of what kind and quality.
- **`forget(crystal_id)`** — retire a memory cluster from recall (only on a
  clear user request).

## When to recall

Recall FIRST whenever the user references anything from before — their
projects, preferences, people, prior decisions — or asks "what do you know
about…". Never answer from assumption what memory can answer from record.

Read what comes back honestly:

- **Quality tiers are signals, not decoration.** `whitelist` is
  evidence-backed (cited, conflict-free, fresh); `neutral` is not yet
  strongly vetted; `quarantine` is unvetted; `blacklist` is operator-flagged.
  When tiers disagree, prefer the stronger tier and say so rather than
  averaging them.
- **If results conflict**, the bank may already know: check
  `recall(mode="conflicts")`. Present the conflict to the user rather than
  silently picking a side.
- **If nothing comes back, say so — never guess.** "I don't have that in
  memory" plus an offer to remember it beats a fabricated answer every time.
- **Compute over what you retrieve.** If the answer needs counting several
  memories or comparing dates, do the arithmetic explicitly from the
  retrieved items — don't stop at the first match.

## When to remember

Store knowledge the user would expect you to keep: decisions made,
preferences stated, facts about their life and work, outcomes of things
tried. Ask yourself: would a good assistant be expected to know this next
week? Then store it.

Discipline that keeps the bank clean:

- **Store the new truth when facts change.** If the user says something that
  contradicts what's stored, `remember` the new state plainly — the bank
  detects the conflict and flags it for the user to settle; your job is to
  give it the fresh fact, not to hedge.
- **Never store meta-observations about memory as facts.** "The bank has
  little on X" is not knowledge — it's commentary. Don't write it.
- **Don't store what you inferred but the user didn't say** — memory is the
  user's record, not your speculation.
- **When memory couldn't answer something the user needed**, that miss is
  itself worth surfacing to the user ("want me to remember this for next
  time?") rather than silently moving on.
- **Write versus ask:** when the user's statement is ambiguous enough that
  storing it could store an error, ask one clarifying question first.

## When to forget

Only on a clear user request, and prefer the gentler tool: if a fact is
merely outdated, `remember` the new truth and let curation settle it.
`forget` removes a whole cluster from recall and from the bank. There is
no restore: the full text of each fact stays in the bank's append-only
ledger for audit, but the memory is gone from recall. Confirm which memory
the user means (recall it, show it, then forget its `crystal_id`).

## What makes this memory different (worth telling the user when relevant)

The bank curates itself: it notices contradictions between things it was
told, records questions it couldn't answer as explicit gaps, grades its own
knowledge by quality tier, and consolidates related memories in the
background. When the user asks "what don't you know?" or "does anything
conflict?", those are real queries — `recall(mode="gaps")` and
`recall(mode="conflicts")` — not rhetorical ones.
