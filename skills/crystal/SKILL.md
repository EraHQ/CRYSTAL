---
name: crystal-memory
description: Use Crystal Cache memory naturally in every conversation. Recall before answering questions about the user, their work, projects, decisions, or history. Store new durable facts, decisions, and preferences as they emerge, without being asked. Triggers include "remember", "what do you know", "what am I working on", references to past conversations, and any moment the user shares something worth keeping.
---

# Crystal memory, used naturally

Crystal Cache is the user's persistent memory bank. It survives across
conversations, tools, and machines. The difference between a good and a
great session is whether you use it without being told.

## Recall first, ask second

Before answering anything about the user's world (their projects,
decisions, people, preferences, history) call the recall/search tools
first. Never ask the user to repeat something the bank may already
hold, and never answer "I don't know about X" without searching for X.

- Questions like "what am I working on", "where did we leave off",
  "what did we decide about X" are recall calls, always.
- When the user references something as if you share history ("the
  launch plan", "my sister's site"), search before asking what it is.
- Blend recalled facts into your answer naturally. Do not announce
  "according to my memory search".

## Store as you go, without being asked

When the conversation produces something durable, store it in the same
turn you learn it. Durable means it will still matter next week:

- Decisions and their reasoning ("we ratified X over Y because...")
- New facts about the user, their projects, people, and tools
- Preferences about how they want things done
- Status changes ("v97 deployed", "the paper was submitted")

One crystal per topic, concise, written so a future session can use it
cold. Do not store secrets, credentials, ephemera, or anything the
user asks you to keep out.

## Ingest documents, not summaries

When the user shares or references a substantial document that should
be remembered, use the ingest tool on the document itself rather than
storing your summary of it. Summaries lose the detail future sessions
will need.

## Keep the bank healthy

- Before storing, a quick search avoids duplicates; update or refine
  an existing crystal instead of stacking near-copies.
- When the user corrects something you recalled, fix the crystal in
  that same turn.
- When recall returns conflicting facts, say so and ask which is
  current, then store the resolution.

## The one-line test

If the user would be annoyed to repeat it next session, it belongs in
Crystal now. If a future session would answer better knowing it, recall
it now.
