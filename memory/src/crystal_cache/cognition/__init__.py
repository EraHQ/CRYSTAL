"""Multi-Agent Cognition Environment — ported verbatim from v1 in Phase 6 Wave C.

Originated as v1's Phase 10 work. v2 preserves the orchestrator /
worker / validator loop and the per-task ephemeral environment
semantics; the only deltas are two SQL refactors that route through
MetadataStore methods instead of inline SQLAlchemy:

  - roles._worker_crystal_key_scan → store.list_facts_by_key_prefix
    (new in CognitionExtensionsMixin)
  - engine._commit_and_finalize → store.create_document_upload
    (already in AuditTablesMixin from Phase 5)

The agent reframe (D-A6) treats this whole subsystem as a single
tool `cognition_run` exposed to the agent. The Orchestrator/Worker/
Validator loop is internal to that tool — agents do not see plans
or step outputs.
"""
