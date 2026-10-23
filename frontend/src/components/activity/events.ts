// The curation feed's event types in plain words. Shared by the
// Assumptions page's Activity drawer and the Activity page's Background
// view (Q90=A, 2026-10-19). An unknown type shows with its underscores
// turned into spaces, never hidden.
export const EVENT_TEXT: Record<string, string> = {
  assumption_written: "Assumption written", assumption_approved: "Assumption approved", assumption_invalidated: "Assumption invalidated",
  assumption_deleted: "Assumption deleted", assumption_responded: "You responded", assumptions_superseded: "Revisions superseded",
  gap_filled: "Gap filled", gap_reopened: "Gap reopened", conflict_found: "Conflict found", conflict_reopened: "Conflict reopened",
  conflict_scan_completed: "Conflict scan finished", verification_spawned: "Verification queued", verification_completed: "Verification finished",
  bank_erased: "Memories erased", turn_started: "Agent turn started", turn_completed: "Agent turn finished", tool_called: "Agent used a tool",
  consequence_written: "Consequence kept", consequence_recallable: "Consequence entered recall",
  steward_action: "Steward acted", steward_session: "Steward session", steward_paused: "Steward paused", steward_undone: "Action undone", control_action: "Acted from chat",
  steward_enabled: "Steward switched on", steward_disabled: "Steward switched off",
  memory_allowed: "Let into recall by a rule",
  auto_action: "Crystal settled it",
};

export const eventText = (type: string): string => EVENT_TEXT[type] ?? type.replace(/_/g, " ");
