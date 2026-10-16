// Console Refine PR 5 (docs/CONSOLE_UX_AUDIT.md 5.3): bits the Knowledge
// tabs share.
import type { StatusVariant } from "@/components/ui";
import { fmtRelative } from "@/lib/utils";

export type KnowledgeTab = "documents" | "repos" | "shapes" | "general";

export const TABS: Array<{ id: KnowledgeTab; label: string }> = [
  { id: "documents", label: "Documents" },
  { id: "repos", label: "Watched repos" },
  { id: "shapes", label: "Data shapes" },
  { id: "general", label: "General knowledge" },
];

// "12.3k characters" — the server counts characters, not bytes.
export function fmtChars(n: number | null | undefined): string {
  const v = n ?? 0;
  if (v < 1000) return `${v} characters`;
  return `${(v / 1000).toFixed(1)}k characters`;
}

// Watch clocks: epoch-adjacent timestamps are the scheduler's "due now"
// sentinel on fresh watches, not a real check (2026-07-27).
export function fmtChecked(iso: string | null | undefined): string {
  if (!iso) return "never";
  const t = new Date(iso).getTime();
  if (!Number.isFinite(t) || t < 946684800000) return "never";
  return fmtRelative(iso);
}

// Document status (endpoints/documents.py) in plain words.
export const DOC_STATUS: Record<string, { label: string; variant: StatusVariant }> = {
  pending: { label: "Queued", variant: "neutral" },
  extracting: { label: "Reading", variant: "info" },
  review: { label: "Ready for review", variant: "warn" },
  approved: { label: "Crystallizing", variant: "info" },
  crystallizing: { label: "Crystallizing", variant: "info" },
  crystallized: { label: "Crystallized", variant: "ok" },
  error: { label: "Couldn't read", variant: "bad" },
};

// Watch sync_state (endpoints/admin.py admin_list_watches) in plain words.
export const WATCH_STATE: Record<string, { label: string; variant: StatusVariant }> = {
  syncing: { label: "Syncing", variant: "info" },
  synced: { label: "Synced", variant: "ok" },
  attention: { label: "Needs attention", variant: "bad" },
  waiting: { label: "Waiting for first sync", variant: "neutral" },
  paused: { label: "Paused", variant: "neutral" },
};

export const REVIEW_MODE: Record<string, string> = {
  auto: "Saves automatically",
  gated: "Review first",
};

// Upload failures explained with a cause and a fix (5.3 item 5). The
// server's 400 for a scanned or empty file says "no extractable text".
export function explainUploadError(message: string): string {
  if (/no extractable text/i.test(message)) {
    return "This looks like a scanned image or an empty file, so there was no text to read. Upload a text PDF or the original document instead.";
  }
  return message;
}
