import { ReactNode } from "react";
import { cn } from "@/lib/utils";

// Part 3: replaces the native confirm dialog. Renders INSIDE the card it belongs
// to: one sentence of consequence, a danger button, Cancel. Nothing
// native, nothing modal. `permanent` adds the sentence the spec asks
// for when there is no Undo.
export function ConfirmInline({ open, message, confirmLabel = "Delete", onConfirm, onCancel, busy = false, permanent = false, className }: {
  open: boolean;
  message: ReactNode;
  confirmLabel?: string;
  onConfirm: () => void;
  onCancel: () => void;
  busy?: boolean;
  permanent?: boolean;
  className?: string;
}) {
  if (!open) return null;
  return (
    <div role="group" aria-label="Confirm" className={cn("mt-2 flex flex-wrap items-center gap-2 rounded-lg border border-bad/40 bg-bad-soft px-3 py-2 text-[13px] text-fg", className)}>
      <span className="min-w-0 flex-1">
        {message}
        {permanent && <span className="text-fg-2"> This is permanent.</span>}
      </span>
      <button type="button" onClick={onCancel} disabled={busy}
        className="rounded-md border border-line px-2.5 py-1 text-[12.5px] font-medium text-fg-2 hover:bg-inset hover:text-fg disabled:opacity-50">
        Cancel
      </button>
      <button type="button" onClick={onConfirm} disabled={busy} autoFocus
        className="rounded-md bg-bad px-2.5 py-1 text-[12.5px] font-semibold text-canvas hover:bg-bad/90 disabled:opacity-50">
        {busy ? "Working…" : confirmLabel}
      </button>
    </div>
  );
}
