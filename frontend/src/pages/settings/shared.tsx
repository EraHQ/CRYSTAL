// Settings tabs — Console Refine PR 9 (docs/CONSOLE_UX_AUDIT.md 5.8).
// What every tab shares: the tab list (each deep-linkable), one card
// shape, and the three button styles so a tab has one primary.
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export const SETTINGS_TABS = [
  { id: "connections", label: "Connections" },
  { id: "model", label: "Model & inference" },
  { id: "curation", label: "Curation" },
  { id: "api-key", label: "API key" },
  { id: "team", label: "Team" },
  { id: "your-data", label: "Your data" },
] as const;
export type SettingsTab = (typeof SETTINGS_TABS)[number]["id"];

export const BTN_PRIMARY = "inline-flex items-center gap-1.5 rounded-md bg-accent px-3 py-1.5 text-[13px] font-medium text-on-accent hover:bg-accent-hover disabled:opacity-50";
export const BTN_SECONDARY = "inline-flex items-center gap-1.5 rounded-md border border-line px-3 py-1.5 text-[13px] font-medium text-fg-2 hover:bg-inset hover:text-fg disabled:opacity-50";
export const BTN_QUIET = "text-[12.5px] text-fg-3 hover:text-fg disabled:opacity-50";
export const INPUT = "w-full rounded-md border border-line bg-inset px-2.5 py-1.5 text-[13px] text-fg outline-none placeholder:text-fg-3 focus:border-accent";

export function Section({ title, help, children, danger = false, className }: {
  title: string; help?: ReactNode; children: ReactNode; danger?: boolean; className?: string;
}) {
  return (
    <section aria-label={title} className={cn("rounded-xl border bg-surface p-4 min-[720px]:p-5", danger ? "border-bad/40" : "border-line", className)}>
      <h2 className={cn("text-[15px] font-semibold", danger ? "text-bad" : "text-fg")}>{title}</h2>
      {help && <p className="mt-1 text-[13px] text-fg-2">{help}</p>}
      <div className="mt-3">{children}</div>
    </section>
  );
}

// A choice card (inference mode, model): one of a radiogroup.
export function ChoiceCard({ checked, disabled = false, onPick, title, body, aside, raw }: {
  checked: boolean; disabled?: boolean; onPick: () => void; title: string; body: string; aside?: ReactNode; raw?: string;
}) {
  return (
    <button type="button" role="radio" aria-checked={checked} aria-disabled={disabled || undefined} title={raw}
      onClick={() => { if (!disabled) onPick(); }}
      className={cn("flex w-full flex-col items-start rounded-lg border px-3 py-2.5 text-left",
        checked ? "border-accent bg-accent-soft" : "border-line bg-inset hover:border-line-strong",
        disabled && "cursor-not-allowed opacity-70")}>
      <span className="flex w-full items-center gap-2">
        <span className={cn("h-3 w-3 shrink-0 rounded-full border", checked ? "border-accent bg-accent" : "border-line-strong")} aria-hidden />
        <span className="text-[13.5px] font-medium text-fg">{title}</span>
        <span className="flex-1" />
        {aside}
      </span>
      <span className="mt-1 pl-5 text-[12.5px] text-fg-2">{body}</span>
    </button>
  );
}
