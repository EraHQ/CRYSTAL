import { ReactNode } from "react";
import { cn } from "@/lib/utils";

export type StatusVariant = "ok" | "warn" | "bad" | "info" | "vio" | "neutral";

const STYLES: Record<StatusVariant, string> = {
  ok: "bg-ok-soft text-ok ring-ok/30",
  warn: "bg-warn-soft text-warn ring-warn/30",
  bad: "bg-bad-soft text-bad ring-bad/30",
  info: "bg-info-soft text-info ring-info/30",
  vio: "bg-vio-soft text-vio ring-vio/30",
  neutral: "bg-inset text-fg-2 ring-line",
};

// Part 3: a status is a WORD, never color alone; the raw enum rides in
// `title` (and aria-label when given) so the term is one hover away.
export function StatusPill({ variant = "neutral", children, raw, dot = false, className }: {
  variant?: StatusVariant;
  children: ReactNode;
  raw?: string;
  dot?: boolean;
  className?: string;
}) {
  return (
    <span title={raw} className={cn("inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[12px] font-medium ring-1 ring-inset", STYLES[variant], className)}>
      {dot && <span aria-hidden className="h-1.5 w-1.5 rounded-full bg-current" />}
      {children}
    </span>
  );
}
