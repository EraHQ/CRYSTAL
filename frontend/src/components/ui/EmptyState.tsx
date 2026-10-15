import { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

// Part 3: ONE empty state for every screen. Icon, one-line title,
// one-line help, optional primary action. `description` is accepted as
// an alias of `help` for the callers that predate this component.
export function EmptyState({ icon: Icon, title, help, description, action, className }: {
  icon?: LucideIcon;
  title: string;
  help?: string;
  description?: string;
  action?: ReactNode;
  className?: string;
}) {
  const line = help ?? description;
  return (
    <div className={cn("flex flex-col items-center justify-center rounded-xl border border-dashed border-line-strong bg-prism-subtle px-6 py-12 text-center", className)}>
      {Icon ? (
        <Icon aria-hidden className="mb-3 h-8 w-8 text-fg-3" />
      ) : (
        <svg aria-hidden className="mb-3 h-10 w-10 opacity-80" viewBox="0 0 20 20" fill="none">
          <path d="M10 2L17 7V13L10 18L3 13V7L10 2Z" style={{ fill: "rgb(var(--ac-sf))" }} />
          <path d="M10 2L17 7L10 10L3 7L10 2Z" style={{ fill: "rgb(var(--ac) / 0.55)" }} />
          <path d="M10 10L17 7V13L10 18V10Z" style={{ fill: "rgb(var(--ac) / 0.35)" }} />
        </svg>
      )}
      <p className="text-[14px] font-semibold text-fg">{title}</p>
      {line && <p className="mt-1 max-w-md text-[13px] leading-relaxed text-fg-2">{line}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
