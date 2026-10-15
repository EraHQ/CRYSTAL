import { ReactNode } from "react";
import { cn } from "@/lib/utils";

// Part 3: one card row for search, filter chips and a group-by control.
// The search slot grows; everything wraps on narrow screens.
export function Toolbar({ search, children, className }: {
  search?: ReactNode;
  children?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-wrap items-center gap-2 rounded-lg border border-line bg-surface p-2", className)}>
      {search && <div className="min-w-[180px] flex-1">{search}</div>}
      {children}
    </div>
  );
}
