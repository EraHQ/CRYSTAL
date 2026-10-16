import { cn } from "@/lib/utils";

// Refine PR 5: a block-level loading placeholder for card grids and
// lists (LoadingRows is the <tr> skeleton for tables and must live in a
// <tbody>). `rows` bars of varying width, announced once as busy.
export function Skeleton({ rows = 4, className }: { rows?: number; className?: string }) {
  const widths = ["w-3/4", "w-1/2", "w-2/3", "w-5/6", "w-1/3"];
  return (
    <div role="status" aria-busy="true" aria-label="Loading" className={cn("space-y-2.5", className)}>
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className={cn("h-3.5 animate-pulse rounded bg-inset", widths[i % widths.length])} />
      ))}
    </div>
  );
}
