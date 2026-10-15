import { ReactNode } from "react";
import { cn } from "@/lib/utils";

// Part 3: every page opens with the same header. h1 22px/650, meta
// 13.5px fg-2, actions on the right, optional toolbar row below; the
// row wraps under 720px instead of overflowing.
export function PageHeader({ title, meta, actions, toolbar, className }: {
  title: ReactNode;
  meta?: ReactNode;
  actions?: ReactNode;
  toolbar?: ReactNode;
  className?: string;
}) {
  return (
    <header className={cn("mb-4", className)}>
      <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
        <div className="min-w-0">
          <h1 className="text-[22px] font-[650] leading-tight tracking-tight text-fg">{title}</h1>
          {meta && <p className="mt-1 text-[13.5px] text-fg-2">{meta}</p>}
        </div>
        {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
      </div>
      {toolbar && <div className="mt-3">{toolbar}</div>}
    </header>
  );
}
