import { ReactNode, useState } from "react";
import { ChevronRight, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

// NOTE (dark redesign): the palette is remapped in tailwind.config.js —
// "white" is the card surface, gray-50/100/200 are dark insets/borders,
// gray-600..900 are light text. True-light text on vivid backgrounds
// uses zinc (untouched by the remap).

export function Disclosure({
  title, children, defaultOpen = false, className,
}: { title: ReactNode; children: ReactNode; defaultOpen?: boolean; className?: string }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className={cn("rounded-lg border border-gray-200 bg-white", className)}>
      <button type="button" onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2.5 text-left text-sm hover:bg-gray-50 transition-colors rounded-lg">
        {open ? <ChevronDown className="h-4 w-4 shrink-0 text-gray-400" /> : <ChevronRight className="h-4 w-4 shrink-0 text-gray-400" />}
        <span className="flex-1 text-gray-700 font-medium">{title}</span>
      </button>
      {open && <div className="border-t border-gray-100 px-3 py-3 text-sm text-gray-600">{children}</div>}
    </div>
  );
}

export function JsonView({ value, className }: { value: unknown; className?: string }) {
  const text = (() => { try { return JSON.stringify(value, null, 2); } catch { return String(value); } })();
  return <pre className={cn("json max-h-96 overflow-auto rounded-lg px-3 py-2", className)}>{text}</pre>;
}

function pill(cls: string, label: string, dot?: string) {
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-xs font-medium ring-1 ring-inset", cls)}>
      {dot && <span className={cn("h-1.5 w-1.5 rounded-full", dot)} />}
      {label}
    </span>
  );
}

export function MatchPill({ matchType }: { matchType: string }) {
  switch (matchType) {
    case "high": return pill("bg-emerald-50 text-emerald-600 ring-emerald-200/60", "high", "bg-emerald-400");
    case "medium": return pill("bg-amber-50 text-amber-600 ring-amber-200/60", "medium", "bg-amber-400");
    case "low": return pill("bg-red-50 text-red-600 ring-red-200/60", "low", "bg-red-400");
    default: return pill("bg-gray-50 text-gray-500 ring-gray-200/60", matchType, "bg-gray-400");
  }
}

export function QualityPill({ tier }: { tier: string }) {
  switch (tier) {
    case "whitelist": return pill("bg-emerald-50 text-emerald-600 ring-emerald-200/60", tier);
    case "neutral": return pill("bg-gray-50 text-gray-500 ring-gray-200/60", tier);
    case "quarantine": return pill("bg-amber-50 text-amber-600 ring-amber-200/60", tier);
    case "blacklist": return pill("bg-red-50 text-red-600 ring-red-200/60", tier);
    default: return pill("bg-gray-50 text-gray-500 ring-gray-200/60", tier);
  }
}

export function EmptyState({ title, description, action }: { title: string; description?: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-gray-300 bg-prism-subtle px-8 py-16 text-center">
      <div className="mb-4">
        <svg className="mx-auto h-12 w-12 opacity-80" viewBox="0 0 20 20" fill="none">
          <path d="M10 2L17 7V13L10 18L3 13V7L10 2Z" fill="#313868" />
          <path d="M10 2L17 7L10 10L3 7L10 2Z" fill="#3d4480" />
          <path d="M10 10L17 7V13L10 18V10Z" fill="#282e5c" />
        </svg>
      </div>
      <p className="text-sm font-semibold text-gray-900">{title}</p>
      {description && <p className="mt-1.5 max-w-md text-sm leading-relaxed text-gray-500">{description}</p>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

export function ErrorBanner({ title, message }: { title: string; message?: string }) {
  return (
    <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm">
      <p className="font-medium text-red-700">{title}</p>
      {message && <p className="mt-1 text-red-600/90">{message}</p>}
    </div>
  );
}

export function LoadingRows({ rows = 5, cols = 4 }: { rows?: number; cols?: number }) {
  return (
    <>
      {Array.from({ length: rows }).map((_, i) => (
        <tr key={i} className="border-t border-gray-100">
          {Array.from({ length: cols }).map((_, j) => (
            <td key={j} className="px-4 py-3"><div className="h-3 w-3/4 animate-pulse rounded bg-gray-100" /></td>
          ))}
        </tr>
      ))}
    </>
  );
}

export function CrystalButton({
  children, variant = "primary", size = "md", disabled = false, className, ...props
}: {
  children: ReactNode; variant?: "primary" | "secondary" | "ghost"; size?: "sm" | "md";
  disabled?: boolean; className?: string;
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  const base = "inline-flex items-center justify-center gap-1.5 rounded-lg font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed";
  const sizes = { sm: "px-2.5 py-1.5 text-xs", md: "px-3.5 py-2 text-sm" };
  const variants = {
    primary: "bg-brand-600 text-zinc-50 shadow-glow hover:bg-brand-500 active:bg-brand-600",
    secondary: "bg-gray-50 text-gray-800 border border-gray-200 hover:border-gray-300 hover:bg-gray-100",
    ghost: "text-gray-500 hover:text-gray-900 hover:bg-gray-100",
  };
  return <button className={cn(base, sizes[size], variants[variant], className)} disabled={disabled} {...props}>{children}</button>;
}

export function TypeBadge({ type }: { type: string }) {
  const cls = (() => {
    switch (type) {
      case "fact": return "bg-blue-50 text-blue-600";
      case "entity": return "bg-emerald-50 text-emerald-600";
      case "relationship": return "bg-violet-50 text-violet-600";
      case "process": return "bg-amber-50 text-amber-600";
      case "qa": return "bg-cyan-50 text-cyan-600";
      case "definition": return "bg-rose-50 text-rose-600";
      default: return "bg-gray-50 text-gray-500";
    }
  })();
  return <span className={cn("inline-flex rounded-md px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider", cls)}>{type}</span>;
}
