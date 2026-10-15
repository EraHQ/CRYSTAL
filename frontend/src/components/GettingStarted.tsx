// Getting-started checklist (T2b, 2026-09-26): the console landing's
// bridge from onboarding to habit. Hosted tenant sessions only; hides
// itself when every row is done or when hidden by the operator
// (localStorage; coming back after a browser change is acceptable).
//
// Console Refine PR 3 (4.2): a card inside the content column with a
// progress line, every row a link naming its destination, a "Check now"
// that refetches, and a one-line collapsed form for Chat. Row 4 ("first
// recall from a tool") completes when the status carries first_recall_at;
// the backend does not send that yet (Part 6: tool name per query-log
// row), so the row reads as not done until it does, never as done by
// guess.
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Check, ChevronRight, RotateCw, X } from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { IconButton } from "@/components/ui";
import { cn } from "@/lib/utils";

const DISMISS_KEY = "cc.gettingstarted.dismissed";

function readDismissed(): boolean {
  try { return localStorage.getItem(DISMISS_KEY) === "1"; } catch { return false; }
}

interface Row { done: boolean; label: string; to: string; destination: string }

export function GettingStarted({ collapsed = false }: { collapsed?: boolean }) {
  const { me } = useAuth();
  const [dismissed, setDismissed] = useState(readDismissed);
  const status = useQuery({
    queryKey: ["onboarding-status"],
    queryFn: () => api.onboardingStatus(),
    enabled: !dismissed && me?.kind === "user",
    staleTime: 30_000,
  });

  if (dismissed || me?.kind !== "user") return null;
  const connected = status.data?.connected === true;
  const recalled = (status.data as { first_recall_at?: string | null } | undefined)?.first_recall_at != null;
  const facts = me?.usage?.facts_used ?? null;
  const hasFacts = facts != null && facts > 0;
  if (!status.data && !me?.usage) return null;

  const rows: Row[] = [
    { done: true, label: "Create your workspace", to: "/settings", destination: "Workspace settings" },
    { done: connected, label: "Connect a tool", to: "/settings", destination: "Manage connections" },
    { done: hasFacts, label: "Store your first memory", to: "/", destination: "Open Crystal Bank" },
    { done: recalled, label: "Ask “what do you remember about me?” from a connected tool", to: "/queries", destination: "Query log" },
  ];
  const doneCount = rows.filter((r) => r.done).length;
  if (doneCount === rows.length) return null;

  const hide = () => {
    try { localStorage.setItem(DISMISS_KEY, "1"); } catch { /* storage off */ }
    setDismissed(true);
  };

  if (collapsed) {
    const next = rows.find((r) => !r.done);
    return (
      <div className="flex items-center gap-3 border-b border-line bg-surface px-4 py-1.5 text-[12.5px]">
        <span className="font-medium text-fg">Getting started</span>
        <span className="text-fg-3">{doneCount} of {rows.length} done</span>
        {next && (
          <Link to={next.to} className="min-w-0 truncate text-accent hover:underline">
            Next: {next.label}
          </Link>
        )}
        <span className="ml-auto"><IconButton label="Hide getting started" icon={X} size="sm" onClick={hide} /></span>
      </div>
    );
  }

  return (
    <section aria-label="Getting started" className="mb-6 rounded-xl border border-line bg-surface p-4">
      <div className="flex items-center gap-3">
        <h2 className="text-[14px] font-semibold text-fg">Getting started</h2>
        <span className="text-[12.5px] text-fg-3">{doneCount} of {rows.length} done</span>
        <span className="ml-auto flex items-center gap-1">
          <IconButton label="Check now" icon={RotateCw} size="sm" onClick={() => void status.refetch()} disabled={status.isFetching} />
          <IconButton label="Hide getting started" icon={X} size="sm" onClick={hide} />
        </span>
      </div>
      <div role="progressbar" aria-label="Getting started" aria-valuemin={0} aria-valuemax={rows.length} aria-valuenow={doneCount}
        className="mt-2.5 h-1.5 w-full overflow-hidden rounded-full bg-inset">
        <div className="h-full rounded-full bg-accent transition-[width]" style={{ width: `${(doneCount / rows.length) * 100}%` }} />
      </div>
      <ul className="mt-3 divide-y divide-line">
        {rows.map((r) => (
          <li key={r.label}>
            <Link to={r.to} className="group flex items-center gap-3 py-2 text-[13px] hover:text-fg">
              {r.done ? (
                <Check aria-hidden className="h-4 w-4 shrink-0 text-ok" />
              ) : (
                <span aria-hidden className="h-4 w-4 shrink-0 rounded border-[1.5px] border-accent" />
              )}
              <span className={cn("min-w-0 flex-1", r.done ? "text-fg-3 line-through" : "text-fg-2 group-hover:text-fg")}>
                {r.label}
                {r.done && <span className="sr-only"> (done)</span>}
              </span>
              <span className="flex shrink-0 items-center gap-1 text-[12.5px] text-accent">
                {r.destination} <ChevronRight aria-hidden className="h-3.5 w-3.5" />
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
}
