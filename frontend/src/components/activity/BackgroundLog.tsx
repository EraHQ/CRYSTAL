// Activity → Background (Q90=A, 2026-10-19): what Crystal did to its own
// knowledge without being asked, newest first: conflicts found, gaps
// filled or reopened, assumptions written and judged, verifications,
// scans. The same witness stream the Assumptions page shows in a drawer,
// here as the audit view with a type filter and a link to the subject.
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { History } from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "@/lib/api";
import type { CurationEvent } from "@/lib/types";
import { EmptyState, ErrorBanner, FilterChips, Skeleton } from "@/components/ui";
import { fmtDateTime, fmtRelative } from "@/lib/utils";
import { eventText } from "./events";

/** Where an event's subject lives in the console, when it is somewhere a person can open. */
export function subjectHref(e: CurationEvent): string | null {
  const id = e.subject_id;
  if (!id) return null;
  if (e.event_type.startsWith("assumption")) return `/assumptions/${encodeURIComponent(id)}`;
  if (e.event_type.startsWith("conflict")) return "/conflicts";
  if (e.event_type.startsWith("gap")) return "/cognition";
  if (e.event_type.startsWith("verification")) return `/assumptions/${encodeURIComponent(id)}`;
  return null;
}

export function BackgroundLog({ customerId }: { customerId: string }) {
  const [type, setType] = useState<string | null>(null);
  const activity = useQuery({ queryKey: ["curation-activity", customerId, 200], queryFn: () => api.listCurationActivity(customerId, 200), refetchInterval: 15_000 });
  const events: CurationEvent[] = activity.data?.events ?? [];
  const types = useMemo(() => {
    const seen = new Map<string, number>();
    for (const e of events) seen.set(e.event_type, (seen.get(e.event_type) ?? 0) + 1);
    return [...seen.entries()].sort((a, b) => b[1] - a[1]);
  }, [events]);
  const rows = type ? events.filter((e) => e.event_type === type) : events;
  return (
    <div>
      <p className="mb-2 text-[13px] text-fg-2">{activity.isLoading ? "Loading…" : `${events.length.toLocaleString()} event${events.length === 1 ? "" : "s"} · the latest 200. Nothing Crystal does to its own knowledge is silent.`}</p>
      {types.length > 1 && (
        <div className="mb-4">
          <FilterChips label="Event" value={type ?? "any"} onChange={(v) => setType(v === "any" || v === null ? null : v)}
            options={[{ value: "any", label: "Any", count: events.length }, ...types.map(([t, n]) => ({ value: t, label: eventText(t), count: n, title: t }))]} />
        </div>
      )}
      {activity.isError && <ErrorBanner title="Couldn't load activity" error={activity.error} onRetry={() => activity.refetch()} />}
      {activity.isLoading && <Skeleton rows={5} />}
      {!activity.isLoading && !activity.isError && events.length === 0 && (
        <EmptyState icon={History} title="No activity yet" help="Conflict scans, gap fills, assumptions and verifications land here as they happen." />
      )}
      {rows.length > 0 && (
        <ul className="divide-y divide-line rounded-xl border border-line bg-surface">
          {rows.map((e) => {
            const href = subjectHref(e);
            return (
              <li key={e.id} className="flex flex-wrap items-start gap-x-3 gap-y-1 px-3 py-2 text-[13px]">
                <span className="w-24 shrink-0 whitespace-nowrap text-[12px] text-fg-3" title={e.created_at ? fmtDateTime(e.created_at) : undefined}>{fmtRelative(e.created_at)}</span>
                <span className="w-40 shrink-0 text-[12px] font-medium text-fg-2" title={e.event_type}>{eventText(e.event_type)}</span>
                <span className="min-w-0 flex-1 break-words text-fg">{e.label || ""}</span>
                {href && <Link to={href} className="text-[12.5px] text-accent hover:underline">Open</Link>}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
