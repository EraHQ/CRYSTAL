// Query log — Console Refine PR 11 (docs/CONSOLE_UX_AUDIT.md 5.7, Q65=A).
// Every question asked of this memory, by which tool, with the memories
// that answered as text (never ids). Search and the chips work over the
// rows loaded so far; Load more pages through the rest. The proxy-era
// metrics (injection, cache split, shadow delta, concept) stay behind an
// admin-only Raw disclosure. Under 720px the Tool and Latency columns
// hide and the table scrolls inside its card.
// Q90=A (2026-10-19): this is the Questions view of the Activity page
// (/activity?view=questions); /queries redirects there. The panel owns
// its count line and toolbar so the page header stays the page's.
import { Fragment, useMemo, useState } from "react";
import { useInfiniteQuery } from "@tanstack/react-query";
import { ChevronDown, ChevronRight, MessageSquare, Search } from "lucide-react";
import { api } from "@/lib/api";
import { Disclosure, EmptyState, ErrorBanner, FilterChips, JsonView, Skeleton, StatusPill, Toolbar } from "@/components/ui";
import type { StatusVariant } from "@/components/ui";
import type { QueryLogSummary } from "@/lib/types";
import { cn, fmtDateTime, fmtRelative } from "@/lib/utils";
import { modelLabel } from "@/pages/settings/ModelInference";

const PAGE = 100;

// Match strength in plain words (raw high/medium/low/none in title).
export const MATCH: Record<string, { label: string; variant: StatusVariant }> = {
  high: { label: "Strong", variant: "ok" },
  medium: { label: "Partial", variant: "info" },
  low: { label: "Weak", variant: "warn" },
  none: { label: "None", variant: "neutral" },
};

// The client label the server stored (User-Agent family or "console") in
// the user's words. Unknown families show as recorded.
export function toolLabel(client: string | null | undefined): string {
  if (!client) return "Unknown";
  const c = client.toLowerCase();
  if (c === "console") return "Console";
  if (c.includes("claude")) return "Claude";
  if (c.includes("cursor")) return "Cursor";
  if (c.includes("chatgpt") || c.includes("openai")) return "ChatGPT";
  if (c.includes("windsurf") || c.includes("codeium")) return "Windsurf";
  if (c.includes("cline")) return "Cline";
  if (c.includes("zed")) return "Zed";
  if (c.includes("gemini")) return "Gemini";
  if (c.includes("vscode") || c.includes("visual studio")) return "VS Code";
  if (c.includes("python") || c.includes("httpx") || c.includes("requests")) return "Python client";
  if (c.includes("node") || c.includes("undici")) return "Node client";
  if (c.includes("curl")) return "curl";
  return client;
}

export function matchesSearch(q: QueryLogSummary, term: string): boolean {
  const t = term.trim().toLowerCase();
  if (!t) return true;
  if (q.query_text.toLowerCase().includes(t)) return true;
  if ((q.response_text ?? "").toLowerCase().includes(t)) return true;
  return (q.matched_memories ?? []).some((m) => m.text.toLowerCase().includes(t));
}

export function QueryLogPanel({ customerId, admin }: { customerId: string; admin: boolean }) {
  const isPlatformAdmin = admin;
  const [term, setTerm] = useState("");
  const [match, setMatch] = useState<string | null>(null);
  const [tool, setTool] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const list = useInfiniteQuery({
    queryKey: ["query_logs", customerId],
    queryFn: ({ pageParam }) => api.listQueryLogs(customerId, { offset: pageParam, limit: PAGE }),
    initialPageParam: 0,
    getNextPageParam: (last, pages) => {
      const loaded = pages.reduce((n, p) => n + p.items.length, 0);
      return loaded < last.total && last.items.length > 0 ? loaded : undefined;
    },
  });
  const pages = list.data?.pages ?? [];
  const items: QueryLogSummary[] = useMemo(() => pages.flatMap((p) => p.items), [pages]);
  const total = pages.length ? pages[pages.length - 1].total : 0;

  const tools = useMemo(() => {
    const seen = new Map<string, number>();
    for (const q of items) {
      const k = toolLabel(q.client);
      seen.set(k, (seen.get(k) ?? 0) + 1);
    }
    return [...seen.entries()].sort((a, b) => b[1] - a[1]);
  }, [items]);
  const searched = useMemo(() => items.filter((q) => matchesSearch(q, term)), [items, term]);
  const rows = searched.filter((q) => (!match || q.match_type === match) && (!tool || toolLabel(q.client) === tool));
  const countMatch = (m: string | null) => searched.filter((q) => (!m || q.match_type === m) && (!tool || toolLabel(q.client) === tool)).length;
  const countTool = (t: string | null) => searched.filter((q) => (!t || toolLabel(q.client) === t) && (!match || q.match_type === match)).length;

  const meta = list.isLoading ? "Loading…" : `${total.toLocaleString()} question${total === 1 ? "" : "s"} · showing the latest ${items.length.toLocaleString()}`;

  return (
    <div>
      <p className="mb-2 text-[13px] text-fg-2">{meta}</p>
      <div className="mb-4">
          <Toolbar search={
            <label className="relative block">
              <Search aria-hidden className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-fg-3" />
              <input value={term} onChange={(e) => setTerm(e.target.value)} placeholder="Search questions, answers and memories" aria-label="Search"
                className="w-full rounded-md border border-line bg-inset py-1.5 pl-8 pr-2.5 text-[13px] text-fg outline-none placeholder:text-fg-3 focus:border-accent" />
            </label>
          }>
            <FilterChips label="Match" value={match ?? "all"} onChange={(v) => setMatch(v === "all" || v === null ? null : v)}
              options={[{ value: "all", label: "All", count: countMatch(null) },
                ...Object.entries(MATCH).map(([k, v]) => ({ value: k, label: v.label, count: countMatch(k), title: k }))]} />
            {tools.length > 0 && (
              <FilterChips label="Tool" value={tool ?? "any"} onChange={(v) => setTool(v === "any" || v === null ? null : v)}
                options={[{ value: "any", label: "Any", count: countTool(null) }, ...tools.map(([t]) => ({ value: t, label: t, count: countTool(t) }))]} />
            )}
          </Toolbar>
      </div>

      {list.isError && <ErrorBanner title="Couldn't load the log" error={list.error} onRetry={() => list.refetch()} />}
      {list.isLoading && <Skeleton rows={6} />}
      {!list.isLoading && !list.isError && items.length === 0 && (
        <EmptyState icon={MessageSquare} title="No questions yet" help="Ask a connected tool “what do you remember about me?” and it shows up here." />
      )}
      {!list.isLoading && items.length > 0 && rows.length === 0 && (
        <EmptyState icon={Search} title="Nothing matches" help="Try another phrase, or Load more below to search older rows." />
      )}

      {rows.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-line bg-surface">
          <table className="w-full min-w-[520px] text-[13px]">
            <thead>
              <tr className="text-left text-[11px] uppercase tracking-[0.08em] text-fg-3">
                <th className="px-3 py-2 font-medium">Time</th>
                <th className="hidden px-3 py-2 font-medium min-[720px]:table-cell">Tool</th>
                <th className="px-3 py-2 font-medium">Question</th>
                <th className="px-3 py-2 font-medium">Match</th>
                <th className="px-3 py-2 font-medium">Answered with</th>
                <th className="hidden px-3 py-2 text-right font-medium min-[720px]:table-cell">Latency</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((q) => {
                const open = expandedId === q.id;
                const mem = q.matched_memories ?? [];
                const m = MATCH[q.match_type] ?? MATCH.none;
                return (
                  <Fragment key={q.id}>
                    <tr className={cn("cursor-pointer border-t border-line hover:bg-inset", open && "bg-inset")} onClick={() => setExpandedId(open ? null : q.id)}
                      aria-expanded={open}>
                      <td className="whitespace-nowrap px-3 py-2 text-[12px] text-fg-3" title={fmtDateTime(q.timestamp)}>
                        <span className="inline-flex items-center gap-1">{open ? <ChevronDown aria-hidden className="h-3.5 w-3.5" /> : <ChevronRight aria-hidden className="h-3.5 w-3.5" />}{fmtRelative(q.timestamp)}</span>
                      </td>
                      <td className="hidden whitespace-nowrap px-3 py-2 text-fg-2 min-[720px]:table-cell" title={q.client ?? undefined}>{toolLabel(q.client)}</td>
                      <td className="max-w-[320px] truncate px-3 py-2 text-fg" title={q.query_text}>{q.query_text || "—"}</td>
                      <td className="px-3 py-2"><StatusPill variant={m.variant} raw={q.match_type}>{m.label}</StatusPill></td>
                      <td className="max-w-[320px] truncate px-3 py-2 text-fg-2" title={mem[0]?.text}>
                        {mem.length === 0 ? <span className="text-fg-3">—</span> : <>{mem[0].text}{mem.length > 1 && <span className="ml-1 text-fg-3">+{mem.length - 1} more</span>}</>}
                      </td>
                      <td className="hidden whitespace-nowrap px-3 py-2 text-right text-fg-2 min-[720px]:table-cell">{q.latency_ms != null ? `${q.latency_ms} ms` : "—"}</td>
                    </tr>
                    {open && (
                      <tr className="border-t border-line bg-inset/60">
                        <td colSpan={6} className="px-3 py-3"><Expanded q={q} admin={isPlatformAdmin} /></td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {list.hasNextPage && (
        <div className="mt-3 flex justify-center">
          <button type="button" onClick={() => void list.fetchNextPage()} disabled={list.isFetchingNextPage}
            className="rounded-md border border-line px-3 py-1.5 text-[13px] font-medium text-fg-2 hover:bg-inset hover:text-fg disabled:opacity-50">
            {list.isFetchingNextPage ? "Loading…" : `Load more (${(total - items.length).toLocaleString()} older)`}
          </button>
        </div>
      )}
    </div>
  );
}

function Expanded({ q, admin }: { q: QueryLogSummary; admin: boolean }) {
  const mem = q.matched_memories ?? [];
  return (
    <div className="grid gap-4 min-[720px]:grid-cols-[minmax(0,1fr)_240px]">
      <div>
        <p className="text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">Matched memories</p>
        {mem.length === 0 ? (
          <p className="mt-1 text-[13px] text-fg-3">No memory matched this question.</p>
        ) : (
          <ul className="mt-1 space-y-1.5">
            {mem.map((m) => <li key={m.id} className="rounded-md border border-line bg-surface px-2.5 py-1.5 text-[13px] text-fg">{m.text}</li>)}
          </ul>
        )}
        {q.response_text && (
          <div className="mt-3">
            <Disclosure title="See full answer"><p className="whitespace-pre-wrap text-[13px] text-fg-2">{q.response_text}</p></Disclosure>
          </div>
        )}
      </div>
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[12.5px]">
        <dt className="text-fg-3">Asked by</dt><dd className="text-fg" title={q.client ?? undefined}>{toolLabel(q.client)}</dd>
        {q.model && <><dt className="text-fg-3">Model</dt><dd className="text-fg" title={q.model}>{modelLabel(q.model)}</dd></>}
        <dt className="text-fg-3">Asked</dt><dd className="text-fg">{fmtDateTime(q.timestamp)}</dd>
        <dt className="text-fg-3">Took</dt><dd className="text-fg">{q.latency_ms != null ? `${q.latency_ms} ms` : "—"}</dd>
        {q.prompt_tokens != null && <><dt className="text-fg-3">Context sent</dt><dd className="text-fg">{q.prompt_tokens.toLocaleString()} tokens</dd></>}
        <dt className="text-fg-3">Memories checked</dt><dd className="text-fg">{(q.matched_facts ?? []).length}</dd>
        {admin && (
          <dd className="col-span-2 mt-2"><Disclosure title="Raw (admin)"><JsonView value={q} /></Disclosure></dd>
        )}
      </dl>
    </div>
  );
}
