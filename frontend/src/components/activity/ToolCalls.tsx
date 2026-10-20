// Activity → Tool calls (2026-10-19): every call a connected client made
// to Crystal's tools (Claude Desktop, Cursor, ChatGPT, the CLI, the
// console): which tool, in words; what it was asked; how it ended; how
// long it took. A recall links to its question. Read from
// tool_invocations, which the MCP boundary writes for every tool.
import { Fragment, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronDown, ChevronRight, Search, Wrench } from "lucide-react";
import { api } from "@/lib/api";
import type { ToolCall } from "@/lib/types";
import { EmptyState, ErrorBanner, FilterChips, JsonView, Skeleton, StatusPill, Toolbar } from "@/components/ui";
import { toolLabel as clientLabel } from "@/pages/QueryLog";
import { cn, fmtDateTime, fmtRelative } from "@/lib/utils";

// MCP tool names in the user's words; the raw name rides in `title`.
export const MCP_TOOL_TEXT: Record<string, string> = {
  remember: "Saved a memory", memory_store: "Saved a memory", memory_learn: "Learned a solution",
  recall: "Recalled memory", memory_recall: "Recalled memory", memory_search: "Searched memory",
  memory_search_documents: "Searched documents", memory_outline: "Outlined memory", memory_keys: "Scanned memory keys",
  memory_synthesize: "Synthesized an answer", status: "Checked status", memory_stats: "Checked status",
  forget: "Forgot a memory", memory_forget: "Forgot a memory", memory_ingest: "Ingested a document",
  memory_import: "Imported memories", memory_export: "Exported memories", memory_list: "Listed memories",
  memory_conflicts: "Listed conflicts", memory_gaps: "Listed gaps", memory_record_gap: "Recorded a gap",
  memory_schedule: "Scheduled a run", memory_schedules: "Listed scheduled runs", memory_unschedule: "Removed a scheduled run",
};
export const mcpToolText = (tool: string): string => MCP_TOOL_TEXT[tool] ?? tool.replace(/^memory_/, "").replace(/_/g, " ");

/** The one argument worth a column: the question, the text, the key. */
export function askedText(args: Record<string, unknown> | null): string {
  if (!args) return "";
  for (const k of ["query", "question", "text", "value", "key", "task", "crystal_id", "document", "topic"]) {
    const v = args[k];
    if (typeof v === "string" && v.trim()) return v;
  }
  const first = Object.values(args).find((v) => typeof v === "string" && (v as string).trim());
  return typeof first === "string" ? first : "";
}

function matches(c: ToolCall, term: string): boolean {
  const t = term.trim().toLowerCase();
  if (!t) return true;
  return [mcpToolText(c.tool), c.tool, clientLabel(c.client), askedText(c.args), c.result_summary ?? ""].some((s) => s.toLowerCase().includes(t));
}

export function ToolCalls({ customerId }: { customerId: string }) {
  const [term, setTerm] = useState("");
  const [tool, setTool] = useState<string | null>(null);
  const [client, setClient] = useState<string | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);
  const list = useQuery({ queryKey: ["tool-calls", customerId], queryFn: () => api.listToolCalls(customerId, { limit: 200 }), refetchInterval: 15_000 });
  const calls = list.data?.calls ?? [];
  const searched = useMemo(() => calls.filter((c) => matches(c, term)), [calls, term]);
  const rows = searched.filter((c) => (!tool || mcpToolText(c.tool) === tool) && (!client || clientLabel(c.client) === client));
  const tools = [...new Set(searched.map((c) => mcpToolText(c.tool)))];
  const clients = [...new Set(searched.map((c) => clientLabel(c.client)))];
  const countTool = (t: string | null) => searched.filter((c) => (!t || mcpToolText(c.tool) === t) && (!client || clientLabel(c.client) === client)).length;
  const countClient = (k: string | null) => searched.filter((c) => (!k || clientLabel(c.client) === k) && (!tool || mcpToolText(c.tool) === tool)).length;
  return (
    <div>
      <p className="mb-2 text-[13px] text-fg-2">{list.isLoading ? "Loading…" : `${calls.length.toLocaleString()} call${calls.length === 1 ? "" : "s"} · the latest 200`}</p>
      <div className="mb-4">
        <Toolbar search={
          <label className="relative block">
            <Search aria-hidden className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-fg-3" />
            <input value={term} onChange={(e) => setTerm(e.target.value)} placeholder="Search tool calls" aria-label="Search tool calls"
              className="w-full rounded-md border border-line bg-inset py-1.5 pl-8 pr-2.5 text-[13px] text-fg outline-none placeholder:text-fg-3 focus:border-accent" />
          </label>
        }>
          {tools.length > 1 && (
            <FilterChips label="Tool" value={tool ?? "any"} onChange={(v) => setTool(v === "any" || v === null ? null : v)}
              options={[{ value: "any", label: "Any", count: countTool(null) }, ...tools.map((t) => ({ value: t, label: t, count: countTool(t) }))]} />
          )}
          {clients.length > 1 && (
            <FilterChips label="Client" value={client ?? "any"} onChange={(v) => setClient(v === "any" || v === null ? null : v)}
              options={[{ value: "any", label: "Any", count: countClient(null) }, ...clients.map((k) => ({ value: k, label: k, count: countClient(k) }))]} />
          )}
        </Toolbar>
      </div>
      {list.isError && <ErrorBanner title="Couldn't load tool calls" error={list.error} onRetry={() => list.refetch()} />}
      {list.isLoading && <Skeleton rows={5} />}
      {!list.isLoading && !list.isError && calls.length === 0 && (
        <EmptyState icon={Wrench} title="No tool calls yet" help="Every call a connected tool makes (remember, recall, schedule, forget…) shows up here as it happens." />
      )}
      {!list.isLoading && calls.length > 0 && rows.length === 0 && <EmptyState icon={Search} title="Nothing matches" help="Try another phrase or filter." />}
      {rows.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-line bg-surface">
          <table className="w-full min-w-[560px] text-[13px]">
            <thead>
              <tr className="text-left text-[11px] uppercase tracking-[0.08em] text-fg-3">
                <th className="px-3 py-2 font-medium">Time</th>
                <th className="hidden px-3 py-2 font-medium min-[720px]:table-cell">Client</th>
                <th className="px-3 py-2 font-medium">Tool</th>
                <th className="px-3 py-2 font-medium">Asked</th>
                <th className="px-3 py-2 font-medium">Outcome</th>
                <th className="hidden px-3 py-2 text-right font-medium min-[720px]:table-cell">Took</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((c) => {
                const open = openId === c.id;
                const asked = askedText(c.args);
                return (
                  <Fragment key={c.id}>
                    <tr className={cn("cursor-pointer border-t border-line hover:bg-inset", open && "bg-inset")} onClick={() => setOpenId(open ? null : c.id)} aria-expanded={open}>
                      <td className="whitespace-nowrap px-3 py-2 text-[12px] text-fg-3" title={c.created_at ? fmtDateTime(c.created_at) : undefined}>
                        <span className="inline-flex items-center gap-1">{open ? <ChevronDown aria-hidden className="h-3.5 w-3.5" /> : <ChevronRight aria-hidden className="h-3.5 w-3.5" />}{fmtRelative(c.created_at)}</span>
                      </td>
                      <td className="hidden whitespace-nowrap px-3 py-2 text-fg-2 min-[720px]:table-cell" title={c.client ?? undefined}>{clientLabel(c.client)}</td>
                      <td className="whitespace-nowrap px-3 py-2 text-fg" title={c.tool}>{mcpToolText(c.tool)}</td>
                      <td className="max-w-[320px] truncate px-3 py-2 text-fg-2" title={asked}>{asked || <span className="text-fg-3">—</span>}</td>
                      <td className="px-3 py-2"><StatusPill variant={c.ok ? "ok" : "bad"} raw={c.code ?? (c.ok ? "ok" : "error")}>{c.ok ? "OK" : c.code || "Failed"}</StatusPill></td>
                      <td className="hidden whitespace-nowrap px-3 py-2 text-right text-fg-2 min-[720px]:table-cell">{c.duration_ms != null ? `${c.duration_ms} ms` : "—"}</td>
                    </tr>
                    {open && (
                      <tr className="border-t border-line bg-inset/60">
                        <td colSpan={6} className="px-3 py-3">
                          <div className="grid gap-4 min-[720px]:grid-cols-[minmax(0,1fr)_240px]">
                            <div className="min-w-0 space-y-3">
                              {c.result_summary && (
                                <div>
                                  <p className="text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">{c.ok ? "What came back" : "What went wrong"}</p>
                                  <p className="mt-1 whitespace-pre-wrap break-words text-[13px] text-fg">{c.result_summary}</p>
                                </div>
                              )}
                              <div>
                                <p className="text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">Arguments</p>
                                <JsonView value={c.args ?? {}} className="mt-1" />
                              </div>
                            </div>
                            <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[12.5px]">
                              <dt className="text-fg-3">Client</dt><dd className="text-fg" title={c.client ?? undefined}>{clientLabel(c.client)}</dd>
                              <dt className="text-fg-3">Tool</dt><dd className="text-fg">{c.tool}</dd>
                              <dt className="text-fg-3">Surface</dt><dd className="text-fg">{c.surface}</dd>
                              {c.created_at && <><dt className="text-fg-3">At</dt><dd className="text-fg">{fmtDateTime(c.created_at)}</dd></>}
                              {c.query_log_id && <><dt className="text-fg-3">Question</dt><dd className="text-fg">Listed under Questions</dd></>}
                            </dl>
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
