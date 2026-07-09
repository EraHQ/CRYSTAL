import { ReactNode, useState, Fragment } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowDown, ArrowUp } from "lucide-react";
import { api } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { Disclosure, EmptyState, ErrorBanner, JsonView, LoadingRows, MatchPill } from "@/components/ui";
import type { QueryLogSummary } from "@/lib/types";
import { cn, fmtDateTime, fmtNum, fmtSigned, truncate } from "@/lib/utils";

type SortKey = "timestamp" | "match_type" | "completion" | "latency";
type SortDir = "asc" | "desc";

export function QueryLog() {
  const { selectedCustomerId } = useSelectedCustomer();
  const [sortKey, setSortKey] = useState<SortKey>("timestamp");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const list = useQuery({ queryKey: ["query_logs", selectedCustomerId], queryFn: () => api.listQueryLogs(selectedCustomerId!, { limit: 100 }), enabled: !!selectedCustomerId });

  if (!selectedCustomerId) return <EmptyState title="No customer selected" />;
  if (list.isError) return <ErrorBanner title="Couldn't load logs" message={String(list.error)} />;

  const items = (list.data?.items ?? []).slice();
  const total = list.data?.total ?? 0;

  const sorted = items.sort((a, b) => {
    const dir = sortDir === "asc" ? 1 : -1;
    switch (sortKey) {
      case "timestamp": return (new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()) * dir;
      case "match_type": { const o: Record<string, number> = { high: 3, medium: 2, low: 1, none: 0 }; return ((o[a.match_type] ?? 0) - (o[b.match_type] ?? 0)) * dir; }
      case "completion": return ((a.completion_tokens ?? 0) - (b.completion_tokens ?? 0)) * dir;
      case "latency": return ((a.latency_ms ?? 0) - (b.latency_ms ?? 0)) * dir;
    }
  });

  const toggleSort = (key: SortKey) => { if (sortKey === key) setSortDir((d) => d === "asc" ? "desc" : "asc"); else { setSortKey(key); setSortDir("desc"); } };

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-lg font-semibold text-gray-900">Query Log</h1>
        <p className="text-sm text-gray-500">{Math.min(items.length, 100)} of {total} queries</p>
      </div>

      <div className="rounded-lg border border-gray-200 bg-white overflow-hidden">
        <table className="min-w-full divide-y divide-gray-100 text-sm">
          <thead className="bg-gray-50/60">
            <tr>
              <SortTh label="Time" k="timestamp" current={sortKey} dir={sortDir} toggle={toggleSort} />
              <th className="px-4 py-2.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Query</th>
              <SortTh label="Match" k="match_type" current={sortKey} dir={sortDir} toggle={toggleSort} />
              <th className="px-4 py-2.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Crystal</th>
              <SortTh label="Tokens" k="completion" current={sortKey} dir={sortDir} toggle={toggleSort} />
              <SortTh label="Latency" k="latency" current={sortKey} dir={sortDir} toggle={toggleSort} />
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-50">
            {list.isLoading && <LoadingRows rows={6} cols={6} />}
            {!list.isLoading && sorted.map((q) => (
              <Fragment key={q.id}>
                <tr onClick={() => setExpandedId(expandedId === q.id ? null : q.id)} className="cursor-pointer hover:bg-gray-50 transition-colors">
                  <td className="px-4 py-2.5 text-xs text-gray-400 whitespace-nowrap">{fmtDateTime(q.timestamp)}</td>
                  <td className="px-4 py-2.5 text-gray-700">{truncate(q.query_text, 60)}</td>
                  <td className="px-4 py-2.5"><MatchPill matchType={q.match_type} /></td>
                  <td className="px-4 py-2.5 font-mono text-xs text-gray-500">
                    {q.matched_facts[0] ?? <span className="text-gray-300">—</span>}
                    {q.matched_facts.length > 1 && <span className="text-gray-300 ml-1">+{q.matched_facts.length - 1}</span>}
                  </td>
                  <td className="px-4 py-2.5 text-gray-600">
                    {fmtNum(q.completion_tokens)}
                    {q.shadow_delta !== null && <span className={cn("ml-1 text-xs", q.shadow_delta < 0 ? "text-emerald-600" : "text-red-500")}>({fmtSigned(Math.round(q.shadow_delta))})</span>}
                  </td>
                  <td className="px-4 py-2.5 text-gray-600">{q.latency_ms != null ? `${q.latency_ms}ms` : "—"}</td>
                </tr>
                {expandedId === q.id && (
                  <tr className="bg-gray-50/50"><td colSpan={6} className="px-4 py-4"><ExpandedRow log={q} /></td></tr>
                )}
              </Fragment>
            ))}
            {!list.isLoading && sorted.length === 0 && <tr><td colSpan={6} className="px-4 py-12 text-center text-sm text-gray-400">No queries yet.</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SortTh({ label, k, current, dir, toggle }: { label: string; k: SortKey; current: SortKey; dir: SortDir; toggle: (k: SortKey) => void }) {
  const active = k === current;
  return (
    <th onClick={() => toggle(k)} className="cursor-pointer select-none px-4 py-2.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wider hover:text-gray-900">
      <span className="inline-flex items-center gap-1">{label}{active && (dir === "asc" ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />)}</span>
    </th>
  );
}

function ExpandedRow({ log }: { log: QueryLogSummary }) {
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm md:grid-cols-4">
        <Field label="Match" value={<MatchPill matchType={log.match_type} />} />
        <Field label="Injection" value={<code className="font-mono text-xs text-brand-600">{log.injection_method}</code>} />
        <Field label="Prompt tokens" value={fmtNum(log.prompt_tokens)} />
        {/* S12: prompt-caching split — prompt_tokens is only the
            NON-cached delta on agent turns. */}
        <Field label="Cache read" value={fmtNum(log.cache_read_tokens)} />
        <Field label="Cache write" value={fmtNum(log.cache_creation_tokens)} />
        <Field label="Overhead" value={log.prompt_token_overhead != null ? fmtSigned(log.prompt_token_overhead) : "—"} />
        <Field label="Completion" value={fmtNum(log.completion_tokens)} />
        <Field label="Shadow delta" value={log.shadow_delta != null ? fmtSigned(Math.round(log.shadow_delta)) : "—"} />
        <Field label="Latency" value={log.latency_ms != null ? `${log.latency_ms}ms` : "—"} />
        <Field label="Concept" value={log.concept_top_score != null ? log.concept_top_score.toFixed(3) : "—"} />
      </div>
      {log.query_text && <Disclosure title="Query" defaultOpen><p className="text-sm text-gray-600 whitespace-pre-wrap">{log.query_text}</p></Disclosure>}
      {log.response_text && <Disclosure title="Response"><p className="text-sm text-gray-600 whitespace-pre-wrap">{log.response_text}</p></Disclosure>}
      {log.concept_payload && <Disclosure title="Decomposition"><JsonView value={log.concept_payload} /></Disclosure>}
      <Disclosure title="Full log"><JsonView value={log} /></Disclosure>
    </div>
  );
}

function Field({ label, value }: { label: string; value: ReactNode }) {
  return <div><p className="text-[11px] uppercase tracking-wider text-gray-400">{label}</p><p className="mt-0.5 text-sm text-gray-900">{value}</p></div>;
}
