// Activity → Model calls (2026-10-19): every model call Crystal made for
// this memory (the llm_calls ledger), newest first: when, what it was
// for, which model, tokens in/out/cached; platform admins also see the
// cost. A call that belongs to a run links to its pipeline. The same
// table serves a run's Calls tab and a question's trace.
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { Cpu } from "lucide-react";
import { api } from "@/lib/api";
import { EmptyState, ErrorBanner, FilterChips, Skeleton } from "@/components/ui";
import { activityOf, ACTIVITIES, originText } from "@/lib/activities";
import { modelLabel } from "@/pages/settings/ModelInference";
import { fmtDateTime, fmtRelative } from "@/lib/utils";
import { fmtUsd } from "./Trace";

export function ModelCalls({ customerId, admin }: { customerId: string; admin: boolean }) {
  const [activity, setActivity] = useState<string | null>(null);
  const [model, setModel] = useState<string | null>(null);
  const list = useQuery({ queryKey: ["model-calls", customerId], queryFn: () => api.listModelCalls(customerId, { limit: 200 }), refetchInterval: 15_000 });
  const calls = list.data?.calls ?? [];
  const rows = useMemo(() => calls.filter((c) => (!activity || activityOf(c.origin) === activity) && (!model || modelLabel(c.model) === model)), [calls, activity, model]);
  const activities = ACTIVITIES.filter((a) => calls.some((c) => activityOf(c.origin) === a));
  const models = [...new Set(calls.map((c) => modelLabel(c.model)))];
  const countActivity = (a: string | null) => calls.filter((c) => (!a || activityOf(c.origin) === a) && (!model || modelLabel(c.model) === model)).length;
  const countModel = (m: string | null) => calls.filter((c) => (!m || modelLabel(c.model) === m) && (!activity || activityOf(c.origin) === activity)).length;
  const total = rows.reduce((n, c) => n + c.input_tokens + c.output_tokens + c.cache_creation_tokens + c.cache_read_tokens, 0);
  return (
    <div>
      <p className="mb-2 text-[13px] text-fg-2">{list.isLoading ? "Loading…" : `${calls.length.toLocaleString()} call${calls.length === 1 ? "" : "s"} · the latest 200 · ${total.toLocaleString()} tokens shown`}</p>
      {(activities.length > 1 || models.length > 1) && (
        <div className="mb-4 flex flex-wrap gap-2">
          {activities.length > 1 && (
            <FilterChips label="Activity" value={activity ?? "any"} onChange={(v) => setActivity(v === "any" || v === null ? null : v)}
              options={[{ value: "any", label: "Any", count: countActivity(null) }, ...activities.map((a) => ({ value: a, label: a, count: countActivity(a) }))]} />
          )}
          {models.length > 1 && (
            <FilterChips label="Model" value={model ?? "any"} onChange={(v) => setModel(v === "any" || v === null ? null : v)}
              options={[{ value: "any", label: "Any", count: countModel(null) }, ...models.map((m) => ({ value: m, label: m, count: countModel(m) }))]} />
          )}
        </div>
      )}
      {list.isError && <ErrorBanner title="Couldn't load model calls" error={list.error} onRetry={() => list.refetch()} />}
      {list.isLoading && <Skeleton rows={5} />}
      {!list.isLoading && !list.isError && calls.length === 0 && (
        <EmptyState icon={Cpu} title="No model calls yet" help="Every call to a model, for answers, research or curation, shows up here as it happens." />
      )}
      {rows.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-line bg-surface">
          <table className="w-full min-w-[560px] text-[13px]">
            <thead>
              <tr className="text-left text-[11px] uppercase tracking-[0.08em] text-fg-3">
                <th className="px-3 py-2 font-medium">Time</th>
                <th className="px-3 py-2 font-medium">What</th>
                <th className="px-3 py-2 font-medium">Model</th>
                <th className="px-3 py-2 text-right font-medium">In</th>
                <th className="px-3 py-2 text-right font-medium">Out</th>
                <th className="hidden px-3 py-2 text-right font-medium min-[720px]:table-cell">Cached</th>
                {admin && <th className="px-3 py-2 text-right font-medium">Cost</th>}
                <th className="hidden px-3 py-2 font-medium min-[720px]:table-cell">Part of</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((c) => {
                const run = c.session_id && c.session_id.startsWith("env_") ? c.session_id : null;
                return (
                  <tr key={c.id} className="border-t border-line">
                    <td className="whitespace-nowrap px-3 py-2 text-[12px] text-fg-3" title={c.created_at ? fmtDateTime(c.created_at) : undefined}>{fmtRelative(c.created_at)}</td>
                    <td className="px-3 py-2 text-fg" title={c.origin}>{originText(c.origin)}</td>
                    <td className="px-3 py-2 text-fg-2" title={c.model}>{modelLabel(c.model)}</td>
                    <td className="px-3 py-2 text-right tabular-nums text-fg-2">{c.input_tokens.toLocaleString()}</td>
                    <td className="px-3 py-2 text-right tabular-nums text-fg-2">{c.output_tokens.toLocaleString()}</td>
                    <td className="hidden px-3 py-2 text-right tabular-nums text-fg-2 min-[720px]:table-cell">{(c.cache_read_tokens + c.cache_creation_tokens).toLocaleString()}</td>
                    {admin && <td className="px-3 py-2 text-right tabular-nums text-fg-2">{fmtUsd(c.cost_micro_usd)}</td>}
                    <td className="hidden px-3 py-2 text-fg-2 min-[720px]:table-cell">
                      {run ? <Link to={`/activity/run/${encodeURIComponent(run)}`} className="text-accent hover:underline">a run</Link>
                        : c.session_id ? <span className="text-fg-3" title={c.session_id}>a conversation</span> : <span className="text-fg-3">—</span>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
