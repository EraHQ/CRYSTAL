// Activity → Questions → one question's trace (2026-10-19): the agent
// turn's self-report (every tool call with its input, what came back and
// how long it took, rendered the same way the chat's trace renders it),
// the model calls of that turn, the connected-tool call that produced a
// recall row, and the raw records behind each for anyone who wants every
// little detail. Loaded on demand when the disclosure opens.
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { AgentRunResponse, ModelCall, QueryTrace as QueryTraceShape, ToolCall } from "@/lib/types";
import { Disclosure, ErrorBanner, JsonView, Skeleton } from "@/components/ui";
import { buildTrace, TraceStepRow } from "@/pages/ChatPlayground";
import { modelLabel } from "@/pages/settings/ModelInference";
import { originText } from "@/lib/activities";
import { fmtDateTime, fmtRelative } from "@/lib/utils";
import { mcpToolText } from "./ToolCalls";

const fmtTok = (n: number | null | undefined) => (n == null ? "—" : n.toLocaleString());
export const fmtUsd = (micro: number | undefined) => (micro == null ? "" : `$${(micro / 1_000_000).toFixed(4)}`);

/** The model ledger for one turn or one run, in a small table. */
export function ModelCallsTable({ calls, admin }: { calls: ModelCall[]; admin: boolean }) {
  if (calls.length === 0) return <p className="text-[12.5px] text-fg-3">No model calls recorded.</p>;
  const total = calls.reduce((n, c) => n + c.input_tokens + c.output_tokens + c.cache_creation_tokens + c.cache_read_tokens, 0);
  return (
    <div className="overflow-x-auto rounded-lg border border-line bg-surface">
      <table className="w-full min-w-[480px] text-[12.5px]">
        <thead>
          <tr className="text-left text-[11px] uppercase tracking-[0.08em] text-fg-3">
            <th className="px-2.5 py-1.5 font-medium">When</th>
            <th className="px-2.5 py-1.5 font-medium">What</th>
            <th className="px-2.5 py-1.5 font-medium">Model</th>
            <th className="px-2.5 py-1.5 text-right font-medium">In</th>
            <th className="px-2.5 py-1.5 text-right font-medium">Out</th>
            <th className="px-2.5 py-1.5 text-right font-medium">Cached</th>
            {admin && <th className="px-2.5 py-1.5 text-right font-medium">Cost</th>}
          </tr>
        </thead>
        <tbody>
          {calls.map((c) => (
            <tr key={c.id} className="border-t border-line">
              <td className="whitespace-nowrap px-2.5 py-1.5 text-fg-3" title={c.created_at ? fmtDateTime(c.created_at) : undefined}>{fmtRelative(c.created_at)}</td>
              <td className="px-2.5 py-1.5 text-fg" title={c.origin}>{originText(c.origin)}</td>
              <td className="px-2.5 py-1.5 text-fg-2" title={c.model}>{modelLabel(c.model)}</td>
              <td className="px-2.5 py-1.5 text-right tabular-nums text-fg-2">{fmtTok(c.input_tokens)}</td>
              <td className="px-2.5 py-1.5 text-right tabular-nums text-fg-2">{fmtTok(c.output_tokens)}</td>
              <td className="px-2.5 py-1.5 text-right tabular-nums text-fg-2">{fmtTok(c.cache_read_tokens + c.cache_creation_tokens)}</td>
              {admin && <td className="px-2.5 py-1.5 text-right tabular-nums text-fg-2">{fmtUsd(c.cost_micro_usd)}</td>}
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr className="border-t border-line text-fg-3">
            <td colSpan={3} className="px-2.5 py-1.5">{calls.length} call{calls.length === 1 ? "" : "s"}</td>
            <td colSpan={admin ? 4 : 3} className="px-2.5 py-1.5 text-right tabular-nums">{total.toLocaleString()} tokens</td>
          </tr>
        </tfoot>
      </table>
    </div>
  );
}

function ConnectedToolCall({ c }: { c: ToolCall }) {
  return (
    <div className="rounded-lg border border-line bg-surface px-3 py-2 text-[12.5px]">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium text-fg" title={c.tool}>{mcpToolText(c.tool)}</span>
        {c.client && <span className="text-fg-3">from {c.client}</span>}
        <span className={c.ok ? "text-ok" : "text-bad"}>{c.ok ? "ok" : c.code || "failed"}</span>
        {c.duration_ms != null && <span className="text-fg-3">{c.duration_ms} ms</span>}
      </div>
      {c.result_summary && <p className="mt-1 break-words text-fg-2">{c.result_summary}</p>}
      {c.args && <Disclosure title="Arguments" className="mt-2"><JsonView value={c.args} /></Disclosure>}
    </div>
  );
}

function Items({ title, items }: { title: string; items: Record<string, unknown>[] }) {
  if (items.length === 0) return null;
  return (
    <div>
      <p className="text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">{title}</p>
      <ul className="mt-1 space-y-1">
        {items.map((it, i) => {
          const text = typeof it === "string" ? it : String((it as { text?: unknown; claim?: unknown; gap?: unknown; summary?: unknown }).text
            ?? (it as { claim?: unknown }).claim ?? (it as { gap?: unknown }).gap ?? (it as { summary?: unknown }).summary ?? JSON.stringify(it));
          return <li key={i} className="rounded-md bg-inset px-2.5 py-1.5 text-[12.5px] text-fg">{text}</li>;
        })}
      </ul>
    </div>
  );
}

export function QueryTrace({ customerId, queryLogId, admin }: { customerId: string; queryLogId: string; admin: boolean }) {
  const q = useQuery({ queryKey: ["query-trace", customerId, queryLogId], queryFn: () => api.getQueryTrace(customerId, queryLogId) });
  if (q.isLoading) return <Skeleton rows={3} />;
  if (q.isError) return <ErrorBanner title="Couldn't load the trace" error={q.error} onRetry={() => q.refetch()} />;
  const t: QueryTraceShape | undefined = q.data;
  if (!t) return null;
  const calls = t.trace?.tool_calls ?? [];
  const stub: AgentRunResponse = { id: t.query_log_id, model: t.model ?? "", messages: [], final_text: "", stop_reason: "", iterations: 0, prompt_tokens: 0, completion_tokens: 0, tool_calls: calls };
  const { steps } = buildTrace(stub);
  const noTurn = !t.trace && t.tool_calls.length === 0;
  return (
    <div className="space-y-4">
      {noTurn && <p className="text-[12.5px] text-fg-3">No trace was recorded for this question (it predates tracing, or it came through the proxy).</p>}
      {t.trace && (
        <div>
          <p className="mb-1 text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">Steps{t.model ? ` · ${modelLabel(t.model)}` : ""}</p>
          {steps.length === 0 ? <p className="text-[12.5px] text-fg-3">No lookups: the answer came from the conversation alone.</p> : (
            <ol aria-label="Trace" className="space-y-1.5">
              {steps.map((s) => (
                <li key={s.call.tool_use_id}>
                  <TraceStepRow step={s} highlighted={false} />
                  <div className="mt-1 flex flex-wrap items-center gap-2 pl-3 text-[11.5px] text-fg-3">
                    {s.call.duration_ms != null && <span>{s.call.duration_ms} ms</span>}
                    <span>round {s.call.iteration}</span>
                  </div>
                  <Disclosure title="Raw call" className="mt-1"><JsonView value={{ input: s.call.input, output: s.call.output }} /></Disclosure>
                </li>
              ))}
            </ol>
          )}
        </div>
      )}
      {t.tool_calls.length > 0 && (
        <div>
          <p className="mb-1 text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">Connected tool</p>
          <div className="space-y-1.5">{t.tool_calls.map((c) => <ConnectedToolCall key={c.id} c={c} />)}</div>
        </div>
      )}
      {t.trace && (t.trace.inferences.length > 0 || t.trace.borders_crossed.length > 0 || t.trace.gaps_felt.length > 0) && (
        <div className="space-y-3">
          <Items title="Inferences the agent made" items={t.trace.inferences} />
          <Items title="Where it went beyond memory" items={t.trace.borders_crossed} />
          <Items title="Gaps it felt" items={t.trace.gaps_felt} />
        </div>
      )}
      <div>
        <p className="mb-1 text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">Model calls</p>
        <ModelCallsTable calls={t.model_calls} admin={admin} />
      </div>
      {t.trace && <Disclosure title="Raw self-report"><JsonView value={t.trace} /></Disclosure>}
    </div>
  );
}
