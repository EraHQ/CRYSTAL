// The Command center's pulse (Q110 B, 2026-10-24): six panels and a
// 12-week heatmap over GET .../pulse. Chart rules (dataviz method): one
// axis per chart, thin marks, categorical slots --c1..--c5 in a fixed
// order (validated in both themes; light c3-c5 sit under 3:1, so every
// legend carries its numbers), the heatmap is one hue light->dark, every
// chart has a hover tooltip and a "Show numbers" table, and text wears
// text tokens, never a series color.
import { useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { Area, AreaChart, Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "@/lib/api";
import type { Pulse as PulseData } from "@/lib/types";
import { ErrorBanner, Meter, Skeleton } from "@/components/ui";
import { cn } from "@/lib/utils";

const SERIES = ["rgb(var(--c1))", "rgb(var(--c2))", "rgb(var(--c3))", "rgb(var(--c4))", "rgb(var(--c5))"];
const MUTED = "rgb(var(--bd))";
const ACCENT = "rgb(var(--ac))";
const TOOLTIP_STYLE = { fontSize: 11, backgroundColor: "rgb(var(--sf))", border: "1px solid rgb(var(--bd))", borderRadius: 8, color: "rgb(var(--tx))" };
const TOOLTIP_ITEM = { color: "rgb(var(--tx2))" };

export const KIND_LABEL: Record<string, string> = {
  conflicts: "Conflicts settled", assumptions: "Assumptions decided", gaps: "Gaps handled", facts: "Facts tidied", goals: "Goal sends",
};
const TIERS: { key: string; label: string; cls: string }[] = [
  { key: "whitelist", label: "Trusted", cls: "bg-ok" },
  { key: "neutral", label: "Neutral", cls: "bg-info" },
  { key: "quarantine", label: "Needs review", cls: "bg-warn" },
  { key: "blacklist", label: "Blocked", cls: "bg-bad" },
];

export function shortDay(iso: string): string {
  const d = new Date(`${iso}T00:00:00Z`);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", timeZone: "UTC" });
}

/** Minutes in words: "42 min", "3.5 h", "2 days". */
export function minutesText(m: number | null): string {
  if (m == null) return "No replies yet";
  if (m < 60) return `${Math.round(m)} min`;
  if (m < 48 * 60) return `${(m / 60).toFixed(m < 600 ? 1 : 0)} h`;
  return `${Math.round(m / 1440)} days`;
}

/** The heatmap's five steps: 0 is empty; the rest split the busiest day. */
export function heatLevel(n: number, max: number): number {
  if (n <= 0 || max <= 0) return 0;
  return Math.min(4, Math.max(1, Math.ceil((n / max) * 4)));
}

function Panel({ title, headline, sub, children, table }: {
  title: string; headline?: string; sub?: string; children?: ReactNode; table?: { head: string[]; rows: (string | number)[][] };
}) {
  const [showTable, setShowTable] = useState(false);
  return (
    <section aria-label={title} className="flex min-w-0 flex-col rounded-xl border border-line bg-surface p-4">
      <div className="flex items-start justify-between gap-2">
        <h3 className="text-[12.5px] font-medium text-fg-2">{title}</h3>
        {table && (
          <button type="button" className="text-[11.5px] text-fg-3 hover:text-fg" aria-pressed={showTable}
            onClick={() => setShowTable((v) => !v)}>{showTable ? "Show chart" : "Show numbers"}</button>
        )}
      </div>
      {headline && <p className="mt-1 text-[24px] font-[650] leading-tight tracking-tight text-fg tabular-nums">{headline}</p>}
      {sub && <p className="text-[12px] text-fg-3">{sub}</p>}
      {showTable && table ? (
        <div className="mt-2 max-h-[180px] overflow-auto">
          <table className="w-full text-[12px]">
            <thead><tr>{table.head.map((h) => <th key={h} className="pb-1 text-left font-medium text-fg-3">{h}</th>)}</tr></thead>
            <tbody>{table.rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j} className="py-0.5 text-fg-2 tabular-nums">{c}</td>)}</tr>)}</tbody>
          </table>
        </div>
      ) : children && <div className="mt-2 min-w-0 flex-1">{children}</div>}
    </section>
  );
}

function Growth({ p }: { p: PulseData }) {
  const g = p.growth;
  return (
    <Panel title="Memory" headline={g.total.toLocaleString()} sub={`facts · ${g.week_added >= 0 ? "+" : ""}${g.week_added.toLocaleString()} this week`}
      table={{ head: ["Day", "Facts"], rows: g.days.map((d) => [shortDay(d.day), d.facts]) }}>
      <ResponsiveContainer width="100%" height={84}>
        <AreaChart data={g.days} margin={{ top: 4, right: 0, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="growthFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={ACCENT} stopOpacity={0.28} />
              <stop offset="100%" stopColor={ACCENT} stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis dataKey="day" hide />
          <YAxis hide domain={["dataMin", "dataMax"]} />
          <Tooltip contentStyle={TOOLTIP_STYLE} itemStyle={TOOLTIP_ITEM} labelStyle={TOOLTIP_ITEM}
            labelFormatter={(d) => shortDay(String(d))} formatter={(v) => [Number(v).toLocaleString(), "facts"]} />
          <Area type="monotone" dataKey="facts" stroke={ACCENT} strokeWidth={2} fill="url(#growthFill)" isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </Panel>
  );
}

function Recall({ p }: { p: PulseData }) {
  const r = p.recall;
  const data = r.days.map((d) => ({ day: d.day, found: d.found, missed: d.asked - d.found }));
  return (
    <Panel title="Recall, 14 days" headline={r.found_pct == null ? "No questions yet" : `${r.found_pct}%`}
      sub={`${r.found.toLocaleString()} of ${r.asked.toLocaleString()} questions found memory`}
      table={{ head: ["Day", "Asked", "Found"], rows: r.days.map((d) => [shortDay(d.day), d.asked, d.found]) }}>
      <ResponsiveContainer width="100%" height={84}>
        <BarChart data={data} margin={{ top: 4, right: 0, bottom: 0, left: 0 }} barCategoryGap={2}>
          <XAxis dataKey="day" hide />
          <YAxis hide />
          <Tooltip contentStyle={TOOLTIP_STYLE} itemStyle={TOOLTIP_ITEM} labelStyle={TOOLTIP_ITEM} cursor={{ fill: "rgb(var(--in))" }}
            labelFormatter={(d) => shortDay(String(d))} />
          <Bar dataKey="found" name="Found memory" stackId="r" fill={SERIES[0]} isAnimationActive={false} />
          <Bar dataKey="missed" name="Nothing found" stackId="r" fill={MUTED} radius={[3, 3, 0, 0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
      <Legend items={[{ color: SERIES[0], label: "Found memory", value: r.found }, { color: MUTED, label: "Nothing found", value: r.asked - r.found }]} />
    </Panel>
  );
}

function Legend({ items }: { items: { color: string; label: string; value: number }[] }) {
  return (
    <ul className="mt-2 flex flex-wrap gap-x-3 gap-y-1" aria-label="Legend">
      {items.map((i) => (
        <li key={i.label} className="flex items-center gap-1.5 text-[11.5px] text-fg-2">
          <span aria-hidden className="h-2 w-2 rounded-sm" style={{ backgroundColor: i.color }} />
          {i.label} <span className="tabular-nums text-fg">{i.value.toLocaleString()}</span>
        </li>
      ))}
    </ul>
  );
}

function Autonomy({ p }: { p: PulseData }) {
  const a = p.autonomy;
  const totals = a.kinds.map((k) => a.days.reduce((s, d) => s + (d[k as keyof typeof d] as number), 0));
  return (
    <Panel title="Crystal on its own, 14 days" headline={a.week.toLocaleString()} sub="acts nobody had to click, this week"
      table={{ head: ["Day", ...a.kinds.map((k) => KIND_LABEL[k] ?? k)], rows: a.days.map((d) => [shortDay(d.day), ...a.kinds.map((k) => d[k as keyof typeof d] as number)]) }}>
      <ResponsiveContainer width="100%" height={84}>
        <BarChart data={a.days} margin={{ top: 4, right: 0, bottom: 0, left: 0 }} barCategoryGap={2}>
          <XAxis dataKey="day" hide />
          <YAxis hide />
          <Tooltip contentStyle={TOOLTIP_STYLE} itemStyle={TOOLTIP_ITEM} labelStyle={TOOLTIP_ITEM} cursor={{ fill: "rgb(var(--in))" }}
            labelFormatter={(d) => shortDay(String(d))} />
          {a.kinds.map((k, i) => (
            <Bar key={k} dataKey={k} name={KIND_LABEL[k] ?? k} stackId="a" fill={SERIES[i % SERIES.length]}
              stroke="rgb(var(--sf))" strokeWidth={1} isAnimationActive={false} />
          ))}
        </BarChart>
      </ResponsiveContainer>
      <Legend items={a.kinds.map((k, i) => ({ color: SERIES[i], label: KIND_LABEL[k] ?? k, value: totals[i] }))} />
    </Panel>
  );
}

function Health({ p }: { p: PulseData }) {
  const h = p.health;
  const total = Object.values(h.tiers).reduce((s, n) => s + n, 0);
  const trusted = h.tiers.whitelist ?? 0;
  return (
    <Panel title="Bank health" headline={total ? `${Math.round((trusted / total) * 100)}%` : "Empty"} sub="of facts are trusted">
      {total > 0 && (
        <div className="flex h-2.5 w-full gap-0.5 overflow-hidden rounded-full" role="img"
          aria-label={TIERS.map((t) => `${t.label} ${h.tiers[t.key] ?? 0}`).join(", ")}>
          {TIERS.filter((t) => (h.tiers[t.key] ?? 0) > 0).map((t) => (
            <div key={t.key} className={t.cls} style={{ width: `${((h.tiers[t.key] ?? 0) / total) * 100}%` }}
              title={`${t.label}: ${(h.tiers[t.key] ?? 0).toLocaleString()}`} />
          ))}
        </div>
      )}
      <ul className="mt-2 flex flex-wrap gap-x-3 gap-y-1" aria-label="Tiers">
        {TIERS.map((t) => (
          <li key={t.key} className="flex items-center gap-1.5 text-[11.5px] text-fg-2">
            <span aria-hidden className={cn("h-2 w-2 rounded-sm", t.cls)} />{t.label} <span className="tabular-nums text-fg">{(h.tiers[t.key] ?? 0).toLocaleString()}</span>
          </li>
        ))}
      </ul>
      <p className="mt-2 text-[12px] text-fg-2"><span className="tabular-nums text-fg">{h.open_conflicts}</span> open conflicts · <span className="tabular-nums text-fg">{h.open_gaps}</span> open gaps</p>
    </Panel>
  );
}

function GoalsAtWork({ p }: { p: PulseData }) {
  const g = p.goals;
  return (
    <Panel title="Goals at work, this week" headline={minutesText(g.median_to_draft_minutes)}
      sub={g.replies ? `median from an email arriving to a drafted reply (${g.replies} email${g.replies === 1 ? "" : "s"})` : "Reply time shows once a goal drafts from your mail"}>
      <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-[12px] min-[420px]:grid-cols-4">
        {([["Sent", g.week.sent], ["Drafted", g.week.drafted], ["To-dos filed", g.week.filed], ["Done", g.week.done]] as const).map(([k, v]) => (
          <div key={k}><dt className="text-fg-3">{k}</dt><dd className="text-[15px] font-semibold tabular-nums text-fg">{v}</dd></div>
        ))}
      </dl>
      {g.median_to_sent_minutes != null && <p className="mt-2 text-[12px] text-fg-2">Sent replies went out in a median {minutesText(g.median_to_sent_minutes)}.</p>}
    </Panel>
  );
}

function Capacity({ p }: { p: PulseData }) {
  const pct = p.capacity.pct;
  return (
    <Panel title="AI capacity today" headline={pct == null ? "Your key" : `${Math.round(pct)}%`}
      sub={pct == null ? "You bring your own model key, so there is no daily allowance." : "of today's allowance used; it refills at midnight UTC"}>
      {pct != null && <Meter value={pct} label="AI capacity used today" hideLabel />}
    </Panel>
  );
}

function Heatmap({ p }: { p: PulseData }) {
  const cells = p.heatmap.map((h) => ({ ...h, total: h.writes + h.recalls + h.acts }));
  const max = Math.max(0, ...cells.map((c) => c.total));
  // Columns are weeks, rows Sunday..Saturday; pad the first week so days line up.
  const first = new Date(`${cells[0]?.day ?? "2026-01-01"}T00:00:00Z`).getUTCDay();
  const padded: (typeof cells[number] | null)[] = [...Array(first).fill(null), ...cells];
  const weeks: (typeof cells[number] | null)[][] = [];
  for (let i = 0; i < padded.length; i += 7) weeks.push(padded.slice(i, i + 7));
  const active = cells.filter((c) => c.total > 0).length;
  const steps = ["bg-inset", "bg-accent/25", "bg-accent/50", "bg-accent/75", "bg-accent"];
  return (
    <section aria-label="Activity, 12 weeks" className="rounded-xl border border-line bg-surface p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-[12.5px] font-medium text-fg-2">Activity, 12 weeks</h3>
        <p className="text-[12px] text-fg-3"><span className="tabular-nums text-fg">{active}</span> active days · writes, recalls and Crystal's own acts</p>
      </div>
      <div className="mt-3 overflow-x-auto">
        <div className="flex gap-[3px]" role="grid" aria-label="One cell per day">
          {weeks.map((w, i) => (
            <div key={i} className="flex flex-col gap-[3px]" role="row">
              {w.map((c, j) => c ? (
                <div key={c.day} role="gridcell" data-level={heatLevel(c.total, max)}
                  title={`${shortDay(c.day)}: ${c.writes} written, ${c.recalls} recalled, ${c.acts} on its own`}
                  aria-label={`${shortDay(c.day)}: ${c.total} events`}
                  className={cn("h-3 w-3 rounded-[3px] min-[720px]:h-4 min-[720px]:w-4", steps[heatLevel(c.total, max)])} />
              ) : <div key={`pad${j}`} className="h-3 w-3 min-[720px]:h-4 min-[720px]:w-4" aria-hidden />)}
            </div>
          ))}
        </div>
      </div>
      <div className="mt-2 flex items-center gap-1 text-[11px] text-fg-3" aria-hidden>
        Less {steps.map((s) => <span key={s} className={cn("h-2.5 w-2.5 rounded-[3px]", s)} />)} More
      </div>
    </section>
  );
}

export function Pulse({ customerId }: { customerId: string }) {
  const pulse = useQuery({ queryKey: ["pulse", customerId], queryFn: () => api.getPulse(customerId), refetchInterval: 120_000 });
  if (pulse.isError) return <ErrorBanner title="Couldn't load the stats" error={pulse.error} onRetry={() => pulse.refetch()} />;
  if (!pulse.data) return <Skeleton className="h-[220px] w-full rounded-xl" />;
  const p = pulse.data;
  return (
    <div className="space-y-3">
      <div className="grid gap-3 min-[720px]:grid-cols-2 min-[1100px]:grid-cols-3">
        <Growth p={p} />
        <Recall p={p} />
        <Autonomy p={p} />
        <Health p={p} />
        <GoalsAtWork p={p} />
        <Capacity p={p} />
      </div>
      <Heatmap p={p} />
    </div>
  );
}
