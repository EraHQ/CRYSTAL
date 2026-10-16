// System critiques — Console Refine PR 13 (docs/CONSOLE_UX_AUDIT.md 5.10).
// Crystal's structured complaints about ITSELF: tools, ingestion, retrieval
// (the substrate stream) and the critics' verdicts on its own answers (the
// response stream). Recorded and surfaced, never auto-acted (MCR Principle
// 9). Dismiss hides an observation (status dropped); Undo restores it.
// Platform-admin only (the route is not mounted for tenants).
import { useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, ChevronRight, MessageSquareWarning } from "lucide-react";
import { Link } from "react-router-dom";
import { api, errorMessage } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { ConfirmInline, EmptyState, ErrorBanner, FilterChips, PageHeader, Segmented, Skeleton, StatusPill, useToast } from "@/components/ui";
import type { StatusVariant } from "@/components/ui";
import { cn, fmtDateTime, fmtRelative, truncate } from "@/lib/utils";

type Severity = "high" | "medium" | "low";

export interface SubstrateObs {
  action_item: {
    id: string;
    created_at?: string | null;
    content?: { subsystem?: string; severity?: string; complaint?: string; crystal_id?: string | null } | null;
  };
  critique?: { critic_role?: string | null; critic_model?: string | null } | null;
  trace_summary?: { sequence_id?: string | null; turn_index?: number | null } | null;
}

interface QualityGroup {
  observation_type: string;
  count: number;
  latest: { critique_id: string; critic_role: string; created_at?: string | null; sequence_id?: string | null; detail?: Record<string, unknown> | null; summary_text?: string | null }[];
}

const SEVERITY: Record<Severity, { label: string; variant: StatusVariant }> = {
  high: { label: "High", variant: "bad" },
  medium: { label: "Medium", variant: "warn" },
  low: { label: "Low", variant: "neutral" },
};

export function severityOf(o: SubstrateObs): Severity {
  const s = o.action_item?.content?.severity;
  return s === "high" || s === "medium" || s === "low" ? s : "low";
}

export function subsystemOf(o: SubstrateObs): string {
  return o.action_item?.content?.subsystem || "unspecified";
}

// The subsystem names the agent and the structural critics write; shown
// capitalized, with the raw token in title.
export function subsystemLabel(subsystem: string): string {
  const words = subsystem.replace(/[_-]+/g, " ").trim();
  return words ? words.charAt(0).toUpperCase() + words.slice(1) : "Unspecified";
}

// Group observations by subsystem; newest first inside a group, groups by
// their newest observation. Counts are exact over what is loaded.
export function groupBySubsystem(observations: SubstrateObs[]): { subsystem: string; items: SubstrateObs[]; high: number; medium: number; latest: string | null }[] {
  const map = new Map<string, SubstrateObs[]>();
  for (const o of observations) {
    const k = subsystemOf(o);
    if (!map.has(k)) map.set(k, []);
    map.get(k)!.push(o);
  }
  const ts = (o: SubstrateObs) => new Date(o.action_item?.created_at ?? 0).getTime() || 0;
  return [...map.entries()]
    .map(([subsystem, items]) => {
      const sorted = [...items].sort((a, b) => ts(b) - ts(a));
      return {
        subsystem,
        items: sorted,
        high: sorted.filter((o) => severityOf(o) === "high").length,
        medium: sorted.filter((o) => severityOf(o) === "medium").length,
        latest: sorted[0]?.action_item?.created_at ?? null,
      };
    })
    .sort((a, b) => (new Date(b.latest ?? 0).getTime() || 0) - (new Date(a.latest ?? 0).getTime() || 0));
}

// The response-quality observation types the critics emit, in words.
export const OBS_LABELS: Record<string, string> = {
  assumption_identified: "Assumption not in evidence",
  generalization_from_thin_evidence: "Generalized from thin evidence",
  source_contradiction: "Contradicts a consulted source",
  tool_output_questionable: "Trusted a questionable tool output",
  gap_papered_over: "Gap papered over",
  border_crossing_unflagged: "Evidence to inference, unflagged",
  reasoning_skip: "Skipped a reasoning step",
};

function weekAgoIso(now = Date.now()): string {
  return new Date(now - 7 * 86_400_000).toISOString().replace(/\.\d{3}Z$/, "+00:00");
}

export function Critiques() {
  const { selectedCustomerId } = useSelectedCustomer();
  const queryClient = useQueryClient();
  const toast = useToast();
  const [tab, setTab] = useState<"substrate" | "response">("substrate");
  const [severity, setSeverity] = useState<Severity | null>(null);
  const [openGroup, setOpenGroup] = useState<string | null>(null);
  const [hidden, setHidden] = useState<Set<string>>(() => new Set());
  const [confirmAll, setConfirmAll] = useState(false);
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const flat = useQuery({
    queryKey: ["substrate-flat", selectedCustomerId],
    queryFn: () => api.listSubstrateObservations(selectedCustomerId!),
    enabled: !!selectedCustomerId,
    refetchInterval: 15_000,
  });
  const quality = useQuery({
    queryKey: ["quality-grouped", selectedCustomerId],
    queryFn: () => api.groupedQualityObservations(selectedCustomerId!),
    enabled: !!selectedCustomerId,
    refetchInterval: 15_000,
  });
  const week = useQuery({
    queryKey: ["quality-week", selectedCustomerId],
    queryFn: () => api.listQualityObservations(selectedCustomerId!, weekAgoIso()),
    enabled: !!selectedCustomerId,
    refetchInterval: 60_000,
  });

  const observations = useMemo(
    () => ((flat.data?.observations ?? []) as SubstrateObs[]).filter((o) => o.action_item?.id && !hidden.has(o.action_item.id)),
    [flat.data, hidden]
  );
  const shown = severity ? observations.filter((o) => severityOf(o) === severity) : observations;
  const groups = groupBySubsystem(shown);
  const countSev = (s: Severity | null) => (s ? observations.filter((o) => severityOf(o) === s).length : observations.length);

  const refresh = () => queryClient.invalidateQueries({ queryKey: ["substrate-flat"] });

  const restore = async (ids: string[]) => {
    try {
      for (const id of ids) await api.restoreSubstrateObservation(id);
    } catch (e) {
      setActionError(errorMessage(e, "Couldn't restore that observation."));
    }
    setHidden((prev) => { const next = new Set(prev); ids.forEach((id) => next.delete(id)); return next; });
    void refresh();
  };

  const dismiss = async (ids: string[]) => {
    setBusy(true);
    setActionError(null);
    const done: string[] = [];
    try {
      for (const id of ids) {
        await api.dismissSubstrateObservation(id);
        done.push(id);
      }
    } catch (e) {
      setActionError(errorMessage(e, "Couldn't dismiss that observation."));
    } finally {
      setBusy(false);
    }
    if (done.length > 0) {
      setHidden((prev) => new Set([...prev, ...done]));
      void refresh();
      toast(done.length === 1 ? "Observation dismissed" : `${done.length} observations dismissed`, { action: { label: "Undo", onClick: () => void restore(done) } });
    }
  };

  if (!selectedCustomerId) {
    return (
      <>
        <PageHeader title="System critiques" />
        <EmptyState icon={MessageSquareWarning} title="No customer selected" help="Pick a customer above to see what Crystal flagged about itself." />
      </>
    );
  }

  const openCount = observations.length;
  const weekCount = week.data?.total ?? 0;
  const meta = flat.isLoading ? "Loading…" : `${openCount} open about tools and pipeline · ${weekCount} answer-quality flag${weekCount === 1 ? "" : "s"} this week`;

  return (
    <div>
      <PageHeader title="System critiques" meta={meta}
        actions={tab === "substrate" && shown.length > 0 ? (
          <button type="button" onClick={() => setConfirmAll(true)} disabled={busy || confirmAll}
            className="rounded-md border border-line px-3 py-1.5 text-[13px] font-medium text-fg-2 hover:bg-inset hover:text-fg disabled:opacity-50">
            Dismiss shown
          </button>
        ) : undefined} />

      <p className="mb-3 text-[13px] text-fg-2">
        Crystal records where its own tools, ingestion and answers fell short. Nothing here acts on its own; use it to decide what to fix.
      </p>

      <div className="mb-3 flex flex-wrap items-center gap-3">
        <Segmented label="Stream" value={tab} onChange={setTab}
          options={[{ value: "substrate", label: "Tools and pipeline" }, { value: "response", label: "Answer quality" }]} />
        <span className="font-mono text-[11px] text-fg-3">{tab === "substrate" ? "substrate" : "response"}</span>
        {tab === "substrate" && observations.length > 0 && (
          <FilterChips label="Severity" value={severity ?? "all"} onChange={(v) => setSeverity(v === "all" || v === null ? null : (v as Severity))}
            options={[{ value: "all", label: "All", count: countSev(null) },
              ...(Object.keys(SEVERITY) as Severity[]).map((s) => ({ value: s, label: SEVERITY[s].label, count: countSev(s), title: s }))]} />
        )}
      </div>

      <ConfirmInline open={confirmAll} className="mb-3" busy={busy} confirmLabel="Dismiss"
        message={`Dismiss the ${shown.length} observation${shown.length === 1 ? "" : "s"} shown? They stay in the record and can be restored.`}
        onCancel={() => setConfirmAll(false)}
        onConfirm={() => { setConfirmAll(false); void dismiss(shown.map((o) => o.action_item.id)); }} />

      {actionError && <ErrorBanner title="That didn't go through" message={actionError} />}

      {tab === "response" ? (
        <QualityStream groups={(quality.data?.groups ?? []) as QualityGroup[]} loading={quality.isLoading} error={quality.isError ? quality.error : null} onRetry={() => quality.refetch()} />
      ) : flat.isError ? (
        <ErrorBanner title="Couldn't load critiques" error={flat.error} onRetry={() => flat.refetch()} />
      ) : flat.isLoading ? (
        <Skeleton rows={5} />
      ) : observations.length === 0 ? (
        <EmptyState icon={MessageSquareWarning} title="Nothing flagged" help="When Crystal or its structural scans find something in the system worth complaining about, it lands here." />
      ) : shown.length === 0 ? (
        <EmptyState icon={MessageSquareWarning} title={`No ${severity} observations`} help="Pick another severity." />
      ) : (
        <div className="space-y-2">
          {groups.map((g) => {
            const open = openGroup === g.subsystem;
            return (
              <section key={g.subsystem} className="rounded-xl border border-line bg-surface" aria-label={subsystemLabel(g.subsystem)}>
                <button type="button" onClick={() => setOpenGroup(open ? null : g.subsystem)} aria-expanded={open}
                  className="flex w-full items-center justify-between gap-3 px-3 py-2.5 text-left hover:bg-inset">
                  <span className="flex min-w-0 items-center gap-2">
                    {open ? <ChevronDown aria-hidden className="h-4 w-4 shrink-0 text-fg-3" /> : <ChevronRight aria-hidden className="h-4 w-4 shrink-0 text-fg-3" />}
                    <span className="text-[14px] font-semibold text-fg" title={g.subsystem}>{subsystemLabel(g.subsystem)}</span>
                  </span>
                  <span className="shrink-0 text-[12px] text-fg-3">
                    {g.items.length} open · {g.high} high · {g.medium} medium{g.latest ? ` · latest ${fmtRelative(g.latest)}` : ""}
                  </span>
                </button>
                {!open && g.items[0]?.action_item?.content?.complaint && (
                  <p className="truncate px-9 pb-2.5 text-[12.5px] text-fg-2">{g.items[0].action_item.content.complaint}</p>
                )}
                {open && (
                  <ul className="divide-y divide-line border-t border-line">
                    {g.items.map((o) => <ObservationRow key={o.action_item.id} o={o} busy={busy} onDismiss={() => void dismiss([o.action_item.id])} />)}
                  </ul>
                )}
              </section>
            );
          })}
        </div>
      )}
    </div>
  );
}

function ObservationRow({ o, busy, onDismiss }: { o: SubstrateObs; busy: boolean; onDismiss: () => void }) {
  const sev = severityOf(o);
  const critic = o.critique?.critic_role || "critic";
  const model = o.critique?.critic_model;
  const when = o.action_item?.created_at;
  const turn = o.trace_summary?.turn_index;
  const seq = o.trace_summary?.sequence_id;
  const crystal = o.action_item?.content?.crystal_id;
  return (
    <li className="flex flex-wrap items-start gap-x-3 gap-y-1.5 px-3 py-2.5 min-[720px]:pl-9">
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2 text-[12px] text-fg-3">
          <StatusPill variant={SEVERITY[sev].variant} raw={sev}>{SEVERITY[sev].label}</StatusPill>
          <span>{critic}{model ? ` (${model})` : ""}{when ? ` · ${fmtRelative(when)}` : ""}</span>
        </div>
        <p className="mt-1 text-[13.5px] text-fg">{o.action_item?.content?.complaint || "(no complaint text)"}</p>
        {(turn != null || seq) && (
          <p className="mt-0.5 text-[12.5px] text-fg-2">
            During: {turn != null ? `turn ${turn + 1}` : "a turn"}{seq ? ` of conversation ${truncate(seq, 14)}` : ""}
          </p>
        )}
        {crystal && (
          <Link to={`/bank/${encodeURIComponent(crystal)}`} className="mt-1 inline-flex rounded-md border border-line bg-inset px-1.5 py-0.5 font-mono text-[11px] text-fg-2 hover:text-fg">
            Crystal {truncate(crystal, 18)}
          </Link>
        )}
      </div>
      <button type="button" onClick={onDismiss} disabled={busy} title={when ? fmtDateTime(when) : undefined}
        className="rounded-md border border-line px-2.5 py-1 text-[12.5px] font-medium text-fg-2 hover:bg-inset hover:text-fg disabled:opacity-50">
        Dismiss
      </button>
    </li>
  );
}

// The response-quality stream. Read-only by design: observations live inside
// critique rows and carry no dismissal state; this exists so an operator
// can SEE what the shadow and self critics think of the agent's answers.
function QualityStream({ groups, loading, error, onRetry }: { groups: QualityGroup[]; loading: boolean; error: unknown; onRetry: () => void }) {
  const [openType, setOpenType] = useState<string | null>(null);
  if (error) return <ErrorBanner title="Couldn't load answer-quality flags" error={error} onRetry={onRetry} />;
  if (loading) return <Skeleton rows={4} />;
  if (groups.length === 0) {
    return <EmptyState icon={MessageSquareWarning} title="No answer-quality flags" help="When the shadow or self critic flags an assumption, a skipped step or a contradiction, it lands here. Empty over real traffic is the good outcome." />;
  }
  return (
    <div className="space-y-2">
      {groups.map((g) => {
        const open = openType === g.observation_type;
        return (
          <section key={g.observation_type} className="rounded-xl border border-line bg-surface" aria-label={OBS_LABELS[g.observation_type] ?? g.observation_type}>
            <button type="button" onClick={() => setOpenType(open ? null : g.observation_type)} aria-expanded={open}
              className="flex w-full items-center justify-between gap-3 px-3 py-2.5 text-left hover:bg-inset">
              <span className="flex min-w-0 items-center gap-2">
                {open ? <ChevronDown aria-hidden className="h-4 w-4 shrink-0 text-fg-3" /> : <ChevronRight aria-hidden className="h-4 w-4 shrink-0 text-fg-3" />}
                <span className="text-[14px] font-semibold text-fg" title={g.observation_type}>{OBS_LABELS[g.observation_type] ?? g.observation_type}</span>
              </span>
              <span className="shrink-0 rounded-full bg-inset px-2 py-0.5 text-[12px] font-medium text-fg-2">{g.count}</span>
            </button>
            {open && (
              <ul className="divide-y divide-line border-t border-line">
                {(g.latest ?? []).map((o, i) => (
                  <li key={`${o.critique_id}-${i}`} className="px-3 py-2.5 min-[720px]:pl-9">
                    <div className="flex flex-wrap items-center gap-2 text-[12px] text-fg-3">
                      <StatusPill variant={o.critic_role === "shadow" ? "vio" : "info"} raw={o.critic_role}>{o.critic_role === "shadow" ? "Shadow critic" : "Self critic"}</StatusPill>
                      {o.created_at && <span>{fmtRelative(o.created_at)}</span>}
                      {o.sequence_id && <span className="font-mono">conversation {truncate(o.sequence_id, 14)}</span>}
                    </div>
                    <p className="mt-1 text-[13.5px] text-fg">
                      {typeof o.detail?.text === "string" ? o.detail.text : typeof o.detail?.description === "string" ? o.detail.description : JSON.stringify(o.detail ?? {})}
                    </p>
                    {o.summary_text && <p className={cn("mt-0.5 text-[12.5px] text-fg-2")}>Critic summary: {truncate(o.summary_text, 240)}</p>}
                  </li>
                ))}
              </ul>
            )}
          </section>
        );
      })}
    </div>
  );
}
