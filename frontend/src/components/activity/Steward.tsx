// Activity → Steward (Q100 A, 2026-10-23). Anthony: "let a model have the
// hands... the agent keeps the mind going with a simple task of keep
// improving my crystal bank until my cost runs out. Then ask for a
// refill." This view is the Steward's accountability: whether it runs,
// what its last session did and why, every action with a 24-hour Undo,
// and the refill ask when it stopped at the day's capacity.
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bot } from "lucide-react";
import { Link } from "react-router-dom";
import { api, errorMessage } from "@/lib/api";
import type { StewardAction, StewardStatus } from "@/lib/types";
import { EmptyState, ErrorBanner, Skeleton, useToast } from "@/components/ui";
import { cn, fmtDateTime, fmtRelative } from "@/lib/utils";

export const ACTION_WORD: Record<string, string> = {
  decide_conflict: "Settled a conflict", reopen_conflict: "Reopened a conflict", approve_assumption: "Approved", reject_assumption: "Rejected",
  verify_assumption: "Queued verification", research_gap: "Queued research", dismiss_gap: "Dismissed a gap", reopen_gap: "Reopened a gap",
  answer_gap: "Answered a gap", edit_fact: "Edited a fact", retire_fact: "Retired a fact", set_tier: "Moved a tier", reprobe_fact: "Asked for a re-probe",
};

/** Who used the controls, in a word. */
export const actorWord = (actor: string | undefined): string => (!actor || actor === "steward" ? "Steward" : "Chat");

/** Where an action's subject lives in the console. */
export function actionHref(a: StewardAction): string | null {
  if (a.subject_kind === "conflict") return "/conflicts";
  if (a.subject_kind === "assumption") return `/assumptions/${encodeURIComponent(a.subject_id)}`;
  if (a.subject_kind === "gap") return "/cognition";
  if (a.subject_kind === "crystal") return `/bank/${encodeURIComponent(a.subject_id)}`;
  return null;
}

/** The refill ask, in words, for a paused Steward. */
export function pausedText(p: NonNullable<StewardStatus["paused"]>): string {
  if (p.code === "session_cap") return "Today's sessions are used. It resumes tomorrow.";
  if (p.code === "daily_capacity") return `Today's AI capacity is used. Upgrade your plan, raise the cap, or wait: it resumes ${fmtRelative(p.resets_at)} and picks up where it left off.`;
  if (p.code === "monthly_budget") return "This month's AI budget is used. Upgrade your plan or raise the cap; it resumes with the next month.";
  if (p.code === "memory_full") return "The bank is at its fact cap. Upgrade your plan or make room; it resumes on its own.";
  return p.message || "Paused at a plan wall. It resumes on its own.";
}

export function Steward({ customerId }: { customerId: string }) {
  const qc = useQueryClient();
  const toast = useToast();
  const [error, setError] = useState<string | null>(null);
  const status = useQuery({ queryKey: ["steward", customerId], queryFn: () => api.stewardStatus(customerId), refetchInterval: 15_000 });
  const refresh = () => { qc.invalidateQueries({ queryKey: ["steward", customerId] }); qc.invalidateQueries({ queryKey: ["curation-activity", customerId] }); };
  const undo = useMutation({
    mutationFn: (id: string) => api.undoStewardAction(customerId, id),
    onSuccess: () => { setError(null); toast("Undone. The bank is as it was."); refresh(); },
    onError: (e) => { setError(errorMessage(e, "Couldn't undo that.")); refresh(); },
  });
  const s = status.data;
  if (status.isLoading) return <Skeleton rows={4} />;
  if (status.isError || !s) return <ErrorBanner title="Couldn't load the Steward" error={status.error} onRetry={() => status.refetch()} />;
  const actions = s.actions ?? [];
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[13px]">
        <span className={cn("inline-flex items-center gap-1.5 font-medium", s.enabled ? "text-ok" : "text-fg-2")}>
          <Bot aria-hidden className="h-4 w-4" />
          {s.enabled ? "Crystal runs itself" : "Off. A person curates."}
        </span>
        {!s.platform_enabled && <span className="text-fg-3">(switched off platform-wide)</span>}
        <span className="text-fg-3">·</span>
        <span className="text-fg-2">Today: {s.today.sessions} of {s.today.sessions_per_day} session{s.today.sessions_per_day === 1 ? "" : "s"}, {s.today.actions} action{s.today.actions === 1 ? "" : "s"}</span>
        <span className="flex-1" />
        <Link to="/settings/curation" className="text-accent hover:underline">{s.enabled ? "Switch it off" : "Switch it on"} in Settings</Link>
      </div>
      {s.paused && (
        <div role="status" className="rounded-lg border border-warn/40 bg-warn-soft px-3 py-2 text-[13px] text-fg">
          <span className="font-medium text-warn">Paused.</span> {pausedText(s.paused)}
        </div>
      )}
      {error && <ErrorBanner title="That didn't go through" message={error} />}
      {s.latest ? (
        <section aria-label="Last session" className="rounded-xl border border-line bg-surface p-4">
          <p className="text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">
            Last session · {s.latest.at ? fmtRelative(s.latest.at) : ""} · {s.latest.actions} action{s.latest.actions === 1 ? "" : "s"}{s.latest.ok ? "" : " · failed"}
          </p>
          <p className="mt-2 whitespace-pre-wrap break-words text-[13.5px] text-fg">{s.latest.brief || (s.latest.ok ? "The session left no brief." : "The session failed before it could write a brief.")}</p>
          {s.latest.run_id && <Link to={`/activity/run/${encodeURIComponent(s.latest.run_id)}`} className="mt-2 inline-block text-[12.5px] text-accent hover:underline">Open the run</Link>}
        </section>
      ) : (
        <EmptyState icon={Bot} title={s.enabled ? "No session yet" : "The Steward is off"}
          help={s.enabled ? "The first session runs in the next idle cycle. What it does lands here, with a 24-hour undo." : "Switch it on in Settings › Curation and Crystal keeps improving this bank on its own: settling what the facts decide, approving what agrees, researching what it can. Every action lands here with a 24-hour undo."} />
      )}
      {actions.length > 0 && (
        <section aria-label="Actions">
          <p className="mb-2 text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">Every action, by the Steward or from chat, newest first · undo within 24 hours</p>
          <ul className="divide-y divide-line rounded-xl border border-line bg-surface">
            {actions.map((a) => {
              const href = actionHref(a);
              return (
                <li key={a.id} className="flex flex-wrap items-start gap-x-3 gap-y-1 px-3 py-2.5 text-[13px]">
                  <span className="w-24 shrink-0 whitespace-nowrap text-[12px] text-fg-3" title={a.created_at ? fmtDateTime(a.created_at) : undefined}>{fmtRelative(a.created_at)}</span>
                  <div className="min-w-0 flex-1">
                    <p className={cn("break-words text-fg", a.undone_at && "line-through text-fg-3")}>
                      <span className={cn("mr-1.5 rounded px-1.5 py-0.5 text-[11px] font-medium", actorWord(a.actor) === "Steward" ? "bg-accent-soft text-accent" : "bg-inset text-fg-2")}>{actorWord(a.actor)}</span>
                      <span className="font-medium text-fg-2">{ACTION_WORD[a.action] ?? a.action.replace(/_/g, " ")}</span>{" · "}{a.label.replace(/^(Steward|Chat) /, "")}
                    </p>
                    {a.reason && <p className="mt-0.5 text-[12.5px] text-fg-2">{a.reason}</p>}
                  </div>
                  {href && <Link to={href} className="text-[12.5px] text-accent hover:underline">Open</Link>}
                  {a.undone_at ? (
                    <span className="text-[12px] text-fg-3">Undone {fmtRelative(a.undone_at)}</span>
                  ) : a.undoable ? (
                    <button type="button" disabled={undo.isPending} onClick={() => undo.mutate(a.id)}
                      className="rounded-md border border-line px-2.5 py-1 text-[12.5px] font-medium text-fg-2 hover:bg-inset hover:text-fg disabled:opacity-50">
                      Undo
                    </button>
                  ) : (
                    <span className="text-[12px] text-fg-3">{a.action === "reprobe_fact" || a.action === "retire_fact" ? "Nothing to undo" : "Undo window closed"}</span>
                  )}
                </li>
              );
            })}
          </ul>
        </section>
      )}
    </div>
  );
}
