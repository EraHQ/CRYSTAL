// The goal page (Series 382, Q139 A, Q140 B, 2026-10-27). One goal on its
// own page: what it is for, each step with its completion criteria measured
// live, every check-in counted, a line to the next check-in, and going back
// to any step. Anthony on the first draft: "It needs to be abstracted more.
// If a user wants to edit rows they have the lattice page. And not
// everything will follow this format." So criteria read as plain sentences
// (the exact numbers sit behind Advanced) and going back says what it undoes
// in general words, never in one goal's terms.
import { useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Check, Circle, RotateCw, Target, X } from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import type { Goal, GoalComment, GoalProblem, GoalCriterion, GoalCriterionKind, GoalCriterionResult, GoalItem, GoalProgressStep, GoalRun, GoalStep } from "@/lib/types";
import { ConfirmInline, EmptyState, ErrorBanner, Modal, PageHeader, StatusPill, Switch, useToast } from "@/components/ui";
import type { StatusVariant } from "@/components/ui";
import { cn, fmtNum, fmtRelative } from "@/lib/utils";
import { BTN_PRIMARY, BTN_QUIET, BTN_SECONDARY, INPUT } from "./settings/shared";
import { APP_NAMES, Decisions, DraftCard, MISSING_LABEL, cadenceText, goalState } from "./Goals";

const MEASURED: GoalCriterionKind[] = ["rows", "coverage", "distinct", "reference", "sources", "saturation", "groups"];

/** How a criterion is checked, in a word. */
export function checkedBy(kind: GoalCriterionKind): { label: string; variant: StatusVariant } {
  if (kind === "rubric") return { label: "Reviewer", variant: "vio" };
  if (kind === "person") return { label: "You", variant: "warn" };
  return { label: "Counted", variant: "info" };
}

/** Why a check-in stopped, in words. */
export function stopText(reason: string | null | undefined): string {
  if (reason === "max_iterations") return "Stopped: tool-call budget";
  if (reason === "deadline") return "Stopped: time limit";
  if (reason === "error") return "Stopped: an error";
  if (reason === "end_turn" || !reason) return "Finished";
  return `Stopped: ${reason}`;
}

function when(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  const today = new Date();
  const time = d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  return d.toDateString() === today.toDateString() ? `Today ${time}` : `${d.toLocaleDateString(undefined, { month: "short", day: "numeric" })}, ${time}`;
}

function duration(seconds: number | null | undefined): string {
  if (seconds == null) return "";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  return `${Math.floor(seconds / 60)}m ${String(Math.round(seconds % 60)).padStart(2, "0")}s`;
}

function stepState(s: GoalStep, current: boolean): { text: string; variant: StatusVariant } {
  if (s.repeats) return { text: "Repeats every check-in", variant: "neutral" };
  if (s.done) return { text: s.accepted_at ? `Accepted ${when(s.accepted_at)}` : "Done", variant: "ok" };
  if (s.state === "awaiting_review") return { text: "Waits for your approval", variant: "warn" };
  if (s.state === "rejected") return { text: `Rejected${s.attempts ? ` · attempt ${s.attempts}` : ""}`, variant: "bad" };
  if (current) return { text: "Working on it", variant: "info" };
  const n = (s.criteria ?? []).length;
  return { text: `Not started${n ? ` · ${n} ${n === 1 ? "criterion" : "criteria"}` : ""}`, variant: "neutral" };
}

function ResultLine({ r }: { r: GoalCriterionResult }) {
  const how = checkedBy(r.kind);
  const pct = typeof r.value === "number" && typeof r.target === "number" && r.target <= 1 ? ` · ${(r.value * 100).toFixed(r.value >= 0.995 ? 0 : 1)}%` : "";
  return (
    <li className="rounded-lg border border-line px-3 py-2">
      <div className="flex flex-wrap items-center gap-2">
        {r.passed === true ? <Check aria-label="Met" className="h-4 w-4 shrink-0 text-ok" />
          : r.passed === false ? <X aria-label="Short" className="h-4 w-4 shrink-0 text-bad" />
            : <Circle aria-label="Not checked yet" className="h-4 w-4 shrink-0 text-fg-3" />}
        <span className="min-w-0 flex-1 text-[13px] font-medium text-fg">{r.label}</span>
        <StatusPill variant={how.variant}>{how.label}</StatusPill>
        {r.measured && <span className="text-[12.5px] tabular-nums text-fg-2">{r.measured}{pct}{r.passed ? " · met" : ""}</span>}
      </div>
      {r.passed === null && r.kind === "rubric" && <p className="mt-1 text-[12px] text-fg-3">Checked once the counted ones pass.</p>}
      {r.passed === null && r.kind === "person" && <p className="mt-1 text-[12px] text-fg-3">You approve it when the rest pass.</p>}
      {r.gap && <p className={cn("mt-1 text-[12px]", r.invalid ? "text-bad" : "text-fg-2")}>{r.gap}</p>}
      {r.basis === "measured" && r.basis_detail && !r.passed && <p className="mt-0.5 text-[11.5px] text-fg-3">Target from research: {r.basis_detail}</p>}
      {(r.missing_sample ?? []).length > 0 && (
        <p className="mt-1 flex flex-wrap gap-1" aria-label="Not yet">
          {(r.missing_sample ?? []).slice(0, 6).map((x) => (
            <span key={x} className="rounded bg-inset px-1.5 py-0.5 font-mono text-[11px] text-fg-2">{x}</span>
          ))}
        </p>
      )}
    </li>
  );
}

/** "When is step n done?": plain sentences; the exact checks behind Advanced (Q140 B). */
function CriteriaDialog({ customerId, goal, step, results, onClose }: {
  customerId: string; goal: Goal; step: GoalStep; results: GoalCriterionResult[]; onClose: () => void;
}) {
  const qc = useQueryClient();
  const toast = useToast();
  const [rows, setRows] = useState<GoalCriterion[]>(() => (step.criteria ?? []).map((c) => ({ ...c })));
  const [extra, setExtra] = useState("");
  const [error, setError] = useState<string | null>(null);
  const lattices = goal.plan?.lattices ?? [];
  const save = useMutation({
    mutationFn: () => {
      const out = rows.map((c) => (c.kind === "rubric" || c.kind === "person" ? { ...c, text: c.label } : c));
      if (extra.trim()) out.push({ kind: "rubric", label: extra.trim(), text: extra.trim(), basis: "estimated", basis_detail: "" });
      return api.setStepCriteria(customerId, goal.id, step.n, out);
    },
    onSuccess: () => {
      toast("Saved. The next submission is checked against these.");
      for (const k of ["goal", "goal-progress"]) qc.invalidateQueries({ queryKey: [k, customerId, goal.id] });
      onClose();
    },
    onError: (e) => setError(errorMessage(e, "Couldn't save these.")),
  });
  const set = (i: number, patch: Partial<GoalCriterion>) => setRows(rows.map((c, j) => (j === i ? { ...c, ...patch } : c)));
  const num = (v: string) => (v.trim() === "" ? undefined : Number(v));
  return (
    <Modal open onClose={onClose} title={`When is step ${step.n} done?`} className="max-w-[600px]"
      footer={<>
        <button type="button" className={BTN_SECONDARY} onClick={onClose}>Cancel</button>
        <button type="button" className={BTN_PRIMARY} disabled={save.isPending || rows.some((c) => c.label.trim().length < 3)
          || (!!extra.trim() && extra.trim().length < 12)} onClick={() => save.mutate()}>Save</button>
      </>}>
      <div className="max-h-[65vh] space-y-4 overflow-y-auto pr-1">
        <p className="text-[13px] text-fg-2">Each line is checked by counting, by an independent reviewer, or by you. A line you add in your own words goes to the reviewer; open Advanced to set an exact count.</p>
        {error && <ErrorBanner title="That didn't go through" message={error} />}
        {rows.map((c, i) => {
          const how = checkedBy(c.kind);
          const r = results.find((x) => x.label === c.label);
          const cols = lattices.find((l) => l.name === c.lattice)?.columns?.map((x) => x.key) ?? [];
          return (
            <div key={i} className="space-y-1.5">
              <div className="flex items-center gap-2">
                <label htmlFor={`crit-${i}`} className="flex-1 text-[12.5px] font-semibold text-fg-2">{i + 1}</label>
                <button type="button" className={BTN_QUIET} onClick={() => setRows(rows.filter((_c, j) => j !== i))}>Remove</button>
              </div>
              <textarea id={`crit-${i}`} className={INPUT + " min-h-[56px]"} value={c.label} maxLength={200}
                onChange={(e) => set(i, { label: e.target.value })} />
              <div className="flex flex-wrap items-center gap-2 text-[12.5px] text-fg-2">
                <StatusPill variant={how.variant}>{how.label}</StatusPill>
                <span>{r?.measured ? `Now ${r.measured}.` : c.kind === "rubric" ? "Checked once the counted ones pass." : c.kind === "person" ? "You approve it." : ""}
                  {c.basis === "measured" && c.basis_detail ? ` Target from research: ${c.basis_detail}` : ""}</span>
              </div>
              {MEASURED.includes(c.kind) && (
                <details className="rounded-lg border border-line px-3 py-2 text-[12.5px]">
                  <summary className="cursor-pointer font-medium text-fg-2">Advanced</summary>
                  <datalist id={`cols-${i}`}>{cols.map((k) => <option key={k} value={k} />)}</datalist>
                  <div className="grid gap-2 pt-2 min-[520px]:grid-cols-2">
                    <label className="block text-[12px] font-medium text-fg-2">Check
                      <select className={INPUT + " mt-1"} value={c.kind} onChange={(e) => set(i, { kind: e.target.value as GoalCriterionKind })}>
                        <option value="rows">Has at least this many rows</option>
                        <option value="coverage">Rows have these columns filled</option>
                        <option value="distinct">Has this many different values</option>
                        <option value="reference">Each item has enough entries</option>
                        <option value="sources">Came from this many pages</option>
                        <option value="saturation">Recent check-ins found nothing new</option>
                        <option value="groups">Each group has a set number of rows</option>
                      </select>
                    </label>
                    <label className="block text-[12px] font-medium text-fg-2">Lattice
                      <select className={INPUT + " mt-1"} value={c.lattice ?? ""} onChange={(e) => set(i, { lattice: e.target.value })}>
                        {lattices.map((l) => <option key={l.name} value={l.name}>{l.name}</option>)}
                      </select>
                    </label>
                    {(c.kind === "rows" || c.kind === "distinct" || c.kind === "sources" || c.kind === "groups") && (
                      <label className="block text-[12px] font-medium text-fg-2">{c.kind === "groups" ? "At least, per group" : "At least"}
                        <input className={INPUT + " mt-1"} inputMode="numeric" value={c.min ?? ""} onChange={(e) => set(i, { min: num(e.target.value) })} />
                      </label>
                    )}
                    {(c.kind === "rows" || c.kind === "distinct" || c.kind === "groups") && (
                      <label className="block text-[12px] font-medium text-fg-2">{c.kind === "groups" ? "At most, per group (blank: no limit)" : "At most (blank: no limit)"}
                        <input className={INPUT + " mt-1"} inputMode="numeric" value={c.max || ""} onChange={(e) => set(i, { max: num(e.target.value) ?? 0 })} />
                      </label>
                    )}
                    {c.kind === "groups" && (<>
                      <label className="block text-[12px] font-medium text-fg-2">Grouped by (column)
                        <input className={INPUT + " mt-1"} list={`cols-${i}`} value={c.column ?? ""} onChange={(e) => set(i, { column: e.target.value })} />
                      </label>
                      <label className="block text-[12px] font-medium text-fg-2 min-[520px]:col-span-2">Values that must be there (comma separated, optional)
                        <input className={INPUT + " mt-1"} value={(c.values ?? []).join(", ")}
                          onChange={(e) => set(i, { values: e.target.value.split(",").map((x) => x.trim()).filter(Boolean) })} />
                      </label>
                      <label className="flex items-center gap-2 text-[12px] font-medium text-fg-2 min-[520px]:col-span-2">
                        <input type="checkbox" checked={!!c.exclusive} disabled={!(c.values ?? []).length} onChange={(e) => set(i, { exclusive: e.target.checked })} />
                        Only these values belong (any other fails)
                      </label>
                    </>)}
                    {(c.kind === "distinct" || c.kind === "sources" || c.kind === "reference") && (
                      <label className="block text-[12px] font-medium text-fg-2">{c.kind === "reference" ? "Entries' column" : "Column"}
                        <input className={INPUT + " mt-1"} list={`cols-${i}`} value={c.column ?? ""} onChange={(e) => set(i, { column: e.target.value })} />
                      </label>
                    )}
                    {c.kind === "coverage" && (
                      <label className="block text-[12px] font-medium text-fg-2 min-[520px]:col-span-2">Columns that must be filled (comma separated)
                        <input className={INPUT + " mt-1"} value={(c.columns ?? []).join(", ")}
                          onChange={(e) => set(i, { columns: e.target.value.split(",").map((x) => x.trim()).filter(Boolean) })} />
                      </label>
                    )}
                    {c.kind === "reference" && (<>
                      <label className="block text-[12px] font-medium text-fg-2">Items (lattice)
                        <select className={INPUT + " mt-1"} value={c.of_lattice ?? ""} onChange={(e) => set(i, { of_lattice: e.target.value })}>
                          {lattices.map((l) => <option key={l.name} value={l.name}>{l.name}</option>)}
                        </select>
                      </label>
                      <label className="block text-[12px] font-medium text-fg-2">At least, per item
                        <input className={INPUT + " mt-1"} inputMode="numeric" value={c.min_per ?? ""} onChange={(e) => set(i, { min_per: num(e.target.value) })} />
                      </label>
                    </>)}
                    {(c.kind === "coverage" || c.kind === "reference") && (
                      <label className="block text-[12px] font-medium text-fg-2">Share that must pass (%)
                        <input className={INPUT + " mt-1"} inputMode="decimal" value={c.min_fraction != null ? Math.round(c.min_fraction * 1000) / 10 : ""}
                          onChange={(e) => { const v = num(e.target.value); set(i, { min_fraction: v == null ? undefined : v / 100 }); }} />
                      </label>
                    )}
                    {c.kind === "saturation" && (
                      <label className="block text-[12px] font-medium text-fg-2">Check-ins in a row
                        <input className={INPUT + " mt-1"} inputMode="numeric" value={c.runs ?? ""} onChange={(e) => set(i, { runs: num(e.target.value) })} />
                      </label>
                    )}
                    <p className="text-[12px] text-fg-3 min-[520px]:col-span-2">Target came from: {c.basis === "measured" ? "research, measured" : "an estimate"}{c.basis_detail ? `. ${c.basis_detail}` : ""}</p>
                  </div>
                </details>
              )}
            </div>
          );
        })}
        <label className="block text-[12.5px] font-semibold text-fg-2">Add another (checked by the reviewer)
          <textarea className={INPUT + " mt-1 min-h-[56px]"} value={extra} maxLength={200}
            placeholder="For example: I've looked at a sample and it reads right." onChange={(e) => setExtra(e.target.value)} />
        </label>
      </div>
    </Modal>
  );
}

/** "Go back to step n?": what it undoes, counted first, in general words. */
function GoBackDialog({ customerId, goal, n, onClose }: { customerId: string; goal: Goal; n: number; onClose: () => void }) {
  const qc = useQueryClient();
  const toast = useToast();
  const [reason, setReason] = useState("");
  const [each, setEach] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const peek = useQuery({ queryKey: ["goal-rollback", customerId, goal.id, n], queryFn: () => api.previewGoalRollback(customerId, goal.id, n), retry: false });
  const go = useMutation({
    mutationFn: () => api.rollbackGoal(customerId, goal.id, n, reason.trim()),
    onSuccess: () => {
      toast(`Back to step ${n}. The next check-in picks up from there.`);
      for (const k of ["goal", "goal-progress", "goal-runs", "goal-comments"]) qc.invalidateQueries({ queryKey: [k, customerId, goal.id] });
      qc.invalidateQueries({ queryKey: ["goals", customerId] });
      onClose();
    },
    onError: (e) => setError(errorMessage(e, "Couldn't go back.")),
  });
  const p = peek.data;
  const sent = (p?.sent_items.length ?? 0) + (p?.not_undone.filter((x) => x.why === "sent").length ?? 0);
  return (
    <Modal open onClose={onClose} title={`Go back to step ${n}?`}
      footer={<>
        <button type="button" className={BTN_SECONDARY} onClick={onClose}>Cancel</button>
        <button type="button" className="inline-flex items-center rounded-md border border-warn/50 bg-warn-soft px-3 py-1.5 text-[13px] font-semibold text-warn hover:border-warn disabled:opacity-50"
          disabled={go.isPending || !p} onClick={() => go.mutate()}>Go back</button>
      </>}>
      <div className="space-y-3">
        <p className="text-[13.5px] text-fg">
          {n === 1 ? "The goal starts over from when its plan was written." : `The goal picks up from where it stood when step ${n - 1} was done.`} Everything it did after that is undone.
        </p>
        {peek.isError && <ErrorBanner title="Can't go back to this step" message={errorMessage(peek.error, "")} />}
        {error && <ErrorBanner title="That didn't go through" message={error} />}
        {p && (
          <ul className="space-y-1.5 text-[13.5px]" aria-label="What going back does">
            <li className="flex justify-between gap-3"><span>Changes to your memory</span><span>{fmtNum(p.undo)} undone</span></li>
            <li className="flex justify-between gap-3"><span>Drafts and to-dos waiting for you</span><span>{fmtNum(p.dismiss)} dismissed</span></li>
            {sent > 0 && <li className="flex justify-between gap-3 text-warn"><span>Sent or posted</span><span>{fmtNum(sent)} {sent === 1 ? "stays" : "stay"} sent</span></li>}
          </ul>
        )}
        {p && (p.undo + p.dismiss + sent > 0) && (
          <button type="button" className="text-[12.5px] text-accent hover:underline" aria-expanded={each} onClick={() => setEach((v) => !v)}>
            {each ? "Hide the list" : "See each one"}
          </button>
        )}
        {p && each && (
          <ul className="max-h-48 space-y-0.5 overflow-y-auto text-[12px] text-fg-2">
            {p.undo_labels.map((l, i) => <li key={`u${i}`}>Undone: {l}</li>)}
            {p.dismiss_labels.map((l, i) => <li key={`d${i}`}>Dismissed: {l}</li>)}
            {p.sent_items.map((l, i) => <li key={`s${i}`}>Stays sent: {l}</li>)}
            {p.not_undone.map((x, i) => <li key={`n${i}`}>Stays: {x.label}</li>)}
          </ul>
        )}
        <label className="block text-[12.5px] font-medium text-fg-2">What should it do differently? (optional)
          <textarea className={INPUT + " mt-1 min-h-[56px]"} value={reason} maxLength={500} placeholder="The next check-in reads this first."
            onChange={(e) => setReason(e.target.value)} />
        </label>
      </div>
    </Modal>
  );
}

function StepCard({ customerId, goal, step, progress, current, canGoBack, busy }: {
  customerId: string; goal: Goal; step: GoalStep; progress?: GoalProgressStep; current: boolean; canGoBack: boolean; busy: boolean;
}) {
  const qc = useQueryClient();
  const toast = useToast();
  const [editing, setEditing] = useState(false);
  const [goingBack, setGoingBack] = useState(false);
  const [note, setNote] = useState("");
  const [declining, setDeclining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const st = stepState(step, current);
  const results = progress?.results ?? [];
  const review = useMutation({
    mutationFn: (approve: boolean) => api.reviewGoalStep(customerId, goal.id, step.n, approve, approve ? undefined : note.trim()),
    onSuccess: (_o, approve) => {
      toast(approve ? `Step ${step.n} accepted.` : "Sent back. The next check-in reads your note.");
      setDeclining(false);
      for (const k of ["goal", "goal-progress"]) qc.invalidateQueries({ queryKey: [k, customerId, goal.id] });
    },
    onError: (e) => setError(errorMessage(e, "That didn't go through.")),
  });
  return (
    <li className={cn("rounded-xl border bg-surface p-4", current ? "border-accent/50" : "border-line")} aria-label={`Step ${step.n}`}>
      <div className="flex flex-wrap items-center gap-2">
        <span aria-hidden className={cn("flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[12px] font-semibold",
          step.done ? "bg-ok-soft text-ok" : current ? "bg-accent text-on-accent" : "bg-inset text-fg-2")}>
          {step.repeats ? <RotateCw className="h-3.5 w-3.5" /> : step.done ? <Check className="h-3.5 w-3.5" /> : step.n}
        </span>
        <span className={cn("min-w-0 flex-1 text-[14px] font-semibold", step.done ? "text-fg-2" : "text-fg")}>{step.title}</span>
        <StatusPill variant={st.variant}>{st.text}</StatusPill>
      </div>
      {error && <div className="mt-2"><ErrorBanner title="That didn't go through" message={error} /></div>}
      {(current || step.state === "rejected") && step.detail && <p className="mt-2 text-[13px] text-fg-2">{step.detail}</p>}
      {step.note && !step.done && <p className="mt-1 text-[12.5px] text-fg-3">Last submitted with: {step.note}</p>}
      {!step.done && !step.repeats && results.length > 0 && (
        <ul className="mt-3 space-y-2" aria-label="When it's done">{results.map((r, i) => <ResultLine key={i} r={r} />)}</ul>
      )}
      {step.done && (step.criteria ?? []).length > 0 && (
        <p className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[12.5px] text-fg-2">
          {(step.criteria ?? []).map((c, i) => <span key={i}><Check aria-hidden className="mr-1 inline h-3.5 w-3.5 text-ok" />{c.label}</span>)}
        </p>
      )}
      {step.state === "awaiting_review" && (
        <div className="mt-3 space-y-2 rounded-lg border border-warn/40 bg-warn-soft px-3 py-2.5">
          <p className="text-[13px] text-fg">It passed every other check. Is this step done?</p>
          {declining && (
            <textarea aria-label="What it still needs" className={INPUT + " min-h-[56px]"} value={note} maxLength={4000}
              placeholder="What it still needs. The next check-in reads this." onChange={(e) => setNote(e.target.value)} />
          )}
          <div className="flex flex-wrap gap-2">
            {!declining && <button type="button" className={BTN_PRIMARY} disabled={review.isPending} onClick={() => review.mutate(true)}>Yes, it's done</button>}
            {declining
              ? <button type="button" className={BTN_SECONDARY} disabled={review.isPending || note.trim().length < 2} onClick={() => review.mutate(false)}>Send back</button>
              : <button type="button" className={BTN_SECONDARY} onClick={() => setDeclining(true)}>Not yet</button>}
          </div>
        </div>
      )}
      {!step.repeats && (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          {!step.done && <button type="button" className={BTN_QUIET} disabled={busy} onClick={() => setEditing(true)}>Edit when it's done</button>}
          {canGoBack && <button type="button" className={BTN_QUIET} disabled={busy} onClick={() => setGoingBack(true)}>
            {step.done ? "Go back to here" : `Go back to step ${step.n}`}</button>}
        </div>
      )}
      {editing && <CriteriaDialog customerId={customerId} goal={goal} step={step} results={results} onClose={() => setEditing(false)} />}
      {goingBack && <GoBackDialog customerId={customerId} goal={goal} n={step.n} onClose={() => setGoingBack(false)} />}
    </li>
  );
}

function RunEntry({ run, number }: { run: GoalRun; number: number | null }) {
  const [open, setOpen] = useState(false);
  if (run.phase === "plan") {
    return (
      <li className="rounded-lg border border-line px-3 py-2">
        <div className="flex flex-wrap items-center gap-2 text-[13px]">
          <span className="font-semibold text-fg">Plan</span>
          <span className="text-fg-3">{when(run.started_at)}{run.seconds != null ? ` · ${duration(run.seconds)}` : ""}</span>
          <span className="flex-1" />
          <StatusPill variant={run.status === "complete" ? "ok" : run.status === "failed" ? "bad" : "neutral"}>
            {run.status === "complete" ? "Written" : run.status === "failed" ? "Failed" : "Writing"}
          </StatusPill>
        </div>
        <p className="mt-1 text-[12.5px] text-fg-2">
          {[run.survey ? `${run.survey} survey findings` : "", run.steps != null ? `${run.steps} steps` : "", run.lattices ? `${run.lattices} lattices` : ""].filter(Boolean).join(" · ")}
        </p>
      </li>
    );
  }
  const tools = Object.entries(run.tools ?? {}).sort((a, b) => b[1] - a[1]);
  const shown = tools.slice(0, 5);
  const otherCount = tools.slice(5).reduce((t, [, c]) => t + c, 0);
  return (
    <li className="rounded-lg border border-line px-3 py-2">
      <button type="button" className="flex w-full flex-wrap items-center gap-2 text-left text-[13px]" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
        <span className="font-semibold text-fg">#{number}</span>
        <span className="text-fg-3">{when(run.started_at)}{run.seconds != null ? ` · ${duration(run.seconds)}` : ""}</span>
        <span className="flex-1" />
        {(run.submissions ?? []).map((s, i) => (
          <StatusPill key={i} variant={s.verdict === "accepted" ? "ok" : s.verdict === "awaiting_review" ? "warn" : "bad"}>
            Step {s.n} {s.verdict === "accepted" ? "accepted" : s.verdict === "awaiting_review" ? "waits for you" : "rejected"}
          </StatusPill>
        ))}
        {run.status === "failed" && <StatusPill variant="bad">Failed</StatusPill>}
        <span className="text-[12px] tabular-nums text-fg-2">{run.calls ?? 0} calls · {fmtNum(run.rows?.inserted ?? 0)} rows new</span>
      </button>
      {open && (
        <div className="mt-2 space-y-2 border-t border-line pt-2">
          <dl className="grid grid-cols-2 gap-2 text-[12.5px] min-[640px]:grid-cols-4">
            <div><dt className="text-fg-3">Tool calls</dt><dd className="tabular-nums text-fg">{run.calls ?? 0}{run.budget ? ` / ${run.budget.tool_calls}` : ""}</dd></div>
            <div><dt className="text-fg-3">Actions</dt><dd className="tabular-nums text-fg">{run.actions ?? 0}{run.budget ? ` / ${run.budget.actions}` : ""}</dd></div>
            <div><dt className="text-fg-3">Rows new</dt><dd className="tabular-nums text-fg">{fmtNum(run.rows?.inserted ?? 0)}</dd></div>
            <div><dt className="text-fg-3">Rows changed</dt><dd className="tabular-nums text-fg">{fmtNum(run.rows?.updated ?? 0)}</dd></div>
          </dl>
          {tools.length > 0 && (
            <p className="flex flex-wrap gap-1.5" aria-label="Tools used">
              {shown.map(([t, c]) => <span key={t} className="rounded bg-inset px-1.5 py-0.5 font-mono text-[11px] text-fg-2">{t} ×{c}</span>)}
              {otherCount > 0 && <span className="rounded bg-inset px-1.5 py-0.5 text-[11px] text-fg-2">other ×{otherCount}</span>}
            </p>
          )}
          {(run.comments ?? []).length > 0 && (
            <p className="text-[12.5px] text-fg-2">Read first: {(run.comments ?? []).map((c) => `“${c.body}”`).join(" ")}</p>
          )}
          {run.summary && <p className="whitespace-pre-wrap text-[13px] text-fg">{run.summary}</p>}
          {run.error && <p className="text-[12.5px] text-bad">{run.error}</p>}
          <p className="text-[12px] text-fg-3">{stopText(run.stop_reason)}{run.model ? ` · ${run.model}` : ""}</p>
        </div>
      )}
    </li>
  );
}

/** A note's state, in words (Series 385). */
export function noteState(c: GoalComment): { label: string; variant: StatusVariant } {
  if (c.always) return { label: "Every check-in", variant: "vio" };
  if (c.status === "done") return { label: "Done", variant: "ok" };
  if (c.status === "dismissed") return { label: "Dismissed", variant: "neutral" };
  if (c.status === "cant") return { label: "Can't yet", variant: "bad" };
  if (c.status === "in_progress") return { label: "In progress", variant: "info" };
  return { label: "Open", variant: "warn" };
}

/** Series 385 (Q153 A): a note stays in every check-in until it is handled. */
function NoteEntry({ customerId, goal, c, numbers }: { customerId: string; goal: Goal; c: GoalComment; numbers: Map<string, number> }) {
  const qc = useQueryClient();
  const set = useMutation({
    mutationFn: (change: { status?: "open" | "done" | "dismissed"; always?: boolean }) => api.setGoalNote(customerId, goal.id, c.id, change),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["goal-comments", customerId, goal.id] }),
  });
  const st = noteState(c);
  const closed = !c.always && (c.status === "done" || c.status === "dismissed");
  const by = c.reported_by_run_id ? numbers.get(c.reported_by_run_id) : undefined;
  return (
    <li className="rounded-lg border border-dashed border-line px-3 py-2" aria-label="Your note">
      <div className="flex flex-wrap items-center gap-2">
        <p className="min-w-0 flex-1 text-[12px] text-fg-3">You · {when(c.created_at)}</p>
        <StatusPill variant={st.variant}>{st.label}</StatusPill>
      </div>
      <p className="mt-0.5 whitespace-pre-wrap text-[13px] text-fg">{c.body}</p>
      {c.report && <p className="mt-1 text-[12.5px] text-fg-2">{by ? `#${by}: ` : ""}{c.report}</p>}
      <div className="mt-1 flex flex-wrap gap-2">
        {closed
          ? <button type="button" className={BTN_QUIET} disabled={set.isPending} onClick={() => set.mutate({ status: "open" })}>Reopen</button>
          : !c.always && <button type="button" className={BTN_QUIET} disabled={set.isPending} onClick={() => set.mutate({ status: "done" })}>Mark handled</button>}
        <button type="button" className={BTN_QUIET} disabled={set.isPending} onClick={() => set.mutate({ always: !c.always })}>
          {c.always ? "Stop applying every check-in" : "Apply every check-in"}</button>
        {!closed && !c.always && <button type="button" className={BTN_QUIET} disabled={set.isPending} onClick={() => set.mutate({ status: "dismissed" })}>Dismiss</button>}
      </div>
    </li>
  );
}

function CheckIns({ customerId, goal }: { customerId: string; goal: Goal }) {
  const qc = useQueryClient();
  const toast = useToast();
  const [text, setText] = useState("");
  const [always, setAlways] = useState(false);
  const [all, setAll] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const runs = useQuery({ queryKey: ["goal-runs", customerId, goal.id], queryFn: () => api.goalRuns(customerId, goal.id), refetchInterval: 30_000 });
  const comments = useQuery({ queryKey: ["goal-comments", customerId, goal.id], queryFn: () => api.goalComments(customerId, goal.id), refetchInterval: 30_000 });
  const send = useMutation({
    mutationFn: () => api.addGoalComment(customerId, goal.id, text.trim(), always),
    onSuccess: () => {
      setText(""); setAlways(false); setError(null);
      toast(always ? "Every check-in follows it from now on." : "Every check-in works on it until it's handled.");
      qc.invalidateQueries({ queryKey: ["goal-comments", customerId, goal.id] });
    },
    onError: (e) => setError(errorMessage(e, "Couldn't send that.")),
  });
  const list = runs.data?.runs ?? [];
  // Check-ins are numbered oldest first; plans aren't counted.
  const numbers = useMemo(() => {
    const out = new Map<string, number>();
    let k = 0;
    for (const r of [...list].reverse()) if (r.phase !== "plan") out.set(r.run_id, ++k);
    return out;
  }, [list]);
  const nextNumber = numbers.size + 1;
  type Entry = { at: string; run?: GoalRun; comment?: GoalComment };
  const notes = comments.data?.comments ?? [];
  const live = notes.filter((c) => c.always || !["done", "dismissed"].includes(c.status ?? "open"));
  const entries: Entry[] = [
    ...list.map((r) => ({ at: r.started_at ?? "", run: r })),
    ...notes.filter((c) => !live.includes(c)).map((c) => ({ at: c.created_at ?? "", comment: c })),
  ].sort((a, b) => (a.at < b.at ? 1 : -1));
  const visible = all ? entries : entries.slice(0, 5);
  return (
    <section aria-labelledby="checkins-h" className="space-y-3">
      <h2 id="checkins-h" className="text-[15px] font-semibold text-fg">Check-ins</h2>
      <form className="space-y-2 rounded-xl border border-line bg-surface p-3" onSubmit={(e) => { e.preventDefault(); send.mutate(); }}>
        <label className="block text-[12.5px] font-semibold text-fg-2">Tell the next check-in
          <textarea className={INPUT + " mt-1 min-h-[64px]"} value={text} maxLength={4000} onChange={(e) => setText(e.target.value)}
            placeholder="Skip the preseason. Start with November." />
        </label>
        <label className="flex items-center gap-2 text-[12.5px] text-fg-2">
          <input type="checkbox" checked={always} onChange={(e) => setAlways(e.target.checked)} />
          Apply every check-in (a standing note that never closes)
        </label>
        {error && <ErrorBanner title="That didn't go through" message={error} />}
        <div className="flex flex-wrap items-center gap-2">
          <span className="flex-1 text-[12px] text-fg-3">
            {always ? "Every check-in follows it, ahead of the plan." : `Check-in #${nextNumber} works on it first, and every check-in after until it's handled.`}
          </span>
          <button type="submit" className={BTN_PRIMARY} disabled={send.isPending || text.trim().length < 2}>Send to next check-in</button>
        </div>
      </form>
      {live.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-[13px] font-semibold text-fg">Your notes in force ({live.length})</h3>
          <ul className="space-y-2" aria-label="Your notes in force">
            {live.map((c) => <NoteEntry key={c.id} customerId={customerId} goal={goal} c={c} numbers={numbers} />)}
          </ul>
        </div>
      )}
      {runs.isError && <ErrorBanner title="Couldn't load the check-ins" error={runs.error} onRetry={() => runs.refetch()} />}
      {!runs.isLoading && entries.length === 0 && <p className="text-[13px] text-fg-3">No check-ins yet.</p>}
      <ul className="space-y-2" aria-label="Check-in history">
        {visible.map((e) => e.run
          ? <RunEntry key={e.run.run_id} run={e.run} number={numbers.get(e.run.run_id) ?? null} />
          : <NoteEntry key={e.comment!.id} customerId={customerId} goal={goal} c={e.comment!} numbers={numbers} />)}
      </ul>
      {entries.length > 5 && (
        <button type="button" className={BTN_QUIET} aria-expanded={all} onClick={() => setAll((v) => !v)}>
          {all ? "Show fewer" : `Show ${entries.length - 5} earlier`}
        </button>
      )}
    </section>
  );
}

function NeedsYou({ customerId, goal, items, busy, act, onError, rerun, release }: {
  customerId: string; goal: Goal; items: GoalItem[]; busy: boolean; onError: (m: string | null) => void;
  act: (id: string, body: Parameters<typeof api.goalItemAction>[2]) => void; rerun: () => void; release: () => void;
}) {
  const open = items.filter((i) => i.status === "open");
  const notes = open.filter((i) => i.kind === "note");
  const drafts = open.filter((i) => i.kind === "draft");
  const actions = open.filter((i) => i.kind === "action");
  const missing = goal.plan?.missing ?? [];
  const held = goal.status === "needs_setup";
  const deciding = goal.status === "deciding" && (goal.plan?.decisions ?? []).some((d) => !d.answer);
  if (!deciding && !missing.length && !notes.length && !drafts.length && !actions.length) return null;
  return (
    <section aria-label="Waiting on you" className="space-y-3">
      {deciding && <Decisions key={goal.updated_at ?? goal.id} customerId={customerId} goal={goal} onError={onError} />}
      {missing.length > 0 && (
        <div className={cn("rounded-lg border px-3 py-2.5", held ? "border-warn/40 bg-warn-soft" : "border-line")} role="note" aria-label="What this goal needs">
          <p className="text-[12.5px] font-medium text-fg">
            {held ? "On hold until this is set up. It won't check in or spend anything meanwhile." : "This goal works around these for now"}
          </p>
          <ul className="mt-1.5 space-y-1.5">
            {missing.map((m, i) => (
              <li key={i} className="text-[13px] text-fg">
                <span className="flex flex-wrap items-center gap-2">
                  <StatusPill variant={m.blocking ? "warn" : "neutral"}>{MISSING_LABEL[m.kind] ?? m.kind}</StatusPill>
                  <span className="min-w-0 flex-1 font-medium">{m.need}</span>
                  {m.kind === "app" && <Link to="/settings/connections" className="text-[12.5px] text-accent hover:underline">Connect</Link>}
                </span>
                {m.why && <span className="block text-[12px] text-fg-2">{m.why}</span>}
                {m.suggestion && <span className="block text-[12px] text-fg-3">{m.suggestion}</span>}
              </li>
            ))}
          </ul>
          {held && (
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <button type="button" className={BTN_SECONDARY} disabled={busy} onClick={rerun}>Check again</button>
              <button type="button" className={BTN_QUIET} disabled={busy} onClick={release}>Run anyway</button>
            </div>
          )}
        </div>
      )}
      {notes.length > 0 && <Questions goal={goal} notes={notes} busy={busy} act={act} />}
      {(drafts.length > 0 || actions.length > 0) && (
        <div>
          <h2 className="text-[15px] font-semibold text-fg">Waiting for you</h2>
          <ul className="mt-2 space-y-2" aria-label="Drafts">
            {drafts.map((d) => <DraftCard key={d.id} item={d} busy={busy} act={(body) => act(d.id, body)} />)}
          </ul>
          <ul className="mt-2 space-y-1.5" aria-label="Action items">
            {actions.map((a) => (
              <li key={a.id} className="flex flex-wrap items-start gap-2 rounded-lg border border-line px-3 py-2">
                <div className="min-w-0 flex-1">
                  <p className="text-[13px] text-fg">{a.title}</p>
                  {a.body && <p className="text-[12px] text-fg-2">{a.body}</p>}
                  {a.due_at && <p className={cn("text-[12px]", new Date(a.due_at).getTime() < Date.now() ? "text-bad" : "text-fg-3")}>Due {fmtRelative(a.due_at)}</p>}
                </div>
                <button type="button" className={BTN_SECONDARY} disabled={busy} onClick={() => act(a.id, { action: "done" })}>Done</button>
                <button type="button" className={BTN_QUIET} disabled={busy} onClick={() => act(a.id, { action: "dismiss" })}>Dismiss</button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

const PROBLEM_WORDS: Record<string, string> = {
  blocked_source: "Blocked source", app: "App to connect", permission: "Permission", choice: "Your choice", other: "Needs you",
};

export function problemOf(n: GoalItem): GoalProblem | null {
  return ((n.target as unknown as { problem?: GoalProblem }).problem) ?? null;
}

/** Series 386 (Q154 C, Q155): each thing the goal needs is one question with
 * answers to pick; repeats of it are counted on it, never listed again.
 * Reports from before questions existed show as one card. */
function Questions({ goal, notes, busy, act }: {
  goal: Goal; notes: GoalItem[]; busy: boolean; act: (id: string, body: Parameters<typeof api.goalItemAction>[2]) => void;
}) {
  const asked = notes.filter((n) => problemOf(n));
  const legacy = notes.filter((n) => !problemOf(n)).sort((a, b) => ((a.created_at ?? "") < (b.created_at ?? "") ? 1 : -1));
  const cards = asked.length + (legacy.length ? 1 : 0);
  return (
    <section aria-label="Needs you" className="space-y-3">
      <div className="flex flex-wrap items-baseline gap-2">
        <h2 className="flex-1 text-[15px] font-semibold text-fg">Waiting on you</h2>
        <span className="text-[12.5px] text-fg-3">{cards} question{cards === 1 ? "" : "s"} · 3 open pauses check-ins</span>
      </div>
      {goal.plan?.waiting_on_you && (
        <p role="status" className="rounded-lg border border-warn/40 bg-warn-soft px-3 py-2 text-[13px] text-fg">
          Check-ins are paused until you answer or handle every question below.
        </p>
      )}
      <ul className="space-y-3" aria-label="Questions">
        {asked.map((n) => <QuestionCard key={n.id} n={n} p={problemOf(n)!} busy={busy} act={act} />)}
        {legacy.length > 0 && (
          <QuestionCard key="legacy" n={legacy[0]} busy={busy} act={act} also={legacy.slice(1)}
            p={{ kind: "other", about: "", step: 0, question: legacy[0].title, options: [], recommended: -1, count: legacy.length,
                 first_seen: legacy[legacy.length - 1].created_at ?? "", last_seen: legacy[0].created_at ?? "" }} />
        )}
      </ul>
    </section>
  );
}

function QuestionCard({ n, p, busy, act, also = [] }: {
  n: GoalItem; p: GoalProblem; busy: boolean; act: (id: string, body: Parameters<typeof api.goalItemAction>[2]) => void; also?: GoalItem[];
}) {
  const [pick, setPick] = useState<number | "other">(p.options.length ? (p.recommended >= 0 ? p.recommended : 0) : "other");
  const [own, setOwn] = useState("");
  const [showAll, setShowAll] = useState(false);
  const answer = pick === "other" ? own.trim() : p.options[pick];
  const closeAlso = (action: "done" | "dismiss") => { for (const x of also) act(x.id, { action }); };
  return (
    <li className="space-y-3 rounded-xl border border-warn/40 bg-warn-soft p-4" aria-label={p.question}>
      <div className="flex flex-wrap items-center gap-2">
        <StatusPill variant="warn">{PROBLEM_WORDS[p.kind] ?? "Needs you"}</StatusPill>
        {p.step > 0 && <StatusPill variant="info">Step {p.step}</StatusPill>}
        <span className="flex-1" />
        <span className="text-[12px] text-fg-3">
          {p.count > 1 ? `Reported ${p.count} times · ` : ""}waiting since {when(p.first_seen)}
        </span>
      </div>
      <h3 className="text-[16px] font-semibold leading-snug text-fg">{p.question}</h3>
      {n.title && n.title !== p.question && <p className="text-[13px] text-fg-2">{n.title}</p>}
      <form className="space-y-2" aria-label={`Answer: ${p.question}`}
        onSubmit={(e) => { e.preventDefault(); act(n.id, { action: "answer", body: answer }); closeAlso("done"); }}>
        {p.options.length > 0 && (
          <div role="radiogroup" aria-label="Pick one" className="space-y-2">
            {p.options.map((o, i) => (
              <label key={i} className={cn("flex cursor-pointer items-start gap-3 rounded-lg border px-3 py-2.5 text-[13.5px]",
                pick === i ? "border-accent bg-surface" : "border-line")}>
                <input type="radio" name={`pick-${n.id}`} checked={pick === i} onChange={() => setPick(i)} className="mt-1" />
                <span className="flex-1 text-fg">{o}</span>
                {i === p.recommended && <StatusPill variant="info">Recommended</StatusPill>}
              </label>
            ))}
            <label className={cn("flex cursor-pointer items-start gap-3 rounded-lg border px-3 py-2.5 text-[13.5px]",
              pick === "other" ? "border-accent bg-surface" : "border-line")}>
              <input type="radio" name={`pick-${n.id}`} checked={pick === "other"} onChange={() => setPick("other")} className="mt-1" />
              <span className="flex-1 text-fg">Something else</span>
            </label>
          </div>
        )}
        {pick === "other" && (
          <textarea aria-label="Your answer" className={INPUT + " min-h-[56px]"} value={own} maxLength={4000}
            placeholder="Say it in your own words." onChange={(e) => setOwn(e.target.value)} />
        )}
        <div className="flex flex-wrap items-center gap-2">
          <button type="submit" className={BTN_PRIMARY} disabled={busy || !answer || answer.length < 2}>Answer</button>
          <button type="button" className={BTN_SECONDARY} disabled={busy} onClick={() => { act(n.id, { action: "done" }); closeAlso("done"); }}>
            Already handled</button>
          <span className="flex-1" />
          {also.length > 0 && (
            <button type="button" className={BTN_QUIET} aria-expanded={showAll} onClick={() => setShowAll((v) => !v)}>
              {showAll ? "Hide" : `${also.length} earlier report${also.length === 1 ? "" : "s"} of this`}</button>
          )}
        </div>
      </form>
      {showAll && (
        <ul className="space-y-1 text-[12.5px] text-fg-2" aria-label="Earlier reports">
          {also.map((x) => <li key={x.id}><span className="text-fg-3">{when(x.created_at)}</span> · {x.title}</li>)}
        </ul>
      )}
    </li>
  );
}

function History({ items, busy, act }: { items: GoalItem[]; busy: boolean; act: (id: string, body: Parameters<typeof api.goalItemAction>[2]) => void }) {
  const [show, setShow] = useState(false);
  const history = items.filter((i) => i.status !== "open");
  if (!history.length) return null;
  return (
    <section aria-label="Sent and handled">
      <button type="button" className={BTN_QUIET} aria-expanded={show} onClick={() => setShow((v) => !v)}>
        {show ? "Hide history" : `History (${history.length})`}
      </button>
      {show && (
        <ul className="mt-2 space-y-1.5" aria-label="History">
          {history.map((h) => (
            <li key={h.id} className="rounded-lg border border-line px-3 py-2">
              <div className="flex flex-wrap items-center gap-2">
                <StatusPill variant={h.status === "sent" ? "ok" : h.status === "failed" ? "bad" : "neutral"}>
                  {h.status === "sent" ? "Sent" : h.status === "failed" ? "Failed" : h.status === "done" ? "Done" : "Dismissed"}
                </StatusPill>
                <span className="min-w-0 flex-1 text-[13px] text-fg">{h.title}</span>
                <span className="text-[11.5px] text-fg-3">{h.updated_at ? fmtRelative(h.updated_at) : ""}</span>
                {(h.status === "done" || h.status === "dismissed") && (
                  <button type="button" className={BTN_QUIET} disabled={busy} onClick={() => act(h.id, { action: "reopen" })}>Reopen</button>
                )}
              </div>
              {h.status === "sent" && <p className="mt-1 whitespace-pre-wrap text-[12.5px] text-fg-2">{h.body}</p>}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function Survey({ goal }: { goal: Goal }) {
  const [all, setAll] = useState(false);
  const findings = goal.plan?.survey?.findings ?? [];
  if (!findings.length) return null;
  const shown = all ? findings : findings.slice(0, 4);
  return (
    <section aria-labelledby="survey-h" className="rounded-xl border border-line bg-surface p-4">
      <h2 id="survey-h" className="text-[14px] font-semibold text-fg">What the plan was sized on</h2>
      <ul className="mt-2 space-y-2.5">
        {shown.map((f, i) => {
          let host = "";
          try { host = f.source ? new URL(f.source).hostname.replace(/^www\./, "") : ""; } catch { /* keep empty */ }
          return (
            <li key={i} className="text-[12.5px]">
              <p className="text-fg-2">{f.question}</p>
              <p className="flex flex-wrap items-center gap-2">
                <span className="font-semibold text-fg">{f.number != null ? fmtNum(f.number) : f.answer}</span>
                <StatusPill variant={f.measured ? "ok" : "neutral"}>{f.measured ? "measured" : "estimated"}</StatusPill>
              </p>
              {f.number != null && f.answer && <p className="text-fg-3">{f.answer}</p>}
              {host && <a href={f.source} target="_blank" rel="noreferrer" className="text-[12px] text-accent hover:underline">{host}</a>}
            </li>
          );
        })}
      </ul>
      {findings.length > 4 && (
        <button type="button" className={BTN_QUIET + " mt-2"} aria-expanded={all} onClick={() => setAll((v) => !v)}>
          {all ? "Fewer" : `All ${findings.length} findings`}
        </button>
      )}
    </section>
  );
}

export function GoalPage({ customerId, goalId }: { customerId: string; goalId: string }) {
  const qc = useQueryClient();
  const toast = useToast();
  const navigate = useNavigate();
  const [error, setError] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const detail = useQuery({ queryKey: ["goal", customerId, goalId], queryFn: () => api.getGoal(customerId, goalId), refetchInterval: 30_000 });
  const progress = useQuery({ queryKey: ["goal-progress", customerId, goalId], queryFn: () => api.goalProgress(customerId, goalId),
    refetchInterval: 60_000, enabled: !!detail.data?.goal.plan });
  const runs = useQuery({ queryKey: ["goal-runs", customerId, goalId], queryFn: () => api.goalRuns(customerId, goalId), refetchInterval: 30_000 });
  const lattices = useQuery({ queryKey: ["lattices", customerId], queryFn: () => api.listLattices(customerId) });
  const refresh = () => {
    for (const k of ["goal", "goal-progress", "goal-runs"]) qc.invalidateQueries({ queryKey: [k, customerId, goalId] });
    qc.invalidateQueries({ queryKey: ["goals", customerId] });
  };
  const fail = (fallback: string) => (e: unknown) => setError(errorMessage(e, fallback));
  const update = useMutation({ mutationFn: (body: Parameters<typeof api.updateGoal>[2]) => api.updateGoal(customerId, goalId, body),
    onSuccess: () => { setError(null); refresh(); }, onError: fail("Couldn't change the goal.") });
  const replan = useMutation({ mutationFn: () => api.replanGoal(customerId, goalId),
    onSuccess: () => { setError(null); toast("A fresh plan is on its way. Finished steps stay finished."); refresh(); }, onError: fail("Couldn't ask for a new plan.") });
  const runAnyway = useMutation({ mutationFn: () => api.runGoalAnyway(customerId, goalId),
    onSuccess: () => { setError(null); toast("Running with what it has. It checks in within a minute."); refresh(); }, onError: fail("Couldn't release the goal.") });
  const runNow = useMutation({ mutationFn: () => api.runGoalNow(customerId, goalId),
    onSuccess: () => { setError(null); toast("It checks in within a minute."); refresh(); }, onError: fail("Couldn't start a check-in.") });
  const remove = useMutation({ mutationFn: () => api.deleteGoal(customerId, goalId),
    onSuccess: () => { toast("Goal deleted."); qc.invalidateQueries({ queryKey: ["goals", customerId] }); navigate("/goals"); }, onError: fail("Couldn't delete the goal.") });
  const item = useMutation({
    mutationFn: ({ id, body }: { id: string; body: Parameters<typeof api.goalItemAction>[2] }) => api.goalItemAction(customerId, id, body),
    onSuccess: (_o, v) => {
      setError(null);
      toast(v.body.action === "send" ? "Sent." : v.body.action === "done" ? "Marked done." : v.body.action === "reopen" ? "Reopened."
        : v.body.action === "answer" ? "Answered. Every check-in works on it until it's handled." : "Dismissed.");
      refresh();
      if (v.body.action === "answer") qc.invalidateQueries({ queryKey: ["goal-comments", customerId, goalId] });
    },
    onError: fail("That didn't go through."),
  });
  const back = <Link to="/goals" className={BTN_SECONDARY}><ArrowLeft className="h-4 w-4" />All goals</Link>;
  if (detail.isError) {
    return (<><PageHeader title="Goal" actions={back} /><ErrorBanner title="Couldn't load the goal" error={detail.error} onRetry={() => detail.refetch()} /></>);
  }
  const g = detail.data?.goal;
  if (!g) return <PageHeader title="Goal" meta="Loading…" actions={back} />;
  const items = detail.data?.items ?? [];
  const plan = g.plan;
  const steps = plan?.steps ?? [];
  const finite = steps.filter((s) => !s.repeats);
  const repeating = steps.length - finite.length;
  const currentN = finite.find((s) => !s.done)?.n ?? null;
  const st = goalState(g);
  const busy = update.isPending || replan.isPending || runNow.isPending || runAnyway.isPending || remove.isPending || item.isPending;
  const checkIns = (runs.data?.runs ?? []).filter((r) => r.phase !== "plan");
  const calls = checkIns.reduce((t, r) => t + (r.calls ?? 0), 0);
  const ids = new Set((plan?.lattices ?? []).map((l) => l.lattice_id).filter(Boolean));
  const rowsByName = new Map((lattices.data?.lattices ?? []).map((l) => [l.lattice_id, l.rows]));
  const rows = (lattices.data?.lattices ?? []).filter((l) => ids.has(l.lattice_id)).reduce((t, l) => t + l.rows, 0);
  const next = g.status !== "active" ? (g.status === "paused" ? "Paused" : st.label)
    : g.next_run_at ? (new Date(g.next_run_at).getTime() <= Date.now() ? "Within a minute" : fmtRelative(g.next_run_at)) : "Within a minute";
  const uses = (plan?.needs.apps ?? []).map((a) => APP_NAMES[a] ?? a);
  return (
    <div className="space-y-5">
      <nav aria-label="Breadcrumb" className="text-[12.5px] text-fg-3">
        <Link to="/goals" className="text-accent hover:underline">Goals</Link> / {g.title}
      </nav>
      <PageHeader title={g.title} meta={<StatusPill variant={st.variant} dot>{st.label}</StatusPill>}
        actions={<div className="flex flex-wrap gap-2">
          {g.status === "active" && <button type="button" className={BTN_QUIET} disabled={busy} onClick={() => update.mutate({ status: "paused" })}>Pause</button>}
          {(g.status === "paused" || g.status === "done" || g.status === "archived") && (
            <button type="button" className={BTN_PRIMARY} disabled={busy} onClick={() => update.mutate({ status: "active" })}>Resume</button>)}
          {g.status === "active" && <button type="button" className={BTN_SECONDARY} disabled={busy || !g.planned_at} onClick={() => runNow.mutate()}>Run now</button>}
          {g.status !== "archived" && g.status !== "done" && g.status !== "needs_setup" && (
            <button type="button" className={BTN_SECONDARY} disabled={busy} onClick={() => replan.mutate()}>Replan</button>)}
        </div>} />
      {error && <ErrorBanner title="That didn't go through" message={error} />}
      <div className="space-y-1">
        <p className="whitespace-pre-wrap text-[14px] text-fg">{g.directive}</p>
        {plan?.success_measure && <p className="text-[13px] text-fg-2"><span className="font-medium text-fg">Success:</span> {plan.success_measure}</p>}
        {plan?.standing_instructions && <p className="text-[13px] text-fg-2"><span className="font-medium text-fg">Always:</span> {plan.standing_instructions}</p>}
      </div>

      <dl className="grid grid-cols-2 gap-3 min-[900px]:grid-cols-4" aria-label="At a glance">
        {[
          ["Steps accepted", `${finite.filter((s) => s.done).length} of ${finite.length}`, repeating ? `plus ${repeating} repeating` : ""],
          ["Check-ins", fmtNum(checkIns.length), calls ? `${fmtNum(calls)} tool calls total` : ""],
          ["Rows written", fmtNum(rows), ids.size ? `${ids.size} lattice${ids.size === 1 ? "" : "s"}` : ""],
          ["Next check-in", next, plan ? cadenceText(plan.check_every_minutes) : ""],
        ].map(([k, v, sub]) => (
          <div key={k} className="rounded-xl border border-line bg-surface px-3 py-2.5">
            <dt className="text-[11.5px] font-semibold uppercase tracking-wide text-fg-3">{k}</dt>
            <dd className="mt-0.5 text-[18px] font-semibold tabular-nums text-fg">{v}</dd>
            {sub && <dd className="text-[12px] text-fg-3">{sub}</dd>}
          </div>
        ))}
      </dl>

      <NeedsYou customerId={customerId} goal={g} items={items} busy={busy} onError={setError}
        act={(id, body) => item.mutate({ id, body })} rerun={() => replan.mutate()} release={() => runAnyway.mutate()} />

      <div className="grid gap-5 min-[1100px]:grid-cols-[minmax(0,1fr)_320px]">
        <div className="min-w-0 space-y-6">
          <section aria-labelledby="steps-h" className="space-y-3">
            <div className="flex flex-wrap items-baseline gap-2">
              <h2 id="steps-h" className="flex-1 text-[15px] font-semibold text-fg">Steps</h2>
              {progress.dataUpdatedAt > 0 && <span className="text-[12px] text-fg-3">Measured live · updated {fmtRelative(new Date(progress.dataUpdatedAt).toISOString())}</span>}
            </div>
            {!plan || steps.length === 0 ? (
              <p className="text-[13px] text-fg-3">{g.status === "deciding" ? "Written once you answer the questions above." : g.status === "active" ? "Being written now." : "No plan yet."}</p>
            ) : (
              <ol className="space-y-3">
                {steps.map((s) => (
                  <StepCard key={s.n} customerId={customerId} goal={g} step={s} busy={busy}
                    progress={progress.data?.steps.find((p) => p.n === s.n)} current={s.n === currentN}
                    canGoBack={!s.repeats && currentN !== null && s.n <= currentN && g.status !== "done" && g.status !== "archived"} />
                ))}
              </ol>
            )}
          </section>
          <CheckIns customerId={customerId} goal={g} />
        </div>
        <aside className="space-y-4">
          <Survey goal={g} />
          {(plan?.lattices ?? []).length > 0 && (
            <section aria-labelledby="lat-h" className="rounded-xl border border-line bg-surface p-4">
              <h2 id="lat-h" className="text-[14px] font-semibold text-fg">Lattices</h2>
              <ul className="mt-2 space-y-1.5 text-[13px]">
                {(plan?.lattices ?? []).map((l) => (
                  <li key={l.name} className="flex items-center gap-2">
                    {l.lattice_id ? <Link to={`/lattices/${encodeURIComponent(l.lattice_id)}`} className="min-w-0 flex-1 truncate font-mono text-[12.5px] text-accent hover:underline">{l.name}</Link>
                      : <span className="min-w-0 flex-1 truncate font-mono text-[12.5px]">{l.name}</span>}
                    <span className="tabular-nums text-fg-2">{l.lattice_id && rowsByName.has(l.lattice_id) ? `${fmtNum(rowsByName.get(l.lattice_id) ?? 0)} rows` : ""}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}
          <section aria-label="Permissions" className="space-y-3 rounded-xl border border-line bg-surface p-4">
            <h2 className="text-[14px] font-semibold text-fg">Permissions</h2>
            <Switch checked={g.may_send} disabled={busy} onChange={(v) => update.mutate({ may_send: v })} label="Send email on its own"
              hint={g.may_send ? "Sends replies that follow its instructions; every send is recorded." : "Replies wait for you to approve."} />
            <Switch checked={g.may_post} disabled={busy} onChange={(v) => update.mutate({ may_post: v })} label="Post or create on its own"
              hint={g.may_post ? "Posts, events and page additions go out; each is recorded." : "They wait for you to approve."} />
            {uses.length > 0 && <p className="text-[12px] text-fg-3">Uses {uses.join(", ")}.</p>}
          </section>
          <History items={items} busy={busy} act={(id, body) => item.mutate({ id, body })} />
          <div className="flex flex-wrap items-center gap-3 text-[12.5px]">
            {g.status !== "done" && <button type="button" className={BTN_QUIET} disabled={busy} onClick={() => update.mutate({ status: "done" })}>Mark done</button>}
            <span className="flex-1" />
            <button type="button" className="text-fg-3 hover:text-bad disabled:opacity-50" disabled={busy} onClick={() => setConfirmDelete(true)}>Delete goal</button>
          </div>
          <ConfirmInline open={confirmDelete} permanent busy={remove.isPending} confirmLabel="Delete goal"
            message="Delete this goal and everything it filed? What it already sent stays sent." onConfirm={() => remove.mutate()} onCancel={() => setConfirmDelete(false)} />
        </aside>
      </div>
    </div>
  );
}

/** /goals/:goalId */
export function GoalRoute() {
  const { selectedCustomerId } = useSelectedCustomer();
  const { goalId } = useParams();
  if (!selectedCustomerId || !goalId) {
    return (<><PageHeader title="Goal" /><EmptyState icon={Target} title="No customer selected" help="Pick a customer above to see their goals." /></>);
  }
  return <GoalPage key={goalId} customerId={selectedCustomerId} goalId={goalId} />;
}
