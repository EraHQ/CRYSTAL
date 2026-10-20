// Scheduled runs (launch feature, 2026-10-18; Q78-Q81): the Schedules
// section on the Cognition page. A list with next run, last outcome,
// Pause / Resume and Delete, and a small create form. Copy rule: percent
// only, never dollars; "Scheduled runs use your daily AI capacity. If it
// is used up, that run is skipped."
// Delivery (Q90=A): an optional https webhook per schedule (Slack,
// Discord, or any URL such as Zapier / Make / n8n). The signing secret
// is shown ONCE, right after it is minted, in a notice the person
// dismisses; it is never fetched again.
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CalendarClock } from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import type { EditScheduleRequest, ScheduledRun, ScheduleWriteResponse } from "@/lib/types";
import { ConfirmInline, EmptyState, ErrorBanner, Skeleton, StatusPill, useToast } from "@/components/ui";
import type { StatusVariant } from "@/components/ui";
import { BTN_PRIMARY, BTN_SECONDARY, INPUT } from "@/pages/settings/shared";
import { fmtDateTime, fmtRelative } from "@/lib/utils";

export const OUTCOME: Record<string, { label: string; variant: StatusVariant }> = {
  queued: { label: "Queued", variant: "info" },
  completed: { label: "Completed", variant: "ok" },
  failed: { label: "Failed", variant: "bad" },
  skipped_capacity: { label: "Skipped: no capacity", variant: "warn" },
};
export const STATUS: Record<string, { label: string; variant: StatusVariant }> = {
  active: { label: "Active", variant: "ok" },
  paused: { label: "Paused", variant: "neutral" },
  parked: { label: "Parked after 3 failures", variant: "bad" },
  done: { label: "Done", variant: "neutral" },
};
const WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"];

export function browserTimeZone(): string {
  try { return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC"; } catch { return "UTC"; }
}

export function hostOf(url: string | null | undefined): string {
  if (!url) return "";
  try { return new URL(url).host; } catch { return url; }
}

export function deliveryLabel(s: { last_delivery: string | null; last_delivery_at: string | null }): string {
  if (!s.last_delivery) return "No delivery yet";
  const when = s.last_delivery_at ? ` ${fmtRelative(s.last_delivery_at)}` : "";
  return s.last_delivery === "sent" ? `Delivered${when}` : `Delivery ${s.last_delivery}${when}`;
}

function SecretNotice({ title, secret, onDone }: { title: string; secret: string; onDone: () => void }) {
  const toast = useToast();
  const copy = async () => {
    try { await navigator.clipboard.writeText(secret); toast("Secret copied."); } catch { toast("Couldn't copy. Select the secret and copy it."); }
  };
  return (
    <div role="status" className="mb-3 rounded-xl border border-line bg-inset p-3 text-[12.5px] text-fg-2">
      <p className="font-medium text-fg">Signing secret for "{title}", shown once</p>
      <p className="mt-0.5">Every delivery to a plain URL carries <code>X-Crystal-Signature</code>, an HMAC-SHA256 of the body with this secret. Slack and Discord ignore it; keep it if you verify signatures.</p>
      <code className="mt-2 block select-all break-all rounded-md border border-line bg-surface px-2 py-1.5 text-[12px] text-fg">{secret}</code>
      <div className="mt-2 flex gap-2">
        <button type="button" onClick={() => void copy()} className={BTN_SECONDARY}>Copy</button>
        <button type="button" onClick={onDone} className={BTN_SECONDARY}>Done</button>
      </div>
    </div>
  );
}

function WebhookEditor({ customerId, schedule, onSecret, onClose, onError }: {
  customerId: string; schedule: ScheduledRun; onSecret: (title: string, secret: string) => void; onClose: () => void; onError: (m: string) => void;
}) {
  const qc = useQueryClient();
  const toast = useToast();
  const [url, setUrl] = useState(schedule.deliver_url ?? "");
  const save = useMutation({
    mutationFn: (value: string) => api.setScheduleDelivery(customerId, schedule.id, value),
    onSuccess: (r, value) => {
      if (r.deliver_secret) onSecret(schedule.title, r.deliver_secret);
      toast(value ? "Webhook saved." : "Webhook removed.");
      qc.invalidateQueries({ queryKey: ["schedules", customerId] });
      onClose();
    },
    onError: (e) => onError(errorMessage(e, "Couldn't save the webhook.")),
  });
  return (
    <form aria-label={`Webhook for ${schedule.title}`} onSubmit={(e) => { e.preventDefault(); save.mutate(url.trim()); }}
      className="mt-2 space-y-2 rounded-lg border border-line bg-inset p-3">
      <label className="block text-[12.5px] font-medium text-fg">Send results to
        <input type="url" value={url} onChange={(e) => setUrl(e.target.value)} maxLength={2048} placeholder="https://hooks.slack.com/services/…" className={INPUT + " mt-1"} />
      </label>
      <p className="text-[12px] text-fg-3">A Slack or Discord webhook, or any https URL (Zapier, Make, n8n). The result is posted when a run completes.</p>
      <div className="flex flex-wrap gap-2">
        <button type="submit" disabled={save.isPending || !url.trim()} className={BTN_PRIMARY}>Save</button>
        {schedule.deliver_url && <button type="button" disabled={save.isPending} onClick={() => save.mutate("")} className={BTN_SECONDARY}>Remove webhook</button>}
        <button type="button" onClick={onClose} className={BTN_SECONDARY}>Cancel</button>
      </div>
    </form>
  );
}

export function Schedules({ customerId }: { customerId: string }) {
  const qc = useQueryClient();
  const toast = useToast();
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [editingHook, setEditingHook] = useState<string | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [secret, setSecret] = useState<{ title: string; secret: string } | null>(null);
  const list = useQuery({ queryKey: ["schedules", customerId], queryFn: () => api.listSchedules(customerId), refetchInterval: 30_000 });
  const refresh = () => qc.invalidateQueries({ queryKey: ["schedules", customerId] });
  const fail = (fallback: string) => (e: unknown) => setError(errorMessage(e, fallback));
  const setStatus = useMutation({
    mutationFn: ({ id, status }: { id: string; status: "active" | "paused" }) => api.setScheduleStatus(customerId, id, status),
    onSuccess: (_r, v) => { setError(null); toast(v.status === "paused" ? "Schedule paused." : "Schedule resumed."); refresh(); },
    onError: fail("Couldn't change the schedule."),
  });
  const remove = useMutation({
    mutationFn: (id: string) => api.deleteSchedule(customerId, id),
    onSuccess: () => { setError(null); setConfirmDelete(null); toast("Schedule deleted."); refresh(); },
    onError: fail("Couldn't delete the schedule."),
  });
  // Run now (2026-10-20): one run this minute, the same run the scheduler
  // would make; the result lands under Results like any other.
  const runNow = useMutation({
    mutationFn: (id: string) => api.runScheduleNow(customerId, id),
    onSuccess: () => { setError(null); toast("Queued. The result appears under Results in a few minutes."); refresh(); qc.invalidateQueries({ queryKey: ["schedule-results", customerId] }); },
    onError: fail("Couldn't start the run."),
  });
  const rows = list.data?.schedules ?? [];
  const atCap = (list.data?.active ?? 0) >= (list.data?.max_active ?? Infinity);

  return (
    <section aria-labelledby="schedules-title">
      <div className="mb-2 flex flex-wrap items-end justify-between gap-2">
        <div>
          <h2 id="schedules-title" className="text-[15px] font-semibold text-fg">Scheduled runs{rows.length ? ` ${rows.length}` : ""}</h2>
          <p className="text-[12.5px] text-fg-3">Something it should do on its own, in your words. Scheduled runs use your daily AI capacity. If it is used up, that run is skipped.</p>
        </div>
        {!creating && (
          <button type="button" onClick={() => setCreating(true)} disabled={atCap} className={BTN_SECONDARY}
            title={atCap ? `This plan allows ${list.data?.max_active} schedule${list.data?.max_active === 1 ? "" : "s"}` : undefined}>
            New schedule
          </button>
        )}
      </div>
      {error && <ErrorBanner title="Schedules" message={error} />}
      {secret && <SecretNotice title={secret.title} secret={secret.secret} onDone={() => setSecret(null)} />}
      {creating && (
        <ScheduleForm customerId={customerId} onCancel={() => setCreating(false)}
          onDone={(r) => { setCreating(false); if (r.deliver_secret) setSecret({ title: r.schedule.title, secret: r.deliver_secret }); refresh(); }} />
      )}
      {list.isError && <ErrorBanner title="Couldn't load schedules" error={list.error} onRetry={() => list.refetch()} />}
      {list.isLoading && <Skeleton rows={2} />}
      {!list.isLoading && !list.isError && rows.length === 0 && !creating && (
        <EmptyState icon={CalendarClock} title="Nothing scheduled" help="News relevant to your knowledge bank every morning, a brief on weekdays, a question researched every Friday. Create one above." />
      )}
      {rows.length > 0 && (
        <ul className="divide-y divide-line rounded-xl border border-line bg-surface">
          {rows.map((s) => {
            const st = STATUS[s.status] ?? { label: s.status, variant: "neutral" as StatusVariant };
            const oc = s.last_outcome ? OUTCOME[s.last_outcome] ?? { label: s.last_outcome, variant: "neutral" as StatusVariant } : null;
            const inFlight = s.last_outcome === "queued";
            if (editing === s.id) {
              return (
                <li key={s.id} className="px-3 py-2.5">
                  <ScheduleForm customerId={customerId} initial={s} onCancel={() => setEditing(null)}
                    onDone={() => { setEditing(null); toast("Schedule updated."); refresh(); }} />
                </li>
              );
            }
            return (
              <li key={s.id} className="px-3 py-2.5">
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-[13.5px] font-medium text-fg">{s.title}</span>
                      <StatusPill variant={st.variant} raw={s.status}>{st.label}</StatusPill>
                    </div>
                    <p className="mt-0.5 text-[12.5px] text-fg-2">{s.when}</p>
                    <p className="mt-0.5 text-[12px] text-fg-3">
                      {s.status === "active" && s.next_run_at ? <span title={fmtDateTime(s.next_run_at)}>Next run {nextRelative(s.next_run_at)}</span> : s.status === "done" ? "Ran once" : "Not running"}
                      {oc && <> · Last run {s.last_run_at ? fmtRelative(s.last_run_at) : ""}: <StatusPill variant={oc.variant} raw={s.last_outcome ?? undefined}>{oc.label}</StatusPill></>}
                    </p>
                    <p className="mt-1 text-[12.5px] text-fg-2">{s.prompt.length > 200 ? s.prompt.slice(0, 200) + "…" : s.prompt}</p>
                    {s.delivers && (
                      <p className="mt-0.5 text-[12px] text-fg-3">
                        Delivers to {hostOf(s.deliver_url)} · <span className={s.last_delivery && s.last_delivery !== "sent" ? "text-bad" : undefined}>{deliveryLabel(s)}</span>
                      </p>
                    )}
                    {s.last_result_summary && (
                      <p className="mt-1 text-[12.5px] text-fg-2">
                        Last result: {s.last_result_summary.length > 240 ? s.last_result_summary.slice(0, 240) + "…" : s.last_result_summary}
                        {s.last_crystal_id && <> <a href={`/bank/${encodeURIComponent(s.last_crystal_id)}`} className="text-accent hover:underline">Open</a></>}
                      </p>
                    )}
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <button type="button" onClick={() => runNow.mutate(s.id)} disabled={runNow.isPending || inFlight} className={BTN_PRIMARY}
                      title={inFlight ? "A run is already in progress" : "Run it now, without changing the schedule"}>{inFlight ? "Running…" : "Run now"}</button>
                    <button type="button" onClick={() => { setEditing(s.id); setEditingHook(null); }} className={BTN_SECONDARY}>Edit</button>
                    {s.status === "active" && <button type="button" onClick={() => setStatus.mutate({ id: s.id, status: "paused" })} disabled={setStatus.isPending} className={BTN_SECONDARY}>Pause</button>}
                    {(s.status === "paused" || s.status === "parked") && <button type="button" onClick={() => setStatus.mutate({ id: s.id, status: "active" })} disabled={setStatus.isPending} className={BTN_SECONDARY}>Resume</button>}
                    <button type="button" onClick={() => setEditingHook(editingHook === s.id ? null : s.id)} className={BTN_SECONDARY}>{s.delivers ? "Webhook" : "Add webhook"}</button>
                    <button type="button" onClick={() => setConfirmDelete(s.id)} disabled={remove.isPending} className={BTN_SECONDARY}>Delete</button>
                  </div>
                </div>
                {editingHook === s.id && (
                  <WebhookEditor customerId={customerId} schedule={s} onClose={() => setEditingHook(null)} onError={setError}
                    onSecret={(title, value) => setSecret({ title, secret: value })} />
                )}
                <ConfirmInline open={confirmDelete === s.id} message={`Delete "${s.title}"? Runs already made stay in the run list.`} confirmLabel="Delete"
                  busy={remove.isPending} onCancel={() => setConfirmDelete(null)} onConfirm={() => remove.mutate(s.id)} />
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}

export function nextRelative(iso: string, now: number = Date.now()): string {
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return "";
  const s = Math.round((t - now) / 1000);
  if (s <= 60) return "within a minute";
  const m = Math.round(s / 60);
  if (m < 60) return `in ${m} min`;
  const h = Math.round(m / 60);
  if (h < 36) return `in ${h} hour${h === 1 ? "" : "s"}`;
  const d = Math.round(h / 24);
  return `in ${d} day${d === 1 ? "" : "s"}`;
}

// The fixed brief wording, offered as a one-click template. The engine's
// "brief" kind is the same words; the console sends the task as text so
// the person can edit it.
export const BRIEF_TEMPLATE = "Summarize what changed in my memory since the last run, the open gaps, and what is waiting for my review.";
export const NEWS_TEMPLATE = "Gather news from the last 24 hours relevant to the subjects in my knowledge bank and cite every source.";

type Cadence = "once" | "daily" | "weekdays" | "weekly" | "hourly";

/** HH:MM from minutes after midnight, for the time input. */
export function timeOf(atMinute: number | null | undefined): string {
  if (atMinute == null) return "08:00";
  return `${String(Math.floor(atMinute / 60)).padStart(2, "0")}:${String(atMinute % 60).padStart(2, "0")}`;
}
/** A datetime-local value (the browser's zone) from an ISO instant. */
export function localInputOf(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`;
}

// One form for New schedule and Edit (2026-10-20): with `initial` it is
// prefilled from the row and PATCHes the definition; the webhook keeps
// its own editor on the row.
function ScheduleForm({ customerId, initial, onDone, onCancel }: { customerId: string; initial?: ScheduledRun; onDone: (r: ScheduleWriteResponse) => void; onCancel: () => void }) {
  const cadences: Cadence[] = ["once", "daily", "weekdays", "weekly", "hourly"];
  const [cadence, setCadence] = useState<Cadence>(initial && cadences.includes(initial.cadence as Cadence) ? (initial.cadence as Cadence) : "daily");
  const [time, setTime] = useState(timeOf(initial?.at_minute));
  const [weekday, setWeekday] = useState(initial?.weekday != null ? WEEKDAYS[initial.weekday] ?? "monday" : "monday");
  const [everyHours, setEveryHours] = useState(initial?.every_hours ? String(initial.every_hours) : "6");
  const [runAt, setRunAt] = useState(localInputOf(initial?.run_once_at));
  const [tz] = useState(initial?.tz || browserTimeZone());
  const [question, setQuestion] = useState(initial?.prompt ?? "");
  const [title, setTitle] = useState(initial?.title ?? "");
  const [deliverUrl, setDeliverUrl] = useState("");
  const [error, setError] = useState<string | null>(null);
  const definition = (): EditScheduleRequest => ({
    kind: "research", cadence, tz,
    time: ["daily", "weekdays", "weekly"].includes(cadence) ? time : undefined,
    weekday: cadence === "weekly" ? weekday : undefined,
    every_hours: cadence === "hourly" ? Number(everyHours) || 1 : undefined,
    run_at: cadence === "once" && runAt ? new Date(runAt).toISOString() : undefined,
    question: question.trim(),
    title: title.trim() || undefined,
  });
  const create = useMutation({
    mutationFn: () => initial
      ? api.editSchedule(customerId, initial.id, definition())
      : api.createSchedule(customerId, { ...definition(), kind: "research", cadence, tz, deliver_url: deliverUrl.trim() || undefined }),
    onSuccess: onDone,
    onError: (e) => setError(errorMessage(e, initial ? "Couldn't save the schedule." : "Couldn't create the schedule.")),
  });
  const needsTime = ["daily", "weekdays", "weekly"].includes(cadence);
  return (
    <form aria-label={initial ? `Edit ${initial.title}` : "New schedule"} onSubmit={(e) => { e.preventDefault(); create.mutate(); }}
      className="mb-3 space-y-3 rounded-xl border border-line bg-surface p-4">
      <label className="block text-[12.5px] font-medium text-fg">What should it do?
        <textarea value={question} onChange={(e) => setQuestion(e.target.value)} rows={3} maxLength={4000} required
          placeholder={NEWS_TEMPLATE} className={INPUT + " mt-1"} />
        <span className="mt-1 block text-[12px] font-normal text-fg-3">Limited to the web and your knowledge bank for now; a run can't act on other services yet.</span>
      </label>
      <div className="flex flex-wrap gap-2">
        <button type="button" onClick={() => setQuestion(NEWS_TEMPLATE)} className={BTN_SECONDARY}>Use the news template</button>
        <button type="button" onClick={() => setQuestion(BRIEF_TEMPLATE)} className={BTN_SECONDARY}>Use the brief template</button>
      </div>
      <div className="grid gap-3 min-[720px]:grid-cols-2">
        <label className="block text-[12.5px] font-medium text-fg">When
          <select value={cadence} onChange={(e) => setCadence(e.target.value as typeof cadence)} className={INPUT + " mt-1"}>
            <option value="weekdays">Weekdays</option>
            <option value="daily">Every day</option>
            <option value="weekly">Weekly</option>
            <option value="hourly">Every N hours</option>
            <option value="once">Once</option>
          </select>
        </label>
        {needsTime && (
          <label className="block text-[12.5px] font-medium text-fg">Time ({tz})
            <input type="time" value={time} onChange={(e) => setTime(e.target.value)} className={INPUT + " mt-1"} required />
          </label>
        )}
        {cadence === "weekly" && (
          <label className="block text-[12.5px] font-medium text-fg">Day
            <select value={weekday} onChange={(e) => setWeekday(e.target.value)} className={INPUT + " mt-1"}>
              {WEEKDAYS.map((d) => <option key={d} value={d}>{d.charAt(0).toUpperCase() + d.slice(1)}</option>)}
            </select>
          </label>
        )}
        {cadence === "hourly" && (
          <label className="block text-[12.5px] font-medium text-fg">Every how many hours
            <input type="number" min={1} max={168} value={everyHours} onChange={(e) => setEveryHours(e.target.value)} className={INPUT + " mt-1"} />
          </label>
        )}
        {cadence === "once" && (
          <label className="block text-[12.5px] font-medium text-fg">Date and time ({tz})
            <input type="datetime-local" value={runAt} onChange={(e) => setRunAt(e.target.value)} className={INPUT + " mt-1"} required />
          </label>
        )}
      </div>
      <label className="block text-[12.5px] font-medium text-fg">Name <span className="font-normal text-fg-3">(optional)</span>
        <input value={title} onChange={(e) => setTitle(e.target.value)} maxLength={200} placeholder="Morning brief" className={INPUT + " mt-1"} />
      </label>
      {!initial && (
        <label className="block text-[12.5px] font-medium text-fg">Send results to <span className="font-normal text-fg-3">(optional)</span>
          <input type="url" value={deliverUrl} onChange={(e) => setDeliverUrl(e.target.value)} maxLength={2048} placeholder="https://hooks.slack.com/services/…" className={INPUT + " mt-1"} />
          <span className="mt-1 block text-[12px] font-normal text-fg-3">A Slack or Discord webhook, or any https URL (Zapier, Make, n8n). Without one, results wait in your bank and in any connected client's status.</span>
        </label>
      )}
      {error && <ErrorBanner title={initial ? "Couldn't save the schedule" : "Couldn't create the schedule"} message={error} />}
      <div className="flex flex-wrap gap-2">
        <button type="submit" disabled={create.isPending || !question.trim()} className={BTN_PRIMARY}>
          {initial ? (create.isPending ? "Saving…" : "Save changes") : (create.isPending ? "Creating…" : "Create schedule")}
        </button>
        <button type="button" onClick={onCancel} className={BTN_SECONDARY}>Cancel</button>
      </div>
    </form>
  );
}
