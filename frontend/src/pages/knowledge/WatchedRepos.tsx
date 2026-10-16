// Knowledge › Watched repos (Console Refine PR 5, 5.3 item 7). Modes read
// in plain words; a failed sync shows the server's message with an inline
// token replacement (PATCH {token}, new in this PR) so a revoked GitHub
// token no longer means removing the watch; Remove asks inline.
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { GitBranch } from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import { ConfirmInline, EmptyState, ErrorBanner, Skeleton, StatusPill } from "@/components/ui";
import type { Watch, WatchActivity } from "@/lib/types";
import { fmtRelative } from "@/lib/utils";
import { fmtChecked, REVIEW_MODE, WATCH_STATE } from "./shared";

const EMPTY_FORM = { source_name: "", repo: "", branch: "main", review_mode: "auto" as "auto" | "gated", cadence_minutes: 15, token: "" };

export function WatchedRepos({ customerId }: { customerId: string }) {
  const qc = useQueryClient();
  const [adding, setAdding] = useState(false);
  const [form, setForm] = useState(EMPTY_FORM);
  const [error, setError] = useState<string | null>(null);

  const watches = useQuery({
    queryKey: ["watches", customerId],
    queryFn: () => api.listWatches(customerId),
    refetchInterval: (q) => ((q.state.data ?? []).some((w) => w.sync_state === "syncing") ? 10000 : 60000),
  });
  const refresh = () => qc.invalidateQueries({ queryKey: ["watches", customerId] });

  const create = useMutation({
    mutationFn: () => api.createWatch(customerId, {
      scheme: "git", source_name: form.source_name.trim(),
      config: { repo: form.repo.trim(), branch: form.branch.trim() || "main" },
      review_mode: form.review_mode, cadence_minutes: form.cadence_minutes,
      token: form.token || undefined,
    }),
    onSuccess: () => { setError(null); setForm(EMPTY_FORM); setAdding(false); refresh(); },
    onError: (e) => setError(errorMessage(e, "Couldn't add this watch.")),
  });

  const rows = watches.data ?? [];
  const field = "mt-0.5 block w-full rounded-md border border-line bg-inset px-2.5 py-1.5 text-[13px] text-fg outline-none focus:border-accent";

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <p className="text-[12.5px] text-fg-2">A push to a watched repo becomes memory on its own. Private repos need a GitHub token; it is stored encrypted and never shown again.</p>
        <button type="button" onClick={() => setAdding((v) => !v)} aria-expanded={adding}
          className="rounded-md border border-line px-3 py-1.5 text-[12.5px] font-medium text-fg-2 hover:bg-inset hover:text-fg">
          {adding ? "Cancel" : "Watch a repo"}
        </button>
      </div>
      {error && <ErrorBanner title="Watched repos" message={error} className="mb-3" />}

      {adding && (
        <form onSubmit={(e) => { e.preventDefault(); if (form.source_name.trim() && form.repo.trim()) create.mutate(); else setError("Source name and repo are required."); }}
          className="mb-4 grid gap-3 rounded-lg border border-line bg-surface p-3 min-[720px]:grid-cols-2">
          <label className="block text-[12px] text-fg-2 min-[720px]:col-span-2">Source name
            <input value={form.source_name} onChange={(e) => setForm({ ...form, source_name: e.target.value })} className={field}
              placeholder="e.g. crystal-cache-v2" maxLength={200} />
            <span className="mt-0.5 block text-fg-3">The name imports resolve under. It survives a clone rename.</span>
          </label>
          <label className="block text-[12px] text-fg-2 min-[720px]:col-span-2">Repo
            <input value={form.repo} onChange={(e) => setForm({ ...form, repo: e.target.value })} className={field}
              placeholder="owner/name or a GitHub URL" />
          </label>
          <label className="block text-[12px] text-fg-2">Branch
            <input value={form.branch} onChange={(e) => setForm({ ...form, branch: e.target.value })} className={field} />
          </label>
          <label className="block text-[12px] text-fg-2">When a push arrives
            <select value={form.review_mode} onChange={(e) => setForm({ ...form, review_mode: e.target.value as "auto" | "gated" })} className={field}>
              <option value="auto">Saves automatically (auto)</option>
              <option value="gated">Review first (gated)</option>
            </select>
          </label>
          <label className="block text-[12px] text-fg-2">GitHub token
            <input type="password" value={form.token} onChange={(e) => setForm({ ...form, token: e.target.value })} className={field}
              autoComplete="off" placeholder="ghp_… (required for GitHub)" />
          </label>
          <label className="block text-[12px] text-fg-2">Check every (minutes)
            <input type="number" min={15} max={10080} value={form.cadence_minutes}
              onChange={(e) => setForm({ ...form, cadence_minutes: Math.max(15, Number(e.target.value) || 15) })} className={field} />
          </label>
          <div className="flex justify-end min-[720px]:col-span-2">
            <button type="submit" disabled={create.isPending}
              className="rounded-md bg-accent px-3 py-1.5 text-[13px] font-medium text-on-accent hover:bg-accent-hover disabled:opacity-50">
              {create.isPending ? "Adding…" : "Start watching"}
            </button>
          </div>
        </form>
      )}

      {watches.isLoading && <Skeleton rows={3} />}
      {watches.isError && <ErrorBanner title="Couldn't load watched repos" error={watches.error} onRetry={() => watches.refetch()} />}
      {!watches.isLoading && !watches.isError && rows.length === 0 && !adding && (
        <EmptyState icon={GitBranch} title="No watched repos"
          help="Watch a repo and every push is read into memory without an upload."
          action={<button type="button" onClick={() => setAdding(true)}
            className="rounded-md bg-accent px-3 py-1.5 text-[13px] font-medium text-on-accent hover:bg-accent-hover">Watch a repo</button>} />
      )}
      <div className="space-y-2">
        {rows.map((w) => <WatchRow key={w.id} customerId={customerId} w={w} onError={setError} onChanged={refresh} />)}
      </div>
    </div>
  );
}

function WatchRow({ customerId, w, onError, onChanged }: {
  customerId: string; w: Watch; onError: (m: string | null) => void; onChanged: () => void;
}) {
  const [showActivity, setShowActivity] = useState(false);
  const [removing, setRemoving] = useState(false);
  const [newToken, setNewToken] = useState("");
  const state = WATCH_STATE[w.sync_state ?? "waiting"] ?? WATCH_STATE.waiting;
  const target = w.scheme === "gdrive" ? (w.config?.folder_name || w.config?.folder_id || "") : `${w.config?.repo ?? ""}@${w.config?.branch || "main"}`;

  const update = useMutation({
    mutationFn: (body: { action?: "sync_now"; status?: "active" | "paused"; token?: string }) => api.updateWatch(customerId, w.id, body),
    onSuccess: () => { onError(null); setNewToken(""); onChanged(); },
    onError: (e) => onError(errorMessage(e, "Watch action failed.")),
  });
  const remove = useMutation({
    mutationFn: () => api.deleteWatch(customerId, w.id),
    onSuccess: () => { onError(null); setRemoving(false); onChanged(); },
    onError: (e) => { setRemoving(false); onError(errorMessage(e, "Couldn't remove this watch.")); },
  });
  const btn = "rounded-md border border-line px-2 py-0.5 text-[12px] text-fg-2 hover:bg-inset hover:text-fg disabled:opacity-50";

  return (
    <article className="rounded-lg border border-line bg-surface">
      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 px-3 py-2">
        <GitBranch aria-hidden className="h-3.5 w-3.5 shrink-0 text-fg-3" />
        <span className="text-[13.5px] font-medium text-fg">{w.source_name}</span>
        <span className="min-w-0 truncate font-mono text-[11.5px] text-fg-3" title={target}>{target}</span>
        <StatusPill variant={state.variant} raw={w.sync_state} dot={w.sync_state === "syncing"}>
          {state.label}{w.sync_state === "syncing" && w.inflight ? ` · ${w.inflight} in flight` : ""}
        </StatusPill>
        <span className="text-[11.5px] text-fg-3" title={`review_mode ${w.review_mode}`}>{REVIEW_MODE[w.review_mode] ?? w.review_mode}</span>
        <span className="text-[11.5px] text-fg-3">checked {fmtChecked(w.last_checked_at)}{w.last_state?.head ? ` · ${w.last_state.head.slice(0, 7)}` : ""}</span>
        <span className="ml-auto flex flex-wrap items-center gap-1.5">
          <button type="button" className={btn} onClick={() => setShowActivity((v) => !v)} aria-expanded={showActivity}>
            {showActivity ? "Hide activity" : "Activity"}
          </button>
          <button type="button" className={btn} disabled={update.isPending} onClick={() => update.mutate({ action: "sync_now" })}>Sync now</button>
          <button type="button" className={btn} disabled={update.isPending}
            onClick={() => update.mutate({ status: w.status === "active" ? "paused" : "active" })}>
            {w.status === "active" ? "Pause" : "Resume"}
          </button>
          <button type="button" onClick={() => setRemoving(true)} disabled={removing}
            className="rounded-md border border-line px-2 py-0.5 text-[12px] text-fg-2 hover:bg-bad-soft hover:text-bad disabled:opacity-50">Remove</button>
        </span>
      </div>
      {w.last_error && (
        <div className="border-t border-line px-3 py-2 text-[12.5px]">
          <p className="text-bad">{w.last_error}</p>
          {w.scheme === "git" && (
            <form className="mt-1.5 flex flex-wrap items-center gap-2"
              onSubmit={(e) => { e.preventDefault(); if (newToken.trim()) update.mutate({ token: newToken.trim() }); }}>
              <span className="text-fg-2">If the repo's token was revoked or expired, paste a new one and sync.</span>
              <input type="password" value={newToken} onChange={(e) => setNewToken(e.target.value)} autoComplete="off"
                aria-label="New GitHub token" placeholder="ghp_…" maxLength={512}
                className="min-w-[180px] rounded-md border border-line bg-inset px-2.5 py-1 text-[12.5px] text-fg outline-none focus:border-accent" />
              <button type="submit" disabled={!newToken.trim() || update.isPending}
                className="rounded-md bg-accent px-2.5 py-1 text-[12.5px] font-medium text-on-accent hover:bg-accent-hover disabled:opacity-40">
                {update.isPending ? "Saving…" : "Save and sync"}
              </button>
            </form>
          )}
        </div>
      )}
      <ConfirmInline open={removing} className="mx-3 mb-2" busy={remove.isPending}
        message={`Stop watching ${w.source_name}? Its crystals stay in the bank.`}
        confirmLabel="Remove watch" onCancel={() => setRemoving(false)} onConfirm={() => remove.mutate()} />
      {showActivity && <Activity customerId={customerId} watchId={w.id} />}
    </article>
  );
}

// workers/source_sync.py _emit event types, in plain words.
const EVENT_TEXT: Record<string, string> = {
  sync_started: "Sync started", cycle_completed: "Sync finished", error: "Error",
  file_retired: "File retired", file_ingested: "File read", file_skipped: "File skipped",
};

function Activity({ customerId, watchId }: { customerId: string; watchId: string }) {
  const activity = useQuery({
    queryKey: ["watch-activity", watchId],
    queryFn: () => api.watchActivity(customerId, watchId),
    refetchInterval: 15000,
  });
  const rows: WatchActivity[] = activity.data ?? [];
  if (activity.isLoading) return <p className="border-t border-line px-3 py-2 text-[12px] text-fg-3">Loading activity…</p>;
  if (activity.isError) return <ErrorBanner title="Couldn't load activity" error={activity.error} onRetry={() => activity.refetch()} className="m-3" />;
  if (rows.length === 0) return <p className="border-t border-line px-3 py-2 text-[12px] text-fg-3">No activity yet.</p>;
  return (
    <ul className="max-h-56 overflow-y-auto border-t border-line">
      {rows.map((a, i) => (
        <li key={`${a.created_at ?? ""}-${i}`} className="flex items-center gap-2 px-3 py-1 text-[12px]">
          <span className={a.event_type === "error" ? "text-bad" : "text-fg-3"} title={a.event_type}>{EVENT_TEXT[a.event_type] ?? a.event_type.replace(/_/g, " ")}</span>
          <span className="min-w-0 flex-1 truncate font-mono text-fg-2" title={a.label}>{a.label}</span>
          <span className="whitespace-nowrap text-fg-3">{fmtRelative(a.created_at)}</span>
        </li>
      ))}
    </ul>
  );
}
