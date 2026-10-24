// Admin › Beta (2026-10-24). Anthony: "Make a beta list for private
// invites to features like this. Everything but keep Gmail private." Each
// feature is open (everyone), invite (only invited emails or customers)
// or closed (nobody). An email invite binds to the account when that
// person signs in.
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, errorMessage } from "@/lib/api";
import type { BetaFeature } from "@/lib/types";
import { ErrorBanner, PageHeader, Segmented, StatusPill, useToast } from "@/components/ui";
import { fmtRelative } from "@/lib/utils";
import { BTN_PRIMARY, BTN_QUIET, INPUT } from "./settings/shared";

const STATUS_OPTIONS: { value: BetaFeature["status"]; label: string }[] = [
  { value: "open", label: "Everyone" },
  { value: "invite", label: "Invite only" },
  { value: "closed", label: "Off" },
];

export function AdminBetaPage() {
  const qc = useQueryClient();
  const toast = useToast();
  const [error, setError] = useState<string | null>(null);
  const [feature, setFeature] = useState("gmail");
  const [target, setTarget] = useState("");
  const [note, setNote] = useState("");
  const beta = useQuery({ queryKey: ["beta"], queryFn: () => api.getBeta() });
  const refresh = () => qc.invalidateQueries({ queryKey: ["beta"] });
  const setStatus = useMutation({
    mutationFn: ({ key, status }: { key: string; status: BetaFeature["status"] }) => api.setBetaFeature(key, status),
    onSuccess: () => { setError(null); toast("Saved."); refresh(); },
    onError: (e) => setError(errorMessage(e, "Couldn't change the feature.")),
  });
  const invite = useMutation({
    mutationFn: () => {
      const t = target.trim();
      return api.inviteBeta({ feature_key: feature, note: note.trim(), ...(t.includes("@") ? { email: t } : { customer_id: t }) });
    },
    onSuccess: () => { setError(null); setTarget(""); setNote(""); toast("Invited."); refresh(); },
    onError: (e) => setError(errorMessage(e, "Couldn't invite.")),
  });
  const uninvite = useMutation({
    mutationFn: (id: string) => api.uninviteBeta(id),
    onSuccess: () => { setError(null); toast("Invite removed."); refresh(); },
    onError: (e) => setError(errorMessage(e, "Couldn't remove the invite.")),
  });
  const features = beta.data?.features ?? [];
  const invites = beta.data?.invites ?? [];
  const busy = setStatus.isPending || invite.isPending || uninvite.isPending;
  return (
    <div className="space-y-4">
      <PageHeader title="Beta" meta="Who sees features before everyone does." />
      {error && <ErrorBanner title="That didn't go through" message={error} />}
      {beta.isError && <ErrorBanner title="Couldn't load the beta list" error={beta.error} onRetry={() => beta.refetch()} />}
      <section aria-label="Features" className="space-y-2">
        {features.map((f) => (
          <div key={f.key} className="flex flex-wrap items-center gap-3 rounded-xl border border-line bg-surface p-4">
            <div className="min-w-0 flex-1">
              <p className="text-[14px] font-semibold text-fg">{f.title}</p>
              <p className="text-[12.5px] text-fg-2">{f.description}</p>
            </div>
            <Segmented label={`${f.title} availability`} value={f.status} size="sm"
              options={STATUS_OPTIONS.map((o) => ({ value: o.value, label: o.label }))}
              onChange={(v) => setStatus.mutate({ key: f.key, status: v })} />
          </div>
        ))}
      </section>
      <section aria-label="Invites" className="rounded-xl border border-line bg-surface p-4">
        <h2 className="text-[15px] font-semibold text-fg">Invites</h2>
        <form className="mt-3 flex flex-wrap items-center gap-2" aria-label="Invite"
          onSubmit={(e) => { e.preventDefault(); if (target.trim()) invite.mutate(); }}>
          <select aria-label="Feature" className={INPUT + " w-auto"} value={feature} onChange={(e) => setFeature(e.target.value)}>
            {features.map((f) => <option key={f.key} value={f.key}>{f.title}</option>)}
          </select>
          <input aria-label="Email or customer id" className={INPUT + " min-w-[200px] flex-1"} placeholder="dana@example.com or cus_…"
            value={target} maxLength={320} onChange={(e) => setTarget(e.target.value)} />
          <input aria-label="Note" className={INPUT + " min-w-[140px] flex-1"} placeholder="Note (optional)" value={note} maxLength={256}
            onChange={(e) => setNote(e.target.value)} />
          <button type="submit" className={BTN_PRIMARY} disabled={busy || !target.trim()}>Invite</button>
        </form>
        {invites.length === 0 ? (
          <p className="mt-3 text-[13px] text-fg-3">No invites yet.</p>
        ) : (
          <ul className="mt-3 divide-y divide-line" aria-label="Invite list">
            {invites.map((i) => (
              <li key={i.id} className="flex flex-wrap items-center gap-2 py-2 text-[13px]">
                <StatusPill variant="vio">{features.find((f) => f.key === i.feature_key)?.title ?? i.feature_key}</StatusPill>
                <span className="min-w-0 flex-1 truncate text-fg">{i.email ?? i.customer_id}</span>
                {i.note && <span className="text-[12px] text-fg-3">{i.note}</span>}
                <span className="text-[12px] text-fg-3">{i.accepted_at ? `joined ${fmtRelative(i.accepted_at)}` : "not signed in yet"}</span>
                <button type="button" className={BTN_QUIET} disabled={busy} onClick={() => uninvite.mutate(i.id)}>Remove</button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
