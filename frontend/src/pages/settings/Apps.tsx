// Settings › Connections › Apps (2026-10-24, Q106-Q108). Anthony: "Can
// users choose what apps they want ... With a checkbox on what you want to
// do with the data this connector makes ... Only make crystals for x, y,
// or z." Each app is a card; each sign-in to it carries three switches:
// Watch (goals see what it shows, kept 7 days), Remember (only what the
// rules allow becomes memory) and Act (goals may send, post or create).
// Connecting opens Nango's Connect UI; the server binds the connection
// when Nango's webhook arrives, so the list refreshes a few times after.
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, errorMessage } from "@/lib/api";
import { openConnectUI } from "@/lib/nangoConnect";
import type { AppCatalogEntry, AppConnection, RememberRule } from "@/lib/types";
import { ConfirmInline, ErrorBanner, StatusPill, Switch, useToast } from "@/components/ui";
import { fmtRelative } from "@/lib/utils";
import { BTN_PRIMARY, BTN_QUIET, BTN_SECONDARY, INPUT, Section } from "./shared";

const FIELDS: { value: RememberRule["field"]; label: string }[] = [
  { value: "from", label: "From" }, { value: "subject", label: "Subject" }, { value: "text", label: "Text" },
  { value: "folder", label: "Folder" }, { value: "label", label: "Label" }, { value: "channel", label: "Channel" },
];
const MATCHES: { value: RememberRule["match"]; label: string }[] = [
  { value: "contains", label: "contains" }, { value: "starts_with", label: "starts with" }, { value: "equals", label: "is" },
];

export function statusPill(c: AppConnection) {
  if (c.status === "needs_reauth") return <StatusPill variant="bad" dot>Sign in again</StatusPill>;
  if (c.status === "paused") return <StatusPill variant="neutral" dot>Paused</StatusPill>;
  return <StatusPill variant="ok" dot>Connected</StatusPill>;
}

function RecentItems({ customerId, connectionId }: { customerId: string; connectionId: string }) {
  const items = useQuery({ queryKey: ["app-items", customerId, connectionId], queryFn: () => api.listAppItems(customerId, connectionId, 15) });
  if (items.isLoading) return <p className="text-[12.5px] text-fg-3">Loading…</p>;
  if (items.isError) return <ErrorBanner title="Couldn't load items" error={items.error} onRetry={() => items.refetch()} />;
  const list = items.data?.items ?? [];
  if (!list.length) return <p className="text-[12.5px] text-fg-3">Nothing yet. Items show here after the first check.</p>;
  return (
    <ul className="divide-y divide-line rounded-lg border border-line" aria-label="Recent items">
      {list.map((i) => (
        <li key={i.id} className="px-3 py-2">
          <div className="flex flex-wrap items-center gap-2">
            <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-fg">{i.title || i.external_id}</span>
            {i.remembered && <StatusPill variant="info">Remembered</StatusPill>}
            <span className="text-[11.5px] text-fg-3">{i.occurred_at ? fmtRelative(i.occurred_at) : ""}</span>
          </div>
          {(i.author || i.snippet) && <p className="mt-0.5 line-clamp-2 text-[12px] text-fg-2">{i.author ? `${i.author}: ` : ""}{i.snippet}</p>}
        </li>
      ))}
    </ul>
  );
}

function Rules({ conn, busy, onSave }: { conn: AppConnection; busy: boolean; onSave: (rules: RememberRule[]) => void }) {
  const [adding, setAdding] = useState(false);
  const [rule, setRule] = useState<RememberRule>({ decision: "allow", field: "subject", match: "contains", value: "" });
  const rules = conn.remember_rules;
  return (
    <div className="mt-2 rounded-lg border border-line px-3 py-2.5">
      <p className="text-[12.5px] font-medium text-fg">What becomes memory</p>
      <p className="text-[12px] text-fg-3">The first rule that matches decides. Nothing matching means it is only watched.</p>
      {rules.length === 0 && <p className="mt-2 text-[12.5px] text-fg-2">No rules: nothing is remembered yet.</p>}
      <ol className="mt-2 space-y-1" aria-label="Remember rules">
        {conn.rules_text.map((t, i) => (
          <li key={i} className="flex items-start gap-2 text-[12.5px] text-fg">
            <span className="w-4 shrink-0 text-fg-3 tabular-nums">{i + 1}.</span>
            <span className="min-w-0 flex-1">{t}</span>
            <button type="button" disabled={busy} className={BTN_QUIET} aria-label={`Remove rule ${i + 1}`}
              onClick={() => onSave(rules.filter((_r, j) => j !== i))}>Remove</button>
          </li>
        ))}
      </ol>
      {adding ? (
        <form className="mt-2 flex flex-wrap items-center gap-2" aria-label="New rule"
          onSubmit={(e) => { e.preventDefault(); if (!rule.value.trim()) return; onSave([...rules, { ...rule, value: rule.value.trim() }]); setAdding(false); setRule({ ...rule, value: "" }); }}>
          <select aria-label="Decision" className={INPUT + " w-auto"} value={rule.decision} onChange={(e) => setRule({ ...rule, decision: e.target.value as RememberRule["decision"] })}>
            <option value="allow">Remember</option><option value="deny">Never remember</option>
          </select>
          <select aria-label="Field" className={INPUT + " w-auto"} value={rule.field} onChange={(e) => setRule({ ...rule, field: e.target.value as RememberRule["field"] })}>
            {FIELDS.map((f) => <option key={f.value} value={f.value}>{f.label}</option>)}
          </select>
          <select aria-label="Match" className={INPUT + " w-auto"} value={rule.match} onChange={(e) => setRule({ ...rule, match: e.target.value as RememberRule["match"] })}>
            {MATCHES.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
          </select>
          <input aria-label="Value" className={INPUT + " min-w-[140px] flex-1"} placeholder="invoice" value={rule.value} maxLength={200}
            onChange={(e) => setRule({ ...rule, value: e.target.value })} />
          <button type="submit" disabled={busy || !rule.value.trim()} className={BTN_PRIMARY}>Add</button>
          <button type="button" onClick={() => setAdding(false)} className={BTN_SECONDARY}>Cancel</button>
        </form>
      ) : (
        <button type="button" disabled={busy || rules.length >= 30} onClick={() => setAdding(true)} className={BTN_SECONDARY + " mt-2"}>Add a rule</button>
      )}
    </div>
  );
}

function ConnectionRow({ customerId, app, conn, onError, onReconnect }: {
  customerId: string; app: AppCatalogEntry; conn: AppConnection; onError: (m: string) => void; onReconnect: (id: string) => void;
}) {
  const qc = useQueryClient();
  const toast = useToast();
  const [confirm, setConfirm] = useState(false);
  const [showItems, setShowItems] = useState(false);
  const refresh = () => qc.invalidateQueries({ queryKey: ["apps", customerId] });
  const update = useMutation({
    mutationFn: (body: Parameters<typeof api.updateApp>[2]) => api.updateApp(customerId, conn.id, body),
    onSuccess: () => refresh(),
    onError: (e) => onError(errorMessage(e, "Couldn't save that.")),
  });
  const sync = useMutation({
    mutationFn: () => api.syncApp(customerId, conn.id),
    onSuccess: () => toast("Checking it within a minute."),
    onError: (e) => onError(errorMessage(e, "Couldn't start a check.")),
  });
  const remove = useMutation({
    mutationFn: () => api.disconnectApp(customerId, conn.id),
    onSuccess: () => { setConfirm(false); toast(`${app.name} disconnected. Memory made from it stays.`); refresh(); },
    onError: (e) => onError(errorMessage(e, "Couldn't disconnect.")),
  });
  const busy = update.isPending || remove.isPending;
  const u = conn.uses;
  return (
    <li className="rounded-lg border border-line bg-inset px-3 py-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="min-w-0 truncate text-[13.5px] font-medium text-fg">{conn.account_label || app.name}</span>
        {statusPill(conn)}
        <span className="text-[12px] text-fg-3">{conn.last_synced_at ? `checked ${fmtRelative(conn.last_synced_at)}` : "not checked yet"}</span>
        <span className="flex-1" />
        {conn.status === "needs_reauth" ? (
          <button type="button" className={BTN_PRIMARY} onClick={() => onReconnect(conn.id)}>Sign in again</button>
        ) : (
          <>
            <button type="button" disabled={busy || conn.status === "paused"} className={BTN_QUIET} onClick={() => sync.mutate()}>Check now</button>
            <button type="button" disabled={busy} className={BTN_QUIET} onClick={() => update.mutate({ paused: conn.status !== "paused" })}>
              {conn.status === "paused" ? "Resume" : "Pause"}
            </button>
          </>
        )}
        <button type="button" disabled={busy} className="text-[12.5px] text-fg-3 hover:text-bad disabled:opacity-50" onClick={() => setConfirm(true)}>Disconnect</button>
      </div>
      {conn.last_error && conn.status !== "needs_reauth" && <p className="mt-1 text-[12px] text-warn">{conn.last_error}</p>}
      <ConfirmInline open={confirm} busy={remove.isPending} confirmLabel="Disconnect"
        message={`Disconnect ${conn.account_label || app.name}? Crystal stops watching it and forgets its sign-in. Memory already made from it stays.`}
        onConfirm={() => remove.mutate()} onCancel={() => setConfirm(false)} />
      <div className="mt-3 grid gap-3 min-[720px]:grid-cols-3" role="group" aria-label={`What Crystal does with ${app.name}`}>
        <Switch checked={u.watch} disabled={busy} label="Watch"
          hint="Goals see what it shows. Kept 7 days, never memory."
          onChange={(next) => update.mutate({ uses: next ? { watch: true } : { watch: false, remember: false } })} />
        <Switch checked={u.remember} disabled={busy || !u.watch} label="Remember"
          hint={u.watch ? `${app.remember_hint} Only what your rules allow.` : "Needs Watch on."}
          onChange={(next) => update.mutate({ uses: { remember: next } })} />
        {app.acts.length > 0 ? (
          <Switch checked={u.act} disabled={busy} label="Act"
            hint={`Goals may ${app.acts.map((a) => a.label.toLowerCase()).join(", ")}. Each goal still needs your permission.`}
            onChange={(next) => update.mutate({ uses: { act: next } })} />
        ) : (
          <p className="text-[12px] text-fg-3">Crystal only reads {app.name}.</p>
        )}
      </div>
      {u.remember && <Rules conn={conn} busy={busy} onSave={(rules) => update.mutate({ rules })} />}
      <div className="mt-2">
        <button type="button" className={BTN_QUIET} aria-expanded={showItems} onClick={() => setShowItems((v) => !v)}>
          {showItems ? "Hide recent items" : "Show recent items"}
        </button>
        {showItems && <div className="mt-2"><RecentItems customerId={customerId} connectionId={conn.id} /></div>}
      </div>
    </li>
  );
}

/** Starting a sign-in (Connect) or a re-sign-in (Reconnect): mint the
 *  session on our server, open Nango's Connect UI, refresh the app list a
 *  few times after it reports a connection (the webhook binds it). Shared
 *  by Settings and the Command center's Add connection modal. */
export function useConnectApp(customerId: string, onError: (message: string | null) => void, onOpened?: () => void) {
  const qc = useQueryClient();
  const toast = useToast();
  const refreshSoon = () => {
    for (const ms of [1500, 5000, 12000]) setTimeout(() => qc.invalidateQueries({ queryKey: ["apps", customerId] }), ms);
  };
  const open = (token: string, name: string) => {
    openConnectUI(token, (e) => {
      if (e.type === "connect") { toast(`${name} connected. It shows here in a moment.`); refreshSoon(); }
    });
  };
  const connect = useMutation({
    mutationFn: (app: AppCatalogEntry) => api.startAppSession(customerId, app.integration),
    onSuccess: (s, app) => { onError(null); onOpened?.(); open(s.token, app.name); },
    onError: (e) => onError(errorMessage(e, "Couldn't start connecting.")),
  });
  const reconnect = useMutation({
    mutationFn: (connectionId: string) => api.reconnectApp(customerId, connectionId),
    onSuccess: (s) => { onError(null); onOpened?.(); open(s.token, "The app"); },
    onError: (e) => onError(errorMessage(e, "Couldn't start signing in again.")),
  });
  return { connect, reconnect };
}

export function Apps({ customerId }: { customerId: string }) {
  const [error, setError] = useState<string | null>(null);
  const apps = useQuery({ queryKey: ["apps", customerId], queryFn: () => api.listApps(customerId) });
  const { connect, reconnect } = useConnectApp(customerId, setError);
  const data = apps.data;
  return (
    <Section title="Apps" help="Connect the apps you use. For each one, choose what Crystal does: watch it for your goals, remember what your rules allow, and act through it.">
      {error && <ErrorBanner title="That didn't go through" message={error} className="mb-3" />}
      {apps.isError && <ErrorBanner title="Couldn't load apps" error={apps.error} onRetry={() => apps.refetch()} />}
      {data && !data.configured && <p className="mb-3 text-[13px] text-fg-2">Connected apps aren't set up on this server yet.</p>}
      <ul className="space-y-3" aria-label="Apps">
        {(data?.apps ?? []).map((app) => (
          <li key={app.integration} className="rounded-xl border border-line p-3">
            <div className="flex flex-wrap items-start gap-2">
              <div className="min-w-0 flex-1">
                <p className="flex flex-wrap items-center gap-2 text-[14px] font-semibold text-fg">
                  {app.name}
                  {app.beta && <StatusPill variant="vio">Private beta</StatusPill>}
                </p>
                <p className="mt-0.5 text-[12.5px] text-fg-2">{app.blurb}</p>
                {app.note && <p className="mt-0.5 text-[12px] text-fg-3">{app.note}</p>}
              </div>
              {app.available ? (
                <button type="button" className={app.connections.length ? BTN_SECONDARY : BTN_PRIMARY}
                  disabled={!data?.configured || connect.isPending} onClick={() => connect.mutate(app)}>
                  {app.connections.length ? "Connect another" : "Connect"}
                </button>
              ) : (
                <span className="text-[12.5px] text-fg-3">Invite only</span>
              )}
            </div>
            {app.connections.length > 0 && (
              <ul className="mt-3 space-y-2" aria-label={`${app.name} connections`}>
                {app.connections.map((c) => (
                  <ConnectionRow key={c.id} customerId={customerId} app={app} conn={c} onError={setError}
                    onReconnect={(id) => reconnect.mutate(id)} />
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
    </Section>
  );
}
