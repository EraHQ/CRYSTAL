// Command center (2026-10-24). Anthony: "Lets create a page called
// command center. That's where you can click 'add connection' and a modal
// will pop up with the option. You can also add goals from the command
// center. The default command center will be your goals, active
// connections (not too big), and ... stats and graphs ... for the system
// as a whole." Q110 B: the stats are six panels and a 12-week heatmap
// (components/command/Pulse.tsx). Q111 A: this is home ("/"). Goals and connections are read from the same routes as
// the Goals page and Settings › Connections; the modals reuse their
// forms (NewGoal, useConnectApp) so the three surfaces cannot drift.
import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Gauge, Plus, Target } from "lucide-react";
import { api } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import type { AppCatalogEntry, AppConnection, Goal } from "@/lib/types";
import { EmptyState, ErrorBanner, Modal, PageHeader, StatusPill } from "@/components/ui";
import { cn, fmtRelative } from "@/lib/utils";
import { NewGoal, cadenceText, goalState } from "./Goals";
import { useConnectApp } from "./settings/Apps";
import { Pulse } from "@/components/command/Pulse";
import { BTN_PRIMARY, BTN_QUIET, BTN_SECONDARY } from "./settings/shared";

/** Open items across every goal, by what they ask of the person. */
export function waitingTotals(goals: Goal[]): { drafts: number; actions: number; needs: number } {
  return goals.reduce((t, g) => ({
    drafts: t.drafts + (g.open?.draft ?? 0), actions: t.actions + (g.open?.action ?? 0),
    // A goal held for setup needs you even with nothing filed yet.
    needs: t.needs + ((g.open?.note ?? 0) > 0 || g.status === "needs_setup" || g.status === "deciding" ? 1 : 0),
  }), { drafts: 0, actions: 0, needs: 0 });
}

/** Apps grouped for the Add connection list, groups in first-seen order. */
export function groupApps(apps: AppCatalogEntry[]): [string, AppCatalogEntry[]][] {
  const out = new Map<string, AppCatalogEntry[]>();
  for (const a of apps) {
    const g = a.group || "Other";
    out.set(g, [...(out.get(g) ?? []), a]);
  }
  return [...out.entries()];
}

function AddConnectionModal({ customerId, open, onClose }: { customerId: string; open: boolean; onClose: () => void }) {
  const [error, setError] = useState<string | null>(null);
  const apps = useQuery({ queryKey: ["apps", customerId], queryFn: () => api.listApps(customerId), enabled: open });
  const { connect } = useConnectApp(customerId, setError, onClose);
  const data = apps.data;
  return (
    <Modal open={open} onClose={onClose} title="Add a connection" className="max-w-[560px]">
      {error && <ErrorBanner title="That didn't go through" message={error} className="mb-3" />}
      {apps.isError && <ErrorBanner title="Couldn't load apps" error={apps.error} onRetry={() => apps.refetch()} />}
      {data && !data.configured && <p className="mb-3 text-[13px] text-fg-2">Connected apps aren't set up on this server yet.</p>}
      <p className="mb-3 text-[13px] text-fg-2">Pick an app. You sign in with it directly; after that you choose what Crystal does with it.</p>
      {data && data.configured && data.apps.length === 0 && (
        <p className="text-[13px] text-fg-3">No apps are set up yet. Each one appears here once it is configured in Nango.</p>
      )}
      <div className="max-h-[60vh] space-y-3 overflow-y-auto pr-1" aria-label="Apps to connect" role="list">
        {groupApps(data?.apps ?? []).map(([group, apps]) => (
          <div key={group} role="listitem">
            <p className="mb-1.5 text-[11px] font-medium uppercase tracking-[0.12em] text-fg-3">{group}</p>
            <ul className="grid gap-2 min-[520px]:grid-cols-2" aria-label={group}>
              {apps.map((app: AppCatalogEntry) => {
                const usable = app.available && !!data?.configured;
                return (
                  <li key={app.integration}>
                    <button type="button" disabled={!usable || connect.isPending} onClick={() => connect.mutate(app)}
                      className={cn("flex h-full w-full flex-col items-start rounded-lg border px-3 py-2.5 text-left",
                        usable ? "border-line bg-inset hover:border-accent" : "cursor-not-allowed border-line opacity-60")}>
                      <span className="flex w-full items-center gap-2">
                        <span className="text-[13.5px] font-medium text-fg">{app.name}</span>
                        <span className="flex-1" />
                        {app.beta && <StatusPill variant="vio">Beta</StatusPill>}
                        {app.connections.length > 0 && <StatusPill variant="ok">{app.connections.length} connected</StatusPill>}
                      </span>
                      <span className="mt-1 text-[12px] text-fg-2">
                        {app.available ? app.blurb : app.beta ? "Invite only. Ask for an invite." : "No longer offered here; manage it under Settings."}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </div>
    </Modal>
  );
}

function GoalCard({ goal }: { goal: Goal }) {
  const st = goalState(goal);
  const waiting = (goal.open?.draft ?? 0) + (goal.open?.action ?? 0) + (goal.open?.note ?? 0);
  const steps = goal.plan?.steps ?? [];
  const done = steps.filter((s) => s.done).length;
  return (
    <li>
      <Link to={`/goals?goal=${encodeURIComponent(goal.id)}`}
        className="block h-full rounded-xl border border-line bg-surface p-4 hover:border-accent">
        <div className="flex flex-wrap items-center gap-2">
          <span className="min-w-0 flex-1 truncate text-[14.5px] font-semibold text-fg">{goal.title}</span>
          {waiting > 0 && <StatusPill variant="warn">{waiting} waiting</StatusPill>}
          <StatusPill variant={st.variant} dot>{st.label}</StatusPill>
        </div>
        <p className="mt-1 line-clamp-2 text-[12.5px] text-fg-2">{goal.plan?.summary ?? goal.directive}</p>
        {steps.length > 0 && (
          <div className="mt-2" aria-label={`${done} of ${steps.length} steps done`}>
            <div className="h-1.5 overflow-hidden rounded-full bg-inset">
              <div className="h-full rounded-full bg-accent" style={{ width: `${Math.round((done / steps.length) * 100)}%` }} />
            </div>
          </div>
        )}
        <p className="mt-2 text-[11.5px] text-fg-3">
          {goal.last_run_at ? `Checked in ${fmtRelative(goal.last_run_at)}` : "Not checked in yet"}
          {goal.plan ? ` · ${cadenceText(goal.plan.check_every_minutes)}` : ""}
        </p>
      </Link>
    </li>
  );
}

function ConnectionChip({ app, conn }: { app: AppCatalogEntry; conn: AppConnection }) {
  const tone = conn.status === "needs_reauth" ? "bg-bad" : conn.status === "paused" ? "bg-fg-3" : "bg-ok";
  const uses = (["watch", "remember", "act"] as const).filter((u) => conn.uses[u]);
  return (
    <li className="flex min-w-0 items-center gap-2 rounded-lg border border-line bg-inset px-2.5 py-1.5"
      title={conn.status === "needs_reauth" ? "Sign in again under Settings › Connections" : undefined}>
      <span aria-hidden className={cn("h-2 w-2 shrink-0 rounded-full", tone)} />
      <span className="min-w-0 truncate text-[12.5px] font-medium text-fg">{app.name}</span>
      {conn.account_label && <span className="hidden min-w-0 truncate text-[11.5px] text-fg-3 min-[720px]:inline">{conn.account_label}</span>}
      <span className="text-[11px] text-fg-3">{uses.map((u) => u[0].toUpperCase() + u.slice(1)).join(" · ") || "Off"}</span>
      <span className="sr-only">{conn.status === "needs_reauth" ? "needs sign-in" : conn.status}</span>
    </li>
  );
}

export function CommandCenterPage() {
  const { selectedCustomerId } = useSelectedCustomer();
  const [addGoal, setAddGoal] = useState(false);
  const [addConnection, setAddConnection] = useState(false);
  const goals = useQuery({ queryKey: ["goals", selectedCustomerId], queryFn: () => api.listGoals(selectedCustomerId!),
    enabled: !!selectedCustomerId, refetchInterval: 60_000 });
  const apps = useQuery({ queryKey: ["apps", selectedCustomerId], queryFn: () => api.listApps(selectedCustomerId!),
    enabled: !!selectedCustomerId });
  if (!selectedCustomerId) {
    return (
      <>
        <PageHeader title="Command center" />
        <EmptyState icon={Gauge} title="No customer selected" help="Pick a customer above to see their command center." />
      </>
    );
  }
  const list = (goals.data?.goals ?? []).filter((g) => g.status !== "archived");
  const live = list.filter((g) => g.status === "active");
  const t = waitingTotals(list);
  const conns = (apps.data?.apps ?? []).flatMap((a) => a.connections.map((c) => ({ app: a, conn: c })));
  const needSignIn = conns.filter((x) => x.conn.status === "needs_reauth").length;
  return (
    <div className="space-y-6">
      <PageHeader title="Command center"
        meta={goals.isLoading ? "Loading…" : `${live.length} active goal${live.length === 1 ? "" : "s"} · ${conns.length} connection${conns.length === 1 ? "" : "s"}`}
        actions={<>
          <button type="button" className={BTN_SECONDARY} onClick={() => setAddConnection(true)}><Plus aria-hidden className="h-4 w-4" />Add connection</button>
          <button type="button" className={BTN_PRIMARY} onClick={() => setAddGoal(true)}><Plus aria-hidden className="h-4 w-4" />Add goal</button>
        </>} />

      {(t.drafts + t.actions + t.needs > 0 || needSignIn > 0) && (
        <section aria-label="Waiting for you" className="flex flex-wrap gap-2">
          {t.drafts > 0 && <StatusPill variant="warn">{t.drafts} draft{t.drafts === 1 ? "" : "s"} to approve</StatusPill>}
          {t.actions > 0 && <StatusPill variant="info">{t.actions} action item{t.actions === 1 ? "" : "s"}</StatusPill>}
          {t.needs > 0 && <StatusPill variant="warn">{t.needs} goal{t.needs === 1 ? " needs" : "s need"} you</StatusPill>}
          {needSignIn > 0 && <StatusPill variant="bad">{needSignIn} app{needSignIn === 1 ? "" : "s"} to sign in again</StatusPill>}
        </section>
      )}

      <section aria-label="Your goals">
        <div className="mb-2 flex items-center justify-between gap-2">
          <h2 className="text-[15px] font-semibold text-fg">Your goals</h2>
          {list.length > 0 && <Link to="/goals" className={BTN_QUIET}>All goals</Link>}
        </div>
        {goals.isError && <ErrorBanner title="Couldn't load goals" error={goals.error} onRetry={() => goals.refetch()} />}
        {!goals.isLoading && !goals.isError && list.length === 0 ? (
          <EmptyState icon={Target} title="No goals yet"
            help="A goal is a standing instruction, like “reply to every email within 3 hours”. Crystal plans it and works it." />
        ) : (
          <ul className="grid gap-3 min-[720px]:grid-cols-2 min-[1100px]:grid-cols-3" aria-label="Goals">
            {list.map((g) => <GoalCard key={g.id} goal={g} />)}
          </ul>
        )}
      </section>

      <section aria-label="Connections">
        <div className="mb-2 flex items-center justify-between gap-2">
          <h2 className="text-[15px] font-semibold text-fg">Connections</h2>
          <Link to="/settings/connections" className={BTN_QUIET}>Manage</Link>
        </div>
        {apps.isError && <ErrorBanner title="Couldn't load connections" error={apps.error} onRetry={() => apps.refetch()} />}
        {!apps.isLoading && !apps.isError && conns.length === 0 ? (
          <p className="text-[13px] text-fg-3">No apps connected. <button type="button" className="text-accent hover:underline" onClick={() => setAddConnection(true)}>Add one</button> so goals can see your mail, messages and calendar.</p>
        ) : (
          <ul className="flex flex-wrap gap-2" aria-label="Active connections">
            {conns.map(({ app, conn }) => <ConnectionChip key={conn.id} app={app} conn={conn} />)}
          </ul>
        )}
      </section>

      <section aria-label="System">
        <h2 className="mb-2 text-[15px] font-semibold text-fg">System</h2>
        <Pulse customerId={selectedCustomerId} />
      </section>

      <Modal open={addGoal} onClose={() => setAddGoal(false)} title="Add a goal" className="max-w-[560px]">
        <NewGoal customerId={selectedCustomerId} onDone={() => setAddGoal(false)} bare />
      </Modal>
      <AddConnectionModal customerId={selectedCustomerId} open={addConnection} onClose={() => setAddConnection(false)} />
    </div>
  );
}
