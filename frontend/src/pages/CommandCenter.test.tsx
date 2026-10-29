// @vitest-environment jsdom
// Command center (2026-10-24): goals as cards that link into the Goals
// page, what waits across every goal, connections as a small strip, and
// the two modals: Add goal (the Goals page's form) and Add connection
// (the app list; picking one opens the Connect UI and closes the modal).
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AppsResponse, Goal, Pulse } from "@/lib/types";

// recharts' ResponsiveContainer measures with ResizeObserver, which jsdom lacks.
class RO { observe() {} unobserve() {} disconnect() {} }
(globalThis as unknown as { ResizeObserver: typeof RO }).ResizeObserver ??= RO;

const goal: Goal = {
  id: "goal_1", title: "Inbox zero", directive: "Reply within 3 hours.", status: "active",
  plan: { summary: "Every email gets a reply within 3 hours.", success_measure: "", check_every_minutes: 30, needs: { apps: [], acts: [] },
    standing_instructions: "", steps: [{ n: 1, title: "a", detail: "", done: true, note: "" }, { n: 2, title: "b", detail: "", done: false, note: "" }] },
  plan_model: null, planned_at: "2026-10-24T00:00:00Z", next_run_at: null, last_run_at: null, last_task_id: null, last_summary: "",
  may_send: false, may_post: false, created_at: null, updated_at: null, open: { action: 2, draft: 1, note: 0 },
};
const apps: AppsResponse = {
  configured: true, features: {},
  apps: [
    { integration: "slack", name: "Slack", blurb: "Messages.", kind: "message", acts: [], beta: null, available: true, note: "", remember_hint: "", group: "Chat",
      connections: [{ id: "ac_1", integration: "slack", account_label: "Kerning", status: "needs_reauth", uses: { watch: true, remember: false, act: true },
        remember_rules: [], rules_text: [], watch_cadence_minutes: 15, last_synced_at: null, last_error: null, created_at: null }] },
    { integration: "notion", name: "Notion", blurb: "Pages you share.", kind: "page", acts: [], beta: null, available: true, note: "", remember_hint: "", group: "Docs and knowledge", connections: [] },
    { integration: "google-mail", name: "Gmail", blurb: "Mail.", kind: "email", acts: [], beta: "gmail", available: false, note: "", remember_hint: "", group: "Email and calendar", connections: [] },
  ],
};
const days = (n: number) => Array.from({ length: n }, (_, i) => new Date(Date.UTC(2026, 9, 24 - (n - 1 - i))).toISOString().slice(0, 10));
const pulse: Pulse = {
  as_of: "2026-10-24T16:00:00Z",
  growth: { days: days(30).map((d, i) => ({ day: d, facts: 1000 + i * 10 })), total: 1290, week_added: 70 },
  recall: { days: days(14).map((d) => ({ day: d, asked: 4, found: 3 })), asked: 56, found: 42, found_pct: 75 },
  autonomy: { days: days(14).map((d) => ({ day: d, conflicts: 1, assumptions: 3, gaps: 0, facts: 0, goals: 1 })),
    kinds: ["conflicts", "assumptions", "gaps", "facts", "goals"], week: 28 },
  health: { tiers: { whitelist: 900, neutral: 300, quarantine: 90 }, open_conflicts: 3, open_gaps: 7 },
  goals: { week: { sent: 4, done: 2, dismissed: 1, drafted: 6, filed: 3 }, replies: 6, median_to_draft_minutes: 42, median_to_sent_minutes: 130 },
  capacity: { pct: 37 },
  heatmap: days(84).map((d, i) => ({ day: d, writes: i % 3, recalls: i % 5, acts: i % 2 })),
};
const m = {
  listGoals: vi.fn(async () => ({ goals: [goal], count: 1 })),
  listApps: vi.fn(async () => apps),
  createGoal: vi.fn(async () => ({ goal })),
  startAppSession: vi.fn(async () => ({ token: "nst_1", expires_at: null, integration: "notion" })),
  getPulse: vi.fn(async () => pulse),
};
vi.mock("@/lib/api", () => ({
  api: new Proxy({}, { get: (_t, k: string) => (...a: unknown[]) => (m as Record<string, (...x: unknown[]) => unknown>)[k](...a) }),
  errorMessage: (e: unknown, fallback = "Something went wrong.") => (e instanceof Error ? e.message : fallback),
}));
vi.mock("@/lib/selected-customer", () => ({ useSelectedCustomer: () => ({ selectedCustomerId: "cus_1" }) }));

import { ToastProvider } from "@/components/ui";
import { CommandCenterPage, waitingTotals } from "./CommandCenter";

function mount() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={qc}><ToastProvider><MemoryRouter><CommandCenterPage /></MemoryRouter></ToastProvider></QueryClientProvider>);
}

afterEach(() => { cleanup(); vi.clearAllMocks(); document.getElementById("connect-ui")?.remove(); });

describe("Command center", () => {
  it("adds up what waits across goals", () => {
    expect(waitingTotals([goal, { ...goal, open: { action: 1, draft: 0, note: 2 } }, { ...goal, status: "needs_setup", open: undefined }]))
      .toEqual({ drafts: 1, actions: 3, needs: 2 });
  });

  it("shows goals as cards that link to the goal, what waits, and a small connection strip", async () => {
    mount();
    const card = (await screen.findByText("Inbox zero")).closest("a")!;
    expect(card.getAttribute("href")).toBe("/goals/goal_1");
    expect(within(card).getByLabelText("1 of 2 steps done")).toBeTruthy();
    const waiting = screen.getByRole("region", { name: "Waiting for you" });
    expect(within(waiting).getByText("1 draft to approve")).toBeTruthy();
    expect(within(waiting).getByText("2 action items")).toBeTruthy();
    expect(await within(waiting).findByText("1 app to sign in again")).toBeTruthy();
    const strip = screen.getByRole("list", { name: "Active connections" });
    expect(within(strip).getByText("Slack")).toBeTruthy();
    expect(within(strip).getByText("Watch · Act")).toBeTruthy();
  });

  it("the System section shows the pulse: memory, recall, autonomy, health, goals, capacity and 84 heatmap days", async () => {
    mount();
    const system = await screen.findByRole("region", { name: "System" });
    expect(await within(system).findByText("1,290")).toBeTruthy();
    expect(within(system).getByText("+70 this week", { exact: false })).toBeTruthy();
    expect(within(system).getByText("75%")).toBeTruthy();
    const auto = within(system).getByRole("region", { name: "Crystal on its own, 14 days" });
    expect(within(auto).getByText("28")).toBeTruthy();
    expect(within(auto).getByText("Conflicts settled")).toBeTruthy();
    fireEvent.click(within(auto).getByRole("button", { name: "Show numbers" }));
    expect(within(auto).getAllByRole("row")).toHaveLength(15);
    expect(within(system).getByText("70%")).toBeTruthy(); // 900 of 1,290 trusted
    expect(within(system).getByText("42 min")).toBeTruthy();
    expect(within(system).getByText("Sent replies went out in a median 2.2 h.")).toBeTruthy();
    expect(within(system).getByRole("progressbar", { name: "AI capacity used today" }).getAttribute("aria-valuenow")).toBe("37");
    expect(within(system).getAllByRole("gridcell")).toHaveLength(84);
  });

  it("Add goal opens the goal form in a modal", async () => {
    mount();
    fireEvent.click(await screen.findByRole("button", { name: "Add goal" }));
    const dialog = screen.getByRole("dialog");
    fireEvent.change(within(dialog).getByLabelText("Name"), { target: { value: "Weekly review" } });
    fireEvent.change(within(dialog).getByLabelText("The goal, in your words"), { target: { value: "Every Friday, summarize my week." } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Add goal" }));
    await waitFor(() => expect(m.createGoal).toHaveBeenCalledWith("cus_1", { title: "Weekly review", directive: "Every Friday, summarize my week.", may_send: false, may_post: false }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });

  it("Add connection lists the apps; picking one opens the Connect UI and closes the modal", async () => {
    mount();
    fireEvent.click(await screen.findByRole("button", { name: "Add connection" }));
    const dialog = screen.getByRole("dialog");
    const list = await within(dialog).findByRole("list", { name: "Apps to connect" });
    expect(within(list).getByText("Chat")).toBeTruthy(); // grouped
    expect(within(list).getByRole("list", { name: "Docs and knowledge" })).toBeTruthy();
    const gmail = within(list).getByText("Gmail").closest("button")!;
    expect(gmail.hasAttribute("disabled")).toBe(true);
    expect(within(list).getByText("Invite only. Ask for an invite.")).toBeTruthy();
    fireEvent.click(within(list).getByText("Notion").closest("button")!);
    await waitFor(() => expect(m.startAppSession).toHaveBeenCalledWith("cus_1", "notion"));
    await waitFor(() => expect(document.getElementById("connect-ui")).toBeTruthy());
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
