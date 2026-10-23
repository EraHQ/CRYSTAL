// @vitest-environment jsdom
// Activity → Steward (Q100 A, 2026-10-23): the switch state, the refill
// ask when paused, the last brief, and every action with its 24-hour Undo.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { StewardStatus } from "@/lib/types";

const now = Date.now();
const iso = (minutesAgo: number) => new Date(now - minutesAgo * 60_000).toISOString();
let status: StewardStatus;
const m = {
  stewardStatus: vi.fn(async () => status),
  undoStewardAction: vi.fn(async (_c: string, id: string) => {
    status = { ...status, actions: status.actions.map((a) => (a.id === id ? { ...a, undoable: false, undone_at: iso(0) } : a)) };
    return { undone: true, action: status.actions.find((a) => a.id === id)! };
  }),
};
vi.mock("@/lib/api", () => ({
  api: new Proxy({}, { get: (_t, k: string) => (...a: unknown[]) => (m as Record<string, (...x: unknown[]) => unknown>)[k](...a) }),
  errorMessage: (e: unknown, fallback = "Something went wrong.") => (e instanceof Error ? e.message : fallback),
}));

import { ToastProvider } from "@/components/ui";
import { ACTION_WORD, Steward, actionHref, pausedText } from "./Steward";

function mount() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={qc}><ToastProvider><MemoryRouter><Steward customerId="cus_1" /></MemoryRouter></ToastProvider></QueryClientProvider>);
}

beforeEach(() => {
  vi.clearAllMocks();
  status = {
    customer_id: "cus_1", enabled: true, platform_enabled: true,
    today: { sessions: 3, actions: 5, sessions_per_day: 12 },
    paused: { code: "daily_capacity", message: "Daily AI capacity is used up for this plan.", at: iso(20), resets_at: new Date(now + 6 * 3600_000).toISOString() },
    latest: { run_id: "cog_9", at: iso(25), brief: "Did: settled the Pro price conflict.\nLeft alone: the refund consequence.\nNext: nothing.", actions: 2, ok: true, status: "complete" },
    actions: [
      { id: "stw_1", run_id: "cog_9", action: "decide_conflict", subject_kind: "conflict", subject_id: "k1", label: "Steward settled a conflict - Pricing: retired the outdated claim", reason: "The $15 price is dated and newer.", inverse: { reopen_conflict: "k1" }, undoable: true, undone_at: null, undo_until: iso(-1400), created_at: iso(25) },
      { id: "stw_2", run_id: "cog_9", action: "approve_assumption", subject_kind: "assumption", subject_id: "asm_1", label: "Steward approved a consequence - Pro is billed every month", reason: "The invoice fact agrees.", inverse: {}, undoable: false, undone_at: iso(5), undo_until: iso(-1400), created_at: iso(26) },
      { id: "stw_3", run_id: "cog_8", action: "reprobe_fact", subject_kind: "fact", subject_id: "f1", label: "Steward asked for a re-probe - Pro is $15", reason: "New facts landed.", inverse: {}, undoable: false, undone_at: null, undo_until: iso(-1000), created_at: iso(300) },
      { id: "stw_4", run_id: "cog_1", action: "dismiss_gap", subject_kind: "gap", subject_id: "g1", label: "Steward dismissed a gap - office coffee", reason: "Trivia.", inverse: { reopen_gap: "g1" }, undoable: false, undone_at: null, undo_until: iso(60), created_at: iso(1500) },
    ],
  };
});
afterEach(cleanup);

describe("Steward", () => {
  it("shows the state, the refill ask, the brief, and undoes an action", async () => {
    mount();
    expect(await screen.findByText("Crystal runs itself")).toBeTruthy();
    expect(screen.getByText("Today: 3 of 12 sessions, 5 actions")).toBeTruthy();
    expect(screen.getByRole("status").textContent).toContain("Today's AI capacity is used. Upgrade your plan, raise the cap, or wait: it resumes");
    const last = screen.getByRole("region", { name: "Last session" });
    expect(within(last).getByText(/Last session · .* · 2 actions/)).toBeTruthy();
    expect(within(last).getByText(/Did: settled the Pro price conflict/)).toBeTruthy();
    expect(within(last).getByRole("link", { name: "Open the run" }).getAttribute("href")).toBe("/activity/run/cog_9");
    const list = screen.getByRole("region", { name: "Actions" });
    const rows = within(list).getAllByRole("listitem");
    expect(rows).toHaveLength(4);
    expect(within(rows[0]).getByText("Settled a conflict")).toBeTruthy();
    expect(within(rows[0]).getByText("The $15 price is dated and newer.")).toBeTruthy();
    expect(within(rows[0]).getByRole("link", { name: "Open" }).getAttribute("href")).toBe("/conflicts");
    expect(within(rows[1]).getByText(/Undone/)).toBeTruthy();
    expect(within(rows[1]).queryByRole("button", { name: "Undo" })).toBeNull();
    expect(within(rows[2]).getByText("Nothing to undo")).toBeTruthy();
    expect(within(rows[3]).getByText("Undo window closed")).toBeTruthy();
    fireEvent.click(within(rows[0]).getByRole("button", { name: "Undo" }));
    await waitFor(() => expect(m.undoStewardAction).toHaveBeenCalledWith("cus_1", "stw_1"));
    await waitFor(() => expect(within(screen.getByRole("region", { name: "Actions" })).getAllByText(/Undone/)).toHaveLength(2));
    expect(actionHref(status.actions[1])).toBe("/assumptions/asm_1");
    expect(actionHref(status.actions[2])).toBeNull();
    expect(ACTION_WORD.research_gap).toBe("Queued research");
    expect(pausedText({ code: "session_cap", message: null, at: iso(1), resets_at: iso(-60) })).toBe("Today's sessions are used. It resumes tomorrow.");
  });

  it("explains itself when off and empty", async () => {
    status = { ...status, enabled: false, paused: null, latest: null, actions: [] };
    mount();
    expect(await screen.findByText("Off. A person curates.")).toBeTruthy();
    expect(screen.getByText("The Steward is off")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Switch it on in Settings" }).getAttribute("href")).toBe("/settings/curation");
    expect(screen.queryByRole("status")).toBeNull();
  });
});
