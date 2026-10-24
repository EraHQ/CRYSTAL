// @vitest-environment jsdom
// Goals (Q109 A, 2026-10-24): the list says what each goal is doing and
// what waits; a new goal goes to the planner; the detail shows the plan,
// what the goal needs, drafts to approve (edited first if the person
// wants) and the history of what was sent.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Goal, GoalItem } from "@/lib/types";

const now = Date.now();
const goal: Goal = {
  id: "goal_1", title: "Inbox zero", directive: "Reply to every email within 3 hours.", status: "active",
  plan: {
    summary: "Every email gets a reply within 3 hours.", success_measure: "No email older than 3 hours unanswered.",
    check_every_minutes: 30, needs: { apps: ["google-mail"], acts: ["send_email"] }, standing_instructions: "Short, warm replies.",
    steps: [
      { n: 1, title: "Each check-in: triage new mail", detail: "Find mail that needs a reply.", done: false, note: "" },
      { n: 2, title: "Set up labels", detail: "", done: true, note: "Done on day one." },
    ],
  },
  plan_model: "frontier", planned_at: new Date(now - 86_400_000).toISOString(), next_run_at: new Date(now + 600_000).toISOString(),
  last_run_at: new Date(now - 1_200_000).toISOString(), last_task_id: "cog_1",
  last_summary: "Did: drafted a reply to Theo.\nWaiting: your approval.", may_send: false, may_post: false,
  created_at: null, updated_at: null, open: { action: 1, draft: 1, note: 1 },
};
const items: GoalItem[] = [
  { id: "gi_d", goal_id: "goal_1", kind: "draft", title: "Re: Contract question", body: "Hi Theo, confirmed.", watched_item_id: "wi_1",
    target: { app: "google-mail", action: "send_email", to: "theo@acme.test", subject: "Contract question" }, status: "open", due_at: null, created_at: null, updated_at: null },
  { id: "gi_a", goal_id: "goal_1", kind: "action", title: "Call Theo back", body: "", watched_item_id: null, target: {}, status: "open",
    due_at: new Date(now + 3_600_000).toISOString(), created_at: null, updated_at: null },
  { id: "gi_n", goal_id: "goal_1", kind: "note", title: "Turn on Act for Gmail so replies can go out.", body: "", watched_item_id: null, target: {},
    status: "open", due_at: null, created_at: null, updated_at: null },
  { id: "gi_s", goal_id: "goal_1", kind: "draft", title: "Re: Lunch", body: "Thursday works.", watched_item_id: null,
    target: { app: "google-mail", action: "send_email", to: "ana@x.test", subject: "Lunch", sent_by: "goal:goal_1" }, status: "sent",
    due_at: null, created_at: null, updated_at: new Date(now - 7_200_000).toISOString() },
];
const m = {
  listGoals: vi.fn(async () => ({ goals: [goal, { ...goal, id: "goal_2", title: "Weekly review", status: "active" as const, plan: null, planned_at: null, last_run_at: null, open: { action: 0, draft: 0, note: 0 } }], count: 2 })),
  getGoal: vi.fn(async () => ({ goal, items })),
  createGoal: vi.fn(async (_c: string, b: { title: string }) => ({ goal: { ...goal, id: "goal_3", title: b.title, plan: null, planned_at: null } })),
  updateGoal: vi.fn(async () => ({ goal })),
  replanGoal: vi.fn(async () => ({ goal })),
  runGoalNow: vi.fn(async () => ({ goal })),
  deleteGoal: vi.fn(async () => ({ deleted: true })),
  goalItemAction: vi.fn(async (_c: string, id: string) => ({ item: { ...items.find((i) => i.id === id)!, status: "sent" } })),
};
vi.mock("@/lib/api", () => ({
  api: new Proxy({}, { get: (_t, k: string) => (...a: unknown[]) => (m as Record<string, (...x: unknown[]) => unknown>)[k](...a) }),
  errorMessage: (e: unknown, fallback = "Something went wrong.") => (e instanceof Error ? e.message : fallback),
}));
vi.mock("@/lib/selected-customer", () => ({ useSelectedCustomer: () => ({ selectedCustomerId: "cus_1" }) }));

import { ToastProvider } from "@/components/ui";
import { GoalsPage, cadenceText, draftWhere, goalState } from "./Goals";

function mount() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}><ToastProvider><MemoryRouter><GoalsPage /></MemoryRouter></ToastProvider></QueryClientProvider>,
  );
}

afterEach(() => { cleanup(); vi.clearAllMocks(); });

describe("Goals (Q109 A)", () => {
  it("words: state, cadence and where a draft goes", () => {
    expect(goalState({ ...goal, planned_at: null }).label).toBe("Planning");
    expect(goalState(goal).label).toBe("Active");
    expect(cadenceText(30)).toBe("every 30 minutes");
    expect(cadenceText(180)).toBe("every 3 hours");
    expect(cadenceText(1440)).toBe("every 1 day");
    expect(cadenceText(10080)).toBe("every week");
    expect(draftWhere(items[0])).toBe("Gmail to theo@acme.test, “Contract question”");
  });

  it("lists goals with what waits, and a new goal goes to the planner", async () => {
    mount();
    await screen.findByText("Inbox zero");
    const list = screen.getByRole("list", { name: "Goals" });
    expect(within(list).getByText("Inbox zero")).toBeTruthy();
    expect(within(list).getByText("3 waiting")).toBeTruthy();
    expect(within(list).getByText("Planning")).toBeTruthy();
    expect(screen.getByText("2 active")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "New goal" }));
    const form = screen.getByRole("form", { name: "New goal" });
    fireEvent.change(within(form).getByLabelText("Name"), { target: { value: "Customers first" } });
    fireEvent.change(within(form).getByLabelText("The goal, in your words"), { target: { value: "Flag every customer email within an hour." } });
    fireEvent.click(within(form).getByRole("switch", { name: "May send email on its own" }));
    fireEvent.click(within(form).getByRole("button", { name: "Add goal" }));
    await waitFor(() => expect(m.createGoal).toHaveBeenCalledWith("cus_1", {
      title: "Customers first", directive: "Flag every customer email within an hour.", may_send: true, may_post: false }));
  });

  it("the detail shows the plan, what it needs, and approves an edited draft", async () => {
    mount();
    fireEvent.click(await screen.findByRole("button", { name: /Inbox zero/ }));
    await screen.findByTestId("draft-gi_d");
    expect(screen.getByText("Short, warm replies.", { exact: false })).toBeTruthy();
    expect(screen.getAllByText("Every email gets a reply within 3 hours.")).toHaveLength(2);
    expect(screen.getByText("Checks in every 30 minutes. Uses Gmail.")).toBeTruthy();
    const steps = screen.getByRole("list", { name: "Steps" });
    expect(within(steps).getAllByLabelText("Done")).toHaveLength(1);
    const needs = screen.getByRole("note", { name: "Needs you" });
    expect(within(needs).getByText("Turn on Act for Gmail so replies can go out.")).toBeTruthy();
    const draft = screen.getByTestId("draft-gi_d");
    expect(within(draft).getByText("Gmail to theo@acme.test, “Contract question”")).toBeTruthy();
    fireEvent.click(within(draft).getByRole("button", { name: "Edit" }));
    fireEvent.change(within(draft).getByLabelText("Draft text"), { target: { value: "Hi Theo, confirmed for Friday." } });
    fireEvent.click(within(draft).getByRole("button", { name: "Approve and send" }));
    await waitFor(() => expect(m.goalItemAction).toHaveBeenCalledWith("cus_1", "gi_d", { action: "send", body: "Hi Theo, confirmed for Friday." }));
    fireEvent.click(screen.getByRole("button", { name: "History (1)" }));
    const history = screen.getByRole("list", { name: "History" });
    expect(within(history).getByText("Sent")).toBeTruthy();
    expect(within(history).getByText(/sent by the goal/)).toBeTruthy();
    fireEvent.click(screen.getByRole("switch", { name: "May send email on its own" }));
    await waitFor(() => expect(m.updateGoal).toHaveBeenCalledWith("cus_1", "goal_1", { may_send: true }));
    fireEvent.click(screen.getByRole("button", { name: "Check in now" }));
    await waitFor(() => expect(m.runGoalNow).toHaveBeenCalled());
    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete goal" }));
    await waitFor(() => expect(m.deleteGoal).toHaveBeenCalledWith("cus_1", "goal_1"));
  });
});
