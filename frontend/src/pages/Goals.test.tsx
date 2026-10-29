// @vitest-environment jsdom
// Goals (Q109 A, 2026-10-24; the goal page, Series 382, 2026-10-27): the
// list says where each goal stands and what waits; a new goal goes to the
// planner; each goal has its own page with its steps' criteria measured
// live, its check-ins, a line to the next check-in, editing when a step is
// done, going back to a step, drafts to approve and what was sent.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, onTestFinished, vi } from "vitest";
import type { Goal, GoalItem, GoalProgressStep, GoalRun } from "@/lib/types";

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
const sports: Goal = {
  ...goal, id: "goal_s", title: "Sports Predictions", directive: "Catalog every NBA game of three seasons.", open: { action: 0, draft: 0, note: 0 },
  plan: { ...goal.plan!, check_every_minutes: 120, needs: { apps: [], acts: [] },
    lattices: [{ name: "nba_games", lattice_id: "lat_g", columns: [{ key: "game_id" }, { key: "home_score" }] }],
    survey: { findings: [
      { question: "Games per regular season", answer: "82 games a team.", number: 1230, source: "https://www.basketball-reference.com/x", measured: true },
      { question: "Distinct players", answer: "about 1,600", number: null, source: "", measured: false }] },
    steps: [
      { n: 1, title: "Import the schedules", detail: "", done: true, note: "", accepted_at: new Date(now - 3_600_000).toISOString(),
        criteria: [{ kind: "rows", label: "Every game is listed", lattice: "nba_games", min: 3000 }] },
      { n: 2, title: "Box scores", detail: "One box score page per game.", done: false, note: "", state: "rejected", attempts: 2,
        criteria: [{ kind: "coverage", label: "Every game has its final score", lattice: "nba_games", columns: ["home_score"], min_fraction: 0.98, basis: "measured", basis_detail: "1,230 a season" },
                   { kind: "rubric", label: "The totals add up to each final score", text: "The totals add up to each final score" }] },
      { n: 3, title: "Each check-in: new games", detail: "", done: false, note: "", repeats: true },
    ] },
};
const progress: GoalProgressStep[] = [
  { n: 1, title: "Import the schedules", done: true, state: "accepted", repeats: false, results: [], last_check: null },
  { n: 2, title: "Box scores", done: false, state: "rejected", repeats: false, last_check: null, results: [
    { kind: "coverage", label: "Every game has its final score", passed: false, value: 0.41, target: 0.98, measured: "1,512 of 3,690 filled",
      gap: "2,178 of 3,690 rows lack home_score.", missing_sample: ["202212010BOS", "202212010CHI"], basis: "measured", basis_detail: "1,230 a season" },
    { kind: "rubric", label: "The totals add up to each final score", passed: null, gap: "" }] },
  { n: 3, title: "Each check-in: new games", done: false, state: null, repeats: true, results: [], last_check: null },
];
const runs: GoalRun[] = [
  { run_id: "r3", phase: "run", status: "complete", started_at: new Date(now - 600_000).toISOString(), completed_at: null, seconds: 372, error: null,
    summary: "Did: imported 62 box scores.", model: "m", stop_reason: "max_iterations", calls: 40, tools: { web_tables: 16, lattice_import: 14 },
    actions: 14, rows: { inserted: 1240, updated: 0 }, submissions: [{ n: 2, verdict: "rejected" }], budget: { tool_calls: 40, actions: 30 },
    comments: [{ id: "c1", body: "Skip the preseason." }] },
  { run_id: "r2", phase: "run", status: "complete", started_at: new Date(now - 7_200_000).toISOString(), completed_at: null, seconds: 350, error: null,
    summary: "Did: step 1.", model: "m", stop_reason: "end_turn", calls: 40, tools: {}, actions: 9, rows: { inserted: 2980, updated: 0 },
    submissions: [{ n: 1, verdict: "accepted" }], budget: null, comments: [] },
  { run_id: "r1", phase: "plan", status: "complete", started_at: new Date(now - 9_000_000).toISOString(), completed_at: null, seconds: 90, error: null,
    summary: "", model: "m", stop_reason: null, steps: 3, lattices: 1, survey: 2 },
];
const m = {
  listGoals: vi.fn(async () => ({ goals: [goal, { ...goal, id: "goal_2", title: "Weekly review", status: "active" as const, plan: null, planned_at: null, last_run_at: null, open: { action: 0, draft: 0, note: 0 } }, sports], count: 3 })),
  getGoal: vi.fn(async (_c: string, id: string) => (id === "goal_s" ? { goal: sports, items: [] } : { goal, items })),
  createGoal: vi.fn(async (_c: string, b: { title: string }) => ({ goal: { ...goal, id: "goal_3", title: b.title, plan: null, planned_at: null } })),
  updateGoal: vi.fn(async () => ({ goal })),
  replanGoal: vi.fn(async () => ({ goal })),
  runGoalNow: vi.fn(async () => ({ goal })),
  runGoalAnyway: vi.fn(async () => ({ goal })),
  decideGoal: vi.fn(async () => ({ goal })),
  deleteGoal: vi.fn(async () => ({ deleted: true })),
  goalItemAction: vi.fn(async (_c: string, id: string) => ({ item: { ...items.find((i) => i.id === id)!, status: "sent" } })),
  goalRuns: vi.fn(async (_c: string, id: string) => ({ runs: id === "goal_s" ? runs : [], count: 0 })),
  goalProgress: vi.fn(async (_c: string, id: string) => ({ steps: id === "goal_s" ? progress : [], snapshots: [] })),
  goalComments: vi.fn(async () => ({ comments: [
    { id: "c1", goal_id: "goal_s", body: "Skip the preseason.", after_run_id: "r2", seen_by_run_id: "r3", seen_at: null,
      created_at: new Date(now - 1_800_000).toISOString(), status: "done", always: false, report: "Started at November.", reported_by_run_id: "r3" },
    { id: "c2", goal_id: "goal_s", body: "If a site won't load, find another source.", after_run_id: "r3", seen_by_run_id: "r3", seen_at: null,
      created_at: new Date(now - 900_000).toISOString(), status: "cant", always: false, report: "Tried sports-reference.com; blocked too.", reported_by_run_id: "r3" },
  ], count: 2 })),
  setGoalNote: vi.fn(async () => ({ comment: {} })),
  addGoalComment: vi.fn(async () => ({ comment: {} })),
  setStepCriteria: vi.fn(async () => ({ goal: sports })),
  reviewGoalStep: vi.fn(async () => ({ goal: sports })),
  previewGoalRollback: vi.fn(async () => ({ preview: true, to_step: 2, returned_to: "", undo: 3, undo_labels: ["a", "b", "c"], not_undone: [],
    dismiss: 2, dismiss_labels: ["d1", "d2"], sent_items: ["Recap"] })),
  rollbackGoal: vi.fn(async () => ({ undone: 3, dismissed_items: 2, sent_items: ["Recap"] })),
  listLattices: vi.fn(async () => ({ lattices: [{ lattice_id: "lat_g", name: "nba_games", rows: 3690, columns: [], key_columns: [], description: "", updated_at: null }], count: 1, rows: { used: 3690, cap: null } })),
};
vi.mock("@/lib/api", () => ({
  api: new Proxy({}, { get: (_t, k: string) => (...a: unknown[]) => (m as Record<string, (...x: unknown[]) => unknown>)[k](...a) }),
  errorMessage: (e: unknown, fallback = "Something went wrong.") => (e instanceof Error ? e.message : fallback),
}));
vi.mock("@/lib/selected-customer", () => ({ useSelectedCustomer: () => ({ selectedCustomerId: "cus_1" }) }));

import { ToastProvider } from "@/components/ui";
import { GoalsPage, cadenceText, draftWhere, goalState, whereItStands } from "./Goals";
import { GoalRoute, checkedBy, stopText } from "./GoalPage";

function mount(at = "/goals") {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}><ToastProvider><MemoryRouter initialEntries={[at]}>
      <Routes><Route path="/goals" element={<GoalsPage />} /><Route path="/goals/:goalId" element={<GoalRoute />} /></Routes>
    </MemoryRouter></ToastProvider></QueryClientProvider>,
  );
}

afterEach(() => { cleanup(); vi.clearAllMocks(); });

describe("Goals (Q109 A)", () => {
  it("words: state, cadence, where a draft goes, where a goal stands", () => {
    expect(goalState({ ...goal, planned_at: null }).label).toBe("Planning");
    expect(goalState(goal).label).toBe("Active");
    expect(cadenceText(30)).toBe("every 30 minutes");
    expect(cadenceText(180)).toBe("every 3 hours");
    expect(cadenceText(1440)).toBe("every 1 day");
    expect(cadenceText(10080)).toBe("every week");
    expect(draftWhere(items[0])).toBe("Gmail to theo@acme.test, “Contract question”");
    expect(draftWhere({ ...items[0], title: "Sports catalog", target: { app: "google-sheet", action: "create_spreadsheet" } }))
      .toBe("Google Sheets: a new spreadsheet, “Sports catalog”");
    expect(draftWhere({ ...items[0], target: { app: "google-sheet", action: "append_rows", sheet: "Games", spreadsheet_id: "s1" } }))
      .toBe("Google Sheets: rows into “Games” in spreadsheet s1");
    expect(whereItStands(sports).line).toBe("Step 2 of 2 rejected · Box scores");
    expect(whereItStands({ ...sports, plan: { ...sports.plan!, steps: sports.plan!.steps.map((s) => ({ ...s, done: true })) } }).line)
      .toBe("Repeating · Each check-in: new games");
    expect(checkedBy("coverage").label).toBe("Counted");
    expect(checkedBy("rubric").label).toBe("Reviewer");
    expect(checkedBy("person").label).toBe("You");
    expect(stopText("max_iterations")).toBe("Stopped: tool-call budget");
    expect(stopText("end_turn")).toBe("Finished");
  });

  it("lists goals with where they stand and what waits; a new goal goes to the planner", async () => {
    mount();
    await screen.findByText("Inbox zero");
    const list = screen.getByRole("list", { name: "Goals" });
    expect(within(list).getByText("3 waiting on you")).toBeTruthy();
    expect(within(list).getByText("Planning")).toBeTruthy();
    expect(within(list).getByText("Step 2 of 2 rejected · Box scores")).toBeTruthy();
    expect(within(list).getByRole("link", { name: /Sports Predictions/ }).getAttribute("href")).toBe("/goals/goal_s");
    fireEvent.click(screen.getByRole("button", { name: /Waiting on you/ }));
    expect(within(screen.getByRole("list", { name: "Goals" })).queryByText("Weekly review")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "New goal" }));
    const form = screen.getByRole("form", { name: "New goal" });
    fireEvent.change(within(form).getByLabelText("Name"), { target: { value: "Customers first" } });
    fireEvent.change(within(form).getByLabelText("The goal, in your words"), { target: { value: "Flag every customer email within an hour." } });
    fireEvent.click(within(form).getByRole("switch", { name: "May send email on its own" }));
    fireEvent.click(within(form).getByRole("button", { name: "Add goal" }));
    await waitFor(() => expect(m.createGoal).toHaveBeenCalledWith("cus_1", {
      title: "Customers first", directive: "Flag every customer email within an hour.", may_send: true, may_post: false }));
  });

  it("an old ?goal= link opens the goal's page", async () => {
    mount("/goals?goal=goal_s");
    expect(await screen.findByRole("heading", { name: "Sports Predictions" })).toBeTruthy();
  });

  it("the goal page: steps measured live, check-ins counted, a line to the next check-in", async () => {
    mount("/goals/goal_s");
    expect(await screen.findByRole("heading", { name: "Sports Predictions" })).toBeTruthy();
    await screen.findByText("1 of 2");
    expect(screen.getByText("plus 1 repeating")).toBeTruthy();
    expect(screen.getByText("80 tool calls total")).toBeTruthy();
    await screen.findByText("3,690");
    const step2 = screen.getByRole("listitem", { name: "Step 2" });
    expect(within(step2).getByText("Rejected · attempt 2")).toBeTruthy();
    expect(await within(step2).findByText("1,512 of 3,690 filled · 41.0%")).toBeTruthy();
    expect(within(step2).getByText("2,178 of 3,690 rows lack home_score.")).toBeTruthy();
    expect(within(step2).getByText("202212010BOS")).toBeTruthy();
    expect(within(step2).getByText("Reviewer")).toBeTruthy();
    expect(within(step2).getByText("Checked once the counted ones pass.")).toBeTruthy();
    const step1 = screen.getByRole("listitem", { name: "Step 1" });
    expect(within(step1).getByText("Every game is listed")).toBeTruthy();
    expect(within(step1).getByRole("button", { name: "Go back to here" })).toBeTruthy();
    expect(screen.queryByRole("listitem", { name: "Step 3" }) && within(screen.getByRole("listitem", { name: "Step 3" })).getByText("Repeats every check-in")).toBeTruthy();
    // Check-ins: numbered oldest first, plans not counted; the comment says which check-in read it.
    const history = screen.getByRole("list", { name: "Check-in history" });
    expect(await within(history).findByText("#2")).toBeTruthy();
    expect(within(history).getByText("#1")).toBeTruthy();
    // A handled note sits in the history with what the check-in said; one still in force sits above it.
    expect(within(history).getByText("#2: Started at November.")).toBeTruthy();
    const force = screen.getByRole("list", { name: "Your notes in force" });
    expect(within(force).getByText("If a site won't load, find another source.")).toBeTruthy();
    expect(within(force).getByText("Can't yet")).toBeTruthy();
    expect(within(force).getByText("#2: Tried sports-reference.com; blocked too.")).toBeTruthy();
    fireEvent.click(within(force).getByRole("button", { name: "Apply every check-in" }));
    await waitFor(() => expect(m.setGoalNote).toHaveBeenCalledWith("cus_1", "goal_s", "c2", { always: true }));
    expect(within(history).getByText("Step 2 rejected")).toBeTruthy();
    fireEvent.click(within(history).getByText("#2"));
    expect(within(history).getByText("Stopped: tool-call budget · m")).toBeTruthy();
    expect(within(history).getByText("web_tables ×16")).toBeTruthy();
    expect(screen.getByText("Check-in #3 works on it first, and every check-in after until it's handled.")).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Tell the next check-in"), { target: { value: "Start with November." } });
    fireEvent.click(screen.getByRole("button", { name: "Send to next check-in" }));
    await waitFor(() => expect(m.addGoalComment).toHaveBeenCalledWith("cus_1", "goal_s", "Start with November.", false));
    // The survey and the lattices.
    expect(screen.getByText("What the plan was sized on")).toBeTruthy();
    expect(screen.getByText("1,230")).toBeTruthy();
    expect(screen.getByText("estimated")).toBeTruthy();
    expect(screen.getByRole("link", { name: "nba_games" }).getAttribute("href")).toBe("/lattices/lat_g");
  });

  it("editing when a step is done: plain sentences, a new line goes to the reviewer", async () => {
    mount("/goals/goal_s");
    const step2 = await screen.findByRole("listitem", { name: "Step 2" });
    fireEvent.click(within(step2).getByRole("button", { name: "Edit when it's done" }));
    const dialog = await screen.findByRole("dialog", { name: "When is step 2 done?" });
    fireEvent.change(within(dialog).getByLabelText("1"), { target: { value: "Every game has a final score" } });
    fireEvent.change(within(dialog).getByLabelText("Add another (checked by the reviewer)"), { target: { value: "A sample of games reads right." } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Save" }));
    await waitFor(() => expect(m.setStepCriteria).toHaveBeenCalled());
    const sent = (m.setStepCriteria.mock.calls[0] as unknown[])[3] as { kind: string; label: string; text?: string; min_fraction?: number }[];
    expect(sent.map((c) => [c.kind, c.label])).toEqual([["coverage", "Every game has a final score"],
      ["rubric", "The totals add up to each final score"], ["rubric", "A sample of games reads right."]]);
    expect(sent[0].min_fraction).toBe(0.98);
    expect(sent[2].text).toBe("A sample of games reads right.");
  });

  it("going back to a step counts what it undoes first, and says what to do differently", async () => {
    mount("/goals/goal_s");
    const step2 = await screen.findByRole("listitem", { name: "Step 2" });
    fireEvent.click(within(step2).getByRole("button", { name: "Go back to step 2" }));
    const dialog = await screen.findByRole("dialog", { name: "Go back to step 2?" });
    expect(await within(dialog).findByText("3 undone")).toBeTruthy();
    expect(within(dialog).getByText("2 dismissed")).toBeTruthy();
    expect(within(dialog).getByText("1 stays sent")).toBeTruthy();
    expect(m.previewGoalRollback).toHaveBeenCalledWith("cus_1", "goal_s", 2);
    fireEvent.change(within(dialog).getByLabelText(/What should it do differently/), { target: { value: "Use the box score pages." } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Go back" }));
    await waitFor(() => expect(m.rollbackGoal).toHaveBeenCalledWith("cus_1", "goal_s", 2, "Use the box score pages."));
  });

  it("a step waiting for the person: approve, or send back with a note", async () => {
    const waiting: Goal = { ...sports, plan: { ...sports.plan!, steps: sports.plan!.steps.map((s) => (s.n === 2 ? { ...s, state: "awaiting_review" as const } : s)) } };
    const getImpl = m.getGoal.getMockImplementation()!;
    m.getGoal.mockImplementation(async () => ({ goal: waiting, items: [] }));
    onTestFinished(() => { m.getGoal.mockImplementation(getImpl); });
    mount("/goals/goal_s");
    const step2 = await screen.findByRole("listitem", { name: "Step 2" });
    expect(within(step2).getByText("Waits for your approval")).toBeTruthy();
    fireEvent.click(within(step2).getByRole("button", { name: "Not yet" }));
    fireEvent.change(within(step2).getByLabelText("What it still needs"), { target: { value: "December is missing." } });
    fireEvent.click(within(step2).getByRole("button", { name: "Send back" }));
    await waitFor(() => expect(m.reviewGoalStep).toHaveBeenCalledWith("cus_1", "goal_s", 2, false, "December is missing."));
  });

  it("the page shows what the goal needs and approves an edited draft", async () => {
    mount("/goals/goal_1");
    await screen.findByTestId("draft-gi_d");
    expect(screen.getByText("Short, warm replies.", { exact: false })).toBeTruthy();
    expect(screen.getByText("Uses Gmail.")).toBeTruthy();
    const needs = screen.getByRole("region", { name: "Needs you" });
    expect(within(needs).getByRole("heading", { name: "Turn on Act for Gmail so replies can go out." })).toBeTruthy();
    fireEvent.change(within(needs).getByLabelText("Your answer"), { target: { value: "Turned it on." } });
    fireEvent.click(within(needs).getByRole("button", { name: "Answer" }));
    await waitFor(() => expect(m.goalItemAction).toHaveBeenCalledWith("cus_1", "gi_n", { action: "answer", body: "Turned it on." }));
    const draft = screen.getByTestId("draft-gi_d");
    expect(within(draft).getByText("Gmail to theo@acme.test, “Contract question”")).toBeTruthy();
    fireEvent.click(within(draft).getByRole("button", { name: "Edit" }));
    fireEvent.change(within(draft).getByLabelText("Draft text"), { target: { value: "Hi Theo, confirmed for Friday." } });
    fireEvent.click(within(draft).getByRole("button", { name: "Approve and send" }));
    await waitFor(() => expect(m.goalItemAction).toHaveBeenCalledWith("cus_1", "gi_d", { action: "send", body: "Hi Theo, confirmed for Friday." }));
    fireEvent.click(screen.getByRole("button", { name: "History (1)" }));
    const history = screen.getByRole("list", { name: "History" });
    expect(within(history).getByText("Sent")).toBeTruthy();
    fireEvent.click(screen.getByRole("switch", { name: "Send email on its own" }));
    await waitFor(() => expect(m.updateGoal).toHaveBeenCalledWith("cus_1", "goal_1", { may_send: true }));
    fireEvent.click(screen.getByRole("button", { name: "Run now" }));
    await waitFor(() => expect(m.runGoalNow).toHaveBeenCalled());
    fireEvent.click(screen.getByRole("button", { name: "Delete goal" }));
    const confirm = screen.getByRole("group", { name: "Confirm" });
    fireEvent.click(within(confirm).getByRole("button", { name: "Delete goal" }));
    await waitFor(() => expect(m.deleteGoal).toHaveBeenCalledWith("cus_1", "goal_1"));
  });

  it("what a goal needs is one question with answers to pick, counted when repeated (Series 386)", async () => {
    const problem = { kind: "blocked_source", about: "pro-football-reference.com", step: 2, question: "Where should the NFL schedules come from?",
      options: ["Any reliable public source", "Skip the NFL for now"], recommended: 0, count: 18,
      first_seen: new Date(now - 86_400_000).toISOString(), last_seen: new Date(now).toISOString() };
    const q: GoalItem = { id: "gi_q", goal_id: "goal_s", kind: "note", title: "pro-football-reference.com is blocked (Cloudflare).", body: "",
      watched_item_id: null, target: { problem } as unknown as GoalItem["target"], status: "open", due_at: null, created_at: null, updated_at: null };
    const waiting: Goal = { ...sports, plan: { ...sports.plan!, waiting_on_you: true } };
    const getImpl = m.getGoal.getMockImplementation()!;
    m.getGoal.mockImplementation(async () => ({ goal: waiting, items: [q] }));
    onTestFinished(() => { m.getGoal.mockImplementation(getImpl); });
    mount("/goals/goal_s");
    const card = await screen.findByRole("listitem", { name: "Where should the NFL schedules come from?" });
    expect(screen.getByRole("status").textContent).toContain("Check-ins are paused until you answer or handle every question");
    expect(within(card).getByText("Blocked source")).toBeTruthy();
    expect(within(card).getByText("Step 2")).toBeTruthy();
    expect(within(card).getByText(/Reported 18 times/)).toBeTruthy();
    expect(within(card).getByText("Recommended")).toBeTruthy();
    expect((within(card).getAllByRole("radio")[0] as HTMLInputElement).checked).toBe(true);
    fireEvent.click(within(card).getByRole("button", { name: "Answer" }));
    await waitFor(() => expect(m.goalItemAction).toHaveBeenCalledWith("cus_1", "gi_q", { action: "answer", body: "Any reliable public source" }));
  });

  it("a goal held for setup says what it needs, and offers Check again and Run anyway (Q112 A)", async () => {
    const held: Goal = { ...goal, id: "goal_h", title: "Sports catalog", status: "needs_setup", open: { action: 0, draft: 0, note: 0 },
      plan: { ...goal.plan!, blocked: true, missing: [
        { kind: "data_source", need: "A sports statistics API", why: "Every step reads box scores.", suggestion: "Pick a provider, or run on web search.", blocking: true },
        { kind: "app", integration: "slack", need: "Connect Slack", why: "Posts the daily recap.", suggestion: "Add it.", blocking: false },
      ] } };
    const getImpl = m.getGoal.getMockImplementation()!;
    m.getGoal.mockImplementation(async () => ({ goal: held, items: [] }));
    onTestFinished(() => { m.getGoal.mockImplementation(getImpl); });
    mount("/goals/goal_h");
    const needs = await screen.findByRole("note", { name: "What this goal needs" });
    expect(within(needs).getByText(/On hold until this is set up/)).toBeTruthy();
    expect(within(needs).getByText("A sports statistics API")).toBeTruthy();
    expect(within(needs).getByText("Data source")).toBeTruthy();
    expect(within(needs).getByRole("link", { name: "Connect" }).getAttribute("href")).toBe("/settings/connections");
    expect(screen.queryByRole("button", { name: "Replan" })).toBeNull();
    fireEvent.click(within(needs).getByRole("button", { name: "Run anyway" }));
    await waitFor(() => expect(m.runGoalAnyway).toHaveBeenCalledWith("cus_1", "goal_h"));
    fireEvent.click(within(needs).getByRole("button", { name: "Check again" }));
    await waitFor(() => expect(m.replanGoal).toHaveBeenCalledWith("cus_1", "goal_h"));
  });

  it("a goal waiting for answers asks its questions; picks and your own words go back to the planner", async () => {
    const asking: Goal = { ...goal, id: "goal_q", title: "Sports catalog", status: "deciding", open: { action: 0, draft: 0, note: 0 },
      plan: { ...goal.plan!, steps: [], deciding: true, missing: [], decisions: [
        { question: "Where should the catalog live?", why: "Every step writes there.", recommended: 0, answer: null,
          options: [{ label: "Crystal memory", detail: "Facts in your bank" }, { label: "Notion", detail: "Connect Notion first" }] },
        { question: "How far back?", why: "", recommended: 1, answer: null,
          options: [{ label: "From today", detail: "" }, { label: "This season", detail: "" }] },
      ] } };
    const getImpl = m.getGoal.getMockImplementation()!;
    m.getGoal.mockImplementation(async () => ({ goal: asking, items: [] }));
    onTestFinished(() => { m.getGoal.mockImplementation(getImpl); });
    mount("/goals/goal_q");
    const form = await screen.findByRole("form", { name: "Questions before the plan" });
    expect(screen.getByText("Written once you answer the questions above.")).toBeTruthy();
    const where = within(form).getByRole("radiogroup", { name: "Where should the catalog live?" });
    expect(within(where).getAllByRole("radio")[0].getAttribute("aria-checked")).toBe("true"); // the recommendation is preselected
    expect(within(where).getByText("Recommended")).toBeTruthy();
    const back = within(form).getByRole("radiogroup", { name: "How far back?" });
    fireEvent.click(within(back).getByText("Something else"));
    const submit = within(form).getByRole("button", { name: "Plan with these answers" });
    expect(submit.hasAttribute("disabled")).toBe(true);
    fireEvent.change(within(form).getByLabelText("Your answer: How far back?"), { target: { value: "Only the playoffs" } });
    fireEvent.click(submit);
    await waitFor(() => expect(m.decideGoal).toHaveBeenCalledWith("cus_1", "goal_q", [{ index: 0, choice: 0 }, { index: 1, other: "Only the playoffs" }]));
  });
});
