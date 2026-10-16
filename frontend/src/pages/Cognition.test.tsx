// @vitest-environment jsdom
// Console Refine PR 6 (docs/CONSOLE_UX_AUDIT.md 5.4, bug 7.1): tenants see
// no dollars, the gaps list renders gaps with their verbs, a run opens as
// a drawer at /cognition/run/:id, the checklist reads Met / Partly / Not
// yet, and the Background research switch writes the auto_research row.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { ReactNode } from "react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const RUNS = [
  { id: "env_1", customer_id: "cus_1", status: "working", trigger_type: "research", trigger_id: "task_1", goal_title: "Find the deploy window", attempts: 2, max_attempts: 3, cycle: 2, cycle_cap: 3, step_count: 4, steps_complete: 1, cost_usd: 0.007, tokens_used: 1200, created_at: "2026-01-01T00:00:00Z", open_critiques: 1, capacity_share_pct: 1.4 },
  { id: "env_2", customer_id: "cus_1", status: "needs_human_review", trigger_type: "research", trigger_id: "task_2", goal_title: "Pricing page facts", attempts: 3, max_attempts: 3, step_count: 4, steps_complete: 4, cost_usd: 0.02, tokens_used: 5000, created_at: "2026-01-01T00:00:00Z", open_critiques: 0, capacity_share_pct: 4 },
  { id: "env_3", customer_id: "cus_1", status: "complete", trigger_type: "fill_gap", goal_title: "Team roster", attempts: 1, max_attempts: 3, step_count: 3, steps_complete: 3, cost_usd: 0.001, tokens_used: 300, created_at: "2026-01-01T00:00:00Z", open_critiques: 0, capacity_share_pct: 0.2 },
];
const DETAIL = {
  id: "env_1", customer_id: "cus_1", status: "working", trigger_type: "research", trigger_id: "task_1", attempts: 2, max_attempts: 3, cycle: 2, cycle_cap: 3,
  cost_usd: 0.007, tokens_used: 1200, created_at: "2026-01-01T00:00:00Z", capacity_share_pct: 1.4,
  goal: { title: "Find the deploy window", description: "Find when deploys run.", acceptance_criteria: ["Names the day", "Names the hour", "Cites a source"], amendments: [] },
  plan: { reasoning: "Search docs then compose.", steps: [{ id: 1, action: "crystal_search", description: "look in memory", parallel_group: null, model: "x" }, { id: 2, action: "compose", description: "write it up", parallel_group: null, model: "x" }], suggested_key: "Ops|Deploy" },
  steps: { "1": { step_id: 1, action: "crystal_search", status: "complete", model_used: "none", duration_ms: 1200, output: { results_count: 3 }, error: null }, "2": { step_id: 2, action: "compose", status: "running", model_used: "x", duration_ms: 0, output: {}, error: null } },
  deliverables: {}, validation: { approved: false, score: 0.5, reasoning: "Hour missing.", criteria_evaluation: [{ criterion: "Names the day", status: "MET", evidence: "Tuesday" }, { criterion: "Names the hour", status: "PARTIALLY_MET", evidence: "morning" }, { criterion: "Cites a source", status: "NOT_MET", evidence: "" }], issues: ["No hour"] },
  rejection_log: [{ attempt: 1, reasoning: "Too vague.", score: 0.2 }], events: [], attempt_history: [],
};
const GAPS = [
  { id: "g_1", customer_id: "cus_1", domain: "Ops", subject: "Deploys", missing: "When do deploys run?", priority: "medium", status: "open", disposition: "needs_document", source: "llm_observation", filled_by_crystal_id: null, filled_snippet: null, run_count: 0, last_run_id: null, last_run_status: null, created_at: "2026-01-01T00:00:00Z", resolved_at: null },
  { id: "g_2", customer_id: "cus_1", domain: null, subject: "Pricing", missing: "What is the enterprise price?", priority: "medium", status: "open", disposition: "researchable", source: "llm_observation", filled_by_crystal_id: null, filled_snippet: null, run_count: 1, last_run_id: "env_1", last_run_status: "working", created_at: "2026-01-01T00:00:00Z", resolved_at: null },
  { id: "g_3", customer_id: "cus_1", domain: null, subject: "Team", missing: "Who owns billing?", priority: "medium", status: "filled", disposition: "researchable", source: "llm_observation", filled_by_crystal_id: "cr_9", filled_snippet: "Dana owns billing.", run_count: 1, last_run_id: "env_3", last_run_status: "complete", created_at: "2026-01-01T00:00:00Z", resolved_at: "2026-01-02T00:00:00Z" },
  { id: "g_4", customer_id: "cus_1", domain: null, subject: "Legal", missing: "Which DPA version applies?", priority: "medium", status: "open", disposition: "workable", source: "llm_observation", filled_by_crystal_id: null, filled_snippet: null, run_count: 0, last_run_id: null, last_run_status: null, created_at: "2026-01-01T00:00:00Z", resolved_at: null },
];
const m = {
  listCognitionRuns: vi.fn(async () => RUNS),
  getCognitionRun: vi.fn(async () => DETAIL),
  listKnowledgeGaps: vi.fn(async () => ({ total: GAPS.length, items: GAPS, cycle_cap: 3 })),
  listCognitionTasks: vi.fn(async () => ({ total: 0, items: [] })),
  listReviewQueue: vi.fn(async () => ({ total: 1, items: [{ id: "p_1", key: "Pricing|Enterprise", value: "Enterprise is custom priced.", confidence: 0.62, status: "pending", created_at: "2026-01-01T00:00:00Z" }] })),
  listBudgets: vi.fn(async () => ({ budgets: [{ function: "auto_research", cap_micro_usd: 0, period: "monthly" }] })),
  upsertBudget: vi.fn(async () => ({})),
  dailyCost: vi.fn(async () => ({ day_start: "2026-01-01", total_micro_usd: 200_000, budget_per_customer_usd: 1, families: [{ origin: "direct", cost_micro_usd: 120_000, calls: 10 }, { origin: "cognition", cost_micro_usd: 80_000, calls: 2 }] })),
  promoteGapToResearch: vi.fn(async () => ({ task_id: "t", gap_id: "g_2" })),
  answerGap: vi.fn(async () => ({ gap_id: "g_1", crystal_id: "cr_1", status: "filled" })),
  dismissGap: vi.fn(async () => ({ gap_id: "g_4", status: "closed" })),
  reopenGap: vi.fn(async () => ({ gap_id: "g_4", status: "open" })),
  approveReviewItem: vi.fn(async () => ({ approved: true })),
  rejectReviewItem: vi.fn(async () => ({ rejected: true })),
  cancelCognitionTask: vi.fn(async () => ({ cancelled: "requested" })),
  requeueCognitionTask: vi.fn(async () => ({})),
};
vi.mock("@/lib/api", () => ({
  api: new Proxy({}, { get: (_t, k: string) => (...a: unknown[]) => (m as Record<string, (...x: unknown[]) => unknown>)[k](...a) }),
  authedFetch: vi.fn(async () => new Response(JSON.stringify({ critiques: [] }), { status: 200 })),
  errorMessage: (e: unknown, fallback = "Something went wrong.") => (e instanceof Error ? e.message : fallback),
}));
let ME: Record<string, unknown> = { kind: "user", role: "owner", customer_id: "cus_1", usage: { facts_used: 10, fact_cap: 100, fact_state: "ok", ai_capacity_pct: 38 } };
vi.mock("@/lib/auth", () => ({ useAuth: () => ({ me: ME, refreshMe: async () => {} }) }));

import { SelectedCustomerProvider } from "@/lib/selected-customer";
import { ToastProvider } from "@/components/ui";
import { BACKGROUND_RESEARCH_ON_CAP, Cognition, activityShares, gapNeedsYou } from "./Cognition";

function mount(path = "/cognition") {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const tree = (children: ReactNode) => (
    <QueryClientProvider client={qc}>
      <ToastProvider>
        <MemoryRouter initialEntries={[path]}>
          <SelectedCustomerProvider pinnedId="cus_1">{children}</SelectedCustomerProvider>
        </MemoryRouter>
      </ToastProvider>
    </QueryClientProvider>
  );
  return render(tree(
    <Routes>
      <Route path="/cognition" element={<Cognition />} />
      <Route path="/cognition/run/:runId" element={<Cognition />} />
      <Route path="/knowledge" element={<p>knowledge page</p>} />
    </Routes>,
  ));
}

beforeEach(() => { vi.clearAllMocks(); ME = { kind: "user", role: "owner", customer_id: "cus_1", usage: { facts_used: 10, fact_cap: 100, fact_state: "ok", ai_capacity_pct: 38 } }; });
afterEach(cleanup);

describe("Cognition (Refine PR 6)", () => {
  it("counts what works and what needs you, shows capacity without dollars, and lists runs and gaps", async () => {
    const { container } = mount();
    await screen.findByText("1 run working · 3 need you · 3 gaps open");
    expect(screen.getByText("38%")).toBeTruthy();
    expect(screen.getByText(/Background research stops on its own/)).toBeTruthy();
    // Runs with plain-word statuses and capacity shares.
    await screen.findByText("Find the deploy window");
    const pricingRun = screen.getByText("Pricing page facts").closest("button")!;
    expect(within(pricingRun).getByText("Needs you")).toBeTruthy();
    expect(screen.getByText("1.4%")).toBeTruthy();
    // The gaps list renders gaps (bug 7.1) with their verbs.
    await screen.findByText("When do deploys run?");
    expect(screen.getByText("What is the enterprise price?")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Upload a document" }).getAttribute("href")).toBe("/knowledge");
    expect(screen.getAllByRole("button", { name: "Type the answer" }).length).toBeGreaterThan(0);
    expect(within(screen.getByText("What is the enterprise price?").closest("article")!).getByText("Researching")).toBeTruthy();
    expect(screen.getByText(/Filled with:/)).toBeTruthy();
    // Review queue in Assumptions' words.
    expect(screen.getByText("62% sure")).toBeTruthy();
    expect(screen.getByRole("link", { name: /Assumptions/ }).getAttribute("href")).toBe("/assumptions");
    // Acceptance: a tenant session shows no "$" anywhere on the page.
    expect(container.textContent).not.toMatch(/\$/);
  });

  it("opens a run as a drawer at /cognition/run/:id with the checklist in Met / Partly / Not yet", async () => {
    mount("/cognition/run/env_1");
    const dialog = await screen.findByRole("dialog");
    await within(dialog).findByText("Find when deploys run.");
    expect(m.getCognitionRun).toHaveBeenCalledWith("env_1");
    expect(within(dialog).getByText(/attempt 2 of 3 · cycle 2 of 3 · 1.4% of today's capacity/)).toBeTruthy();
    expect(within(dialog).getByText("Met")).toBeTruthy();
    expect(within(dialog).getByText("Partly")).toBeTruthy();
    expect(within(dialog).getByText("Not yet")).toBeTruthy();
    expect(dialog.textContent).not.toMatch(/\$/);
    fireEvent.click(within(dialog).getByRole("tab", { name: /Steps/ }));
    expect(within(dialog).getByText("crystal_search")).toBeTruthy();
    fireEvent.click(within(dialog).getByRole("tab", { name: "Verdicts" }));
    expect(within(dialog).getByText("Hour missing.")).toBeTruthy();
    fireEvent.click(within(dialog).getByRole("button", { name: /Stop/ }));
    await waitFor(() => expect(m.cancelCognitionTask).toHaveBeenCalledWith("task_1"));
    fireEvent.click(within(dialog).getByRole("button", { name: "Close" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });

  it("types an answer into a gap, dismisses with Undo, and queues research", async () => {
    mount();
    await screen.findByText("When do deploys run?");
    const row = screen.getByText("When do deploys run?").closest("article")!;
    fireEvent.click(within(row).getByRole("button", { name: "Type the answer" }));
    fireEvent.change(within(row).getByLabelText("Your answer"), { target: { value: "Tuesdays at 10:00 UTC." } });
    fireEvent.click(within(row).getByRole("button", { name: "Save answer" }));
    await waitFor(() => expect(m.answerGap).toHaveBeenCalledWith("g_1", "Tuesdays at 10:00 UTC."));
    expect(await screen.findByText("Saved to memory. The gap is filled.")).toBeTruthy();
    const legal = screen.getByText("Which DPA version applies?").closest("article")!;
    fireEvent.click(within(legal).getByRole("button", { name: "Dismiss" }));
    await waitFor(() => expect(m.dismissGap).toHaveBeenCalledWith("g_4"));
    fireEvent.click(await screen.findByRole("button", { name: "Undo" }));
    await waitFor(() => expect(m.reopenGap).toHaveBeenCalledWith("g_4"));
    fireEvent.click(within(legal).getByRole("button", { name: "Research it" }));
    await waitFor(() => expect(m.promoteGapToResearch).toHaveBeenCalledWith("g_4"));
  });

  it("the Background research switch writes the auto_research budget row", async () => {
    mount();
    const sw = await screen.findByRole("switch", { name: "Background research" });
    await waitFor(() => expect((sw as HTMLButtonElement).disabled).toBe(false));
    expect(sw.getAttribute("aria-checked")).toBe("false");
    fireEvent.click(sw);
    await waitFor(() => expect(m.upsertBudget).toHaveBeenCalledWith("cus_1", "auto_research", BACKGROUND_RESEARCH_ON_CAP));
    expect(await screen.findByText("Background research is on.")).toBeTruthy();
  });

  it("admins keep the dollar panel", async () => {
    ME = { kind: "user", role: "platform_admin", customer_id: null, usage: null };
    mount();
    await screen.findByText("LLM spend today (admin)");
    expect(screen.getByText("$0.20")).toBeTruthy();
  });

  it("helpers group spend into activities and know which gaps need a person", () => {
    const shares = activityShares({ day_start: "", total_micro_usd: 100, budget_per_customer_usd: null, families: [{ origin: "direct", cost_micro_usd: 50, calls: 1 }, { origin: "shadow_critic", cost_micro_usd: 30, calls: 1 }, { origin: "something_new", cost_micro_usd: 20, calls: 1 }] });
    expect(shares).toEqual([{ label: "Answering your tools", fraction: 0.5 }, { label: "Curation", fraction: 0.3 }, { label: "Other", fraction: 0.2 }]);
    expect(gapNeedsYou(GAPS[0])).toBe(true);
    expect(gapNeedsYou(GAPS[1])).toBe(false);
    expect(gapNeedsYou(GAPS[2])).toBe(false);
  });
});
