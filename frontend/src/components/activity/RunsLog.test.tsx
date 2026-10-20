// @vitest-environment jsdom
// Activity → Runs: tasks joined to runs and schedules, newest first, in
// plain words; a run that ran out of attempts says "Couldn't finish";
// a scheduled run says which schedule and where its brief went; the
// expanded row offers the pipeline drawer and the bank.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { CognitionRun, CognitionTask, ScheduledRun } from "@/lib/types";

const tasks: CognitionTask[] = [
  { id: "cog_sched", task_type: "agent_research", status: "complete", priority: "background", created_at: "2026-10-19T12:00:00Z", started_at: "2026-10-19T12:00:10Z", completed_at: "2026-10-19T12:04:10Z",
    payload: { schedule_id: "sched_a", schedule_title: "Morning news", topic: "news relevant to my knowledge base" }, result: { findings: "# Brief\n\nThree stories.", crystal_id: "crys_1", outcome: "approved" }, result_crystal_id: "crys_1" },
  { id: "cog_gap", task_type: "research", status: "complete", priority: "background", created_at: "2026-10-19T11:00:00Z", completed_at: "2026-10-19T11:09:00Z",
    payload: { gap_id: "gap_1", topic: "what is the deploy window?" }, result: { action: "no_actionable_findings", outcome: "rejected_exhausted", reason: "Failed after 3 attempts. attempt 1: score 0%" } },
  { id: "cog_tool", task_type: "agent_research", status: "running", priority: "background", created_at: "2026-10-19T13:00:00Z", payload: { topic: "compare vector stores" } },
  { id: "cog_asm", task_type: "assumption_verification", status: "failed", priority: "background", created_at: "2026-10-19T10:00:00Z", completed_at: "2026-10-19T10:01:00Z",
    payload: { assumption_crystal_id: "asm_1", statement: "deploys are on Tuesdays" }, error_message: "search provider timed out" },
];
const runs: CognitionRun[] = [
  { id: "env_1", customer_id: "cus_1", status: "completed", trigger_type: "agent", trigger_id: "cog_sched", goal_title: "Morning news", attempts: 1, max_attempts: 3, step_count: 4, steps_complete: 4, cost_usd: 0.01, tokens_used: 12345, created_at: "2026-10-19T12:00:11Z" },
  { id: "env_2", customer_id: "cus_1", status: "needs_human_review", trigger_type: "research", trigger_id: "cog_gap", goal_title: "Deploy window", attempts: 3, max_attempts: 3, step_count: 3, steps_complete: 3, cost_usd: 0.02, tokens_used: 9000, created_at: "2026-10-19T11:00:05Z" },
];
const schedules: ScheduledRun[] = [
  { id: "sched_a", customer_id: "cus_1", kind: "research", title: "Morning news", prompt: "news", cadence: "daily", at_minute: 480, weekday: null, every_hours: null, run_once_at: null, tz: "UTC",
    next_run_at: null, status: "active", series_failures: 0, last_task_id: "cog_sched", last_run_at: "2026-10-19T12:00:00Z", last_outcome: "completed", last_result_summary: "Three stories.", last_crystal_id: "crys_1",
    deliver_url: "https://hooks.slack.com/services/x", delivers: true, last_delivery: "sent", last_delivery_at: "2026-10-19T12:04:20Z", when: "Daily at 08:00 UTC" } as ScheduledRun,
];
const m = {
  listCognitionTasks: vi.fn(async () => ({ total: tasks.length, items: tasks })),
  listCognitionRuns: vi.fn(async () => runs),
  listSchedules: vi.fn(async () => ({ schedules, active: 1, max_active: 3 })),
};
vi.mock("@/lib/api", () => ({
  api: new Proxy({}, { get: (_t, k: string) => (...a: unknown[]) => (m as Record<string, (...x: unknown[]) => unknown>)[k](...a) }),
  errorMessage: (e: unknown, fallback = "Something went wrong.") => (e instanceof Error ? e.message : fallback),
}));

import { RunsLog, fmtDuration, runEntries } from "./RunsLog";

afterEach(cleanup);

describe("runEntries", () => {
  it("joins tasks to their run and schedule, newest first, in plain words", () => {
    const out = runEntries(tasks, runs, schedules);
    expect(out.map((e) => e.id)).toEqual(["cog_tool", "cog_sched", "cog_gap", "cog_asm"]);
    const sched = out[1];
    expect(sched.source_label).toBe("Scheduled · Morning news");
    expect(sched.state).toBe("completed");
    expect(sched.run_id).toBe("env_1");
    expect(sched.attempts).toBe("attempt 1/3");
    expect(sched.tokens).toBe(12345);
    expect(sched.took).toBe("4 min");
    expect(sched.delivery).toBe("Sent to hooks.slack.com");
    expect(sched.crystal_id).toBe("crys_1");
    const gap = out[2];
    expect(gap.source_label).toBe("Knowledge gap");
    expect(gap.state).toBe("unfinished");
    expect(gap.reason).toBe("Failed after 3 attempts. attempt 1: score 0%");
    expect(gap.text).toBe("");
    expect(out[0].state).toBe("running");
    expect(out[0].source_label).toBe("Asked by a tool");
    expect(out[0].run_id).toBeNull();
    expect(out[3].source_label).toBe("Assumption check");
    expect(out[3].state).toBe("failed");
    expect(out[3].reason).toBe("search provider timed out");
    expect(out[3].task).toBe("deploys are on Tuesdays");
  });
  it("formats durations", () => {
    expect(fmtDuration(42_000)).toBe("42s");
    expect(fmtDuration(250_000)).toBe("4 min 10s");
    expect(fmtDuration(3_900_000)).toBe("1 h 5 min");
  });
});

describe("RunsLog", () => {
  it("lists runs, filters by state, and the expanded row offers the pipeline and the bank", async () => {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const onPipeline = vi.fn();
    render(<QueryClientProvider client={qc}><RunsLog customerId="cus_1" onPipeline={onPipeline} /></QueryClientProvider>);
    expect(await screen.findByText("4 runs · the latest 200")).toBeTruthy();
    expect(m.listCognitionTasks).toHaveBeenCalledWith("cus_1", { limit: 200 });
    const table = within(screen.getByRole("table"));
    expect(table.getByText("Scheduled · Morning news")).toBeTruthy();
    expect(table.getByText("Couldn't finish")).toBeTruthy();
    expect(table.getByText("Your bank · Sent to hooks.slack.com")).toBeTruthy();
    expect(document.body.textContent).not.toMatch(/\$/);
    // Expand the scheduled run.
    fireEvent.click(table.getByText("news relevant to my knowledge base").closest("tr")!);
    expect(screen.getByText("Delivered").nextElementSibling?.textContent).toBe("Sent to hooks.slack.com");
    expect(screen.getByText("Attempts").nextElementSibling?.textContent).toBe("attempt 1/3");
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe("Brief");
    expect(screen.getByText("Three stories.")).toBeTruthy();
    expect((screen.getByRole("link", { name: "Open in bank" }) as HTMLAnchorElement).getAttribute("href")).toBe("/bank/crys_1");
    fireEvent.click(screen.getByRole("button", { name: "Pipeline details" }));
    expect(onPipeline).toHaveBeenCalledWith("env_1");
    // The state chips narrow the list.
    const chips = within(screen.getByRole("group", { name: "Run state" }));
    expect(chips.getByRole("button", { name: /Couldn't finish/ }).textContent).toContain("2");
    fireEvent.click(chips.getByRole("button", { name: /Working/ }));
    expect(screen.queryByText("Scheduled · Morning news")).toBeNull();
    expect(screen.getByText("compare vector stores")).toBeTruthy();
    const sources = within(screen.getByRole("group", { name: "Source" }));
    expect(sources.getByRole("button", { name: /Asked by a tool/ })).toBeTruthy();
  });
});
