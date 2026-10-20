// @vitest-environment jsdom
// Scheduled runs: results under the schedules, from the task list
// (payload.schedule_id), newest first, long text behind Show all,
// "Couldn't finish" with the reason when the engine ran out of attempts,
// failed runs say why.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { CognitionTask } from "@/lib/types";

const tasks: CognitionTask[] = [
  { id: "cog_old", task_type: "agent_research", status: "complete", priority: "background", created_at: "2026-10-18T12:00:00Z", completed_at: "2026-10-18T12:05:00Z",
    payload: { schedule_id: "sched_a", schedule_title: "Morning news", topic: "news" }, result: { findings: "Old " + "x".repeat(900), crystal_id: "crys_old" } },
  { id: "cog_plain", task_type: "agent_research", status: "complete", priority: "background", created_at: "2026-10-18T13:00:00Z", completed_at: "2026-10-18T13:05:00Z",
    payload: { topic: "a manual research run" }, result: { findings: "not a schedule" } },
  { id: "cog_new", task_type: "agent_research", status: "complete", priority: "background", created_at: "2026-10-19T12:00:00Z", completed_at: "2026-10-19T12:04:00Z",
    payload: { schedule_id: "sched_a", schedule_title: "Morning news" }, result: { action: "no_actionable_findings", outcome: "rejected_exhausted", reason: "Failed after 3 attempts. attempt 1: score 0%" } },
  { id: "cog_bad", task_type: "agent_research", status: "failed", priority: "background", created_at: "2026-10-19T08:00:00Z", completed_at: "2026-10-19T08:01:00Z",
    payload: { schedule_id: "sched_b", schedule_title: "MCP spec" }, error_message: "search provider timed out", result: null },
];
const m = { listCognitionTasks: vi.fn(async () => ({ total: tasks.length, items: tasks })) };
vi.mock("@/lib/api", () => ({
  api: new Proxy({}, { get: (_t, k: string) => (...a: unknown[]) => (m as Record<string, (...x: unknown[]) => unknown>)[k](...a) }),
  errorMessage: (e: unknown, fallback = "Something went wrong.") => (e instanceof Error ? e.message : fallback),
}));

import { ScheduleResults, bankHref, scheduleResults } from "./ScheduleResults";

afterEach(cleanup);

describe("scheduleResults", () => {
  it("keeps only scheduled runs, newest first, with text and crystal", () => {
    const out = scheduleResults(tasks);
    expect(out.map((r) => r.task_id)).toEqual(["cog_new", "cog_bad", "cog_old"]);
    expect(out[0].unfinished).toBe(true);
    expect(out[0].text).toBe("Failed after 3 attempts. attempt 1: score 0%");
    expect(out[2].unfinished).toBe(false);
    expect(out[1].status).toBe("failed");
    expect(out[1].text).toBe("search provider timed out");
    expect(out[2].crystal_id).toBe("crys_old");
  });
  it("opens a crystal in the reader and a document as its crystals", () => {
    expect(bankHref("crys_1", "Morning news")).toBe("/bank/crys_1");
    expect(bankHref("doc_39875bff1d9b4ad9", "Morning news")).toBe("/?doc=doc_39875bff1d9b4ad9&from=Morning%20news");
  });
});

describe("ScheduleResults", () => {
  it("lists results with their status and expands long text", async () => {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={qc}><ScheduleResults customerId="cus_1" /></QueryClientProvider>);
    expect(await screen.findByText("Results 3")).toBeTruthy();
    expect(m.listCognitionTasks).toHaveBeenCalledWith("cus_1", { limit: 200 });
    expect(screen.getByText("Couldn't finish")).toBeTruthy();
    expect(screen.getByText("Failed after 3 attempts. attempt 1: score 0%")).toBeTruthy();
    expect(screen.getByText("Failed")).toBeTruthy();
    expect(screen.getByText("search provider timed out")).toBeTruthy();
    expect(screen.queryByText("not a schedule")).toBeNull();
    expect((screen.getByRole("link", { name: "Open in bank" }) as HTMLAnchorElement).getAttribute("href")).toBe("/bank/crys_old");
    const body = screen.getByText(/^Old x{900}$/).closest("[data-collapsed]")!;
    expect(body.getAttribute("data-collapsed")).toBe("true");
    fireEvent.click(screen.getByRole("button", { name: "Show all" }));
    expect(body.getAttribute("data-collapsed")).toBe("false");
    fireEvent.click(screen.getByRole("button", { name: "Show less" }));
    expect(body.getAttribute("data-collapsed")).toBe("true");
  });
});
