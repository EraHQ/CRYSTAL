// @vitest-environment jsdom
// Scheduled runs (launch feature, 2026-10-18; Q78-Q81): the Schedules
// section lists runs with their next run and last outcome, pauses and
// resumes, deletes behind an inline confirm, creates a research question
// on weekdays with the browser's time zone, and disables New schedule at
// the plan cap with the cap in the title.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ScheduledRun } from "@/lib/types";

const row = (id: string, over: Partial<ScheduledRun> = {}): ScheduledRun => ({
  id, customer_id: "cus_1", kind: "brief", title: "Morning brief", prompt: "brief", cadence: "weekdays", at_minute: 480, weekday: null, every_hours: null,
  run_once_at: null, tz: "America/New_York", next_run_at: new Date(Date.now() + 3 * 3600_000).toISOString(), status: "active", series_failures: 0,
  last_task_id: null, last_run_at: new Date(Date.now() - 86_400_000).toISOString(), last_outcome: "completed", when: "Weekdays at 08:00 America/New_York",
  last_result_summary: null, last_crystal_id: null, deliver_url: null, delivers: false, last_delivery: null, last_delivery_at: null,
  created_at: "2026-10-18T10:00:00Z", updated_at: "2026-10-18T10:00:00Z", ...over,
});
let rows: ScheduledRun[] = [];
let maxActive = 10;
const m = {
  listSchedules: vi.fn(async () => ({ schedules: rows, count: rows.length, active: rows.filter((r) => r.status === "active").length, max_active: maxActive })),
  createSchedule: vi.fn(async (_c: string, body: Record<string, unknown>) => {
    const url = (body.deliver_url as string | undefined) ?? null;
    const s = row("sched_new", { kind: body.kind as string, prompt: (body.question as string) ?? "brief", title: (body.title as string) ?? "Research", when: "Weekdays at 08:00 UTC", deliver_url: url, delivers: !!url });
    rows = [...rows, s]; return url ? { schedule: s, deliver_secret: "whsec_once" } : { schedule: s };
  }),
  setScheduleStatus: vi.fn(async (_c: string, id: string, status: string) => { rows = rows.map((r) => (r.id === id ? { ...r, status } : r)); return { schedule: rows.find((r) => r.id === id) }; }),
  setScheduleDelivery: vi.fn(async (_c: string, id: string, url: string) => {
    const had = rows.find((r) => r.id === id)?.delivers;
    rows = rows.map((r) => (r.id === id ? { ...r, deliver_url: url || null, delivers: !!url } : r));
    return url && !had ? { schedule: rows.find((r) => r.id === id), deliver_secret: "whsec_minted" } : { schedule: rows.find((r) => r.id === id) };
  }),
  deleteSchedule: vi.fn(async (_c: string, id: string) => { rows = rows.filter((r) => r.id !== id); return { deleted: true }; }),
};
vi.mock("@/lib/api", () => ({
  api: new Proxy({}, { get: (_t, k: string) => (...a: unknown[]) => (m as Record<string, (...x: unknown[]) => unknown>)[k](...a) }),
  errorMessage: (e: unknown, fallback = "Something went wrong.") => (e instanceof Error ? e.message : fallback),
}));

import { ToastProvider } from "@/components/ui";
import { Schedules, deliveryLabel, hostOf, nextRelative } from "./Schedules";

function mount(ui: ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={qc}><ToastProvider>{ui}</ToastProvider></QueryClientProvider>);
}
beforeEach(() => {
  vi.clearAllMocks();
  maxActive = 10;
  rows = [
    row("sched_a", { deliver_url: "https://hooks.slack.com/services/T/B/x", delivers: true, last_delivery: "failed: 404", last_delivery_at: new Date(Date.now() - 3600_000).toISOString(), last_result_summary: "Two things changed overnight.", last_crystal_id: "crys_brief" }),
    row("sched_b", { kind: "research", title: "MCP spec", prompt: "What changed in the MCP authorization spec?", status: "parked", last_outcome: "failed", series_failures: 3, next_run_at: null }),
  ];
});
afterEach(cleanup);

describe("deliveryLabel and hostOf", () => {
  it("names the host and the last delivery outcome", () => {
    expect(hostOf("https://hooks.slack.com/services/T/B/x")).toBe("hooks.slack.com");
    expect(hostOf(null)).toBe("");
    expect(deliveryLabel({ last_delivery: null, last_delivery_at: null })).toBe("No delivery yet");
    expect(deliveryLabel({ last_delivery: "sent", last_delivery_at: new Date(Date.now() - 120_000).toISOString() })).toMatch(/^Delivered /);
    expect(deliveryLabel({ last_delivery: "failed: 404", last_delivery_at: null })).toBe("Delivery failed: 404");
  });
});

describe("nextRelative", () => {
  it("speaks in minutes, hours and days", () => {
    const now = Date.now();
    expect(nextRelative(new Date(now + 30_000).toISOString(), now)).toBe("within a minute");
    expect(nextRelative(new Date(now + 20 * 60_000).toISOString(), now)).toBe("in 20 min");
    expect(nextRelative(new Date(now + 3 * 3600_000).toISOString(), now)).toBe("in 3 hours");
    expect(nextRelative(new Date(now + 5 * 86_400_000).toISOString(), now)).toBe("in 5 days");
  });
});

describe("Schedules", () => {
  it("lists runs with status, next run and last outcome", async () => {
    mount(<Schedules customerId="cus_1" />);
    expect(await screen.findByText("Morning brief")).toBeTruthy();
    expect(screen.getAllByText("Weekdays at 08:00 America/New_York", { exact: false }).length).toBe(2);
    expect(screen.getByText("Next run in 3 hours")).toBeTruthy();
    expect(screen.getByText("Parked after 3 failures")).toBeTruthy();
    expect(screen.getAllByText("Failed").length).toBe(1);
    expect(screen.getByText("What changed in the MCP authorization spec?")).toBeTruthy();
    expect(screen.getByText("Scheduled runs use your daily AI capacity. If it is used up, that run is skipped.", { exact: false })).toBeTruthy();
    expect(screen.getByText("Delivers to hooks.slack.com", { exact: false })).toBeTruthy();
    expect(screen.getByText("Delivery failed: 404", { exact: false })).toBeTruthy();
    expect(screen.getByText("Two things changed overnight.", { exact: false })).toBeTruthy();
    expect((screen.getByRole("link", { name: "Open" }) as HTMLAnchorElement).getAttribute("href")).toBe("/bank/crys_brief");
    expect(screen.getByRole("button", { name: "Webhook" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Add webhook" })).toBeTruthy();
  });

  it("creates with a webhook and shows the signing secret once", async () => {
    mount(<Schedules customerId="cus_1" />);
    await screen.findByText("Morning brief");
    fireEvent.click(screen.getByRole("button", { name: "New schedule" }));
    const form = screen.getByRole("form", { name: "New schedule" });
    fireEvent.change(within(form).getByLabelText(/^Send results to/), { target: { value: "https://hooks.slack.com/services/T/B/new" } });
    fireEvent.click(within(form).getByRole("button", { name: "Create schedule" }));
    await waitFor(() => expect(m.createSchedule).toHaveBeenCalledTimes(1));
    expect((m.createSchedule.mock.calls[0][1] as Record<string, unknown>).deliver_url).toBe("https://hooks.slack.com/services/T/B/new");
    expect(await screen.findByText("whsec_once")).toBeTruthy();
    expect(screen.getByText(/shown once/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Done" }));
    expect(screen.queryByText("whsec_once")).toBeNull();
  });

  it("adds and removes a webhook on an existing schedule", async () => {
    mount(<Schedules customerId="cus_1" />);
    await screen.findByText("MCP spec");
    fireEvent.click(screen.getByRole("button", { name: "Add webhook" }));
    const editor = screen.getByRole("form", { name: "Webhook for MCP spec" });
    const save = within(editor).getByRole("button", { name: "Save" }) as HTMLButtonElement;
    expect(save.disabled).toBe(true);
    expect(within(editor).queryByRole("button", { name: "Remove webhook" })).toBeNull();
    fireEvent.change(within(editor).getByLabelText("Send results to"), { target: { value: "https://discord.com/api/webhooks/1/abc" } });
    fireEvent.click(save);
    await waitFor(() => expect(m.setScheduleDelivery).toHaveBeenCalledWith("cus_1", "sched_b", "https://discord.com/api/webhooks/1/abc"));
    expect(await screen.findByText("whsec_minted")).toBeTruthy();
    await waitFor(() => expect(screen.getByText("Delivers to discord.com", { exact: false })).toBeTruthy());
    // The brief already delivers: Webhook opens the editor with Remove.
    fireEvent.click(within(screen.getByText("Morning brief").closest("li")!).getByRole("button", { name: "Webhook" }));
    const editor2 = screen.getByRole("form", { name: "Webhook for Morning brief" });
    fireEvent.click(within(editor2).getByRole("button", { name: "Remove webhook" }));
    await waitFor(() => expect(m.setScheduleDelivery).toHaveBeenCalledWith("cus_1", "sched_a", ""));
    await waitFor(() => expect(screen.queryByText("Delivers to hooks.slack.com", { exact: false })).toBeNull());
  });

  it("pauses an active run and resumes a parked one", async () => {
    mount(<Schedules customerId="cus_1" />);
    await screen.findByText("Morning brief");
    const rowOf = (title: string) => screen.getByText(title).closest("li")!;
    expect(within(rowOf("Morning brief")).queryByRole("button", { name: "Resume" })).toBeNull();
    fireEvent.click(within(rowOf("Morning brief")).getByRole("button", { name: "Pause" }));
    await waitFor(() => expect(m.setScheduleStatus).toHaveBeenCalledWith("cus_1", "sched_a", "paused"));
    await waitFor(() => expect(within(rowOf("Morning brief")).getByRole("button", { name: "Resume" })).toBeTruthy());
    fireEvent.click(within(rowOf("MCP spec")).getByRole("button", { name: "Resume" }));
    await waitFor(() => expect(m.setScheduleStatus).toHaveBeenCalledWith("cus_1", "sched_b", "active"));
  });

  it("deletes behind an inline confirm", async () => {
    mount(<Schedules customerId="cus_1" />);
    await screen.findByText("Morning brief");
    fireEvent.click(screen.getAllByRole("button", { name: "Delete" })[0]);
    expect(m.deleteSchedule).not.toHaveBeenCalled();
    const confirm = screen.getByText('Delete "Morning brief"?', { exact: false }).closest("div")!;
    fireEvent.click(within(confirm).getByRole("button", { name: "Delete" }));
    await waitFor(() => expect(m.deleteSchedule).toHaveBeenCalledWith("cus_1", "sched_a"));
    await waitFor(() => expect(screen.queryByText("Morning brief")).toBeNull());
  });

  it("creates a weekday research question with the browser time zone", async () => {
    mount(<Schedules customerId="cus_1" />);
    await screen.findByText("Morning brief");
    fireEvent.click(screen.getByRole("button", { name: "New schedule" }));
    const form = screen.getByRole("form", { name: "New schedule" });
    fireEvent.change(within(form).getByLabelText("What"), { target: { value: "research" } });
    const submit = within(form).getByRole("button", { name: "Create schedule" }) as HTMLButtonElement;
    expect(submit.disabled).toBe(true);
    fireEvent.change(within(form).getByLabelText("Question"), { target: { value: "  What changed in Qdrant?  " } });
    fireEvent.change(within(form).getByLabelText(/^Time/), { target: { value: "09:30" } });
    expect(submit.disabled).toBe(false);
    fireEvent.click(submit);
    await waitFor(() => expect(m.createSchedule).toHaveBeenCalledTimes(1));
    const body = m.createSchedule.mock.calls[0][1] as Record<string, unknown>;
    expect(body.kind).toBe("research");
    expect(body.cadence).toBe("weekdays");
    expect(body.time).toBe("09:30");
    expect(body.question).toBe("What changed in Qdrant?");
    expect(typeof body.tz).toBe("string");
    expect(body.weekday).toBeUndefined();
    expect(body.every_hours).toBeUndefined();
    expect(body.run_at).toBeUndefined();
    await waitFor(() => expect(screen.queryByRole("form", { name: "New schedule" })).toBeNull());
  });

  it("disables New schedule at the plan cap and says the cap", async () => {
    maxActive = 1;
    rows = [row("sched_a")];
    mount(<Schedules customerId="cus_1" />);
    await screen.findByText("Morning brief");
    const btn = screen.getByRole("button", { name: "New schedule" }) as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
    expect(btn.title).toBe("This plan allows 1 schedule");
  });
});
