// @vitest-environment jsdom
// Activity → Background: the curation feed in plain words with a type
// filter and a link to the subject where one exists.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { CurationEvent } from "@/lib/types";

const events: CurationEvent[] = [
  { id: "ev_1", subject_id: "asm_1", event_type: "assumption_written", label: "Deploys happen on Tuesdays", payload: null, created_at: "2026-10-19T12:00:00Z" },
  { id: "ev_2", subject_id: "conf_1", event_type: "conflict_found", label: "Rate is 95 vs rate is 120", payload: null, created_at: "2026-10-19T11:00:00Z" },
  { id: "ev_3", subject_id: null, event_type: "conflict_scan_completed", label: "Scanned 1,398 facts", payload: null, created_at: "2026-10-19T10:00:00Z" },
];
const m = { listCurationActivity: vi.fn(async () => ({ total: events.length, events })) };
vi.mock("@/lib/api", () => ({
  api: new Proxy({}, { get: (_t, k: string) => (...a: unknown[]) => (m as Record<string, (...x: unknown[]) => unknown>)[k](...a) }),
  errorMessage: (e: unknown, fallback = "Something went wrong.") => (e instanceof Error ? e.message : fallback),
}));

import { BackgroundLog, subjectHref } from "./BackgroundLog";
import { eventText } from "./events";

afterEach(cleanup);

describe("BackgroundLog", () => {
  it("names events in plain words, links subjects, and filters by type", async () => {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={qc}><MemoryRouter><BackgroundLog customerId="cus_1" /></MemoryRouter></QueryClientProvider>);
    expect(await screen.findByText(/3 events · the latest 200/)).toBeTruthy();
    expect(m.listCurationActivity).toHaveBeenCalledWith("cus_1", 200);
    const list = within(screen.getByRole("list"));
    expect(list.getByText("Assumption written")).toBeTruthy();
    expect(list.getByText("Conflict scan finished")).toBeTruthy();
    const links = screen.getAllByRole("link", { name: "Open" }).map((l) => l.getAttribute("href"));
    expect(links).toEqual(["/assumptions/asm_1", "/conflicts"]);
    const chips = within(screen.getByRole("group", { name: "Event" }));
    fireEvent.click(chips.getByRole("button", { name: /Conflict found/ }));
    expect(screen.queryByText("Deploys happen on Tuesdays")).toBeNull();
    expect(screen.getByText("Rate is 95 vs rate is 120")).toBeTruthy();
    expect(eventText("something_new")).toBe("something new");
    expect(subjectHref({ id: "x", subject_id: null, event_type: "gap_filled", label: "", payload: null, created_at: null })).toBeNull();
  });
});
