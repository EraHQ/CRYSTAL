// @vitest-environment jsdom
// Console Refine PR 7 (docs/CONSOLE_UX_AUDIT.md 5.5): claim cards carry
// dates, sources and a Newer tag; one primary verb per side; a decision
// leaves the Open list at once (the nav badge reads the same cache) and
// toasts with Undo; the Resolved tab lists history with Reopen; every
// Backlog row is a link; the scan reports in the page.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { ReactNode } from "react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const C1 = {
  id: "k_1", customer_id: "cus_1", fact_a_id: "f_old", fact_b_id: "f_new", crystal_a_id: "cR", crystal_b_id: "cR", subject: "Rate",
  claim_a: "rate is 120", claim_b: "rate is 95", provenance_a: null, provenance_b: null, detector: "contradiction_scan", status: "open",
  resolution: null, pair_key: "pk", created_at: "2026-01-10T00:00:00Z", resolved_at: null,
  facts: {
    a: { learned_at: "2025-12-01T00:00:00Z", source_kind: "document_extraction", source_path: "docs/rates.md", crystal_id: "cR", active: true },
    b: { learned_at: "2026-01-05T00:00:00Z", source_kind: "operator_stated", source_path: null, crystal_id: "cR", active: true },
  },
};
const C2 = { ...C1, id: "k_2", subject: "Owner", claim_a: "Dana owns billing", claim_b: "Dana owns billing.", detector: "dedup_scan" };
const R1 = { ...C1, id: "k_9", subject: "Region", status: "resolved", resolution: "superseded", resolved_at: "2026-01-11T00:00:00Z" };
// The mock keeps state so a refetch after a decision agrees with it.
const state = { open: [C1, C2] as typeof C1[] };
const m = {
  listConflicts: vi.fn(async (_c: string, status: string) => status === "open" ? { total: state.open.length, items: state.open } : status === "resolved" ? { total: 1, items: [R1] } : { total: 0, items: [] }),
  listBacklog: vi.fn(async () => ({ total: 3, items: [
    { kind: "conflict", id: "k_1", subject: "Rate", status: "open", priority_score: 50, created_at: "2026-01-10T00:00:00Z" },
    { kind: "gap", id: "g_1", subject: "Deploys", status: "open", priority_score: 40, created_at: "2026-01-10T00:00:00Z" },
    { kind: "verification", id: "v_1", subject: "Pricing", status: "pending", priority_score: 30, created_at: "2026-01-10T00:00:00Z" },
  ] })),
  listCurationActivity: vi.fn(async () => ({ total: 1, events: [{ id: "e1", subject_id: null, event_type: "conflict_scan_completed", label: "x", payload: { facts_scanned: 2354 }, created_at: "2026-01-09T00:00:00Z" }] })),
  resolveConflict: vi.fn(async (_c: string, id: string) => { state.open = state.open.filter((x) => x.id !== id); return { conflict: { ...C1, status: "resolved", resolution: "superseded" } }; }),
  reopenConflict: vi.fn(async () => { state.open = [C1, ...state.open.filter((x) => x.id !== "k_1")]; return { conflict: C1 }; }),
  scanConflicts: vi.fn(async () => ({ scan: { facts_scanned: 2354, pairs_evaluated: 812, conflicts_found: 1 } })),
};
vi.mock("@/lib/api", () => ({
  api: new Proxy({}, { get: (_t, k: string) => (...a: unknown[]) => (m as Record<string, (...x: unknown[]) => unknown>)[k](...a) }),
  errorMessage: (e: unknown, fallback = "Something went wrong.") => (e instanceof Error ? e.message : fallback),
}));
vi.mock("@/lib/auth", () => ({ useAuth: () => ({ me: { kind: "user", role: "owner", customer_id: "cus_1" }, refreshMe: async () => {} }) }));

import { SelectedCustomerProvider } from "@/lib/selected-customer";
import { ToastProvider } from "@/components/ui";
import { Conflicts, backlogHref, newerSide } from "./Conflicts";

function mount() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const tree = (children: ReactNode) => (
    <QueryClientProvider client={qc}>
      <ToastProvider>
        <MemoryRouter initialEntries={["/conflicts"]}>
          <SelectedCustomerProvider pinnedId="cus_1">{children}</SelectedCustomerProvider>
        </MemoryRouter>
      </ToastProvider>
    </QueryClientProvider>
  );
  return { qc, ...render(tree(<Routes><Route path="/conflicts" element={<Conflicts />} /></Routes>)) };
}

beforeEach(() => { vi.clearAllMocks(); state.open = [C1, C2]; });
afterEach(cleanup);

describe("Conflicts (Refine PR 7)", () => {
  it("shows what the decision needs: dates, sources, the newer claim, kinds and the last scan", async () => {
    mount();
    expect(await screen.findByText(/2 open · last scan .* over 2,354 facts/)).toBeTruthy();
    const card = (await screen.findByText("rate is 120")).closest("article")!;
    expect(within(card).getByText("Newer")).toBeTruthy();
    expect(within(card).getByText("From docs/rates.md")).toBeTruthy();
    expect(within(card).getByText("From you")).toBeTruthy();
    expect(within(card).getAllByText(/^Learned /).length).toBe(2);
    expect(within(card).getByText("Contradiction")).toBeTruthy();
    expect(newerSide(C1)).toBe("b");
    // Kind chips carry the raw detector.
    const chips = within(screen.getByRole("group", { name: "Kind" }));
    expect(chips.getByRole("button", { name: /Duplicates/ }).getAttribute("title")).toBe("dedup_scan");
    fireEvent.click(chips.getByRole("button", { name: /Duplicates/ }));
    expect(screen.queryByText("rate is 120")).toBeNull();
    expect(screen.getByText("Dana owns billing")).toBeTruthy();
  });

  it("keeps one claim: the row leaves Open at once, the toast offers Undo, and Undo reopens", async () => {
    const { qc } = mount();
    const card = (await screen.findByText("rate is 120")).closest("article")!;
    expect(screen.getByRole("tab", { name: "Open 2" })).toBeTruthy();
    const newer = within(card).getByText("rate is 95").closest("div")!;
    fireEvent.click(within(newer).getByRole("button", { name: "Keep this one" }));
    // The shared ["conflicts", id] cache (the nav badge's source) drops it with the click.
    await waitFor(() => expect(screen.getByRole("tab", { name: "Open 1" })).toBeTruthy());
    expect((qc.getQueryData(["conflicts", "cus_1"]) as { total: number }).total).toBe(1);
    await waitFor(() => expect(m.resolveConflict).toHaveBeenCalledWith("cus_1", "k_1", "superseded", "a"));
    fireEvent.click(await screen.findByRole("button", { name: "Undo" }));
    await waitFor(() => expect(m.reopenConflict).toHaveBeenCalledWith("cus_1", "k_1"));
    await waitFor(() => expect(screen.getByRole("tab", { name: "Open 2" })).toBeTruthy());
  });

  it("marks a claim wrong, keeps both, and dismisses with the right verbs", async () => {
    mount();
    const card = (await screen.findByText("rate is 120")).closest("article")!;
    fireEvent.click(within(card).getAllByRole("button", { name: "Mark wrong" })[0]);
    await waitFor(() => expect(m.resolveConflict).toHaveBeenCalledWith("cus_1", "k_1", "blacklisted", "a"));
    const other = (await screen.findByText("Dana owns billing")).closest("article")!;
    fireEvent.click(within(other).getByRole("button", { name: "Both are true" }));
    await waitFor(() => expect(m.resolveConflict).toHaveBeenCalledWith("cus_1", "k_2", "qualified", undefined));
  });

  it("lists decided conflicts under Resolved with Reopen, and every Backlog row is a link", async () => {
    mount();
    await screen.findByText("rate is 120");
    fireEvent.click(screen.getByRole("tab", { name: "Resolved 1" }));
    const row = (await screen.findByText("Region")).closest("article")!;
    expect(within(row).getByText("One kept")).toBeTruthy();
    fireEvent.click(within(row).getByRole("button", { name: "Reopen" }));
    await waitFor(() => expect(m.reopenConflict).toHaveBeenCalledWith("cus_1", "k_9"));
    const backlog = screen.getByRole("complementary");
    const links = within(backlog).getAllByRole("link");
    expect(links.map((l) => l.getAttribute("href"))).toEqual(["/conflicts#conflict-k_1", "/cognition", "/assumptions"]);
    expect(backlogHref({ kind: "push_review", id: "p", subject: "", status: "", priority_score: 0, created_at: null }, false)).toBe("/cognition#review");
    expect(backlogHref({ kind: "agent_task", id: "a", subject: "", status: "", priority_score: 0, created_at: null }, true)).toBe("/agents");
  });

  it("Scan now reports in the page", async () => {
    mount();
    fireEvent.click(await screen.findByRole("button", { name: "Scan now" }));
    expect(await screen.findByText("Scanned 2,354 facts, 812 pairs checked: 1 new conflict.")).toBeTruthy();
  });
});
