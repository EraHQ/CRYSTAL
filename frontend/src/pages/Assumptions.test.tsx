// @vitest-environment jsdom
// Console Refine PR 7 (docs/CONSOLE_UX_AUDIT.md 5.6): the default view is
// what waits; Approve is the primary verb and moves the badge's cache at
// once; Delete confirms inline; parents link to the bank; the deep link
// highlights a card; Activity opens in a drawer.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { ReactNode } from "react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const base = { diagnostic_tags: [] as string[], parent_crystal_id: null, gap_id: null, invalidated_parents: [] as string[], thread_id: null, responds_to: null, iteration: 0, replaces_parent: false, superseded_by: null };
const A = [
  { ...base, id: "as_1", statement: "Dana also owns invoicing.", quality_tier: "quarantine", recall_gated: true, created_at: "2026-01-10T00:00:00Z", confidence: 0.72, rationale: "Billing and invoicing share an owner.", parents: [{ id: "cr_1", summary_text: "Dana owns billing" }, { id: "cr_2", summary_text: null, label: "ops/billing.md", claims: ["Invoicing sits under billing", "Refunds inside 14 days are automatic"] }] },
  { ...base, id: "as_2", statement: "Deploys skip Fridays.", quality_tier: "neutral", recall_gated: false, created_at: "2026-01-09T00:00:00Z", confidence: 0.9, rationale: null, parents: [] },
  { ...base, id: "as_3", statement: "The old rate still applies.", quality_tier: "blacklist", recall_gated: true, created_at: "2026-01-08T00:00:00Z", confidence: 0.4, rationale: null, parents: [], superseded_by: "as_4" },
  // Q99 C: a consequence, in its own section, recallable but not approved.
  { ...base, id: "asm_c1", kind: "consequence" as const, state: "waiting" as const, statement: "Pro is billed every month.", quality_tier: "quarantine", recall_gated: false, created_at: "2026-01-11T00:00:00Z", confidence: null, rationale: null, parents: [{ id: "cr_p", summary_text: "Pricing" }],
    parent_fact: { id: "f_pro", claim: "Pro is $15 since Oct 1.", key: "Pricing|Pro|Price", crystal_id: "cr_p" },
    checks: [{ fact_id: "f_inv", verdict: "CONSISTENT" as const, claim: "Pro customers get an invoice every month." }, { fact_id: "f_x", verdict: "UNRELATED" as const, claim: "Launch day is November 12." }], consistent_edges: 1, contradicted_edges: 0, approved_by: null },
];
// The mock keeps state so a refetch after an action agrees with it.
let rows = A.map((a) => ({ ...a }));
const m = {
  listAssumptions: vi.fn(async () => ({ total: rows.length, items: rows })),
  assumptionThread: vi.fn(async () => ({ root_id: "x", thread: [], count: 0 })),
  approveAssumption: vi.fn(async (id: string) => { rows = rows.map((a) => (a.id === id ? { ...a, recall_gated: false } : a)); return { crystal_id: id, recall_gated: false }; }),
  verifyAssumption: vi.fn(async () => ({ crystal_id: "as_1", task_id: "t" })),
  respondAssumption: vi.fn(async () => ({ root_id: "as_1", created: ["as_5"], thread: [], count: 2 })),
  deleteCrystal: vi.fn(async (id: string) => { rows = rows.filter((a) => a.id !== id); return { crystal_id: id, deleted: true }; }),
  listCurationActivity: vi.fn(async () => ({ total: 1, events: [{ id: "e1", subject_id: "as_1", event_type: "assumption_written", label: "Dana also owns invoicing.", payload: null, created_at: "2026-01-10T00:00:00Z" }] })),
};
vi.mock("@/lib/api", () => ({
  api: new Proxy({}, { get: (_t, k: string) => (...a: unknown[]) => (m as Record<string, (...x: unknown[]) => unknown>)[k](...a) }),
  errorMessage: (e: unknown, fallback = "Something went wrong.") => (e instanceof Error ? e.message : fallback),
}));

import { SelectedCustomerProvider } from "@/lib/selected-customer";
import { ToastProvider } from "@/components/ui";
import { Assumptions, assumptionState } from "./Assumptions";

function mount(path = "/assumptions") {
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
  return { qc, ...render(tree(<Routes><Route path="/assumptions" element={<Assumptions />} /><Route path="/assumptions/:crystalId" element={<Assumptions />} /></Routes>)) };
}

beforeEach(() => { vi.clearAllMocks(); rows = A.map((a) => ({ ...a })); });
afterEach(cleanup);

describe("Assumptions (Refine PR 7)", () => {
  it("defaults to pending, counts in the header, and links parents to the bank", async () => {
    mount();
    await screen.findByText("1 waiting for review · 3 in all");
    expect(screen.getByText("Dana also owns invoicing.")).toBeTruthy();
    expect(screen.queryByText("Deploys skip Fridays.")).toBeNull();
    expect(screen.queryByText("The old rate still applies.")).toBeNull();
    expect(screen.getByText("72% confident")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Dana owns billing" }).getAttribute("href")).toBe("/bank/cr_1");
    fireEvent.click(within(screen.getByRole("group", { name: "State" })).getByRole("button", { name: /All/ }));
    expect(await screen.findByText("Deploys skip Fridays.")).toBeTruthy();
    expect(screen.getByText("Superseded")).toBeTruthy();
    expect(assumptionState(A[2]).label).toBe("Superseded");
  });

  it("Approve is primary and moves the pending count at once; Delete confirms inline", async () => {
    const confirmSpy = vi.spyOn(window, "confirm");
    const { qc } = mount();
    const card = (await screen.findByText("Dana also owns invoicing.")).closest("article")!;
    fireEvent.click(within(card).getByRole("button", { name: "Approve" }));
    // The ["assumptions", id] cache (the nav badge's source) flips the row with the click.
    await waitFor(() => expect(screen.getByText("0 waiting for review · 3 in all")).toBeTruthy());
    const cached = qc.getQueryData(["assumptions", "cus_1"]) as { items: Array<{ id: string; recall_gated: boolean }> };
    expect(cached.items.find((a) => a.id === "as_1")!.recall_gated).toBe(false);
    await waitFor(() => expect(m.approveAssumption).toHaveBeenCalledWith("as_1"));
    expect(await screen.findByText("Approved. Your tools can recall it now.")).toBeTruthy();
    fireEvent.click(within(screen.getByRole("group", { name: "State" })).getByRole("button", { name: /All/ }));
    const other = (await screen.findByText("Deploys skip Fridays.")).closest("article")!;
    fireEvent.click(within(other).getByRole("button", { name: "Delete" }));
    fireEvent.click(within(within(other).getByRole("group", { name: "Confirm" })).getByRole("button", { name: "Delete" }));
    await waitFor(() => expect(m.deleteCrystal).toHaveBeenCalledWith("as_2"));
    expect(confirmSpy).not.toHaveBeenCalled();
  });

  it("Respond opens inline and sends; Verify queues", async () => {
    mount();
    const card = (await screen.findByText("Dana also owns invoicing.")).closest("article")!;
    fireEvent.click(within(card).getByRole("button", { name: "Respond" }));
    fireEvent.change(within(card).getByLabelText("What is actually true?"), { target: { value: "Dana owns billing only." } });
    fireEvent.click(within(card).getByRole("button", { name: "Send response" }));
    await waitFor(() => expect(m.respondAssumption).toHaveBeenCalledWith("as_1", "Dana owns billing only."));
    expect(await screen.findByText("1 revision added to the thread.")).toBeTruthy();
    fireEvent.click(within(card).getByRole("button", { name: "Verify" }));
    await waitFor(() => expect(m.verifyAssumption).toHaveBeenCalledWith("as_1"));
  });

  it("shows the thinking under the statement and quotes the evidence it was built from", async () => {
    mount();
    const card = (await screen.findByText("Dana also owns invoicing.")).closest("article")!;
    const why = within(card).getByRole("region", { name: "Why" });
    expect(within(why).getByText("Billing and invoicing share an owner.")).toBeTruthy();
    expect(within(card).getByText("Built from")).toBeTruthy();
    // A parent with no summary is named by its file and quotes its first claim.
    const doc = within(card).getByRole("link", { name: /ops\/billing\.md/ });
    expect(doc.getAttribute("href")).toBe("/bank/cr_2");
    expect(within(doc).getByText(/“Invoicing sits under billing” and 1 more/)).toBeTruthy();
  });

  it("deep-links an approved card by switching the filter and highlighting it; Activity opens in a drawer", async () => {
    mount("/assumptions/as_2");
    const card = (await screen.findByText("Deploys skip Fridays.")).closest("article")!;
    expect(card.className).toMatch(/ring-accent/);
    expect(within(screen.getByRole("group", { name: "State" })).getByRole("button", { name: /All/ }).getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(screen.getByRole("button", { name: "Activity" }));
    const dialog = await screen.findByRole("dialog");
    expect(await within(dialog).findByText("Assumption written")).toBeTruthy();
  });

  // Q99 C (2026-10-23): consequences have their own section below the
  // queue, quote the fact they came from and every fact they were checked
  // against, never count in the header, and approve with the same verb.
  it("a consequence sits in its own section with its parent and its checks, and approves", async () => {
    mount();
    await screen.findByText("1 waiting for review · 3 in all");
    const section = screen.getByRole("region", { name: "Consequences" });
    expect(within(section).getByText(/1 waiting · 1 in all/)).toBeTruthy();
    const card = within(section).getByText("Pro is billed every month.").closest("article")!;
    expect(within(card).getByText("Recallable, waiting for review")).toBeTruthy();
    expect(within(card).getByText("1 agrees · 2 checked")).toBeTruthy();
    expect(within(card).getByRole("link", { name: /Pro is \$15 since Oct 1\./ }).getAttribute("href")).toBe("/bank/cr_p");
    expect(within(card).getByText("agrees")).toBeTruthy();
    expect(within(card).getByText("unrelated")).toBeTruthy();
    expect(within(card).getByText("Pro customers get an invoice every month.")).toBeTruthy();
    fireEvent.click(within(card).getByRole("button", { name: "Approve" }));
    await waitFor(() => expect(m.approveAssumption).toHaveBeenCalledWith("asm_c1"));
    // The queue count never moved: consequences are not in the badge.
    expect(screen.getByText("1 waiting for review · 3 in all")).toBeTruthy();
    expect(assumptionState({ ...A[3], state: "approved" } as never).label).toBe("Approved");
    expect(assumptionState({ ...A[3], recall_gated: true } as never).label).toBe("Waiting for review");
  });
});
