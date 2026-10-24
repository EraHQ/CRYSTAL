// @vitest-environment jsdom
// Console Refine PR 4 (docs/CONSOLE_UX_AUDIT.md 5.1): the bank pages at
// 100, searches and filters on the server, opens the reader as a drawer
// at /bank/:id, and never calls a native dialog for a tier change or a
// delete.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { ReactNode } from "react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const TOTAL = 250;
function crystal(i: number) {
  const tiers = ["whitelist", "neutral", "quarantine", "blacklist"];
  return {
    id: `cr_${i}`, summary_text: `crystal ${i}`, fact_count: 3, quality_tier: tiers[i % 4],
    diagnostic_tags: [], created_at: "2026-01-01T00:00:00Z", last_activity: "2026-01-01T00:00:00Z",
    crystal_type: "customer:legacy", build_method: "kmeans",
    headline_key: `Domain ${i % 3}|Topic|k${i}`, headline_claim: null, headline_source_kind: "model_reasoning",
    source_path: null, source_uri: null, recall_hits: i % 7,
  };
}
import type { BankOverview } from "@/lib/types";
const BANK: BankOverview = { crystals: TOTAL, facts: 750, updated_at: "2026-01-01T00:00:00Z", tier_counts: { whitelist: 63, neutral: 63, quarantine: 62, blacklist: 62 } };

const listCrystals = vi.fn(async (_cid: string, opts: { offset?: number; limit?: number; q?: string; tier?: string }) => {
  const offset = opts.offset ?? 0, limit = opts.limit ?? 100;
  if (opts.q === "zebra") return { total: 1, items: [crystal(245)], bank: BANK };
  const all = Array.from({ length: TOTAL }, (_, i) => crystal(i)).filter((c) => !opts.tier || c.quality_tier === opts.tier);
  return { total: all.length, items: all.slice(offset, offset + limit), bank: BANK };
});
const getCrystal = vi.fn(async (_cid: string, id: string) => ({
  ...crystal(Number(id.slice(3))), customer_id: "cus_1", keyword_fingerprint: [], cluster_tightness: null, parent_crystal_id: null,
  facts: [
    { id: "f1", claim_text: "The zebra has stripes.", pair_type: "question_answer", source_kind: "model_reasoning", prompt_text: "Topic|k1", answer_value: null, created_at: "2026-01-01T00:00:00Z", hit_count: 4, source_doc_id: null },
    { id: "f2", claim_text: "Zebras live in herds.", pair_type: "question_answer", source_kind: "document_extraction", prompt_text: "Topic|k2", answer_value: null, created_at: "2026-01-01T00:00:00Z", hit_count: 1, source_doc_id: "doc_1" },
  ],
}));
const getCrystalLedger = vi.fn(async () => []);
const setCrystalTier = vi.fn(async (id: string, tier: string) => ({ crystal_id: id, tier }));
const deleteCrystal = vi.fn(async (id: string) => ({ crystal_id: id, deleted: true }));
const supersedeFact = vi.fn(async () => ({}));
const retireFact = vi.fn(async () => ({ retired: "f1" }));

vi.mock("@/lib/api", () => ({
  api: {
    listCrystals: (...a: unknown[]) => (listCrystals as any)(...a),
    getCrystal: (...a: unknown[]) => (getCrystal as any)(...a),
    getCrystalLedger: (...a: unknown[]) => (getCrystalLedger as any)(...a),
    setCrystalTier: (...a: unknown[]) => (setCrystalTier as any)(...a),
    deleteCrystal: (...a: unknown[]) => (deleteCrystal as any)(...a),
    supersedeFact: (...a: unknown[]) => (supersedeFact as any)(...a),
    retireFact: (...a: unknown[]) => (retireFact as any)(...a),
  },
  authedFetch: vi.fn(),
  errorMessage: (e: unknown, fallback = "Something went wrong.") => (e instanceof Error ? e.message : fallback),
}));

import { SelectedCustomerProvider } from "@/lib/selected-customer";
import { ToastProvider } from "@/components/ui";
import { fmtRelative } from "@/lib/utils";
import { BankBrowser, TRUST, classify } from "./BankBrowser";

function mount(path = "/bank") {
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
      <Route path="/bank" element={<BankBrowser />} />
      <Route path="/bank/:crystalId" element={<BankBrowser />} />
    </Routes>,
  ));
}

beforeEach(() => { vi.clearAllMocks(); });
afterEach(cleanup);

describe("Crystal Bank (Refine PR 4)", () => {
  it("shows the whole-bank overview, pages at 100 and loads more", async () => {
    mount();
    await screen.findByText(/250 crystals · 750 facts · updated/);
    expect(listCrystals).toHaveBeenLastCalledWith("cus_1", { offset: 0, limit: 100, q: "", tier: undefined });
    expect(screen.getByPlaceholderText("Search all 250 crystals by name, key or fact")).toBeTruthy();
    expect(screen.getByText("Showing 100 of 250 crystals.")).toBeTruthy();
    // Trust chips carry the raw tier and its count.
    const chips = within(screen.getByRole("group", { name: "Trust" }));
    const trusted = chips.getByRole("button", { name: /Trusted/ });
    expect(trusted.getAttribute("title")).toBe("whitelist");
    expect(within(trusted).getByText("63")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Load 100 more/ }));
    await screen.findByText("Showing 200 of 250 crystals.");
    expect(listCrystals).toHaveBeenLastCalledWith("cus_1", { offset: 100, limit: 100, q: "", tier: undefined });
  });

  it("searches and filters on the server, so a term reaches a crystal past the loaded page", async () => {
    mount();
    await screen.findByText("Showing 100 of 250 crystals.");
    fireEvent.change(screen.getByLabelText("Search crystals"), { target: { value: "zebra" } });
    await waitFor(() => expect(listCrystals).toHaveBeenLastCalledWith("cus_1", { offset: 0, limit: 100, q: "zebra", tier: undefined }));
    await screen.findByText("Showing 1 of 1 crystal.");
    expect(screen.getByText("k245")).toBeTruthy();
    fireEvent.click(within(screen.getByRole("group", { name: "Trust" })).getByRole("button", { name: /Blocked/ }));
    await waitFor(() => expect(listCrystals).toHaveBeenLastCalledWith("cus_1", { offset: 0, limit: 100, q: "zebra", tier: "blacklist" }));
  });

  it("opens the reader as a drawer straight from /bank/:id and closes back to the list", async () => {
    mount("/bank/cr_1");
    const dialog = await screen.findByRole("dialog");
    expect(getCrystal).toHaveBeenCalledWith("cus_1", "cr_1");
    await within(dialog).findByText("The zebra has stripes.");
    expect(within(dialog).getByText(/3 facts · learned from 2 sources · created/)).toBeTruthy();
    expect(within(dialog).getByText("Recalled 4 times")).toBeTruthy();
    expect(within(dialog).getByRole("tab", { name: "History 0" })).toBeTruthy();
    fireEvent.click(within(dialog).getByRole("button", { name: "Close" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(await screen.findByText("Showing 100 of 250 crystals.")).toBeTruthy();
  });

  it("changes trust without a native dialog; Blocked asks inline first", async () => {
    const confirmSpy = vi.spyOn(window, "confirm");
    const alertSpy = vi.spyOn(window, "alert");
    mount("/bank/cr_1");
    const dialog = await screen.findByRole("dialog");
    await within(dialog).findByText("The zebra has stripes.");
    // cr_1 is neutral; every tier shows its own consequence line.
    expect(within(dialog).getByText(TRUST.neutral.line)).toBeTruthy();
    fireEvent.click(within(dialog).getByRole("radio", { name: "Trusted" }));
    await waitFor(() => expect(setCrystalTier).toHaveBeenCalledWith("cr_1", "whitelist"));
    fireEvent.click(within(dialog).getByRole("radio", { name: "Blocked" }));
    expect(setCrystalTier).toHaveBeenCalledTimes(1);
    expect(within(dialog).getByText(TRUST.blacklist.line)).toBeTruthy();
    fireEvent.click(within(dialog).getByRole("button", { name: "Block" }));
    await waitFor(() => expect(setCrystalTier).toHaveBeenCalledWith("cr_1", "blacklist"));
    expect(confirmSpy).not.toHaveBeenCalled();
    expect(alertSpy).not.toHaveBeenCalled();
  });

  it("deletes from the overflow menu through ConfirmInline and reports a failure in a banner", async () => {
    const confirmSpy = vi.spyOn(window, "confirm");
    deleteCrystal.mockRejectedValueOnce(new Error("Not allowed on this plan"));
    mount("/bank/cr_2");
    const dialog = await screen.findByRole("dialog");
    await within(dialog).findByText("The zebra has stripes.");
    fireEvent.click(within(dialog).getByRole("button", { name: "More" }));
    fireEvent.click(within(dialog).getByRole("menuitem", { name: /Delete crystal/ }));
    fireEvent.click(within(dialog).getByRole("button", { name: "Delete crystal" }));
    await within(dialog).findByText("Not allowed on this plan");
    expect(deleteCrystal).toHaveBeenCalledWith("cr_2");
    expect(confirmSpy).not.toHaveBeenCalled();
    // Second try succeeds and closes the drawer.
    fireEvent.click(within(dialog).getByRole("button", { name: "More" }));
    fireEvent.click(within(dialog).getByRole("menuitem", { name: /Delete crystal/ }));
    fireEvent.click(within(dialog).getByRole("button", { name: "Delete crystal" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(await screen.findByText("Crystal deleted.")).toBeTruthy();
  });

  it("edits a fact as a new version with the History toast, and retires through an inline confirm", async () => {
    mount("/bank/cr_1");
    const dialog = await screen.findByRole("dialog");
    await within(dialog).findByText("The zebra has stripes.");
    fireEvent.click(within(dialog).getAllByRole("button", { name: "Edit" })[0]);
    fireEvent.change(within(dialog).getByLabelText("Fact text"), { target: { value: "The zebra has black and white stripes." } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Save as new version" }));
    await waitFor(() => expect(supersedeFact).toHaveBeenCalledWith("cr_1", "f1", "The zebra has black and white stripes."));
    expect(await screen.findByText("Saved as a new version. The original stays in History.")).toBeTruthy();
    fireEvent.click(within(dialog).getAllByRole("button", { name: "Retire" })[0]);
    expect(within(dialog).getByText(/Retire this fact\?/)).toBeTruthy();
    expect(retireFact).not.toHaveBeenCalled();
    fireEvent.click(within(within(dialog).getByRole("group", { name: "Confirm" })).getByRole("button", { name: "Retire" }));
    await waitFor(() => expect(retireFact).toHaveBeenCalledWith("cr_1", "f1"));
  });

  it("shows the connect/upload empty state without the retired proxy copy", async () => {
    listCrystals.mockResolvedValueOnce({ total: 0, items: [], bank: { crystals: 0, facts: 0, updated_at: null, tier_counts: {} } });
    mount();
    await screen.findByText("No crystals yet");
    expect(screen.getByText(/Crystals appear here as your tools save memories/)).toBeTruthy();
    expect(screen.getByRole("link", { name: /Settings › Connections/ }).getAttribute("href")).toBe("/settings/connections");
    expect(screen.getByRole("link", { name: /Knowledge/ }).getAttribute("href")).toBe("/knowledge");
    expect(screen.queryByText(/proxy/i)).toBeNull();
  });
});

describe("helpers", () => {
  it("classify labels file crystals by path and others by the last key segment", () => {
    expect(classify({ headline_key: "Code|pkg|<module>", crystal_type: "customer:legacy", headline_source_kind: "document_chunk", build_method: "content_chunk", source_path: "src/app/main.py", summary_text: null }))
      .toEqual({ kind: "ingested", breadcrumb: "src › app", title: "main.py" });
    expect(classify({ headline_key: "Python|Async|Cancellation", crystal_type: "customer:legacy", headline_source_kind: "model_reasoning", build_method: "kmeans", source_path: null, summary_text: null }))
      .toEqual({ kind: "knowledge", breadcrumb: "Python › Async", title: "Cancellation" });
  });

  it("fmtRelative never prints Invalid Date", () => {
    const now = Date.parse("2026-10-15T12:00:00Z");
    expect(fmtRelative("2026-10-15T11:59:50Z", now)).toBe("just now");
    expect(fmtRelative("2026-10-15T11:30:00Z", now)).toBe("30 min ago");
    expect(fmtRelative("2026-10-15T09:00:00Z", now)).toBe("3 hours ago");
    expect(fmtRelative("2026-10-13T12:00:00Z", now)).toBe("2 days ago");
    expect(fmtRelative("2026-07-15T12:00:00Z", now)).toBe("3 months ago");
    expect(fmtRelative(null, now)).toBe("");
    expect(fmtRelative("not a date", now)).toBe("");
  });
});
