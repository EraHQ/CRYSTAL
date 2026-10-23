// @vitest-environment jsdom
// Mind (2026-10-21): the HUD around the scene. The scene itself is
// three.js (no WebGL in jsdom) and is mocked to a handle that records
// what the page asked of it: selections fly, the scrub replays, the
// inspector shows the right card for every kind of thing picked.
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MindScene } from "@/lib/types";

const handle = { select: vi.fn(), highlight: vi.fn(), setScrub: vi.fn(), setAutoRotate: vi.fn(), resize: vi.fn(), destroy: vi.fn() };
let captured: { onSelect: (s: unknown) => void; onHover: (s: unknown, x: number, y: number) => void } | null = null;
vi.mock("@/components/mind/scene", () => ({
  createMindScene: (_el: HTMLElement, _data: MindScene, opts: { onSelect: (s: unknown) => void; onHover: (s: unknown, x: number, y: number) => void }) => { captured = opts; return handle; },
}));

const SINCE = "2026-10-20T23:00:00Z";
const DATA: MindScene = {
  generated_at: "2026-10-21T23:00:00Z",
  window: { since: SINCE, until: "2026-10-21T23:00:00Z", hours: 24 },
  totals: { facts: 3, clusters: 2, regions: 2, sources: 1, contradictions: 1, duplicates: 0, assumptions_waiting: 1, gaps: 1, runs: 1, recalls: 1 },
  regions: [
    { id: 0, name: "Pricing", center: [40, 0, 0], facts: 2, clusters: 1, trusted: 1, hits: 5 },
    { id: 1, name: "Launch", center: [-40, 0, 0], facts: 1, clusters: 1, trusted: 0, hits: 0 },
  ],
  facts: [
    { id: "f1", crystal_id: "c1", region: 0, pos: [40, 1, 1], tier: "trusted", hits: 5, key: "Pricing|Pro", claim: "Pro is $15 since Oct 1", source_kind: "document_chunk", doc_id: "d1", learned_at: "2026-10-21T00:10:00Z" },
    { id: "f2", crystal_id: "c1", region: 0, pos: [41, 2, 0], tier: "new", hits: 0, key: "Pricing|Pro old", claim: "Pro: $12 per month", source_kind: "document_chunk", doc_id: "d1", learned_at: "2026-10-21T00:12:00Z" },
    { id: "f3", crystal_id: "c2", region: 1, pos: [-40, 0, 1], tier: "reviewed", hits: 0, key: "Launch|Date", claim: "Launch day is November 12", source_kind: "model_reasoning", doc_id: null, learned_at: "2026-10-21T00:20:00Z" },
  ],
  sources: [{ id: "d1", label: "Kerning: product brief", region: 0, facts: 2, learned_at: "2026-10-21T00:05:00Z" }],
  bonds: [{ a: "c1", b: "c2", a_pos: [40, 1, 0], b_pos: [-40, 0, 1], region_a: 0, region_b: 1, type: "co_cited", weight: 0.7 }],
  conflicts: [{ id: "k1", kind: "contradiction", subject: "Pricing", status: "open", resolution: null, a: { fact_id: "f2", claim: "Pro: $12 per month", pos: [41, 2, 0], region: 0 }, b: { fact_id: "f1", claim: "Pro is $15 since Oct 1", pos: [40, 1, 1], region: 0 }, reasoning: "Two prices, one plan.", recommendation: { resolution: "superseded", loser: "a", reason: "B is dated." }, found_at: "2026-10-21T04:40:00Z" }],
  assumptions: [{ id: "as1", statement: "Pennant will outgrow three seats", confidence: 0.74, rationale: "A says x; B says y; so z", state: "waiting", parents: [{ crystal_id: "c1", label: "Pricing cluster", pos: [40, 1, 0], region: 0 }, { crystal_id: "c2", label: "Launch cluster", pos: [-40, 0, 1], region: 1 }], formed_at: "2026-10-21T06:00:00Z" }],
  consequences: [{ id: "cq1", statement: "Pro is billed every month", parent_fact_id: "f1", parent_claim: "Pro is $15 since Oct 1", pos: [40, 1, 1], region: 0, state: "waiting", recallable: true,
    checks: [{ fact_id: "f2", verdict: "CONTRADICTS", pos: [41, 2, 0], claim: "Pro: $12 per month" }, { fact_id: "f3", verdict: "UNRELATED", pos: [-40, 0, 1], claim: "Launch day is November 12" }], approved_by: null, formed_at: "2026-10-21T07:00:00Z" }],
  gaps: [{ id: "g1", missing: "Does Kerning charge EU VAT?", subject: "VAT", priority: "medium", source: "gap_discovery", disposition: "researchable", region: 0, opened_at: "2026-10-21T06:30:00Z" }],
  runs: [{ id: "r1", type: "research", origin: null, title: "Token format for color", status: "completed", outcome: "approved", summary: "The 2025.10 format writes a color as an object.", started_at: "2026-10-21T11:31:00Z", finished_at: "2026-10-21T11:33:25Z", result_crystal_id: "c2", region: 1, landing: [-40, 0, 1] }],
  recalls: [{ id: "q1", question: "what does Pro cost?", at: "2026-10-21T14:12:00Z", client: "Claude Desktop", model: "nvidia/nemotron-3-super-120b-a12b", match: "high", latency_ms: 900, region: 0, touched: ["f1", "f2"] }],
  events: [{ id: "e1", type: "conflict_found", label: "Contradiction found - Pricing", subject_id: "k1", at: "2026-10-21T04:40:00Z" }],
  lanes: [{ name: "Contradictions", hours: Array.from({ length: 24 }, (_, i) => ({ calls: i === 5 ? 50 : 0, tokens: i === 5 ? 31000 : 0 })) }, { name: "Other", hours: Array.from({ length: 24 }, () => ({ calls: 0, tokens: 0 })) }],
};

const m = { getMind: vi.fn(async () => DATA) };
vi.mock("@/lib/api", () => ({
  api: new Proxy({}, { get: (_t, k: string) => (...a: unknown[]) => (m as Record<string, (...x: unknown[]) => unknown>)[k](...a) }),
  errorMessage: (e: unknown, fallback = "Something went wrong.") => (e instanceof Error ? e.message : fallback),
}));

import { SelectedCustomerProvider } from "@/lib/selected-customer";
import { focusOf, relationsOf, searchMind } from "@/components/mind/relations";
import { Mind, doingAt, nameOf } from "./Mind";

function mount() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={["/mind"]}>
        <SelectedCustomerProvider pinnedId="cus_1"><Mind /></SelectedCustomerProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => { vi.clearAllMocks(); captured = null; });
afterEach(cleanup);

describe("Mind", () => {
  it("loads the scene once, opens on the whole mind, and the strip shows the lanes that worked", async () => {
    mount();
    await screen.findByText("3 facts in 2 regions");
    await waitFor(() => expect(captured).not.toBeNull());
    expect(m.getMind).toHaveBeenCalledWith("cus_1", 24);
    expect(screen.getByText("Disagreements").previousSibling?.textContent).toBe("1");
    expect(screen.getByText("Contradictions")).toBeTruthy();
    expect(screen.queryByText("Other")).toBeNull();
    expect(screen.getByTestId("mind-when").textContent).toMatch(/^Now/);
  });

  it("shows the right card for everything the scene can pick, with the page it opens on", async () => {
    mount();
    await screen.findByText("3 facts in 2 regions");
    await waitFor(() => expect(captured).not.toBeNull());
    const aside = screen.getByRole("complementary", { name: "Inspector" });
    const pickFromScene = (sel: unknown) => { captured!.onSelect(sel); };

    pickFromScene({ kind: "conflict", id: "k1" });
    await within(aside).findByText("Two facts that cannot both be true");
    expect(within(aside).getByText("Why it looks like a contradiction")).toBeTruthy();
    expect(within(aside).getByText(/Suggested: Keep claim B; claim A is outdated\./)).toBeTruthy();
    expect(within(aside).getByRole("link", { name: "Decide in Conflicts" }).getAttribute("href")).toBe("/conflicts");

    pickFromScene({ kind: "assumption", id: "as1" });
    await within(aside).findByText("Pennant will outgrow three seats");
    expect(within(aside).getByText("A says x; B says y; so z")).toBeTruthy();
    expect(within(aside).getByText("Built from")).toBeTruthy();
    expect(within(aside).getByRole("link", { name: "Review in Assumptions" }).getAttribute("href")).toBe("/assumptions/as1");

    pickFromScene({ kind: "gap", id: "g1" });
    await within(aside).findByText("Does Kerning charge EU VAT?");
    pickFromScene({ kind: "run", id: "r1" });
    await within(aside).findByText("Token format for color");
    expect(within(aside).getByRole("link", { name: "Open the run" }).getAttribute("href")).toBe("/activity/run/r1");
    pickFromScene({ kind: "recall", id: "q1" });
    await within(aside).findByText("what does Pro cost?");
    expect(within(aside).getByText(/2 facts touched/)).toBeTruthy();
    pickFromScene({ kind: "bond", id: "c1|c2" });
    await within(aside).findByText("Two clusters that travel together");
    expect(within(aside).getByText(/Cited together in answers/)).toBeTruthy();
    pickFromScene({ kind: "fact", id: "f1" });
    await within(aside).findByText("Pro is $15 since Oct 1");
    expect(within(aside).getByRole("link", { name: "Open its cluster" }).getAttribute("href")).toBe("/bank/c1");
    pickFromScene({ kind: "source", id: "d1" });
    await within(aside).findByText("Kerning: product brief");

    // Flying to a region from a card asks the scene to go there.
    fireEvent.click(within(aside).getByRole("button", { name: "Fly to Pricing" }));
    expect(handle.select).toHaveBeenCalledWith({ kind: "region", id: 0 }, true);
    await within(aside).findByText("2 facts in 1 clusters · 1 trusted");
    fireEvent.click(within(aside).getByRole("button", { name: "Back to the whole mind" }));
    await within(aside).findByText("3 facts in 2 regions");
  });

  it("scrubbing replays the window: the scene follows and the strip says when and what", async () => {
    mount();
    await screen.findByText("3 facts in 2 regions");
    await waitFor(() => expect(captured).not.toBeNull());
    const scrub = screen.getByLabelText("Replay the window") as HTMLInputElement;
    fireEvent.change(scrub, { target: { value: "6" } });
    await waitFor(() => expect(handle.setScrub).toHaveBeenCalledWith(6));
    expect(screen.getByTestId("mind-when").textContent).toContain("Contradiction found - Pricing");
    expect(doingAt(DATA, 1)).toBe("quiet");
    expect(doingAt(DATA, 6)).toBe("Contradiction found - Pricing");
    expect(doingAt(DATA, 5.5)).toBe("contradictions (50 model calls this hour)");
    expect(nameOf(DATA, { kind: "bond", id: "c1|c2" })).toBe("Bond: Pricing and Launch");
    expect(nameOf(DATA, { kind: "conflict", id: "k1" })).toBe("Disagreement: Pricing");
    expect(nameOf(DATA, { kind: "gap", id: "g1" })).toBe("Gap: Does Kerning charge EU VAT?");
  });

  // The walk (2026-10-23): search is the starting place, every card says
  // what it came from and led to, the scene lights exactly that, and Back
  // retraces the steps.
  it("the walk: search, step along the relations, the scene lights the path, Back retraces", async () => {
    mount();
    await screen.findByText("3 facts in 2 regions");
    await waitFor(() => expect(captured).not.toBeNull());
    const aside = screen.getByRole("complementary", { name: "Inspector" });

    fireEvent.change(screen.getByLabelText("Search the mind"), { target: { value: "VAT" } });
    const results = within(aside).getByRole("listbox", { name: "Search results" });
    fireEvent.click(within(results).getByRole("button", { name: /Does Kerning charge EU VAT\?/ }));
    await within(aside).findByText("Does Kerning charge EU VAT?");
    expect(within(aside).queryByRole("listbox")).toBeNull();
    expect(handle.select).toHaveBeenLastCalledWith({ kind: "gap", id: "g1" }, true);
    await waitFor(() => expect(handle.highlight).toHaveBeenLastCalledWith(expect.objectContaining({
      keys: expect.arrayContaining(["gap:g1", "region:0", "fact:f1", "fact:f2"]), facts: expect.arrayContaining(["f1", "f2"]),
    })));
    expect(screen.getByLabelText("Trail").textContent).toContain("The whole mind ›");

    // A gap sits alongside its region and that region's best facts; stepping onto a fact walks on.
    const alongside = within(aside).getByRole("group", { name: "Alongside" });
    expect(within(alongside).getByRole("button", { name: /^Pricing/ })).toBeTruthy();
    fireEvent.click(within(alongside).getByRole("button", { name: /Pro is \$15 since Oct 1/ }));
    await within(aside).findByText("Pro is $15 since Oct 1");
    expect(within(within(aside).getByRole("group", { name: "Came from" })).getByRole("button", { name: /Kerning: product brief/ })).toBeTruthy();
    const ledTo = within(aside).getByRole("group", { name: "Led to" });
    expect(within(ledTo).getByRole("button", { name: /Pennant will outgrow three seats/ })).toBeTruthy();
    expect(within(ledTo).getByRole("button", { name: /what does Pro cost\?/ })).toBeTruthy();
    expect(within(ledTo).getByRole("button", { name: /disagreement/ })).toBeTruthy();
    await waitFor(() => expect(handle.highlight).toHaveBeenLastCalledWith(expect.objectContaining({
      keys: expect.arrayContaining(["fact:f1", "source:d1", "fact:f2", "conflict:k1", "assumption:as1", "recall:q1"]),
    })));
    expect(screen.getByLabelText("Trail").textContent).toContain("Gap: Does Kerning charge EU");

    // Back retraces: to the gap, then to the whole mind, which clears the focus.
    fireEvent.click(screen.getByRole("button", { name: "Back" }));
    await within(aside).findByText("Does Kerning charge EU VAT?");
    expect(handle.select).toHaveBeenLastCalledWith({ kind: "gap", id: "g1" }, true);
    fireEvent.click(screen.getByRole("button", { name: "Back" }));
    await within(aside).findByText("3 facts in 2 regions");
    await waitFor(() => expect(handle.highlight).toHaveBeenLastCalledWith(null));
    expect(screen.queryByLabelText("Trail")).toBeNull();

    // Picks made in the scene join the trail too.
    captured!.onSelect({ kind: "conflict", id: "k1" });
    await within(aside).findByText("Two facts that cannot both be true");
    expect(screen.getByLabelText("Trail").textContent).toContain("The whole mind ›");
    fireEvent.change(screen.getByLabelText("Search the mind"), { target: { value: "zzqq" } });
    expect(within(aside).getByText("Nothing in the mind matches that.")).toBeTruthy();
  });

  // Q99 C (2026-10-23): a consequence has its own card, walks to the fact
  // it came from and the facts it was checked against, and the fact walks
  // to what it implies.
  it("a consequence: card, walk both ways, search", async () => {
    mount();
    await screen.findByText("3 facts in 2 regions");
    await waitFor(() => expect(captured).not.toBeNull());
    const aside = screen.getByRole("complementary", { name: "Inspector" });
    captured!.onSelect({ kind: "consequence", id: "cq1" });
    await within(aside).findByText("Pro is billed every month");
    expect(within(aside).getByText(/Consequence · recallable, waiting for you · Pricing/)).toBeTruthy();
    expect(within(aside).getByText("Reasoned from one fact and checked against 2: 0 agree, 1 contradicts.")).toBeTruthy();
    expect(within(within(aside).getByRole("group", { name: "Came from" })).getByRole("button", { name: /Pro is \$15 since Oct 1 derived from/ })).toBeTruthy();
    const ledTo = within(aside).getByRole("group", { name: "Led to" });
    expect(within(ledTo).getByRole("button", { name: /Pro: \$12 per month contradicts/ })).toBeTruthy();
    expect(within(ledTo).getByRole("button", { name: /Launch day is November 12 unrelated/ })).toBeTruthy();
    expect(within(aside).getByRole("link", { name: "Review in Assumptions" }).getAttribute("href")).toBe("/assumptions/cq1");
    // The parent fact walks to what it implies; the checked fact sees it alongside.
    const rels = relationsOf(DATA, { kind: "fact", id: "f1" });
    expect(rels.to.find((r) => r.sel.kind === "consequence")?.note).toBe("it implies");
    expect(relationsOf(DATA, { kind: "fact", id: "f2" }).alongside.find((r) => r.sel.kind === "consequence")?.note).toBe("contradicts this");
    expect(searchMind(DATA, "billed").map((r) => r.sel.kind)).toEqual(["consequence"]);
    expect(nameOf(DATA, { kind: "consequence", id: "cq1" })).toBe("Consequence: Pro is billed every month");
  });

  it("relations are read off the one payload: parents, clusters, sources, conflicts, recalls", () => {
    const a = relationsOf(DATA, { kind: "assumption", id: "as1" });
    expect(a.from.map((r) => r.label)).toEqual(expect.arrayContaining(["Pro is $15 since Oct 1", "Launch day is November 12", "Kerning: product brief"]));
    expect(a.from.find((r) => r.label === "Launch day is November 12")?.note).toBe("from Launch cluster");
    const b = relationsOf(DATA, { kind: "bond", id: "c1|c2" });
    expect(b.from.map((r) => r.label)).toEqual(["Pro is $15 since Oct 1", "Pro: $12 per month"]);
    expect(b.to.map((r) => r.label)).toEqual(["Launch day is November 12"]);
    const c = relationsOf(DATA, { kind: "conflict", id: "k1" });
    expect(c.from.map((r) => r.note)).toEqual(["A", "2 facts", "B"]);
    const r = relationsOf(DATA, { kind: "run", id: "r1" });
    expect(r.to.map((x) => x.label)).toEqual(["Launch day is November 12"]);
    expect(relationsOf(DATA, { kind: "recall", id: "q1" }).to.map((x) => x.note)).toEqual(["answered with", "answered with"]);
    expect(relationsOf(DATA, { kind: "overview" })).toEqual({ from: [], alongside: [], to: [] });
    expect(focusOf(DATA, { kind: "region", id: 0 }).facts.sort()).toEqual(["f1", "f2"]);
    expect(searchMind(DATA, "pro").map((x) => x.sel.kind)).toEqual(expect.arrayContaining(["fact", "source", "recall"]));
    expect(searchMind(DATA, "p")).toEqual([]);
  });
});
