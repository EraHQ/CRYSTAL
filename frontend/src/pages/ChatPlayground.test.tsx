// @vitest-environment jsdom
// Console Refine PR 8 (docs/CONSOLE_UX_AUDIT.md 5.2): no key field or CRYS
// chip in Chat; the empty state suggests the bank's own topics; each
// answer cites the memories that answered, with numbers that map to rows
// in the expanded trace (plain words, no JSON); feedback is Helpful / Not
// right with an inline correction; a failed turn shows an ErrorBanner
// with Retry; no native prompt anywhere.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { ReactNode } from "react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const RESULT = {
  id: "run_1", model: "claude-sonnet-5-5", messages: [], final_text: "Deploys run **Tuesdays** at 10:00 UTC.", stop_reason: "end_turn", iterations: 2, prompt_tokens: 1200, completion_tokens: 80,
  tool_calls: [
    { iteration: 1, tool_name: "knowledge_search", tool_use_id: "tu_1", input: { query: "deploy window" }, output: { injection_text: "Deploys happen on Tuesday mornings.", matched_fact_ids: ["f1", "f2", "f3"], matched_crystal_ids: ["cr_1"], top_score: 0.9 }, is_error: false },
    { iteration: 1, tool_name: "content_search", tool_use_id: "tu_2", input: { query: "deploy runbook" }, output: { injection_text: "Runbook: step 3 deploys.", matched_fact_ids: ["d1"], matched_crystal_ids: ["cr_9"] }, is_error: false },
    { iteration: 2, tool_name: "create_document", tool_use_id: "tu_3", input: { content: "# Deploys" }, output: { type: "document", filename: "deploys.md", format: "md", mime: "text/markdown", title: "Deploys", content: "# Deploys", bytes: 9 }, is_error: false },
  ],
};
const m = {
  getCustomer: vi.fn(async () => ({ id: "cus_1", provider: "anthropic", model_id: "m" })),
  listCustomers: vi.fn(async () => ({ items: [] })),
  listCrystals: vi.fn(async () => ({ total: 3, bank: { crystals: 3, facts: 3, updated_at: null, tier_counts: {} }, items: [
    { id: "a", headline_key: "Ops|Deploys|Window" }, { id: "b", headline_key: "Ops|Deploys|Owner" }, { id: "c", headline_key: "Pricing|Enterprise" },
  ] })),
  adminAgentStream: vi.fn(async (_c: string, _b: unknown, onEvent: (e: string, p: unknown) => void) => { onEvent("tool_calls", { calls: [{ tool_use_id: "tu_1", name: "knowledge_search" }] }); return RESULT; }),
  adminAgent: vi.fn(async () => RESULT),
  adminLearn: vi.fn(async () => ({ crystals_written: 1 })),
  listChatSessions: vi.fn(async () => ({ sessions: [], count: 0 })),
  getChatSession: vi.fn(async () => ({ sequence_id: "s", turns: [], count: 0 })),
};
vi.mock("@/lib/api", () => ({
  api: new Proxy({}, { get: (_t, k: string) => (...a: unknown[]) => (m as Record<string, (...x: unknown[]) => unknown>)[k](...a) }),
  errorMessage: (e: unknown, fallback = "Something went wrong.") => (e instanceof Error ? e.message : fallback),
}));
vi.mock("@/lib/auth", () => ({ useAuth: () => ({ status: "signedIn", me: { kind: "user", role: "owner", customer_id: "cus_1" }, refreshMe: async () => {} }) }));

import { SelectedCustomerProvider } from "@/lib/selected-customer";
import { ToastProvider } from "@/components/ui";
import { ChatPlayground, buildTrace, toolLabel } from "./ChatPlayground";

function mount() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const tree = (children: ReactNode) => (
    <QueryClientProvider client={qc}>
      <ToastProvider>
        <MemoryRouter initialEntries={["/playground"]}>
          <SelectedCustomerProvider pinnedId="cus_1">{children}</SelectedCustomerProvider>
        </MemoryRouter>
      </ToastProvider>
    </QueryClientProvider>
  );
  return render(tree(<ChatPlayground />));
}

async function ask(text: string) {
  const box = await screen.findByLabelText("Message");
  await waitFor(() => expect((box as HTMLTextAreaElement).disabled).toBe(false));
  fireEvent.change(box, { target: { value: text } });
  fireEvent.keyDown(box, { key: "Enter" });
}

beforeEach(() => { vi.clearAllMocks(); if (!crypto.randomUUID) Object.defineProperty(crypto, "randomUUID", { value: () => `u${Math.random()}` }); });
afterEach(cleanup);

describe("Chat (Refine PR 8)", () => {
  it("has no key field or CRYS chip, and suggests the bank's own topics", async () => {
    const { container } = mount();
    expect(await screen.findByText("What do you want to check?")).toBeTruthy();
    expect(screen.getByText(/Ask the way you would in Claude, Cursor or ChatGPT/)).toBeTruthy();
    expect(await screen.findByRole("button", { name: "What do you know about Ops?" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "What do you remember about me?" })).toBeTruthy();
    expect(screen.getByRole("button", { name: /History/ })).toBeTruthy();
    expect(screen.queryByPlaceholderText(/sk-ant/)).toBeNull();
    expect(screen.queryByText("CRYS")).toBeNull();
    expect(container.querySelector('input[type="password"]')).toBeNull();
    expect(screen.getByText("Enter to send · Shift Enter for a new line")).toBeTruthy();
  });

  it("cites the memories that answered and expands to a plain-word trace with no JSON", async () => {
    mount();
    await ask("When do deploys run?");
    await screen.findByText("Tuesdays");
    expect(m.adminAgentStream).toHaveBeenCalledTimes(1);
    // One citation per recall step, numbered in order.
    const cites = screen.getByText("From memory").parentElement!;
    expect(within(cites).getByRole("button", { name: "1" }).getAttribute("title")).toBe("Recalled memory: deploy window");
    expect(within(cites).getByRole("button", { name: "2" }).getAttribute("title")).toBe("Searched documents: deploy runbook");
    // The one-line trace.
    const line = screen.getByText(/Looked up 3 memories and 1 document passage, wrote a document · \d+\.\ds · Sonnet$/);
    fireEvent.click(line);
    const trace = screen.getByRole("list", { name: "Trace" });
    expect(trace.querySelectorAll(":scope > li").length).toBe(3);
    expect(screen.getByText("Deploys happen on Tuesday mornings.")).toBeTruthy();
    expect(screen.getByText("Wrote a document")).toBeTruthy();
    const traceText = trace.textContent ?? "";
    expect(traceText).not.toMatch(/[{}]/);
    expect(screen.getAllByRole("link", { name: "open crystal" }).map((l) => l.getAttribute("href"))).toEqual(["/bank/cr_1", "/bank/cr_9"]);
    // The document the run wrote is a card, with the trace naming the step.
    expect(screen.getByRole("button", { name: /Download/ })).toBeTruthy();
  });

  it("a citation number opens the trace on its row", async () => {
    mount();
    await ask("When do deploys run?");
    await screen.findByText("Tuesdays");
    fireEvent.click(within(screen.getByText("From memory").parentElement!).getByRole("button", { name: "2" }));
    const row = await screen.findByText("Searched documents");
    expect(row.closest("li")!.className).toMatch(/border-accent/);
  });

  it("Helpful saves and toasts; Not right asks inline and sends a correction; no native prompt", async () => {
    const promptSpy = vi.spyOn(window, "prompt");
    mount();
    await ask("When do deploys run?");
    await screen.findByText("Tuesdays");
    fireEvent.click(screen.getByRole("button", { name: "Not right" }));
    expect(screen.getByText("What was wrong? Your note is saved to memory as a correction.")).toBeTruthy();
    expect(screen.getByText("Uses a little AI capacity.")).toBeTruthy();
    fireEvent.change(screen.getByLabelText(/What was wrong\?/), { target: { value: "Wednesdays, not Tuesdays." } });
    fireEvent.click(screen.getByRole("button", { name: "Send correction" }));
    await waitFor(() => expect(m.adminLearn).toHaveBeenCalledWith("cus_1", { prompt: "When do deploys run?", response: RESULT.final_text, outcome: "fail", signal: "Wednesdays, not Tuesdays." }));
    expect(await screen.findByText(/Correction saved · 1 new memory/)).toBeTruthy();
    expect(promptSpy).not.toHaveBeenCalled();
    // A second turn, marked helpful.
    await ask("And the hour?");
    await waitFor(() => expect(screen.getAllByText("Tuesdays").length).toBe(2));
    fireEvent.click(screen.getByRole("button", { name: "Helpful" }));
    await waitFor(() => expect(m.adminLearn).toHaveBeenLastCalledWith("cus_1", expect.objectContaining({ outcome: "pass" })));
    expect(await screen.findByText("Saved. Similar questions reuse this answer.")).toBeTruthy();
  });

  it("a failed turn shows an ErrorBanner with Retry, never raw JSON", async () => {
    m.adminAgentStream.mockRejectedValueOnce(new Error("stream down"));
    m.adminAgent.mockRejectedValueOnce(new Error("Daily AI capacity is used up for this plan."));
    mount();
    await ask("When do deploys run?");
    const alert = await screen.findByRole("alert");
    expect(within(alert).getByText("Daily AI capacity is used up for this plan.")).toBeTruthy();
    expect(alert.textContent).not.toMatch(/[{}]/);
    fireEvent.click(within(alert).getByRole("button", { name: /Retry/ }));
    await screen.findByText("Tuesdays");
    expect(m.adminAgentStream).toHaveBeenCalledTimes(2);
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("buildTrace reads crystal_recall facts and counts document passages apart from memories", () => {
    const trace = buildTrace({
      ...RESULT, tool_calls: [
        { iteration: 1, tool_name: "crystal_recall", tool_use_id: "a", input: { query: "owner" }, output: { crystals: [{ crystal_id: "cr_2", facts: [{ fact_id: "f", prompt_text: "Team|Billing", claim_text: "Dana owns billing.", pair_type: "qa", score: 0.8 }] }], count: 1 }, is_error: false },
        { iteration: 1, tool_name: "content_search", tool_use_id: "b", input: { query: "x" }, output: { injection_text: null, matched_fact_ids: [], matched_crystal_ids: [] }, is_error: false },
        { iteration: 2, tool_name: "web_search", tool_use_id: "c", input: { query: "y" }, output: "ok", is_error: false },
      ],
    });
    expect(trace.memories).toBe(1);
    expect(trace.documents).toBe(0);
    expect(trace.steps.map((s) => s.n)).toEqual([1, 2, null]);
    expect(trace.steps[0].memories).toEqual([{ text: "Dana owns billing.", source: "Team|Billing", crystalId: "cr_2" }]);
    expect(trace.steps[1].note).toBe("Nothing matched.");
    expect(toolLabel("some_new_tool")).toBe("some new tool");
  });
});
