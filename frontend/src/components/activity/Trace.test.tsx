// @vitest-environment jsdom
// One question's trace: the agent's tool calls render like the chat
// trace (with durations and the raw call behind a disclosure), the
// connected-tool call that produced a recall row, the model calls of
// the turn; dollars for platform admins only.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { QueryTrace as Shape } from "@/lib/types";

const trace: Shape = {
  query_log_id: "qlog_1", sequence_id: "seq_1", turn_index: 0, model: "nvidia/nemotron-3-super-120b-a12b",
  trace: {
    id: "tr_1", created_at: "2026-10-19T22:00:05Z", crystals_used: ["f1"], events: [],
    tool_calls: [
      { iteration: 1, tool_name: "knowledge_search", tool_use_id: "tu_1", input: { query: "crystal bank" }, is_error: false, duration_ms: 812,
        output: { matched_fact_ids: ["f1"], crystals: [{ crystal_id: "cr_1", facts: [{ fact_id: "f1", claim_text: "A crystal bank is self-curating memory", prompt_text: "Crystal|Bank" }] }] } },
      { iteration: 2, tool_name: "web_search", tool_use_id: "tu_2", input: { query: "nemotron pricing" }, is_error: true, duration_ms: 3000, output: "provider timed out" },
    ],
    inferences: [{ claim: "The user builds Crystal Cache" }], borders_crossed: [], gaps_felt: [{ gap: "pricing model" }],
  },
  model_calls: [
    { id: "llm_1", session_id: "seq_1", parent_session_id: null, origin: "agent", model: "nvidia/nemotron-3-super-120b-a12b", input_tokens: 30246, output_tokens: 900, cache_creation_tokens: 0, cache_read_tokens: 12000, cost_micro_usd: 9800, created_at: "2026-10-19T22:00:00Z" },
    { id: "llm_2", session_id: "seq_1", parent_session_id: null, origin: "self_critique", model: "claude-haiku-5-5", input_tokens: 2000, output_tokens: 100, cache_creation_tokens: 0, cache_read_tokens: 0, cost_micro_usd: 300, created_at: "2026-10-19T22:00:20Z" },
  ],
  tool_calls: [{ id: "tcall_1", surface: "mcp", client: "Cursor", tool: "recall", args: { query: "crystal bank" }, ok: true, code: null, result_summary: "count: 1", duration_ms: 40, query_log_id: "qlog_1", created_at: "2026-10-19T22:00:00Z" }],
};
const m = { getQueryTrace: vi.fn(async () => trace) };
vi.mock("@/lib/api", () => ({
  api: new Proxy({}, { get: (_t, k: string) => (...a: unknown[]) => (m as Record<string, (...x: unknown[]) => unknown>)[k](...a) }),
  errorMessage: (e: unknown, fallback = "Something went wrong.") => (e instanceof Error ? e.message : fallback),
}));

import { QueryTrace } from "./Trace";

afterEach(cleanup);

function mount(admin: boolean) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={qc}><MemoryRouter><QueryTrace customerId="cus_1" queryLogId="qlog_1" admin={admin} /></MemoryRouter></QueryClientProvider>);
}

describe("QueryTrace", () => {
  it("shows every step, the connected tool, the self-report and the model calls, without dollars for a tenant", async () => {
    mount(false);
    const steps = within(await screen.findByRole("list", { name: "Trace" }));
    expect(m.getQueryTrace).toHaveBeenCalledWith("cus_1", "qlog_1");
    expect(steps.getByText("Recalled memory")).toBeTruthy();
    expect(steps.getByText("A crystal bank is self-curating memory")).toBeTruthy();
    expect(steps.getByText("812 ms")).toBeTruthy();
    expect(steps.getByText("Searched the web")).toBeTruthy();
    expect(steps.getByText("failed")).toBeTruthy();
    fireEvent.click(steps.getAllByText("Raw call")[0]);
    expect(steps.getByText(/"query": "crystal bank"/)).toBeTruthy();
    expect(screen.getByText("Recalled memory", { selector: "span.font-medium.text-fg[title='recall']" })).toBeTruthy();
    expect(screen.getByText("from Cursor")).toBeTruthy();
    expect(screen.getByText("The user builds Crystal Cache")).toBeTruthy();
    expect(screen.getByText("pricing model")).toBeTruthy();
    const table = within(screen.getByRole("table"));
    expect(table.getByText("Chat answer")).toBeTruthy();
    expect(table.getByText("Self-critique")).toBeTruthy();
    expect(table.getByText("Nemotron Super")).toBeTruthy();
    expect(table.getByText("30,246")).toBeTruthy();
    expect(table.getByText("33,246 tokens · 12,000 from cache")).toBeTruthy();
    expect(document.body.textContent).not.toMatch(/\$/);
  });
  it("shows the cost column to a platform admin", async () => {
    mount(true);
    const table = within(await screen.findByRole("table"));
    expect(table.getByText("Cost")).toBeTruthy();
    expect(table.getByText("$0.0098")).toBeTruthy();
  });
});
