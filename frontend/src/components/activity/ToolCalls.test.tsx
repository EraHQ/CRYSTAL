// @vitest-environment jsdom
// Tool calls: every connected-tool call in words, with its client, what
// it asked, the outcome and the arguments behind the row.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ToolCall } from "@/lib/types";

const calls: ToolCall[] = [
  { id: "t1", surface: "mcp", client: "Claude-User", tool: "remember", args: { text: "deploys are on Tuesdays", key: "Ops|Deploys" }, ok: true, code: null, result_summary: "stored: True; crystal_id: c1", duration_ms: 120, query_log_id: null, created_at: "2026-10-19T22:00:00Z" },
  { id: "t2", surface: "mcp", client: "Cursor", tool: "recall", args: { query: "deploy window" }, ok: true, code: null, result_summary: "count: 2", duration_ms: 40, query_log_id: "qlog_1", created_at: "2026-10-19T21:00:00Z" },
  { id: "t3", surface: "mcp", client: "Cursor", tool: "memory_forget", args: { crystal_id: "c9" }, ok: false, code: "forbidden", result_summary: "not yours", duration_ms: 5, query_log_id: null, created_at: "2026-10-19T20:00:00Z" },
];
const m = { listToolCalls: vi.fn(async () => ({ calls, count: calls.length })) };
vi.mock("@/lib/api", () => ({
  api: new Proxy({}, { get: (_t, k: string) => (...a: unknown[]) => (m as Record<string, (...x: unknown[]) => unknown>)[k](...a) }),
  errorMessage: (e: unknown, fallback = "Something went wrong.") => (e instanceof Error ? e.message : fallback),
}));

import { ToolCalls, askedText, mcpToolText } from "./ToolCalls";

afterEach(cleanup);

describe("ToolCalls", () => {
  it("lists calls in words with client, ask, outcome and arguments", async () => {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={qc}><ToolCalls customerId="cus_1" /></QueryClientProvider>);
    expect(await screen.findByText("3 calls · the latest 200")).toBeTruthy();
    expect(m.listToolCalls).toHaveBeenCalledWith("cus_1", { limit: 200 });
    const table = within(screen.getByRole("table"));
    expect(table.getByText("Saved a memory")).toBeTruthy();
    expect(table.getByText("deploys are on Tuesdays")).toBeTruthy();
    expect(table.getByText("Claude")).toBeTruthy();
    expect(table.getByText("Forgot a memory")).toBeTruthy();
    expect(table.getByText("forbidden")).toBeTruthy();
    expect(table.getAllByText("OK").length).toBe(2);
    fireEvent.click(table.getByText("Forgot a memory").closest("tr")!);
    expect(screen.getByText("What went wrong").nextElementSibling?.textContent).toBe("not yours");
    expect(screen.getByText(/"crystal_id": "c9"/)).toBeTruthy();
    const chips = within(screen.getByRole("group", { name: "Client" }));
    fireEvent.click(chips.getByRole("button", { name: /Claude/ }));
    expect(within(screen.getByRole("table")).queryByText("Forgot a memory")).toBeNull();
    expect(mcpToolText("memory_something_new")).toBe("something new");
    expect(askedText({ k: 5, query: "q" })).toBe("q");
    expect(askedText(null)).toBe("");
  });
});
