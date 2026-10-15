// @vitest-environment jsdom
// Console Refine PR 2 (docs/CONSOLE_UX_AUDIT.md 4.3 item 1, bug 5): a
// non-ok response from authedFetch announces the plan wall exactly as
// jsonFetch does, and still hands the Response back so the thirty
// callers that read res.ok keep working. Before this, every upload,
// tier change, delete, fact edit, watch, review and import/export call
// skipped the wall and failed silently on a 402.
import { afterEach, describe, expect, it, vi } from "vitest";
import { authedFetch, authedFetchOrThrow, responseMessage } from "./api";

function fakeResponse(status: number, body: unknown, headers: Record<string, string> = {}) {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } });
}

afterEach(() => vi.unstubAllGlobals());

describe("authedFetch and the plan wall", () => {
  it("a 402 with X-Plan-Wall dispatches crystal:plan-wall and returns the Response", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => fakeResponse(402, { error: { message: "Memory is full." } }, { "X-Plan-Wall": "memory_full" })));
    const seen: Array<{ code: string; message: string; status: number }> = [];
    const listener = (e: Event) => seen.push((e as CustomEvent).detail);
    window.addEventListener("crystal:plan-wall", listener);
    const res = await authedFetch("/v1/documents/upload", { method: "POST" });
    window.removeEventListener("crystal:plan-wall", listener);
    expect(res.ok).toBe(false);
    expect(res.status).toBe(402);
    expect(seen).toEqual([{ code: "memory_full", message: "Memory is full.", status: 402 }]);
    // The body is still readable by the caller (the wall read a clone).
    expect(await res.json()).toEqual({ error: { message: "Memory is full." } });
  });

  it("a plain failure dispatches nothing and the message helper reads the envelope", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => fakeResponse(400, { error: { message: "Label is too long." } })));
    const seen: unknown[] = [];
    const listener = (e: Event) => seen.push(e);
    window.addEventListener("crystal:plan-wall", listener);
    const res = await authedFetch("/v1/x");
    window.removeEventListener("crystal:plan-wall", listener);
    expect(seen).toEqual([]);
    expect(await responseMessage(res, "Save failed")).toBe("Label is too long.");
  });

  it("a non-JSON failure falls back to the caller's words plus the status", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("<html>", { status: 502 })));
    const res = await authedFetch("/v1/x");
    expect(await responseMessage(res, "Save failed")).toBe("Save failed (502)");
  });

  it("authedFetchOrThrow announces the wall once and throws the humane message", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => fakeResponse(402, { error: { message: "Daily capacity used." } }, { "X-Plan-Wall": "daily_capacity" })));
    const seen: unknown[] = [];
    const listener = (e: Event) => seen.push(e);
    window.addEventListener("crystal:plan-wall", listener);
    await expect(authedFetchOrThrow("/v1/x")).rejects.toThrow("Daily capacity used.");
    window.removeEventListener("crystal:plan-wall", listener);
    expect(seen).toHaveLength(1);
  });
});
