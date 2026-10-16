// @vitest-environment jsdom
// Console Refine PR 2 (docs/CONSOLE_UX_AUDIT.md 4.3 item 1, bug 5): a
// non-ok response from authedFetch announces the plan wall exactly as
// jsonFetch does, and still hands the Response back so the thirty
// callers that read res.ok keep working. Before this, every upload,
// tier change, delete, fact edit, watch, review and import/export call
// skipped the wall and failed silently on a 402.
import { afterEach, describe, expect, it, vi } from "vitest";
import { authedFetch, authedFetchOrThrow, responseMessage, uploadWithProgress } from "./api";

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

// Console Refine PR 5 (5.3 item 3, acceptance 1): the XHR upload reports
// progress and announces a plan wall exactly like the fetch paths, so an
// upload over the memory cap opens the plan wall.
class FakeXHR {
  static next: { status: number; body: unknown; headers?: Record<string, string>; abort?: boolean } = { status: 200, body: {} };
  static sent: FormData | null = null;
  status = 0; statusText = ""; responseText = "";
  upload = { onprogress: null as null | ((e: { lengthComputable: boolean; loaded: number; total: number }) => void) };
  onload: null | (() => void) = null;
  onerror: null | (() => void) = null;
  onabort: null | (() => void) = null;
  headers: Record<string, string> = {};
  open() {}
  setRequestHeader(k: string, v: string) { this.headers[k] = v; }
  getResponseHeader(k: string) { return FakeXHR.next.headers?.[k] ?? null; }
  abort() { this.onabort?.(); }
  send(form: FormData) {
    FakeXHR.sent = form;
    this.upload.onprogress?.({ lengthComputable: true, loaded: 50, total: 100 });
    this.status = FakeXHR.next.status;
    this.responseText = JSON.stringify(FakeXHR.next.body);
    this.upload.onprogress?.({ lengthComputable: true, loaded: 100, total: 100 });
    this.onload?.();
  }
}

describe("uploadWithProgress", () => {
  it("reports progress and resolves the JSON body on success", async () => {
    vi.stubGlobal("XMLHttpRequest", FakeXHR);
    FakeXHR.next = { status: 200, body: { id: "doc_1" } };
    const seen: number[] = [];
    const form = new FormData();
    const body = await uploadWithProgress("/v1/documents/upload", form, { onProgress: (f) => seen.push(f) });
    expect(body).toEqual({ id: "doc_1" });
    expect(seen).toEqual([0.5, 1]);
    expect(FakeXHR.sent).toBe(form);
  });

  it("a 402 with X-Plan-Wall dispatches crystal:plan-wall and throws the humane message", async () => {
    vi.stubGlobal("XMLHttpRequest", FakeXHR);
    FakeXHR.next = { status: 402, body: { error: { message: "Memory is full." } }, headers: { "X-Plan-Wall": "memory_full" } };
    const walls: Array<{ code: string; message: string; status: number }> = [];
    const listener = (e: Event) => walls.push((e as CustomEvent).detail);
    window.addEventListener("crystal:plan-wall", listener);
    const err = await uploadWithProgress("/v1/documents/upload", new FormData()).catch((e: { status: number; message: string }) => e) as { status: number; message: string };
    window.removeEventListener("crystal:plan-wall", listener);
    expect(walls).toEqual([{ code: "memory_full", message: "Memory is full.", status: 402 }]);
    expect(err.status).toBe(402);
    expect(err.message).toBe("Memory is full.");
  });

  it("a plain 400 carries the server's words and dispatches nothing", async () => {
    vi.stubGlobal("XMLHttpRequest", FakeXHR);
    FakeXHR.next = { status: 400, body: { detail: "File contains no extractable text" } };
    const walls: unknown[] = [];
    const listener = (e: Event) => walls.push(e);
    window.addEventListener("crystal:plan-wall", listener);
    const err = await uploadWithProgress("/v1/documents/upload", new FormData()).catch((e: { status: number; message: string }) => e) as { status: number; message: string };
    window.removeEventListener("crystal:plan-wall", listener);
    expect(walls).toEqual([]);
    expect(err.message).toBe("File contains no extractable text");
  });
});
