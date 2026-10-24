// @vitest-environment jsdom
// Settings › Connections › Apps (Q106-Q108): every app with who may
// connect it; each sign-in carries Watch / Remember / Act; Remember needs
// Watch and only remembers what the rules allow; Connect opens Nango's
// Connect UI and hands it the session token when it says it is ready.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AppsResponse } from "@/lib/types";

const conn = {
  id: "ac_1", integration: "slack", account_label: "Kerning", status: "active", uses: { watch: true, remember: true, act: false },
  remember_rules: [{ decision: "allow" as const, field: "channel" as const, match: "equals" as const, value: "launch" }],
  rules_text: ['Remember messages in channel "launch".'], watch_cadence_minutes: 15,
  last_synced_at: new Date(Date.now() - 300_000).toISOString(), last_error: null, created_at: null,
};
const data: AppsResponse = {
  configured: true, features: { connections: true, goals: true, gmail: false },
  apps: [
    { integration: "slack", name: "Slack", blurb: "Messages in the channels you choose.", kind: "message",
      acts: [{ key: "post_message", label: "Post a message" }], beta: null, available: true, note: "",
      remember_hint: "Keep messages that pass your rules as memory.", connections: [conn] },
    { integration: "google-drive", name: "Google Drive", blurb: "Files you pick are watched.", kind: "file", acts: [], beta: null,
      available: true, note: "", remember_hint: "Turn file contents into memory.", connections: [] },
    { integration: "google-mail", name: "Gmail", blurb: "New mail.", kind: "email", acts: [{ key: "send_email", label: "Send email" }],
      beta: "gmail", available: false, note: "Private beta.", remember_hint: "", connections: [] },
  ],
};
const m = {
  listApps: vi.fn(async () => data),
  startAppSession: vi.fn(async () => ({ token: "nst_1", expires_at: null, integration: "google-drive" })),
  updateApp: vi.fn(async () => ({ connection: conn })),
  listAppItems: vi.fn(async () => ({ items: [{ id: "wi_1", connection_id: "ac_1", kind: "message", external_id: "1", title: "Ship it", author: "theo",
    occurred_at: null, url: null, snippet: "Shipping Friday", remembered: true, expires_at: null }], count: 1 })),
  syncApp: vi.fn(async () => ({ queued: true })),
  disconnectApp: vi.fn(async () => ({ deleted: true })),
  reconnectApp: vi.fn(),
};
vi.mock("@/lib/api", () => ({
  api: new Proxy({}, { get: (_t, k: string) => (...a: unknown[]) => (m as Record<string, (...x: unknown[]) => unknown>)[k](...a) }),
  errorMessage: (e: unknown, fallback = "Something went wrong.") => (e instanceof Error ? e.message : fallback),
}));

import { ToastProvider } from "@/components/ui";
import { Apps } from "./Apps";

function mount() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={qc}><ToastProvider><Apps customerId="cus_1" /></ToastProvider></QueryClientProvider>);
}

afterEach(() => { cleanup(); vi.clearAllMocks(); document.getElementById("connect-ui")?.remove(); });

describe("Apps (Q106-Q108)", () => {
  it("shows who may connect what, and each sign-in's switches", async () => {
    mount();
    await screen.findByText("Private beta");
    const apps = screen.getByRole("list", { name: "Apps" });
    expect(within(apps).getByText("Invite only")).toBeTruthy();
    expect(within(apps).getByText("Connect another")).toBeTruthy();
    const row = screen.getByRole("list", { name: "Slack connections" });
    expect(within(row).getByText("Kerning")).toBeTruthy();
    expect(within(row).getByRole("switch", { name: "Watch" }).getAttribute("aria-checked")).toBe("true");
    expect(within(row).getByText('Remember messages in channel "launch".')).toBeTruthy();
    fireEvent.click(within(row).getByRole("switch", { name: "Act" }));
    await waitFor(() => expect(m.updateApp).toHaveBeenCalledWith("cus_1", "ac_1", { uses: { act: true } }));
    // Watch off takes Remember with it.
    fireEvent.click(within(row).getByRole("switch", { name: "Watch" }));
    await waitFor(() => expect(m.updateApp).toHaveBeenCalledWith("cus_1", "ac_1", { uses: { watch: false, remember: false } }));
  });

  it("adds and removes remember rules, and shows recent items", async () => {
    mount();
    const row = await screen.findByRole("list", { name: "Slack connections" });
    fireEvent.click(within(row).getByRole("button", { name: "Add a rule" }));
    const form = within(row).getByRole("form", { name: "New rule" });
    fireEvent.change(within(form).getByLabelText("Field"), { target: { value: "text" } });
    fireEvent.change(within(form).getByLabelText("Value"), { target: { value: "invoice" } });
    fireEvent.click(within(form).getByRole("button", { name: "Add" }));
    await waitFor(() => expect(m.updateApp).toHaveBeenCalledWith("cus_1", "ac_1", { rules: [conn.remember_rules[0],
      { decision: "allow", field: "text", match: "contains", value: "invoice" }] }));
    fireEvent.click(within(row).getByRole("button", { name: "Remove rule 1" }));
    await waitFor(() => expect(m.updateApp).toHaveBeenCalledWith("cus_1", "ac_1", { rules: [] }));
    fireEvent.click(within(row).getByRole("button", { name: "Show recent items" }));
    expect(await within(row).findByText("Ship it")).toBeTruthy();
    expect(within(row).getByText("Remembered")).toBeTruthy();
  });

  it("Connect opens the Connect UI and hands it the token when it is ready", async () => {
    mount();
    await screen.findByText("Google Drive");
    const apps = screen.getByRole("list", { name: "Apps" });
    const drive = within(apps).getByText("Google Drive").closest("li")!;
    fireEvent.click(within(drive as HTMLElement).getByRole("button", { name: "Connect" }));
    await waitFor(() => expect(document.getElementById("connect-ui")).toBeTruthy());
    const iframe = document.getElementById("connect-ui") as HTMLIFrameElement;
    expect(iframe.src.startsWith("https://connect.nango.dev/")).toBe(true);
    const posted: unknown[] = [];
    Object.defineProperty(iframe, "contentWindow", { value: { postMessage: (d: unknown) => posted.push(d) } });
    window.dispatchEvent(new MessageEvent("message", { origin: "https://evil.test", data: { type: "ready" } }));
    expect(posted).toEqual([]);
    window.dispatchEvent(new MessageEvent("message", { origin: "https://connect.nango.dev", data: { type: "ready" } }));
    expect(posted).toEqual([{ type: "session_token", sessionToken: "nst_1" }]);
    window.dispatchEvent(new MessageEvent("message", { origin: "https://connect.nango.dev", data: { type: "close" } }));
    expect(document.getElementById("connect-ui")).toBeNull();
  });
});
