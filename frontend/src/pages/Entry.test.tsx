// @vitest-environment jsdom
// Console Refine PR 12 (docs/CONSOLE_UX_AUDIT.md 5.13 to 5.15): sign in
// and create account share one card and the email; a wrong password
// offers the reset; forgot password sends Firebase's reset email; the
// provider button says which popup is open; the consent page names the
// app, says what it can and cannot do, and shows an outcome; the wizard
// has a named stepper, the key behind a warning, no key for OAuth tools.
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const m = {
  oauthClientInfo: vi.fn(async () => ({ client_id: "c1", client_name: "Claude", redirect_hosts: ["claude.ai"], scopes: ["memory"] })),
  oauthApprove: vi.fn(async () => ({ redirect_to: "https://claude.ai/cb?code=x" })),
  signup: vi.fn(async () => ({ api_key: "crys_live_abcdef1234567890", customer_id: "cus_new" })),
  onboardingStatus: vi.fn(async () => ({ connected: false, last_seen_at: null, ai_tools: [] })),
  createDocumentText: vi.fn(async () => ({})),
};
vi.mock("@/lib/api", () => ({
  api: new Proxy({}, { get: (_t, k: string) => (...a: unknown[]) => (m as Record<string, (...x: unknown[]) => unknown>)[k](...a) }),
  errorMessage: (e: unknown, fallback = "Something went wrong.") => (e instanceof Error ? e.message : fallback),
}));
let resolveGoogle: () => void = () => {};
const auth = {
  status: "signedOut", me: { kind: "user", customer_id: "cus_1", subscription_tier: "free" }, email: "a@example.com",
  error: null as string | null, errorCode: null as string | null,
  signInGoogle: vi.fn(() => new Promise<void>((r) => { resolveGoogle = r; })),
  signInGitHub: vi.fn(async () => {}),
  signInEmail: vi.fn(async () => { auth.error = "That password doesn't match this email."; auth.errorCode = "auth/invalid-credential"; }),
  signUpEmail: vi.fn(async () => {}),
  resetPassword: vi.fn(async () => {}),
  signOut: vi.fn(async () => {}), refreshMe: vi.fn(async () => {}), resendVerification: vi.fn(async () => {}), reloadUser: vi.fn(async () => {}),
};
vi.mock("@/lib/auth", () => ({ useAuth: () => auth }));

import { Login, wrongPassword } from "./Login";
import { OAuthConsent, identify } from "./OAuthConsent";
import { OnboardingSetup } from "./OnboardingSetup";

beforeEach(() => { vi.clearAllMocks(); auth.error = null; auth.errorCode = null; });
afterEach(cleanup);

describe("Sign in (Refine PR 12)", () => {
  it("one card, two tabs sharing the email, a popup state, a wrong-password reset path and the reset flow", async () => {
    render(<Login />);
    expect(screen.getByRole("radio", { name: "Sign in" }).getAttribute("aria-checked")).toBe("true");
    fireEvent.change(screen.getByLabelText("Email"), { target: { value: "a@example.com" } });
    fireEvent.click(screen.getByRole("radio", { name: "Create account" }));
    expect((screen.getByLabelText("Work email") as HTMLInputElement).value).toBe("a@example.com");
    expect(screen.getByText("Free plan: 2,500 facts, no card needed.")).toBeTruthy();
    expect(screen.getByText("We'll send a link to confirm your email.")).toBeTruthy();
    expect(screen.queryByText(/Terms/)).toBeNull();
    expect(screen.getByText(/Your memories stay yours/)).toBeTruthy();
    fireEvent.click(screen.getByRole("radio", { name: "Sign in" }));
    // The Google popup: that button says so, the others dim.
    fireEvent.click(screen.getByRole("button", { name: "Continue with Google" }));
    expect(await screen.findByText("Opening Google…")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Continue with GitHub" }) as HTMLButtonElement).disabled).toBe(true);
    resolveGoogle();
    await waitFor(() => expect(screen.queryByText("Opening Google…")).toBeNull());
    // Wrong password names the email and offers the reset.
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "nope12" } });
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() => expect(auth.signInEmail).toHaveBeenCalledWith("a@example.com", "nope12"));
    cleanup();
    render(<Login />);
    fireEvent.change(screen.getByLabelText("Email"), { target: { value: "a@example.com" } });
    expect(screen.getByRole("alert").textContent).toContain("That password doesn't match a@example.com");
    fireEvent.click(within(screen.getByRole("alert")).getByRole("button", { name: "reset it" }));
    expect(screen.getByText("We'll email a reset link. It works for 30 minutes.")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Send reset link" }));
    await waitFor(() => expect(auth.resetPassword).toHaveBeenCalledWith("a@example.com"));
    expect(await screen.findByRole("status")).toBeTruthy();
    expect(screen.getByText(/Signed up with Google or GitHub\?/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Back to sign in" }));
    expect(screen.getByRole("radio", { name: "Sign in" })).toBeTruthy();
    expect(wrongPassword("auth/wrong-password")).toBe(true);
    expect(wrongPassword("auth/user-not-found")).toBe(false);
  });
});

describe("OAuth consent (Refine PR 12)", () => {
  it("names a known app, states what it can and cannot do, shows details, and an outcome on connect", async () => {
    window.history.pushState({}, "", "/admin/oauth/consent?client_id=c1&redirect_uri=https%3A%2F%2Fclaude.ai%2Fcb&state=s1&code_challenge=abc");
    render(<OAuthConsent />);
    expect(await screen.findByText("Known app")).toBeTruthy();
    expect(screen.getByText(/by Anthropic/)).toBeTruthy();
    expect(screen.getByText("Claude will be able to")).toBeTruthy();
    expect(screen.getByText(/Forget or export memories when you ask it to/)).toBeTruthy();
    expect(screen.getByText(/change your settings, plan or billing/)).toBeTruthy();
    expect(screen.getByText(/Free plan/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Technical details" }));
    expect(screen.getByText("S256")).toBeTruthy();
    expect(screen.getByText("memory")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Connect Claude" }));
    await waitFor(() => expect(m.oauthApprove).toHaveBeenCalled());
    expect((m.oauthApprove.mock.calls[0] as unknown as [{ state: string; code_challenge: string }])[0]).toMatchObject({ state: "s1", code_challenge: "abc" });
    expect(await screen.findByRole("status")).toBeTruthy();
    expect(screen.getByRole("status").textContent).toBe("Claude is connected. Returning you to claude.ai.");
    expect(identify("localhost:3000")).toEqual({ kind: "local", host: "localhost:3000" });
    expect(identify("evil.example")).toEqual({ kind: "unverified", host: "evil.example" });
    expect(identify("chatgpt.com")).toEqual({ kind: "known", publisher: "OpenAI", host: "chatgpt.com" });
  });

  it("warns on an unverified app and refuses a redirect the server did not register", async () => {
    m.oauthClientInfo.mockResolvedValueOnce({ client_id: "c2", client_name: "Thing", redirect_hosts: ["evil.example"], scopes: [] });
    window.history.pushState({}, "", "/admin/oauth/consent?client_id=c2&redirect_uri=https%3A%2F%2Fother.example%2Fcb&state=s");
    render(<OAuthConsent />);
    expect(await screen.findByText("Unverified app")).toBeTruthy();
    expect(screen.getByText(/Only approve a tool you set up yourself/)).toBeTruthy();
    expect((screen.getByRole("button", { name: "Deny" }) as HTMLButtonElement).disabled).toBe(true);
  });
});

describe("Onboarding (Refine PR 12)", () => {
  it("named steps, Next gated on the name, tiles to a workspace, the key behind a warning, no key for OAuth tools", async () => {
    render(<OnboardingSetup />);
    const next = screen.getByRole("button", { name: "Next" }) as HTMLButtonElement;
    expect(next.disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("Your name"), { target: { value: "Anthony" } });
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getByText("Where do you use AI?")).toBeTruthy();
    const tiles = within(screen.getByRole("group", { name: "Tools" }));
    fireEvent.click(tiles.getByRole("button", { name: "Claude Desktop" }));
    fireEvent.click(tiles.getByRole("button", { name: "Cursor" }));
    expect(tiles.getByRole("button", { name: "Cursor" }).getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(screen.getByRole("button", { name: "Create workspace" }));
    await waitFor(() => expect(m.signup).toHaveBeenCalledWith({ operator_name: "Anthony", tools: ["claude_desktop", "cursor"], model: "claude-sonnet-5" }));
    expect(await screen.findByText(/This key is shown once/)).toBeTruthy();
    expect(screen.getByText("crys_live_abcdef1234567890")).toBeTruthy();
    // Claude Desktop is first and signs in: no key in its snippet.
    expect(screen.getByText(/no key needed, sign in/)).toBeTruthy();
    expect(screen.getByText("https://api.erahq.ai/mcp")).toBeTruthy();
    fireEvent.click(screen.getByRole("tab", { name: "Cursor" }));
    expect(screen.getByText(/your key is inside/)).toBeTruthy();
    expect(screen.getByText(/crys_live_abcdef1234567890"/)).toBeTruthy();
    // Done steps are clickable: back to Name keeps the name.
    const steps = within(screen.getByRole("list", { name: "Steps" }));
    expect(steps.getByText("Connect").closest("li")?.getAttribute("aria-current")).toBe("step");
    fireEvent.click(steps.getByRole("button", { name: /Name/ }));
    expect((screen.getByLabelText("Your name") as HTMLInputElement).value).toBe("Anthony");
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    // Tools now steps forward instead of creating a second workspace.
    expect(screen.queryByRole("button", { name: "Create workspace" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getByText("Tell Crystal who it is remembering for")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Open my console" }));
    await waitFor(() => expect(auth.refreshMe).toHaveBeenCalled());
  });
});
