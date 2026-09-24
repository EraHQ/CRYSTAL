// The frontend half of the error-envelope wire contract (live-found
// 2026-09-24; server half pinned in tests/test_error_envelope_contract.py).
// The bug this guards against: the console read FastAPI's default
// {"detail": ...} while the API speaks the OpenAI envelope
// {"error": {"message": ...}} — so every humane 402/403 message rendered
// as a generic error and the verify-gate inbox panel could never appear.
import { describe, expect, it } from "vitest";
import { errorMessageFrom } from "./api";

const ENVELOPE_403 = {
  error: {
    message:
      "Verify your email to finish creating your account: click the " +
      "link we sent to your inbox, then return here.",
    type: "permission_error",
    param: null,
    code: null,
  },
};

describe("errorMessageFrom", () => {
  it("reads the OpenAI envelope this API actually speaks", () => {
    expect(errorMessageFrom(ENVELOPE_403)).toContain("Verify your email");
  });

  it("still reads plain-FastAPI detail (self-host deployments)", () => {
    expect(errorMessageFrom({ detail: "Trial expired: writes paused" }))
      .toBe("Trial expired: writes paused");
  });

  it("feeds the onboarding verify-state detection end to end", () => {
    // OnboardingSetup switches to the inbox panel on this phrase — the
    // exact check that silently failed in prod on 2026-09-24.
    const msg = errorMessageFrom(ENVELOPE_403) ?? "";
    expect(msg.includes("Verify your email")).toBe(true);
  });

  it("reads the envelope for 402 capacity walls too", () => {
    expect(
      errorMessageFrom({
        error: {
          message: "Memory is full (550 crystals; your plan holds 500).",
          type: "api_error",
          param: null,
          code: null,
        },
      }),
    ).toContain("Memory is full");
  });

  it("returns undefined for shapes carrying no message", () => {
    expect(errorMessageFrom(null)).toBeUndefined();
    expect(errorMessageFrom({})).toBeUndefined();
    expect(errorMessageFrom({ error: {} })).toBeUndefined();
    expect(errorMessageFrom({ error: { message: 42 } })).toBeUndefined();
    expect(errorMessageFrom({ detail: ["not", "a", "string"] }))
      .toBeUndefined();
  });
});
