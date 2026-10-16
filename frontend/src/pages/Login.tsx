// Sign in — Console Refine PR 12 (docs/CONSOLE_UX_AUDIT.md 5.13). One card
// with Sign in / Create account as Segmented tabs (the email persists),
// provider buttons that show which popup is open while the others dim, a
// wrong-password message that offers the reset, a forgot-password state
// on Firebase's sendPasswordResetEmail, and the legal footer.
import { FormEvent, useState } from "react";
import { Loader2 } from "lucide-react";
import { useAuth } from "@/lib/auth";
import { Segmented } from "@/components/ui";
import { BTN_PRIMARY, BTN_SECONDARY, Brand, EntryShell, GitHubMark, GoogleMark, INPUT, LegalFooter } from "./entry";

type Mode = "signin" | "signup";
type View = Mode | "reset";

export function wrongPassword(code: string | null | undefined): boolean {
  return !!code && (code.includes("invalid-credential") || code.includes("wrong-password"));
}

export function Login() {
  const { signInGoogle, signInGitHub, signInEmail, signUpEmail, resetPassword, error, errorCode } = useAuth();
  const [view, setView] = useState<View>("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [popup, setPopup] = useState<"google" | "github" | null>(null);
  const [resetSent, setResetSent] = useState(false);
  const [resetError, setResetError] = useState<string | null>(null);

  const provider = async (which: "google" | "github") => {
    setPopup(which);
    try {
      if (which === "google") await signInGoogle(); else await signInGitHub();
    } finally {
      setPopup(null);
    }
  };
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    try {
      if (view === "signin") await signInEmail(email, password);
      else await signUpEmail(email, password);
    } finally {
      setBusy(false);
    }
  };
  const sendReset = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setResetError(null);
    try {
      await resetPassword(email);
      setResetSent(true);
    } catch {
      setResetError("Couldn't send the reset email. Check the address and try again.");
    } finally {
      setBusy(false);
    }
  };
  const anyBusy = busy || popup !== null;

  if (view === "reset") {
    return (
      <EntryShell>
        <Brand sub="Reset your password" />
        {resetSent ? (
          <p role="status" className="text-[13.5px] text-fg">We emailed a reset link to <b>{email}</b>. It works for 30 minutes.</p>
        ) : (
          <form onSubmit={sendReset} className="space-y-3">
            <p className="text-[13px] text-fg-2">We'll email a reset link. It works for 30 minutes.</p>
            <input type="email" required aria-label="Email" placeholder="you@company.com" value={email} onChange={(e) => setEmail(e.target.value)} className={INPUT} />
            {resetError && <p role="alert" className="rounded-md bg-bad-soft px-3 py-2 text-[12.5px] text-bad">{resetError}</p>}
            <button type="submit" disabled={anyBusy || !email} className={BTN_PRIMARY}>{busy ? "Sending…" : "Send reset link"}</button>
          </form>
        )}
        <button type="button" onClick={() => { setView("signin"); setResetSent(false); }} className={BTN_SECONDARY + " mt-3"}>Back to sign in</button>
        <p className="mt-4 text-[12px] text-fg-3">Signed up with Google or GitHub? Use that button instead; there is no password to reset.</p>
      </EntryShell>
    );
  }

  const mode: Mode = view;
  return (
    <EntryShell>
      <Brand sub={mode === "signin" ? "Sign in to your memory" : "Create your workspace"} />
      <Segmented label="Sign in or create account" className="mb-4 w-full" value={mode} onChange={(v) => setView(v)}
        options={[{ value: "signin", label: "Sign in" }, { value: "signup", label: "Create account" }]} />

      <div className="space-y-2">
        <button type="button" onClick={() => void provider("google")} disabled={anyBusy} className={BTN_SECONDARY}>
          {popup === "google" ? <Loader2 aria-hidden className="h-4 w-4 animate-spin" /> : <GoogleMark />}
          {popup === "google" ? "Opening Google…" : mode === "signin" ? "Continue with Google" : "Sign up with Google"}
        </button>
        <button type="button" onClick={() => void provider("github")} disabled={anyBusy} className={BTN_SECONDARY}>
          {popup === "github" ? <Loader2 aria-hidden className="h-4 w-4 animate-spin" /> : <GitHubMark />}
          {popup === "github" ? "Opening GitHub…" : mode === "signin" ? "Continue with GitHub" : "Sign up with GitHub"}
        </button>
      </div>

      <div className="my-4 flex items-center gap-3 text-[11px] uppercase tracking-[0.1em] text-fg-3">
        <span className="h-px flex-1 bg-line" />or<span className="h-px flex-1 bg-line" />
      </div>

      <form onSubmit={submit} className="space-y-3">
        <div>
          <label htmlFor="login-email" className="block text-[12.5px] font-medium text-fg-2">{mode === "signin" ? "Email" : "Work email"}</label>
          <input id="login-email" type="email" required autoComplete="email" placeholder="you@company.com" value={email} onChange={(e) => setEmail(e.target.value)} className={INPUT + " mt-1"} />
        </div>
        <div>
          <div className="flex items-center justify-between">
            <label htmlFor="login-password" className="text-[12.5px] font-medium text-fg-2">Password</label>
            {mode === "signin" && <button type="button" onClick={() => setView("reset")} className="text-[12px] text-accent hover:underline">Forgot password?</button>}
          </div>
          <input id="login-password" type="password" required minLength={6} autoComplete={mode === "signin" ? "current-password" : "new-password"} placeholder={mode === "signin" ? "Your password" : "At least 6 characters"}
            value={password} onChange={(e) => setPassword(e.target.value)} className={INPUT + " mt-1"} />
        </div>
        {mode === "signup" && <p className="text-[12px] text-fg-3">We'll send a link to confirm your email.</p>}
        {error && (
          <p role="alert" className="rounded-md bg-bad-soft px-3 py-2 text-[12.5px] text-bad">
            {wrongPassword(errorCode) && mode === "signin"
              ? <>That password doesn't match {email || "this email"}. Try again, or <button type="button" onClick={() => setView("reset")} className="underline">reset it</button>.</>
              : error}
          </p>
        )}
        <button type="submit" disabled={anyBusy} className={BTN_PRIMARY}>
          {busy ? (mode === "signin" ? "Signing in…" : "Creating…") : mode === "signin" ? "Sign in" : "Create account"}
        </button>
        {mode === "signup" && <p className="text-center text-[12px] text-fg-3">Free plan: 2,500 facts, no card needed.</p>}
      </form>
      <LegalFooter />
    </EntryShell>
  );
}
