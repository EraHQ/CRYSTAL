// Hosted sign-in (Accounts Phase C). Rendered only when Firebase is
// configured and there is no session. Google one-tap-style button +
// email/password with a sign-in / create-account toggle.
import { useState } from "react";
import { useAuth } from "@/lib/auth";

function GoogleMark() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 48 48">
      <path fill="#FFC107" d="M43.6 20.1H42V20H24v8h11.3C33.7 32.7 29.2 36 24 36c-6.6 0-12-5.4-12-12s5.4-12 12-12c3.1 0 5.9 1.2 8 3l5.7-5.7C34.3 6.1 29.4 4 24 4 13 4 4 13 4 24s9 20 20 20 20-9 20-20c0-1.3-.1-2.6-.4-3.9z"/>
      <path fill="#FF3D00" d="M6.3 14.7l6.6 4.8C14.6 15.1 18.9 12 24 12c3.1 0 5.9 1.2 8 3l5.7-5.7C34.3 6.1 29.4 4 24 4 16.3 4 9.7 8.3 6.3 14.7z"/>
      <path fill="#4CAF50" d="M24 44c5.2 0 9.9-2 13.4-5.2l-6.2-5.2C29.2 35.1 26.7 36 24 36c-5.2 0-9.6-3.3-11.3-8l-6.5 5C9.5 39.6 16.2 44 24 44z"/>
      <path fill="#1976D2" d="M43.6 20.1H42V20H24v8h11.3c-.8 2.2-2.2 4.2-4.1 5.6l6.2 5.2C41 35.4 44 30.2 44 24c0-1.3-.1-2.6-.4-3.9z"/>
    </svg>
  );
}

function GitHubMark() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 16 16" fill="currentColor">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
    </svg>
  );
}

export function Login() {
  const { signInGoogle, signInGitHub, signInEmail, signUpEmail, error } = useAuth();
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    try {
      if (mode === "signin") await signInEmail(email, password);
      else await signUpEmail(email, password);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex h-screen items-center justify-center bg-[#0b0e17]">
      <div className="w-full max-w-sm rounded-2xl border border-white/10 bg-[#10131d] p-8 shadow-2xl">
        <div className="mb-6 flex items-center gap-3">
          <svg className="h-8 w-8" viewBox="0 0 20 20" fill="none">
            <path d="M10 2L17 7V13L10 18L3 13V7L10 2Z" fill="#6f72f7" />
            <path d="M10 2L17 7L10 10L3 7L10 2Z" fill="#8487fb" />
            <path d="M10 10L17 7V13L10 18V10Z" fill="#5d60ee" />
          </svg>
          <div>
            <p className="text-[15px] font-semibold text-white">CRYSTAL</p>
            <p className="text-[11px] uppercase tracking-[0.14em] text-gray-500">
              {mode === "signin" ? "Sign in" : "Create your account"}
            </p>
          </div>
        </div>

        <button
          onClick={() => void signInGoogle()}
          className="mb-4 flex w-full items-center justify-center gap-2.5 rounded-lg border border-white/15 bg-white px-4 py-2.5 text-[13px] font-medium text-gray-800 transition hover:bg-gray-100"
        >
          <GoogleMark />
          Continue with Google
        </button>

        <button
          onClick={() => void signInGitHub()}
          className="mb-4 flex w-full items-center justify-center gap-2.5 rounded-lg border border-white/15 bg-[#1b1f2a] px-4 py-2.5 text-[13px] font-medium text-gray-200 transition hover:bg-[#242938]"
        >
          <GitHubMark />
          Continue with GitHub
        </button>

        <div className="mb-4 flex items-center gap-3 text-[11px] uppercase tracking-wider text-gray-600">
          <span className="h-px flex-1 bg-white/10" />
          or
          <span className="h-px flex-1 bg-white/10" />
        </div>

        <form onSubmit={submit} className="space-y-3">
          <input
            type="email"
            required
            placeholder="you@company.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full rounded-lg border border-white/10 bg-[#0b0e17] px-3.5 py-2.5 text-[13px] text-gray-200 placeholder-gray-600 outline-none focus:border-[#6f72f7]"
          />
          <input
            type="password"
            required
            minLength={6}
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded-lg border border-white/10 bg-[#0b0e17] px-3.5 py-2.5 text-[13px] text-gray-200 placeholder-gray-600 outline-none focus:border-[#6f72f7]"
          />
          {error && (
            <p className="rounded-lg bg-red-500/10 px-3 py-2 text-[12px] text-red-400">
              {error}
            </p>
          )}
          <button
            type="submit"
            disabled={busy}
            className="w-full rounded-lg bg-[#6f72f7] px-4 py-2.5 text-[13px] font-semibold text-white transition hover:bg-[#5d60ee] disabled:opacity-50"
          >
            {mode === "signin" ? "Sign in" : "Create account"}
          </button>
        </form>

        <p className="mt-5 text-center text-[12px] text-gray-500">
          {mode === "signin" ? (
            <>
              New to CRYSTAL?{" "}
              <button
                className="font-medium text-[#8487fb] hover:underline"
                onClick={() => setMode("signup")}
              >
                Create an account
              </button>
            </>
          ) : (
            <>
              Already have an account?{" "}
              <button
                className="font-medium text-[#8487fb] hover:underline"
                onClick={() => setMode("signin")}
              >
                Sign in
              </button>
            </>
          )}
        </p>
      </div>
    </div>
  );
}
