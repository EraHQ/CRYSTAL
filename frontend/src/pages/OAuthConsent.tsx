// OAuth consent (L2-S5b, 2026-09-24) — the page Claude's "Add custom
// connector" flow lands on. Rendered by the Gate BEFORE the console
// shell, so signed-out users fall into the normal Login (URL and all
// OAuth params intact) and brand-new users get the wizard first.
// PALETTE NOTE: gray-900 is the brightest text; #ffffffXX for overlays.
import { useEffect, useMemo, useState } from "react";
import { ShieldCheck, X } from "lucide-react";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/auth";

export function OAuthConsent() {
  const { email, signOut } = useAuth();
  const [clientName, setClientName] = useState<string | null>(null);
  const [host, setHost] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const p = useMemo(
    () => new URLSearchParams(window.location.search),
    [],
  );
  const clientId = p.get("client_id") ?? "";
  const redirectUri = p.get("redirect_uri") ?? "";
  const state = p.get("state") ?? "";

  useEffect(() => {
    if (!clientId) return;
    api.oauthClientInfo(clientId)
      .then((c) => {
        setClientName(c.client_name);
        setHost(c.redirect_hosts[0] ?? "");
      })
      .catch(() => setError("This connection request is invalid or expired."));
  }, [clientId]);

  const approve = async () => {
    setBusy(true);
    setError(null);
    try {
      const out = await api.oauthApprove({
        client_id: clientId,
        redirect_uri: redirectUri,
        state,
        code_challenge: p.get("code_challenge") ?? "",
        scopes: p.get("scopes") ?? "",
        explicit_redirect: p.get("explicit_redirect") !== "0",
        resource: p.get("resource") ?? undefined,
      });
      window.location.assign(out.redirect_to);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not approve");
      setBusy(false);
    }
  };

  const deny = () => {
    const sep = redirectUri.includes("?") ? "&" : "?";
    const back = `${redirectUri}${sep}error=access_denied${
      state ? `&state=${encodeURIComponent(state)}` : ""
    }`;
    window.location.assign(back);
  };

  if (!clientId || !redirectUri) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#0b0e17] px-4">
        <p className="text-[13px] text-gray-500">
          This connection request is missing its parameters. Start again
          from your AI tool.
        </p>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-[#0b0e17] px-4 py-10">
      <div className="m-auto w-full max-w-md rounded-2xl border border-[#ffffff1a] bg-[#10131d] p-8">
        <div className="mb-5 flex h-12 w-12 items-center justify-center rounded-xl bg-[#6f72f7]/15">
          <ShieldCheck className="h-6 w-6 text-[#8487fb]" />
        </div>
        <h1 className="mb-1 text-[19px] font-semibold text-gray-900">
          {clientName ?? "An MCP client"} wants to connect
        </h1>
        <p className="mb-1 text-[13px] leading-relaxed text-gray-400">
          {clientName ?? "This client"}
          {host ? ` (${host})` : ""} is asking to use the Crystal memory
          belonging to <b className="text-gray-700">{email}</b>: recalling
          what you have stored and remembering new things as you work.
        </p>
        <button
          onClick={() => void signOut()}
          className="mb-4 block text-[12px] text-[#8487fb] hover:underline"
        >
          Not you? Switch account
        </button>
        <ul className="mb-5 space-y-1.5 text-[13px] text-gray-600">
          <li>Acts as your seat: everything it stores is owned by you</li>
          <li>Access lasts one hour at a time and renews while in use</li>
          <li>Revocable any time from Settings</li>
        </ul>
        {error && <p className="mb-3 text-[12px] text-red-400">{error}</p>}
        <div className="flex gap-2">
          <button
            disabled={busy}
            onClick={() => void approve()}
            className="flex-1 rounded-lg bg-[#6f72f7] px-4 py-2.5 text-[13px] font-semibold text-gray-900 transition hover:bg-[#5d60ee] disabled:opacity-40"
          >
            {busy ? "Connecting…" : "Approve"}
          </button>
          <button
            disabled={busy}
            onClick={deny}
            className="flex items-center gap-1.5 rounded-lg border border-[#ffffff26] px-4 py-2.5 text-[13px] text-gray-600 hover:bg-[#ffffff0d]"
          >
            <X className="h-4 w-4" /> Deny
          </button>
        </div>
      </div>
    </div>
  );
}
