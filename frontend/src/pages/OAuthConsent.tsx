// OAuth consent — Console Refine PR 12 (docs/CONSOLE_UX_AUDIT.md 5.15).
// The page Claude's "Add custom connector" flow lands on, rendered by the
// Gate before the console shell. Client identity first (a known
// publisher by redirect host, a local app, or an Unverified warning with
// the host in mono), the account row, what the tool will and will not
// be able to do, access duration, the technical details, and an outcome
// state before the browser leaves. What it CAN do is stated from the
// server's grant: the one scope is full access to the memory, which
// includes forgetting and exporting through MCP, so the page never
// claims otherwise. There is no per-tool Disconnect yet (no list or
// revoke route), so the ending line is the honest one.
import { useEffect, useMemo, useState } from "react";
import { ChevronDown, ShieldAlert, ShieldCheck, X } from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { StatusPill } from "@/components/ui";
import { planFor } from "@/lib/plans";
import { BTN_PRIMARY, BTN_SECONDARY, Brand, EntryShell } from "./entry";

const KNOWN: { match: RegExp; publisher: string }[] = [
  { match: /(^|\.)claude\.ai$|(^|\.)anthropic\.com$/i, publisher: "Anthropic" },
  { match: /(^|\.)chatgpt\.com$|(^|\.)openai\.com$/i, publisher: "OpenAI" },
  { match: /(^|\.)cursor\.(com|sh)$/i, publisher: "Cursor" },
];
const LOOPBACK = ["localhost", "127.0.0.1", "[::1]", "::1"];

export type Identity = { kind: "known"; publisher: string; host: string } | { kind: "local"; host: string } | { kind: "unverified"; host: string };

export function identify(host: string): Identity {
  const bare = host.replace(/:\d+$/, "");
  if (LOOPBACK.includes(bare)) return { kind: "local", host };
  const k = KNOWN.find((x) => x.match.test(bare));
  return k ? { kind: "known", publisher: k.publisher, host } : { kind: "unverified", host };
}

export function OAuthConsent() {
  const { email, signOut, me } = useAuth();
  const [clientName, setClientName] = useState<string | null>(null);
  const [host, setHost] = useState<string>("");
  const [allowedHosts, setAllowedHosts] = useState<string[] | null>(null);
  const [scopes, setScopes] = useState<string[]>([]);
  // B4-6 (Lockdown PR-7): the server states what the grant IS; the card
  // shows that statement verbatim above the plain-words list.
  const [grant, setGrant] = useState<string>(
    "Full access to this workspace's memory as your seat: it can read, store, and forget anything you can",
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<"approved" | "denied" | null>(null);
  const [details, setDetails] = useState(false);

  const p = useMemo(() => new URLSearchParams(window.location.search), []);
  const clientId = p.get("client_id") ?? "";
  const redirectUri = p.get("redirect_uri") ?? "";
  const state = p.get("state") ?? "";
  const pkce = (p.get("code_challenge") ?? "") !== "";

  useEffect(() => {
    if (!clientId) return;
    api.oauthClientInfo(clientId)
      .then((c) => {
        setClientName(c.client_name);
        setHost(c.redirect_hosts[0] ?? "");
        setAllowedHosts(c.redirect_hosts ?? []);
        setScopes(c.scopes ?? []);
        if (c.grant) setGrant(c.grant);
      })
      .catch(() => setError("This connection request is invalid or expired."));
  }, [clientId]);

  // B3 (security sweep 2026-09-30): Deny navigates only to a URI whose
  // scheme is safe and whose host the server confirmed for this client.
  const safeRedirect = useMemo(() => {
    if (!allowedHosts) return null;
    let u: URL;
    try { u = new URL(redirectUri); } catch { return null; }
    const loopback = LOOPBACK.includes(u.hostname);
    const schemeOk = u.protocol === "https:" || (u.protocol === "http:" && loopback);
    if (!schemeOk) return null;
    if (!allowedHosts.includes(u.host) && !allowedHosts.includes(u.hostname)) return null;
    return u;
  }, [allowedHosts, redirectUri]);

  const name = clientName ?? "This app";
  const who = useMemo(() => identify(host || safeRedirect?.host || ""), [host, safeRedirect]);
  const plan = planFor(me?.subscription_tier);
  const workspace = me?.customer_id ? `workspace ${me.customer_id}` : "your workspace";

  const approve = async () => {
    setBusy(true);
    setError(null);
    try {
      const out = await api.oauthApprove({
        client_id: clientId, redirect_uri: redirectUri, state,
        code_challenge: p.get("code_challenge") ?? "", scopes: p.get("scopes") ?? "",
        explicit_redirect: p.get("explicit_redirect") !== "0", resource: p.get("resource") ?? undefined,
      });
      setOutcome("approved");
      window.setTimeout(() => window.location.assign(out.redirect_to), 900);
    } catch (e) {
      setError(errorMessage(e, "Couldn't connect. Try again from your AI tool."));
      setBusy(false);
    }
  };
  const deny = () => {
    if (!safeRedirect) {
      setError("This connection request cannot be completed. Close this page and start again from your AI tool.");
      return;
    }
    const back = new URL(safeRedirect.toString());
    back.searchParams.set("error", "access_denied");
    if (state) back.searchParams.set("state", state);
    setOutcome("denied");
    window.setTimeout(() => window.location.assign(back.toString()), 900);
  };

  if (!clientId || !redirectUri) {
    return (
      <EntryShell>
        <Brand sub="Connect a tool" />
        <p className="text-[13.5px] text-fg-2">This connection request is missing its parameters. Start again from your AI tool.</p>
      </EntryShell>
    );
  }

  if (outcome) {
    return (
      <EntryShell>
        <Brand sub="Connect a tool" />
        <p role="status" className="text-[15px] font-medium text-fg">
          {outcome === "approved" ? `${name} is connected. Returning you to ${who.host || "your tool"}.` : `Not connected. Nothing was shared with ${name}.`}
        </p>
      </EntryShell>
    );
  }

  return (
    <EntryShell>
      <Brand sub="Connect a tool" />
      <section aria-label="App" className="rounded-lg border border-line bg-inset p-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[16px] font-semibold text-fg">{name}</span>
          {who.kind === "known" && <StatusPill variant="ok" dot>Known app</StatusPill>}
          {who.kind === "local" && <StatusPill variant="info">On this computer</StatusPill>}
          {who.kind === "unverified" && <StatusPill variant="warn" dot>Unverified app</StatusPill>}
        </div>
        <p className="mt-1 text-[12.5px] text-fg-2">
          {who.kind === "known" && <>by {who.publisher} · <span className="font-mono">{who.host}</span></>}
          {who.kind === "local" && <>Runs on this computer and returns to <span className="font-mono">{who.host}</span>.</>}
          {who.kind === "unverified" && <><ShieldAlert aria-hidden className="mr-1 inline h-3.5 w-3.5 text-warn" />Crystal can't vouch for it. It returns to <span className="font-mono">{who.host || "an unknown host"}</span>. Only approve a tool you set up yourself.</>}
        </p>
      </section>

      <section aria-label="Account" className="mt-3 flex flex-wrap items-center gap-x-2 gap-y-1 text-[13px]">
        <span className="text-fg">{email}</span>
        <span className="text-fg-3">· {workspace} · {plan.name} plan</span>
        <button type="button" onClick={() => void signOut()} className="text-accent hover:underline">Switch account</button>
      </section>

      <h2 className="mt-5 text-[13.5px] font-semibold text-fg">{name} will be able to</h2>
      <p className="mt-1 text-[12.5px] text-fg-3">{grant}.</p>
      <ul className="mt-1.5 space-y-1.5 text-[13px]">
        <li className="flex gap-2"><ShieldCheck aria-hidden className="mt-0.5 h-4 w-4 shrink-0 text-ok" /><span><span className="text-fg">Recall your memories.</span> <span className="text-fg-2">Search and read what you have saved, to answer you.</span></span></li>
        <li className="flex gap-2"><ShieldCheck aria-hidden className="mt-0.5 h-4 w-4 shrink-0 text-ok" /><span><span className="text-fg">Save new memories.</span> <span className="text-fg-2">Saved as you. You can edit or forget them later.</span></span></li>
        <li className="flex gap-2"><ShieldCheck aria-hidden className="mt-0.5 h-4 w-4 shrink-0 text-ok" /><span><span className="text-fg">Forget or export memories when you ask it to.</span> <span className="text-fg-2">The same things you can do in the console's memory.</span></span></li>
      </ul>
      <p className="mt-3 text-[13px] text-fg-2"><span className="font-medium text-fg">It cannot</span> change your settings, plan or billing, create or regenerate keys, or delete your account.</p>
      <p className="mt-3 text-[12.5px] text-fg-3">Access lasts one hour and renews while you use it. It ends when you delete your account in Settings; regenerating your API key does not affect it.</p>

      <button type="button" onClick={() => setDetails((d) => !d)} aria-expanded={details} className="mt-3 inline-flex items-center gap-1 text-[12.5px] text-fg-2 hover:text-fg">
        <ChevronDown aria-hidden className={"h-3.5 w-3.5 transition-transform " + (details ? "rotate-180" : "")} /> Technical details
      </button>
      {details && (
        <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 rounded-md border border-line bg-inset p-3 font-mono text-[11.5px] text-fg-2">
          <dt className="text-fg-3">client_id</dt><dd className="break-all">{clientId}</dd>
          <dt className="text-fg-3">redirect_uri</dt><dd className="break-all">{redirectUri}</dd>
          <dt className="text-fg-3">scope</dt><dd>{scopes.length ? scopes.join(" ") : (p.get("scopes") || "memory")}</dd>
          <dt className="text-fg-3">pkce</dt><dd>{pkce ? "S256" : "none"}</dd>
        </dl>
      )}

      {error && <p role="alert" className="mt-3 rounded-md bg-bad-soft px-3 py-2 text-[12.5px] text-bad">{error}</p>}
      <div className="mt-5 flex gap-2">
        <button type="button" disabled={busy || !clientName} onClick={() => void approve()} className={BTN_PRIMARY}>{busy ? "Connecting…" : `Connect ${name}`}</button>
        <button type="button" disabled={busy || !safeRedirect} onClick={deny} className={BTN_SECONDARY + " w-auto"}><X aria-hidden className="h-4 w-4" /> Deny</button>
      </div>
    </EntryShell>
  );
}
