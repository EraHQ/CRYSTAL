// First-run onboarding — Console Refine PR 12 (docs/CONSOLE_UX_AUDIT.md
// 5.14; flow from T2b 2026-09-26). One card with a named stepper (Name,
// Tools, Connect, About you; done steps are clickable), the verify gate
// as step zero of the same card that advances by itself, tiles that
// reflow to two columns under 720px, the one-time key behind a warning
// callout with a beforeunload guard while it is uncopied, OAuth tools
// (Claude Desktop, Claude.ai) shown without a key, the live first-contact
// strip, and Back / Next in one footer on every step.
import { FormEvent, useEffect, useRef, useState } from "react";
import { AlertTriangle, Check, Copy } from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { CONNECT_TOOLS } from "@/lib/connect-tools";
import { cn } from "@/lib/utils";
import { BTN_PRIMARY, BTN_SECONDARY, Brand, EntryShell, INPUT } from "./entry";

export const STEPS = ["Name", "Tools", "Connect", "About you"] as const;
type Step = 1 | 2 | 3 | 4;

export function Stepper({ step, done, onGo }: { step: Step; done: Step; onGo: (s: Step) => void }) {
  return (
    <ol aria-label="Steps" className="mb-5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[12px]">
      {STEPS.map((label, i) => {
        const n = (i + 1) as Step;
        const isDone = n < done || (n < step);
        const current = n === step;
        return (
          <li key={label} aria-current={current ? "step" : undefined} className="flex items-center gap-1.5">
            {isDone ? (
              <button type="button" onClick={() => onGo(n)} className="inline-flex items-center gap-1 text-fg-2 hover:text-fg">
                <span className="flex h-4 w-4 items-center justify-center rounded-full bg-ok-soft text-ok"><Check aria-hidden className="h-3 w-3" /></span>{label}
              </button>
            ) : (
              <span className={cn("inline-flex items-center gap-1", current ? "font-medium text-fg" : "text-fg-3")}>
                <span className={cn("flex h-4 w-4 items-center justify-center rounded-full text-[10px]", current ? "bg-accent text-on-accent" : "bg-inset text-fg-3")}>{n}</span>{label}
              </span>
            )}
            {i < STEPS.length - 1 && <span aria-hidden className="ml-1.5 h-px w-4 bg-line" />}
          </li>
        );
      })}
    </ol>
  );
}

export function OnboardingSetup() {
  const { email, refreshMe, signOut, resendVerification, reloadUser } = useAuth();
  const [step, setStep] = useState<Step>(1);
  const [reached, setReached] = useState<Step>(1);
  const [name, setName] = useState("");
  const [tools, setTools] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [needsVerify, setNeedsVerify] = useState(false);
  const [resent, setResent] = useState(false);
  // The one-time reveal, held in memory across the connect step.
  const [apiKey, setApiKey] = useState<string | null>(null);
  const [customerId, setCustomerId] = useState<string | null>(null);
  const [activeTool, setActiveTool] = useState<string>("");
  const [connected, setConnected] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const [keyCopied, setKeyCopied] = useState(false);
  const [who, setWho] = useState("");
  const [working, setWorking] = useState("");
  const [goal, setGoal] = useState("");
  const [seeded, setSeeded] = useState(false);
  const pollRef = useRef<number | null>(null);

  const go = (s: Step) => { setStep(s); setReached((r) => (s > r ? s : r)); };

  // T2a's first-contact signal: poll while the connect step is showing.
  useEffect(() => {
    if (step !== 3 || connected) return;
    const tick = async () => {
      try {
        const s = await api.onboardingStatus();
        if (s.connected) setConnected(true);
      } catch { /* polling must never surface errors */ }
    };
    void tick();
    pollRef.current = window.setInterval(() => void tick(), 3000);
    return () => { if (pollRef.current) window.clearInterval(pollRef.current); };
  }, [step, connected]);

  // 5.14 item 5: the key is shown once; leaving the page with it uncopied
  // asks first.
  useEffect(() => {
    if (step !== 3 || !apiKey || keyCopied) return;
    const warn = (e: BeforeUnloadEvent) => { e.preventDefault(); e.returnValue = ""; };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [step, apiKey, keyCopied]);

  const toggleTool = (id: string) => setTools((t) => (t.includes(id) ? t.filter((x) => x !== id) : [...t, id]));

  // Signup fires at the end of step 2: the seat gets its name, the
  // picker's tools persist, and the key comes back for the connect step.
  const provision = async () => {
    setBusy(true);
    setError(null);
    try {
      const out = await api.signup({ operator_name: name.trim(), tools, model: "claude-sonnet-5" });
      if (out.api_key) {
        setApiKey(out.api_key);
        setCustomerId(out.customer_id);
        setActiveTool(tools[0] ?? "other_mcp");
        setNeedsVerify(false);
        go(3);
      } else {
        await refreshMe(); // already provisioned: straight to the console
      }
    } catch (e) {
      const msg = errorMessage(e, "");
      if (msg.includes("Verify your email")) setNeedsVerify(true);
      else setError("Couldn't create your workspace. Please try again.");
    } finally {
      setBusy(false);
    }
  };

  // 5.14 item 2: the verify gate advances by itself once the link is clicked.
  useEffect(() => {
    if (!needsVerify) return;
    const t = window.setInterval(async () => {
      try {
        await reloadUser();
        await provision();
      } catch { /* keep waiting */ }
    }, 5000);
    return () => window.clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [needsVerify]);

  const seedAndEnter = async () => {
    setBusy(true);
    setError(null);
    try {
      const text = [
        who.trim() && `Who I am: ${who.trim()}`,
        working.trim() && `What I'm working on right now: ${working.trim()}`,
        goal.trim() && `What Crystal should never lose track of: ${goal.trim()}`,
      ].filter(Boolean).join("\n\n");
      if (text && customerId) {
        await api.createDocumentText(customerId, { text, label: "About me", scope: "personal", auto_crystallize: true });
        setSeeded(true);
      }
      await refreshMe();
    } catch {
      await refreshMe(); // seeding must never strand the user outside their console
    } finally {
      setBusy(false);
    }
  };

  const copy = async (what: string, value: string) => {
    try { await navigator.clipboard.writeText(value); } catch { /* selectable */ }
    setCopied(what);
    if (what === "key" || what === "snippet") setKeyCopied(true);
    setTimeout(() => setCopied(null), 1600);
  };

  const footer = (back: Step | null, next: { label: string; onClick: () => void; disabled?: boolean; secondary?: { label: string; onClick: () => void } }) => (
    <div className="mt-5 flex items-center justify-between gap-3 border-t border-line pt-4">
      <div>{back && <button type="button" onClick={() => go(back)} className="text-[13px] text-fg-2 hover:text-fg">Back</button>}</div>
      <div className="flex items-center gap-3">
        {next.secondary && <button type="button" onClick={next.secondary.onClick} className="text-[13px] text-fg-3 hover:text-fg">{next.secondary.label}</button>}
        <button type="button" disabled={next.disabled} onClick={next.onClick} className={BTN_PRIMARY + " w-auto px-6"}>{next.label}</button>
      </div>
    </div>
  );

  if (needsVerify) {
    return (
      <EntryShell>
        <Brand sub="Create your workspace" />
        <Stepper step={2} done={reached} onGo={(s) => { setNeedsVerify(false); go(s); }} />
        <h1 className="text-[19px] font-semibold text-fg">Check your inbox</h1>
        <p className="mt-1 text-[13.5px] text-fg-2">We sent a confirmation link to <b className="text-fg">{email}</b>. Open it on any device. This page continues by itself once you do.</p>
        {error && <p role="alert" className="mt-3 rounded-md bg-bad-soft px-3 py-2 text-[12.5px] text-bad">{error}</p>}
        <div className="mt-4 flex flex-wrap gap-2">
          <button type="button" disabled={busy} onClick={async () => { await reloadUser(); await provision(); }} className={BTN_PRIMARY + " w-auto"}>{busy ? "Checking…" : "I clicked the link"}</button>
          <button type="button" onClick={async () => { await resendVerification(); setResent(true); }} className={BTN_SECONDARY + " w-auto"}>{resent ? "Sent again" : "Resend email"}</button>
        </div>
        <button type="button" onClick={() => void signOut()} className="mt-4 text-[12.5px] text-fg-3 hover:text-fg">Wrong address? Use a different account</button>
      </EntryShell>
    );
  }

  if (step === 1) {
    const next = (e?: FormEvent) => { e?.preventDefault(); if (name.trim()) go(2); };
    return (
      <EntryShell>
        <Brand sub="Create your workspace" />
        <Stepper step={1} done={reached} onGo={go} />
        <form onSubmit={next}>
          <h1 className="text-[19px] font-semibold text-fg">Welcome. What's your name?</h1>
          <p className="mt-1 text-[13.5px] text-fg-2">Your tools will use it when they talk about you, and every memory you save is credited to it.</p>
          <label className="mt-4 block text-[12.5px] font-medium text-fg-2" htmlFor="opname">Your name</label>
          <input id="opname" value={name} onChange={(e) => setName(e.target.value)} placeholder={email ? email.split("@")[0] : "Your name"} className={INPUT + " mt-1"} autoFocus />
        </form>
        {footer(null, { label: "Next", onClick: () => next(), disabled: !name.trim() })}
      </EntryShell>
    );
  }

  if (step === 2) {
    return (
      <EntryShell wide>
        <Brand sub="Create your workspace" />
        <Stepper step={2} done={reached} onGo={go} />
        <h1 className="text-[19px] font-semibold text-fg">Where do you use AI?</h1>
        <p className="mt-1 text-[13.5px] text-fg-2">Pick everything you use. The next step gives you working setup for exactly these.</p>
        {error && <p role="alert" className="mt-3 rounded-md bg-bad-soft px-3 py-2 text-[12.5px] text-bad">{error}</p>}
        <div role="group" aria-label="Tools" className="mt-4 grid grid-cols-2 gap-2 min-[720px]:grid-cols-3">
          {CONNECT_TOOLS.map((t) => {
            const on = tools.includes(t.id);
            return (
              <button key={t.id} type="button" aria-pressed={on} onClick={() => toggleTool(t.id)}
                className={cn("flex min-w-0 items-center gap-2 rounded-lg border px-3 py-2.5 text-left text-[13px]",
                  on ? "border-accent bg-accent-soft font-medium text-fg" : "border-line bg-inset text-fg-2 hover:border-line-strong")}>
                {on ? <Check aria-hidden className="h-4 w-4 shrink-0 text-accent" /> : <span aria-hidden className="h-4 w-4 shrink-0 rounded border border-line-strong" />}
                <span className="min-w-0 break-words">{t.label}</span>
              </button>
            );
          })}
        </div>
        <p className="mt-3 text-[12px] text-fg-3">Crystal speaks MCP, the open protocol these tools share. If yours is not listed, Other MCP client gives you the universal config.</p>
        {footer(1, apiKey
          ? { label: "Next", onClick: () => go(3) }
          : { label: busy ? "Creating your workspace…" : "Create workspace", onClick: () => void provision(), disabled: busy })}
      </EntryShell>
    );
  }

  if (step === 3 && apiKey) {
    const tabs = CONNECT_TOOLS.filter((t) => (tools.length ? tools.includes(t.id) : t.id === "other_mcp"));
    const active = tabs.find((t) => t.id === activeTool) ?? tabs[0];
    const needsKey = active.auth === "key";
    const snip = active.snippet(apiKey);
    const first = tools.map((id) => CONNECT_TOOLS.find((t) => t.id === id)?.label).filter(Boolean)[0] ?? "your tool";
    return (
      <EntryShell wide>
        <Brand sub="Create your workspace" />
        <Stepper step={3} done={reached} onGo={go} />
        <h1 className="text-[19px] font-semibold text-fg">Connect Crystal to your tools</h1>
        <div role="alert" className="mt-3 rounded-lg border border-warn/40 bg-warn-soft p-3">
          <p className="flex items-start gap-2 text-[13px] text-fg"><AlertTriangle aria-hidden className="mt-0.5 h-4 w-4 shrink-0 text-warn" /><span>This key is shown once. Copy it before you leave this page. Lost it later? Create a new one in Settings › API key.</span></p>
          <div className="mt-2 flex items-center gap-2 rounded-md border border-line bg-surface px-3 py-2">
            <code className="min-w-0 flex-1 break-all font-mono text-[12px] text-fg">{apiKey}</code>
            <button type="button" onClick={() => void copy("key", apiKey)} aria-label="Copy key" className="shrink-0 rounded-md border border-line p-1.5 text-fg-2 hover:bg-inset hover:text-fg">
              {copied === "key" ? <Check aria-hidden className="h-4 w-4 text-ok" /> : <Copy aria-hidden className="h-4 w-4" />}
            </button>
          </div>
        </div>
        <div role="tablist" aria-label="Tool" className="mt-4 flex flex-wrap gap-1.5">
          {tabs.map((t) => (
            <button key={t.id} type="button" role="tab" aria-selected={t.id === active.id} onClick={() => setActiveTool(t.id)}
              className={cn("rounded-md border px-2.5 py-1 text-[12.5px]", t.id === active.id ? "border-accent bg-accent-soft font-medium text-fg" : "border-line text-fg-2 hover:bg-inset hover:text-fg")}>
              {t.label}
            </button>
          ))}
        </div>
        <ol className="mt-3 space-y-2">
          {active.steps.map((s, i) => (
            <li key={i} className="flex gap-3 text-[13px] leading-relaxed text-fg">
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-accent-soft text-[11px] font-semibold text-accent">{i + 1}</span>
              <span className="min-w-0">{s}</span>
            </li>
          ))}
        </ol>
        <div className="mt-3 rounded-lg border border-line bg-inset p-3">
          <div className="mb-2 flex items-center justify-between gap-3">
            <span className="text-[11px] font-semibold uppercase tracking-[0.08em] text-fg-3">{active.label}{needsKey ? " · your key is inside" : " · no key needed, sign in"}</span>
            <button type="button" onClick={() => void copy("snippet", snip)} className="inline-flex items-center gap-1.5 rounded-md border border-line px-2 py-0.5 text-[11.5px] text-fg-2 hover:bg-surface hover:text-fg">
              {copied === "snippet" ? <Check aria-hidden className="h-3.5 w-3.5" /> : <Copy aria-hidden className="h-3.5 w-3.5" />}{copied === "snippet" ? "Copied" : "Copy"}
            </button>
          </div>
          <pre className="overflow-x-auto whitespace-pre-wrap break-all font-mono text-[12px] leading-relaxed text-fg">{snip}</pre>
        </div>
        <div role="status" className={cn("mt-3 flex flex-wrap items-center justify-between gap-2 rounded-lg border px-3 py-2 text-[13px]", connected ? "border-ok/40 bg-ok-soft text-fg" : "border-accent/40 bg-accent-soft text-fg")}>
          <span className="flex items-center gap-2"><span aria-hidden className={cn("h-2 w-2 rounded-full", connected ? "bg-ok" : "animate-pulse bg-warn")} />{connected ? `Connected: Crystal just heard from ${first}.` : "Listening for your first tool call…"}</span>
          {!connected && <span className="text-[12px] text-fg-3">Try: “what do you remember about me?”</span>}
        </div>
        {footer(2, { label: "Next", onClick: () => go(4), secondary: connected ? undefined : { label: "I'll connect later", onClick: () => go(4) } })}
      </EntryShell>
    );
  }

  return (
    <EntryShell wide>
      <Brand sub="Create your workspace" />
      <Stepper step={4} done={reached} onGo={go} />
      {connected && <p className="mb-3 flex items-center gap-2 rounded-md border border-ok/40 bg-ok-soft px-3 py-1.5 text-[12.5px] text-fg"><Check aria-hidden className="h-4 w-4 text-ok" /> Connected: your tools are live.</p>}
      <h1 className="text-[19px] font-semibold text-fg">Tell Crystal who it is remembering for</h1>
      <p className="mt-1 text-[13.5px] text-fg-2">This goes through the same extraction as everything you will ever store: it becomes your first memories, and self-curation starts from them.</p>
      {[
        { id: "who", label: "Who are you?", v: who, set: setWho },
        { id: "working", label: "What are you working on right now?", v: working, set: setWorking },
        { id: "goal", label: "What should Crystal never lose track of?", v: goal, set: setGoal },
      ].map((f) => (
        <label key={f.id} className="mt-3 block text-[12.5px] font-medium text-fg-2" htmlFor={f.id}>{f.label}
          <textarea id={f.id} value={f.v} onChange={(e) => f.set(e.target.value)} rows={2} className={INPUT + " mt-1 resize-none"} />
        </label>
      ))}
      {seeded && <p role="status" className="mt-3 text-[12.5px] text-accent">Forming memories from what you shared…</p>}
      {footer(3, { label: busy ? "Saving…" : "Open my console", onClick: () => void seedAndEnter(), disabled: busy, secondary: { label: "Skip", onClick: () => void refreshMe() } })}
    </EntryShell>
  );
}
