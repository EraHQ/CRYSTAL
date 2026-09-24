// First-run onboarding wizard (T2b, 2026-09-26 — supersedes the
// single-form Phase C version; mockup ratified 2026-09-25 on the Design
// canvas). Flow: verify (server-gated) → name → environment → signup →
// connect (key baked into per-tool snippets, live first-contact poll) →
// about-you (a real document through extraction) → console.
//
// PALETTE NOTE (2026-09-26, live-found): tailwind.config REMAPS the
// semantic scale for the dark theme — "white" IS the card surface
// (#151823) and gray-900 is the BRIGHTEST text. Text therefore uses
// gray-900/700/600/500/400 (bright → muted); faint light overlays and
// borders use arbitrary #ffffffXX values, never white/NN.
import { useEffect, useRef, useState } from "react";
import { Check, Copy } from "lucide-react";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { CONNECT_TOOLS } from "@/lib/connect-tools";

function StepDots({ n }: { n: number }) {
  return (
    <div className="flex items-center justify-between">
      <div className="text-[11px] tracking-[0.12em] text-gray-500">
        STEP {n} OF 4
      </div>
      <div className="flex gap-1.5">
        {[1, 2, 3, 4].map((i) => (
          <span
            key={i}
            className={`h-1 w-[22px] rounded-full ${i <= n ? "bg-[#6f72f7]" : "bg-[#ffffff1a]"}`}
          />
        ))}
      </div>
    </div>
  );
}

function Shell({ wide, children }: { wide?: boolean; children: React.ReactNode }) {
  return (
    <div className="flex h-screen items-center justify-center overflow-y-auto bg-[#0b0e17]">
      <div
        className={`w-full ${wide ? "max-w-2xl" : "max-w-md"} rounded-2xl border border-[#ffffff1a] bg-[#10131d] p-8`}
      >
        {children}
      </div>
    </div>
  );
}

export function OnboardingSetup() {
  const { email, refreshMe, signOut, resendVerification, reloadUser } = useAuth();
  const [step, setStep] = useState<1 | 2 | 3 | 4>(1);
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
  const [who, setWho] = useState("");
  const [working, setWorking] = useState("");
  const [goal, setGoal] = useState("");
  const [seeded, setSeeded] = useState(false);
  const pollRef = useRef<number | null>(null);

  // T2a's first-contact signal: poll while the connect step is showing.
  useEffect(() => {
    if (step !== 3 || connected) return;
    const tick = async () => {
      try {
        const s = await api.onboardingStatus();
        if (s.connected) setConnected(true);
      } catch {
        /* polling must never surface errors */
      }
    };
    void tick();
    pollRef.current = window.setInterval(() => void tick(), 3000);
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, [step, connected]);

  const toggleTool = (id: string) =>
    setTools((t) => (t.includes(id) ? t.filter((x) => x !== id) : [...t, id]));

  // Signup fires at the end of step 2: the seat gets its name, the
  // picker's tools persist, and the key comes back for the connect step.
  const provision = async () => {
    setBusy(true);
    setError(null);
    try {
      const out = await api.signup({
        operator_name: name.trim(),
        tools,
        model: "claude-sonnet-5",
      });
      if (out.api_key) {
        setApiKey(out.api_key);
        setCustomerId(out.customer_id);
        setActiveTool(tools[0] ?? "other_mcp");
        setStep(3);
      } else {
        await refreshMe(); // already provisioned: straight to the console
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : "";
      if (msg.includes("Verify your email")) {
        setNeedsVerify(true);
      } else {
        setError("Could not create your workspace. Please try again.");
      }
    } finally {
      setBusy(false);
    }
  };

  const seedAndEnter = async () => {
    setBusy(true);
    setError(null);
    try {
      const text = [
        who.trim() && `Who I am: ${who.trim()}`,
        working.trim() && `What I'm working on right now: ${working.trim()}`,
        goal.trim() && `What Crystal should never lose track of: ${goal.trim()}`,
      ]
        .filter(Boolean)
        .join("\n\n");
      if (text && customerId) {
        await api.createDocumentText(customerId, {
          text,
          label: "About me",
          scope: "personal",
          auto_crystallize: true,
        });
        setSeeded(true);
      }
      await refreshMe();
    } catch {
      // Seeding must never strand the user outside their console.
      await refreshMe();
    } finally {
      setBusy(false);
    }
  };

  const copy = async (what: string, value: string) => {
    await navigator.clipboard.writeText(value);
    setCopied(what);
    setTimeout(() => setCopied(null), 1600);
  };

  if (needsVerify) {
    return (
      <Shell>
        <h1 className="mb-1 text-[17px] font-semibold text-gray-900">
          Check your inbox
        </h1>
        <p className="mb-5 text-[13px] leading-relaxed text-gray-400">
          We sent a verification link to{" "}
          <b className="text-gray-700">{email}</b>. Click it, then come back
          here — your workspace is created the moment your email is verified.
        </p>
        <div className="flex gap-2">
          <button
            className="flex-1 rounded-lg bg-[#6f72f7] px-3 py-2.5 text-[13px] font-semibold text-gray-900 hover:bg-[#5d60ee]"
            onClick={async () => {
              await reloadUser();
              setNeedsVerify(false);
              await provision();
            }}
          >
            I clicked the link — continue
          </button>
          <button
            className="rounded-lg border border-[#ffffff1a] px-3 py-2.5 text-[13px] text-gray-600 hover:bg-[#ffffff0d]"
            onClick={async () => {
              await resendVerification();
              setResent(true);
            }}
          >
            {resent ? "Sent again" : "Resend email"}
          </button>
        </div>
        <button
          className="mt-4 text-[12px] text-gray-500 hover:text-gray-600"
          onClick={() => void signOut()}
        >
          Use a different account
        </button>
      </Shell>
    );
  }

  if (step === 1) {
    return (
      <Shell>
        <div className="mb-5"><StepDots n={1} /></div>
        <h1 className="mb-1 text-[19px] font-semibold text-gray-900">
          Welcome. Who is remembering?
        </h1>
        <p className="mb-5 text-[13px] leading-relaxed text-gray-400">
          Every memory in Crystal has an owner. This names your seat, and it
          is how your crystals are attributed from the very first one.
        </p>
        <label className="mb-1.5 block text-[12px] text-gray-400" htmlFor="opname">
          Your name
        </label>
        <input
          id="opname"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder={email ? email.split("@")[0] : "Your name"}
          className="mb-5 w-full rounded-lg border border-[#ffffff26] bg-[#0d1019] px-3.5 py-2.5 text-[14px] text-gray-900 outline-none focus:border-[#6f72f7]"
        />
        <button
          disabled={!name.trim()}
          onClick={() => setStep(2)}
          className="w-full rounded-lg bg-[#6f72f7] px-4 py-2.5 text-[13px] font-semibold text-gray-900 transition hover:bg-[#5d60ee] disabled:opacity-40"
        >
          Continue
        </button>
      </Shell>
    );
  }

  if (step === 2) {
    return (
      <Shell wide>
        <div className="mb-5"><StepDots n={2} /></div>
        <h1 className="mb-1 text-[19px] font-semibold text-gray-900">
          Where do you work with AI?
        </h1>
        <p className="mb-5 text-[13px] leading-relaxed text-gray-400">
          Pick everything you use. The next step gives you working setup for
          exactly these, nothing generic.
        </p>
        {error && <p className="mb-3 text-[12px] text-red-400">{error}</p>}
        <div className="mb-3 grid grid-cols-3 gap-2.5">
          {CONNECT_TOOLS.map((t) => {
            const on = tools.includes(t.id);
            return (
              <button
                key={t.id}
                onClick={() => toggleTool(t.id)}
                className={`flex items-center gap-2.5 rounded-xl border px-3.5 py-3 text-left text-[13px] transition ${
                  on
                    ? "border-[#6f72f7] bg-[#6f72f7]/10 font-semibold text-gray-900"
                    : "border-[#ffffff26] bg-[#0d1019] text-gray-600 hover:border-[#ffffff40]"
                }`}
              >
                {on ? (
                  <Check className="h-4 w-4 shrink-0 text-[#8487fb]" />
                ) : (
                  <span className="h-4 w-4 shrink-0 rounded border-[1.5px] border-gray-300" />
                )}
                {t.label}
              </button>
            );
          })}
        </div>
        <p className="mb-5 text-[11px] text-gray-500">
          Crystal speaks MCP, the open protocol these tools share. If yours is
          not listed, Other MCP client gives you the universal config.
        </p>
        <div className="flex items-center justify-between">
          <button
            className="text-[12px] text-gray-500 hover:text-gray-600"
            onClick={() => setStep(1)}
          >
            Back
          </button>
          <button
            disabled={busy}
            onClick={() => void provision()}
            className="rounded-lg bg-[#6f72f7] px-7 py-2.5 text-[13px] font-semibold text-gray-900 transition hover:bg-[#5d60ee] disabled:opacity-40"
          >
            {busy ? "Creating your workspace…" : "Continue"}
          </button>
        </div>
      </Shell>
    );
  }

  if (step === 3 && apiKey) {
    const tabs = CONNECT_TOOLS.filter((t) =>
      tools.length ? tools.includes(t.id) : t.id === "other_mcp",
    );
    const active = tabs.find((t) => t.id === activeTool) ?? tabs[0];
    const snip = active.snippet(apiKey);
    return (
      <Shell wide>
        <div className="mb-5"><StepDots n={3} /></div>
        <h1 className="mb-1 text-[19px] font-semibold text-gray-900">
          Connect Crystal to your tools
        </h1>
        <p className="mb-4 text-[13px] leading-relaxed text-gray-400">
          Your key is already inside these snippets. It is shown{" "}
          <span className="font-semibold text-gray-700">only this once</span>,
          so finish this step now (or copy the raw key below). Paste, restart
          the tool, and this screen notices the moment Crystal hears from it.
        </p>
        <div className="mb-3 flex flex-wrap gap-1.5">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setActiveTool(t.id)}
              className={`rounded-lg px-3.5 py-1.5 text-[12px] transition ${
                t.id === active.id
                  ? "bg-[#6f72f7] font-semibold text-gray-900"
                  : "border border-[#ffffff26] text-gray-400 hover:text-gray-700"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div className="mb-3 rounded-xl border border-[#ffffff1a] bg-[#0a0d15] p-4">
          <div className="mb-2 flex items-center justify-between gap-3">
            <span className="min-w-0 truncate text-[11px] text-gray-500">
              {active.pasteLine}
            </span>
            <button
              onClick={() => void copy("snippet", snip)}
              className="flex shrink-0 items-center gap-1.5 rounded-md border border-[#ffffff26] px-2.5 py-1 text-[11px] text-gray-600 hover:bg-[#ffffff0d]"
            >
              {copied === "snippet" ? (
                <Check className="h-3 w-3 text-emerald-400" />
              ) : (
                <Copy className="h-3 w-3" />
              )}
              Copy
            </button>
          </div>
          <pre className="overflow-x-auto whitespace-pre-wrap text-[12px] leading-relaxed text-indigo-300">
            {snip}
          </pre>
        </div>
        <div className="mb-3 flex items-center gap-2 rounded-lg border border-[#ffffff1a] bg-[#0b0e17] px-3 py-2">
          <code className="min-w-0 flex-1 truncate text-[11px] text-emerald-400">
            {apiKey}
          </code>
          <button
            onClick={() => void copy("key", apiKey)}
            className="shrink-0 rounded-md p-1.5 text-gray-400 hover:bg-[#ffffff1a] hover:text-gray-900"
            title="Copy raw key"
          >
            {copied === "key" ? (
              <Check className="h-4 w-4 text-emerald-400" />
            ) : (
              <Copy className="h-4 w-4" />
            )}
          </button>
        </div>
        <div
          className={`mb-5 flex items-center justify-between rounded-xl border px-4 py-3 ${
            connected
              ? "border-emerald-500/40 bg-emerald-500/10"
              : "border-[#6f72f7]/30 bg-[#6f72f7]/10"
          }`}
        >
          <div className="flex items-center gap-2.5 text-[13px]">
            <span
              className={`h-2.5 w-2.5 rounded-full ${connected ? "bg-emerald-400" : "animate-pulse bg-amber-400"}`}
            />
            <span className={connected ? "text-emerald-700" : "text-gray-700"}>
              {connected
                ? "Connected — Crystal just heard from your tools."
                : "Listening for your first tool call…"}
            </span>
          </div>
          {!connected && (
            <span className="text-[11px] text-gray-500">
              Try: "what do you remember about me?"
            </span>
          )}
        </div>
        <div className="flex items-center justify-between">
          <button
            className="text-[12px] text-gray-500 hover:text-gray-600"
            onClick={() => setStep(4)}
          >
            I'll connect later
          </button>
          <button
            onClick={() => setStep(4)}
            className={`rounded-lg px-7 py-2.5 text-[13px] font-semibold transition ${
              connected
                ? "bg-[#6f72f7] text-gray-900 hover:bg-[#5d60ee]"
                : "bg-[#ffffff1a] text-gray-400 hover:bg-[#ffffff26]"
            }`}
          >
            Continue
          </button>
        </div>
      </Shell>
    );
  }

  // Step 4 — About you (seeds first crystals through real extraction).
  return (
    <Shell wide>
      <div className="mb-5"><StepDots n={4} /></div>
      {connected && (
        <div className="mb-4 flex items-center gap-2 rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-3.5 py-2 text-[12px] text-emerald-700">
          <Check className="h-4 w-4" /> Connected — your tools are live.
        </div>
      )}
      <h1 className="mb-1 text-[19px] font-semibold text-gray-900">
        Tell Crystal who it is remembering for
      </h1>
      <p className="mb-4 text-[13px] leading-relaxed text-gray-400">
        This goes through the same extraction pipeline as everything you will
        ever store: it becomes your first crystals, and self-curation starts
        working from them immediately.
      </p>
      {[
        { id: "who", label: "Who are you?", v: who, set: setWho },
        { id: "working", label: "What are you working on right now?", v: working, set: setWorking },
        { id: "goal", label: "What should Crystal help you never lose track of?", v: goal, set: setGoal },
      ].map((f) => (
        <div key={f.id} className="mb-3">
          <label className="mb-1 block text-[12px] text-gray-400" htmlFor={f.id}>
            {f.label}
          </label>
          <textarea
            id={f.id}
            value={f.v}
            onChange={(e) => f.set(e.target.value)}
            className="h-[64px] w-full resize-none rounded-lg border border-[#ffffff26] bg-[#0d1019] px-3.5 py-2.5 text-[13px] text-gray-900 outline-none focus:border-[#6f72f7]"
          />
        </div>
      ))}
      {seeded && (
        <p className="mb-3 text-[12px] text-[#8487fb]">
          Forming crystals from what you shared…
        </p>
      )}
      <div className="flex items-center justify-between">
        <button
          className="text-[12px] text-gray-500 hover:text-gray-600"
          onClick={() => void refreshMe()}
        >
          Skip
        </button>
        <button
          disabled={busy}
          onClick={() => void seedAndEnter()}
          className="rounded-lg bg-[#6f72f7] px-7 py-2.5 text-[13px] font-semibold text-gray-900 transition hover:bg-[#5d60ee] disabled:opacity-40"
        >
          {busy ? "Saving…" : "Open my console"}
        </button>
      </div>
    </Shell>
  );
}
