// Settings › Connections (Refine PR 9, 5.8 item 3). The permanent home
// of the connect instructions (the wizard is signup-only), rendered from
// the same registry so the two cannot drift. Honest limits: the key is
// shown once at creation and stored as a hash, so the snippet carries a
// placeholder (no prefix is stored either); calls are not tagged with
// the tool that made them, so there is no per-tool list or Disconnect
// yet, only whether any tool has reached the memory.
import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Check, Copy } from "lucide-react";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { CONNECT_TOOLS } from "@/lib/connect-tools";
import { StatusPill } from "@/components/ui";
import { cn, fmtRelative } from "@/lib/utils";
import { Section } from "./shared";

export const KEY_PLACEHOLDER = "<your-api-key>";

export function toolLabel(id: string): string {
  return CONNECT_TOOLS.find((t) => t.id === id)?.label ?? id;
}

export function Connections() {
  const { me } = useAuth();
  const [activeId, setActiveId] = useState("claude_desktop");
  const [copied, setCopied] = useState(false);
  const active = CONNECT_TOOLS.find((t) => t.id === activeId) ?? CONNECT_TOOLS[0];
  const snippet = active.snippet(KEY_PLACEHOLDER);
  const isUser = me?.kind === "user";
  const status = useQuery({ queryKey: ["onboarding-status"], queryFn: () => api.onboardingStatus(), enabled: isUser, refetchInterval: 30_000 });

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(snippet);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch { /* the text stays selectable */ }
  };

  const said = (status.data?.ai_tools ?? []).filter(Boolean);

  return (
    <div className="space-y-4">
      <Section title="Connect a tool" help="Crystal speaks MCP, so any of these connects to the same memory.">
        <div role="tablist" aria-label="Tool" className="flex flex-wrap gap-1.5">
          {CONNECT_TOOLS.map((t) => (
            <button key={t.id} type="button" role="tab" aria-selected={t.id === active.id} onClick={() => setActiveId(t.id)}
              className={cn("rounded-md border px-2.5 py-1 text-[12.5px]",
                t.id === active.id ? "border-accent bg-accent-soft font-medium text-fg" : "border-line text-fg-2 hover:bg-inset hover:text-fg")}>
              {t.label}
            </button>
          ))}
        </div>
        <ol className="mt-4 space-y-2">
          {active.steps.map((s, i) => (
            <li key={i} className="flex gap-3 text-[13px] leading-relaxed text-fg">
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-accent-soft text-[11px] font-semibold text-accent">{i + 1}</span>
              <span className="min-w-0">{s}</span>
            </li>
          ))}
        </ol>
        <div className="mt-3 rounded-lg border border-line bg-inset p-3">
          <div className="mb-2 flex items-center justify-between gap-3">
            <span className="text-[11px] font-semibold uppercase tracking-[0.08em] text-fg-3">{active.label}</span>
            <button type="button" onClick={() => void copy()} className="inline-flex items-center gap-1.5 rounded-md border border-line px-2 py-0.5 text-[11.5px] text-fg-2 hover:bg-surface hover:text-fg">
              {copied ? <Check aria-hidden className="h-3.5 w-3.5" /> : <Copy aria-hidden className="h-3.5 w-3.5" />}
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
          <pre className="overflow-x-auto whitespace-pre-wrap break-all font-mono text-[12px] leading-relaxed text-fg">{snippet}</pre>
        </div>
        <p className="mt-2 text-[12.5px] text-fg-2">
          {active.auth === "key" ? (
            <>
              The snippet shows a placeholder: your key was shown once when the workspace was created and is not stored.
              Lost it? <Link to="/settings/api-key" className="text-accent hover:underline">Create a new one in API key</Link>; it replaces the old one in one step.
            </>
          ) : (
            <>No key needed: paste the URL and sign in with your Crystal account.</>
          )}
        </p>
      </Section>

      {isUser && (
        <Section title="Connected tools">
          {status.data?.connected ? (
            <p className="flex flex-wrap items-center gap-2 text-[13px] text-fg">
              <StatusPill variant="ok" dot>Connected</StatusPill>
              A tool last reached your memory {fmtRelative(status.data.last_seen_at)}.
            </p>
          ) : (
            <p className="text-[13px] text-fg-2">No tool has connected yet. Follow the steps above, then ask it “what do you remember about me?”</p>
          )}
          {said.length > 0 && (
            <p className="mt-2 text-[12.5px] text-fg-2">At signup you said you use {said.map(toolLabel).join(", ")}.</p>
          )}
          <p className="mt-2 text-[12px] text-fg-3">Calls are not yet tagged with the tool that made them, so there is no per-tool list or Disconnect here. A connection ends when its key is regenerated, or with your account.</p>
        </Section>
      )}
    </div>
  );
}
