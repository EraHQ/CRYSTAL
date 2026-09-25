// Connect a tool — the PERMANENT home of the connect instructions
// (L2-S5c follow-up, live-found 2026-09-25: the Getting Started
// checklist pointed returning users at Settings, and Settings had no
// such section; the wizard is signup-only by design). Renders the same
// registry the wizard uses (frontend/src/lib/connect-tools.ts), so the
// two can never drift. Key-based tools show a placeholder — the static
// key is revealed exactly once at workspace creation — while the
// Claude tabs need no key at all: URL plus sign-in.
import { useState } from "react";
import { Check, Copy } from "lucide-react";
import { CONNECT_TOOLS } from "@/lib/connect-tools";

const KEY_PLACEHOLDER = "<your-api-key>";

export function ConnectTools() {
  const [activeId, setActiveId] = useState("claude_desktop");
  const [copied, setCopied] = useState(false);
  const active =
    CONNECT_TOOLS.find((t) => t.id === activeId) ?? CONNECT_TOOLS[0];
  const snip = active.snippet(KEY_PLACEHOLDER);
  const needsKey = snip.includes(KEY_PLACEHOLDER);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(snip);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard denied — the text is selectable */
    }
  };

  return (
    <section className="rounded-xl border border-gray-200 bg-white p-5 shadow-card">
      <h2 className="text-sm font-semibold text-gray-900">Connect a tool</h2>
      <p className="mb-3 mt-1 text-[13px] text-gray-500">
        Crystal speaks MCP, so any of these connects to the same memory.
      </p>
      <div className="mb-4 flex flex-wrap gap-1.5">
        {CONNECT_TOOLS.map((t) => (
          <button
            key={t.id}
            onClick={() => setActiveId(t.id)}
            className={`rounded-lg px-2.5 py-1.5 text-[12px] transition ${
              t.id === active.id
                ? "bg-[#6f72f7] font-semibold text-gray-900"
                : "border border-[#ffffff26] text-gray-600 hover:bg-[#ffffff0d]"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
      <ol className="mb-3 space-y-2">
        {active.steps.map((s, i) => (
          <li
            key={i}
            className="flex gap-3 text-[13px] leading-relaxed text-gray-700"
          >
            <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[#6f72f7]/20 text-[11px] font-semibold text-[#8487fb]">
              {i + 1}
            </span>
            <span className="min-w-0">{s}</span>
          </li>
        ))}
      </ol>
      <div className="rounded-xl border border-[#ffffff1a] bg-[#0a0d15] p-4">
        <div className="mb-2 flex items-center justify-between gap-3">
          <span className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">
            {active.label}
          </span>
          <button
            onClick={() => void copy()}
            className="flex shrink-0 items-center gap-1.5 rounded-md border border-[#ffffff26] px-2.5 py-1 text-[11px] text-gray-600 hover:bg-[#ffffff0d]"
          >
            {copied ? (
              <Check className="h-3.5 w-3.5" />
            ) : (
              <Copy className="h-3.5 w-3.5" />
            )}
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
        <pre className="overflow-x-auto whitespace-pre-wrap break-all text-[12px] leading-relaxed text-gray-700">
          {snip}
        </pre>
      </div>
      {needsKey && (
        <p className="mt-2 text-[12px] text-gray-500">
          Replace {KEY_PLACEHOLDER} with your workspace key — it was shown
          once when the workspace was created. The Claude Desktop and
          Claude.ai tabs need no key: paste the URL and sign in.
        </p>
      )}
    </section>
  );
}
