// Getting-started checklist (T2b, 2026-09-26): the console landing's
// bridge from onboarding to habit. Renders only for hosted tenant
// sessions, self-hides when every row is done or when dismissed
// (localStorage; reappearing after a browser change is acceptable).
import { useEffect, useState } from "react";
import { Check, X } from "lucide-react";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/auth";

const DISMISS_KEY = "cc.gettingstarted.dismissed";

export function GettingStarted() {
  const { me } = useAuth();
  const [connected, setConnected] = useState<boolean | null>(null);
  const [dismissed, setDismissed] = useState(
    () => localStorage.getItem(DISMISS_KEY) === "1",
  );

  useEffect(() => {
    if (dismissed || me?.kind !== "user") return;
    let cancelled = false;
    api
      .onboardingStatus()
      .then((s) => {
        if (!cancelled) setConnected(s.connected);
      })
      .catch(() => {
        /* the checklist must never surface errors */
      });
    return () => {
      cancelled = true;
    };
  }, [dismissed, me?.kind]);

  if (dismissed || me?.kind !== "user") return null;
  const crystals = me?.usage?.crystals_used ?? null;
  const hasCrystals = crystals != null && crystals > 0;
  // Everything done: get out of the way permanently.
  if (connected && hasCrystals) return null;
  if (connected === null && !me?.usage) return null;

  const rows: Array<{ done: boolean; label: string }> = [
    { done: true, label: "Create your workspace" },
    { done: connected === true, label: "Connect a tool: your snippets are in Settings" },
    { done: hasCrystals, label: "Store your first memory" },
    { done: false, label: 'Ask "what do you remember about me?" from any tool' },
  ];

  return (
    <div className="mx-4 mt-4 rounded-xl border border-[#ffffff1a] bg-[#10131d] p-4">
      <div className="mb-2.5 flex items-center justify-between">
        <span className="text-[13px] font-semibold text-gray-900">
          Getting started
        </span>
        <button
          onClick={() => {
            localStorage.setItem(DISMISS_KEY, "1");
            setDismissed(true);
          }}
          className="rounded p-1 text-gray-500 hover:bg-[#ffffff0d] hover:text-gray-600"
          aria-label="Dismiss"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
      <div className="flex flex-col gap-2">
        {rows.map((r) => (
          <div key={r.label} className="flex items-center gap-2.5 text-[13px]">
            {r.done ? (
              <Check className="h-4 w-4 shrink-0 text-emerald-400" />
            ) : (
              <span className="h-4 w-4 shrink-0 rounded border-[1.5px] border-[#6f72f7]" />
            )}
            <span className={r.done ? "text-gray-500 line-through" : "text-gray-700"}>
              {r.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
