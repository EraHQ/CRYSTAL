// L6=B rider (boarded 2026-09-05, shipped in v49): surface the /health
// self_curation signal as a console banner. A keyless deployment serves
// memory but does not curate it. Admin/self-host surfaces only (tenants
// on the hosted platform always have a platform key behind them).
// Dismissal is session-scoped on purpose: the state is worth re-seeing
// tomorrow. Console Refine PR 3 (4.2): one line, the spec's copy, tokens.
import { useEffect, useState } from "react";
import { AlertTriangle, X } from "lucide-react";
import { IconButton } from "@/components/ui";

export function SelfCurationBanner() {
  const [idle, setIdle] = useState(false);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch("/health")
      .then((r) => (r.ok ? r.json() : null))
      .then((body) => {
        if (!cancelled && body?.self_curation === "idle_no_model_key") {
          setIdle(true);
        }
      })
      .catch(() => {
        /* liveness probe hiccups must not break the console */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (!idle || dismissed) return null;
  return (
    <div role="status" className="flex items-center gap-3 border-b border-warn/40 bg-warn-soft px-4 py-2 text-[13px] text-fg">
      <AlertTriangle aria-hidden className="h-4 w-4 shrink-0 text-warn" />
      <span className="min-w-0 truncate">
        Self-curation is idle. Set <code className="font-mono text-[12px]">CC_ANTHROPIC_API_KEY</code> on the server to turn it on.
      </span>
      <span className="ml-auto"><IconButton label="Dismiss" icon={X} size="sm" onClick={() => setDismissed(true)} /></span>
    </div>
  );
}
