// L6=B rider (boarded 2026-09-05, shipped in v49): surface the /health
// self_curation signal as a console banner. A keyless deployment serves
// memory but does not curate it — the single biggest expectation gap in
// the functionality audit. Admin/self-host surfaces only (tenants on the
// hosted platform always have a platform key behind them). Dismissal is
// session-scoped on purpose: the state is worth re-seeing tomorrow.
import { useEffect, useState } from "react";
import { AlertTriangle, X } from "lucide-react";

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
    <div className="flex items-center gap-3 border-b border-amber-300 bg-amber-50 px-4 py-2 text-sm text-amber-900">
      <AlertTriangle className="h-4 w-4 shrink-0" />
      <span className="min-w-0">
        <b>Self-curation is idle:</b> no internal model key is configured —
        contradiction/duplicate/gap scans, gap filling, and multi-segment
        keys are all skipping. Set <code>CC_ANTHROPIC_API_KEY</code> (or your
        provider&apos;s key) and restart to enable the part that makes this
        memory different.
      </span>
      <button
        onClick={() => setDismissed(true)}
        className="ml-auto rounded p-1 hover:bg-amber-100"
        aria-label="Dismiss"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  );
}
