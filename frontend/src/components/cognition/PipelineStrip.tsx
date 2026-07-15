// PipelineStrip — the signature element of the cognition tracker
// (2026-07-14, ratified Q2 B). One glance answers "where is this run
// right now": Orchestrate → Research → Compose → Validate as an
// animated strip. The active stage breathes (spring pulse), completed
// stages pop their check, and the connector toward the active stage
// carries a traveling shimmer. Respects prefers-reduced-motion.
import { motion, useReducedMotion } from "framer-motion";
import { Brain, Search, Zap, ShieldCheck } from "lucide-react";

const STAGES = [
  { key: "orchestrate", label: "Orchestrate", Icon: Brain },
  { key: "research", label: "Research", Icon: Search },
  { key: "compose", label: "Compose", Icon: Zap },
  { key: "validate", label: "Validate", Icon: ShieldCheck },
] as const;

const RETRIEVAL = new Set([
  "crystal_search", "crystal_key_scan", "web_search", "web_fetch",
  "research", "source_lookup",
]);

/** Map env status + step states onto the strip's active stage index.
 * -1 = nothing active (terminal states). */
export function activeStage(
  status: string,
  steps: Record<string, { action: string; status: string }>,
): number {
  if (status === "orchestrating") return 0;
  if (status === "validating") return 3;
  if (status === "working" || status === "rejected") {
    const running = Object.values(steps).find((s) => s.status === "running");
    if (running) return RETRIEVAL.has(running.action) ? 1 : 2;
    return 1;
  }
  return -1;
}

export function PipelineStrip({ status, steps }: {
  status: string;
  steps: Record<string, { action: string; status: string }>;
}) {
  const reduce = useReducedMotion();
  const active = activeStage(status, steps);
  const terminal = active === -1;
  const doneUpTo = terminal
    ? (status === "complete" ? STAGES.length : -1)
    : active;

  if (terminal && status !== "complete") return null;

  return (
    <div className="flex items-center gap-0 py-1" aria-label="run pipeline">
      {STAGES.map((stage, i) => {
        const isActive = i === active;
        const isDone = i < doneUpTo || (terminal && status === "complete");
        return (
          <div key={stage.key} className="flex items-center flex-1 last:flex-none">
            <div className="flex flex-col items-center gap-1">
              <motion.div
                className={`w-8 h-8 rounded-full flex items-center justify-center border
                  ${isDone ? "bg-green-50 border-green-300 text-green-600" :
                    isActive ? "bg-blue-50 border-blue-400 text-blue-600" :
                    "bg-gray-50 border-gray-200 text-gray-300"}`}
                animate={isActive && !reduce
                  ? { scale: [1, 1.12, 1] }
                  : { scale: 1 }}
                transition={isActive && !reduce
                  ? { duration: 1.6, repeat: Infinity, ease: "easeInOut" }
                  : { type: "spring", stiffness: 300, damping: 20 }}
              >
                <stage.Icon className="h-4 w-4" />
              </motion.div>
              <span className={`text-[10px] font-medium
                ${isDone ? "text-green-600" : isActive ? "text-blue-600" : "text-gray-300"}`}>
                {stage.label}
              </span>
            </div>
            {i < STAGES.length - 1 && (
              <div className="flex-1 h-0.5 mx-2 mb-4 rounded bg-gray-100 relative overflow-hidden">
                <div
                  className={`absolute inset-y-0 left-0 rounded transition-all duration-700
                    ${i < doneUpTo ? "bg-green-300 w-full" : "w-0"}`}
                />
                {i === active && !reduce && (
                  <motion.div
                    className="absolute inset-y-0 w-8 rounded bg-gradient-to-r from-transparent via-blue-300 to-transparent"
                    animate={{ x: ["-2rem", "100%"] }}
                    transition={{ duration: 1.4, repeat: Infinity, ease: "linear" }}
                  />
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
