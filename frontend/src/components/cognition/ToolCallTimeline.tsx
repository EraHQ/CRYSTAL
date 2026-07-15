// ToolCallTimeline (2026-07-14, Q1 visibility): what an agentic
// worker DID — every tool call with its input and the head of its
// output, staggered in on expand. This is the workers-as-CRYS trace
// (agentic.py attaches it to step output).
import { useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ChevronDown, ChevronUp, Wrench } from "lucide-react";

export interface ToolCall {
  tool: string;
  input: Record<string, unknown> | null;
  output_head: string;
  iteration: number | null;
}

function inputSummary(input: ToolCall["input"]): string {
  if (!input) return "";
  const parts: string[] = [];
  for (const [k, v] of Object.entries(input)) {
    if (Array.isArray(v)) parts.push(`${k}: ${v.slice(0, 3).join(" · ")}${v.length > 3 ? ` (+${v.length - 3})` : ""}`);
    else if (typeof v === "string") parts.push(`${k}: ${v.slice(0, 80)}`);
  }
  return parts.join("  |  ");
}

export function ToolCallTimeline({ calls }: { calls: ToolCall[] }) {
  const [open, setOpen] = useState(false);
  const [openCall, setOpenCall] = useState<number | null>(null);
  const reduce = useReducedMotion();
  if (!calls.length) return null;

  return (
    <div className="mt-1">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-[11px] text-indigo-600 hover:text-indigo-800"
      >
        <Wrench className="h-3 w-3" />
        {calls.length} tool call{calls.length === 1 ? "" : "s"}
        {open ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={reduce ? false : { opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="mt-1.5 space-y-1 border-l-2 border-indigo-100 pl-2.5">
              {calls.map((c, i) => (
                <motion.div
                  key={i}
                  initial={reduce ? false : { opacity: 0, x: -6 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: reduce ? 0 : i * 0.05 }}
                  className="text-[11px]"
                >
                  <button
                    className="flex items-center gap-1.5 text-left w-full"
                    onClick={() => setOpenCall(openCall === i ? null : i)}
                  >
                    <span className="text-[9px] font-semibold text-indigo-400 bg-indigo-50 rounded px-1">
                      {c.iteration ?? i + 1}
                    </span>
                    <span className="font-mono text-gray-700">{c.tool}</span>
                    <span className="text-gray-400 truncate flex-1">{inputSummary(c.input)}</span>
                  </button>
                  {openCall === i && c.output_head && (
                    <pre className="mt-1 bg-gray-50 border border-gray-100 rounded p-1.5 text-[10px] text-gray-600 whitespace-pre-wrap max-h-28 overflow-y-auto">
                      {c.output_head}
                    </pre>
                  )}
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
