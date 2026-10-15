import { KeyboardEvent, useRef } from "react";
import { cn } from "@/lib/utils";

export interface SegmentedOption<V extends string> { value: V; label: string }

// Part 3: a labeled set of options with exactly one selected. Arrow keys
// move the selection (radiogroup semantics), so it works without a mouse.
export function Segmented<V extends string>({ options, value, onChange, label, className, size = "md" }: {
  options: SegmentedOption<V>[];
  value: V;
  onChange: (v: V) => void;
  label: string;
  className?: string;
  size?: "sm" | "md";
}) {
  const refs = useRef<Array<HTMLButtonElement | null>>([]);
  const onKey = (e: KeyboardEvent<HTMLButtonElement>, idx: number) => {
    const delta = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
    if (!delta) return;
    e.preventDefault();
    const next = (idx + delta + options.length) % options.length;
    onChange(options[next].value);
    refs.current[next]?.focus();
  };
  return (
    <div role="radiogroup" aria-label={label} className={cn("inline-flex rounded-lg border border-line bg-inset p-0.5", className)}>
      {options.map((o, i) => {
        const on = o.value === value;
        return (
          <button key={o.value} type="button" role="radio" aria-checked={on} tabIndex={on ? 0 : -1}
            ref={(el) => { refs.current[i] = el; }}
            onClick={() => onChange(o.value)} onKeyDown={(e) => onKey(e, i)}
            className={cn(
              "rounded-md font-medium transition-colors",
              size === "sm" ? "px-2 py-0.5 text-[12px]" : "px-2.5 py-1 text-[12.5px]",
              on ? "bg-surface text-fg shadow-card" : "text-fg-2 hover:text-fg",
            )}>
            {o.label}
          </button>
        );
      })}
    </div>
  );
}
