import { cn } from "@/lib/utils";

// Refine PR 6: a labeled on/off control (role=switch). The label is the
// accessible name; `hint` is the one line under it that says what the
// switch does right now.
export function Switch({ checked, onChange, label, hint, disabled = false, className }: {
  checked: boolean;
  onChange: (next: boolean) => void;
  label: string;
  hint?: string;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <label className={cn("flex items-start gap-2.5", disabled && "opacity-60", className)}>
      <button type="button" role="switch" aria-checked={checked} aria-label={label} disabled={disabled}
        onClick={() => onChange(!checked)}
        className={cn("relative mt-0.5 h-5 w-9 shrink-0 rounded-full transition-colors disabled:cursor-not-allowed",
          checked ? "bg-accent" : "bg-line-strong")}>
        {/* The knob is anchored at the pill's left edge: without left-0 its
            static position sat at the button's centred content box, so the
            on-state knob slid off the pill and over the label (2026-10-23). */}
        <span aria-hidden className={cn("absolute left-0 top-0.5 h-4 w-4 rounded-full bg-surface shadow-card transition-transform",
          checked ? "translate-x-[18px]" : "translate-x-0.5")} />
      </button>
      <span className="min-w-0">
        <span className="block text-[13px] font-medium text-fg">{label}</span>
        {hint && <span className="block text-[12px] text-fg-3">{hint}</span>}
      </span>
    </label>
  );
}
