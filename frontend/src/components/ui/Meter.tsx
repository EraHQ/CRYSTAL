import { cn } from "@/lib/utils";

// Part 3: a 6px bar colored by threshold, accent under 85%, warn from 85
// to 99, bad at 100. Label and value sit above it. Exported so the pin
// and the Billing meters agree on the thresholds.
export function meterTone(pct: number): "accent" | "warn" | "bad" {
  if (pct >= 100) return "bad";
  if (pct >= 85) return "warn";
  return "accent";
}

export function Meter({ value, max = 100, label, valueLabel, className }: {
  value: number;
  max?: number;
  label: string;
  valueLabel?: string;
  className?: string;
}) {
  const pct = max > 0 ? Math.max(0, Math.min(100, (value / max) * 100)) : 0;
  const tone = meterTone(pct);
  const fill = { accent: "bg-accent", warn: "bg-warn", bad: "bg-bad" }[tone];
  return (
    <div className={cn("min-w-0", className)}>
      <div className="mb-1 flex items-baseline justify-between gap-2 text-[12.5px]">
        <span className="truncate text-fg-2">{label}</span>
        {valueLabel && <span className={cn("shrink-0 tabular-nums", tone === "accent" ? "text-fg-2" : tone === "warn" ? "text-warn" : "text-bad")}>{valueLabel}</span>}
      </div>
      <div role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(pct)}
        data-tone={tone} className="h-1.5 w-full overflow-hidden rounded-full bg-inset">
        <div className={cn("h-full rounded-full transition-[width]", fill)} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
