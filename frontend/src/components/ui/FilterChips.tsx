import { cn } from "@/lib/utils";

export interface FilterChipOption {
  value: string;
  label: string;
  count?: number;
  /** The raw term the label stands for (e.g. "whitelist" behind "Trusted"); shown as a tooltip. */
  title?: string;
}

// Part 3: single or multi select chips, each with an optional count and
// the raw term in `title`. Selection is announced with aria-pressed.
export function FilterChips(props: {
  options: FilterChipOption[];
  className?: string;
  label?: string;
} & ({ multi?: false; value: string | null; onChange: (v: string | null) => void }
   | { multi: true; value: string[]; onChange: (v: string[]) => void })) {
  const { options, className, label } = props;
  const isOn = (v: string) => (props.multi ? props.value.includes(v) : props.value === v);
  const toggle = (v: string) => {
    if (props.multi) {
      props.onChange(props.value.includes(v) ? props.value.filter((x) => x !== v) : [...props.value, v]);
    } else {
      props.onChange(props.value === v ? null : v);
    }
  };
  return (
    <div role="group" aria-label={label} className={cn("flex flex-wrap items-center gap-1.5", className)}>
      {options.map((o) => {
        const on = isOn(o.value);
        return (
          <button key={o.value} type="button" aria-pressed={on} title={o.title} onClick={() => toggle(o.value)}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-[12.5px] font-medium transition-colors",
              on ? "border-accent/40 bg-accent-soft text-accent" : "border-line text-fg-2 hover:bg-inset hover:text-fg",
            )}>
            {o.label}
            {o.count !== undefined && <span className={cn("tabular-nums", on ? "text-accent/80" : "text-fg-3")}>{o.count}</span>}
          </button>
        );
      })}
    </div>
  );
}
