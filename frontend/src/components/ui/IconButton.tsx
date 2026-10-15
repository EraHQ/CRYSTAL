import { ButtonHTMLAttributes } from "react";
import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

// Part 3: an icon-only button MUST say what it does. `label` is required
// (it becomes aria-label and the native title) and a visible tooltip
// appears on hover and keyboard focus. No unlabeled icon buttons.
export function IconButton({ label, icon: Icon, size = "md", variant = "ghost", className, ...props }: {
  label: string;
  icon: LucideIcon;
  size?: "sm" | "md";
  variant?: "ghost" | "outline" | "danger";
} & Omit<ButtonHTMLAttributes<HTMLButtonElement>, "aria-label" | "title" | "children">) {
  const variants = {
    ghost: "text-fg-3 hover:bg-inset hover:text-fg",
    outline: "border border-line text-fg-2 hover:bg-inset hover:text-fg",
    danger: "text-fg-3 hover:bg-bad-soft hover:text-bad",
  };
  return (
    <span className="group relative inline-flex">
      <button type="button" aria-label={label} title={label}
        className={cn("inline-flex items-center justify-center rounded-md transition-colors disabled:opacity-40",
          size === "sm" ? "h-6 w-6" : "h-8 w-8", variants[variant], className)}
        {...props}>
        <Icon aria-hidden className={size === "sm" ? "h-3.5 w-3.5" : "h-4 w-4"} />
      </button>
      <span role="tooltip" aria-hidden
        className="pointer-events-none absolute left-1/2 top-full z-10 mt-1 -translate-x-1/2 whitespace-nowrap rounded-md border border-line bg-surface px-2 py-0.5 text-[11.5px] text-fg-2 opacity-0 shadow-card transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
        {label}
      </span>
    </span>
  );
}
