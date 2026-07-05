import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

// Standard cn helper. Used by every component that conditionally
// composes Tailwind classes.
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Pretty-print a number of bytes / tokens / milliseconds. Returns "—"
// for null so the table cells render consistently.
export function fmtNum(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  return n.toLocaleString();
}

export function fmtSigned(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  if (n > 0) return `+${n.toLocaleString()}`;
  return n.toLocaleString();
}

export function fmtFloat(
  n: number | null | undefined,
  digits = 3
): string {
  if (n === null || n === undefined) return "—";
  return n.toFixed(digits);
}

export function fmtDateTime(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleString();
  } catch {
    return iso;
  }
}

// Truncate a string to maxLen chars, adding ellipsis if truncated.
export function truncate(s: string, maxLen: number): string {
  if (s.length <= maxLen) return s;
  return s.slice(0, maxLen - 1) + "…";
}

// Match-type to a Tailwind color class. Used for colored pills and
// dot indicators.
export function matchColor(
  matchType: string
): { bg: string; text: string; ring: string } {
  switch (matchType) {
    case "high":
      return {
        bg: "bg-green-100",
        text: "text-green-800",
        ring: "ring-green-600/20",
      };
    case "medium":
      return {
        bg: "bg-yellow-100",
        text: "text-yellow-800",
        ring: "ring-yellow-600/20",
      };
    case "low":
      return {
        bg: "bg-red-100",
        text: "text-red-800",
        ring: "ring-red-600/20",
      };
    default:
      return {
        bg: "bg-zinc-100",
        text: "text-zinc-700",
        ring: "ring-zinc-600/10",
      };
  }
}
